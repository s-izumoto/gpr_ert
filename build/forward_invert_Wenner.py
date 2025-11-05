"""
ERT forward–inversion data generator for a 1D electrode line (Wenner-ready).

Overview
--------
This script reconstructs log10-resistivity fields from latent vectors (Z) using a
PCA model, pads them to a target depth, and samples cell values onto a 2D pyGIMLi
mesh. It then builds ERT measurement schemes either as (a) full Wenner sequences
or (b) randomized AB/MN pairs under simple geometric constraints, simulates
apparent resistivity (rho_a), and writes a surrogate dataset bundle (.npz).
Optionally, it inverts the simulated data (per selected field) and saves
illustration PNGs plus an inversion bundle.

Key outputs (saved under --out):
- ert_surrogate.npz
    Z, D (meters), Dnorm ([0,1]), ABMN (indices), y=log10(rho_a), rho_a, k, xs, field_ids, meta
- inversion_fieldXXX.png / _log.png  (optional)
- inversion_bundle_Wenner.npz        (optional; inverted cell models per field and metadata)

Typical use
-----------
python forward_invert_Wenner.py \\
  --pca path/to/pca_model.joblib \\
  --Z path/to/Z.npy \\
  --out ./outputs_wenner \\
  --design-type wenner --wenner-a-min 1 --wenner-a-max 10 \\
  --invert --inv-save-all-png

Notes
-----
- The script forces headless plotting (Qt/MPL offscreen) and limits BLAS/OpenMP
  threads to avoid CPU oversubscription during parallel runs.
- Dnorm columns are normalized by the inner survey width (world length minus margins):
  [ dAB/L, dMN/L, (mAB - margin)/L, (mMN - margin)/L ].
- Set --dx-elec > 0 to derive world length from the electrode spacing; otherwise
  --world-Lx is used as-is.
"""

from __future__ import annotations
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import os, json, tempfile, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any, Tuple, List, Optional

import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import matplotlib.pyplot as plt
import concurrent.futures

# Limit BLAS/OpenMP threads per worker (avoid oversubscription)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

# ==== Config ====
@dataclass
class ERTForwardConfig:
    """Configuration for ERT forward simulation and optional inversion.

    Important fields
    ----------------
    pca : str
        Path to a joblib file with keys ["mean", "components", "nz", "nx"].
    Z_path : str
        Path to latent array (N, k) in .npy/.npz.
    out : str
        Output directory for all bundles and images.
    design_type : {"random","wenner"}
        Pair generator. "wenner" enumerates all valid A--a-->M--a-->N--a-->B.
    dx_elec : float
        If > 0, sets world length = 2*margin + (n_elec-1)*dx_elec.
    noise_rel : float
        Relative Gaussian noise applied to rho_a.
    invert / invert_all / inv_npz_out / inv_save_all_png
        Inversion toggles and output paths.
    workers : int or None
        Number of process workers; None picks a safe default.
    """
    pca: str
    Z_path: str
    out: str
    # selection
    n_fields: int = -1
    field_offset: int = 0
    fields: str | list[int] | None = None
    field_from_npz: str | None = None
    # geometry
    nz_full: int = 100
    nx_full: int = 400
    world_Lx: float = 31.0     # used if dx_elec <= 0
    margin: float = 3.0
    n_elec: int = 32
    dx_elec: float = 1.0       # if >0, world_Lx is computed as 2*margin + (n_elec-1)*dx
    mesh_area: float = 0.1
    # design
    design_type: str = "random"  # "random" or "wenner"
    wenner_a_min: int = 1
    wenner_a_max: int = 10
    n_AB: int = 32
    n_MN_per_AB: int = 16
    dAB_min: float = 2.0
    dAB_max: float = 18.0
    dMN_min: float = 1.0
    dMN_max: float = 12.0
    # forward
    mode: str = "2d"
    noise_rel: float = 0.02
    # misc
    seed: int = 0
    # inversion
    invert: bool = True
    invert_all: bool = True         # ★ 既定 True
    invert_out: str | None = None   # 先頭PNGの保存先（既定名は自動）
    inv_npz_out: str | None = None  # バンドルNPZ保存先
    inv_save_all_png: bool = False  # PNGは既定で先頭のみ
    workers: int | None = None  # Noneなら自動


# ==== Small utils ====
class TempDir:
    """Context manager to run pyGIMLi operations in a throwaway folder.

    Some pyGIMLi functions create temporary files in the CWD. This ensures
    a clean per-task directory that is removed on exit.
    """
    def __init__(self, base: str | None = None, prefix: str = "pg_worker_"):
        self.base = base; self.prefix = prefix; self.tmp = None; self.cwd = None
    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix=self.prefix, dir=self.base)
        self.cwd = os.getcwd()
        os.chdir(self.tmp)
        return self.tmp
    def __exit__(self, exc_type, exc, tb):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

# ==== PCA ====
def load_pca(p: str | Path):
    """Load PCA pack from joblib and return (mean, components, nz, nx).

    Expects a dict with float arrays "mean", "components" and ints "nz", "nx".
    """
    from joblib import load as joblib_load
    mdl = joblib_load(p)
    return (
        mdl["mean"].astype(np.float32),
        mdl["components"].astype(np.float32),
        int(mdl["nz"]), int(mdl["nx"]),
    )

def load_Z_any(path: str | Path, k: int | None = None) -> np.ndarray:
    """Load Z latent matrix from .npy/.npz (or a generic np.load file).

    Parameters
    ----------
    path : str | Path
        File containing Z with shape (N, k).
    k : int | None
        If provided, truncate columns to k.

    Returns
    -------
    np.ndarray
        Float32 array of shape (N, <=k).

    Raises
    ------
    ValueError
        If Z is not 2D.
    """
    path = str(path)
    if path.lower().endswith(".npy"):
        Z = np.load(path).astype(np.float32)
    elif path.lower().endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        key = "Z" if "Z" in d.files else d.files[0]
        Z = d[key].astype(np.float32)
    else:
        Z = np.load(path, allow_pickle=True)
        if isinstance(Z, np.lib.npyio.NpzFile):
            key = "Z" if "Z" in Z.files else Z.files[0]
            Z = Z[key].astype(np.float32)
        else:
            Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2:
        raise ValueError(f"Expected Z to be 2D (N, k). Got shape {Z.shape} from {path}.")
    return Z[:, :k] if (k is not None and Z.shape[1] > k) else Z

def recon_shallow_from_Z(Z: np.ndarray, comps: np.ndarray, mean: np.ndarray, nz: int, nx: int) -> np.ndarray:
    """Reconstruct shallow log10-resistivity images from latent Z using PCA.

    Returns array shaped (N, nz, nx).
    """
    Xf = (Z @ comps + mean).astype(np.float32)
    return Xf.reshape((-1, nz, nx))

def pad_deep(shallow: np.ndarray, nz_full: int) -> np.ndarray:
    """Pad the shallow model to nz_full by repeating the last row (simple extrapolation)."""
    N, nz, nx = shallow.shape
    if nz_full <= nz:
        return shallow[:, :nz_full, :]
    last = shallow[:, nz - 1:nz, :]
    deep = np.repeat(last, nz_full - nz, axis=1)
    return np.concatenate([shallow, deep], axis=1)

# ==== Geometry / mesh ====
def make_sensor_positions(n_elec: int, L_world: float, margin: float):
    """Create 1D surface electrode positions along x within [margin, Lx-margin].

    Returns
    -------
    sensors : list[pg.Pos]
    xs : np.ndarray
        Electrode x coordinates.
    dx_eff : float
        Effective electrode spacing.
    L_inner : float
        World length minus margins (normalization base for Dnorm).
    """
    L_inner = L_world - 2.0 * margin
    if L_inner <= 0: raise ValueError("Margin too large for world length.")
    dx = L_inner / float(n_elec - 1)
    xs = margin + dx * np.arange(n_elec, dtype=np.float32)
    sensors = [pg.Pos(float(x), 0.0) for x in xs]
    return sensors, xs.astype(np.float32), float(dx), float(L_inner)

def build_mesh_world(L_world: float, Lz: float, sensors: Iterable[Any], dz_under: float = 0.05,
                     area: float | None = None, quality: int = 34):
    """Build a 2D world mesh for ERT with optional area control.

    Adds shallow 'dip' nodes below each electrode to stabilize near-surface cells.
    """
    world = mt.createWorld(start=[0.0, 0.0], end=[L_world, -Lz], worldMarker=True)
    for p in sensors:
        world.createNode(p)
        world.createNode(pg.Pos(p[0], -dz_under * Lz))
    kwargs = dict(quality=quality)
    if isinstance(area, (int, float)) and area > 0:
        kwargs["area"] = float(area)
    return mt.createMesh(world, **kwargs)

def sample_logR_to_cells(logR: np.ndarray, mesh: pg.Mesh, L_world: float, Lz: float) -> np.ndarray:
    """Nearest-grid sampling of log10-resistivity (NZ×NX) onto mesh cell centers.

    Coordinates are mapped to [0, L_world] × [0, -Lz], with a flip along z to
    match image coordinates (top row = shallow cells).
    """
    NZ, NX = logR.shape
    xs = np.linspace(0.0, L_world, NX, dtype=np.float32)
    zs = np.linspace(0.0, -Lz, NZ, dtype=np.float32)
    cx = np.array([c.center()[0] for c in mesh.cells()], dtype=np.float32)
    cz = np.array([c.center()[1] for c in mesh.cells()], dtype=np.float32)
    ix = np.clip(np.round((cx - xs[0]) / (xs[-1] - xs[0]) * (NX - 1)).astype(int), 0, NX - 1)
    iz = np.clip(np.round((cz - zs[-1]) / (zs[0] - zs[-1]) * (NZ - 1)).astype(int), 0, NZ - 1)
    iz = (NZ - 1) - iz
    return logR[iz, ix]

# ==== Designs ====
def generate_wenner_pairs(n_elec: int, a_min: int = 1, a_max: int | None = None):
    """Enumerate all valid Wenner quadruples (a,b,m,n) for n_elec.

    Pattern: A--a-->M--a-->N--a-->B with a in [a_min, a_max].
    Returns four int arrays (A, B, M, N).
    """
    if a_max is None:
        a_max = n_elec // 3
    aa_all, bb_all, mm_all, nn_all = [], [], [], []
    for a in range(int(max(1, a_min)), int(a_max) + 1):
        for A in range(0, n_elec - 3 * a):
            M = A + a; N = A + 2 * a; B = A + 3 * a
            aa_all.append(A); bb_all.append(B); mm_all.append(M); nn_all.append(N)
    return (np.asarray(aa_all, dtype=np.int32),
            np.asarray(bb_all, dtype=np.int32),
            np.asarray(mm_all, dtype=np.int32),
            np.asarray(nn_all, dtype=np.int32))

def sample_unique_pairs_by_AB(rng: np.random.Generator, xs: np.ndarray, n_AB: int, n_MN_per_AB: int,
                              dAB_min: float, dAB_max: float, dMN_min: float, dMN_max: float):
    """Randomly sample unique AB backbones and unique MN per AB.

    Constraints
    -----------
    - dAB in [dAB_min, dAB_max], dMN in [dMN_min, dMN_max]
    - MN electrodes never equal A or B
    - All pairs snapped to nearest electrodes on xs

    Returns
    -------
    a, b, m, n : int arrays (indices)
    out_arr : float32 (M, 4)
        Columns: [dAB, dMN, mAB, mMN] in meters.
    """
    def snap_pair(x1: float, x2: float):
        i1 = int(np.argmin(np.abs(xs - x1)))
        i2 = int(np.argmin(np.abs(xs - x2)))
        if i1 == i2: i2 = min(i2 + 1, len(xs) - 1)
        if i1 > i2: i1, i2 = i2, i1
        d = abs(xs[i2] - xs[i1]); m = 0.5 * (xs[i2] + xs[i1])
        return i1, i2, d, m

    Lx = float(xs[-1] - xs[0])

    # Unique AB
    AB_seen = set(); AB_list = []
    tries = 0; max_try_AB = 1000 * max(1, n_AB)
    while len(AB_list) < n_AB and tries < max_try_AB:
        tries += 1
        mAB = rng.uniform(xs[0] + 0.05 * Lx, xs[-1] - 0.05 * Lx)
        dAB = rng.uniform(dAB_min, dAB_max)
        xA, xB = mAB - 0.5 * dAB, mAB + 0.5 * dAB
        ia, ib, dAB_s, mAB_s = snap_pair(xA, xB)
        key_ab = (ia, ib)
        if key_ab in AB_seen: continue
        AB_seen.add(key_ab); AB_list.append((ia, ib, dAB_s, mAB_s))
    if len(AB_list) < n_AB:
        raise RuntimeError(f"Could not get {n_AB} unique AB pairs.")

    # Unique MN per AB (no overlap with A or B)
    a_all=[]; b_all=[]; m_all=[]; n_all=[]; out=[]
    max_try_MN = 2000 * max(1, n_MN_per_AB)
    for (ia, ib, dAB_s, mAB_s) in AB_list:
        MN_seen = set(); tries = 0
        while len(MN_seen) < n_MN_per_AB and tries < max_try_MN:
            tries += 1
            mMN = rng.uniform(xs[0] + 0.05 * Lx, xs[-1] - 0.05 * Lx)
            dMN = rng.uniform(dMN_min, dMN_max)
            xM, xN = mMN - 0.5 * dMN, mMN + 0.5 * dMN
            im, in_ = int(np.argmin(np.abs(xs - xM))), int(np.argmin(np.abs(xs - xN)))
            if im == in_:
                in_ = min(in_ + 1, len(xs)-1)
            if (im in (ia, ib)) or (in_ in (ia, ib)): continue
            if im > in_: im, in_ = in_, im
            dMN_s, mMN_s = abs(xs[in_] - xs[im]), 0.5 * (xs[in_] + xs[im])
            key_mn = (im, in_)
            if key_mn in MN_seen: continue
            MN_seen.add(key_mn)
            a_all.append(ia); b_all.append(ib); m_all.append(im); n_all.append(in_)
            out.append((dAB_s, dMN_s, mAB_s, mMN_s))
        if len(MN_seen) < n_MN_per_AB:
            raise RuntimeError(f"Could not get {n_MN_per_AB} unique MN for AB=({ia},{ib}).")

    a = np.array(a_all, dtype=np.int32)
    b = np.array(b_all, dtype=np.int32)
    m = np.array(m_all, dtype=np.int32)
    n = np.array(n_all, dtype=np.int32)
    out_arr = np.array(out, dtype=np.float32)
    return a, b, m, n, out_arr

def make_scheme(sensors: Iterable[Any], a: np.ndarray, b: np.ndarray, m: np.ndarray, n: np.ndarray) -> pg.DataContainerERT:
    """Build a pg.DataContainerERT with given (a,b,m,n) indices and attached sensors."""
    dc = pg.DataContainerERT()
    for p in sensors:
        dc.createSensor(p)
    dc.resize(len(a))
    dc["a"] = pg.Vector(a); dc["b"] = pg.Vector(b)
    dc["m"] = pg.Vector(m); dc["n"] = pg.Vector(n)
    return dc

# ==== Inversion helper ====
def invert_and_save_image(mesh: pg.Mesh,
                          scheme: pg.DataContainerERT,
                          rhoa: np.ndarray,
                          out_png_linear: str,
                          out_png_log: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Invert rho_a on the provided mesh/scheme and save linear/log PNGs.

    Also writes a sidecar .npz next to the PNG with:
      inv_rho_cells, cell_centers, cmin/cmax, abmn, rhoa, world bounds (meters).
    Returns (path_to_linear_png, path_to_log_png_or_None).
    """
    mgr = ert.ERTManager()

    data = scheme.copy()
    data["rhoa"] = rhoa
    data["err"] = np.full_like(rhoa, 0.03, dtype=float) 

    inv_res = mgr.invert(data, mesh=mesh, lam=20, robust=True, verbose=False)
    inv_arr = np.asarray(inv_res, dtype=float)
    cmin = float(np.nanmin(inv_arr))
    cmax = float(np.nanmax(inv_arr))

    fig, ax = plt.subplots(figsize=(6, 3))
    _ = mgr.showResult(ax=ax, cMin=cmin, cMax=cmax, cMap="Spectral_r", logScale=False)
    ax.set_title("Inverted resistivity (linear color)")
    fig.tight_layout()
    fig.savefig(out_png_linear, dpi=200)
    plt.close(fig)

    out_log_path = None
    if out_png_log:
        cmin_log = max(cmin, 1e-12)  
        fig, ax = plt.subplots(figsize=(6, 3))
        _ = mgr.showResult(ax=ax, cMin=cmin_log, cMax=cmax, cMap="Spectral_r", logScale=True)
        ax.set_title("Inverted resistivity (log color)")
        fig.tight_layout()
        fig.savefig(out_png_log, dpi=200)
        plt.close(fig)
        out_log_path = out_png_log

    try:
        npz_path = str(Path(out_png_linear).with_suffix(".npz"))
        cx = np.array([c.center()[0] for c in mesh.cells()], dtype=np.float32)
        cz = np.array([c.center()[1] for c in mesh.cells()], dtype=np.float32)
        xs = np.array([v[0] for v in mesh.nodes()], dtype=np.float32)
        zs = np.array([v[1] for v in mesh.nodes()], dtype=np.float32)
        xmin, xmax = float(xs.min()), float(xs.max())
        zmin, zmax = float(zs.min()), float(zs.max())
        L_world = xmax - xmin
        Lz = abs(zmax - zmin)
        abmn = np.stack([scheme["a"], scheme["b"], scheme["m"], scheme["n"]], axis=1).astype(np.int32)
        rhoa_f = np.asarray(rhoa, dtype=np.float64)

        np.savez_compressed(
            npz_path,
            inv_rho_cells=np.asarray(inv_arr, dtype=np.float64),
            cell_centers=np.stack([cx, cz], axis=1),
            cmin=float(cmin),
            cmax=float(cmax),
            abmn=abmn,
            rhoa=rhoa_f,
            world_xmin=float(xmin),
            world_xmax=float(xmax),
            world_zmin=float(zmin),
            world_zmax=float(zmax),
            L_world=float(L_world),
            Lz=float(Lz),
        )
        print(f"[save] inversion NPZ -> {npz_path}")
    except Exception as e:
        print(f"[WARN] failed to save inversion NPZ: {e}")

    return out_png_linear, out_log_path



# ==== Worker ====
def simulate_one_field(z_row: np.ndarray, comps: np.ndarray, mean: np.ndarray, nz_crop: int, nx_crop: int,
                       cfg: ERTForwardConfig, L_world: float, Lz: float, seed: int):
    """Simulate one resistivity field from a single latent Z row.

    Steps
    -----
    1) PCA reconstruct → pad to nz_full → sample onto mesh cells.
    2) Build ABMN pairs (Wenner or randomized).
    3) Forward simulate rho_a with pyGIMLi and apply relative noise.
    4) Build design matrix D (meters) and Dnorm ([0,1]) using inner width L_inner.
    5) Return all arrays needed to assemble the surrogate dataset.
    """
    rng = np.random.default_rng(seed)

    # Reconstruct field
    shallow = recon_shallow_from_Z(z_row[None, :], comps, mean, nz_crop, nx_crop)  # (1, nz_crop, nx_crop)
    logR_full = pad_deep(shallow, cfg.nz_full)[0]  # (nz_full, nx_crop)

    # Sensors, mesh, cell model
    sensors, xs, dx_eff, L_inner = make_sensor_positions(cfg.n_elec, L_world, cfg.margin)
    mesh = build_mesh_world(L_world, Lz, sensors, dz_under=0.05,
                            area=(cfg.mesh_area if cfg.mesh_area > 0 else None), quality=34)
    rho_cells = (10.0 ** sample_logR_to_cells(logR_full, mesh, L_world, Lz)).astype(np.float32)

    # Designs
    if cfg.design_type == "wenner":
        a_idx, b_idx, m_idx, n_idx = generate_wenner_pairs(cfg.n_elec, cfg.wenner_a_min, cfg.wenner_a_max)
        xA, xB, xM, xN = xs[a_idx], xs[b_idx], xs[m_idx], xs[n_idx]
        dAB = np.abs(xB - xA); dMN = np.abs(xN - xM)
        mAB = 0.5*(xA + xB);   mMN = 0.5*(xM + xN)
        designs_snapped = np.stack([dAB, dMN, mAB, mMN], axis=1).astype(np.float32)
        a, b, m, n = a_idx, b_idx, m_idx, n_idx
    else:
        a, b, m, n, designs_snapped = sample_unique_pairs_by_AB(
            rng, xs, cfg.n_AB, cfg.n_MN_per_AB, cfg.dAB_min, cfg.dAB_max, cfg.dMN_min, cfg.dMN_max
        )

    # Scheme & forward
    scheme = make_scheme(sensors, a, b, m, n)
    scheme.createGeometricFactors()

    with TempDir():
        data = ert.simulate(mesh=mesh, scheme=scheme, res=rho_cells,
                            noiseLevel=0.0, noiseAbs=0.0,
                            seed=int(rng.integers(0, 2**31 - 1)))
        rhoa = np.asarray(data["rhoa"], dtype=np.float32)
        kfac = np.asarray(data["k"], dtype=np.float32)

    # Relative Gaussian noise on rhoa
    if cfg.noise_rel and cfg.noise_rel > 0:
        rhoa = rhoa * (1.0 + (cfg.noise_rel * rng.standard_normal(size=rhoa.shape)).astype(np.float32))

    rhoa = np.maximum(rhoa.astype(np.float32), 1e-12)
    y = np.log10(rhoa).astype(np.float32)

    # Normalize designs by inner width
    D = designs_snapped.astype(np.float32)
    Dnorm = np.empty_like(D)
    Dnorm[:, 0] = D[:, 0] / L_inner  # dAB/L
    Dnorm[:, 1] = D[:, 1] / L_inner  # dMN/L
    Dnorm[:, 2] = (D[:, 2] - cfg.margin) / L_inner  # mAB normalized
    Dnorm[:, 3] = (D[:, 3] - cfg.margin) / L_inner  # mMN normalized

    ABMN = np.stack([a, b, m, n], axis=1).astype(np.int32)
    return (
        D.astype(np.float32),
        Dnorm.astype(np.float32),
        ABMN.astype(np.int32),
        y.astype(np.float32),
        rhoa.astype(np.float32),
        kfac.astype(np.float32),
        xs.astype(np.float32),
        float(dx_eff),
    )

def _simulate_one_field_payload(args):
    """Thin wrapper for parallel execution: returns a dict payload per field."""
    (z_row, comps, mean, nz_crop, nx_crop, cfg, L_world, Lz, seed, field_id) = args
    D, Dn, ABMN, y, rhoa, kfac, xs, dx_eff = simulate_one_field(
        z_row, comps, mean, nz_crop, nx_crop, cfg, L_world, Lz, seed
    )
    return {
        "field_id": int(field_id),
        "Z_row": z_row.astype(np.float32),
        "D": D, "Dn": Dn, "ABMN": ABMN,
        "y": y, "rhoa": rhoa, "k": kfac,
        "xs": xs, "dx_eff": float(dx_eff),
    }


def _parse_fields(s: str) -> list[int]:
    """Parse a comma/range string like '0,4,7-10' into a sorted unique int list."""
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    parts = s.replace(" ", "").split(",")
    out = []
    for p in parts:
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))
        else:
            out.append(int(p))
    return sorted(set(out))

def _read_fields_from_seqnpz(path: str | os.PathLike) -> list[int]:
    """Extract field indices from a seq_log*.npz produced by gpr_seq_core.py.

    Priority order:
      - 'field' (scalar)
      - 'fields' (array)
      - 'field_index' or 'field_id' (scalar)
      - any npz key ending with '__fieldNNN'
    Raises if nothing can be found.
    """
    import re
    with np.load(path, allow_pickle=False) as z:
        if "field" in z.files:
            return [int(np.asarray(z["field"]).item())]
        if "fields" in z.files:
            arr = np.asarray(z["fields"]).astype(int).ravel()
            if arr.size == 0:
                raise ValueError(f"'fields' is empty in {path}")
            return sorted(set(int(x) for x in arr))
        for k in ("field_index", "field_id"):
            if k in z.files:
                return [int(np.asarray(z[k]).item())]
        cand = []
        for name in z.files:
            m = re.search(r"__field(\d+)$", name)
            if m:
                cand.append(int(m.group(1)))
        if cand:
            return sorted(set(cand))
        raise KeyError("No field keys found in npz.")


# ==== Runner ====
def run(cfg: ERTForwardConfig) -> Path:
    """Main driver: load models/data, pick fields, simulate in parallel, save bundles.

    Writes:
      out/ert_surrogate.npz
      and (optional) inversion images + inversion bundle.
    Returns the Path to ert_surrogate.npz.
    """
    out_dir = Path(cfg.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Load PCA and latent
    mean, comps, nz_crop, nx_crop = load_pca(cfg.pca)
    k_lat = comps.shape[0]
    Zall = load_Z_any(cfg.Z_path, k=k_lat)

    # === Field selection ===
    n_total = Zall.shape[0]

    field_from_npz = getattr(cfg, "field_from_npz", None)
    if field_from_npz not in (None, "", False):
        fields = _read_fields_from_seqnpz(field_from_npz)
        bad = [i for i in fields if i < 0 or i >= n_total]
        if bad:
            raise ValueError(f"field(s) {bad} from {field_from_npz} out of range (n_total={n_total}).")
        fields   = sorted(set(int(i) for i in fields))
        Zslice   = Zall[fields, :]
        n_fields = len(fields)
        start, stop = (min(fields), max(fields)+1)
        print(f"[ERT] simulate: {n_fields} field(s) from npz → {fields} (k={k_lat}) [{field_from_npz}]")

    elif hasattr(cfg, "fields") and cfg.fields not in (None, [], ""):
        if isinstance(cfg.fields, (list, tuple)):
            fields = sorted(set(int(i) for i in cfg.fields))
        else:
            fields = _parse_fields(cfg.fields)
        bad = [i for i in fields if i < 0 or i >= n_total]
        if bad:
            raise ValueError(f"Invalid field indices in 'fields': {bad} (total n_total={n_total})")
        Zslice   = Zall[fields, :]
        n_fields = len(fields)
        print(f"[ERT] simulate: {n_fields} fields (explicit) k={k_lat} indices={fields}")
        start, stop = (min(fields) if fields else 0), (max(fields) + 1 if fields else 0)

    else:
        start = max(0, int(cfg.field_offset))
        stop  = n_total if int(cfg.n_fields) <= 0 else min(n_total, start + int(cfg.n_fields))
        fields   = list(range(start, stop))
        Zslice   = Zall[start:stop, :]
        n_fields = len(fields)
        print(f"[ERT] simulate: {n_fields} fields, k={k_lat} (idx {start}..{stop-1})")

    if cfg.workers in (None, 0):
        cpu = os.cpu_count() or 1
        auto = max(1, min(len(fields), max(1, cpu // 2))) 
        workers = auto
    else:
        workers = max(1, int(cfg.workers))
    print(f"[parallel] workers = {workers}")


    # Geometry: compute world_Lx if dx_elec given
    if cfg.dx_elec and cfg.dx_elec > 0:
        L_world = 2.0 * cfg.margin + (cfg.n_elec - 1) * cfg.dx_elec
    else:
        L_world = float(cfg.world_Lx)
    hy = L_world / float(cfg.nx_full)
    Lz = hy * float(cfg.nz_full)

    # Seeds
    base_rng = np.random.default_rng(getattr(cfg, "seed", 0))
    seeds = base_rng.integers(0, 2**31 - 1, size=max(1, len(fields)), dtype=np.int64)

    # Accumulators
    Z_all, D_all, Dn_all, ABMN_all = [], [], [], []
    y_all, rhoa_all, k_all = [], [], []
    schemes = []
    xs_ref = None; dx_eff_ref = None
    field_ids = [] 

    task_args = []
    for i_local, field_id in enumerate(fields):
        z_row = Zslice[i_local]
        seed_i = int(seeds[i_local % len(seeds)])
        task_args.append((z_row, comps, mean, nz_crop, nx_crop, cfg, L_world, Lz, seed_i, field_id))

    results = []
    if workers == 1:
        for ta in task_args:
            results.append(_simulate_one_field_payload(ta))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_simulate_one_field_payload, task_args):
                results.append(r)

    Z_all, D_all, Dn_all, ABMN_all = [], [], [], []
    y_all, rhoa_all, k_all = [], [], []
    xs_ref = None; dx_eff_ref = None
    field_ids = []

    for r in results:
        field_id = r["field_id"]
        D, Dn, ABMN = r["D"], r["Dn"], r["ABMN"]
        y, rhoa, kfac = r["y"], r["rhoa"], r["k"]
        z_row = r["Z_row"]
        Zrep = np.repeat(z_row[None, :], D.shape[0], axis=0).astype(np.float32)

        Z_all.append(Zrep); D_all.append(D); Dn_all.append(Dn)
        ABMN_all.append(ABMN); y_all.append(y); rhoa_all.append(rhoa); k_all.append(kfac)
        if xs_ref is None:
            xs_ref = r["xs"]; dx_eff_ref = r["dx_eff"]
        field_ids.append(np.full(D.shape[0], field_id, dtype=np.int32))

    if not Z_all:
        raise RuntimeError("No simulations; nothing to save. Check inputs/selection.")

    # Concatenate
    Z_all    = np.vstack(Z_all).astype(np.float32)
    D_all    = np.vstack(D_all).astype(np.float32)
    Dn_all   = np.vstack(Dn_all).astype(np.float32)
    ABMN_all = np.vstack(ABMN_all).astype(np.int32)
    y_all    = np.concatenate(y_all).astype(np.float32)
    rhoa_all = np.concatenate(rhoa_all).astype(np.float32)
    k_all    = np.concatenate(k_all).astype(np.float32)
    field_ids= np.concatenate(field_ids).astype(np.int32)

    meta = {
        "start_index": int(start),
        "n_fields": int(n_fields),
        "selected_fields": fields, 
        "k_lat": int(k_lat),
        "nz_full": int(cfg.nz_full),
        "nx_full": int(cfg.nx_full),
        "world_Lx_m": float(L_world),
        "Lz_m": float(Lz),
        "margin_m": float(cfg.margin),
        "n_elec": int(cfg.n_elec),
        "dx_eff_m": float(dx_eff_ref),
        "design_desc": "[dAB, dMN, mAB, mMN] meters; Dnorm normalized by inner width",
        "mode": cfg.mode,
        "noise_rel": float(cfg.noise_rel),
    }

    out_npz = out_dir / "ert_surrogate.npz"
    np.savez_compressed(
        out_npz,
        Z=Z_all, D=D_all, Dnorm=Dn_all, ABMN=ABMN_all,
        y=y_all, rhoa=rhoa_all, k=k_all,
        xs=xs_ref.astype(np.float32),
        field_ids=field_ids,
        meta=np.array([json.dumps(meta)], dtype=object),
    )
    print("Saved:", out_npz)

    if cfg.invert and len(fields) > 0:
        try:
            inv_log_list, inv_rho_list, inv_field_ids = [], [], []
            png_paths = []
            inv_save_all_png = bool(getattr(cfg, "inv_save_all_png", False))
            inv_npz_out = getattr(cfg, "inv_npz_out", None)

            for i_local, fid in enumerate(fields):
                mask_i = (field_ids == fid)
                rhoa_i = rhoa_all[mask_i]
                abmn_i = ABMN_all[0][0:0]  # dummy; we will compute next line properly
                abmn_i = ABMN_all[0]  
            ABMN_all = np.vstack(ABMN_all).astype(np.int32)

            for i_local, fid in enumerate(fields):
                mask_i = (field_ids == fid)
                rhoa_i = rhoa_all[mask_i]
                abmn_i = ABMN_all[mask_i] 

                sensors, xs_re, _, _ = make_sensor_positions(cfg.n_elec, L_world, cfg.margin)
                scheme_i = make_scheme(sensors,
                                    abmn_i[:,0].astype(np.int32),
                                    abmn_i[:,1].astype(np.int32),
                                    abmn_i[:,2].astype(np.int32),
                                    abmn_i[:,3].astype(np.int32))
                scheme_i.createGeometricFactors()

                mesh_i = build_mesh_world(
                    L_world, Lz, sensors, dz_under=0.05,
                    area=(cfg.mesh_area if cfg.mesh_area > 0 else None), quality=34
                )

                mgr = ert.ERTManager()
                data = scheme_i.copy()
                data["rhoa"] = rhoa_i
                rel = float(getattr(cfg, "inv_rel_err", getattr(cfg, "noise_rel", 0.03)))
                data["err"] = np.full_like(rhoa_i, rel, dtype=np.float32)

                inv_rho = mgr.invert(data, mesh=mesh_i, lam=20, robust=True, verbose=False)
                inv_rho = np.asarray(inv_rho, dtype=np.float32)
                inv_log = np.log10(inv_rho, dtype=np.float32)

                inv_field_ids.append(np.int32(fid))
                inv_rho_list.append(inv_rho)
                inv_log_list.append(inv_log)

                if (i_local == 0) and getattr(cfg, "invert_out", None):
                    out_lin = Path(cfg.invert_out)
                else:
                    out_lin = (out_dir / f"inversion_field{fid:03d}.png")

                out_log = out_lin.with_name(out_lin.stem + "_log" + out_lin.suffix)
                out_lin.parent.mkdir(parents=True, exist_ok=True)

                save_png_this = inv_save_all_png or (i_local == 0)
                if save_png_this:
                    out_lin = (Path(cfg.invert_out) if (i_local == 0 and getattr(cfg, "invert_out", None))
                            else (out_dir / f"inversion_field{fid:03d}.png"))
                    out_log = out_lin.with_name(out_lin.stem + "_log" + out_lin.suffix)

                    fig, ax = plt.subplots(figsize=(6, 3))
                    _ = mgr.showResult(ax=ax, model=inv_rho, logScale=False)
                    ax.set_title(f"Inverted resistivity (linear) field{fid:03d}")
                    fig.tight_layout(); fig.savefig(out_lin, dpi=200); plt.close(fig)

                    fig, ax = plt.subplots(figsize=(6, 3))
                    _ = mgr.showResult(ax=ax, model=np.log10(inv_rho), logScale=False)
                    ax.set_title(f"Inverted resistivity (log10 values) field{fid:03d}")
                    fig.tight_layout(); fig.savefig(out_log, dpi=200); plt.close(fig)

                    png_paths.extend([str(out_lin), str(out_log)])

            if len(inv_rho_list) > 0:
                bundle_path = Path(inv_npz_out) if inv_npz_out else (out_dir / "inversion_bundle_Wenner.npz")
                save_dict = {}
                for fid, logv, rhov in zip(inv_field_ids, inv_log_list, inv_rho_list):
                    save_dict[f"inv_log_cells__field{int(fid):03d}"] = logv
                    save_dict[f"inv_rho_cells__field{int(fid):03d}"] = rhov

                sensors0, _, _, _ = make_sensor_positions(cfg.n_elec, L_world, cfg.margin)
                mesh0 = build_mesh_world(L_world, Lz, sensors0, dz_under=0.05,
                                        area=(cfg.mesh_area if cfg.mesh_area > 0 else None), quality=34)
                cx = np.array([c.center()[0] for c in mesh0.cells()], dtype=np.float32)
                cz = np.array([c.center()[1] for c in mesh0.cells()], dtype=np.float32)
                xs = np.array([v[0] for v in mesh0.nodes()], dtype=np.float32)
                zs = np.array([v[1] for v in mesh0.nodes()], dtype=np.float32)
                xmin, xmax = float(xs.min()), float(xs.max())
                zmin, zmax = float(zs.min()), float(zs.max())

                save_dict["cell_centers"] = np.stack([cx, cz], axis=1)
                save_dict["world_xmin"] = xmin; save_dict["world_xmax"] = xmax
                save_dict["world_zmin"] = zmin; save_dict["world_zmax"] = zmax
                save_dict["L_world"] = float(xmax - xmin); save_dict["Lz"] = float(abs(zmax - zmin))

                for fid in fields:
                    mask_i = (field_ids == fid)
                    abmn_i = ABMN_all[mask_i]
                    rhoa_i = rhoa_all[mask_i]
                    save_dict[f"abmn__field{fid:03d}"] = abmn_i.astype(np.int32)
                    save_dict[f"rhoa__field{fid:03d}"] = np.asarray(rhoa_i, dtype=np.float64)

                save_dict["inv_field_ids"] = np.array(inv_field_ids, dtype=np.int32)
                if png_paths:
                    save_dict["png_paths"] = np.array(png_paths, dtype=object)
                np.savez_compressed(bundle_path, **save_dict)
                print(f"[invert] saved bundle: {bundle_path}")

        except Exception as e:
            print("[WARN] Inversion failed:", e)


    return out_npz


# ==== CLI ====
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", required=True)
    ap.add_argument("--Z", dest="Z_path", required=True)
    ap.add_argument("--out", required=True)
    # selection
    ap.add_argument("--n-fields", type=int, default=-1)
    ap.add_argument("--field-offset", type=int, default=0)
    # geometry
    ap.add_argument("--nz-full", type=int, default=100)
    ap.add_argument("--nx-full", type=int, default=400)
    ap.add_argument("--world-Lx", type=float, default=31.0)
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--n-elec", type=int, default=32)
    ap.add_argument("--dx-elec", type=float, default=1.0)
    ap.add_argument("--mesh-area", type=float, default=0.1)
    # design
    ap.add_argument("--design-type", choices=["random","wenner"], default="random")
    ap.add_argument("--wenner-a-min", type=int, default=1)
    ap.add_argument("--wenner-a-max", type=int, default=10)
    ap.add_argument("--n-AB", type=int, default=32)
    ap.add_argument("--n-MN-per-AB", type=int, default=16)
    ap.add_argument("--dAB-min", type=float, default=2.0)
    ap.add_argument("--dAB-max", type=float, default=18.0)
    ap.add_argument("--dMN-min", type=float, default=1.0)
    ap.add_argument("--dMN-max", type=float, default=12.0)
    # forward
    ap.add_argument("--mode", choices=["2d"], default="2d")
    ap.add_argument("--noise-rel", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    # inversion
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--invert-out", type=str, default=None)
    ap.set_defaults(invert=True)
    ap.add_argument("--no-invert", action="store_false", dest="invert",
                    help="Use this option to explicitly disable inversion.")

    ap.add_argument(
        "--fields",
        type=str, default=None,
        help="Comma-separated list of 0-based field indices (e.g., 0,4,9). "
            "If provided, this overrides --n-fields and --field-offset."
    )

    ap.add_argument("--invert-all", action="store_true",
                help="Invert on all selected fields (default True)")
    ap.add_argument("--inv-npz-out", type=str, default=None,
                    help="Bundle NPZ path for inversion results (default: out/inversions_bundle.npz)")
    ap.add_argument("--inv-save-all-png", action="store_true",
                    help="Save inversion PNG for all fields (default: only first)")
    ap.add_argument(
        "--field-from-npz",
        type=str, default=None,
        help="Path to seq_log.npz exported by gpr_seq_core.py; uses its 'field' value as the only field."
    )
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of parallel workers (default: auto-detected).")
    ap.set_defaults(invert=True, invert_all=True)

    ns = ap.parse_args()

    cfg = ERTForwardConfig(
        pca=ns.pca, Z_path=ns.Z_path, out=ns.out,
        n_fields=ns.n_fields, field_offset=ns.field_offset, fields=ns.fields, 
        nz_full=ns.nz_full, nx_full=ns.nx_full, world_Lx=ns.world_Lx, margin=ns.margin,
        n_elec=ns.n_elec, dx_elec=ns.dx_elec, mesh_area=ns.mesh_area,
        design_type=ns.design_type, wenner_a_min=ns.wenner_a_min, wenner_a_max=ns.wenner_a_max,
        n_AB=ns.n_AB, n_MN_per_AB=ns.n_MN_per_AB, dAB_min=ns.dAB_min, dAB_max=ns.dAB_max, dMN_min=ns.dMN_min, dMN_max=ns.dMN_max,
        mode=ns.mode, noise_rel=ns.noise_rel, seed=ns.seed,
        invert=ns.invert, invert_out=ns.invert_out,
        invert_all=getattr(ns, "invert_all", True), inv_npz_out=ns.inv_npz_out, inv_save_all_png=getattr(ns, "inv_save_all_png", False),
        field_from_npz=getattr(ns, "field_from_npz", None), workers=getattr(ns, "workers", None),

    )
    out = run(cfg)
    print("Done. Output:", out)
