"""
ERT surrogate pair generation with pyGIMLi — **Wenner-alpha on a fixed sensor line**
-------------------------------------------------------------------------------
Purpose
=======
Generate supervised-learning pairs (design metrics → log10 apparent resistivity)
by forward-simulating ERT measurements with pyGIMLi over synthetic 2D fields
reconstructed from a PCA latent (Z). The script supports *Wenner-alpha* patterns
on a subset of *active* electrodes chosen from a larger, fixed sensor line—e.g.,
select 16 active electrodes from 32 physically installed sensors *without
moving sensors*. A light legacy pattern (adjacent-dipole stepping) is also
available for quick tests.

Typical workflowS
----------------
1) Load PCA pack (``joblib`` dict) with keys ``mean``, ``components``, ``nz``, ``nx``.
2) Load latent rows ``Z`` from .npy/.npz and keep the first ``k = components.shape[0]``
   columns.
3) For each field (row of ``Z``):
   - Reconstruct a shallow log10-resistivity raster via ``Z @ components + mean``.
   - Pad vertically to the requested full depth ("copy last row" padding).
   - Build a 2D world + surface sensor line; make a mesh.
   - Sample the log10 resistivity onto mesh cells and exponentiate → ``rho_cells``.
   - Build an acquisition scheme (Wenner-alpha over the *active* subset) and
     simulate with pyGIMLi to obtain apparent resistivity ``rhoa``.
   - Convert to labels ``y = log10(rhoa)`` and assemble design features.
4) Concatenate all fields and save a compressed NPZ bundle.

Outputs (.npz)
---------------
- ``Z``      : (M, k)           — latent vector per simulated row (repeated per design)
- ``D``      : (M, 4)           — raw design metrics in meters ``[dAB, dMN, mAB, mMN]``
- ``Dnorm``  : (M, 4)           — normalized by inner width ``L_inner``
- ``ABMN``   : (M, 4) int32     — global electrode indices (0-based) into the full line
- ``y``      : (M,) float32     — labels ``log10(rhoa)``
- ``rhoa``   : (M,) float32     — apparent resistivity in Ω·m (with optional relative noise)
- ``k``      : (M,) float32     — geometric factors from the scheme
- ``xs``     : (n_elec,) float32— sensor x-positions along the surface (meters)
- ``meta``   : JSON string with geometry, pattern, seeds, etc.

Key notions
-----------
- **Fixed sensor line** (length ``world_Lx``): ``n_elec`` equally spaced sensors
  from ``x = margin`` to ``x = world_Lx - margin``. We never move these.
- **Active subset**: a strictly increasing list of indices into the fixed line
  (policies: ``every_other`` [default], ``first_k``, or ``explicit`` list).
- **Wenner-alpha** over the *active* subset: quadruples (A,M,N,B) =
  ``(i, i+s, i+2s, i+3s)`` in *active-index space* with spacing ``s ≥ 1``, then
  mapped back to global 0-based indices into the 32-sensor line.
- **Design metrics**: distances and midpoints derived from sensor positions:
  ``dAB = |xB-xA|``, ``dMN = |xN-xM|``, ``mAB = (xA+xB)/2``, ``mMN = (xM+xN)/2``.
- **Normalization** (``Dnorm``): divide distances and translate midpoints so that
  all four features are in [0, 1] relative to the inner width ``L_inner``.

CLI example
-----------
>>> python measurements_warmup.py \
...   --pca pca_pack.joblib --Z latent.npz --out outputs/ \
...   --pattern wenner-alpha --n-active-elec 16 --active-policy every_other \
...   --n-fields 200 --noise-rel 0.02 --jobs 8

Notes & tips
------------
- Set ``--jobs 1`` when debugging to run sequentially (cleaner tracebacks).
- ``--mode 2d`` uses ``ert.simulate``; ``--mode 25d`` uses a simplified
  reference operator (slightly faster, fewer features).
- To pin the active subset explicitly, pass ``--active-policy explicit`` and
  ``--active-indices 0 2 4 ...`` (length must match ``--n-active-elec``).
- All BLAS/OpenMP env vars are limited to 1 thread per worker to avoid CPU
  oversubscription during ``ProcessPoolExecutor`` runs.
"""
from __future__ import annotations

import os, json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Any, Sequence, Optional
import numpy as np

# Limit BLAS/OpenMP threads per worker (helps parallel scaling)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile, shutil
from contextlib import contextmanager

# ====== Small utils ======
@contextmanager
def in_temp_workdir(base: str | None = None, prefix: str = "pg_worker_"):
    """Context: create a temporary working directory and chdir into it.
    Cleans up on exit. Useful because some pyGIMLi routines write files.
    """
    tmp = tempfile.mkdtemp(prefix=prefix, dir=base)
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        yield tmp
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

# ====== PCA helpers ======
def load_pca(p: str | Path):
    """Load a PCA pack saved via joblib.

    Expected keys in the dict: ``mean`` (1D), ``components`` (2D), ``nz``, ``nx``.
    Returns ``(mean, components, nz, nx)`` as float32 + ints.
    """
    from joblib import load as joblib_load
    mdl = joblib_load(p)
    return (
        mdl["mean"].astype(np.float32),
        mdl["components"].astype(np.float32),
        int(mdl["nz"]), int(mdl["nx"]),
    )


def load_Z_any(path: str | Path, k: int | None = None) -> np.ndarray:
    """Load latent Z from .npy/.npz/.npz-like; optionally truncate to k components.
    Supports files with a top-level array or an NPZ with a key named ``Z``.
    Ensures a 2D array (N, k').
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
    """Reconstruct shallow (nz × nx) log10-resistivity from latent Z.
    ``Z`` can be (1, k) or (N, k); returns (N, nz, nx).
    """
    Xf = (Z @ comps + mean).astype(np.float32)
    return Xf.reshape((-1, nz, nx))


def pad_deep(shallow: np.ndarray, nz_full: int) -> np.ndarray:
    """Pad the shallow reconstruction to ``nz_full`` by repeating the last row.
    Input shape: (N, nz, nx) → output: (N, nz_full, nx).
    """
    N, nz, nx = shallow.shape
    if nz_full <= nz:
        return shallow[:, :nz_full, :]
    last = shallow[:, nz - 1:nz, :]
    deep_rows = nz_full - nz
    deep = np.repeat(last, deep_rows, axis=1)
    return np.concatenate([shallow, deep], axis=1)

# ====== Geometry / mesh ======
def make_sensor_positions(n_elec: int, L_world: float, margin: float):
    """Create equally spaced surface sensor positions in [margin, L_world - margin].
    Returns (pyGIMLi positions, xs array, effective dx, inner length).
    """
    L_inner = L_world - 2.0 * margin
    assert L_inner > 0.0, "Margin too large for world length."
    dx = L_inner / float(n_elec - 1)
    xs = margin + dx * np.arange(n_elec, dtype=np.float32)
    sensors = [pg.Pos(float(x), 0.0) for x in xs]
    return sensors, xs.astype(np.float32), float(dx), float(L_inner)


def build_mesh_world(L_world: float, Lz: float, sensors: Iterable[Any], dz_under: float = 0.05,
                     area: float | None = None, quality: int = 34):
    """Create a rectangular world [0, L_world] × [0, -Lz] with vertical sticks
    under each sensor (improves near-surface mesh quality), then mesh it.
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
    """Nearest-neighbor sample a (NZ×NX) log10-resistivity raster onto mesh cells.
    Mesh x ∈ [0, L_world], z ∈ [0, -Lz]. Handles z-down indexing.
    Returns per-cell log10-resistivity.
    """
    NZ, NX = logR.shape
    xs = np.linspace(0.0, L_world, NX, dtype=np.float32)
    zs = np.linspace(0.0, -Lz, NZ, dtype=np.float32)
    cx = np.array([c.center()[0] for c in mesh.cells()], dtype=np.float32)
    cz = np.array([c.center()[1] for c in mesh.cells()], dtype=np.float32)
    ix = np.clip(np.round((cx - xs[0]) / (xs[-1] - xs[0]) * (NX - 1)).astype(int), 0, NX - 1)
    iz = np.clip(np.round((cz - zs[-1]) / (zs[0] - zs[-1]) * (NZ - 1)).astype(int), 0, NZ - 1)
    # Flip z index because the image is indexed top-to-bottom but cz is 0 at surface, negative at depth
    iz = (NZ - 1) - iz
    return logR[iz, ix]

# ====== Design helpers ======
def resolve_active_indices(n_total: int, n_active: int, policy: str, explicit: Optional[Sequence[int]] = None) -> np.ndarray:
    """Return a strictly increasing array of *global* electrode indices (0-based)
    of length ``n_active`` chosen from ``range(n_total)``.

    Policies
    --------
    - ``explicit``: take the provided ``explicit`` indices (validated and sorted)
    - ``every_other``: approximately uniform subset across the full line
    - ``first_k``: the first ``n_active`` electrodes
    - default: linspace-based uniform subset
    """
    if n_active > n_total:
        raise ValueError("n_active cannot exceed n_total sensors")
    if policy == "explicit":
        if explicit is None or len(explicit) != n_active:
            raise ValueError("When policy='explicit', provide --active-indices exactly n_active items")
        idx = np.array(sorted(set(int(i) for i in explicit)))
        if len(idx) != n_active or idx.min() < 0 or idx.max() >= n_total:
            raise ValueError("Invalid --active-indices")
        return idx.astype(np.int32)
    if policy == "every_other":
        # choose roughly uniformly: take every k-th to get n_active from n_total
        step = int(np.floor(n_total / n_active))
        if step < 1:
            step = 1
        idx = np.arange(0, n_total, step, dtype=np.int32)[:n_active]
        # ensure endpoints are included to improve spatial spread
        idx[0] = 0
        idx[-1] = n_total - 1
        return np.sort(idx)
    if policy == "first_k":
        return np.arange(n_active, dtype=np.int32)
    # fallback: uniform subset by linspace
    return np.linspace(0, n_total - 1, n_active, dtype=int)


def build_wenner_alpha_from_active(active_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build Wenner-alpha quadruples *within the active list* and map back to global indices.

    Pattern: (A, M, N, B) = (i, i+s, i+2s, i+3s) for spacing ``s ≥ 1`` and start
    index ``i`` such that all indices are inside ``active_idx``. Ensures distinct
    electrodes (by construction). Raises if the active set is too small.
    """
    N = len(active_idx)
    a_list: List[int] = []
    b_list: List[int] = []
    m_list: List[int] = []
    n_list: List[int] = []
    for s in range(1, N - 2):
        # i can go up to N-1-3s inclusive
        max_i = N - 1 - 3 * s
        if max_i < 0:
            break
        for i in range(0, max_i + 1):
            A = active_idx[i]
            M = active_idx[i + s]
            Nn = active_idx[i + 2 * s]
            B = active_idx[i + 3 * s]
            # guard distinctness (should hold by construction)
            if len({A, M, Nn, B}) < 4:
                continue
            a_list.append(A); b_list.append(B); m_list.append(M); n_list.append(Nn)
    if not a_list:
        raise RuntimeError("No Wenner-alpha quadruples constructed; check n_active.")
    return (np.array(a_list, dtype=np.int32),
            np.array(b_list, dtype=np.int32),
            np.array(m_list, dtype=np.int32),
            np.array(n_list, dtype=np.int32))


def make_design_metrics(xs: np.ndarray, a: np.ndarray, b: np.ndarray, m: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Compute design feature vectors in meters: ``[dAB, dMN, mAB, mMN]``.
    ``xs`` are sensor x-positions; ``a,b,m,n`` are global indices into ``xs``.
    """
    xa, xb, xm, xn = xs[a], xs[b], xs[m], xs[n]
    dAB = np.abs(xb - xa)
    dMN = np.abs(xn - xm)
    mAB = 0.5 * (xa + xb)
    mMN = 0.5 * (xm + xn)
    return np.stack([dAB, dMN, mAB, mMN], axis=1).astype(np.float32)


def make_scheme(sensors: Iterable[Any], a: np.ndarray, b: np.ndarray, m: np.ndarray, n: np.ndarray):
    """Create a pyGIMLi ERT data container with sensors and ABMN quadruples."""
    dc = pg.DataContainerERT()
    for p in sensors:
        dc.createSensor(p)
    dc.resize(len(a))
    dc["a"] = pg.Vector(a); dc["b"] = pg.Vector(b)
    dc["m"] = pg.Vector(m); dc["n"] = pg.Vector(n)
    return dc

# ====== Worker ======

def _simulate_one_field(args_tuple):
    """Worker: simulate a single field given all parameters (packed tuple).

    Returns a tuple
      (Zrep, D, Dnorm, ABMN, y, rhoa, kfac, xs, dx_eff)
    with shapes consistent with the number of generated designs.
    """
    ( z_row, comps, mean, nz_crop, nx_crop,
      nz_full, nx_full, L_world, Lz, n_elec, margin, mesh_area,
      mode, noise_rel, seed_base,
      pattern, n_active, active_policy, active_indices
    ) = args_tuple

    rng = np.random.default_rng(seed_base)

    # Reconstruct full field (log10 rho)
    shallow = recon_shallow_from_Z(z_row[None, :], comps, mean, nz_crop, nx_crop)  # (1, nz_crop, nx_crop)
    logR_full = pad_deep(shallow, nz_full)[0]

    # Sensors, mesh, cell model
    sensors, xs, dx_eff, L_inner = make_sensor_positions(n_elec, L_world, margin)
    mesh = build_mesh_world(L_world, Lz, sensors, dz_under=0.05,
                            area=(mesh_area if mesh_area > 0 else None), quality=34)
    rho_cells = (10.0 ** sample_logR_to_cells(logR_full, mesh, L_world, Lz)).astype(np.float32)

    # Designs
    if pattern == "wenner-alpha":
        act_idx = resolve_active_indices(n_elec, n_active, active_policy, active_indices)
        a, b, m, n = build_wenner_alpha_from_active(act_idx)
        designs = make_design_metrics(xs, a, b, m, n)
    else:
        # Fallback: simple adjacent-dipole stepping within ALL electrodes
        a_list=[]; b_list=[]; m_list=[]; n_list=[]
        for s in range(1, n_elec-2):
            max_i = n_elec - 1 - 3*s
            if max_i < 0: break
            for i in range(0, max_i+1):
                a_list.append(i); m_list.append(i+s); n_list.append(i+2*s); b_list.append(i+3*s)
        a = np.array(a_list, dtype=np.int32)
        b = np.array(b_list, dtype=np.int32)
        m = np.array(m_list, dtype=np.int32)
        n = np.array(n_list, dtype=np.int32)
        designs = make_design_metrics(xs, a, b, m, n)

    # Scheme & forward
    scheme = make_scheme(sensors, a, b, m, n)
    scheme.createGeometricFactors()

    # Use a temp working dir because some backends write to CWD
    with in_temp_workdir():
        if mode == "2d":
            data = ert.simulate(mesh=mesh, scheme=scheme, res=rho_cells,
                                noiseLevel=0.0, noiseAbs=0.0,
                                seed=int(rng.integers(0, 2**31 - 1)))
            rhoa = np.asarray(data["rhoa"], dtype=np.float32)
            kfac = np.asarray(data["k"], dtype=np.float32)
        else:
            fop = ert.ERTModellingReference()
            fop.setData(scheme); fop.setMesh(mesh)
            rhoa = np.asarray(fop.response(rho_cells), dtype=np.float32)
            kfac = np.asarray(scheme["k"], dtype=np.float32)

    # Relative Gaussian noise on rhoa (if requested)
    if noise_rel and noise_rel > 0:
        rhoa = rhoa * (1.0 + (noise_rel * rng.standard_normal(size=rhoa.shape)).astype(np.float32))

    # Log-transform labels robustly
    rhoa = np.maximum(rhoa.astype(np.float32), 1e-12)
    y = np.log10(rhoa).astype(np.float32)

    # Normalize designs by inner width
    D = designs.astype(np.float32)
    Dnorm = np.empty_like(D)
    Dnorm[:, 0] = D[:, 0] / L_inner
    Dnorm[:, 1] = D[:, 1] / L_inner
    Dnorm[:, 2] = (D[:, 2] - margin) / L_inner
    Dnorm[:, 3] = (D[:, 3] - margin) / L_inner

    # Repeat Z row to align with number of designs
    Zrep = np.repeat(z_row[None, :], D.shape[0], axis=0).astype(np.float32)
    ABMN = np.stack([a, b, m, n], axis=1).astype(np.int32)

    return (Zrep, D, Dnorm, ABMN, y, rhoa, kfac, xs.astype(np.float32), dx_eff)

# ====== Public API ======
@dataclass
class ERTForwardConfig:
    """Configuration container for ``run``.

    The ``dx_elec`` parameter, if > 0, overrides ``world_Lx`` to ensure that the
    sensor line length is consistent with the electrode spacing and the margin.
    """
    pca: str
    Z_path: str
    out: str
    n_fields: int = -1
    field_offset: int = 0
    nz_full: int = 100
    nx_full: int = 400
    world_Lx: float = 31.0
    margin: float = 3.0
    n_elec: int = 32
    dx_elec: float = 1.0
    mesh_area: float = 0.1
    # pattern controls
    pattern: str = "wenner-alpha"  # "wenner-alpha" or "legacy-alpha"
    n_active_elec: int = 16        # how many electrodes are "active" inside the fixed n_elec
    active_policy: str = "every_other"  # "every_other" | "first_k" | "explicit"
    active_indices: Optional[Sequence[int]] = None
    mode: str = "2d"  # or "25d"
    noise_rel: float = 0.02
    jobs: int = 8
    chunksize: int = 1
    seed: int = 0


def run(cfg: ERTForwardConfig) -> Path:
    """Run the forward simulation over a slice of fields and save an NPZ bundle.

    Returns the path to the saved NPZ.
    """
    out_dir = Path(cfg.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Load PCA and latent
    mean, comps, nz_crop, nx_crop = load_pca(cfg.pca)
    k_lat = comps.shape[0]
    Zall = load_Z_any(cfg.Z_path, k=k_lat)

    n_total = Zall.shape[0]
    start = max(0, int(cfg.field_offset))
    stop = n_total if cfg.n_fields <= 0 else min(n_total, start + int(cfg.n_fields))
    Zslice = Zall[start:stop, :]
    n_fields = Zslice.shape[0]
    print(f"[ERT] simulate: {n_fields} fields, k={k_lat} (idx {start}..{stop-1})")

    # Geometry: compute world_Lx if dx_elec is given (>0)
    if cfg.dx_elec and cfg.dx_elec > 0:
        L_world = 2.0 * cfg.margin + (cfg.n_elec - 1) * cfg.dx_elec
    else:
        L_world = float(cfg.world_Lx)
    hy = L_world / float(cfg.nx_full)  # horizontal pixel size
    Lz = hy * float(cfg.nz_full)       # keep aspect ratio similar to nx_full

    # Seeds per field
    base_rng = np.random.default_rng(cfg.seed)
    seeds = base_rng.integers(0, 2**31 - 1, size=max(1, n_fields), dtype=np.int64)

    # Build task list (common params packed once)
    common = (
        comps, mean, nz_crop, nx_crop,
        cfg.nz_full, cfg.nx_full, L_world, Lz,
        cfg.n_elec, cfg.margin, cfg.mesh_area,
        cfg.mode, cfg.noise_rel,
    )

    tasks = []
    field_source_idx = []
    for i in range(n_fields):
        field_source_idx.append(i)
        seed_i = int(seeds[i % len(seeds)])
        tasks.append((Zslice[i],) + common + (
            seed_i, cfg.pattern, int(cfg.n_active_elec), cfg.active_policy, tuple(cfg.active_indices) if cfg.active_indices else None
        ))
    np.save(out_dir / "field_source_idx.npy", np.array(field_source_idx, dtype=np.int32))

    # Run workers
    results = [None] * len(tasks)

    if cfg.jobs == 1:
        # Sequential mode for easier debugging; store in submission order
        for pos, t in enumerate(tasks):
            try:
                res = _simulate_one_field(t)
                results[pos] = res
            except Exception as e:
                print(f"[ERROR] simulate_one_field failed at task {pos+1}: {e}")
            if (pos+1) % 5 == 0 or (pos+1) == len(tasks):
                print(f"  done {pos+1}/{len(tasks)} (sequential)")
    else:
        # Parallel mode; map futures back to their target slots
        with ProcessPoolExecutor(max_workers=cfg.jobs) as ex:
            fut2pos = {ex.submit(_simulate_one_field, t): pos for pos, t in enumerate(tasks)}
            for i, fut in enumerate(as_completed(fut2pos), 1):
                pos = fut2pos[fut]
                try:
                    res = fut.result()
                    results[pos] = res
                except Exception as e:
                    print(f"[ERROR] worker task failed at pos {pos}: {e}")
                if i % 5 == 0 or i == len(tasks):
                    print(f"  done {i}/{len(tasks)} (parallel)")

    if not results:
        raise RuntimeError("No successful simulations; nothing to save.")

    # Concatenate results
    Z_all    = np.vstack([r[0] for r in results]).astype(np.float32)
    D_all    = np.vstack([r[1] for r in results]).astype(np.float32)
    Dn_all   = np.vstack([r[2] for r in results]).astype(np.float32)
    ABMN_all = np.vstack([r[3] for r in results]).astype(np.int32)
    y_all    = np.concatenate([r[4] for r in results]).astype(np.float32)
    rhoa_all = np.concatenate([r[5] for r in results]).astype(np.float32)
    k_all    = np.concatenate([r[6] for r in results]).astype(np.float32)
    xs       = results[0][7]
    dx_eff   = float(results[0][8])

    meta = {
        "start_index": int(start),
        "n_fields": int(n_fields),
        "k_lat": int(k_lat),
        "nz_full": int(cfg.nz_full),
        "nx_full": int(cfg.nx_full),
        "world_Lx_m": float(L_world),
        "Lz_m": float(Lz),
        "margin_m": float(cfg.margin),
        "n_elec": int(cfg.n_elec),
        "dx_eff_m": float(dx_eff),
        "pattern": cfg.pattern,
        "n_active_elec": int(getattr(cfg, 'n_active_elec', 0)),
        "active_policy": cfg.active_policy,
        "mode": cfg.mode,
        "noise_rel": float(cfg.noise_rel),
        "design_desc": "[dAB, dMN, mAB, mMN] meters; Dnorm normalized by inner width",
        "jobs": int(cfg.jobs),
        "no_split": True,
    }

    out_npz = Path(cfg.out) / "ert_surrogate_wenner.npz"
    np.savez_compressed(
        out_npz,
        Z=Z_all, D=D_all, Dnorm=Dn_all, ABMN=ABMN_all,
        y=y_all, rhoa=rhoa_all, k=k_all,
        xs=xs.astype(np.float32),
        meta=np.array([json.dumps(meta)], dtype=object),
    )

    print("Saved:", out_npz)
    print("Rows:", Z_all.shape[0], "Features: Z", Z_all.shape[1], "+ D", D_all.shape[1], "-> y=log10(rhoa)")
    return out_npz

# --- CLI glue ---
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", required=True, help="Joblib PCA pack with mean/components/nz/nx")
    ap.add_argument("--Z", dest="Z_path", required=True, help="Latent array .npy/.npz (N, k)")
    ap.add_argument("--out", required=True, help="Output directory for the NPZ bundle")

    ap.add_argument("--n-fields", type=int, default=-1, help="How many fields to simulate (<=0 → all)")
    ap.add_argument("--field-offset", type=int, default=0, help="Start index into Z rows")

    ap.add_argument("--nz-full", type=int, default=100, help="Full vertical cells after padding")
    ap.add_argument("--nx-full", type=int, default=400, help="Horizontal resolution for aspect ratio")
    ap.add_argument("--n-elec", type=int, default=32, help="Number of sensors installed on the line")
    ap.add_argument("--dx-elec", type=float, default=1.0, help="Electrode spacing (m); overrides world-Lx if >0")
    ap.add_argument("--margin", type=float, default=3.0, help="Left/right margin without sensors (m)")
    ap.add_argument("--mesh-area", type=float, default=0.1, help="Target mesh cell area (smaller → finer)")
    ap.add_argument("--mode", choices=["2d", "25d"], default="2d", help="Forward engine")
    ap.add_argument("--world-Lx", type=float, default=31.0, help="World width if dx-elec <= 0")

    # Pattern controls
    ap.add_argument("--pattern", choices=["wenner-alpha", "legacy-alpha"], default="wenner-alpha")
    ap.add_argument("--n-active-elec", type=int, default=16, help="Active subset size inside n-elec")
    ap.add_argument("--active-policy", choices=["every_other", "first_k", "explicit"], default="every_other")
    ap.add_argument("--active-indices", type=int, nargs="*", default=None,
                    help="0-based indices into the full sensor line; used when --active-policy explicit")

    ap.add_argument("--noise-rel", type=float, default=0.02, help="Relative Gaussian noise on rhoa (e.g. 0.02 = 2%)")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel workers; set 1 for debugging")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (affects noise and per-field seeds)")
    ap.add_argument("--chunksize", type=int, default=1, help="Reserved; not used by ProcessPoolExecutor")

    ns = ap.parse_args()

    cfg = ERTForwardConfig(
        pca=ns.pca,
        Z_path=ns.Z_path,
        out=ns.out,
        n_fields=ns.n_fields,
        field_offset=ns.field_offset,
        nz_full=ns.nz_full,
        nx_full=ns.nx_full,
        n_elec=ns.n_elec,
        dx_elec=ns.dx_elec,
        margin=ns.margin,
        mesh_area=ns.mesh_area,
        pattern=ns.pattern,
        n_active_elec=ns.n_active_elec,
        active_policy=ns.active_policy,
        active_indices=ns.active_indices,
        mode=ns.mode,
        world_Lx=ns.world_Lx,
        noise_rel=ns.noise_rel,
        jobs=ns.jobs,
        chunksize=ns.chunksize,
        seed=ns.seed,
    )
    out = run(cfg)
    print("Done. Output:", out)
