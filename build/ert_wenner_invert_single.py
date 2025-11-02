
"""
ERT surrogate pair generation (Wenner-ready) + optional inversion image (pyGIMLi).
- Uses PCA (mean, components, nz, nx) to reconstruct log10 resistivity fields from Z.
- Generates ABMN either as: (a) Wenner (A--a-->M--a-->N--a-->B) or (b) random AB/MN with constraints.
- Simulates rho_a with pyGIMLi and stores Z, D (meters), Dnorm ([0,1]) , ABMN, y=log10(rho_a), rho_a, k.
- Optionally inverts the *first* field's data and saves a PNG.

CLI example:
python ert_wenner_invert_fixed.py --pca pca_model.joblib --Z Z.npy --out outputs_wenner \
  --design-type wenner --wenner-a-min 1 --wenner-a-max 10 --invert
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

# Limit BLAS/OpenMP threads per worker (avoid oversubscription)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

# ==== Config ====
@dataclass
class ERTForwardConfig:
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

# ==== Small utils ====
class TempDir:
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
    from joblib import load as joblib_load
    mdl = joblib_load(p)
    return (
        mdl["mean"].astype(np.float32),
        mdl["components"].astype(np.float32),
        int(mdl["nz"]), int(mdl["nx"]),
    )

def load_Z_any(path: str | Path, k: int | None = None) -> np.ndarray:
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
    Xf = (Z @ comps + mean).astype(np.float32)
    return Xf.reshape((-1, nz, nx))

def pad_deep(shallow: np.ndarray, nz_full: int) -> np.ndarray:
    N, nz, nx = shallow.shape
    if nz_full <= nz:
        return shallow[:, :nz_full, :]
    last = shallow[:, nz - 1:nz, :]
    deep = np.repeat(last, nz_full - nz, axis=1)
    return np.concatenate([shallow, deep], axis=1)

# ==== Geometry / mesh ====
def make_sensor_positions(n_elec: int, L_world: float, margin: float):
    L_inner = L_world - 2.0 * margin
    if L_inner <= 0: raise ValueError("Margin too large for world length.")
    dx = L_inner / float(n_elec - 1)
    xs = margin + dx * np.arange(n_elec, dtype=np.float32)
    sensors = [pg.Pos(float(x), 0.0) for x in xs]
    return sensors, xs.astype(np.float32), float(dx), float(L_inner)

def build_mesh_world(L_world: float, Lz: float, sensors: Iterable[Any], dz_under: float = 0.05,
                     area: float | None = None, quality: int = 34):
    world = mt.createWorld(start=[0.0, 0.0], end=[L_world, -Lz], worldMarker=True)
    for p in sensors:
        world.createNode(p)
        world.createNode(pg.Pos(p[0], -dz_under * Lz))
    kwargs = dict(quality=quality)
    if isinstance(area, (int, float)) and area > 0:
        kwargs["area"] = float(area)
    return mt.createMesh(world, **kwargs)

def sample_logR_to_cells(logR: np.ndarray, mesh: pg.Mesh, L_world: float, Lz: float) -> np.ndarray:
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
    """All valid Wenner ABMN for a line of n_elec; A--a-->M--a-->N--a-->B."""
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
    """Return unique (a,b,m,n) and snapped designs [dAB, dMN, mAB, mMN] in meters."""
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
    dc = pg.DataContainerERT()
    for p in sensors:
        dc.createSensor(p)
    dc.resize(len(a))
    dc["a"] = pg.Vector(a); dc["b"] = pg.Vector(b)
    dc["m"] = pg.Vector(m); dc["n"] = pg.Vector(n)
    return dc

# ==== Inversion helper ====
# ==== Inversion helper ====
def invert_and_save_image(mesh: pg.Mesh,
                          scheme: pg.DataContainerERT,
                          rhoa: np.ndarray,
                          out_png_linear: str,
                          out_png_log: Optional[str] = None) -> tuple[str, Optional[str]]:
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    mgr = ert.ERTManager()

    # データをコピーして最小限のエラーを与える（pyGIMLiのinvertが安定）
    data = scheme.copy()
    data["rhoa"] = rhoa
    data["err"] = np.full_like(rhoa, 0.03, dtype=float)  # 例: 3% 相対誤差

    # 反演
    inv_res = mgr.invert(data, mesh=mesh, lam=20, robust=True, verbose=False)
    inv_arr = np.asarray(inv_res, dtype=float)
    cmin = float(np.nanmin(inv_arr))
    cmax = float(np.nanmax(inv_arr))

    # --- linear color 画像 ---
    fig, ax = plt.subplots(figsize=(6, 3))
    _ = mgr.showResult(ax=ax, cMin=cmin, cMax=cmax, cMap="Spectral_r", logScale=False)
    ax.set_title("Inverted resistivity (linear color)")
    fig.tight_layout()
    fig.savefig(out_png_linear, dpi=200)
    plt.close(fig)

    # --- log color 画像（任意） ---
    out_log_path = None
    if out_png_log:
        cmin_log = max(cmin, 1e-12)  # log表示の下限は正に
        fig, ax = plt.subplots(figsize=(6, 3))
        _ = mgr.showResult(ax=ax, cMin=cmin_log, cMax=cmax, cMap="Spectral_r", logScale=True)
        ax.set_title("Inverted resistivity (log color)")
        fig.tight_layout()
        fig.savefig(out_png_log, dpi=200)
        plt.close(fig)
        out_log_path = out_png_log

    # --- ここでNPZサイドカーを保存（必ず関数の中！） ---
    try:
        npz_path = str(Path(out_png_linear).with_suffix(".npz"))
        # セル中心
        cx = np.array([c.center()[0] for c in mesh.cells()], dtype=np.float32)
        cz = np.array([c.center()[1] for c in mesh.cells()], dtype=np.float32)
        # ワールド境界
        xs = np.array([v[0] for v in mesh.nodes()], dtype=np.float32)
        zs = np.array([v[1] for v in mesh.nodes()], dtype=np.float32)
        xmin, xmax = float(xs.min()), float(xs.max())
        zmin, zmax = float(zs.min()), float(zs.max())
        L_world = xmax - xmin
        Lz = abs(zmax - zmin)
        # ABMN と rhoa
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
    return D, Dnorm, ABMN, y, rhoa, kfac, xs.astype(np.float32), dx_eff, scheme, mesh


def _parse_fields(s: str) -> list[int]:
    """
    '0,4,9' 形式の文字列から 0-based の整数リストを返す。
    将来的に '0-3,7' などのレンジ表記に拡張するならここで対応させる。
    """
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
        # レンジ 'a-b' に軽く対応（任意）
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))
        else:
            out.append(int(p))
    # 一意・昇順
    return sorted(set(out))

def _read_fields_from_seqnpz(path: str | os.PathLike) -> list[int]:
    """
    gpr_seq_core の単体ログ(seq_log.npz) / バンドル(seq_logs_bundle.npz)から
    フィールド番号のリストを返す。

    優先順:
      - 'field' (スカラー)      → [field]
      - 'fields' (配列)         → list(fields)
      - 'field_index'/'field_id'→ [その値]
      - キー末尾 '__fieldNNN'   → 重複排除して昇順の一覧
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
        # 最後の保険: __fieldNNN をキー名から拾う
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
    out_dir = Path(cfg.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Load PCA and latent
    mean, comps, nz_crop, nx_crop = load_pca(cfg.pca)
    k_lat = comps.shape[0]
    Zall = load_Z_any(cfg.Z_path, k=k_lat)

    # === Field selection ===
    n_total = Zall.shape[0]

    field_from_npz = getattr(cfg, "field_from_npz", None)
    if field_from_npz not in (None, "", False):
        # ★ 複数OK
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
        # ← 従来どおり（明示 fields の複数処理）
        if isinstance(cfg.fields, (list, tuple)):
            fields = sorted(set(int(i) for i in cfg.fields))
        else:
            fields = _parse_fields(cfg.fields)
        bad = [i for i in fields if i < 0 or i >= n_total]
        if bad:
            raise ValueError(f"fieldsに不正なインデックスがあります: {bad} (総数 n_total={n_total})")
        Zslice   = Zall[fields, :]
        n_fields = len(fields)
        print(f"[ERT] simulate: {n_fields} fields (explicit) k={k_lat} indices={fields}")
        start, stop = (min(fields) if fields else 0), (max(fields) + 1 if fields else 0)

    else:
        # ← 従来どおり（連続スライス）
        start = max(0, int(cfg.field_offset))
        stop  = n_total if int(cfg.n_fields) <= 0 else min(n_total, start + int(cfg.n_fields))
        fields   = list(range(start, stop))
        Zslice   = Zall[start:stop, :]
        n_fields = len(fields)
        print(f"[ERT] simulate: {n_fields} fields, k={k_lat} (idx {start}..{stop-1})")

    # Geometry: compute world_Lx if dx_elec given
    if cfg.dx_elec and cfg.dx_elec > 0:
        L_world = 2.0 * cfg.margin + (cfg.n_elec - 1) * cfg.dx_elec
    else:
        L_world = float(cfg.world_Lx)
    hy = L_world / float(cfg.nx_full)
    Lz = hy * float(cfg.nz_full)

    # Seeds
    base_rng = np.random.default_rng(cfg.seed)
    seeds = base_rng.integers(0, 2**31 - 1, size=max(1, n_fields), dtype=np.int64)

    # Accumulators
    Z_all, D_all, Dn_all, ABMN_all = [], [], [], []
    y_all, rhoa_all, k_all = [], [], []
    schemes = []
    xs_ref = None; dx_eff_ref = None
    field_ids = []  # 実フィールド番号を保持（0..n_total-1）

    for i_local in range(n_fields):
        field_id = fields[i_local]        # ← 実際のフィールド番号
        z_row = Zslice[i_local]

        D, Dn, ABMN, y, rhoa, kfac, xs, dx_eff, scheme, mesh = simulate_one_field(
            z_row, comps, mean, nz_crop, nx_crop, cfg, L_world, Lz, int(seeds[i_local % len(seeds)])
        )

        # Repeat Z per measurement row
        Zrep = np.repeat(z_row[None, :], D.shape[0], axis=0).astype(np.float32)
        Z_all.append(Zrep); D_all.append(D); Dn_all.append(Dn)
        ABMN_all.append(ABMN); y_all.append(y); rhoa_all.append(rhoa); k_all.append(kfac)
        schemes.append((scheme, mesh))
        if xs_ref is None:
            xs_ref = xs; dx_eff_ref = dx_eff
        # 実フィールド番号を書き込む
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
        "selected_fields": fields,            # ← 追加：明示保持
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

    # Optional inversion image on "最初に選ばれた実フィールド" で評価
    # === Inversion: 選ばれた全フィールドで実行（デフォルト）
    if cfg.invert and len(schemes) > 0:
        try:
            inv_log_list, inv_rho_list, inv_field_ids = [], [], []
            png_paths = []
            inv_save_all_png = bool(getattr(cfg, "inv_save_all_png", False))
            inv_npz_out = getattr(cfg, "inv_npz_out", None)

            for i_local, (scheme_i, mesh_i) in enumerate(schemes):
                field_id = fields[i_local]
                mask_i = (field_ids == field_id)
                rhoa_i = rhoa_all[mask_i]

                # --- 反演 ---
                mgr = ert.ERTManager()
                data = scheme_i.copy()
                data["rhoa"] = rhoa_i

                # --- エラー割当（相対 + フロア） ---
                rel = float(getattr(cfg, "inv_rel_err", getattr(cfg, "noise_rel", 0.03)))
                data["err"] = np.full_like(rhoa_i, rel, dtype=np.float32)
                #print(f"[invert] field={field_id} err(min/max/mean)={err.min():.3g}/{err.max():.3g}/{err.mean():.3g}")

                inv_rho = mgr.invert(data, mesh=mesh_i, lam=20, robust=True, verbose=False)
                inv_rho = np.asarray(inv_rho, dtype=np.float32)
                inv_log = np.log10(inv_rho, dtype=np.float32)

                inv_field_ids.append(np.int32(field_id))
                inv_rho_list.append(inv_rho)
                inv_log_list.append(inv_log)

                # --- PNG 保存（既定は先頭のみ。全保存したい場合は cfg.inv_save_all_png=True を渡す） ---
                save_png_this = inv_save_all_png or (i_local == 0)
                if save_png_this:
                    # 先頭のみ明示パスがあればそれを使う。以降は fieldID を埋めた自動名
                    if (i_local == 0) and getattr(cfg, "invert_out", None):
                        out_lin = Path(cfg.invert_out)
                    else:
                        out_lin = (out_dir / f"inversion_field{field_id:03d}.png")
                    out_log = out_lin.with_name(out_lin.stem + "_log" + out_lin.suffix)

                    # 線形/対数の2枚を保存
                    # 線形スケール画像
                    fig, ax = plt.subplots(figsize=(6, 3))
                    _ = mgr.showResult(ax=ax, model=inv_rho, logScale=False)
                    ax.set_title(f"Inverted resistivity (linear) field{field_id:03d}")
                    fig.tight_layout()
                    fig.savefig(out_lin, dpi=200)
                    plt.close(fig)

                    # 対数スケール画像（log10を自分で取って渡す）
                    fig, ax = plt.subplots(figsize=(6, 3))
                    _ = mgr.showResult(ax=ax, model=np.log10(inv_rho), logScale=False)
                    ax.set_title(f"Inverted resistivity (log10 values) field{field_id:03d}")
                    fig.tight_layout()
                    fig.savefig(out_log, dpi=200)
                    plt.close(fig)


                    png_paths.extend([str(out_lin), str(out_log)])

            # --- 反演結果を一つの NPZ に束ねて保存 ---
            if len(inv_rho_list) > 0:
                bundle_path = Path(inv_npz_out) if inv_npz_out else (out_dir / "inversions_bundle.npz")
                save_dict = {}
                for fid, logv, rhov in zip(inv_field_ids, inv_log_list, inv_rho_list):
                    save_dict[f"inv_log_cells__field{int(fid):03d}"] = logv
                    save_dict[f"inv_rho_cells__field{int(fid):03d}"] = rhov

                # ===== ここから【追加】: メッシュ・座標・ABMN/rhoa を格納 =====
                # メッシュは全フィールドで同一の想定なので 0 番だけ採用
                mesh0 = schemes[0][1]  # schemes は (scheme_i, mesh_i) のリスト
                cx = np.array([c.center()[0] for c in mesh0.cells()], dtype=np.float32)
                cz = np.array([c.center()[1] for c in mesh0.cells()], dtype=np.float32)
                xs = np.array([v[0] for v in mesh0.nodes()], dtype=np.float32)
                zs = np.array([v[1] for v in mesh0.nodes()], dtype=np.float32)
                xmin, xmax = float(xs.min()), float(xs.max())
                zmin, zmax = float(zs.min()), float(zs.max())
                L_world = xmax - xmin
                Lz = abs(zmax - zmin)

                save_dict["cell_centers"] = np.stack([cx, cz], axis=1)
                save_dict["world_xmin"] = xmin
                save_dict["world_xmax"] = xmax
                save_dict["world_zmin"] = zmin
                save_dict["world_zmax"] = zmax
                save_dict["L_world"] = float(L_world)
                save_dict["Lz"] = float(Lz)

                # 各フィールドの ABMN と rhoa も保存（後処理が楽）
                for i_local, (scheme_i, _mesh_i) in enumerate(schemes):
                    fid = int(inv_field_ids[i_local])
                    abmn_i = np.stack([scheme_i["a"], scheme_i["b"], scheme_i["m"], scheme_i["n"]], axis=1).astype(np.int32)
                    mask_i = (field_ids == fid)
                    rhoa_i = rhoa_all[mask_i]
                    save_dict[f"abmn__field{fid:03d}"] = abmn_i
                    save_dict[f"rhoa__field{fid:03d}"] = np.asarray(rhoa_i, dtype=np.float64)
                # ===== 追加ここまで =====

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
                    help="明示的に反演を無効化したいときに使用")
    ap.add_argument(
        "--fields",
        type=str, default=None,
        help="0-based のフィールド番号をカンマ区切りで指定 (例: 0,4,9)。指定があれば --n-fields/--field-offset を無視。"
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

    # 既定値（重要：デフォルトで “全フィールド反演” にする）
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
        field_from_npz=getattr(ns, "field_from_npz", None),
    )
    out = run(cfg)
    print("Done. Output:", out)
