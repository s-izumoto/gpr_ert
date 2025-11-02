
"""
ERT surrogate pair generation with pyGIMLi (YAML-friendly), with support for
Wenner-alpha pattern on a subset of "active" electrodes chosen from a larger,
fixed sensor line (e.g., choose 16 out of 32 without moving the 32 sensors).
"""
from __future__ import annotations

import os, json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Any, Sequence, Optional
import numpy as np

# Limit BLAS/OpenMP threads per worker (avoid oversubscription)
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
    deep_rows = nz_full - nz
    deep = np.repeat(last, deep_rows, axis=1)
    return np.concatenate([shallow, deep], axis=1)

# ====== Geometry / mesh ======
def make_sensor_positions(n_elec: int, L_world: float, margin: float):
    L_inner = L_world - 2.0 * margin
    assert L_inner > 0.0, "Margin too large for world length."
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

# ====== Design helpers ======
def resolve_active_indices(n_total: int, n_active: int, policy: str, explicit: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Return a strictly increasing array of electrode indices (0-based) of length n_active,
    selected from range(n_total), without moving the physical sensors.
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
        # ensure first and last included (spread nicely)
        idx[0] = 0
        idx[-1] = n_total - 1
        return np.sort(idx)
    if policy == "first_k":
        return np.arange(n_active, dtype=np.int32)
    # fallback: uniform subset
    return np.linspace(0, n_total - 1, n_active, dtype=int)

def build_wenner_alpha_from_active(active_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Wenner-alpha quadruples over the *active* index list, respecting the order.
    Pattern: (A, M, N, B) = (i, i+s, i+2s, i+3s) with spacing s >= 1.
    Returns arrays of *global* electrode indices (into the 32-sensor line).
    """
    N = len(active_idx)
    a_list: List[int] = []
    b_list: List[int] = []
    m_list: List[int] = []
    n_list: List[int] = []
    for s in range(1, N - 2):
        # start i up to N-1-3s inclusive
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
    """
    Produce [dAB, dMN, mAB, mMN] in meters from sensor x-positions.
    """
    xa, xb, xm, xn = xs[a], xs[b], xs[m], xs[n]
    dAB = np.abs(xb - xa)
    dMN = np.abs(xn - xm)
    mAB = 0.5 * (xa + xb)
    mMN = 0.5 * (xm + xn)
    return np.stack([dAB, dMN, mAB, mMN], axis=1).astype(np.float32)

def make_scheme(sensors: Iterable[Any], a: np.ndarray, b: np.ndarray, m: np.ndarray, n: np.ndarray):
    dc = pg.DataContainerERT()
    for p in sensors:
        dc.createSensor(p)
    dc.resize(len(a))
    dc["a"] = pg.Vector(a); dc["b"] = pg.Vector(b)
    dc["m"] = pg.Vector(m); dc["n"] = pg.Vector(n)
    return dc

# ===== New array builders & utilities (0-based indices) =====
def _canonical_abmn_0b(a:int,b:int,m:int,n:int):
    # canonicalize reciprocity: (A,B,M,N) ~ (M,N,A,B); return lexicographically smallest
    dip1=(a,b); dip2=(m,n)
    if dip2 < dip1:
        dip1, dip2 = dip2, dip1
    return dip1[0], dip1[1], dip2[0], dip2[1]

def build_schlumberger_all(n_elec:int, min_gap:int=1):
    # Schlumberger: A=i, B=i+2*s; potentials half-span ah in [1, s-1]; M=i+s-ah, N=i+s+ah
    a_list=[]; b_list=[]; m_list=[]; n_list=[]
    for s in range(max(1,min_gap), (n_elec//2)+1):
        a_hi = s-1
        for ah in range(1, a_hi+1):
            for i in range(0, n_elec - 2*s):
                A = i; B = i + 2*s
                M = i + s - ah; N = i + s + ah
                if 0 <= M < N < n_elec:
                    A,B,M,N = _canonical_abmn_0b(A,B,M,N)
                    if len({A,B,M,N}) == 4:
                        a_list.append(A); b_list.append(B); m_list.append(M); n_list.append(N)
    if not a_list:
        raise RuntimeError("No Schlumberger quadruples constructed; check n_elec.")
    uniq = sorted({(a_list[i], b_list[i], m_list[i], n_list[i]) for i in range(len(a_list))})
    a,b,m,n = zip(*uniq)
    return (np.array(a, dtype=np.int32),
            np.array(b, dtype=np.int32),
            np.array(m, dtype=np.int32),
            np.array(n, dtype=np.int32))

def build_dipole_dipole_all(n_elec:int, min_gap:int=1, k_min:int=1, k_max:int=6):
    # Dipole–Dipole: dipole length s; separation k>=1: B=A+s; M=A+(k+1)s; N=A+(k+2)s
    a_list=[]; b_list=[]; m_list=[]; n_list=[]
    max_s = n_elec // 4
    for s in range(max(1, min_gap), max_s+1):
        k_hi = (n_elec // s) - 3
        if k_max is not None:
            k_hi = min(k_hi, k_max)
        for k in range(max(1,k_min), k_hi+1):
            for i in range(0, n_elec - (k+2)*s):
                A = i; B = i + s; M = i + (k+1)*s; N = i + (k+2)*s
                A,B,M,N = _canonical_abmn_0b(A,B,M,N)
                if len({A,B,M,N}) == 4:
                    a_list.append(A); b_list.append(B); m_list.append(M); n_list.append(N)
    if not a_list:
        raise RuntimeError("No Dipole–Dipole quadruples constructed; check n_elec.")
    uniq = sorted({(a_list[i], b_list[i], m_list[i], n_list[i]) for i in range(len(a_list))})
    a,b,m,n = zip(*uniq)
    return (np.array(a, dtype=np.int32),
            np.array(b, dtype=np.int32),
            np.array(m, dtype=np.int32),
            np.array(n, dtype=np.int32))

def build_gradient_all(n_elec: int, mn_k_min: int = 1, mn_k_max: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gradient 配列（0-based）:
      - 電流電極はラインの両端に固定: A=0, B=n_elec-1
      - 電位電極は M, N を内部でスライド
      - 「MN の間隔 k」を自由指定: N = M + k （k >= 1）
      - 4 電極はすべて相異（A,B,M,N は重複禁止）
    """
    A_fixed = 0
    B_fixed = n_elec - 1
    a_list, b_list, m_list, n_list = [], [], [], []

    k_hi = (n_elec - 2) if (mn_k_max is None) else min(mn_k_max, n_elec - 2)
    for k in range(max(1, mn_k_min), k_hi + 1):
        # M は [1 .. (n_elec-2-k)] にして M,N が内部＆B_fixed 未衝突
        m_lo = 1
        m_hi = (n_elec - 2) - k
        if m_hi < m_lo:
            continue
        for M in range(m_lo, m_hi + 1):
            N = M + k
            A, B = A_fixed, B_fixed
            # 4点すべて異なることを保証
            if len({A, B, M, N}) != 4:
                continue
            A, B, M, N = _canonical_abmn_0b(A, B, M, N)
            if len({A, B, M, N}) == 4:
                a_list.append(A); b_list.append(B); m_list.append(M); n_list.append(N)

    if not a_list:
        raise RuntimeError("No Gradient quadruples constructed; check n_elec or k range.")

    uniq = sorted({(a_list[i], b_list[i], m_list[i], n_list[i]) for i in range(len(a_list))})
    a, b, m, n = zip(*uniq)
    return (np.array(a, dtype=np.int32),
            np.array(b, dtype=np.int32),
            np.array(m, dtype=np.int32),
            np.array(n, dtype=np.int32))


# ====== Worker ======
def _simulate_one_field(args_tuple):
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
        act_idx = np.arange(n_elec, dtype=np.int32) if (n_active >= n_elec and active_policy != "explicit") else resolve_active_indices(n_elec, n_active, active_policy, active_indices)
        a, b, m, n = build_wenner_alpha_from_active(act_idx)
        designs = make_design_metrics(xs, a, b, m, n)
    elif pattern == "schlumberger":
        a, b, m, n = build_schlumberger_all(n_elec, min_gap=1)
        designs = make_design_metrics(xs, a, b, m, n)
    elif pattern in ("dipole-dipole", "dipole"):
        a, b, m, n = build_dipole_dipole_all(n_elec, min_gap=1)
        designs = make_design_metrics(xs, a, b, m, n)
    elif pattern == "gradient":
        a, b, m, n = build_gradient_all(n_elec, mn_k_min=1, mn_k_max=None)
        designs = make_design_metrics(xs, a, b, m, n)

    elif pattern == "all":
        a1,b1,m1,n1 = build_wenner_alpha_from_active(np.arange(n_elec, dtype=np.int32))
        a2,b2,m2,n2 = build_schlumberger_all(n_elec, min_gap=1)
        a3,b3,m3,n3 = build_dipole_dipole_all(n_elec, min_gap=1)
        a4,b4,m4,n4 = build_gradient_all(n_elec, mn_k_min=1, mn_k_max=None)

        A = np.concatenate([a1,a2,a3,a4]); B = np.concatenate([b1,b2,b3,b4])
        M = np.concatenate([m1,m2,m3,m4]); N = np.concatenate([n1,n2,n3,n4])
        tup = np.stack([A,B,M,N], axis=1)
        canon = np.array([_canonical_abmn_0b(int(r[0]),int(r[1]),int(r[2]),int(r[3])) for r in tup], dtype=np.int32)
        uniq = np.unique(canon, axis=0)
        a,b,m,n = [uniq[:,i].astype(np.int32) for i in range(4)]
        designs = make_design_metrics(xs, a, b, m, n)
    else:
        # legacy fallback
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

    # Relative Gaussian noise on rhoa
    if noise_rel and noise_rel > 0:
        rhoa = rhoa * (1.0 + (noise_rel * rng.standard_normal(size=rhoa.shape)).astype(np.float32))

    rhoa = np.maximum(rhoa.astype(np.float32), 1e-12)
    y = np.log10(rhoa).astype(np.float32)

    # Normalize designs by inner width
    D = designs.astype(np.float32)
    Dnorm = np.empty_like(D)
    Dnorm[:, 0] = D[:, 0] / L_inner
    Dnorm[:, 1] = D[:, 1] / L_inner
    Dnorm[:, 2] = (D[:, 2] - margin) / L_inner
    Dnorm[:, 3] = (D[:, 3] - margin) / L_inner

    Zrep = np.repeat(z_row[None, :], D.shape[0], axis=0).astype(np.float32)
    ABMN = np.stack([a, b, m, n], axis=1).astype(np.int32)

    return (Zrep, D, Dnorm, ABMN, y, rhoa, kfac, xs.astype(np.float32), dx_eff)

# ====== Public API ======
@dataclass
class ERTForwardConfig:
    pca: str
    Z_path: str
    out: str
    n_fields: int = 5
    field_offset: int = 0
    nz_full: int = 100
    nx_full: int = 400
    world_Lx: float = 31.0
    margin: float = 3.0
    n_elec: int = 32
    dx_elec: float = 1.0
    mesh_area: float = 0.1
    # pattern controls
    pattern: str = "all"  # "wenner-alpha" or "legacy-alpha"
    n_active_elec: int = 32        # how many electrodes are "active" inside the fixed n_elec
    active_policy: str = "every_other"  # "every_other" | "first_k" | "explicit"
    active_indices: Optional[Sequence[int]] = None
    mode: str = "2d"  # or "25d"
    noise_rel: float = 0.02
    jobs: int = 8
    chunksize: int = 1
    seed: int = 0

def run(cfg: ERTForwardConfig) -> Path:
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

    # Geometry: compute world_Lx if dx_elec given
    if cfg.dx_elec and cfg.dx_elec > 0:
        L_world = 2.0 * cfg.margin + (cfg.n_elec - 1) * cfg.dx_elec
    else:
        L_world = float(cfg.world_Lx)
    hy = L_world / float(cfg.nx_full)
    Lz = hy * float(cfg.nz_full)

    # Seeds per field
    base_rng = np.random.default_rng(cfg.seed)
    seeds = base_rng.integers(0, 2**31 - 1, size=max(1, n_fields), dtype=np.int64)

    # Build task list
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
        # 逐次：投入順の pos に格納
        for pos, t in enumerate(tasks):
            try:
                res = _simulate_one_field(t)
                results[pos] = res
            except Exception as e:
                print(f"[ERROR] simulate_one_field failed at task {pos+1}: {e}")
            if (pos+1) % 5 == 0 or (pos+1) == len(tasks):
                print(f"  done {pos+1}/{len(tasks)} (sequential)")
    else:
        # 並列：Future -> pos のマップを作って完了後に所定のスロットへ
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

    # Concatenate
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
        "n_active_elec": int(cfg.n_active_elec),
        "active_policy": cfg.active_policy,
        "mode": cfg.mode,
        "noise_rel": float(cfg.noise_rel),
        "design_desc": "[dAB, dMN, mAB, mMN] meters; Dnorm normalized by inner width",
        "jobs": int(cfg.jobs),
        "no_split": True,
    }

    out_npz = Path(cfg.out) / "ert_surrogate_all.npz"
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
    ap.add_argument("--pca", required=True)
    ap.add_argument("--Z", dest="Z_path", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--n-fields", type=int, default=5)
    ap.add_argument("--field-offset", type=int, default=0)

    ap.add_argument("--nz-full", type=int, default=100)
    ap.add_argument("--nx-full", type=int, default=400)
    ap.add_argument("--n-elec", type=int, default=32)
    ap.add_argument("--dx-elec", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--mesh-area", type=float, default=0.1)
    ap.add_argument("--mode", choices=["2d", "25d"], default="2d")
    ap.add_argument("--world-Lx", type=float, default=31.0)

    # New: pattern controls
    ap.add_argument("--pattern",
        choices=["wenner-alpha","schlumberger","dipole-dipole","gradient","all","legacy-alpha"],
        default="all")
    ap.add_argument("--n-active-elec", type=int, default=32)
    ap.add_argument("--active-policy", choices=["every_other", "first_k", "explicit"], default="every_other")
    ap.add_argument("--active-indices", type=int, nargs="*", default=None,
                    help="0-based indices into the full sensor line; used when --active-policy explicit")

    ap.add_argument("--noise-rel", type=float, default=0.02)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunksize", type=int, default=1)

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
