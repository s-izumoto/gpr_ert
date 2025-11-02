# build/invert_ops.py
from __future__ import annotations
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path
from typing import Iterable, Any, Optional, Tuple, List, Dict
import re
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# --------- pygimli helpers (lazy import) ----------
def _import_pg():
    import pygimli as pg
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    return pg, mt, ert


def make_sensor_positions(n_elec: int, L_world: float, margin: float):
    pg, _, _ = _import_pg()
    L_inner = L_world - 2.0 * margin
    if L_inner <= 0:
        raise ValueError(f"Margin too large relative to world length (L_world={L_world}, margin={margin}).")
    dx = L_inner / float(n_elec - 1)
    xs = margin + dx * np.arange(n_elec, dtype=np.float64)
    sensors = [pg.Pos(float(x), 0.0) for x in xs]
    return sensors, xs.astype(np.float64), float(dx), float(L_inner)


def build_mesh_world(L_world: float, Lz: float, sensors: Iterable[Any],
                     dz_under: float = 0.05, area: float | None = None, quality: int = 34):
    pg, mt, _ = _import_pg()
    world = mt.createWorld(start=[0.0, 0.0], end=[L_world, -Lz], worldMarker=True)
    for p in sensors:
        world.createNode(p)
        world.createNode(pg.Pos(p[0], -dz_under * Lz))
    kwargs = dict(quality=quality)
    if isinstance(area, (int, float)) and area > 0:
        kwargs["area"] = float(area)
    return mt.createMesh(world, **kwargs)


def make_scheme(sensors: Iterable[Any], a: np.ndarray, b: np.ndarray, m: np.ndarray, n: np.ndarray):
    pg, _, _ = _import_pg()
    data = pg.DataContainerERT()
    for p in sensors:
        data.createSensor(p)
    nrows = int(len(a))
    data.resize(nrows)
    data.add("a", a.astype(int))
    data.add("b", b.astype(int))
    data.add("m", m.astype(int))
    data.add("n", n.astype(int))
    return data


def invert_core(mesh, scheme, rhoa: np.ndarray):
    """Run pyGIMLi inversion and return (inv_arr, cmin, cmax, manager)."""
    _, _, ert = _import_pg()
    mgr = ert.ERTManager(verbose=False)
    data = scheme.copy()
    data["rhoa"] = rhoa.astype(float)
    data["err"] = np.full_like(rhoa, 0.03, dtype=float)  # 3%
    inv_res = mgr.invert(data, mesh=mesh, lam=20, robust=True, verbose=False)
    inv_arr = np.asarray(inv_res, dtype=float)
    cmin = float(np.nanmin(inv_arr))
    cmax = float(np.nanmax(inv_arr))
    return inv_arr, cmin, cmax, mgr


def save_images(mgr, inv_arr: np.ndarray, out_png_linear: str, out_png_log: Optional[str]):
    """Save images (linear & log color) using mgr.showResult."""
    cmin = float(np.nanmin(inv_arr))
    cmax = float(np.nanmax(inv_arr))

    # linear color
    fig, ax = plt.subplots(figsize=(6, 3))
    _ = mgr.showResult(ax=ax, cMin=cmin, cMax=cmax, cMap="Spectral_r", logScale=False)
    ax.set_title("Inverted resistivity (linear color)")
    fig.tight_layout()
    fig.savefig(out_png_linear, dpi=200)
    plt.close(fig)

    # log color (optional)
    if out_png_log:
        fig, ax = plt.subplots(figsize=(6, 3))
        _ = mgr.showResult(ax=ax, cMin=max(cmin, 1e-12), cMax=cmax, cMap="Spectral_r", logScale=True)
        ax.set_title("Inverted resistivity (log color)")
        fig.tight_layout()
        fig.savefig(out_png_log, dpi=200)
        plt.close(fig)


# ---------------- utilities ----------------
def to_zero_based(designs: np.ndarray, n_elec: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    designs = np.asarray(designs, dtype=int)
    if designs.ndim != 2 or designs.shape[1] != 4:
        raise SystemExit(f"ABMN must have shape [T,4], got {designs.shape}")

    A, B, M, N = designs.T
    mn = designs.min()
    mx = designs.max()
    if mn >= 0 and mx <= (n_elec - 1):
        base = 0
        a, b, m, n = A.copy(), B.copy(), M.copy(), N.copy()
    elif mn >= 1 and mx <= n_elec:
        base = 1
        a, b, m, n = A - 1, B - 1, M - 1, N - 1
    else:
        if (designs == n_elec).any():
            base = 1
            a, b, m, n = A - 1, B - 1, M - 1, N - 1
        else:
            base = 0 if mx <= (n_elec - 1) else 1
            if base == 1:
                a, b, m, n = A - 1, B - 1, M - 1, N - 1
            else:
                a, b, m, n = A.copy(), B.copy(), M.copy(), N.copy()

    for name, arr in zip(["a", "b", "m", "n"], [a, b, m, n]):
        if arr.min() < 0 or arr.max() > (n_elec - 1):
            raise SystemExit(f"Index out of range after base detection for {name}: min={arr.min()} max={arr.max()} n_elec={n_elec}")
    return a.astype(int), b.astype(int), m.astype(int), n.astype(int), base


def find_field_indices(npz: np.lib.npyio.NpzFile) -> List[int]:
    pat = re.compile(r"^y__field(\d{3})$")
    fields = []
    for k in npz.files:
        m = pat.match(k)
        if m:
            fields.append(int(m.group(1)))
    if not fields and "fields" in npz.files:
        arr = np.asarray(npz["fields"]).reshape(-1).astype(int).tolist()
        fields = arr
    return sorted(fields)


def load_field_npz(npz: np.lib.npyio.NpzFile, field_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    k_abmn = f"ABMN__field{field_idx:03d}"
    k_y = f"y__field{field_idx:03d}"
    if k_abmn not in npz.files:
        raise SystemExit(f"Missing key '{k_abmn}' for field {field_idx}")
    if k_y not in npz.files:
        raise SystemExit(f"Missing key '{k_y}' for field {field_idx}")
    ABMN = np.asarray(npz[k_abmn]).astype(int)
    y = np.asarray(npz[k_y]).astype(float)
    if ABMN.shape[0] != y.shape[0]:
        raise SystemExit(f"Mismatched lengths for field {field_idx}: ABMN={ABMN.shape[0]} vs y={y.shape[0]}")
    return ABMN, y


def legacy_single_series(npz: np.lib.npyio.NpzFile) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    keys = set(npz.files)
    if ("designs" in keys or "ABMN" in keys) and ("y" in keys):
        designs = np.asarray(npz["designs"] if "designs" in keys else npz["ABMN"]).astype(int)
        y = np.asarray(npz["y"]).astype(float)
        if designs.shape[0] != y.shape[0]:
            raise SystemExit(f"Mismatched lengths: designs/ABMN={designs.shape[0]} vs y={y.shape[0]}")
        return designs, y
    return None


# -------------- public API --------------
def run_inversion(
    npz_path: str | Path,
    out: str | None = None,
    out_log: str | None = None,
    out_dir: str | None = None,
    bundle_out: str | None = None,
    field_index: int | None = None,
    all_fields: bool = False,
    images_all: bool = False,
    n_elec: int = 32,
    dx_elec: float = 1.0,
    world_Lx: float = 31.0,
    margin: float = 3.0,
    nx_full: int = 400,
    nz_full: int = 100,
    mesh_area: float = 0.1,
) -> Path:
    """Main inversion pipeline. Returns path to saved bundle npz."""
    npz_path = Path(npz_path)
    z = np.load(npz_path, allow_pickle=False)

    # decide fields
    targets: List[Tuple[str, np.ndarray, np.ndarray, int]] = []
    field_ids = find_field_indices(z)
    if field_ids:
        if all_fields:
            chosen = field_ids
        elif field_index is not None:
            if field_index not in field_ids:
                raise SystemExit(f"--field-index {field_index} not found. Available fields: {field_ids}")
            chosen = [field_index]
        else:
            chosen = [field_ids[0]]
        for f in chosen:
            ABMN, y = load_field_npz(z, f)
            targets.append((f"field{f:03d}", ABMN, y, f))
    else:
        legacy = legacy_single_series(z)
        if legacy is None:
            raise SystemExit("NPZ is neither multi-field nor legacy('designs'/'ABMN' + 'y').")
        designs, y = legacy
        targets.append(("single", designs, y, -1))

    # geometry
    if dx_elec and dx_elec > 0:
        L_world = 2.0 * margin + (n_elec - 1) * dx_elec
    else:
        L_world = float(world_Lx)
    hy = L_world / float(nx_full)
    Lz = hy * float(nz_full)

    # output dirs / bundle path
    if out_dir:
        out_dir = Path(out_dir)
    elif out:
        out_dir = Path(out).parent
    else:
        out_dir = npz_path.parent
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    bundle_path = Path(bundle_out) if bundle_out else (Path(out_dir) / "inversion_bundle.npz")

    # mesh (shared)
    sensors, xs, dx_eff, L_inner = make_sensor_positions(n_elec, L_world, margin)
    mesh = build_mesh_world(L_world, Lz, sensors, dz_under=0.05,
                            area=(mesh_area if mesh_area and mesh_area > 0 else None), quality=34)

    # mesh meta
    xs_nodes = np.array([v[0] for v in mesh.nodes()], dtype=np.float32)
    zs_nodes = np.array([v[1] for v in mesh.nodes()], dtype=np.float32)
    xmin, xmax = float(xs_nodes.min()), float(xs_nodes.max())
    zmin, zmax = float(zs_nodes.min()), float(zs_nodes.max())
    L_world_eff = xmax - xmin
    Lz_eff = abs(zmax - zmin)
    cx = np.array([c.center()[0] for c in mesh.cells()], dtype=np.float32)
    cz = np.array([c.center()[1] for c in mesh.cells()], dtype=np.float32)
    cell_centers = np.stack([cx, cz], axis=1)

    bundle: Dict[str, np.ndarray] = {
        "cell_centers": cell_centers,
        "world_xmin": np.array([xmin], dtype=np.float64),
        "world_xmax": np.array([xmax], dtype=np.float64),
        "world_zmin": np.array([zmin], dtype=np.float64),
        "world_zmax": np.array([zmax], dtype=np.float64),
        "L_world":    np.array([L_world_eff], dtype=np.float64),
        "Lz":         np.array([Lz_eff], dtype=np.float64),
    }

    # image base (first field by default)
    base_linear = Path(out).with_suffix("") if out else (Path(out_dir) / "inversion_first")
    base_log = Path(out_log).with_suffix("") if out_log else None

    # run all targets
    for idx, (label, designs, y, _) in enumerate(targets):
        print(f"[run] processing {label}: T={designs.shape[0]}")
        a, b, m, n, base = to_zero_based(designs, n_elec)
        print(f"[base-detect] {label}: {'0-based' if base == 0 else '1-based'} -> using 0-based")
        for i in range(min(5, len(a))):
            print(f"  {i:02d}: a={a[i]}, b={b[i]}, m={m[i]}, n={n[i]}")

        rhoa = np.power(10.0, y).astype(float)
        rhoa = np.maximum(rhoa, 1e-12)

        scheme = make_scheme(sensors, a, b, m, n)
        scheme.createGeometricFactors()

        inv_arr, cmin, cmax, mgr = invert_core(mesh, scheme, rhoa)

        # images (default: only first field)
        if images_all or idx == 0:
            out_png_linear = Path(str(base_linear) + (f"_{label}" if images_all else "") + ".png")
            if base_log:
                out_png_log = Path(str(base_log) + (f"_{label}" if images_all else "") + ".png")
            else:
                out_png_log = Path(out_dir) / (("inversion_log_" + label + ".png") if images_all else "inversion_first_log.png")
            save_images(mgr, inv_arr, str(out_png_linear), str(out_png_log))
            print(f"[saved images] {out_png_linear} , {out_png_log}")

        # bundle
        suffix = f"__{label}"
        bundle[f"inv_rho_cells{suffix}"] = np.asarray(inv_arr, dtype=np.float64)
        bundle[f"cmin{suffix}"] = np.array([cmin], dtype=np.float64)
        bundle[f"cmax{suffix}"] = np.array([cmax], dtype=np.float64)
        bundle[f"abmn{suffix}"] = np.stack([scheme['a'], scheme['b'], scheme['n'], scheme['m']], axis=1).astype(np.int32)[:, [0, 1, 3, 2]]
        bundle[f"rhoa{suffix}"] = np.asarray(rhoa, dtype=np.float64)

    np.savez_compressed(bundle_path, **bundle)
    print(f"[bundle saved] {bundle_path}")
    print("[all done]")
    return bundle_path


# -------------- CLI --------------
def _build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="seq_log .npz")
    ap.add_argument("--out", help="PNG path (linear)")
    ap.add_argument("--out-log", help="PNG path (log-color)")
    ap.add_argument("--out-dir", help="Directory to write outputs")
    ap.add_argument("--bundle-out", help="ONE bundled inversion NPZ path")
    ap.add_argument("--field-index", type=int)
    ap.add_argument("--all-fields", action="store_true")
    ap.add_argument("--images-all", action="store_true")
    ap.add_argument("--n-elec", type=int, default=32)
    ap.add_argument("--dx-elec", type=float, default=1.0)
    ap.add_argument("--world-Lx", type=float, default=31.0)
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--nx-full", type=int, default=400)
    ap.add_argument("--nz-full", type=int, default=100)
    ap.add_argument("--mesh-area", type=float, default=0.1)
    return ap


def main():
    ap = _build_argparser()
    args = ap.parse_args()
    run_inversion(
        npz_path=args.npz,
        out=args.out,
        out_log=args.out_log,
        out_dir=args.out_dir,
        bundle_out=args.bundle_out,
        field_index=args.field_index,
        all_fields=args.all_fields,
        images_all=args.images_all,
        n_elec=args.n_elec,
        dx_elec=args.dx_elec,
        world_Lx=args.world_Lx,
        margin=args.margin,
        nx_full=args.nx_full,
        nz_full=args.nz_full,
        mesh_area=args.mesh_area,
    )


if __name__ == "__main__":
    main()
