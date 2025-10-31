
# -*- coding: utf-8 -*-
"""
invert_from_npz_with_ertgeom_autobase.py

- Reads seq_log_fieldXXX.npz (keys: 'designs' [T,4], 'y' [T])
- Auto-detects whether ABMN indices are 0-based or 1-based and converts to 0-based
- Uses ert_wenner_invert.py-compatible world & sensor geometry
- Runs pyGIMLi ERT inversion and saves PNG

Usage:
  python invert_from_npz_with_ertgeom_autobase.py \
    --npz ./gpr_seq_logs/<run>/seq_log_field000.npz \
    --out ./inversion_field000.png \
    --n-elec 32 --dx-elec 1.0 --margin 3.0 --nx-full 400 --nz-full 100 --mesh-area 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def _import_pg():
    import pygimli as pg
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    return pg, mt, ert

def make_sensor_positions(n_elec: int, L_world: float, margin: float):
    pg, _, _ = _import_pg()
    L_inner = L_world - 2.0 * margin
    if L_inner <= 0:
        raise ValueError("Margin too large relative to world length.")
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

def invert_and_save_image(mesh, scheme, rhoa: np.ndarray,
                          out_png_linear: str,
                          out_png_log: Optional[str] = None) -> tuple[str, Optional[str]]:
    _, _, ert = _import_pg()
    mgr = ert.ERTManager(verbose=False)

    data = scheme.copy()
    data["rhoa"] = rhoa.astype(float)
    data["err"] = np.full_like(rhoa, 0.03, dtype=float)

    inv_res = mgr.invert(data, mesh=mesh, lam=20, robust=True, verbose=False)
    inv_arr = np.asarray(inv_res, dtype=float)

    # 線形スケール（従来どおり）
    cmin = float(np.nanmin(inv_arr))
    cmax = float(np.nanmax(inv_arr))
    fig, ax = plt.subplots(figsize=(6, 3))
    _ = mgr.showResult(ax=ax, cMin=cmin, cMax=cmax, cMap="Spectral_r", logScale=False)
    ax.set_title("Inverted resistivity (linear color)")
    fig.tight_layout()
    fig.savefig(out_png_linear, dpi=200)
    plt.close(fig)

    # ログカラースケール（色だけ log、凡例は Ωm）
    out_log_path = None
    if out_png_log:
        # cMin は正に制限
        cmin_log = max(cmin, 1e-12)
        fig, ax = plt.subplots(figsize=(6, 3))
        _ = mgr.showResult(ax=ax, cMin=cmin_log, cMax=cmax, cMap="Spectral_r", logScale=True)
        ax.set_title("Inverted resistivity (log color)")
        fig.tight_layout()
        fig.savefig(out_png_log, dpi=200)
        plt.close(fig)
        out_log_path = out_png_log

    return out_png_linear, out_log_path


def to_zero_based(designs: np.ndarray, n_elec: int):
    """Auto-detect base and convert to 0-based safely."""
    A, B, M, N = designs.T
    mn = designs.min()
    mx = designs.max()
    # Cases:
    #  (1) Already 0-based if mn >= 0 and mx <= n_elec-1
    #  (2) 1-based if mn >= 1 and mx <= n_elec
    #  (3) Mixed/unknown: try heuristic, else raise
    if mn >= 0 and mx <= (n_elec - 1):
        base = 0
        a, b, m, n = A.copy(), B.copy(), M.copy(), N.copy()
    elif mn >= 1 and mx <= n_elec:
        base = 1
        a, b, m, n = A - 1, B - 1, M - 1, N - 1
    else:
        # Heuristic: if any equals n_elec -> likely 1-based
        if (designs == n_elec).any():
            base = 1
            a, b, m, n = A - 1, B - 1, M - 1, N - 1
        else:
            # If any negative exists, assume already 0-based and just clip-check
            if (designs < 0).any():
                base = 0
                a, b, m, n = A.copy(), B.copy(), M.copy(), N.copy()
            else:
                # Last resort: subtract 1 if mx == n_elec, else use as-is
                base = 0 if mx <= (n_elec - 1) else 1
                if base == 1:
                    a, b, m, n = A - 1, B - 1, M - 1, N - 1
                else:
                    a, b, m, n = A.copy(), B.copy(), M.copy(), N.copy()
    # Validate
    for name, arr in zip(["a","b","m","n"], [a,b,m,n]):
        if arr.min() < 0 or arr.max() > (n_elec - 1):
            raise SystemExit(f"Index out of range after base detection for {name}: "
                             f"min={arr.min()} max={arr.max()} n_elec={n_elec}")
    return a.astype(int), b.astype(int), m.astype(int), n.astype(int), base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="seq_log_fieldXXX.npz (must contain 'designs' and 'y')")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--out-log", default=None, help="Output PNG path for log-color image (optional)")
    ap.add_argument("--n-elec", type=int, default=32)
    ap.add_argument("--dx-elec", type=float, default=1.0)
    ap.add_argument("--world-Lx", type=float, default=31.0)
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--nx-full", type=int, default=400)
    ap.add_argument("--nz-full", type=int, default=100)
    ap.add_argument("--mesh-area", type=float, default=0.1)
    args = ap.parse_args()

    out_png_linear = str(Path(args.out))
    # out-log が未指定なら自動命名: foo.png → foo_log.png
    if args.out_log is None:
        p = Path(out_png_linear)
        out_png_log = str(p.with_name(p.stem + "_log" + p.suffix))
    else:
        out_png_log = str(Path(args.out_log))

    d = np.load(Path(args.npz), allow_pickle=True)
    if "designs" not in d or "y" not in d:
        raise SystemExit("NPZ must contain 'designs' ([T,4] ABMN) and 'y' ([T])")
    designs = d["designs"].astype(int)
    y = d["y"].astype(float)
    if designs.shape[0] != y.shape[0]:
        raise SystemExit(f"Mismatched lengths: designs={designs.shape[0]} vs y={y.shape[0]}")

    # Build world geometry identical to ert_wenner_invert.py policy
    if args.dx_elec and args.dx_elec > 0:
        L_world = 2.0 * args.margin + (args.n_elec - 1) * args.dx_elec
    else:
        L_world = float(args.world_Lx)
    hy = L_world / float(args.nx_full)
    Lz = hy * float(args.nz_full)

    sensors, xs, dx_eff, L_inner = make_sensor_positions(args.n_elec, L_world, args.margin)
    mesh = build_mesh_world(L_world, Lz, sensors, dz_under=0.05,
                            area=(args.mesh_area if args.mesh_area > 0 else None), quality=34)

    # Auto-detect base and convert
    a, b, m, n, base = to_zero_based(designs, args.n_elec)
    print(f"[base-detect] designs appear to be {'0-based' if base==0 else '1-based'}; using 0-based indices.")
    print("[sanity] first 5 ABMN (0-based):")
    for i in range(min(5, len(a))):
        print(f"  {i:02d}: a={a[i]}, b={b[i]}, m={m[i]}, n={n[i]}")

    rhoa = np.power(10.0, y).astype(float)
    rhoa = np.maximum(rhoa, 1e-12)

    scheme = make_scheme(sensors, a, b, m, n)
    scheme.createGeometricFactors()

    out_png = str(Path(args.out))
    print("[mesh] cells:", mesh.cellCount(), "nodes:", mesh.nodeCount())
    print("[mesh] area mean:", np.mean([c.size() for c in mesh.cells()]))
    print("[mesh] bbox x=[%.3f, %.3f] z=[%.3f, %.3f]" % (
        min(n.pos()[0] for n in mesh.nodes()),
        max(n.pos()[0] for n in mesh.nodes()),
        min(n.pos()[1] for n in mesh.nodes()),
        max(n.pos()[1] for n in mesh.nodes()),
    ))

    invert_and_save_image(mesh, scheme, rhoa, out_png_linear, out_png_log)
    print("[done] saved inversion image:", out_png)

if __name__ == "__main__":
    main()
