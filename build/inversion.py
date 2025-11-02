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
# === 追加 import ===
from concurrent.futures import ProcessPoolExecutor, as_completed

def _invert_job(payload):
    """子プロセス側：単一フィールドを逆解析して結果を返す（画像は作らない）"""
    # 内部BLASの並列を殺して再現性＆安定性能
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # ---- 受け取り ----
    label          = payload["label"]
    designs        = payload["designs"]
    y              = payload["y"]
    n_elec         = payload["n_elec"]
    L_world        = payload["L_world"]
    margin         = payload["margin"]
    mesh_area      = payload["mesh_area"]

    # 幾何（各プロセスで同一手順・同一乱数不使用 ⇒ 決定論的）
    sensors, xs, dx_eff, L_inner = make_sensor_positions(n_elec, L_world, margin)
    mesh = build_mesh_world(L_world, payload["Lz"], sensors,
                            dz_under=0.05,
                            area=(mesh_area if mesh_area and mesh_area > 0 else None), quality=34)

    # ABMNを0始まりに
    a, b, m, n, base = to_zero_based(designs, n_elec)

    # ρa（log10(y) → linear Ωm）
    rhoa = np.power(10.0, y).astype(float)
    rhoa = np.maximum(rhoa, 1e-12)

    # 測線作成と幾何係数
    scheme = make_scheme(sensors, a, b, m, n)
    scheme.createGeometricFactors()

    # 逆解析
    inv_arr, cmin, cmax, _mgr = invert_core(mesh, scheme, rhoa)

    # 必要な成果だけ返す（画像生成のための mgr は親で再計算せず、画像は親の最初フィールドのみで作る方針）
    res = {
        "label": label,
        "inv_arr": np.asarray(inv_arr, dtype=np.float64),
        "cmin": float(cmin),
        "cmax": float(cmax),
        "abmn": np.stack([scheme['a'], scheme['b'], scheme['n'], scheme['m']], axis=1).astype(np.int32)[:, [0,1,3,2]],
        "rhoa": np.asarray(rhoa, dtype=np.float64),
    }
    return res


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
    """
    y__field*, ABMN__field* の両方を走査し、フィールド番号を桁数に依らず抽出。
    例: y__field7, y__field007, ABMN__field12, ABMN__field0012 など。
    """
    pat = re.compile(r"^(?:y|ABMN)__field(\d+)$")  # ← \d+ で桁数フリー & y/ABMN 両対応
    fields: List[int] = []
    for k in npz.files:
        m = pat.match(k)
        if m:
            s = m.group(1)                  # '7', '007', '0012' など
            v = int(s.lstrip("0") or "0")   # 先頭0を許容し整数化（'000'→0）
            fields.append(v)

    # 重複除去
    fields = sorted(set(fields))

    # フォールバック: 'fields' 配列キーがあればそれを使う
    if not fields and "fields" in npz.files:
        arr = np.asarray(npz["fields"]).reshape(-1).astype(int).tolist()
        fields = arr

    return fields

def _pick_field_key(npz: np.lib.npyio.NpzFile, base: str, field_idx: int) -> str:
    """
    base='y__field' or 'ABMN__field'
    候補: 03桁, 非ゼロ埋め, 先頭0ありの任意桁を正規表現で検索、の順で探す。
    """
    candidates = [f"{base}{field_idx:03d}", f"{base}{field_idx}"]
    for k in candidates:
        if k in npz.files:
            return k
    # 最後に 0*<idx> を許容する正規表現で探索
    pat = re.compile(rf"^{re.escape(base)}0*{field_idx}$")
    for k in npz.files:
        if pat.match(k):
            return k
    raise SystemExit(f"Missing key for base={base!r} field_idx={field_idx}. Tried {candidates} and regex {pat.pattern}.")


def load_field_npz(npz: np.lib.npyio.NpzFile, field_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    # ここをゼロ埋め固定から柔軟検索に変更
    k_abmn = _pick_field_key(npz, "ABMN__field", field_idx)
    k_y    = _pick_field_key(npz, "y__field",    field_idx)

    ABMN = np.asarray(npz[k_abmn]).astype(int)
    y    = np.asarray(npz[k_y]).astype(float)
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
    workers: int = 1,
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

    sensors, xs, dx_eff, L_inner = make_sensor_positions(n_elec, L_world, margin)
    mesh = build_mesh_world(L_world, Lz, sensors, dz_under=0.05,
                            area=(mesh_area if mesh_area and mesh_area > 0 else None), quality=34)

    # メッシュ・メタは単スレと同じ手順で作成（＝NPZ完全一致）
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

    # 画像用のベースパス（既存ロジック維持）
    base_linear = Path(out).with_suffix("") if out else (Path(out_dir) / "inversion_first")
    base_log = Path(out_log).with_suffix("") if out_log else None

    # === 並列実行の準備 ===
    # 最初のフィールドは親プロセスで実行（画像生成の仕様を単スレと完全一致させるため）
    first_label, first_designs, first_y, _ = targets[0]
    a0, b0, m0, n0, base0 = to_zero_based(first_designs, n_elec)
    rhoa0 = np.maximum(np.power(10.0, first_y).astype(float), 1e-12)
    scheme0 = make_scheme(sensors, a0, b0, m0, n0)
    scheme0.createGeometricFactors()
    inv_arr0, cmin0, cmax0, mgr0 = invert_core(mesh, scheme0, rhoa0)

    # 画像（単スレ版と同じ：デフォは最初のフィールドのみ）
    if images_all:
        # images_all を要求された場合でもNPZの同一性には影響なし
        out_png_linear = Path(str(base_linear) + f"_{first_label}.png")
        out_png_log = Path(str(base_log) + f"_{first_label}.png") if base_log else (Path(out_dir) / f"inversion_log_{first_label}.png")
    else:
        out_png_linear = Path(str(base_linear) + ".png")
        out_png_log = Path(str(base_log) + ".png") if base_log else (Path(out_dir) / "inversion_first_log.png")
    save_images(mgr0, inv_arr0, str(out_png_linear), str(out_png_log))

    # 最初のフィールドを bundle に格納（単スレと同順）
    suffix0 = f"__{first_label}"
    bundle[f"inv_rho_cells{suffix0}"] = np.asarray(inv_arr0, dtype=np.float64)
    bundle[f"cmin{suffix0}"]          = np.array([float(cmin0)], dtype=np.float64)
    bundle[f"cmax{suffix0}"]          = np.array([float(cmax0)], dtype=np.float64)
    bundle[f"abmn{suffix0}"]          = np.stack([scheme0['a'], scheme0['b'], scheme0['n'], scheme0['m']], axis=1).astype(np.int32)[:, [0,1,3,2]]
    bundle[f"rhoa{suffix0}"]          = np.asarray(rhoa0, dtype=np.float64)

    # 残りのフィールドを並列（workers==1 なら逐次）
    rest = targets[1:]
    results = []

    n_total = len(targets) 
    n_tasks = len(rest)
    if n_tasks > 0:
        # ★ workers 自動決定（未指定/0/-1/負なら自動）
        if workers in (None, 0, -1) or (isinstance(workers, int) and workers < 1):
            cpu_env = (
                os.environ.get("SLURM_CPUS_PER_TASK")
                or os.environ.get("OMP_NUM_THREADS")
                or os.cpu_count()
                or 1
            )
            try:
                n_cpu = int(cpu_env)
            except Exception:
                n_cpu = 1
            workers = max(1, min(n_cpu, n_tasks))   # 残りタスクとCPU数の小さい方

            # 任意で上限キャップしたい場合は有効化
            # workers = min(workers, 8)
        print(f"[info] Total fields: {n_total} (first handled in main)")
        print(f"[info] Remaining fields: {n_tasks}")
        print(f"[info] Using {workers} parallel worker(s) for inversion")

        if workers == 1:
            # 逐次（＝単スレ完全一致）
            for (label, designs, y, _) in rest:
                payload = dict(
                    label=label, designs=designs, y=y,
                    n_elec=n_elec, L_world=L_world, margin=margin,
                    mesh_area=mesh_area, Lz=Lz,
                )
                results.append(_invert_job(payload))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = []
                for (label, designs, y, _) in rest:
                    payload = dict(
                        label=label, designs=designs, y=y,
                        n_elec=n_elec, L_world=L_world, margin=margin,
                        mesh_area=mesh_area, Lz=Lz,
                    )
                    futs.append(ex.submit(_invert_job, payload))
                for f in as_completed(futs):
                    results.append(f.result())


        # フィールドラベル順に整列してから bundle へ（＝単スレと同順）
        results_sorted = sorted(results, key=lambda d: d["label"])
        # 注意：targets も label 昇順なので、挿入順は単スレと一致
        for d in results_sorted:
            suffix = f"__{d['label']}"
            bundle[f"inv_rho_cells{suffix}"] = d["inv_arr"]
            bundle[f"cmin{suffix}"]          = np.array([d["cmin"]], dtype=np.float64)
            bundle[f"cmax{suffix}"]          = np.array([d["cmax"]], dtype=np.float64)
            bundle[f"abmn{suffix}"]          = d["abmn"]
            bundle[f"rhoa{suffix}"]          = d["rhoa"]

    bundle_path = Path(bundle_out) if bundle_out else (Path(out_dir) / "inversion_bundle.npz")
    # 既存どおりセーブ
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
