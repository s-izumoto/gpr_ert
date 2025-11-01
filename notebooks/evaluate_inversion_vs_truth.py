# -*- coding: utf-8 -*-
"""
Evaluate ERT inversion vs. "true" field reconstructed from PCA.
- Loads inversion NPZ saved by patched invert scripts.
- Reconstructs the true log10(ρ) field via PCA (from gpr_sequential_design utilities).
- Samples the true field onto the SAME mesh cell centers used in inversion.
- Computes metrics in log and linear domains; optionally depth-weighted.
- Optionally: saves diagnostic plots (parity, residual histogram, residual vs true).

Usage:
  python evaluate_inversion_vs_truth.py \
      --inv-npz path/to/inversion_output.npz \
      --pca path/to/pca_joblib.joblib \
      --Z path/to/Z.npz \
      --field-idx 0 \
      --out-dir ./eval_out \
      [--lambda-depth 0.0]   # meters; 0 disables weighting
      [--no-plots]           # disable saving plots
      [--scatter-max 80000]  # downsample points for scatter plots
"""
# evaluate_inversion_vs_truth.py  (multi-NPZ対応 / SSIMなし版 + plots)
# -*- coding: utf-8 -*-
import argparse
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
from pathlib import Path
import numpy as np
import csv, json

try:
    from gpr_sequential_design import _reconstruct_field_2d_from_pca__pcaops
except Exception:
    _reconstruct_field_2d_from_pca__pcaops = None

def _reconstruct_field_2d_from_pca_local(field_idx: int, pca_joblib_path: Path, z_path: Path) -> np.ndarray:
    from joblib import load as joblib_load
    meta = joblib_load(pca_joblib_path)
    mean = np.asarray(meta["mean"], dtype=np.float32)
    comps = np.asarray(meta["components"], dtype=np.float32)
    nz, nx = int(meta["nz"]), int(meta["nx"])
    Z = np.load(z_path, allow_pickle=True)["Z"]
    z = Z[field_idx, :comps.shape[0]].astype(np.float32)
    x_flat = (z @ comps + mean).astype(np.float32)
    return x_flat.reshape(nz, nx)

def reconstruct_log10_field(field_idx: int, pca_joblib_path: Path, z_path: Path) -> np.ndarray:
    if _reconstruct_field_2d_from_pca__pcaops is not None:
        return _reconstruct_field_2d_from_pca__pcaops(field_idx, pca_joblib_path, z_path)
    return _reconstruct_field_2d_from_pca_local(field_idx, pca_joblib_path, z_path)

def sample_logR_to_cells(logR: np.ndarray, cell_centers: np.ndarray,
                         L_world: float, Lz: float, world_xmin: float, world_zmax: float) -> np.ndarray:
    NZ, NX = logR.shape
    xs = np.linspace(world_xmin, world_xmin + L_world, NX, dtype=np.float32)
    zs = np.linspace(world_zmax, world_zmax - Lz, NZ, dtype=np.float32)  # 上→下
    cx = cell_centers[:, 0].astype(np.float32)
    cz = cell_centers[:, 1].astype(np.float32)
    ix = np.clip(np.round((cx - xs[0]) / (xs[-1] - xs[0]) * (NX - 1)).astype(int), 0, NX - 1)
    iz = np.clip(np.round((cz - zs[-1]) / (zs[0] - zs[-1]) * (NZ - 1)).astype(int), 0, NZ - 1)
    # 画像行方向の上下反転（z上→下）を補正
    iz = (NZ - 1) - iz
    return logR[iz, ix]

def depth_weights(cz: np.ndarray, lambda_depth: float) -> np.ndarray:
    if lambda_depth <= 0:
        return np.ones_like(cz, dtype=float)
    z = -cz  # 正の深さ[m]
    w = np.exp(-z / float(lambda_depth))
    w /= w.mean()
    return w

def safe_log10(x, eps=1e-12):
    return np.log10(np.clip(np.asarray(x, dtype=float), eps, None))

def _safe_corr(a,b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a@b)/denom) if denom>0 else 0.0

# ---- Plot helpers (matplotlib only; one chart per figure; no color specification) ----
def _tripcolor_mesh(save_path, cell_centers, triangles, values, title,
                    xlabel="x (m)", ylabel="z (m)", xlim=None, z_top=None, Lz=None, cb_label=None):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np
    cx = cell_centers[:,0]; cz = cell_centers[:,1]
    # triangles expected as (ntri, 3) indices into cell_centers; if None, fallback to scatter
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    if triangles is not None and len(triangles) > 0:
        tri = mtri.Triangulation(cx, cz, triangles)
        tpc = ax.tripcolor(tri, values, shading='flat')
        cb = fig.colorbar(tpc, ax=ax)
    else:
        sc = ax.scatter(cx, cz, c=values, s=8)
        cb = fig.colorbar(sc, ax=ax)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if (z_top is not None) and (Lz is not None):
        ax.set_ylim(z_top - Lz, z_top)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if cb_label: cb.set_label(cb_label)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)

def _imshow_pca_grid(save_path, logR_2d, xmin, L_world, zmax, Lz, title,
                     xlabel="x (m)", ylabel="z (m)", cb_label="log10(ρ)"):
    import matplotlib.pyplot as plt
    import numpy as np
    NZ, NX = logR_2d.shape
    xs = np.linspace(xmin, xmin + L_world, NX)
    zs = np.linspace(zmax, zmax - Lz, NZ)
    # imshow expects first dim as y (top→bottom). Our zs goes top→bottom already.
    extent = [xs[0], xs[-1], zs[-1], zs[0]]  # [xmin, xmax, zmin, zmax]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    im = ax.imshow(logR_2d, extent=extent, aspect='auto', origin='upper')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cb_label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)

def _downsample_idx(n, k):
    if k is None or k <= 0 or k >= n:
        return np.arange(n, dtype=int)
    # uniform stride downsample
    stride = max(1, int(np.floor(n / k)))
    return np.arange(0, n, stride, dtype=int)[:k]

def _parity_plot_log10(save_path, *, true_log, pred_log, label, metrics, scatter_max=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # デバッグ: 値域チェック（ここで変なら呼び出し側が間違い）
    tmin, tmax = float(np.nanmin(true_log)), float(np.nanmax(true_log))
    pmin, pmax = float(np.nanmin(pred_log)), float(np.nanmax(pred_log))
    print("[parity] true_log10 min/max:", tmin, tmax)
    print("[parity] pred_log10 min/max:", pmin, pmax)
    if (pmin < -0.2 and pmax < 1.0):
        print("[WARN] pred_log のレンジが残差っぽいです（diff を渡している可能性）。")

    # ダウンサンプル
    idx = _downsample_idx(len(true_log), scatter_max)
    lt_s, lp_s = true_log[idx], pred_log[idx]

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(lt_s, lp_s, s=6, alpha=0.4)

    lim_min = min(np.min(true_log), np.min(pred_log))
    lim_max = max(np.max(true_log), np.max(pred_log))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], lw=1)
    ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("True log10(ρ)")
    ax.set_ylabel("Pred log10(ρ)")
    ax.set_title(f"Parity (log10) – {label}")

    txt = (f"n={metrics['Nc']}\n"
           f"MAE={metrics['mae_log10']:.3f}  RMSE={metrics['rmse_log10']:.3f}\n"
           f"bias={metrics['bias_log10']:.3f}  r={metrics['pearson_log10']:.3f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top")

    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


def _residual_hist_log10(save_path, diff, label):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(diff, bins=50)
    ax.axvline(0, lw=1)
    ax.set_xlabel("Residual (pred - true) in log10(ρ)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual histogram – {label}")
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)

def _residual_vs_true(save_path, lt, diff, label, scatter_max=None):
    import matplotlib.pyplot as plt
    idx = _downsample_idx(len(lt), scatter_max)
    lt_s, diff_s = lt[idx], diff[idx]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(lt_s, diff_s, s=6, alpha=0.4)
    ax.axhline(0, lw=1)
    ax.set_xlabel("True log10(ρ)")
    ax.set_ylabel("Residual (pred - true) in log10(ρ)")
    ax.set_title(f"Residual vs True – {label}")
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)

def eval_one(inv_npz: Path, true_log2d: np.ndarray, out_dir: Path, lambda_depth: float, label: str | None,
             make_plots: bool = True, scatter_max: int | None = 80000):
    npz = np.load(inv_npz, allow_pickle=True)
    inv_rho_cells = npz["inv_rho_cells"].astype(float)
    cell_centers  = npz["cell_centers"].astype(np.float32)
    L_world = float(npz["L_world"]); Lz = float(npz["Lz"])
    xmin = float(npz["world_xmin"]); zmax = float(npz["world_zmax"])

    true_log_cells = sample_logR_to_cells(true_log2d, cell_centers, L_world, Lz, xmin, zmax)

    # ★ inv_rho は線形(Ωm)を想定し、log10 を 1 回だけ取る
    inv_rho = np.asarray(inv_rho_cells, dtype=float)
    inv_log_cells = np.log10(np.clip(inv_rho, 1e-12, None))
    pred_log_cells = inv_log_cells.copy() 

    # デバッグ: レンジ確認（期待: inv_rho ≈ 12〜500, inv_log ≈ 1.08〜2.70）
    print("[debug] inv_rho  min/max:", float(inv_rho.min()),  float(inv_rho.max()))
    print("[debug] inv_log10 min/max:", float(inv_log_cells.min()), float(inv_log_cells.max()))

    diff_log = pred_log_cells - true_log_cells

    w = depth_weights(cell_centers[:,1], lambda_depth)
    def wmean(x): return float((w * x).sum() / w.sum())
    def wrmse(e): return float(np.sqrt(wmean(e**2)))

    # log領域
    mae_log = wmean(np.abs(diff_log))
    rmse_log = wrmse(diff_log)
    bias_log = wmean(diff_log)

    # 線形領域
    true_rho = np.power(10.0, true_log_cells)
    inv_rho  = np.power(10.0, pred_log_cells)
    abs_err_lin = np.abs(inv_rho - true_rho)
    rel_err_lin = abs_err_lin / np.clip(true_rho, 1e-12, None)
    mae_rel  = 100.0 * wmean(rel_err_lin)
    rmse_rel = 100.0 * wrmse(rel_err_lin)
    mae_lin  = wmean(abs_err_lin)
    rmse_lin = wrmse(abs_err_lin)


    # For tripcolor: build Delaunay triangles over cell centers (viz only)
    try:
        import matplotlib.tri as mtri
        tri_obj = mtri.Triangulation(cell_centers[:,0], cell_centers[:,1])
        triangles = tri_obj.triangles
    except Exception:
        triangles = None

    # 相関
    pearson_log = _safe_corr(inv_log_cells, true_log_cells)
    try:
        from scipy.stats import spearmanr
        spearman_log = float(spearmanr(inv_log_cells, true_log_cells, nan_policy="omit").correlation)
    except Exception:
        rk1 = np.argsort(np.argsort(inv_log_cells))
        rk2 = np.argsort(np.argsort(true_log_cells))
        spearman_log = _safe_corr(rk1, rk2)

    safe_label = (label or inv_npz.stem).replace(" ", "_")
    metrics = {
        "label": (label or inv_npz.stem),
        "inv_npz": str(inv_npz),
        "Nc": int(inv_rho_cells.size),
        "lambda_depth": float(lambda_depth),
        "mae_log10": mae_log,
        "rmse_log10": rmse_log,
        "bias_log10": bias_log,
        "pearson_log10": pearson_log,
        "spearman_log10": spearman_log,
        "mae_linear": mae_lin,
        "rmse_linear": rmse_lin,
        "mae_relative_percent": mae_rel,
        "rmse_relative_percent": rmse_rel,
    }

    # 保存（個別CSV・JSON）
    (out_dir / f"metrics_{safe_label}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / f"per_cell_log10_{safe_label}.csv").open("w", newline="", encoding="utf-8") as fp:
        wtr = csv.writer(fp)
        wtr.writerow(["cx","cz","inv_log10","true_log10","err_log10","w"])
        for (cx,cz), il, tl, wl in zip(cell_centers, pred_log_cells, true_log_cells, w):
            wtr.writerow([f"{cx:.6g}", f"{cz:.6g}", f"{il:.6g}", f"{tl:.6g}", f"{(il-tl):.6g}", f"{wl:.6g}"])


    # プロット保存
    if make_plots:
        print("[call parity] pred uses pred_log_cells: min/max =",
            float(np.nanmin(pred_log_cells)), float(np.nanmax(pred_log_cells)))
        print("[call parity] diff_log min/max =",
            float(np.nanmin(diff_log)), float(np.nanmax(diff_log)))

        _parity_plot_log10(
            save_path=out_dir / f"parity_{safe_label}.png",
            true_log=true_log_cells,
            pred_log=pred_log_cells,
            label=metrics["label"],
            metrics=metrics, 
            scatter_max=scatter_max
        )
        _residual_hist_log10(out_dir / f"residual_hist_{safe_label}.png", diff_log, metrics["label"])
        _residual_vs_true(out_dir / f"residual_vs_true_{safe_label}.png", true_log_cells, diff_log, metrics["label"], scatter_max)

    # 返却
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inv-npz",
        type=Path,
        nargs="+",
        default=[
            Path("./inversion_MI_field000_mod.npz"),
            Path("../outputs_wenner/inversion.npz"),
        ],
        help="1つ以上の inversion .npz を指定（デフォルト: Wenner と Dipole-Dipole）"
    )

    ap.add_argument("--labels", nargs="*", default=None,
                    help="各inv-npzに対応するラベル（省略可）。個数が合わなければ stem を自動使用")
    ap.add_argument("--pca", type=Path, default=Path("../data/interim/pca/pca_joint.joblib"))
    ap.add_argument("--Z",   type=Path, default=Path("../data/interim/pca/Z.npz"))
    ap.add_argument("--field-idx", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("./eval_out"))
    ap.add_argument("--lambda-depth", type=float, default=0.0)
    ap.add_argument("--no-plots", action="store_true", help="プロット保存を無効化")
    ap.add_argument("--scatter-max", type=int, default=80000, help="散布図の最大点数（ダウンサンプル）")
    args = ap.parse_args()

    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)

    # 真値（log10ρ）を1回だけ再構成
    true_log2d = reconstruct_log10_field(args.field_idx, args.pca, args.Z)

    # ラベル整形
    labels = args.labels or []
    if len(labels) != len(args.inv_npz):
        labels = (labels + [p.stem for p in args.inv_npz])[0:len(args.inv_npz)]

    all_metrics = []
    print("[metrics]")
    for inv_path, lab in zip(args.inv_npz, labels):
        m = eval_one(
            inv_path, true_log2d, out_dir, args.lambda_depth, lab,
            make_plots=(not args.no_plots), scatter_max=args.scatter_max
        )
        all_metrics.append(m)
        # コンソールにも短く出す
        print(f"  [{m['label']}] mae_log10={m['mae_log10']:.4f} rmse_log10={m['rmse_log10']:.4f} "
              f"bias_log10={m['bias_log10']:.4f} pearson={m['pearson_log10']:.4f} "
              f"mae%={m['mae_relative_percent']:.1f} rmse%={m['rmse_relative_percent']:.1f}")

    # 総合サマリー保存
    (out_dir / "summary_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    with (out_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as fp:
        wtr = csv.writer(fp)
        header = ["label","inv_npz","Nc","lambda_depth",
                  "mae_log10","rmse_log10","bias_log10","pearson_log10","spearman_log10",
                  "mae_linear","rmse_linear","mae_relative_percent","rmse_relative_percent"]
        wtr.writerow(header)
        for m in all_metrics:
            wtr.writerow([m[k] for k in header])

if __name__ == "__main__":
    main()
