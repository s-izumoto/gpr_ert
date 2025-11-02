
# -*- coding: utf-8 -*-
"""
Evaluate ERT inversion vs. 'true' field reconstructed from PCA.
Now supports two modes:

(Mode A) Standalone NPZs (old behavior):
  python evaluate_inversion_vs_truth_multi.py \
      --inv-npz path/to/inversion1.npz path/to/inversion2.npz \
      --labels 'Model-1' 'Model-2' \
      --pca ../data/interim/pca/pca_joint.joblib \
      --Z   ../data/interim/pca/Z.npz \
      --field-idx 0 \
      --out-dir ./eval_out

(Mode B) Bundled NPZs for multiple fields (NEW):
  python evaluate_inversion_vs_truth_multi.py \
      --wenner-bundle ../outputs_wenner/inversion.npz \
      --other-bundle  ./inversion_bundle.npz \
      --pca ../data/interim/pca/pca_joint.joblib \
      --Z   ../data/interim/pca/Z.npz \
      --out-dir ./eval_out_multi

In Mode B, the script:
  * Reads available field labels (e.g., field000, field001, ...) from the WENNER bundle.
  * For each label, reconstructs the PCA 'true' field using the corresponding field index.
  * Evaluates BOTH NPZ bundles on that field (if present) using shared mesh metadata stored in each bundle.
  * Saves per-field plots/metrics and a single summary CSV/JSON across all fields and both sources.
"""
import argparse
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

from pathlib import Path
import re
import csv, json
import numpy as np

# ---------------- PCA reconstruction ----------------
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

# ---------------- geometry / sampling ----------------
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

def _safe_corr(a,b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a@b)/denom) if denom>0 else 0.0

# ---------------- plotting helpers (1 fig per chart) ----------------
def _downsample_idx(n, k):
    if k is None or k <= 0 or k >= n:
        return np.arange(n, dtype=int)
    stride = max(1, int(np.floor(n / k)))
    return np.arange(0, n, stride, dtype=int)[:k]

def _parity_plot_log10(save_path, *, true_log, pred_log, label, metrics, scatter_max=None):
    import matplotlib.pyplot as plt
    tmin, tmax = float(np.nanmin(true_log)), float(np.nanmax(true_log))
    pmin, pmax = float(np.nanmin(pred_log)), float(np.nanmax(pred_log))
    if (pmin < -0.2 and pmax < 1.0):
        print("[WARN] pred_log のレンジが残差っぽいです（diff を渡している可能性）。")

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

# ---------------- metrics core ----------------
def _compute_metrics(inv_rho_cells, cell_centers, L_world, Lz, xmin, zmax, true_log2d, lambda_depth, label, make_plots, scatter_max, out_dir):
    # Project truth onto mesh
    true_log_cells = sample_logR_to_cells(true_log2d, cell_centers, L_world, Lz, xmin, zmax)

    # inv_rho(Ωm) → log10(ρ)  (logを 1 回だけ取る)
    inv_log_cells = np.log10(np.clip(np.asarray(inv_rho_cells, dtype=float), 1e-12, None))
    pred_log_cells = inv_log_cells.copy()
    diff_log = pred_log_cells - true_log_cells

    print("[debug] inv_rho  min/max:", float(inv_rho_cells.min()),  float(inv_rho_cells.max()))
    print("[debug] inv_log10 min/max:", float(pred_log_cells.min()), float(pred_log_cells.max()))

    w = depth_weights(cell_centers[:,1], lambda_depth)
    def wmean(x): return float((w * x).sum() / w.sum())
    def wrmse(e): return float(np.sqrt(wmean(e**2)))

    # log domain
    mae_log = wmean(np.abs(diff_log))
    rmse_log = wrmse(diff_log)
    bias_log = wmean(diff_log)

    # linear domain
    true_rho = np.power(10.0, true_log_cells)
    inv_rho  = np.power(10.0, pred_log_cells)
    abs_err_lin = np.abs(inv_rho - true_rho)
    rel_err_lin = abs_err_lin / np.clip(true_rho, 1e-12, None)
    mae_rel  = 100.0 * wmean(rel_err_lin)
    rmse_rel = 100.0 * wrmse(rel_err_lin)
    mae_lin  = wmean(abs_err_lin)
    rmse_lin = wrmse(abs_err_lin)

    pearson_log = _safe_corr(inv_log_cells, true_log_cells)
    try:
        from scipy.stats import spearmanr
        spearman_log = float(spearmanr(inv_log_cells, true_log_cells, nan_policy="omit").correlation)
    except Exception:
        rk1 = np.argsort(np.argsort(inv_log_cells))
        rk2 = np.argsort(np.argsort(true_log_cells))
        spearman_log = _safe_corr(rk1, rk2)

    metrics = {
        "label": label,
        "Nc": int(len(inv_log_cells)),
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

    if make_plots:
        _parity_plot_log10(out_dir / f"parity_{label}.png",
                           true_log=true_log_cells, pred_log=pred_log_cells,
                           label=label, metrics=metrics, scatter_max=scatter_max)
        _residual_hist_log10(out_dir / f"residual_hist_{label}.png", diff_log, label)
        _residual_vs_true(out_dir / f"residual_vs_true_{label}.png", true_log_cells, diff_log, label, scatter_max)

    # per-cell csv
    with (out_dir / f"per_cell_log10_{label}.csv").open("w", newline="", encoding="utf-8") as fp:
        wtr = csv.writer(fp)
        wtr.writerow(["cx","cz","inv_log10","true_log10","err_log10","w"])
        for (cx,cz), il, tl, wl in zip(cell_centers, pred_log_cells, true_log_cells, w):
            wtr.writerow([f"{cx:.6g}", f"{cz:.6g}", f"{il:.6g}", f"{tl:.6g}", f"{(il-tl):.6g}", f"{wl:.6g}"])

    return metrics

# ---------------- bundled helpers ----------------
FIELD_PAT = re.compile(r"^inv_rho_cells__(field\d{3})$")

def list_field_labels_from_bundle(npz) -> list[str]:
    labels = []
    for k in npz.files:
        m = FIELD_PAT.match(k)
        if m:
            labels.append(m.group(1))
    return sorted(labels)

def field_label_to_index(label: str) -> int:
    m = re.match(r"^field(\d{3})$", label)
    if not m:
        raise ValueError(f"Invalid field label: {label}")
    return int(m.group(1))

def load_bundle_meta(npz):
    cell_centers = np.asarray(npz["cell_centers"], dtype=np.float32)
    L_world = float(np.asarray(npz["L_world"]).reshape(-1)[0])
    Lz      = float(np.asarray(npz["Lz"]).reshape(-1)[0])
    xmin    = float(np.asarray(npz["world_xmin"]).reshape(-1)[0])
    zmax    = float(np.asarray(npz["world_zmax"]).reshape(-1)[0])
    return cell_centers, L_world, Lz, xmin, zmax

def try_load_inv_for_label(npz, label: str):
    k = f"inv_rho_cells__{label}"
    if k not in npz.files:
        return None
    return np.asarray(npz[k], dtype=float)

# ---------------- CLI / main ----------------
def main():
    ap = argparse.ArgumentParser()
    # Mode A (old): list of standalone inv NPZs
    ap.add_argument("--inv-npz", type=Path, nargs="*", default=None,
                    help="One or more standalone inversion NPZs (old behavior).")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Labels for --inv-npz (optional).")
    ap.add_argument("--field-idx", type=int, default=None,
                    help="[Mode A] PCA field index. Required in Mode A. Ignored in Mode B.")
    # Mode B (new): two bundles, matched per-field using WENNER bundle's available labels
    ap.add_argument("--wenner-bundle", type=Path, default=Path("../data/processed/outputs_wenner/inversions_bundle.npz"),
                    help="[Mode B] Path to WENNER bundle NPZ (e.g., ../outputs_wenner/inversion.npz).")
    ap.add_argument("--other-bundle", type=Path, default=Path("../data/processed/inversion_out/inversion_bundle.npz"),
                    help="[Mode B] Path to OTHER bundle NPZ (e.g., output of inversion.py).")
    # Alias: allow using --inv-npz (single) as the OTHER bundle when paired with --wenner-bundle
    ap.add_argument("--inv-npz-bundle", type=Path, default=None,
                    help="[Mode B] Path to OTHER bundle NPZ (e.g., output of inversion.py).")
    ap.add_argument("--fields", type=int, nargs="*", default=None,
                    help="[Mode B] Optional explicit list of field indices to evaluate (e.g., 0 1 2).")
    # Common
    ap.add_argument("--pca", type=Path, default=Path("../data/interim/pca/pca_joint.joblib"))
    ap.add_argument("--Z",   type=Path, default=Path("../data/interim/pca/Z.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path("./eval_out"))
    ap.add_argument("--lambda-depth", type=float, default=0.0)
    ap.add_argument("--no-plots", action="store_true", help="Disable saving plots")
    ap.add_argument("--scatter-max", type=int, default=80000, help="Max scatter points for plots")
    args = ap.parse_args()

    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)

    # Decide mode
    mode_b = (args.wenner_bundle is not None) or (args.other_bundle is not None) or (args.inv_npz_bundle is not None)
    # Mode A uses --inv-npz (list). However, if --wenner-bundle is provided and --inv-npz has exactly 1 path,
    # we reinterpret it as a bundle for Mode B.
    mode_a = (args.inv_npz is not None and len(args.inv_npz) > 0 and not (args.wenner_bundle and len(args.inv_npz) == 1))

    if mode_b and mode_a:
        raise SystemExit("Specify either Mode A (--inv-npz ...) OR Mode B (--wenner-bundle ... --other-bundle ...), not both.")

    all_metrics = []
    header = ["label","source","field_idx","Nc","lambda_depth",
              "mae_log10","rmse_log10","bias_log10","pearson_log10","spearman_log10",
              "mae_linear","rmse_linear","mae_relative_percent","rmse_relative_percent"]

    if mode_b:
        # -------- Mode B: two bundles; evaluate multiple fields ----------
        if args.wenner_bundle is None:
            raise SystemExit("Mode B requires --wenner-bundle and an OTHER bundle (via --other-bundle or --inv-npz-bundle, or a single --inv-npz).")
        npz_w = np.load(args.wenner_bundle, allow_pickle=False)
        # Resolve OTHER bundle path
        other_path = None
        if args.other_bundle is not None:
            other_path = args.other_bundle
        elif args.inv_npz_bundle is not None:
            other_path = args.inv_npz_bundle
        elif (args.inv_npz is not None and len(args.inv_npz) == 1):
            other_path = args.inv_npz[0]
        else:
            raise SystemExit("Provide OTHER bundle via --other-bundle or --inv-npz-bundle, or pass a single --inv-npz together with --wenner-bundle.")
        npz_o = np.load(other_path, allow_pickle=False)

        # Available fields from WENNER bundle
        labels = list_field_labels_from_bundle(npz_w)
        if not labels:
            raise SystemExit(f"No per-field keys found in WENNER bundle: {args.wenner_bundle}")

        # Optional user subset
        if args.fields is not None and len(args.fields) > 0:
            allowed = {f"field{f:03d}" for f in args.fields}
            labels = [lb for lb in labels if lb in allowed]
            if not labels:
                raise SystemExit("No matching fields after filtering by --fields.")

        # Evaluate per field (each bundle uses its own mesh/meta)
        print(f"[Mode B] Fields to evaluate: {labels}  (from {args.wenner_bundle})")
        for lab in labels:
            fidx = field_label_to_index(lab)
            true_log2d = reconstruct_log10_field(fidx, args.pca, args.Z)

            # WENNER
            inv_w = try_load_inv_for_label(npz_w, lab)
            if inv_w is not None:
                cc_w, Lx_w, Lz_w, xmin_w, zmax_w = load_bundle_meta(npz_w)
                m_w = _compute_metrics(inv_w, cc_w, Lx_w, Lz_w, xmin_w, zmax_w, true_log2d,
                                       args.lambda_depth, f"WENNER_{lab}",
                                       make_plots=(not args.no_plots), scatter_max=args.scatter_max, out_dir=out_dir)
                m_w["label"] = lab; m_w["source"] = "WENNER"; m_w["field_idx"] = fidx
                all_metrics.append(m_w)
                print(f"[{lab}] WENNER  mae_log10={m_w['mae_log10']:.4f} rmse_log10={m_w['rmse_log10']:.4f} "
                      f"bias={m_w['bias_log10']:.4f} r={m_w['pearson_log10']:.4f} mae%={m_w['mae_relative_percent']:.1f}")
            else:
                print(f"[WARN] {lab} not found in WENNER bundle.")

            # OTHER
            inv_o = try_load_inv_for_label(npz_o, lab)
            if inv_o is not None:
                cc_o, Lx_o, Lz_o, xmin_o, zmax_o = load_bundle_meta(npz_o)
                m_o = _compute_metrics(inv_o, cc_o, Lx_o, Lz_o, xmin_o, zmax_o, true_log2d,
                                       args.lambda_depth, f"OTHER_{lab}",
                                       make_plots=(not args.no_plots), scatter_max=args.scatter_max, out_dir=out_dir)
                m_o["label"] = lab; m_o["source"] = "OTHER"; m_o["field_idx"] = fidx
                all_metrics.append(m_o)
                print(f"[{lab}] OTHER   mae_log10={m_o['mae_log10']:.4f} rmse_log10={m_o['rmse_log10']:.4f} "
                      f"bias={m_o['bias_log10']:.4f} r={m_o['pearson_log10']:.4f} mae%={m_o['mae_relative_percent']:.1f}")
            else:
                print(f"[WARN] {lab} not found in OTHER bundle.")

        # Save summary
        (out_dir / "summary_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
        with (out_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as fp:
            wtr = csv.writer(fp); wtr.writerow(header)
            for m in all_metrics:
                row = [m.get(k, "") for k in header]
                wtr.writerow(row)

    elif mode_a:
        # -------- Mode A: list of standalone NPZs (old behavior) ----------
        if args.field_idx is None:
            raise SystemExit("Mode A requires --field-idx.")
        true_log2d = reconstruct_log10_field(args.field_idx, args.pca, args.Z)
        labels = args.labels or []
        if len(labels) != len(args.inv_npz):
            labels = (labels + [p.stem for p in args.inv_npz])[0:len(args.inv_npz)]

        for inv_path, lab in zip(args.inv_npz, labels):
            npz = np.load(inv_path, allow_pickle=False)

            # Standalone keys expected
            inv_rho_cells = np.asarray(npz["inv_rho_cells"], dtype=float)
            cell_centers  = np.asarray(npz["cell_centers"], dtype=np.float32)
            L_world = float(np.asarray(npz["L_world"]).reshape(-1)[0])
            Lz      = float(np.asarray(npz["Lz"]).reshape(-1)[0])
            xmin    = float(np.asarray(npz["world_xmin"]).reshape(-1)[0])
            zmax    = float(np.asarray(npz["world_zmax"]).reshape(-1)[0])

            m = _compute_metrics(inv_rho_cells, cell_centers, L_world, Lz, xmin, zmax,
                                 true_log2d, args.lambda_depth, lab,
                                 make_plots=(not args.no_plots),
                                 scatter_max=args.scatter_max, out_dir=out_dir)
            m["label"] = lab; m["source"] = "standalone"; m["field_idx"] = int(args.field_idx)
            all_metrics.append(m)
            print(f"[{lab}] mae_log10={m['mae_log10']:.4f} rmse_log10={m['rmse_log10']:.4f} "
                  f"bias={m['bias_log10']:.4f} r={m['pearson_log10']:.4f} mae%={m['mae_relative_percent']:.1f}")

        (out_dir / "summary_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
        with (out_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as fp:
            wtr = csv.writer(fp); wtr.writerow(header)
            for m in all_metrics:
                row = [m.get(k, "") for k in header]
                wtr.writerow(row)

    else:
        raise SystemExit("Specify Mode A (--inv-npz ...) or Mode B (--wenner-bundle ... --other-bundle ...).")

if __name__ == "__main__":
    main()
