# -*- coding: utf-8 -*-
"""
build.evaluate_inversion_vs_truth_multi
A library-style module that evaluates ERT inversion vs. PCA 'true' fields.

Usage from code:
    from build.evaluate_inversion_vs_truth_multi import run_from_cfg
    run_from_cfg(cfg_dict)

Config schema (minimal):
  # Common
  pca: ../data/interim/pca/pca_joint.joblib
  Z:   ../data/interim/pca/Z.npz
  out_dir: ./eval_out
  lambda_depth: 0.0
  plots: true
  scatter_max: 80000

  # Choose ONE mode:

  # (A) Standalone (old behavior)
  mode: standalone
  inv_npz:
    - path/to/inversion1.npz
    - path/to/inversion2.npz
  labels: ["Model-1", "Model-2"]
  field_idx: 0

  # (B) Bundles per-field (new)
  mode: bundle
  wenner_bundle: ../outputs_wenner/inversion.npz
  other_bundle:  ./inversion_bundle.npz        # OR inv_npz_bundle
  fields: [0, 1, 2]                            # optional filter
"""
from __future__ import annotations
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

from pathlib import Path
import csv, json, re
import numpy as np

# ================= PCA reconstruction =================
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

def _reconstruct_field_2d_from_pca__pcaops(field_idx: int, pca_joblib_path: "Path", z_path: "Path") -> np.ndarray:
    from joblib import load as joblib_load
    meta = joblib_load(pca_joblib_path)  # keys: mean, components(k,D), nz, nx, crop_frac, ...
    mean = np.asarray(meta["mean"], dtype=np.float32)            # (D,)
    comps = np.asarray(meta["components"], dtype=np.float32)     # (k, D)
    nz, nx = int(meta["nz"]), int(meta["nx"])
    k = comps.shape[0]

    Z = np.load(z_path, allow_pickle=True)["Z"]                  # (N, k) を想定（kはk_star）
    if not (0 <= field_idx < Z.shape[0]):
        raise IndexError(f"field_index {field_idx} is out of range [0, {Z.shape[0]-1}]")
    z = Z[field_idx, :k].astype(np.float32)                      # (k,)

    x_flat = (z @ comps + mean).astype(np.float32)               # (D,)
    if x_flat.size != nz * nx:
        raise RuntimeError(f"サイズ不一致: flat={x_flat.size} vs nz*nx={nz*nx} (PCA作成時の形状を確認)")
    arr2d = x_flat.reshape(nz, nx)
    return arr2d


def reconstruct_log10_field(field_idx: int, pca_joblib_path: Path, z_path: Path) -> np.ndarray:
    return _reconstruct_field_2d_from_pca__pcaops(field_idx, pca_joblib_path, z_path)

# ================= geometry / sampling =================
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

# ================= plotting helpers =================
def _downsample_idx(n, k):
    if k is None or k <= 0 or k >= n:
        return np.arange(n, dtype=int)
    stride = max(1, int(np.floor(n / k)))
    return np.arange(0, n, stride, dtype=int)[:k]

def _parity_plot_log10(save_path, *, true_log, pred_log, label, metrics, scatter_max=None):
    import matplotlib.pyplot as plt
    idx = _downsample_idx(len(true_log), scatter_max)
    lt_s, lp_s = true_log[idx], pred_log[idx]
    lim_min = min(np.min(true_log), np.min(pred_log))
    lim_max = max(np.max(true_log), np.max(pred_log))
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(lt_s, lp_s, s=6, alpha=0.4)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], lw=1)
    ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("True log10(ρ)"); ax.set_ylabel("Pred log10(ρ)")
    ax.set_title(f"Parity (log10) – {label}")
    txt = (f"n={metrics['Nc']}\\n"
           f"MAE={metrics['mae_log10']:.3f}  RMSE={metrics['rmse_log10']:.3f}\\n"
           f"bias={metrics['bias_log10']:.3f}  r={metrics['pearson_log10']:.3f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top")
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)

def _residual_hist_log10(save_path, diff, label):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(diff, bins=50); ax.axvline(0, lw=1)
    ax.set_xlabel("Residual (pred - true) in log10(ρ)")
    ax.set_ylabel("Count"); ax.set_title(f"Residual histogram – {label}")
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)

def _residual_vs_true(save_path, lt, diff, label, scatter_max=None):
    import matplotlib.pyplot as plt
    idx = _downsample_idx(len(lt), scatter_max)
    lt_s, diff_s = lt[idx], diff[idx]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(lt_s, diff_s, s=6, alpha=0.4); ax.axhline(0, lw=1)
    ax.set_xlabel("True log10(ρ)"); ax.set_ylabel("Residual (pred - true) in log10(ρ)")
    ax.set_title(f"Residual vs True – {label}")
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)

# ================= metrics core =================
def _compute_metrics(inv_rho_cells, cell_centers, L_world, Lz, xmin, zmax,
                     true_log2d, lambda_depth, label, make_plots, scatter_max, out_dir):
    # Project truth onto mesh
    true_log_cells = sample_logR_to_cells(true_log2d, cell_centers, L_world, Lz, xmin, zmax)

    # inv_rho(Ωm) → log10(ρ)（logを1回だけ）
    inv_log_cells = np.log10(np.clip(np.asarray(inv_rho_cells, dtype=float), 1e-12, None))
    pred_log_cells = inv_log_cells.copy()
    diff_log = pred_log_cells - true_log_cells

    w = depth_weights(cell_centers[:,1], lambda_depth)
    def wmean(x): return float((w * x).sum() / w.sum())
    def wrmse(e): return float(np.sqrt(wmean(e**2)))

    # log domain
    mae_log = wmean(np.abs(diff_log)); rmse_log = wrmse(diff_log); bias_log = wmean(diff_log)

    # linear domain
    true_rho = np.power(10.0, true_log_cells); inv_rho = np.power(10.0, pred_log_cells)
    abs_err_lin = np.abs(inv_rho - true_rho); rel_err_lin = abs_err_lin / np.clip(true_rho, 1e-12, None)
    mae_rel = 100.0 * wmean(rel_err_lin); rmse_rel = 100.0 * wrmse(rel_err_lin)
    mae_lin = wmean(abs_err_lin); rmse_lin = wrmse(abs_err_lin)

    pearson_log = _safe_corr(inv_log_cells, true_log_cells)
    try:
        from scipy.stats import spearmanr
        spearman_log = float(spearmanr(inv_log_cells, true_log_cells, nan_policy="omit").correlation)
    except Exception:
        rk1 = np.argsort(np.argsort(inv_log_cells)); rk2 = np.argsort(np.argsort(true_log_cells))
        spearman_log = _safe_corr(rk1, rk2)

    metrics = {
        "label": label, "Nc": int(len(inv_log_cells)), "lambda_depth": float(lambda_depth),
        "mae_log10": mae_log, "rmse_log10": rmse_log, "bias_log10": bias_log,
        "pearson_log10": pearson_log, "spearman_log10": spearman_log,
        "mae_linear": mae_lin, "rmse_linear": rmse_lin,
        "mae_relative_percent": mae_rel, "rmse_relative_percent": rmse_rel,
    }

    if make_plots:
        _parity_plot_log10(Path(out_dir) / f"parity_{label}.png",
                           true_log=true_log_cells, pred_log=pred_log_cells,
                           label=label, metrics=metrics, scatter_max=scatter_max)
        _residual_hist_log10(Path(out_dir) / f"residual_hist_{label}.png", diff_log, label)
        _residual_vs_true(Path(out_dir) / f"residual_vs_true_{label}.png", true_log_cells, diff_log, label, scatter_max)

    # per-cell csv
    out_dir = Path(out_dir)
    with (out_dir / f"per_cell_log10_{label}.csv").open("w", newline="", encoding="utf-8") as fp:
        wtr = csv.writer(fp)
        wtr.writerow(["cx","cz","inv_log10","true_log10","err_log10","w"])
        for (cx,cz), il, tl, wl in zip(cell_centers, pred_log_cells, true_log_cells, w):
            wtr.writerow([f"{cx:.6g}", f"{cz:.6g}", f"{il:.6g}", f"{tl:.6g}", f"{(il-tl):.6g}", f"{wl:.6g}"])

    return metrics

# ================= bundled helpers =================
FIELD_PAT = re.compile(r"^inv_rho_cells__(field\d{3})$")

def list_field_labels_from_bundle(npz) -> list[str]:
    return sorted(m.group(1) for k in npz.files if (m := FIELD_PAT.match(k)))

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

# ================= public runner =================
HEADER = ["label","source","field_idx","Nc","lambda_depth",
          "mae_log10","rmse_log10","bias_log10","pearson_log10","spearman_log10",
          "mae_linear","rmse_linear","mae_relative_percent","rmse_relative_percent"]

def run_from_cfg(cfg: dict) -> dict:
    """
    Execute evaluation based on a config dict.
    Returns a dict with { 'metrics': list[dict], 'out_dir': str }.
    """
    # --- common settings
    pca = Path(cfg.get("pca", "../data/interim/pca/pca_joint.joblib"))
    Z   = Path(cfg.get("Z",   "../data/interim/pca/Z.npz"))
    out_dir = Path(cfg.get("out_dir", "./eval_out")); out_dir.mkdir(parents=True, exist_ok=True)
    lambda_depth = float(cfg.get("lambda_depth", 0.0))
    make_plots = bool(cfg.get("plots", True))
    scatter_max = int(cfg.get("scatter_max", 80000))

    mode = cfg.get("mode", "bundle").lower()
    all_metrics = []

    if mode == "bundle":
        wenner_bundle = cfg.get("wenner_bundle", None)
        other_bundle  = cfg.get("other_bundle", None) or cfg.get("inv_npz_bundle", None) or cfg.get("inv_npz", None)
        if isinstance(other_bundle, list):
            # if mistakenly given a 1-length list, unwrap
            if len(other_bundle) == 1: other_bundle = other_bundle[0]
            else: raise SystemExit("For 'bundle' mode, 'other_bundle' must be a single NPZ path")

        if not wenner_bundle or not other_bundle:
            raise SystemExit("mode=bundle requires 'wenner_bundle' and 'other_bundle' (or 'inv_npz_bundle').")

        npz_w = np.load(wenner_bundle, allow_pickle=False)
        npz_o = np.load(other_bundle, allow_pickle=False)

        labels = list_field_labels_from_bundle(npz_w)
        if not labels:
            raise SystemExit(f"No per-field keys found in WENNER bundle: {wenner_bundle}")

        # Optional field filter
        fields = cfg.get("fields", None)
        if fields:
            allowed = {f"field{int(f):03d}" for f in fields}
            labels = [lb for lb in labels if lb in allowed]
            if not labels:
                raise SystemExit("No matching fields after filtering by 'fields'.")

        print(f"[bundle] fields: {labels}")
        for lab in labels:
            fidx = field_label_to_index(lab)
            true_log2d = reconstruct_log10_field(fidx, pca, Z)

            inv_w = try_load_inv_for_label(npz_w, lab)
            if inv_w is not None:
                cc_w, Lx_w, Lz_w, xmin_w, zmax_w = load_bundle_meta(npz_w)
                m_w = _compute_metrics(inv_w, cc_w, Lx_w, Lz_w, xmin_w, zmax_w, true_log2d,
                                       lambda_depth, f"WENNER_{lab}", make_plots, scatter_max, out_dir)
                m_w["label"] = lab; m_w["source"] = "WENNER"; m_w["field_idx"] = int(fidx)
                all_metrics.append(m_w)
            else:
                print(f"[WARN] {lab} not found in WENNER bundle.")

            inv_o = try_load_inv_for_label(npz_o, lab)
            if inv_o is not None:
                cc_o, Lx_o, Lz_o, xmin_o, zmax_o = load_bundle_meta(npz_o)
                m_o = _compute_metrics(inv_o, cc_o, Lx_o, Lz_o, xmin_o, zmax_o, true_log2d,
                                       lambda_depth, f"OTHER_{lab}", make_plots, scatter_max, out_dir)
                m_o["label"] = lab; m_o["source"] = "OTHER"; m_o["field_idx"] = int(fidx)
                all_metrics.append(m_o)
            else:
                print(f"[WARN] {lab} not found in OTHER bundle.")

    elif mode == "standalone":
        inv_list = cfg.get("inv_npz", [])
        if not inv_list:
            raise SystemExit("mode=standalone requires 'inv_npz': [paths...]")
        field_idx = cfg.get("field_idx", None)
        if field_idx is None:
            raise SystemExit("mode=standalone requires 'field_idx'.")
        labels = cfg.get("labels", None) or [Path(p).stem for p in inv_list]
        if len(labels) != len(inv_list):
            raise SystemExit("'labels' length must match 'inv_npz' length.")

        true_log2d = reconstruct_log10_field(int(field_idx), pca, Z)

        for inv_path, lab in zip(inv_list, labels):
            npz = np.load(inv_path, allow_pickle=False)
            inv_rho_cells = np.asarray(npz["inv_rho_cells"], dtype=float)
            cell_centers  = np.asarray(npz["cell_centers"], dtype=np.float32)
            L_world = float(np.asarray(npz["L_world"]).reshape(-1)[0])
            Lz      = float(np.asarray(npz["Lz"]).reshape(-1)[0])
            xmin    = float(np.asarray(npz["world_xmin"]).reshape(-1)[0])
            zmax    = float(np.asarray(npz["world_zmax"]).reshape(-1)[0])

            m = _compute_metrics(inv_rho_cells, cell_centers, L_world, Lz, xmin, zmax,
                                 true_log2d, lambda_depth, lab, make_plots, scatter_max, out_dir)
            m["label"] = lab; m["source"] = "standalone"; m["field_idx"] = int(field_idx)
            all_metrics.append(m)
    else:
        raise SystemExit("Config 'mode' must be 'bundle' or 'standalone'.")

    # Save summary
    (Path(out_dir) / "summary_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    with (Path(out_dir) / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as fp:
        wtr = csv.writer(fp); wtr.writerow(HEADER)
        for m in all_metrics:
            wtr.writerow([m.get(k, "") for k in HEADER])

    return {"metrics": all_metrics, "out_dir": str(out_dir)}
