# -*- coding: utf-8 -*-
"""
Evaluate ERT inversion results against PCA-reconstructed "true" fields (log10 resistivity),
with support for per-field NPZ bundles (Wenner vs. GPR method) or standalone NPZ files.

This module computes scalar accuracy metrics in both log and linear domains, depth-weighted
if requested, and generates diagnostic plots (parity, residual histogram, residual vs. truth).
It also includes spatial-similarity measures (Fourier spectrum correlation, morphological IoU
via Otsu binarization, and Jensen–Shannon divergence between histograms).

Two operation modes are supported via a configuration dictionary:

Mode A — "bundle":
    Compare two NPZ *bundles* that store multiple fields (e.g., WENNER vs GPR).
    Each bundle must contain keys of the form "inv_rho_cells__fieldNNN" for
    inversion cell values, and shared metadata arrays: "cell_centers", "L_world",
    "Lz", "world_xmin", "world_zmax".

    Required config keys:
        pca: path to PCA joblib (contains mean, components, nz, nx, ...)
        Z:   path to PCA coefficients array (npz with key "Z")
        wenner_bundle: path to WENNER inversion bundle (npz)
        GPR_bundle:  path to GPR inversion bundle (npz) — alias: inv_npz_bundle / inv_npz
        out_dir: output directory (created if missing)

    Optional:
        fields:     list of integer indices to restrict evaluation (0-based)
        lambda_depth: positive float to apply exponential depth weighting; 0 disables
        plots:      bool, write PNG diagnostics
        write_per_cell: bool, write per-cell CSV for each subset (ALL, BOTTOM25, TOP25)
        scatter_max: int cap for scatter points in plots
        verbose:    bool

Mode B — "standalone":
    Evaluate one or more *single-field* inversion NPZ files for a specific field index.

    Required config keys:
        pca, Z, out_dir, lambda_depth, plots, scatter_max (as above)
        mode: "standalone"
        inv_npz: [list of paths] (each NPZ with keys: inv_rho_cells, cell_centers, L_world, Lz, world_xmin, world_zmax)
        field_idx: integer field index to reconstruct truth from PCA/Z

    Optional:
        labels: list of display labels (same length as inv_npz)
        write_per_cell, verbose, etc.

Outputs:
    out_dir/summary_metrics.json — list of metric dicts (one per source × field × subset)
    out_dir/summary_metrics.csv  — CSV with the same metrics in tabular form
    out_dir/parity_*.png, residual_hist_*.png, residual_vs_true_*.png — when plots=True
    out_dir/per_cell_log10_*.csv — when write_per_cell=True

Key metrics per subset (ALL, BOTTOM25, TOP25 based on true log10-ρ quartiles):
    mae_log10, rmse_log10, bias_log10, pearson_log10, spearman_log10,
    mae_linear, rmse_linear, mae_relative_percent, rmse_relative_percent,
    fourier_corr, morph_iou, js_divergence

Typical usage from code:
    from evaluate_GPR_vs_Wenner import run_from_cfg
    result = run_from_cfg(cfg_dict)

Notes:
    * This module sets the Qt and Matplotlib backends to offscreen for headless environments.
    * All "true" maps are reconstructed in log10 domain from PCA meta + Z coefficients.
    * Depth weights w(z) = exp(-depth / lambda_depth), normalized to mean 1, when lambda_depth>0.
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
from skimage.filters import threshold_otsu
from skimage.morphology import disk, closing


def _js_divergence_hist(true_map: np.ndarray, pred_map: np.ndarray, bins: int = 64) -> float:
    """Compute Jensen–Shannon divergence between the histograms of two 2D maps.

    The input arrays are expected to be in log-space already. Both are flattened,
    histogrammed over a common range and bin count, smoothed by a tiny epsilon to
    avoid zero probabilities, and normalized. The result is scaled to [0, 1] by
    dividing by ln(2). Lower values indicate more similar distributions.

    Returns
    -------
    float
        JSD in [0, 1], or NaN if the inputs are degenerate.
    """
    t = np.ravel(np.nan_to_num(true_map, nan=0.0, posinf=0.0, neginf=0.0))
    p = np.ravel(np.nan_to_num(pred_map, nan=0.0, posinf=0.0, neginf=0.0))

    if t.size == 0 or p.size == 0:
        return float("nan")

    lo = float(min(np.min(t), np.min(p)))
    hi = float(max(np.max(t), np.max(p)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return float("nan")

    # density=True → probability density; normalize to a probability mass function
    T, _ = np.histogram(t, bins=bins, range=(lo, hi), density=True)
    P, _ = np.histogram(p, bins=bins, range=(lo, hi), density=True)

    eps = 1e-12
    T = T + eps
    P = P + eps
    T = T / T.sum()
    P = P / P.sum()

    M = 0.5 * (T + P)

    # Symmetrized KL: 0.5*(KL(P||M)+KL(T||M)) ∈ [0, ln 2]
    kl_PM = np.sum(P * (np.log(P) - np.log(M)))
    kl_TM = np.sum(T * (np.log(T) - np.log(M)))
    jsd = 0.5 * (kl_PM + kl_TM)

    denom = np.log(2.0)  # scale to [0, 1]
    return float(jsd / denom) if denom > 0 else float("nan")


# ================= PCA reconstruction =================

def _reconstruct_field_2d_from_pca__pcaops(field_idx: int, pca_joblib_path: "Path", z_path: "Path") -> np.ndarray:
    """Reconstruct a 2D log10-ρ field from PCA components and per-field coefficients.

    The joblib is expected to contain: mean (D,), components (k, D), nz, nx, ...
    The Z npz file must contain array "Z" with shape (N_fields, k).
    """
    from joblib import load as joblib_load
    meta = joblib_load(pca_joblib_path)  # keys: mean, components(k,D), nz, nx, crop_frac, ...
    mean = np.asarray(meta["mean"], dtype=np.float32)            # (D,)
    comps = np.asarray(meta["components"], dtype=np.float32)     # (k, D)
    nz, nx = int(meta["nz"]), int(meta["nx"])
    k = comps.shape[0]

    Z = np.load(z_path, allow_pickle=True)["Z"]                  # (N, k)
    if not (0 <= field_idx < Z.shape[0]):
        raise IndexError(f"field_index {field_idx} is out of range [0, {Z.shape[0]-1}]")
    z = Z[field_idx, :k].astype(np.float32)                      # (k,)

    x_flat = (z @ comps + mean).astype(np.float32)               # (D,)
    if x_flat.size != nz * nx:
        raise RuntimeError(f"Size mismatch: flat={x_flat.size} vs nz*nx={nz*nx}")
    arr2d = x_flat.reshape(nz, nx)
    return arr2d


def reconstruct_log10_field(field_idx: int, pca_joblib_path: Path, z_path: Path) -> np.ndarray:
    """Public wrapper to reconstruct a log10-ρ 2D field for a given index."""
    return _reconstruct_field_2d_from_pca__pcaops(field_idx, pca_joblib_path, z_path)


# ================= geometry / sampling =================

def sample_logR_to_cells(logR: np.ndarray, cell_centers: np.ndarray,
                         L_world: float, Lz: float, world_xmin: float, world_zmax: float) -> np.ndarray:
    """Sample a regular logR image at inversion cell-centers.

    The image spans [world_xmin, world_xmin+L_world] in x and [world_zmax-Lz, world_zmax] in z (top→bottom).
    Cell centers are mapped to nearest indices (with z inverted from up→down image coordinates).
    Returns a 1D vector aligned with the order of `cell_centers`.
    """
    NZ, NX = logR.shape
    xs = np.linspace(world_xmin, world_xmin + L_world, NX, dtype=np.float32)
    zs = np.linspace(world_zmax, world_zmax - Lz, NZ, dtype=np.float32)  # top → bottom
    cx = cell_centers[:, 0].astype(np.float32)
    cz = cell_centers[:, 1].astype(np.float32)
    ix = np.clip(np.round((cx - xs[0]) / (xs[-1] - xs[0]) * (NX - 1)).astype(int), 0, NX - 1)
    iz = np.clip(np.round((cz - zs[-1]) / (zs[0] - zs[-1]) * (NZ - 1)).astype(int), 0, NZ - 1)
    # flip z-index to map from physical up (+z) to image row index (downward)
    iz = (NZ - 1) - iz
    return logR[iz, ix]


def depth_weights(cz: np.ndarray, lambda_depth: float) -> np.ndarray:
    """Compute per-cell depth weights w(z) = exp(-depth / lambda_depth), normalized to mean 1.

    Parameters
    ----------
    cz : np.ndarray
        z-coordinates of cell centers (in world coordinates). Upward is positive; depth uses -cz.
    lambda_depth : float
        Decay length (m). If <= 0, return an all-ones vector.
    """
    if lambda_depth <= 0:
        return np.ones_like(cz, dtype=float)
    z = -cz  # positive depth [m]
    w = np.exp(-z / float(lambda_depth))
    w /= w.mean()
    return w


def _safe_corr(a, b) -> float:
    """Numerically robust Pearson correlation between two 1D arrays (mean-centered)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom > 0 else 0.0


# ================= plotting helpers =================

def _downsample_idx(n: int, k: int | None) -> np.ndarray:
    """Return evenly spaced indices to cap scatter size at `k` points (or all if None)."""
    if k is None or k <= 0 or k >= n:
        return np.arange(n, dtype=int)
    stride = max(1, int(np.floor(n / k)))
    return np.arange(0, n, stride, dtype=int)[:k]


def _parity_plot_log10(save_path: Path, *, true_log: np.ndarray, pred_log: np.ndarray,
                        label: str, metrics: dict, scatter_max: int | None = None) -> None:
    """Save a parity plot for log10-ρ (pred vs true) with a 1:1 reference line and summary text."""
    import matplotlib.pyplot as plt
    idx = _downsample_idx(len(true_log), scatter_max)
    lt_s, lp_s = true_log[idx], pred_log[idx]
    lim_min = min(np.min(true_log), np.min(pred_log))
    lim_max = max(np.max(true_log), np.max(pred_log))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(lt_s, lp_s, s=6, alpha=0.4)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], lw=1)
    ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("True log10(ρ)"); ax.set_ylabel("Pred log10(ρ)")
    ax.set_title(f"Parity (log10) – {label}")
    txt = (f"n={metrics.get('Nc','?')}\n"
           f"MAE={metrics['mae_log10']:.3f}  RMSE={metrics['rmse_log10']:.3f}\n"
           f"bias={metrics['bias_log10']:.3f}  r={metrics['pearson_log10']:.3f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top")
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


def _residual_hist_log10(save_path: Path, diff: np.ndarray, label: str) -> None:
    """Save a histogram of residuals (pred - true) in log10 domain."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diff, bins=50); ax.axvline(0, lw=1)
    ax.set_xlabel("Residual (pred - true) in log10(ρ)")
    ax.set_ylabel("Count"); ax.set_title(f"Residual histogram – {label}")
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


def _residual_vs_true(save_path: Path, lt: np.ndarray, diff: np.ndarray, label: str,
                      scatter_max: int | None = None) -> None:
    """Save a scatter plot of residual (pred - true) vs true log10-ρ."""
    import matplotlib.pyplot as plt
    idx = _downsample_idx(len(lt), scatter_max)
    lt_s, diff_s = lt[idx], diff[idx]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(lt_s, diff_s, s=6, alpha=0.4); ax.axhline(0, lw=1)
    ax.set_xlabel("True log10(ρ)"); ax.set_ylabel("Residual (pred - true) in log10(ρ)")
    ax.set_title(f"Residual vs True – {label}")
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


# ================= metrics core =================

def _compute_metrics(inv_rho_cells: np.ndarray, cell_centers: np.ndarray,
                     L_world: float, Lz: float, xmin: float, zmax: float,
                     true_log2d: np.ndarray, lambda_depth: float, label: str,
                     make_plots: bool, scatter_max: int, out_dir: Path,
                     write_per_cell: bool = False, mask: np.ndarray | None = None) -> dict:
    """Compute metrics (optionally over a masked subset of cells).

    Parameters
    ----------
    inv_rho_cells : (Nc,) array
        Inverted apparent resistivity per cell (Ω·m), linear domain.
    cell_centers : (Nc, 2) array
        Cell center coordinates (cx, cz).
    L_world, Lz, xmin, zmax : float
        Domain dimensions and origin used to project truth onto mesh.
    true_log2d : (NZ, NX) array
        True log10-ρ image to be sampled at cell centers.
    lambda_depth : float
        Depth weighting parameter (≤0 disables weighting).
    label : str
        Base label used for plot/CSV filenames.
    make_plots : bool
        Whether to write diagnostic plots.
    scatter_max : int
        Cap on scatter points for plots (for large Nc).
    out_dir : Path
        Output directory.
    write_per_cell : bool
        If True, write per-cell CSV for this subset.
    mask : optional (Nc,) bool array
        If provided, compute metrics only on cells where mask is True.
    """
    # Project truth (image) onto inversion mesh
    true_log_cells_all = sample_logR_to_cells(true_log2d, cell_centers, L_world, Lz, xmin, zmax)

    # inv_rho (Ω·m) → log10(ρ)
    inv_log_cells_all = np.log10(np.clip(np.asarray(inv_rho_cells, dtype=float), 1e-12, None))

    # Subset if a mask is provided
    if mask is not None:
        true_log_cells = true_log_cells_all[mask]
        pred_log_cells = inv_log_cells_all[mask]
        cc_used = cell_centers[mask]
    else:
        true_log_cells = true_log_cells_all
        pred_log_cells = inv_log_cells_all
        cc_used = cell_centers

    Nc = int(true_log_cells.size)
    if Nc == 0:
        # Empty subset → return NaNs to avoid crashes in downstream processing
        nan = float("nan")
        metrics = {
            "label": str(label), "Nc": 0,
            "mae_log10": nan, "rmse_log10": nan, "bias_log10": nan,
            "pearson_log10": nan, "spearman_log10": nan,
            "mae_linear": nan, "rmse_linear": nan,
            "mae_relative_percent": nan, "rmse_relative_percent": nan
        }
        return metrics

    diff_log = pred_log_cells - true_log_cells

    # Depth weights (normalized to mean 1)
    w = depth_weights(cc_used[:, 1], lambda_depth)
    def wmean(x): return float((w * x).sum() / w.sum())
    def wrmse(e): return float(np.sqrt(wmean(e ** 2)))

    # --- log domain metrics ---
    mae_log = wmean(np.abs(diff_log))
    rmse_log = wrmse(diff_log)
    bias_log = wmean(diff_log)

    # --- linear domain metrics ---
    true_rho = np.power(10.0, true_log_cells)
    inv_rho  = np.power(10.0, pred_log_cells)
    abs_err_lin = np.abs(inv_rho - true_rho)
    rel_err_lin = abs_err_lin / np.clip(true_rho, 1e-12, None)
    mae_rel = 100.0 * wmean(rel_err_lin)
    rmse_rel = 100.0 * wrmse(rel_err_lin)
    mae_lin = wmean(abs_err_lin)
    rmse_lin = wrmse(abs_err_lin)

    pearson_log = _safe_corr(pred_log_cells, true_log_cells)
    try:
        from scipy.stats import spearmanr
        spearman_log = float(spearmanr(pred_log_cells, true_log_cells, nan_policy="omit").correlation)
    except Exception:
        # Fallback: rank by argsort-of-argsort and compute Pearson
        rk1 = np.argsort(np.argsort(pred_log_cells)); rk2 = np.argsort(np.argsort(true_log_cells))
        spearman_log = _safe_corr(rk1, rk2)

    metrics = {
        "label": str(label),
        "Nc": Nc,
        "mae_log10": mae_log,
        "rmse_log10": rmse_log,
        "bias_log10": bias_log,
        "pearson_log10": pearson_log,
        "spearman_log10": spearman_log,
        "mae_linear": mae_lin,
        "rmse_linear": rmse_lin,
        "mae_relative_percent": mae_rel,
        "rmse_relative_percent": rmse_rel
    }

    # Diagnostics and per-cell CSV
    if make_plots:
        _parity_plot_log10(Path(out_dir) / f"parity_{label}.png",
                           true_log=true_log_cells, pred_log=pred_log_cells,
                           label=str(label), metrics=metrics, scatter_max=scatter_max)
        _residual_hist_log10(Path(out_dir) / f"residual_hist_{label}.png", diff_log, label)
        _residual_vs_true(Path(out_dir) / f"residual_vs_true_{label}.png",
                          true_log_cells, diff_log, label, scatter_max)

    out_dir = Path(out_dir)
    if write_per_cell:
        with (out_dir / f"per_cell_log10_{label}.csv").open("w", newline="", encoding="utf-8") as fp:
            wtr = csv.writer(fp)
            wtr.writerow(["cx", "cz", "inv_log10", "true_log10", "err_log10", "w"])
            for (cx, cz), il, tl, wl in zip(cc_used, pred_log_cells, true_log_cells, w):
                wtr.writerow([f"{cx:.6g}", f"{cz:.6g}", f"{il:.6g}", f"{tl:.6g}", f"{(il - tl):.6g}", f"{wl:.6g}"])

    # --- spatial similarity metrics ---
    try:
        # Re-map 1D inversion back onto the true_log2d image grid (approximate binning)
        NZ, NX = true_log2d.shape
        inv_log2d = np.full_like(true_log2d, np.nan)

        xs = np.linspace(xmin, xmin + L_world, NX)
        zs = np.linspace(zmax, zmax - Lz, NZ)
        ix = np.clip(np.round((cell_centers[:, 0] - xs[0]) / (xs[-1] - xs[0]) * (NX - 1)).astype(int), 0, NX - 1)
        iz = np.clip(np.round((cell_centers[:, 1] - zs[-1]) / (zs[0] - zs[-1]) * (NZ - 1)).astype(int), 0, NZ - 1)
        iz = (NZ - 1) - iz  # flip z (up→down)

        inv_log2d[iz, ix] = np.log10(np.clip(inv_rho_cells, 1e-12, None))

        fourier_corr = fourier_spectrum_correlation(true_log2d, inv_log2d)
        morph_iou    = morphological_iou(true_log2d, inv_log2d)
        js_div       = _js_divergence_hist(true_log2d, inv_log2d, bins=64)

    except Exception as e:
        print(f"[WARN] spatial similarity failed for {label}: {e}")
        fourier_corr, morph_iou, js_div = np.nan, np.nan, np.nan

    metrics["fourier_corr"] = fourier_corr
    metrics["morph_iou"] = morph_iou
    metrics["js_divergence"] = js_div

    return metrics


def fourier_spectrum_correlation(true_map: np.ndarray, pred_map: np.ndarray) -> float:
    """
    Compute correlation between the magnitude spectra of 2D Fourier transforms
    of two input maps. This version is more robust against DC bias, boundary
    artifacts, and NaN values.

    Steps:
        1. Replace NaN by the median of valid values.
        2. Apply a 2D Hanning window to reduce edge discontinuities.
        3. Perform 2D FFT and shift the zero-frequency component to the center.
        4. Remove DC component to avoid dominance by low frequencies.
        5. Optionally take log(1 + amplitude) to compress dynamic range.
        6. Compute Pearson correlation between the flattened spectra.
    """
    import numpy as np

    # Ensure float arrays
    t = np.array(true_map, dtype=float, copy=True)
    p = np.array(pred_map, dtype=float, copy=True)

    # Handle NaN: replace with median of valid region
    valid_t = np.isfinite(t)
    valid_p = np.isfinite(p)
    if np.any(valid_t):
        t[~valid_t] = np.nanmedian(t[valid_t])
    else:
        t[:] = 0
    if np.any(valid_p):
        p[~valid_p] = np.nanmedian(p[valid_p])
    else:
        p[:] = 0

    # Apply a 2D Hanning window to reduce spectral leakage
    wy = np.hanning(t.shape[0])
    wx = np.hanning(t.shape[1])
    window = wy[:, None] * wx[None, :]
    t *= window
    p *= window

    # 2D FFT and magnitude spectra
    ft_true = np.abs(np.fft.fftshift(np.fft.fft2(t)))
    ft_pred = np.abs(np.fft.fftshift(np.fft.fft2(p)))

    # Remove DC (center) component to avoid bias from zero frequency
    cy, cx = np.array(ft_true.shape) // 2
    ft_true[cy, cx] = 0.0
    ft_pred[cy, cx] = 0.0

    # Use logarithmic amplitude to compress the dynamic range
    ft_true = np.log1p(ft_true)
    ft_pred = np.log1p(ft_pred)

    # Flatten and compute Pearson correlation
    return float(np.corrcoef(ft_true.ravel(), ft_pred.ravel())[0, 1])

def morphological_iou(true_map: np.ndarray, pred_map: np.ndarray) -> float:
    """
    Morphological IoU with shared thresholding and optional closing (2D-safe).

    - Keep arrays 2D; compute stats (min/max/threshold) on the common-valid mask
    - Normalize to [0, 1] using valid-only min/max, but apply transform to full 2D
    - Threshold both maps with a shared policy; fill invalids with False
    - Optional morphological closing (disk(radius)) on 2D binaries
    """
    shared_threshold = globals().get("_MORPH_SHARED_THRESHOLD", "true_otsu")
    radius = int(globals().get("_MORPH_RADIUS", 0))

    # 2D common-valid mask
    valid = np.isfinite(true_map) & np.isfinite(pred_map)
    if not np.any(valid):
        return float("nan")

    # Copy to float 2D
    t2 = np.array(true_map, dtype=float, copy=True)
    p2 = np.array(pred_map, dtype=float, copy=True)

    # Normalize to [0,1] using min/max computed on valid pixels
    def norm01_2d(a: np.ndarray, valid_mask: np.ndarray):
        vals = a[valid_mask]
        amin = np.nanmin(vals); amax = np.nanmax(vals)
        if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
            return None
        out = (a - amin) / (amax - amin)
        return out

    t_norm = norm01_2d(t2, valid)
    p_norm = norm01_2d(p2, valid)
    if t_norm is None or p_norm is None:
        return float("nan")

    # Shared threshold selection (compute threshold on valid region of t_norm)
    if shared_threshold == "true_otsu":
        thr = float(threshold_otsu(t_norm[valid]))
        t_bin = t_norm > thr
        p_bin = p_norm > thr
    elif shared_threshold == "true_q50":
        thr = float(np.nanquantile(t_norm[valid], 0.5))
        t_bin = t_norm > thr
        p_bin = p_norm > thr
    else:  # "both_otsu" — compute each threshold on its own valid region
        t_thr = float(threshold_otsu(t_norm[valid]))
        p_thr = float(threshold_otsu(p_norm[valid]))
        t_bin = t_norm > t_thr
        p_bin = p_norm > p_thr

    # Invalidate outside the common mask so morphology/IoU ignores them
    t_bin = np.where(valid, t_bin, False)
    p_bin = np.where(valid, p_bin, False)

    # Optional morphological closing (2D structuring element)
    if radius > 0:
        se = disk(radius)
        t_bin = closing(t_bin, se)
        p_bin = closing(p_bin, se)

    # IoU over the common-valid domain
    inter = np.logical_and(t_bin, p_bin)[valid].sum()
    union = np.logical_or(t_bin, p_bin)[valid].sum()
    if union == 0:
        return float("nan")
    return float(inter / union)

# ================= bundled helpers =================
FIELD_PAT = re.compile(r"^inv_rho_cells__(field\d{1,4})$")


def list_field_labels_from_bundle(npz) -> list[str]:
    """Return sorted field labels (e.g., ["field000", ...]) present in a bundle npz."""
    return sorted(m.group(1) for k in npz.files if (m := FIELD_PAT.match(k)))


def field_label_to_index(label: str) -> int:
    """Extract integer index from a bundle field label (e.g., "field007" → 7)."""
    m = re.match(r"^field(\d{1,4})$", label)
    if not m:
        raise ValueError(f"Invalid field label: {label}")
    return int(m.group(1))


def load_bundle_meta(npz):
    """Load common geometry metadata from a bundle NPZ."""
    cell_centers = np.asarray(npz["cell_centers"], dtype=np.float32)
    L_world = float(np.asarray(npz["L_world"]).reshape(-1)[0])
    Lz      = float(np.asarray(npz["Lz"]).reshape(-1)[0])
    xmin    = float(np.asarray(npz["world_xmin"]).reshape(-1)[0])
    zmax    = float(np.asarray(npz["world_zmax"]).reshape(-1)[0])
    return cell_centers, L_world, Lz, xmin, zmax


def try_load_inv_for_label(npz, label: str):
    """Return the inversion vector for a given label in a bundle, or None if missing."""
    k = f"inv_rho_cells__{label}"
    if k not in npz.files:
        return None
    return np.asarray(npz[k], dtype=float)


# ================= public runner =================
HEADER = [
    "label", "subset", "source", "field_idx", "Nc", "lambda_depth",
    "mae_log10", "rmse_log10", "bias_log10", "pearson_log10", "spearman_log10",
    "mae_linear", "rmse_linear", "mae_relative_percent", "rmse_relative_percent",
    "fourier_corr", "morph_iou", "js_divergence"
]


def _eval_all_subsets(inv_rho_cells: np.ndarray, cell_centers: np.ndarray,
                      L_world: float, Lz: float, xmin: float, zmax: float, true_log2d: np.ndarray,
                      lambda_depth: float, base_label: str, make_plots: bool,
                      scatter_max: int, out_dir: Path, write_per_cell: bool) -> list[dict]:
    """Run evaluation for multiple subsets:
       - all
       - bottom25% / top25% by true log10-ρ
       - shallow50% / deep50% by physical depth from the top boundary (zmax)
    """
    # Precompute true log10 on cells (for bottom/top quartiles)
    true_log_cells_all = sample_logR_to_cells(true_log2d, cell_centers, L_world, Lz, xmin, zmax)
    q25 = float(np.quantile(true_log_cells_all, 0.25))
    q75 = float(np.quantile(true_log_cells_all, 0.75))
    m_bottom = true_log_cells_all <= q25
    m_top    = true_log_cells_all >= q75

    # Depth from the top: depth = zmax - cz  (top ~ 0, bottom ~ Lz)
    cz = cell_centers[:, 1].astype(float)
    depth = (float(zmax) - cz)
    # Robustness: ignore NaNs/inf when computing the halfway threshold
    valid_depth = np.isfinite(depth)
    if not np.any(valid_depth):
        # Fallback to unweighted split if geometry is degenerate
        halfway = float(Lz) / 2.0
    else:
        # Use geometric half by domain: Lz/2 is the intended split
        halfway = float(Lz) / 2.0
    m_shallow = depth <= halfway
    m_deep    = depth >  halfway

    out: list[dict] = []

    # all
    m_all = _compute_metrics(inv_rho_cells, cell_centers, L_world, Lz, xmin, zmax, true_log2d,
                             lambda_depth, f"{base_label}__all", make_plots, scatter_max, out_dir,
                             write_per_cell=write_per_cell, mask=None)
    m_all["subset"] = "all"
    out.append(m_all)

    # bottom25
    m_b = _compute_metrics(inv_rho_cells, cell_centers, L_world, Lz, xmin, zmax, true_log2d,
                           lambda_depth, f"{base_label}__bottom25", make_plots, scatter_max, out_dir,
                           write_per_cell=write_per_cell, mask=m_bottom)
    m_b["subset"] = "bottom25"
    out.append(m_b)

    # top25
    m_t = _compute_metrics(inv_rho_cells, cell_centers, L_world, Lz, xmin, zmax, true_log2d,
                           lambda_depth, f"{base_label}__top25", make_plots, scatter_max, out_dir,
                           write_per_cell=write_per_cell, mask=m_top)
    m_t["subset"] = "top25"
    out.append(m_t)

    # shallow50
    m_sh = _compute_metrics(inv_rho_cells, cell_centers, L_world, Lz, xmin, zmax, true_log2d,
                            lambda_depth, f"{base_label}__shallow50", make_plots, scatter_max, out_dir,
                            write_per_cell=write_per_cell, mask=m_shallow)
    m_sh["subset"] = "shallow50"
    out.append(m_sh)

    # deep50
    m_dp = _compute_metrics(inv_rho_cells, cell_centers, L_world, Lz, xmin, zmax, true_log2d,
                            lambda_depth, f"{base_label}__deep50", make_plots, scatter_max, out_dir,
                            write_per_cell=write_per_cell, mask=m_deep)
    m_dp["subset"] = "deep50"
    out.append(m_dp)

    return out


def run_from_cfg(cfg: dict) -> dict:
    """Execute evaluation from a configuration dictionary.

    Returns
    -------
    dict
        { 'metrics': list[dict], 'out_dir': str }
    """
    # --- common settings ---
    pca = Path(cfg.get("pca", "../data/interim/pca/pca_joint.joblib"))
    Z   = Path(cfg.get("Z",   "../data/interim/pca/Z.npz"))
    out_dir = Path(cfg.get("out_dir", "./eval_out")); out_dir.mkdir(parents=True, exist_ok=True)
    lambda_depth = float(cfg.get("lambda_depth", 0.0))
    make_plots    = bool(cfg.get("plots", False))           # default False
    write_per_cell = bool(cfg.get("write_per_cell", False)) # default False
    verbose       = bool(cfg.get("verbose", False))         # default False
    morph_cfg = cfg.get("morph", {}) if isinstance(cfg.get("morph", {}), dict) else {}
    globals()["_MORPH_SHARED_THRESHOLD"] = str(morph_cfg.get("shared_threshold", "true_otsu")).lower()
    globals()["_MORPH_RADIUS"] = int(morph_cfg.get("radius", 0))

    if verbose:
        print(f"[morph] shared_threshold={globals().get('_MORPH_SHARED_THRESHOLD')}, "
              f"radius={globals().get('_MORPH_RADIUS')}")

    scatter_max = int(cfg.get("scatter_max", 80000))

    mode = cfg.get("mode", "bundle").lower()
    all_metrics: list[dict] = []

    if mode == "bundle":
        # Resolve bundle paths; allow aliases for GPR
        wenner_bundle = cfg.get("wenner_bundle", None)
        GPR_bundle  = cfg.get("GPR_bundle", None) or cfg.get("inv_npz_bundle", None) or cfg.get("inv_npz", None)
        if isinstance(GPR_bundle, list):
            if len(GPR_bundle) == 1:
                GPR_bundle = GPR_bundle[0]
            else:
                raise SystemExit("For 'bundle' mode, 'GPR_bundle' must be a single NPZ path")

        if not wenner_bundle or not GPR_bundle:
            raise SystemExit("mode=bundle requires 'wenner_bundle' and 'GPR_bundle' (or 'inv_npz_bundle').")

        npz_w = np.load(wenner_bundle, allow_pickle=False)
        npz_o = np.load(GPR_bundle, allow_pickle=False)

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
                mlist_w = _eval_all_subsets(inv_w, cc_w, Lx_w, Lz_w, xmin_w, zmax_w, true_log2d,
                                            lambda_depth, f"WENNER_{lab}", make_plots, scatter_max, out_dir,
                                            write_per_cell=write_per_cell)
                for m in mlist_w:
                    m["label"] = lab; m["source"] = "WENNER"; m["field_idx"] = int(fidx)
                all_metrics.extend(mlist_w)
            else:
                print(f"[WARN] {lab} not found in WENNER bundle.")

            inv_o = try_load_inv_for_label(npz_o, lab)
            if inv_o is not None:
                cc_o, Lx_o, Lz_o, xmin_o, zmax_o = load_bundle_meta(npz_o)
                mlist_o = _eval_all_subsets(inv_o, cc_o, Lx_o, Lz_o, xmin_o, zmax_o, true_log2d,
                                            lambda_depth, f"GPR_{lab}", make_plots, scatter_max, out_dir,
                                            write_per_cell=write_per_cell)
                for m in mlist_o:
                    m["label"] = lab; m["source"] = "GPR"; m["field_idx"] = int(fidx)
                all_metrics.extend(mlist_o)
            else:
                print(f"[WARN] {lab} not found in GPR bundle.")

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

            mlist = _eval_all_subsets(inv_rho_cells, cell_centers, L_world, Lz, xmin, zmax, true_log2d,
                                      lambda_depth, lab, make_plots, scatter_max, out_dir,
                                      write_per_cell=write_per_cell)
            for m in mlist:
                m["label"] = lab; m["source"] = "standalone"; m["field_idx"] = int(field_idx)
            all_metrics.extend(mlist)
    else:
        raise SystemExit("Config 'mode' must be 'bundle' or 'standalone'.")

    # --- Save summary ---
    (Path(out_dir) / "summary_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    with (Path(out_dir) / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as fp:
        wtr = csv.writer(fp); wtr.writerow(HEADER)
        for m in all_metrics:
            wtr.writerow([
                m.get("label", ""),
                m.get("subset", ""),
                m.get("source", ""),
                m.get("field_idx", ""),
                m.get("Nc", ""),
                lambda_depth,
                m.get("mae_log10", ""),
                m.get("rmse_log10", ""),
                m.get("bias_log10", ""),
                m.get("pearson_log10", ""),
                m.get("spearman_log10", ""),
                m.get("mae_linear", ""),
                m.get("rmse_linear", ""),
                m.get("mae_relative_percent", ""),
                m.get("rmse_relative_percent", ""),
                m.get("fourier_corr", ""),
                m.get("morph_iou", ""),
                m.get("js_divergence", ""),
            ])

    return {"metrics": all_metrics, "out_dir": str(out_dir)}

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    try:
        import yaml  # PyYAML
    except Exception:
        yaml = None

    ap = argparse.ArgumentParser(
        description="Evaluate inversions vs truth from a YAML config."
    )
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"[error] Config not found: {cfg_path}")

    if yaml is None:
        raise SystemExit("[error] PyYAML is required: pip install pyyaml")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result = run_from_cfg(cfg) 
    print(f"[ok] wrote results to: {result['out_dir']}")
