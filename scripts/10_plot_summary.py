"""
10_plot_summary.py — Compare ERT inversion quality between WENNER and OTHER designs

Purpose
-------
This script reads a single CSV file containing per-field/per-run evaluation metrics
and generates two kinds of outputs **for each subset** ("top25", "all", "bottom25"):

1) Side‑by‑side **boxplots** comparing the distributions of metrics
   for **OTHER vs WENNER** (OTHER on the left, WENNER on the right).
2) A **relative‑change analysis** that aligns WENNER/OTHER by field label and
   computes percentage differences for each metric. It saves:
   - a horizontal bar chart of the *average* relative change across fields, and
   - a CSV of per‑field relative changes.

Expected CSV schema (columns)
-----------------------------
Required: "label", "subset", "source" and the following metric columns if available.
Missing metric columns are auto‑filled with NaN so plotting can proceed when possible.

Metrics used here:
- mae_log10, rmse_log10, bias_log10, pearson_log10, spearman_log10
- mae_linear, rmse_linear, mae_relative_percent, rmse_relative_percent
- fourier_corr, morph_iou, js_divergence

Subsets
-------
Rows must include a "subset" column taking one of: "top25", "all", "bottom25"
(the selection logic is assumed to be performed upstream and written to the CSV).

Definitions
-----------
- Boxplots: show the distribution of metric values across rows per source.
- Relative change (per field):
    rel% = 100 * (OTHER - WENNER) / |WENNER|
  The script first averages duplicate rows per (label, source) if they exist,
  then computes the difference on the per‑field means.

Usage
-----
python 10_plot_summary.py \
    --csv data/summary/summary_metrics.csv \
    --outdir data/summary

Outputs
-------
PNG figures and CSVs are written under --outdir. Filenames include the subset and metric.
Boxplots are named:     compare_{subset}__{metric}.png
Relative change figure: relative_change_{subset}.png
Relative change table:  relative_change_{subset}.csv

Notes
-----
- Labels order is intentionally **OTHER, WENNER** to make comparisons consistent.
- If Matplotlib < 3.9 is used, the script falls back from `tick_labels` to `labels`.
- MPLBACKEND is forced to "Agg" so the script runs headless (useful on servers/HPC).
"""

import argparse
import pandas as pd
import os
# Force a headless backend to avoid Qt/GUI issues in batch environments
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def save_boxplot(series_w: pd.Series, series_o: pd.Series, title: str, ylabel: str, outpath: Path) -> None:
    """Save a boxplot that compares OTHER vs WENNER for one metric.

    Parameters
    ----------
    series_w : pd.Series
        Metric values for rows where source == "WENNER".
    series_o : pd.Series
        Metric values for rows where source == "OTHER".
    title : str
        Figure title.
    ylabel : str
        Y‑axis label describing the metric (e.g., "MAE (log10 ρ)").
    outpath : Path
        Output PNG path.
    """
    fig = plt.figure(figsize=(6, 5))

    # Order matters: OTHER first (left), then WENNER (right)
    data = [series_o.dropna().values, series_w.dropna().values]

    # Matplotlib 3.9 renamed `labels` -> `tick_labels`; handle both for compatibility
    try:
        plt.boxplot(
            data,
            tick_labels=["OTHER", "WENNER"],
            showmeans=True,
            meanline=True,
        )
    except TypeError:
        plt.boxplot(
            data,
            labels=["OTHER", "WENNER"],
            showmeans=True,
            meanline=True,
        )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_comparisons(df_subset: pd.DataFrame, subset_name: str, outdir: Path) -> None:
    """Create boxplots (OTHER vs WENNER) for a dataframe subset.

    The subset is typically one of {"top25", "all", "bottom25"}.
    Metrics missing in the dataframe are skipped with a console warning.
    """
    wenner = df_subset[df_subset["source"].str.upper() == "WENNER"]
    other = df_subset[df_subset["source"].str.upper() == "OTHER"]

    metrics = [
        ("mae_log10", "MAE (log10 ρ)"),
        ("rmse_log10", "RMSE (log10 ρ)"),
        ("pearson_log10", "Pearson r"),
        ("mae_linear", "MAE (ρ)"),
        ("rmse_linear", "RMSE (ρ)"),
        ("fourier_corr", "Fourier Spectrum Correlation"),
        ("morph_iou", "Morphological IoU"),
        ("js_divergence", "Jensen–Shannon divergence"),
    ]

    for col, ylabel in metrics:
        if col not in wenner.columns or col not in other.columns:
            print(f"⚠ Column '{col}' missing, skipping plot.")
            continue
        title = f"{ylabel} — OTHER vs WENNER ({subset_name})"
        filename = f"compare_{subset_name}__{col}.png"
        save_boxplot(wenner[col], other[col], title, ylabel, outdir / filename)


def relative_change_table(df_subset: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Compute per‑field relative changes for a list of metric names.

    Steps
    -----
    1) Aggregate duplicates by taking the mean per (label, source).
    2) Pivot to columns {WENNER, OTHER} per metric.
    3) Compute rel% = 100 * (OTHER - WENNER) / |WENNER|.

    Returns
    -------
    pd.DataFrame
        Index: label, Columns: metric names, Values: relative % change.
    """
    out: dict[str, pd.Series] = {}

    # Average possibly multiple rows per (label, source)
    base = df_subset.groupby(["label", "source"], as_index=False).mean(numeric_only=True)

    for col in metrics:
        pivot = base.pivot_table(index="label", columns="source", values=col, aggfunc="mean")

        # Ensure both columns exist to avoid KeyError; fill with NaN otherwise
        for need in ["WENNER", "OTHER"]:
            if need not in pivot.columns:
                pivot[need] = np.nan

        rel = 100.0 * (pivot["OTHER"] - pivot["WENNER"]) / pivot["WENNER"].abs()
        out[col] = rel

    return pd.DataFrame(out)


def plot_relative_changes(df_subset: pd.DataFrame, subset_name: str, outdir: Path) -> None:
    """Plot the mean relative % change for all metrics and save the per‑field table.

    The bar colors encode the *sense* of improvement:
    - green for metrics where larger is better (positive is improvement)
    - red for metrics where smaller is better (negative is improvement)
    """
    metrics_full = [
        ("mae_log10", "MAE (log10 ρ)", "smaller_better"),
        ("rmse_log10", "RMSE (log10 ρ)", "smaller_better"),
        ("pearson_log10", "Pearson r", "larger_better"),
        ("mae_linear", "MAE (ρ)", "smaller_better"),
        ("rmse_linear", "RMSE (ρ)", "smaller_better"),
        ("fourier_corr", "Fourier Spectrum Correlation", "larger_better"),
        ("morph_iou", "Morphological IoU", "larger_better"),
        ("js_divergence", "Jensen–Shannon divergence", "smaller_better"),
    ]

    # Only compare labels that appear in BOTH sources
    has_w = df_subset["source"].str.upper() == "WENNER"
    has_o = df_subset["source"].str.upper() == "OTHER"
    common_labels = set(df_subset[has_w]["label"]).intersection(df_subset[has_o]["label"])
    if not common_labels:
        print(f"⚠ No matching fields between WENNER and OTHER in {subset_name}")
        return

    # Compute per‑field relative change on the intersected set
    metric_names = [m[0] for m in metrics_full]
    df_rel = relative_change_table(df_subset[df_subset["label"].isin(common_labels)], metric_names)

    # Save the detailed table
    df_rel.sort_index().to_csv(outdir / f"relative_change_{subset_name}.csv")

    # Visualize metric‑wise mean relative change (ignoring NaN)
    fig, ax = plt.subplots(figsize=(9, 5))
    means: list[float] = []
    labels: list[str] = []
    colors: list[str] = []

    for col, nice, sense in metrics_full:
        means.append(np.nanmean(df_rel[col].values))
        labels.append(nice)
        colors.append("#2ca02c" if sense == "larger_better" else "#d62728")

    ax.barh(labels, means, color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel(
        "Relative change of OTHER vs WENNER (%)\n"
        "Note: negative is better for error metrics; positive is better for correlation/IoU."
    )
    ax.set_title(f"Relative performance difference per metric — OTHER vs WENNER — {subset_name}")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(outdir / f"relative_change_{subset_name}.png", dpi=200)
    plt.close(fig)

    print(f"✅ Saved relative change plot & CSV for {subset_name}")


def main(csv_path: str, outdir: str) -> None:
    """Entry point: read CSV, validate columns, and generate all outputs per subset."""
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Columns required by downstream code. Missing ones are created as NaN to keep compatibility
    required_cols = [
        "label", "subset", "source",
        "mae_log10", "rmse_log10", "bias_log10",
        "pearson_log10", "spearman_log10",
        "mae_linear", "rmse_linear",
        "mae_relative_percent", "rmse_relative_percent",
        "fourier_corr", "morph_iou", "js_divergence",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠ Missing columns in CSV: {missing}")
        # Fill absent columns with NaN so the rest of the pipeline continues gracefully
        for c in missing:
            df[c] = np.nan

    # For each subset, produce (1) boxplots and (2) relative‑change figures + CSV
    for subset_name in ["top25", "all", "bottom25"]:
        df_subset = df[df["subset"].str.lower() == subset_name]
        if df_subset.empty:
            print(f"⚠ No rows for subset '{subset_name}', skipping.")
            continue
        print(f"Processing subset: {subset_name} (n={len(df_subset)})")

        # (1) Boxplots preserved from existing behavior
        make_comparisons(df_subset, subset_name, outdir_p)

        # (2) NEW: within‑label % difference of OTHER vs WENNER
        plot_relative_changes(df_subset, subset_name, outdir_p)

    print(f"✅ All plots saved under: {outdir_p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default="data/evaluation/summary_metrics.csv",
        help="Input CSV path (must include 'subset' and 'source' columns)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="data/summary",
        help="Directory to save output PNGs and relative‑change CSVs",
    )
    args = ap.parse_args()
    main(args.csv, args.outdir)
