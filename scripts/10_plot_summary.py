
"""
10_plot_summary.py — Compare ERT inversion quality between WENNER and GPR designs

Purpose
-------
This script reads a single CSV file containing per-field/per-run evaluation metrics
and generates outputs **for each subset** ("top25", "all", "bottom25", "shallow50", "deep50"):

A) Distribution comparison:
   - Side-by-side boxplots for GPR vs WENNER (GPR left, WENNER right).

B) Relative-change analysis (GPR vs WENNER within the same label):
   - Horizontal bar chart of the *average* relative change across fields.
   - CSV of per-field relative % changes.
   - Spearman ρ is included; linear-scale RMSE/MAE are excluded in the relative plot.
   - Bars are green where GPR improves vs WENNER, red otherwise.
   - Both "Per-field mean %" and "Overall mean-based %" are plotted (visually distinguished).

C) Raw value exports:
   - `values_<subset>.csv`: raw rows in the subset (sorted).
   - `values_stats_<subset>.csv`: per-source summary stats per metric.
   - `values_paired_<subset>.csv`: per-label paired values (WENNER/GPR/DIFF).

D) Spread / stability:
   - `iqr_<subset>.csv`: per-source×metric Q1, median, Q3, IQR, Relative IQR(%).
   - `iqr_<subset>.png`: bar chart comparing IQR of GPR vs WENNER.
   - `relative_iqr_<subset>.png`: bar chart comparing Relative IQR(%) of GPR vs WENNER.

Relative IQR definition
-----------------------
Relative IQR (%) = 100 * (IQR / abs(median))
(Use abs(median) to avoid sign flips for metrics that can cross zero, e.g., bias.
If you prefer plain IQR/median, remove the abs() in the code where noted.)

Usage
-----
python 10_plot_summary.py \
    --csv data/evaluation/summary_metrics.csv \
    --outdir data/summary

Notes
-----
- Labels order is GPR, WENNER for visual consistency.
- Matplotlib < 3.9 falls back from `tick_labels` to `labels`.
- MPLBACKEND is forced to "Agg" for headless environments.
"""

import argparse
import pandas as pd
import os
# Force a headless backend to avoid Qt/GUI issues in batch environments
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# Central list of metrics used across plotting and exports
METRICS = [
    ("mae_log10", "MAE (log10 ρ)"),
    ("rmse_log10", "RMSE (log10 ρ)"),
    ("bias_log10", "Bias (log10 ρ)"),
    ("pearson_log10", "Pearson r"),
    ("spearman_log10", "Spearman ρ"),
    ("mae_linear", "MAE (ρ)"),
    ("rmse_linear", "RMSE (ρ)"),
    ("fourier_corr", "Fourier Spectrum Correlation"),
    ("morph_iou", "Morphological IoU"),
    ("js_divergence", "Jensen–Shannon divergence"),
]


def save_boxplot(series_w: pd.Series, series_o: pd.Series, title: str, ylabel: str, outpath: Path) -> None:
    """Save a boxplot that compares GPR vs WENNER for one metric."""
    fig = plt.figure(figsize=(6, 5))
    data = [series_o.dropna().values, series_w.dropna().values]

    # Matplotlib 3.9 renamed `labels` -> `tick_labels`; handle both for compatibility
    try:
        plt.boxplot(
            data,
            tick_labels=["GPR", "WENNER"],
            showmeans=True,
            meanline=True,
        )
    except TypeError:
        plt.boxplot(
            data,
            labels=["GPR", "WENNER"],
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
    """Create boxplots (GPR vs WENNER) for a dataframe subset."""
    wenner = df_subset[df_subset["source"].str.upper() == "WENNER"]
    GPR = df_subset[df_subset["source"].str.upper() == "GPR"]

    for col, ylabel in METRICS:
        if col not in wenner.columns or col not in GPR.columns:
            print(f"⚠ Column '{col}' missing, skipping plot.")
            continue
        title = f"{ylabel} — GPR vs WENNER ({subset_name})"
        filename = f"compare_{subset_name}__{col}.png"
        save_boxplot(wenner[col], GPR[col], title, ylabel, outdir / filename)


def relative_change_table(df_subset: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Compute per-field relative changes for the given metric names."""
    out: dict[str, pd.Series] = {}

    # Average possibly multiple rows per (label, source)
    base = df_subset.groupby(["label", "source"], as_index=False).mean(numeric_only=True)

    for col in metrics:
        pivot = base.pivot_table(index="label", columns="source", values=col, aggfunc="mean")
        for need in ["WENNER", "GPR"]:
            if need not in pivot.columns:
                pivot[need] = np.nan
        rel = 100.0 * (pivot["GPR"] - pivot["WENNER"]) / pivot["WENNER"].abs()
        out[col] = rel

    return pd.DataFrame(out)


def plot_relative_changes(df_subset: pd.DataFrame, subset_name: str, outdir: Path) -> None:
    """Plot *two* kinds of relative % changes per metric and save CSVs.

    1) Per-field mean relative change:
       mean_label( 100 * (GPR - WENNER) / |WENNER| )

    2) Overall mean-based relative change:
       100 * ( mean(GPR) - mean(WENNER) ) / | mean(WENNER) |

    Visual distinction:
      - Keep green/red for improvement/worsening.
      - Add hatching (PF='///', OVR='...'), slight alpha difference, and bold edge.
      - Annotate bars with 'PF' / 'OVR' + value.
    """
    metrics_full = [
        ("mae_log10", "MAE (log10 ρ)", "smaller_better"),
        ("rmse_log10", "RMSE (log10 ρ)", "smaller_better"),
        ("pearson_log10", "Pearson r", "larger_better"),
        ("spearman_log10", "Spearman ρ", "larger_better"),
        ("fourier_corr", "Fourier Spectrum Correlation", "larger_better"),
        ("morph_iou", "Morphological IoU", "larger_better"),
        ("js_divergence", "Jensen–Shannon divergence", "smaller_better"),
    ]

    has_w = df_subset["source"].str.upper() == "WENNER"
    has_o = df_subset["source"].str.upper() == "GPR"
    common_labels = set(df_subset[has_w]["label"]).intersection(df_subset[has_o]["label"])
    if not common_labels:
        print(f"⚠ No matching fields between WENNER and GPR in {subset_name}")
        return

    # 1) Per-field mean of relative changes
    metric_names = [m[0] for m in metrics_full]
    df_rel = relative_change_table(df_subset[df_subset["label"].isin(common_labels)], metric_names)
    df_rel.sort_index().to_csv(outdir / f"relative_change_{subset_name}.csv")

    # 2) Overall mean-based relative change per metric
    base = df_subset.groupby(["source"], as_index=False).mean(numeric_only=True)
    overall = {}
    for col, _nice, _sense in metrics_full:
        if col not in base.columns:
            overall[col] = np.nan
            continue
        mean_w = float(base.loc[base["source"].str.upper() == "WENNER", col].mean())
        mean_o = float(base.loc[base["source"].str.upper() == "GPR", col].mean())
        denom = np.abs(mean_w) if (not np.isnan(mean_w) and mean_w != 0.0) else np.nan
        overall[col] = np.nan if (isinstance(denom, float) and np.isnan(denom)) else 100.0 * (mean_o - mean_w) / denom

    # Summary CSV with both definitions
    rows = []
    for col, nice, _sense in metrics_full:
        per_field_mean = float(np.nanmean(df_rel[col].values))
        overall_mean = float(overall.get(col, np.nan))
        rows.append({
            "metric": col,
            "label": nice,
            "per_field_mean_percent": per_field_mean,
            "overall_mean_based_percent": overall_mean,
        })
    pd.DataFrame(rows).to_csv(outdir / f"relative_change_summary_{subset_name}.csv", index=False)

    # Prepare arrays for plotting
    labels = [nice for _col, nice, _sense in metrics_full]
    per_field_vals = [float(np.nanmean(df_rel[col].values)) for col, _nice, _sense in metrics_full]
    overall_vals   = [float(overall.get(col, np.nan))          for col, _nice, _sense in metrics_full]

    # Colors: green if improvement, red otherwise; HATCH/ALPHA distinguishes PF vs OVR
    per_field_colors, overall_colors = [], []
    for (col, _nice, sense), pf, ov in zip(metrics_full, per_field_vals, overall_vals):
        pf_improved = (pf < 0.0) if (sense == "smaller_better") else (pf > 0.0)
        ov_improved = (ov < 0.0) if (sense == "smaller_better") else (ov > 0.0)
        per_field_colors.append("#2ca02c" if pf_improved else "#d62728")
        overall_colors.append("#2ca02c" if ov_improved else "#d62728")

    y = np.arange(len(labels))
    h = 0.36  # bar height
    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw bars with distinct hatches and alpha; thicker edges for legibility
    pf_bars = ax.barh(
        y - h/2, per_field_vals, height=h,
        color=per_field_colors, edgecolor="black", linewidth=1.2,
        hatch="///", alpha=0.9, label="Per-field mean % (PF, hatched)"
    )
    ov_bars = ax.barh(
        y + h/2, overall_vals,   height=h,
        color=overall_colors, edgecolor="black", linewidth=1.2,
        hatch="...", alpha=0.65, label="Overall mean-based % (OVR, dotted)"
    )

    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(
        "Relative change of GPR vs WENNER (%)\n"
        "Negative is better for error/divergence metrics; Positive is better for correlations/IoU."
    )
    ax.set_title(
        f"Relative performance — GPR vs WENNER — {subset_name}\n"
        "(PF = Per-field mean, hatched ///  |  OVR = Overall mean-based, dotted …)"
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Annotate bars with PF/OVR and values (e.g., 'PF +3.2%')
    def _annotate(container, tag):
        try:
            # matplotlib >= 3.4
            labels_txt = []
            for rect in container:
                val = rect.get_width()
                if np.isnan(val):
                    labels_txt.append(f"{tag} n/a")
                else:
                    labels_txt.append(f"{tag} {val:+.1f}%")
            ax.bar_label(container, labels=labels_txt, padding=3, fontsize=8)
        except Exception:
            # Fallback: manual text
            for rect in container:
                x = rect.get_width()
                y_mid = rect.get_y() + rect.get_height()/2
                txt = f"{tag} {x:+.1f}%" if not np.isnan(x) else f"{tag} n/a"
                ax.text(x + (0.5 if np.isfinite(x) else 0.0), y_mid, txt, va="center", fontsize=8)

    _annotate(pf_bars, "PF")
    _annotate(ov_bars, "OVR")

    # Legend: show both meaning (improvement color is described in xlabel/title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(outdir / f"relative_change_{subset_name}.png", dpi=200)
    plt.close(fig)

    print(f"✅ Saved relative change plot & CSVs (per-field + overall, visually distinguished) for {subset_name}")


# === Spread (IQR) exports & plots ================================================================

def compute_iqr_table(df_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Q1, median, Q3, IQR and Relative IQR(%) for each source×metric in the subset.
    Returns a long-form DataFrame with columns:
      [source, metric, q1, median, q3, iqr, rel_iqr_percent]
    """
    rows = []
    for col, _nice in METRICS:
        if col not in df_subset.columns:
            continue
        for src, g in df_subset.groupby("source"):
            v = g[col].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            if v.size == 0:
                q1 = med = q3 = iqr = rel = np.nan
            else:
                q1 = np.nanpercentile(v, 25)
                med = np.nanmedian(v)
                q3 = np.nanpercentile(v, 75)
                iqr = q3 - q1
                denom = np.abs(med) if med not in (0, np.nan) else np.nan  # remove abs() if plain median is desired
                rel = np.nan if (denom is None or np.isnan(denom) or denom == 0.0) else 100.0 * (iqr / denom)
            rows.append({
                "source": str(src),
                "metric": col,
                "q1": q1,
                "median": med,
                "q3": q3,
                "iqr": iqr,
                "rel_iqr_percent": rel,
            })
    return pd.DataFrame(rows)


def save_iqr_csv_and_plots(df_subset: pd.DataFrame, subset_name: str, outdir: Path) -> None:
    """Save IQR table and create bar charts for IQR and Relative IQR comparing GPR vs WENNER."""
    tbl = compute_iqr_table(df_subset)
    if tbl.empty:
        print(f"⚠ IQR table empty for {subset_name}")
        return

    # Save CSV
    tbl.sort_values(["metric", "source"]).to_csv(outdir / f"iqr_{subset_name}.csv", index=False)
    print(f"✅ Saved IQR CSV for {subset_name}")

    # Build arrays for plotting (preserve METRICS order)
    metrics_present = [m for m, _ in METRICS if m in tbl["metric"].unique()]
    iqr_w, iqr_o, ril_w, ril_o, labels = [], [], [], [], []
    for m in metrics_present:
        row_w = tbl[(tbl["metric"] == m) & (tbl["source"].str.upper() == "WENNER")]
        row_o = tbl[(tbl["metric"] == m) & (tbl["source"].str.upper() == "GPR")]
        if row_w.empty and row_o.empty:
            continue
        labels.append(dict(METRICS)[m])
        iqr_w.append(float(row_w["iqr"].iloc[0]) if not row_w.empty else np.nan)
        iqr_o.append(float(row_o["iqr"].iloc[0]) if not row_o.empty else np.nan)
        ril_w.append(float(row_w["rel_iqr_percent"].iloc[0]) if not row_w.empty else np.nan)
        ril_o.append(float(row_o["rel_iqr_percent"].iloc[0]) if not row_o.empty else np.nan)

    x = np.arange(len(labels))
    width = 0.4

    # Plot absolute IQR
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x - width/2, iqr_o, width, label="GPR")
    ax1.bar(x + width/2, iqr_w, width, label="WENNER")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("IQR (Q3 − Q1)")
    ax1.set_title(f"IQR comparison — GPR vs WENNER — {subset_name}")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax1.legend()
    plt.tight_layout()
    fig1.savefig(outdir / f"iqr_{subset_name}.png", dpi=200)
    plt.close(fig1)

    # Plot Relative IQR (%)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(x - width/2, ril_o, width, label="GPR")
    ax2.bar(x + width/2, ril_w, width, label="WENNER")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Relative IQR (%)  = 100 × IQR / |median|")
    ax2.set_title(f"Relative IQR comparison — GPR vs WENNER — {subset_name}")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(outdir / f"relative_iqr_{subset_name}.png", dpi=200)
    plt.close(fig2)

    print(f"✅ Saved IQR and Relative IQR plots for {subset_name}")


# === Raw value export helpers ====================================================================

def save_values_raw(df_subset: pd.DataFrame, subset_name: str, outdir: Path) -> None:
    """Save the raw rows for this subset (sorted by label, source)."""
    keep_cols = ["label", "subset", "source"] + [c for c, _ in METRICS] + [
        "bias_log10", "spearman_log10", "mae_relative_percent", "rmse_relative_percent"
    ]
    keep_cols = [c for c in keep_cols if c in df_subset.columns]
    df_subset.sort_values(["label", "source"]).loc[:, keep_cols] \
        .to_csv(outdir / f"values_{subset_name}.csv", index=False)
    print(f"✅ Saved raw values CSV for {subset_name}")


def save_values_stats(df_subset: pd.DataFrame, subset_name: str, outdir: Path) -> None:
    """Save summary statistics per source×metric (count/mean/std/min/median/max/Q1/Q3)."""
    metric_cols = [c for c, _ in METRICS if c in df_subset.columns]
    if not metric_cols:
        print(f"⚠ No metric columns found for stats in {subset_name}")
        return

    def q1(x): return np.nanpercentile(x, 25)
    def q3(x): return np.nanpercentile(x, 75)

    stats = (
        df_subset
        .groupby("source")[metric_cols]
        .agg(["count", "mean", "std", "min", "median", "max", q1, q3])
        .sort_index()
    )
    # Flatten MultiIndex columns: e.g., mae_log10_mean
    stats.columns = [f"{m}_{stat if isinstance(stat, str) else stat.__name__}"
                     for m, stat in stats.columns]
    stats.to_csv(outdir / f"values_stats_{subset_name}.csv")
    print(f"✅ Saved stats CSV for {subset_name}")


def save_values_paired(df_subset: pd.DataFrame, subset_name: str, outdir: Path) -> None:
    """Save a long-form paired table (label, metric, WENNER, GPR, DIFF)."""
    has_w = df_subset["source"].str.upper() == "WENNER"
    has_o = df_subset["source"].str.upper() == "GPR"
    common_labels = sorted(set(df_subset[has_w]["label"]).intersection(df_subset[has_o]["label"]))
    if not common_labels:
        print(f"⚠ No matching labels to pair in {subset_name}")
        return

    # Average duplicates per (label, source)
    base = df_subset[df_subset["label"].isin(common_labels)] \
        .groupby(["label", "source"], as_index=False).mean(numeric_only=True)

    rows = []
    for col, _nice in METRICS:
        if col not in base.columns:
            continue
        pv = base.pivot_table(index="label", columns="source", values=col, aggfunc="mean")
        w = pv.get("WENNER")
        o = pv.get("GPR")
        # Align and iterate
        for lab in pv.index:
            wv = np.nan if w is None else w.get(lab, np.nan)
            ov = np.nan if o is None else o.get(lab, np.nan)
            rows.append({
                "label": lab,
                "metric": col,
                "WENNER": wv,
                "GPR": ov,
                "DIFF(GPR-WENNER)": (ov - wv) if (pd.notna(ov) and pd.notna(wv)) else np.nan
            })

    paired = pd.DataFrame(rows).sort_values(["metric", "label"])
    paired.to_csv(outdir / f"values_paired_{subset_name}.csv", index=False)
    print(f"✅ Saved paired values CSV for {subset_name}")


def export_all_values(df_subset: pd.DataFrame, subset_name: str, outdir: Path) -> None:
    """Run all value exports for a given subset."""
    save_values_raw(df_subset, subset_name, outdir)
    save_values_stats(df_subset, subset_name, outdir)
    save_values_paired(df_subset, subset_name, outdir)


# =================================================================================================

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
        for c in missing:
            df[c] = np.nan

    # For each subset, produce: (1) raw CSVs, (2) boxplots, (3) relative-change plot + CSV,
    # and (4) IQR/Relative IQR CSV & plots.
    for subset_name in ["top25", "all", "bottom25", "shallow50", "deep50"]:
        df_subset = df[df["subset"].str.lower() == subset_name]
        if df_subset.empty:
            print(f"⚠ No rows for subset '{subset_name}', skipping.")
            continue
        print(f"Processing subset: {subset_name} (n={len(df_subset)})")

        # (1) raw & paired values and summary stats
        export_all_values(df_subset, subset_name, outdir_p)

        # (2) Boxplots (existing behavior)
        make_comparisons(df_subset, subset_name, outdir_p)

        # (3) Relative-change (updated behavior)
        plot_relative_changes(df_subset, subset_name, outdir_p)

        # (4) Spread / stability
        save_iqr_csv_and_plots(df_subset, subset_name, outdir_p)

    print(f"✅ All plots & CSVs saved under: {outdir_p}")


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
        help="Directory to save output PNGs and CSVs",
    )
    args = ap.parse_args()
    main(args.csv, args.outdir)
