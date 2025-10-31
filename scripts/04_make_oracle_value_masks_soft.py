# -*- coding: utf-8 -*-
"""
04_make_oracle_value_masks.py
-----------------------------
YAML runner for build_oracle_value.py (value-based oracle masks).
Usage is analogous to 04_make_oracle_masks.py.

Example YAML:

inputs:
  dataset: /path/to/ert_surrogate_unified.npz   # or use "Z: /path/to/Z_unified.npz"
  pca:      /path/to/pca_latent.joblib
out:        ./oracle_pairs_value
selection:
  rows_per_field: 512            # only needed if dataset meta lacks n_AB, n_MN_per_AB
geom:
  nz_full: 100
  nx_full: 400
  world_Lx: 31.0
mask:
  head: [25, 100]                # coarse grid (Hc, Wc)
  thresh_region: shallow         # or "all"
  downsample: any                # or "quantile"
  pct: 80.0                      # -> top20/bottom20 split
roi:
  apply_roi: false               # set true to mask outside inverted-triangle ROI
  roi_top_width_frac: 1.0
  roi_bottom_width_frac: 0.10
  roi_gamma: 1.0
teacher:
  tau: 0.08                      # soft-rank temperature (smaller = sharper)
misc:
  preview_n: 3
"""
from __future__ import annotations
import argparse, shlex, subprocess, sys
from pathlib import Path
import yaml

# YAML key -> CLI flag mapping for build_oracle_value.py
KEYMAP = {
    # Sources
    "inputs.dataset":      "--dataset",
    "inputs.Z":            "--Z",
    "inputs.pca":          "--pca",
    "out":                 "--out",

    # If using --dataset and meta doesn't include n_AB, n_MN_per_AB
    "selection.rows_per_field": "--rows-per-field",

    # Geometry
    "geom.nz_full":        "--nz-full",
    "geom.nx_full":        "--nx-full",
    "geom.world_Lx":       "--world-Lx",

    # Mask / region
    "mask.head":           "--head",
    "mask.thresh_region":  "--thresh-region",
    "mask.downsample":     "--downsample",
    "mask.pct":            "--pct",

    # ROI
    "roi.apply_roi":           "--apply-roi",
    "roi.roi_top_width_frac":  "--roi-top-width-frac",
    "roi.roi_bottom_width_frac":"--roi-bottom-width-frac",
    "roi.roi_gamma":           "--roi-gamma",

    # Teacher (soft-rank)
    "teacher.tau":         "--tau",

    # Misc
    "misc.preview_n":      "--preview-n",
}

def nested_get(d: dict, path: list[str]):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML file describing oracle build")
    ap.add_argument("--script", default=str(Path("./build/build_oracle_value_soft.py")),
                    help="Path to the value builder script (default: build_oracle_value.py next to this file)")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise SystemExit("Config must be a YAML mapping")

    cmd = [args.python, args.script]

    # Walk the mapping
    for key, flag in KEYMAP.items():
        val = nested_get(cfg, key.split("."))
        if val is None:
            continue

        # bool flags -> add flag only when True
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
            continue

        # list/tuple -> expand elements (e.g., --head H W)
        if isinstance(val, (list, tuple)):
            cmd.append(flag)
            cmd.extend(map(str, val))
            continue

        # scalar
        cmd.extend([flag, str(val)])

    print("[runner] Exec:", shlex.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
