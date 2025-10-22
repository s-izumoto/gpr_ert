# -*- coding: utf-8 -*-
"""
04_make_oracle_masks.py
-----------------------
Thin runner that reads a YAML config and launches build_oracle.py with the
corresponding CLI flags. Mirrors the style of 03_make_surrogate_pairs_pygimli.py.
"""
from __future__ import annotations
import argparse, shlex, subprocess, sys
from pathlib import Path
import yaml

# Map YAML dotted keys -> CLI flags for build_oracle.py
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

    # Mask parameters
    "mask.sigma_m":        "--sigma-m",
    "mask.thresh_pct":     "--thresh-pct",
    "mask.head":           "--head",
    "mask.scale_normalize":"--scale-normalize",
    "mask.downsample":     "--downsample",
    "mask.min_frac":       "--min-frac",
    "mask.coarse_q":       "--coarse-q",
    "mask.thresh_region":  "--thresh-region",
    "mask.min_component_frac": "--min-component-frac",
    
    # ROI
    "roi.apply_roi":           "--apply-roi",
    "roi.roi_top_width_frac":  "--roi-top-width-frac",
    "roi.roi_bottom_width_frac":"--roi-bottom-width-frac",
    "roi.roi_gamma":           "--roi-gamma",
    "roi.thresh_in_roi":       "--thresh-in-roi",

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
    ap.add_argument("--script", default=str(Path("./build/build_oracle.py")),
                    help="Path to the builder script (default: build_oracle.py next to this file)")
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
        # Booleans -> flags without value (only when True)
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
            continue
        # Lists -> multiple flags or a single flag then values (for nargs="+")
        if isinstance(val, (list, tuple)):
            # For flags that accept multiple values in one go (e.g., --sigma-m 0.5 1.0 2.0, --head 25 100)
            cmd.append(flag)
            cmd.extend(map(str, val))
            continue
        # Scalars
        cmd.extend([flag, str(val)])

    print("[runner] Exec:", shlex.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
