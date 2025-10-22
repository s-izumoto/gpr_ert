"""
Thin runner that reads a YAML config and dispatches to build_pca_latent.py
Usage:
  python -m scripts.02_fit_pca_and_project --config configs/pca/pca_randomized.yml

Notes:
- Requires PyYAML: `pip install pyyaml`
- Works on Windows PowerShell / CMD / Git Bash.
"""
from __future__ import annotations
import argparse, sys, subprocess
from pathlib import Path
import yaml


def kv_to_cli(cfg: dict) -> list[str]:
    """Map config keys to build_pca_latent.py CLI flags."""
    m = {
        "ds": "--ds",
        "out": "--out",
        "crop_frac": "--crop-frac",
        "max_components": "--max-components",
        "target_var": "--target-var",
        "solver": "--solver",
        "incremental": "--incremental",
        "batch": "--batch",
        "per_class": "--per-class",
        "save_projections": "--save-projections",
        "save_recon": "--save-recon",
        "dpi": "--dpi",
    }
    args = []
    for k, flag in m.items():
        if k not in cfg or cfg[k] is None:
            continue
        v = cfg[k]
        if isinstance(v, bool):
            if v:
                args.append(flag)  # store_true flags
        else:
            args.extend([flag, str(v)])
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--script", default=str(Path("./build/pca_ops.py")),
                    help="Path to build_pca_latent.py (default: repo root)")
    args = ap.parse_args()

    cfg_path = Path(ns.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise SystemExit("Config must be a YAML mapping (key: value pairs)")

    cli_args = [sys.executable, args.script] + kv_to_cli(cfg)
    print("[runner] Exec:", " ".join(cli_args))
    proc = subprocess.run(cli_args)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()