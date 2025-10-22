#!/usr/bin/env python
# scripts/01_make_fields.py

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # PyYAML
except ImportError:
    print("ERROR: PyYAML not found. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

def build_cli_args(cfg: dict) -> list[str]:
    """Translate YAML keys into CLI args for build_ert_ml_dataset_nosplit.py."""
    args = []
    if "out" in cfg: args += ["--out", str(cfg["out"])]
    if "nx" in cfg: args += ["--nx", str(int(cfg["nx"]))]
    if "nz" in cfg: args += ["--nz", str(int(cfg["nz"]))]
    if "per_class" in cfg: args += ["--per-class", str(int(cfg["per_class"]))]
    if cfg.get("save_png", False): args += ["--save-png"]
    if "previews_per_class" in cfg: args += ["--previews-per-class", str(int(cfg["previews_per_class"]))]
    if "dpi" in cfg: args += ["--dpi", str(int(cfg["dpi"]))]
    if cfg.get("seed") is not None: args += ["--seed", str(int(cfg["seed"]))]
    return args

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", type=str, required=True,
                    help="Path to YAML config (e.g., configs/data/make_fields.yml)")
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python interpreter to use (defaults to current)")
    ap.add_argument("--script", type=str, default="./build/fields_generate.py",
                    help="Path to the build script")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cli_args = [args.python, args.script] + build_cli_args(cfg)

    print("[make_fields] Running:", " ".join(str(a) for a in cli_args))
    # Use shell=False for Windows & POSIX safety
    proc = subprocess.run(cli_args, check=False)
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
