#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/01_make_fields.py

Purpose
-------
This thin wrapper reads a YAML configuration file and launches a separate
"builder" script (default: ./build/make_fields.py) with command‑line
arguments derived from that YAML. It is intended to keep experiment configs
version‑controlled (YAML) while delegating the heavy lifting of dataset/field
generation to the builder.

Typical use
-----------
$ python scripts/01_make_fields.py \
    --config configs/data/make_fields.yml \
    --python /path/to/python \
    --script ./build/make_fields.py

YAML schema (keys understood by this wrapper)
--------------------------------------------
- out: str
    Output directory for generated artifacts (passed as --out)
- nx: int
    Horizontal resolution/size (passed as --nx)
- nz: int
    Vertical resolution/size (passed as --nz)
- per_class: int
    Number of samples per class (passed as --per-class)
- save_png: bool
    If true, request the builder to export preview PNGs (passed as --save-png)
- previews_per_class: int
    Number of example images per class when save_png is enabled (passed as --previews-per-class)
- dpi: int
    DPI for saved PNGs (passed as --dpi)
- seed: int
    Random seed for reproducibility (passed as --seed)

Notes
-----
* This script does not validate semantic correctness of values; it only
  forwards them to the builder. Any additional arguments must be added here.
* On Windows and POSIX, subprocess is invoked with shell=False for safety.
* PyYAML (yaml) is required to parse the config.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # PyYAML
except ImportError:  # pragma: no cover - explicit user guidance
    print("ERROR: PyYAML not found. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def build_cli_args(cfg: dict) -> list[str]:
    """Translate YAML keys into CLI args for the builder script.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.

    Returns
    -------
    list[str]
        Flat list of CLI tokens to append to the Python + script path.

    Mapping
    -------
    out                ->  --out <path>
    nx                 ->  --nx <int>
    nz                 ->  --nz <int>
    per_class          ->  --per-class <int>
    save_png (True)    ->  --save-png
    previews_per_class ->  --previews-per-class <int>
    dpi                ->  --dpi <int>
    seed               ->  --seed <int>
    """
    args: list[str] = []

    # Strings/paths
    if "out" in cfg:
        args += ["--out", str(cfg["out"])]

    # Integers – be explicit about type conversion to surface YAML mistakes early
    if "nx" in cfg:
        args += ["--nx", str(int(cfg["nx"]))]
    if "nz" in cfg:
        args += ["--nz", str(int(cfg["nz"]))]
    if "per_class" in cfg:
        args += ["--per-class", str(int(cfg["per_class"]))]

    # Booleans / feature toggles
    if cfg.get("save_png", False):
        args += ["--save-png"]

    # More integers
    if "previews_per_class" in cfg:
        args += ["--previews-per-class", str(int(cfg["previews_per_class"]))]
    if "dpi" in cfg:
        args += ["--dpi", str(int(cfg["dpi"]))]

    # Seed (optional but useful for reproducibility)
    if cfg.get("seed") is not None:
        args += ["--seed", str(int(cfg["seed"]))]

    return args


def main() -> None:
    """CLI entrypoint: parse args, read YAML, assemble and run the builder."""
    parser = argparse.ArgumentParser(
        description=(
            "Load a YAML config and invoke the builder script with matching CLI flags.\n"
            "Only a fixed set of keys are translated; see module docstring."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML config (e.g., configs/data/make_fields.yml)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use (defaults to the current interpreter)",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="./build/make_fields.py",
        help="Path to the builder script to execute",
    )

    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"ERROR: Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    # Load YAML (empty files become {})
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve the builder script path and construct the full command
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: Builder script not found: {script_path}", file=sys.stderr)
        sys.exit(3)

    cli_args = [args.python, str(script_path)] + build_cli_args(cfg)

    # Echo the command for reproducibility/debugging
    print("[make_fields] Running:", " ".join(str(a) for a in cli_args))

    # Use shell=False for Windows & POSIX safety. Propagate return code.
    proc = subprocess.run(cli_args, check=False)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
