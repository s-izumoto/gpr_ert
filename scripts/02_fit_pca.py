#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_fit_pca.py — Configuration-driven PCA runner
================================================

This thin runner reads a YAML configuration file, converts its key–value
pairs into command-line flags, and then invokes an underlying **PCA builder**
script (default: `./build/fit_pca.py`). It is intended to keep your pipeline
declarative: edit a small YAML file, and this runner will dispatch the right
flags without you having to write complex shell commands.

Why use this?
-------------
- **Single source of truth:** Hyperparameters live in a YAML file that you can
  version-control and reuse.
- **Minimal CLI glue:** The mapping from YAML keys to CLI flags is centralized
  in one function (`kv_to_cli`).
- **Cross-platform:** Works on Windows PowerShell / CMD / Git Bash / Linux / macOS.

Typical YAML schema
-------------------
Below are the recognized YAML keys and the CLI flags they map to. Unknown keys
are ignored (safe by default). Boolean keys behave like `store_true` flags
(i.e., included only when `true`).

| YAML key          | CLI flag              | Type        | Meaning                                               |
|-------------------|-----------------------|-------------|-------------------------------------------------------|
| `ds`              | `--ds`                | str         | Path to dataset or bundle understood by the builder.  |
| `out`             | `--out`               | str         | Output directory for PCA artifacts.                   |
| `crop_frac`       | `--crop-frac`         | float       | Optional spatial/temporal cropping fraction.          |
| `max_components`  | `--max-components`    | int         | Upper bound on number of principal components.        |
| `target_var`      | `--target-var`        | float       | Target cumulative explained variance (0–1).           |
| `solver`          | `--solver`            | str         | PCA solver selection (e.g., 'full', 'randomized').    |
| `incremental`     | `--incremental`       | bool        | Use IncrementalPCA-style fitting if true.             |
| `batch`           | `--batch`             | int         | Batch size for incremental fitting (if supported).    |
| `per_class`       | `--per-class`         | bool        | Fit one PCA per class/label (builder-defined).        |
| `save_projections`| `--save-projections`  | bool        | Persist projected embeddings (X → Z).                 |
| `save_recon`      | `--save-recon`        | bool        | Persist reconstructions (Z → X̂).                     |
| `dpi`             | `--dpi`               | int         | DPI for any figures produced by the builder.          |

Usage
-----
From your repository root (paths are examples; adjust as needed):

    # Windows PowerShell / CMD / Git Bash / Linux / macOS
    python -m scripts.02_fit_pca --config configs/pca/pca_randomized.yml

    # If your builder lives elsewhere, override it with --script
    python -m scripts.02_fit_pca --config configs/pca/pca_randomized.yml ^
                                 --script build/fit_pca.py

Exit status
-----------
This runner returns the exit code of the underlying builder process. A non-zero
code indicates the builder failed.

Requirements
------------
- PyYAML (`pip install pyyaml`)

Notes
-----
- The actual *semantics* of the flags are implemented in the builder script
  you point to via `--script` (default: `./build/fit_pca.py`). This runner
  only translates YAML → CLI flags and executes the builder.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def kv_to_cli(cfg: dict) -> list[str]:
    """
    Convert a YAML config mapping to a flat list of CLI arguments.

    Rules:
    - Only keys present in `m` are considered; other keys are ignored.
    - Boolean values are treated as `store_true` flags (only added when True).
    - Non-boolean values are appended as `--flag value` pairs.

    Parameters
    ----------
    cfg : dict
        A dictionary produced by `yaml.safe_load` representing the YAML file.

    Returns
    -------
    list[str]
        A list such as `["--ds", "path/to/ds", "--out", "outputs/pca", "--incremental"]`
        ready to pass to `subprocess.run`.
    """
    # Mapping from YAML keys → underlying builder's CLI flags.
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

    args: list[str] = []
    for k, flag in m.items():
        if k not in cfg or cfg[k] is None:
            # Skip keys not present or explicitly set to null.
            continue
        v = cfg[k]
        if isinstance(v, bool):
            # Boolean keys act like "store_true" flags; only include when True.
            if v:
                args.append(flag)
        else:
            # Non-boolean: add as "--flag value". Always cast to string.
            args.extend([flag, str(v)])
    return args


def main() -> None:
    # --- Parse user-provided arguments (YAML path, optional builder override) ---
    ap = argparse.ArgumentParser(
        description=(
            "Read a YAML config, map keys to flags, and invoke a PCA builder script."
        )
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML config with PCA settings (key: value mapping).",
    )
    ap.add_argument(
        "--script",
        default=str(Path("./build/fit_pca.py")),
        help=(
            "Path to the underlying PCA builder script to execute "
            "(default: ./build/fit_pca.py)."
        ),
    )
    args = ap.parse_args()

    # --- Load and validate the YAML configuration ---
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"[error] Config file not found: {cfg_path}")
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SystemExit(f"[error] Failed to parse YAML: {e}")
    if not isinstance(cfg, dict):
        raise SystemExit("[error] Config must be a YAML mapping (key: value pairs).")

    # --- Compose the full command line and execute the builder ---
    cli_args = [sys.executable, args.script] + kv_to_cli(cfg)
    print("[runner] Exec:", " ".join(cli_args))
    proc = subprocess.run(cli_args)

    # Mirror the builder's exit code.
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
