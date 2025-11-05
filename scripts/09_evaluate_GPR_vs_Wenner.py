#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_evaluate_GPR_vs_Wenner.py — Path-based runner
================================================

Purpose
-------
This script runs the evaluation process by **executing another Python script**
through its file path using `subprocess`.  
It simply forwards a specified YAML configuration file to the target script via:

    --config <path/to/config.yml>

The script itself performs no data processing or model evaluation.
Its only responsibilities are:
1. Verify that both the YAML configuration and target script exist.
2. Execute the target script with the YAML path as an argument.
3. Display execution timing and return the same exit code as the target.

Usage example
-------------
    python 09_evaluate_GPR_vs_Wenner.py \
        --config configs/eval/evaluate_GPR_vs_Wenner.yml \
        --script build/evaluate_GPR_vs_Wenner.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter


def _timestamp() -> str:
    """Return the current local timestamp as 'YYYY-MM-DD HH:MM:SS'."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _build_cmd(python_exe: str, script_path: Path, config_path: Path) -> list[str]:
    """Construct the command to execute the target script with the given config."""
    return [python_exe, str(script_path), "--config", str(config_path)]


def main() -> None:
    # Argument parsing
    ap = argparse.ArgumentParser(
        description="Execute the evaluation script by path, passing the YAML file via --config."
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    ap.add_argument(
        "--script",
        default=str(Path("./build/evaluate_GPR_vs_Wenner.py")),
        help="Path to the target evaluation script (default: ./build/evaluate_GPR_vs_Wenner.py).",
    )
    args = ap.parse_args()

    # Validate paths
    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"[error] Config file not found: {cfg_path}")

    script_path = Path(args.script).resolve()
    if not script_path.is_file():
        raise SystemExit(f"[error] Target script not found: {script_path}")

    # Build and run the subprocess command
    cmd = _build_cmd(sys.executable, script_path, cfg_path)
    print(f"[time] start:    {_timestamp()}")
    print("[runner] exec:", " ".join(cmd))

    t0 = perf_counter()
    proc = subprocess.run(cmd)
    elapsed = perf_counter() - t0

    # Print timing info and exit with the same code as the subprocess
    print(f"[time] end:      {_timestamp()}  elapsed: {elapsed:.3f}s")
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
