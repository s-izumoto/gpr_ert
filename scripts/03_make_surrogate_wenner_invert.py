#!/usr/bin/env python
"""
Thin runner that reads a YAML config and dispatches to a *script you choose* via --script.
Default target script: ./build/ert_physics_forward.py (changeable).

This keeps your existing forward model file untouched and simply maps YAML keys to its CLI flags.
"""
from __future__ import annotations

import argparse, sys, subprocess
from pathlib import Path
import yaml

# ---- Map YAML keys -> CLI flags of your forward script ----
# Adjust this mapping to match the argparse of your target script.
# Below matches the flags you shared (make_ert_surrogate_dataset_unique_parallel*.py).
KEYMAP = {

    # design selection (new)
    ("design", "type"): "--design-type",
    ("design", "wenner_a_min"): "--wenner-a-min",
    ("design", "wenner_a_max"): "--wenner-a-max",
    # inversion controls (optional)
    ("output", "invert"): "--invert",
    ("output", "invert_out"): "--invert-out",

    # inputs
    ("inputs", "pca"): "--pca",
    ("inputs", "Z"): "--Z",
    ("inputs", "which_split"): "--which-split",

    # selection
    ("selection", "n_fields"): "--n-fields",
    ("selection", "field_offset"): "--field-offset",

    # geometry
    ("geom", "nz_full"): "--nz-full",
    ("geom", "nx_full"): "--nx-full",
    ("geom", "n_elec"): "--n-elec",
    ("geom", "dx_elec"): "--dx-elec",
    ("geom", "margin"): "--margin",
    ("geom", "mesh_area"): "--mesh-area",
    ("geom","world_Lx"): "--world-Lx",

    # design
    ("design", "n_AB"): "--n-AB",
    ("design", "n_MN_per_AB"): "--n-MN-per-AB",
    ("design", "dAB_min"): "--dAB-min",
    ("design", "dAB_max"): "--dAB-max",
    ("design", "dMN_min"): "--dMN-min",
    ("design", "dMN_max"): "--dMN-max",

    # forward
    ("forward", "mode"): "--mode",
    ("forward", "noise_rel"): "--noise-rel",

    # parallel & misc
    ("misc", "seed"): "--seed",

    # top-level convenience
    ("root", "out"): "--out",

    # others
    ("misc","oversample_anomaly"): "--oversample-anomaly",
    ("misc","anomaly_topq"): "--anomaly-topq",
    ("misc","anomaly_factor"): "--anomaly-factor",
}


def _append_flag(args: list[str], flag: str, value):
    if isinstance(value, bool):
        if value:  # store_true
            args.append(flag)
    elif isinstance(value, (list, tuple)):
        # pass repeated flags: --foo v1 --foo v2 ...
        for v in value:
            args.extend([flag, str(v)])
    elif value is not None:
        args.extend([flag, str(value)])


def cfg_to_cli(cfg: dict) -> list[str]:
    cli: list[str] = []

    # flatten with sections
    sections = {
        "inputs": cfg.get("inputs", {}),
        "selection": cfg.get("selection", {}),
        "geom": cfg.get("geom", {}),
        "design": cfg.get("design", {}),
        "forward": cfg.get("forward", {}),
        "output": cfg.get("output", {}),
        "misc": cfg.get("misc", {}),
    }
    # also allow some top-level keys (e.g., out)
    root = {k: v for k, v in cfg.items() if k not in sections}

    for (sect, key), flag in KEYMAP.items():
        if sect == "root":
            val = root.get(key)
        else:
            val = sections.get(sect, {}).get(key)
        _append_flag(cli, flag, val)

    return cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path("./configs/simulate/wenner_invert.yml")), help="Path to YAML config.")
    ap.add_argument(
        "--script",
        default=str(Path("./build/ert_wenner_invert.py")),
        help="Path to the *existing* forward script to run (e.g., make_ert_surrogate_dataset_unique_parallel_nosplit.py)",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise SystemExit("Config must be a YAML mapping (key: value pairs)")

    cli_args = [sys.executable, args.script] + cfg_to_cli(cfg)
    print("[runner] Exec:", " ".join(cli_args))
    proc = subprocess.run(cli_args)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()