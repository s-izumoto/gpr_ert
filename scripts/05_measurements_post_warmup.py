"""
05_measurements_post_warmup.py
--------------------------------
Purpose
    Thin command-line runner that reads a YAML configuration and launches an
    external ERT forward generator (e.g., a post-warmup measurement script).
    It translates YAML keys into command-line flags, applies a few smart
    defaults, normalizes acquisition pattern names (Wenner-alpha, Schlumberger,
    Dipole-Dipole, or "all"), and then calls the target Python script.

How it works
    1) Load a YAML file (given by --config).
    2) Normalize/validate a few fields (e.g., pattern aliases, active policy).
    3) Map the YAML keys to CLI flags using KEYMAP.
    4) Build an argv list and spawn the target script via subprocess.run().
       Use --dry to print the command without executing.

YAML expectations (selected keys)
    pca, Z, out, n_fields, field_offset, nz_full, nx_full,
    n_elec, dx_elec, margin, mesh_area, mode, world_Lx,
    pattern, n_active_elec, active_policy, active_indices,
    noise_rel, jobs, seed, chunksize

Smart defaults
    n_fields: 5
    n_active_elec: 32
    pattern: "wenner-alpha" (if not provided)

Examples
    Dry-run (only show the command):
        python 05_measurements_post_warmup.py --config configs/example.yml --dry

    Execute with a custom target script:
        python 05_measurements_post_warmup.py \\
            --config configs/example.yml \\
            --script build/measurements_post_warmup.py

Exit codes
    0 on success. Non-zero if the spawned process returns a non-zero code
    or if basic validation fails (e.g., active_policy=explicit but wrong
    number of active_indices).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

# Map YAML keys to CLI flags expected by the target forward script.
# NOTE: The target script is referenced via --script (defaults below).
KEYMAP = {
    "pca": "--pca",
    "Z": "--Z",
    "out": "--out",
    "n_fields": "--n-fields",
    "field_offset": "--field-offset",
    "nz_full": "--nz-full",
    "nx_full": "--nx-full",
    "n_elec": "--n-elec",
    "dx_elec": "--dx-elec",
    "margin": "--margin",
    "mesh_area": "--mesh-area",
    "mode": "--mode",
    "world_Lx": "--world-Lx",
    # Acquisition / pattern controls
    "pattern": "--pattern",
    "n_active_elec": "--n-active-elec",
    "active_policy": "--active-policy",
    "active_indices": "--active-indices",
    # Misc / runtime
    "noise_rel": "--noise-rel",
    "jobs": "--jobs",
    "seed": "--seed",
    "chunksize": "--chunksize",
}


def build_argv_from_yaml(cfg: dict, script_path: str) -> list[str]:
    """
    Convert a YAML-backed configuration into an argv list for the target script.

    Rules
    -----
    * bool True -> include the flag without a value (e.g., --some-flag)
    * list/tuple -> "--flag item1 item2 ..."
    * other scalar -> "--flag value"
    * keys not present or None are skipped
    """
    argv: list[str] = [sys.executable, script_path]

    for key, flag in KEYMAP.items():
        if key not in cfg or cfg[key] is None:
            # Skip missing/None values: the target script will use its own defaults
            continue

        value = cfg[key]

        if isinstance(value, bool):
            # Only add boolean flags when True
            if value:
                argv.append(flag)
            continue

        if isinstance(value, (list, tuple)):
            # For sequences, pass the flag once followed by all items
            argv.append(flag)
            argv.extend(str(x) for x in value)
            continue

        # Fallback: scalar value, pass as "--flag value"
        argv.extend([flag, str(value)])

    return argv


def main() -> None:
    # ---- CLI parsing ----
    ap = argparse.ArgumentParser(
        description=(
            "Read a YAML config, map keys to CLI flags, and launch an ERT "
            "forward measurement script (post-warmup)."
        )
    )
    ap.add_argument("--config", required=True, help="Path to YAML configuration file.")
    ap.add_argument(
        "--script",
        default=str(Path("build/measurements_post_warmup.py")),
        help="Path to the target ERT forward runner script to execute.",
    )
    ap.add_argument(
        "--dry",
        action="store_true",
        help="Print the constructed command and exit without running it.",
    )
    ns = ap.parse_args()

    # ---- Load YAML ----
    with open(ns.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- Basic normalization / validation ----

    # Smart defaults suitable for this project (safe to override in YAML)
    cfg.setdefault("n_fields", 5)
    cfg.setdefault("n_active_elec", 32)

    # Normalize pattern aliases to canonical names expected downstream
    if "pattern" in cfg and isinstance(cfg["pattern"], str):
        p = cfg["pattern"].lower()
        if p in ("wenner", "wenner-alpha", "wenner_alpha"):
            cfg["pattern"] = "wenner-alpha"
        elif p in ("schlumberger", "schlum"):
            cfg["pattern"] = "schlumberger"
        elif p in ("dipole", "dipole-dipole", "dipole_dipole", "dd"):
            cfg["pattern"] = "dipole-dipole"
        elif p in ("all", "any"):
            cfg["pattern"] = "all"

    # Default to Wenner-alpha if not specified
    if "pattern" not in cfg:
        cfg["pattern"] = "wenner-alpha"

    # If using an explicit active set, ensure the count matches n_active_elec
    if cfg.get("active_policy") == "explicit":
        indices = cfg.get("active_indices") or []
        n_act = int(cfg.get("n_active_elec", 32))
        if len(indices) != n_act:
            raise SystemExit(
                f"When active_policy=explicit, active_indices must have length "
                f"{n_act} (got {len(indices)})."
            )

    # ---- Build argv and (optionally) run ----
    argv = build_argv_from_yaml(cfg, ns.script)

    # Always show the exact command for reproducibility
    print("[runner] Exec:", shlex.join(argv))

    if ns.dry:
        # Dry-run: do not execute
        return

    # Execute the target script; bubble up a non-zero exit code if it fails
    proc = subprocess.run(argv, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
