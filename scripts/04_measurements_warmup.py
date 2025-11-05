#!/usr/bin/env python3
"""
04_measurements_warmup.py — Thin runner for ERT "warm‑up" measurements
======================================================================

Purpose
-------
This script is a small, focused **launcher** that reads a YAML config file and
constructs the corresponding command‑line invocation for an external forward
simulation script (by default: ``build/measurements_warmup.py`` which mirrors a
Wenner‑α ERT forward generator such as ``ert_physics_forward_wenner.py``).
It converts a curated subset of YAML keys into CLI flags, validates a few
couplings (e.g., explicit active electrode lists), and then executes the
sub‑process.

Typical use
-----------
- You prepare a YAML file with fields describing the desired forward run
  (mesh size, number of fields, electrode geometry, noise model, etc.).
- You call this runner with ``--config path/to/config.yml``.
- The runner prints the fully expanded command for reproducibility and then
  executes it (unless ``--dry`` is given).

Inputs & outputs
----------------
- **Input**: A YAML file (``--config``) whose keys correspond to known command
  flags of the target script. Only keys listed in ``KEYMAP`` are forwarded.
- **Process**: Spawns the target script as a subprocess with the translated
  CLI arguments.
- **Output**: Whatever the target script writes (files/STDOUT/STDERR). This
  launcher itself only prints the composed command and returns the target's
  exit code (propagated).

Key features
------------
- **Deterministic CLI synthesis**: YAML → CLI via ``KEYMAP`` so runs are
  traceable and copy‑pastable.
- **Light validation**: Enforces that when ``active_policy: explicit`` is used,
  the number of ``active_indices`` matches ``n_active_elec``.
- **Safe booleans & lists**: Booleans become simple flags, lists are expanded
  as multiple positional values after their flag.

Assumptions & conventions
-------------------------
- The target script understands the flags as spelled in ``KEYMAP``.
- The YAML uses the *left‑hand* keys of ``KEYMAP``.
- Default Wenner pattern is ``"wenner-alpha"`` when ``pattern`` is omitted.

Failure modes
-------------
- Missing/incorrect YAML keys are simply ignored (not forwarded).
- If ``active_policy: explicit`` and the list length mismatches, the program
  exits with an explanatory error.
- The target script's non‑zero return code is propagated as this program's
  exit status.

Examples
--------
Dry‑run to see the composed command (no execution):

    python 04_measurements_warmup.py \
        --config configs/warmup.yml \
        --dry

Override the target script path:

    python 04_measurements_warmup.py \
        --config configs/warmup.yml \
        --script build/measurements_warmup.py

"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

# -----------------------------------------------------------------------------
# Mapping from YAML keys (left) to the CLI flags (right) expected by the target
# forward model script. Only keys present here are forwarded to the subprocess.
# Keep this list tight and explicit so that the runner is predictable.
#
# Notes
# -----
# - Boolean values are rendered as a bare flag (e.g., ``--pca`` when True).
# - Sequences (list/tuple) are expanded into repeated positionals after the
#   flag (e.g., ``--active-indices 0 1 2 3``).
# - Everything else is passed as ``--flag <value>`` with ``str(value)``.
# -----------------------------------------------------------------------------
KEYMAP = {
    # Data/model selection
    "pca": "--pca",                # Enable PCA basis / PCA‑space inputs
    "Z": "--Z",                    # Latent index / component spec

    # IO and dataset slicing
    "out": "--out",                # Output directory/path (handled by target)
    "n_fields": "--n-fields",      # Number of fields to generate
    "field_offset": "--field-offset",  # Starting field index offset

    # Grid / mesh controls
    "nz_full": "--nz-full",        # Full grid size (z)
    "nx_full": "--nx-full",        # Full grid size (x)
    "mesh_area": "--mesh-area",    # Area scaling for mesh generation
    "world_Lx": "--world-Lx",      # Physical domain width (x‑extent)
    "margin": "--margin",          # Free‑air margin padding, if used

    # Electrode geometry
    "n_elec": "--n-elec",          # Total number of electrodes available
    "dx_elec": "--dx-elec",        # Electrode spacing

    # Mode / pattern specifics
    "mode": "--mode",              # General mode switch understood by target
    "pattern": "--pattern",        # Measurement pattern name (defaulted below)
    # Wenner‑specific/active subset controls
    "n_active_elec": "--n-active-elec",  # Count of concurrently active electrodes
    "active_policy": "--active-policy",  # Policy: auto / explicit / etc.
    "active_indices": "--active-indices",# Explicit active electrode indices

    # Noise / compute controls
    "noise_rel": "--noise-rel",    # Relative noise level
    "jobs": "--jobs",              # Parallel workers for the target script
    "seed": "--seed",              # RNG seed forwarded to target
    "chunksize": "--chunksize",    # Work chunk size for batching
}


def build_argv_from_yaml(cfg: dict, script_path: str) -> list[str]:
    """Translate a YAML ``cfg`` dict into a subprocess ``argv``.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration. Only keys present in ``KEYMAP`` are used.
    script_path : str
        Path to the target script to execute (e.g., ``build/measurements_warmup.py``).

    Returns
    -------
    list[str]
        The full argument vector starting with the current Python executable,
        followed by the target script path and the translated flags.
    """
    argv: list[str] = [sys.executable, script_path]

    for key, flag in KEYMAP.items():
        if key not in cfg or cfg[key] is None:
            # Skip absent or explicitly null keys for a clean CLI.
            continue

        value = cfg[key]

        if isinstance(value, bool):
            # Booleans: include flag only when True, omit otherwise.
            if value:
                argv.append(flag)
            continue

        if isinstance(value, (list, tuple)):
            # Sequences: flag once, then each element as a separate token.
            argv.append(flag)
            argv.extend([str(x) for x in value])
        else:
            # Scalars: flag followed by stringified value.
            argv.extend([flag, str(value)])

    return argv


def main() -> None:
    """CLI entry point.

    - Loads the YAML file specified by ``--config``.
    - Applies small conveniences (default pattern, light validation).
    - Composes the target command and executes it, unless ``--dry``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Read a YAML config and launch a forward ERT warm‑up generator "
            "(Wenner‑α by default) by translating known YAML keys into CLI flags."
        ),
        epilog=(
            "Example: python 04_measurements_warmup.py --config configs/warmup.yml\n"
            "Use --dry to print the command without executing."
        ),
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--script",
        default=str(Path("build/measurements_warmup.py")),
        help=(
            "Path to the target forward script (e.g., ert_physics_forward_wenner.py). "
            "Override if your build/layout differs."
        ),
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Print the composed command and exit without executing.",
    )

    ns = parser.parse_args()

    # --- Load YAML ------------------------------------------------------------
    with open(ns.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- Convenience defaults & light validation -----------------------------
    # Default to Wenner‑alpha if not specified.
    if "pattern" not in cfg:
        cfg["pattern"] = "wenner-alpha"

    # If the user selects explicit active electrodes, enforce count consistency.
    if cfg.get("active_policy") == "explicit":
        indices = cfg.get("active_indices") or []
        n_active = int(cfg.get("n_active_elec", 16))
        if len(indices) != n_active:
            raise SystemExit(
                "When active_policy=explicit, active_indices must have length "
                f"{n_active} (got {len(indices)})."
            )

    # --- Build command --------------------------------------------------------
    argv = build_argv_from_yaml(cfg, ns.script)

    # Print a shell‑safe, human‑readable command for logging/repro.
    print("[runner] Exec:", shlex.join(argv))

    # --- Execute or dry‑run ---------------------------------------------------
    if ns.dry:
        return

    proc = subprocess.run(argv, check=False)

    # Propagate non‑zero exit codes to the caller (e.g., CI).
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
