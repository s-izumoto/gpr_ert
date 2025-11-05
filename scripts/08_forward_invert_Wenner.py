#!/usr/bin/env python
"""
Driver script for ERT forward/inversion runs configured by YAML.

Summary
-------
This *thin runner* reads a YAML configuration file and translates its keys into
command‑line flags for an existing forward/inversion script that you specify via
``--script``. By default it targets ``./build/forward_invert_Wenner.py`` but you
can point it to any compatible script without touching that script’s code.

The runner keeps the physics/IO logic encapsulated in your target script while
centralizing experiment settings in a YAML file. It simply:

1. Loads a YAML mapping (sections like ``inputs``, ``selection``, ``geom``, …).
2. Maps those keys to CLI flags using ``KEYMAP`` below.
3. Spawns a subprocess: ``python <target_script> <mapped flags>``.
4. Prints start/end timestamps and exits with the child process’ return code.

Typical usage
-------------

```
python 08_forward_invert_Wenner.py \
  --config ./configs/forward_invert_Wenner.yml \
  --script ./build/forward_invert_Wenner.py
```

YAML layout (example, abridged)
--------------------------------

```
inputs:
  pca: ./data/PCA.npz
  Z:   ./data/Z.npy
  which_split: train
selection:
  n_fields: 10
  field_offset: 0
  fields: [0, 4, 9]          # If set, overrides n_fields/field_offset
geom:
  nz_full: 60
  nx_full: 30
  n_elec: 32
  dx_elec: 0.1
  margin: 0.5
  mesh_area: H2
  world_Lx: 3.2

design:
  type: wenner
  wenner_a_min: 1
  wenner_a_max: 16
  n_AB: 35
  n_MN_per_AB: 4
  dAB_min: 1
  dAB_max: 20
  dMN_min: 1
  dMN_max: 10

forward:
  mode: noisy
  noise_rel: 0.02

output:
  invert: true               # translates to --invert (boolean flag)
  invert_out: ./out/inv.png  # translates to --invert-out <path>
  invert_all: false
  inv_npz_out: ./out/inv.npz
  inv_save_all_png: false
  no_invert: false

misc:
  seed: 42
  workers: 8
  oversample_anomaly: false
  anomaly_topq: 0.10
  anomaly_factor: 2.0

# top-level convenience (outside sections)
out: ./out/forward_bundle
```

Key mapping
-----------
``KEYMAP`` defines a 1:1 mapping from (section, key) → target flag. Update it if
your target script’s argparse changes. Booleans are emitted as presence/absence
flags (no value). Lists/tuples are repeated as multiple flags, except for
``--fields`` which becomes a single comma‑separated list ("0,4,9").

Exit code & logging
-------------------
• The runner prints ``[time] start:`` and ``[time] end:`` with wall‑clock stamps
  and elapsed seconds.
• It exits with the child process’ return code (non‑zero means the target
  script reported an error).

Limitations / notes
-------------------
• The runner does not validate semantic consistency of YAML values; that is the
  target script’s responsibility.
• Unknown YAML keys are ignored unless added to ``KEYMAP``.
• Paths are passed as‑is to the child process; ensure they are valid relative to
  the working directory from which you run this script.
"""
from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path
from time import perf_counter
from datetime import datetime
import yaml

# Record start time for simple wall/CPU timing in the runner’s own process.
_start_wall_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
_start_t = perf_counter()
print(f"[time] start: {_start_wall_ts}")

# -----------------------------------------------------------------------------
# Map YAML keys → CLI flags of your *target* forward/inversion script
# -----------------------------------------------------------------------------
# Adjust this dictionary to mirror the argparse interface of the script passed
# via --script. The default below matches flags used in the provided pipeline.
KEYMAP = {
    # design selection (new)
    ("design", "type"): "--design-type",
    ("design", "wenner_a_min"): "--wenner-a-min",
    ("design", "wenner_a_max"): "--wenner-a-max",

    # inversion controls (optional; emitted only if truthy/non-None)
    ("output", "invert"): "--invert",
    ("output", "invert_out"): "--invert-out",
    ("output", "invert_all"): "--invert-all",
    ("output", "inv_npz_out"): "--inv-npz-out",
    ("output", "inv_save_all_png"): "--inv-save-all-png",
    ("output", "no_invert"): "--no-invert",

    # inputs
    ("inputs", "pca"): "--pca",
    ("inputs", "Z"): "--Z",
    ("inputs", "which_split"): "--which-split",

    # selection
    ("selection", "n_fields"): "--n-fields",
    ("selection", "field_offset"): "--field-offset",
    ("selection", "fields"): "--fields",
    ("selection", "field_from_seq_npz"): "--field-from-npz",

    # geometry
    ("geom", "nz_full"): "--nz-full",
    ("geom", "nx_full"): "--nx-full",
    ("geom", "n_elec"): "--n-elec",
    ("geom", "dx_elec"): "--dx-elec",
    ("geom", "margin"): "--margin",
    ("geom", "mesh_area"): "--mesh-area",
    ("geom", "world_Lx"): "--world-Lx",

    # design counts/limits
    ("design", "n_AB"): "--n-AB",
    ("design", "n_MN_per_AB"): "--n-MN-per-AB",
    ("design", "dAB_min"): "--dAB-min",
    ("design", "dAB_max"): "--dAB-max",
    ("design", "dMN_min"): "--dMN-min",
    ("design", "dMN_max"): "--dMN-max",

    # forward controls
    ("forward", "mode"): "--mode",
    ("forward", "noise_rel"): "--noise-rel",

    # parallel & misc
    ("misc", "seed"): "--seed",
    ("misc", "workers"): "--workers",

    # top-level convenience (not inside a section)
    ("root", "out"): "--out",

    # additional toggles
    ("misc", "oversample_anomaly"): "--oversample-anomaly",
    ("misc", "anomaly_topq"): "--anomaly-topq",
    ("misc", "anomaly_factor"): "--anomaly-factor",
}


def _append_flag(args: list[str], flag: str, value) -> None:
    """Append a flag and its value(s) to ``args`` according to type conventions.

    Rules
    -----
    • ``bool``: emit the flag only if True (presence/absence style).
    • ``list``/``tuple``: repeat the flag for each element, **except** for
      ``--fields`` where values are joined with commas into a single argument
      (the target script expects comma‑separated indices).
    • any other non‑None value: emit a single ``flag value`` pair.
    """
    if isinstance(value, bool):
        if value:
            args.append(flag)
    elif isinstance(value, (list, tuple)):
        if flag == "--fields":
            # fields are serialized as a single comma‑separated list, e.g. "0,4,9"
            args.extend([flag, ",".join(str(v) for v in value)])
        else:
            for v in value:
                args.extend([flag, str(v)])
    elif value is not None:
        args.extend([flag, str(value)])


def cfg_to_cli(cfg: dict) -> list[str]:
    """Translate a loaded YAML dict into a list of CLI arguments.

    The function gathers supported sections, allows a few top‑level keys (see
    ``root``), and walks ``KEYMAP`` to emit flags only for keys present in
    the YAML. Unknown keys are ignored here.
    """
    cli: list[str] = []

    # Recognized sections (missing sections default to empty dicts)
    sections = {
        "inputs": cfg.get("inputs", {}),
        "selection": cfg.get("selection", {}),
        "geom": cfg.get("geom", {}),
        "design": cfg.get("design", {}),
        "forward": cfg.get("forward", {}),
        "output": cfg.get("output", {}),
        "misc": cfg.get("misc", {}),
    }

    # Top‑level keys are exposed through the pseudo‑section "root"
    root = {k: v for k, v in cfg.items() if k not in sections}

    for (sect, key), flag in KEYMAP.items():
        if sect == "root":
            val = root.get(key)
        else:
            val = sections.get(sect, {}).get(key)
        _append_flag(cli, flag, val)

    return cli


def main() -> None:
    """Parse args, load YAML, build CLI, and execute the target script.

    This function is intentionally minimal: all heavy lifting remains in the
    target script. Any non‑zero return code from the child process is propagated
    as this runner’s exit status.
    """
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default=str(Path("./configs/forward_invert_Wenner.yml")),
        help="Path to YAML config.",
    )

    ap.add_argument(
        "--script",
        default=str(Path("./build/forward_invert_Wenner.py")),
        help=(
            "Path to the existing forward/inversion script to run "
            "(e.g., make_ert_surrogate_dataset_unique_parallel_nosplit.py)"
        ),
    )

    args = ap.parse_args()

    # Load YAML as a mapping (dict). Fail early if the structure is not a map.
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise SystemExit("Config must be a YAML mapping (key: value pairs)")

    # Convert config → CLI list and spawn the subprocess
    cli_args = [sys.executable, args.script] + cfg_to_cli(cfg)
    print("[runner] Exec:", " ".join(cli_args))

    proc = subprocess.run(cli_args)

    # Print end timing and propagate child return code
    _end_t = perf_counter()
    _end_wall_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _elapsed = _end_t - _start_t
    print(f"[time] end:   {_end_wall_ts}  elapsed: {_elapsed:.3f}s")

    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
