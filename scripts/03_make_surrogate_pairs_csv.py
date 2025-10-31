
"""
Thin runner to launch the Wenner-α ERT forward generator from a YAML config.
It mirrors the style of 03_make_surrogate_pairs_pygimli.py but targets
ert_physics_forward_wenner.py by default.
"""
from __future__ import annotations
import argparse, shlex, subprocess, sys
from pathlib import Path
import yaml

# Map YAML keys to CLI flags for ert_physics_forward_wenner.py
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
    # Wenner-specific controls
    "pattern": "--pattern",
    "n_active_elec": "--n-active-elec",
    "active_policy": "--active-policy",
    "active_indices": "--active-indices",
    "abmn_csv": "--abmn-csv",
    # Misc
    "noise_rel": "--noise-rel",
    "jobs": "--jobs",
    "seed": "--seed",
    "chunksize": "--chunksize",
}

DEFAULT_SCRIPT = str(Path("build/ert_physics_forward_csv.py").resolve())

def build_argv_from_yaml(cfg: dict, script_path: str) -> list[str]:
    argv = [sys.executable, script_path]
    for k, flag in KEYMAP.items():
        if k not in cfg or cfg[k] is None:
            continue
        v = cfg[k]
        if isinstance(v, bool):
            if v:
                argv.append(flag)
            continue
        if isinstance(v, (list, tuple)):
            argv.append(flag)
            argv.extend([str(x) for x in v])
        else:
            argv.extend([flag, str(v)])
    return argv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML file path")
    ap.add_argument("--script", default=DEFAULT_SCRIPT, help="Path to ert_physics_forward_wenner.py (override)")
    ap.add_argument("--dry", action="store_true", help="Print the command and exit")
    ns = ap.parse_args()

    with open(ns.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Basic validation / conveniences
    if "pattern" not in cfg:
        cfg["pattern"] = "wenner-alpha"
    if cfg.get("active_policy") == "explicit":
        inds = cfg.get("active_indices") or []
        n_act = int(cfg.get("n_active_elec", 16))
        if len(inds) != n_act:
            raise SystemExit(f"When active_policy=explicit, active_indices must have length {n_act} (got {len(inds)})")

    argv = build_argv_from_yaml(cfg, ns.script)

    print("[runner] Exec:", shlex.join(argv))
    if ns.dry:
        return
    proc = subprocess.run(argv, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

if __name__ == "__main__":
    main()
