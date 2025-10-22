# -*- coding: utf-8 -*-
"""
04_reduce_oracle_diversity.py
-----------------------------
Thin runner that reads a YAML config and launches reduce_oracle_dataset.py
with the corresponding CLI flags. Mirrors the style of 03/04 scripts using
--config / --script / --python.
"""
from __future__ import annotations
import argparse, shlex, subprocess, sys
from pathlib import Path
import yaml

# Map YAML keys -> CLI flags for reduce_oracle_dataset.py
KEYMAP = {
    "npz":             "--npz",
    "out_prefix":      "--out-prefix",
    "rp_dim":          "--rp-dim",
    "seed":            "--seed",
    "dedup":           "--dedup",           # "simhash" | "cosine"
    "bits":            "--bits",            # only when dedup: simhash
    "cosine_thresh":   "--cosine-thresh",   # only when dedup: cosine
    "keep_frac":       "--keep-frac",
    "cap":             "--cap",
    "skip_large_mb":   "--skip-large-mb",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config for diversity reduction")
    ap.add_argument(
        "--script",
        default=str(Path("./build/oracle_reduce.py")),
        help="Path to reduce_oracle_dataset.py (default: ./reduce_oracle_dataset.py)",
    )
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise SystemExit("Config must be a YAML mapping")

    cmd = [args.python, args.script]

    # Walk the mapping
    for key, flag in KEYMAP.items():
        if key not in cfg or cfg[key] is None:
            continue
        val = cfg[key]
        # Booleans -> flags without value (only when True)
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
            continue
        # Lists/tuples -> append flag then all values (nargs='+')
        if isinstance(val, (list, tuple)):
            cmd.append(flag)
            cmd.extend(map(str, val))
            continue
        # Scalars
        cmd.extend([flag, str(val)])

    print("[runner] Exec:", shlex.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
