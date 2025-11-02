# -*- coding: utf-8 -*-
"""
scripts/07_evaluate_inversion_vs_truth_multi.py

Thin runner that reads a YAML config and calls the build module:
    python scripts/07_evaluate_inversion_vs_truth_multi.py --config configs/eval/evaluate_inversion_vs_truth_multi.yml

This avoids a long CLI and keeps logic in build/.
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import yaml
import os, sys
# いまのファイル（scripts/...）から一つ上 = リポジトリ直下
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# すでに入っていなければ先頭に入れる（先頭に入れる＝最優先で解決）
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# リポジトリ内 import を想定（例: REPO_ROOT/build/invert_ops.py）
from build.evaluate_inversion_vs_truth_multi import run_from_cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    # Optional: allow overriding repo root for imports if user runs from elsewhere
    ap.add_argument("--repo-root", default=".", help="Repo root (to ensure 'build' is importable)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise SystemExit("Config must be a YAML mapping (key: value pairs)")

    result = run_from_cfg(cfg)
    print("[done] Wrote:", result["out_dir"])
    print("[done] Metrics:", json.dumps(result["metrics"][:2], indent=2) if result["metrics"] else "[]")

if __name__ == "__main__":
    main()
