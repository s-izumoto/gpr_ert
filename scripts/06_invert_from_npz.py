# scripts/03_invert_from_npz.py
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import os, sys
# いまのファイル（scripts/...）から一つ上 = リポジトリ直下
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# すでに入っていなければ先頭に入れる（先頭に入れる＝最優先で解決）
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# リポジトリ内 import を想定（例: REPO_ROOT/build/invert_ops.py）
from build.inversion import run_inversion


def main():
    ap = argparse.ArgumentParser(description="Thin runner: read YAML and call build.invert_ops.run_inversion()")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise SystemExit("Config must be a YAML mapping (key: value pairs)")

    # 必須
    npz = cfg.get("npz")
    if not npz:
        raise SystemExit("YAML must contain 'npz'.")

    # 任意（デフォルトは build 側）
    out = cfg.get("out")
    out_log = cfg.get("out_log")
    out_dir = cfg.get("out_dir")
    bundle_out = cfg.get("bundle_out")
    field_index = cfg.get("field_index")
    all_fields = bool(cfg.get("all_fields", False))
    images_all = bool(cfg.get("images_all", False))

    n_elec = int(cfg.get("n_elec", 32))
    dx_elec = float(cfg.get("dx_elec", 1.0))
    world_Lx = float(cfg.get("world_Lx", 31.0))
    margin = float(cfg.get("margin", 3.0))
    nx_full = int(cfg.get("nx_full", 400))
    nz_full = int(cfg.get("nz_full", 100))
    mesh_area = float(cfg.get("mesh_area", 0.1))

    bundle = run_inversion(
        npz_path=npz,
        out=out,
        out_log=out_log,
        out_dir=out_dir,
        bundle_out=bundle_out,
        field_index=field_index,
        all_fields=all_fields,
        images_all=images_all,
        n_elec=n_elec,
        dx_elec=dx_elec,
        world_Lx=world_Lx,
        margin=margin,
        nx_full=nx_full,
        nz_full=nz_full,
        mesh_area=mesh_area,
    )

    print(f"[runner] bundle saved at: {bundle}")


if __name__ == "__main__":
    main()
