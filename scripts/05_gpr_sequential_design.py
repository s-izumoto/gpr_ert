# scripts/06_gpr_sequential_design.py

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil
import csv
import numpy as np
# --- add repo root to sys.path (before any other imports) ---
import os, sys
import warnings
try:
    import yaml  # PyYAML 推奨
except Exception:
    yaml = None
# いまのファイル（scripts/...）から一つ上 = リポジトリ直下
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# すでに入っていなければ先頭に入れる（先頭に入れる＝最優先で解決）
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
from build.gpr_seq_core import GPRSeqConfig, run_from_cfg  # 既存の import に合わせて

# === KEYMAP: 表記ゆれ（エイリアス）を正規キーへ統一 ===
KEYMAP = {
    # fields 系
    "field": "fields",
    "fields": "fields",
    "field-index": "fields",
    "field_index": "fields",
    # よくある表記ゆれ（必要に応じて拡張）
    "n_elecs": "n_elec",
    "nElec": "n_elec",
    "minGap": "min_gap",
    "ddKmax": "dd_kmax",
    "dd-kmax": "dd_kmax",
    "schlAMax": "schl_a_max",
    "schl-a-max": "schl_a_max",
}

def _normalize_config_keys(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        out[KEYMAP.get(k, k)] = v
    return out

def _normalize_fields(val) -> list[int]:
    if val is None:
        return []
    if isinstance(val, int):
        return [int(val)]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    if isinstance(val, str):
        # "0,1,3" 形式も許容
        return [int(x.strip()) for x in val.split(",") if x.strip() != ""]
    raise TypeError(f"fields の型が不正です: {type(val)}")


def _append_csv(src_csv: Path, dst_csv: Path):
    """src のデータ行を dst に追記（src の1行目ヘッダはスキップ）"""
    if not src_csv.exists():
        return
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with src_csv.open("r", encoding="utf-8", newline="") as fr:
        reader = csv.reader(fr)
        rows = list(reader)
    if not rows:
        return
    header, data = rows[0], rows[1:]
    write_header = not dst_csv.exists()
    with dst_csv.open("a", encoding="utf-8", newline="") as fw:
        w = csv.writer(fw)
        if write_header:
            w.writerow(header)
        if data:
            w.writerows(data)

def _npz_to_dict(npz_path: Path) -> dict:
    """seq_log.npz を dict に読み出す（np.ndarrayへ変換）"""
    out = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for k in z.files:
            out[k] = np.asarray(z[k])
    return out

def _fields_from_medoids(npz_path: str | Path) -> list[int]:
    with np.load(npz_path, allow_pickle=False) as z:
        if "src_index" not in z.files:
            raise KeyError(f"'src_index' not found in {npz_path}. available={list(z.files)}")
        idx = np.asarray(z["src_index"], dtype=np.int64).ravel()
    if (idx < 0).any():
        raise ValueError(f"src_index contains negative values: {np.unique(idx[idx<0])[:8]}")
    return sorted(set(int(i) for i in idx.tolist()))  # 0-basedのままでOK

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--field-index", type=int, nargs="+", default=[0,2],
        help="対象フィールドの行インデックス。例: --field-index 0 3 7"
    )
    p.add_argument(
        "--run-tag", type=str, default=None,
        help="出力フォルダ名に付ける任意タグ（未指定なら日時）"
    )

    p.add_argument(
        "--fields-from-medoids", type=str, default=None,
        help="cluster_pca_kmeans.py の出力 medoids_index.npz へのパス。中の src_index を fields として使う。"
    )

    # 既存の引数があればここに維持（例: その他のフラグ）

    ns = p.parse_args()

    # === YAML を読み込み ===
    cfg_path = Path(ns.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        if yaml is not None:
            yaml_cfg = yaml.safe_load(f) or {}
        else:
            # PyYAML が無い環境向けフォールバック（YAMLがJSON互換の場合のみ）
            yaml_cfg = json.load(f)

    # キー正規化（field/field-index など → fields）
    yaml_cfg_norm = _normalize_config_keys(yaml_cfg if isinstance(yaml_cfg, dict) else {})

    # YAMLのロード後（yaml_cfg_norm がある状態で）
    fields = []

    # 1) YAMLの fields_from_medoids が最優先
    yaml_medoids = yaml_cfg_norm.get("fields_from_medoids", None)
    cli_medoids  = getattr(ns, "fields_from_medoids", None)

    if yaml_medoids:
        fields = _fields_from_medoids(yaml_medoids)
        print(f"[run] fields_from_medoids (YAML): {fields}")
    elif cli_medoids:
        fields = _fields_from_medoids(cli_medoids)
        print(f"[run] fields_from_medoids (CLI): {fields}")
    else:
        # 従来の fields または --field-index を踏襲
        fields_yaml = _normalize_fields(yaml_cfg_norm.get("fields")) if "fields" in yaml_cfg_norm else []
        fields_cli  = _normalize_fields(ns.field_index) if getattr(ns, "field_index", None) is not None else []
        fields = fields_yaml if fields_yaml else fields_cli
        print(f"[run] fields from YAML/CLI: {fields}")

    if not fields:
        fields = [0]

    fields = sorted(set(int(x) for x in fields))
    print(f"[run] fields to process: {fields}")



    # すでに上で fields は決定済み（YAML/CLI/medoids から）
    # ここで GPRSeqConfig に渡す dict を作るとき、未知キーは入れない
    cfg_payload = dict(yaml_cfg_norm)

    # ★ 重要：GPRSeqConfig が受け取らないキーを必ず除外
    cfg_payload.pop("fields_from_medoids", None)
    cfg_payload.pop("fields", None)  # fields はローカル変数で使うので渡さない

    try:
        cfg = GPRSeqConfig(**cfg_payload)
    except TypeError as e:
        warnings.warn(f"[warn] Unknown keys in YAML may be ignored by GPRSeqConfig: {e}")
        # さらに未知キーを削る必要があればここで対応（通常は不要）
        cfg = GPRSeqConfig(**cfg_payload)


    # === bundle ディレクトリを作成 ===
    stamp = ns.run_tag or f"{datetime.now():%Y%m%d_%H%M%S}"
    bundle_dir = Path(cfg.out_dir) / f"{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # bundle CSV のパス
    bundle_cand_csv = bundle_dir / "bundle_candidate_stats.csv"
    bundle_gprp_csv = bundle_dir / "bundle_gpr_params.csv"

    # NPZ を集約するためのバッファ（フィールド毎に保持）
    # キー命名: "<key>__field%03d"
    bundle_npz_dict = {}
    fields_done = []

    for fid in fields:
        # 各フィールドはサブフォルダに隔離して実行
        sub_dir = bundle_dir / f"field{int(fid):03d}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        # out_dir を差し替えた cfg を作る
        cfg_i = GPRSeqConfig(**{**asdict(cfg), "out_dir": str(sub_dir)})

        # 実行
        out_dir = run_from_cfg(cfg_i, field_index=int(fid))
        print(f"[done] field={fid} -> {out_dir}")

        # ---- CSV を bundle に追記 ----
        cand_csv = Path(out_dir) / "candidate_stats.csv"
        gprp_csv = Path(out_dir) / "gpr_params.csv"
        _append_csv(cand_csv, bundle_cand_csv)
        _append_csv(gprp_csv, bundle_gprp_csv)

        # ---- NPZ を集約 ----
        seq_npz = Path(out_dir) / "seq_log.npz"
        if seq_npz.exists():
            d = _npz_to_dict(seq_npz)
            tag = f"field{int(fid):03d}"
            for k, arr in d.items():
                bundle_npz_dict[f"{k}__{tag}"] = arr
            fields_done.append(int(fid))
        else:
            print(f"[warn] seq_log.npz not found for field={fid}")

    # === bundle npz を保存 ===
    # メタ情報として fields を入れておく
    bundle_npz_dict["fields"] = np.asarray(fields_done, dtype=np.int32)
    np.savez(bundle_dir / "seq_logs_bundle.npz", **bundle_npz_dict)
    print(f"[bundle] saved: {bundle_dir/'seq_logs_bundle.npz'}")

    print("[bundle] all done.")
    print("  dir :", bundle_dir)
    print("  csv :", bundle_cand_csv.name, ",", bundle_gprp_csv.name)
    print("  npz :", "seq_logs_bundle.npz")

if __name__ == "__main__":
    main()
