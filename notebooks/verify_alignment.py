# verify_alignment.py
import json
import numpy as np
from pathlib import Path

ERT = Path("../data/interim/surrogate_ds/ert_surrogate.npz")
ORC = Path("../data/processed/oracle_pairs/latent_oracle_pairs_aligned.npz")

def load_meta(ds):
    meta = {}
    if "meta" in ds.files:
        try:
            meta = json.loads(ds["meta"][0])
        except Exception:
            pass
    return meta

def main():
    assert ERT.exists(), f"Missing: {ERT}"
    assert ORC.exists(), f"Missing: {ORC}"

    ds = np.load(ERT, allow_pickle=True)
    oc = np.load(ORC, allow_pickle=True)

    # 1) R を計算
    meta = load_meta(ds)
    if not meta or "n_AB" not in meta or "n_MN_per_AB" not in meta:
        raise RuntimeError("ERT meta に n_AB / n_MN_per_AB がありません。生成時の設定を確認してください。")
    R = int(meta["n_AB"]) * int(meta["n_MN_per_AB"])

    # 2) rows→Nf
    Z_rows = np.asarray(ds["Z"], dtype=np.float32)  # (N_rows, k)
    if Z_rows.ndim != 2:
        raise RuntimeError(f"ds['Z'] must be 2D, got {Z_rows.shape}")
    rows, k = Z_rows.shape
    if rows % R != 0:
        raise RuntimeError(f"行数 {rows} が R={R} で割り切れません。")
    Nf = rows // R

    # 3) oracle 側のフィールド数と一致？
    if "M_oracle" not in oc.files:
        raise RuntimeError("oracle npz に 'M_oracle' がありません。")
    Nf_orc = oc["M_oracle"].shape[0]
    print(f"[info] R={R}, rows={rows}, Nf(ERT)={Nf}, Nf(oracle)={Nf_orc}")
    if Nf != Nf_orc:
        raise RuntimeError(f"フィールド数が不一致です: ERT={Nf} vs ORACLE={Nf_orc}")

    # 4) Z_fields の数値一致（順序整合の厳密チェック）
    if "Z_fields" in oc.files:
        Z_fields_from_oracle = np.asarray(oc["Z_fields"], dtype=np.float32)   # (Nf, k)
        Z_fields_from_ds = Z_rows.reshape(Nf, R, k)[:, 0, :]                  # (Nf, k) 各フィールド先頭行
        same = np.allclose(Z_fields_from_oracle, Z_fields_from_ds, rtol=1e-5, atol=1e-6)
        print(f"[check] Z_fields alignment: {'OK' if same else 'MISMATCH'}")
        if not same:
            # どのインデックスで不一致かも出す
            diffs = np.where(~np.isclose(Z_fields_from_oracle, Z_fields_from_ds, rtol=1e-5, atol=1e-6))
            bad_f = np.unique(diffs[0])[:10]
            print(f"[debug] mismatch field indices (first 10): {bad_f}")
            raise RuntimeError("Z_fields が一致しません。フィールド順がズレています。")
    else:
        print("[warn] oracle に 'Z_fields' がありません。厳密チェックはスキップします。")

    print("[OK] ERT と ORACLE のフィールド対応は整合しています。")

if __name__ == "__main__":
    main()
