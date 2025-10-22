
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filter surrogate ERT NPZs using a reduced selection, **safely via Z matching**.
#
# Key features:
# - Default: still supports indices/mask, but if Z is available it matches by Z hashes.
# - --force_z: ignore indices/mask and require (or reconstruct) Z, then match by Z only.
# - --source-z: NPZ/NPY providing a full Z array to reconstruct reduced Z from indices when reduced has no Z.
#
# Examples:
#   # Strict Z matching using reduced Z directly
#   python 05_filter_surrogate_by_reduction.py     #     --force_z     #     --reduced path/to/reduced_with_Z.npz     #     --surrogate path/to/ert_surrogate.npz     #     --out path/to/ert_surrogate_reduced.npz
#
#   # Strict Z matching when reduced holds indices only; reconstruct Z from a source Z
#   python 05_filter_surrogate_by_reduction.py     #     --force_z     #     --reduced path/to/reduced_indices.npy     #     --source-z path/to/oracle_full.npz     #     --surrogate path/to/ert_surrogate.npz
#
import argparse, hashlib
from pathlib import Path
import numpy as np
import json
from pathlib import Path
import os

# ---------------- helpers ----------------

def _rows_per_field_from_meta(npz):
    """SUR 側 NPZの meta から R を推定 (n_AB × n_MN_per_AB)。失敗時は None。"""
    if "meta" in npz.files:
        try:
            meta = json.loads(npz["meta"][0])
            n_AB = int(meta.get("n_AB", 0))
            n_MN = int(meta.get("n_MN_per_AB", 0))
            if n_AB > 0 and n_MN > 0:
                return n_AB * n_MN
        except Exception:
            pass
    return None

def _extract_Z_fields_any(M, rows_per_field=None):
    """
    入力が dict/NPZ/ndarray のいずれであっても、(Nf, k) の Z_fields を返す。
    - Z_fields があればそれを返す
    - Z(行ベース) しか無ければ R=rows_per_field で (Nf,R,k)→[:,0,:] に変換
    - ndarray で2次元なら既に (Nf,k) とみなす
    """
    # ndarray 直渡し: 2D なら Z_fields とみなす
    if isinstance(M, np.ndarray):
        if M.ndim == 2:
            return np.asarray(M)
        raise RuntimeError(f"ndarray reduced/surrogate must be 2D for Z_fields, got shape={M.shape}")

    # NPZ/dict 風アクセス
    get = (lambda k: M.get(k)) if isinstance(M, dict) else (lambda k: (M[k] if k in M else None))

    Zf = get("Z_fields")
    if Zf is not None:
        Zf = np.asarray(Zf)
        if Zf.ndim != 2:
            raise RuntimeError(f"Z_fields must be 2D, got shape={Zf.shape}")
        return Zf

    # Z_fields が無ければ Z(行ベース) から復元
    Z = get("Z")
    if Z is None:
        return None  # 本当に何も無い

    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise RuntimeError(f"Z must be 2D, got shape={Z.shape}")
    if rows_per_field is None:
        # rows_per_field は呼び出し側で SUR の meta から与えるのが安全
        raise RuntimeError("rows_per_field is required to derive Z_fields from row-wise Z.")
    if Z.shape[0] % rows_per_field != 0:
        raise RuntimeError(f"Z rows {Z.shape[0]} not divisible by rows_per_field={rows_per_field}")
    Nf = Z.shape[0] // rows_per_field
    return Z.reshape(Nf, rows_per_field, -1)[:, 0, :]

def _row_hashes(X):
    X = np.ascontiguousarray(X)
    flat = X.reshape(X.shape[0], -1)
    hashes = np.empty((flat.shape[0],), dtype=object)
    for i, row in enumerate(flat):
        m = hashlib.sha1(row.view(np.uint8)).hexdigest()
        hashes[i] = m
    return hashes

def _load_npz_or_npy(path: Path):
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=True)
    return np.load(path, allow_pickle=True)  # NPZ mapping-like

def _extract_Z_from_mapping(M, prefer_keys=("Z", "Z_fields")):
    for k in prefer_keys:
        if isinstance(M, dict):
            if k in M:
                Z = np.asarray(M[k])
                if Z.ndim >= 1:
                    return Z
        else:
            # NPZ-like
            if k in M:
                Z = np.asarray(M[k])
                if Z.ndim >= 1:
                    return Z
    return None

def _extract_indices_or_mask(M):
    # Return dict with 'idx' or 'mask' if present; else None
    prefer_idx_keys = ["keep_idx", "kept_indices", "Z_idx_keep", "Z_idx_keeped", "idx_keep", "Z_keep_idx", "indices_keep", "Z_idx"]
    mask_keys = ["keep_mask", "mask_keep", "kept_mask"]
    if isinstance(M, dict):
        get = lambda k: M.get(k, None)
    else:
        get = lambda k: (M[k] if k in M else None)
    for k in prefer_idx_keys:
        a = get(k)
        if a is not None:
            a = np.asarray(a)
            if a.ndim == 1 and np.issubdtype(a.dtype, np.integer):
                return {"idx": a.astype(np.int64)}
    for k in mask_keys:
        a = get(k)
        if a is not None:
            a = np.asarray(a)
            if a.ndim == 1 and a.dtype == np.bool_:
                return {"mask": a}
    return None

def _infer_sample_length(npz):
    dims = []
    if "Z" in npz:
        Z = np.asarray(npz["Z"])
        if Z.ndim >= 1 and Z.shape[0] > 1:
            dims.append(Z.shape[0])
    for k in ("field_ids", "field_idx", "Z_idx"):
        if k in npz:
            a = np.asarray(npz[k])
            if a.ndim >= 1 and a.shape[0] > 1:
                dims.append(a.shape[0])
    for k in npz.files:
        a = np.asarray(npz[k])
        if a.ndim >= 1 and a.shape[0] > 1:
            dims.append(a.shape[0])
    if not dims:
        # fallback include 1s
        for k in npz.files:
            a = np.asarray(npz[k])
            if a.ndim >= 1:
                dims.append(a.shape[0])
    if not dims:
        raise RuntimeError("Cannot infer N for surrogate.")
    return int(max(dims))

def _candidate_sample_keys(npz, N):
    return [k for k in npz.files if (np.asarray(npz[k]).ndim >= 1 and np.asarray(npz[k]).shape[0] == N)]

# -------------- core matching --------------
def _keep_idx_by_Z(Z_red, Z_all, verbose=False):
    if Z_red.ndim == 1:
        Z_red = Z_red.reshape(1, -1)
    if Z_all.ndim == 1:
        Z_all = Z_all.reshape(1, -1)
    h_red = _row_hashes(Z_red)
    h_all = _row_hashes(Z_all)
    pos = {h: i for i, h in enumerate(h_all)}
    keep = []
    missing = 0
    for h in h_red:
        if h in pos:
            keep.append(pos[h])
        else:
            missing += 1
    if missing:
        raise RuntimeError(f"Z-hash matching failed for {missing} rows (check Z alignment/precision).")
    keep = np.array(keep, dtype=np.int64)
    if verbose:
        print(f"[match:Z] matched {len(keep)} rows via Z hashes")
    return keep

def _match_reduced_to_surrogate(reduced_any, surrogate_npz, force_z=False, source_Z_any=None, verbose=False, rows_per_field=None):
    """
    常に Z_fields ベースでマッチする:
      - reduced に Z_fields があればそれを使う
      - なければ indices/mask + source Z から Z を復元し、それを Z_fields とみなす（2D 必須）
      - surrogate 側は Z_fields が無ければ rows_per_field を使って Z→Z_fields に変換
    """
    # 1) reduced 側の Z_fields を取得
    Zf_red = _extract_Z_fields_any(reduced_any, rows_per_field=rows_per_field)

    # force_z で Z_fields がまだ得られない場合は indices/mask + source Z から復元
    if force_z and Zf_red is None:
        iom = _extract_indices_or_mask(reduced_any)
        if iom is None:
            raise RuntimeError("force_z requested but reduced has neither Z_fields nor indices/mask.")
        if source_Z_any is None:
            raise RuntimeError("force_z requested and reduced has only indices/mask; please provide --source-z with a Z array.")
        # source から Z を引く
        Z_source = _extract_Z_from_mapping(source_Z_any, prefer_keys=("Z_fields","Z"))
        if Z_source is None:
            if isinstance(source_Z_any, np.ndarray) and source_Z_any.ndim == 2:
                Z_source = source_Z_any
            else:
                raise RuntimeError("Could not extract Z or Z_fields from --source-z.")
        if "idx" in iom:
            idx = iom["idx"]
            if np.any((idx < 0) | (idx >= Z_source.shape[0])):
                raise RuntimeError("Indices out of bounds for source Z/Z_fields.")
            Z_pick = Z_source[idx]
        else:
            mask = iom["mask"].astype(bool)
            if mask.size != Z_source.shape[0]:
                raise RuntimeError("Mask length mismatches source Z/Z_fields.")
            Z_pick = Z_source[mask]
        # ここで Z_pick は 2D を仮定（Z_fields か Z のいずれかのサブセット）
        if Z_pick.ndim != 2:
            raise RuntimeError(f"Reconstructed reduced array must be 2D, got shape={Z_pick.shape}")
        Zf_red = Z_pick  # すでに (Nf,k) 仕様として扱う

    if Zf_red is None:
        # 非 force_z で reduced が indices/mask のみ → 安全のため Z_fields マッチを要求
        raise RuntimeError("Reduced file provides neither Z_fields nor Z (nor indices/mask with --source-z).")

    # 2) surrogate 側も Z_fields に正規化
    #    - SUR に Z_fields があればそれ
    #    - なければ rows_per_field/meta から R を得て Z→Z_fields
    try:
        Zf_all = _extract_Z_fields_any(surrogate_npz, rows_per_field=rows_per_field or _rows_per_field_from_meta(surrogate_npz))
    except RuntimeError as e:
        raise RuntimeError(f"Surrogate cannot provide Z_fields: {e}")

    # 3) Z_fields 同士でハッシュ一致
    return _keep_idx_by_Z(Zf_red, Zf_all, verbose=verbose)

# -------------- filtering --------------
def _filter_npz_to_out(in_path: Path, keep_fields_idx, out_path: Path, rows_per_field=None, verbose=False):
    import numpy as np, json
    with np.load(in_path, allow_pickle=True) as NPZ:
        # 1) R（rows_per_field）を決める
        R = rows_per_field
        if R is None:
            # meta から取れるなら取る
            if "meta" in NPZ.files:
                try:
                    meta = json.loads(NPZ["meta"][0])
                    if "rows_per_field" in meta:
                        R = int(meta["rows_per_field"])
                except Exception:
                    pass
        if R is None:
            raise RuntimeError("rows_per_field is required (not found in meta and not provided).")

        # 2) N_rows/N_fields を推定
        # 行ベース Z があれば最優先
        N_rows = None
        if "Z" in NPZ.files and NPZ["Z"].ndim >= 1:
            N_rows = int(NPZ["Z"].shape[0])
        else:
            # ほかの「明らかに行ベース」配列から拾う（ABMN、測定値など）
            for k in NPZ.files:
                a = NPZ[k]
                if isinstance(a, np.ndarray) and a.ndim >= 1:
                    # 行配列候補の最頻値を拾う
                    if a.shape[0] % R == 0 and a.shape[0] >= R:
                        N_rows = a.shape[0]
                        break

        if N_rows is None:
            raise RuntimeError("Could not infer N_rows; please ensure NPZ has row-wise arrays (e.g., 'Z').")

        if N_rows % R != 0:
            raise RuntimeError(f"Z rows {N_rows} not divisible by rows_per_field={R}")
        N_fields = N_rows // R

        # 3) keep_rows を展開（フィールド→行）
        keep_fields_idx = np.asarray(keep_fields_idx, dtype=np.int64)
        if np.any((keep_fields_idx < 0) | (keep_fields_idx >= N_fields)):
            raise RuntimeError(f"Field indices out of bounds: N_fields={N_fields}, min={keep_fields_idx.min()}, max={keep_fields_idx.max()}")
        keep_rows = np.concatenate([np.arange(f*R, (f+1)*R, dtype=np.int64) for f in keep_fields_idx])

        # 4) キーごとに、先頭軸が N_rows なら keep_rows、N_fields なら keep_fields を使い分け
        out_dict = {}
        for k in NPZ.files:
            arr = NPZ[k]
            if not (isinstance(arr, np.ndarray) and arr.ndim >= 1):
                out_dict[k] = arr
                continue

            if arr.shape[0] == N_rows:
                out_dict[k] = arr[keep_rows]
            elif arr.shape[0] == N_fields:
                out_dict[k] = arr[keep_fields_idx]
            else:
                out_dict[k] = arr

        # 5) meta を読み/初期化し、rows_per_field を保証
        meta = None
        if "meta" in NPZ.files:
            try:
                meta = json.loads(NPZ["meta"][0])
            except Exception:
                meta = None
        if meta is None:
            meta = {}
        if "rows_per_field" not in meta:
            meta["rows_per_field"] = int(R)

        # 5') n_fields を「保持されたフィールド数」に更新（ここで必ず上書き）
        kept_fields = int(len(keep_fields_idx))
        meta["n_fields"] = kept_fields
        meta["kept_fields"] = kept_fields  # 任意

        # 5.1) sidecar: field_source_idx（あれば持ってくる）
        # 入力NPZと同じフォルダの sidecar を絶対パスで指す
        sidecar = Path(in_path).with_name("field_source_idx.npy").resolve()

        fsrc_keep = None
        try:
            if sidecar.exists() and sidecar.is_file() and sidecar.suffix.lower() == ".npy" and sidecar.stat().st_size > 0:
                fsrc = np.load(os.fspath(sidecar), allow_pickle=True)
                fsrc = np.asarray(fsrc).reshape(-1)
                if fsrc.shape[0] == N_fields:
                    fsrc_keep = np.asarray(fsrc, dtype=np.int64)[keep_fields_idx]
                elif verbose:
                    print(f"[warn] sidecar length {fsrc.shape[0]} != N_fields {N_fields}; will synthesize.")
        except Exception as e:
            if verbose:
                print(f"[warn] sidecar load failed: {e}")
        # フォールバック：減らした元フィールド番号をそのままsource_idxにする
        if fsrc_keep is None:
            if verbose:
                print("[info] synthesizing field_source_idx from keep_fields_idx")
            fsrc_keep = np.asarray(keep_fields_idx, dtype=np.int64)

        # NPZ内にも保存
        out_dict["field_source_idx"] = fsrc_keep

        # ★ 最後に meta を入れて保存（← ここは except の外！）
        out_dict["meta"] = np.array([json.dumps(meta)], dtype=object)
        np.savez_compressed(out_path, **out_dict)
        if verbose:
            print(f"[save] wrote {out_path} (kept fields {kept_fields}/{N_fields}, rows {len(keep_rows)}/{N_rows})")

# -------------- main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduced", required=True, help="Reduced selection file (NPZ/NPY). Prefer Z inside.")
    ap.add_argument("--source-z", help="NPZ/NPY providing full Z to reconstruct reduced Z from indices/mask when --force_z.")
    ap.add_argument("--surrogate", required=True, action="append", help="Surrogate NPZ to filter (repeatable). Must contain Z for Z-matching.")
    ap.add_argument("--out", help="Single output NPZ path (only valid with single --surrogate).")
    ap.add_argument("--suffix", default="_reduced", help="Suffix for outputs when --out not given.")
    ap.add_argument("--force_z", action="store_true", help="Ignore indices/mask; require Z (or reconstruct it with --source-z) and match strictly by Z.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    ap.add_argument("--rows-per-field", type=int, default=None,
        help="SUR が Z しか持たない場合に Z→Z_fields へ変換するための行数 R。meta が読めない場合は指定。")

    args = ap.parse_args()

    red_path = Path(args.reduced)
    if not red_path.exists():
        raise SystemExit(f"Reduced file not found: {red_path}")
    RED = _load_npz_or_npy(red_path)

    SRCZ = None
    if args.source_z:
        srcz_path = Path(args.source_z)
        if not srcz_path.exists():
            raise SystemExit(f"--source-z not found: {srcz_path}")
        SRCZ = _load_npz_or_npy(srcz_path)

    for s in args.surrogate:
        s_path = Path(s)
        if not s_path.exists():
            raise SystemExit(f"Surrogate NPZ not found: {s_path}")
        with np.load(s_path, allow_pickle=True) as SUR:
            keep_idx = _match_reduced_to_surrogate(
                RED, SUR, force_z=args.force_z, source_Z_any=SRCZ,
                verbose=args.verbose, rows_per_field=args.rows_per_field
            )
            if args.out:
                if len(args.surrogate) > 1:
                    raise SystemExit("--out can only be used with a single --surrogate input.")
                out_path = Path(args.out)
            else:
                out_path = s_path.with_name(s_path.stem + args.suffix + s_path.suffix)
        _filter_npz_to_out(s_path, keep_idx, out_path, rows_per_field=args.rows_per_field, verbose=args.verbose)


if __name__ == "__main__":
    main()
