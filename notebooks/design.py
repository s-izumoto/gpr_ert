# vsoed/design/dspace.py
from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
import torch

Pair = Tuple[int, int, int, int]  # (A,B,M,N)

# ----------------------- policy-output → embed -----------------------

@torch.no_grad()  # remove this if you need grads through the mapping
def map_uv_to_embed(u: torch.Tensor, n_elecs: int, min_gap: int = 1) -> torch.Tensor:
    """
    Map policy samples u in (0,1)^4 to the continuous 4D embed:
      [dAB, dMN, mAB, mMN], where for each dipole:
        d ∈ [d_min, 1], d_min = min_gap / (n_elecs-1)
        m ∈ [d/2, 1 - d/2]   (midpoint must lie between electrodes)
    This produces *feasible* continuous points that are then snapped to a
    discrete pair via nearest_k in embed space.

    Args:
      u: (..., 4) tensor with components in (0,1)
      n_elecs: total electrodes
      min_gap: minimal discrete separation (A<B, B-A >= min_gap)

    Returns:
      e: (..., 4) tensor [dAB, dMN, mAB, mMN] on same device/dtype as u
    """
    assert u.shape[-1] == 4, "u must be (..., 4)"
    L = float(max(1, n_elecs - 1))
    d_min = min_gap / L

    u = u.clamp(1e-6, 1 - 1e-6)
    u_dAB, u_dMN, u_mAB, u_mMN = torch.unbind(u, dim=-1)

    # distances in [d_min, 1]
    dAB = d_min + (1.0 - d_min) * u_dAB
    dMN = d_min + (1.0 - d_min) * u_dMN

    # midpoints in [d/2, 1 - d/2]
    mAB = (1.0 - dAB) * u_mAB + 0.5 * dAB
    mMN = (1.0 - dMN) * u_mMN + 0.5 * dMN

    e = torch.stack([dAB, dMN, mAB, mMN], dim=-1)
    return e

# ---------------------------- design space ----------------------------

class ERTDesignSpace:
    """
    Concrete ERT design space:
      - Enumerates all (A,B,M,N) with B-A >= min_gap and N-M >= min_gap
      - Optionally disallows overlap (default)
      - Collapses reciprocity: keep one of (A,B,M,N) and (M,N,A,B)
      - Precomputes normalized embeds: [dAB, dMN, mAB, mMN]
      - Provides nearest_k / nearest_1 in *transformed* embed space (distance scaling)
      - NEW: can be constructed from a pre-filtered list of allowed_pairs
    """

    def __init__(self, n_elecs: int, min_gap: int = 1, allow_overlap: bool = False,
                 metric: str = "identity", allowed_pairs: list | None = None):

        self.n_elecs = int(n_elecs)
        self.min_gap = int(min_gap)
        self.metric  = str(metric).lower()

        # --- enumerate valid discrete pairs or use provided allowed_pairs ---
        if allowed_pairs is None:
            pairs: List[Pair] = []
            for A in range(n_elecs):
                for B in range(A + min_gap, n_elecs):
                    for M in range(n_elecs):
                        for N in range(M + min_gap, n_elecs):
                            if not allow_overlap and len({A, B, M, N}) < 4:
                                continue
                            pairs.append((A, B, M, N))
        else:
            # Assume incoming allowed pairs are 1-based; convert to 0-based internal
            pairs = []
            for (A,B,M,N) in allowed_pairs:
                A0,B0,M0,N0 = A-1, B-1, M-1, N-1
                if not allow_overlap and len({A0,B0,M0,N0}) < 4:
                    continue
                if not (0 <= A0 < n_elecs and 0 <= B0 < n_elecs and 0 <= M0 < n_elecs and 0 <= N0 < n_elecs):
                    continue
                if B0 - A0 < min_gap or N0 - M0 < min_gap:
                    continue
                pairs.append((A0,B0,M0,N0))

        # --- collapse reciprocity globally: keep one canonical orientation ---
        pairs_canon: List[Pair] = []
        seen = set()
        for (A, B, M, N) in pairs:
            dip1 = (A, B); dip2 = (M, N)
            if dip2 < dip1:
                dip1, dip2 = dip2, dip1
            key = (dip1[0], dip1[1], dip2[0], dip2[1])
            if key in seen:
                continue
            seen.add(key)
            pairs_canon.append(key)

        pairs_canon.sort()
        self.pairs: Sequence[Pair] = pairs_canon
        self.embed_dim = 4

        # --- base embeds [P,4] ---
        self.embeds = np.stack([self._pair_to_embed(p) for p in self.pairs], axis=0).astype(np.float32)

        # --- prepare metric transform & transformed embeds ---
        self._prepare_metric_transform()
        self._rebuild_kdt()


    def _pair_to_embed(self, p: Pair) -> np.ndarray:
        """(A,B,M,N) → normalized [dAB, dMN, mAB, mMN] in [0,1]."""
        A, B, M, N = p
        L = float(self.n_elecs - 1)
        dAB = (B - A) / L
        dMN = (N - M) / L
        mAB = (A + B) / (2.0 * L)
        mMN = (M + N) / (2.0 * L)
        return np.array([dAB, dMN, mAB, mMN], dtype=np.float32)

    def dnorm_from_pair(self, p: Pair) -> np.ndarray:
        """Public alias used elsewhere in your code."""
        return self._pair_to_embed(p)

    # ---- k-NN API used by train/design_select.py ----

    def nearest_k(self, e, k: int) -> np.ndarray:
        """
        Return indices of the k nearest discrete designs to the given 4D embed.
        e: array-like shape (4,) or (1,4)
        """
        e_t = self._transform_query(e)  # ★ 変換
        k = min(int(k), len(self.pairs))
        if self._kdt is None or self._use == "brute":
            diffs = self.t_embeds - e_t  # ★ 変換後の埋め込みで距離
            d2 = (diffs * diffs).sum(axis=1)
            return np.argsort(d2)[:k]
        if self._use == "scipy":
            _, idx = self._kdt.query(e_t, k=k)
            return np.asarray(idx).reshape(-1)
        else:  # sklearn KDTree
            idx = self._kdt.query(e_t, k=k, return_distance=False)
            return np.asarray(idx).reshape(-1)


    def nearest_1(self, e) -> int:
        idxs = self.nearest_k(e, 1)
        return int(np.asarray(idxs).reshape(()))

    # (optional) convenience
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _prepare_metric_transform(self):
        """Set up transform for the chosen metric and produce transformed embeds."""
        E = self.embeds.copy()  # [P,4]
        eps = 1e-8

        def _perdim(E):
            mu = E.mean(axis=0, keepdims=True)
            sd = E.std(axis=0, keepdims=True)
            sd = np.where(sd < eps, 1.0, sd)
            return (E - mu) / sd, (mu, sd)

        def _whiten(E):
            mu = E.mean(axis=0, keepdims=True)
            X = E - mu
            # Cov = U S U^T ⇒ Cov^{-1/2} = U S^{-1/2} U^T
            Cov = (X.T @ X) / max(1, X.shape[0] - 1)
            U, S, Vt = np.linalg.svd(Cov + eps * np.eye(Cov.shape[0], dtype=Cov.dtype), full_matrices=False)
            S_inv_sqrt = np.diag(1.0 / np.sqrt(S + eps)).astype(E.dtype)
            W = (U @ S_inv_sqrt @ U.T).astype(E.dtype)  # [4,4]
            return (X @ W), (mu, W)

        def _cdf(E):
            # per-dim ECDF to [0,1]
            Z = np.empty_like(E)
            P = E.shape[0]
            for j in range(E.shape[1]):
                x = E[:, j]
                order = np.argsort(x)
                ranks = np.empty_like(order)
                ranks[order] = np.arange(P)
                # rank in (0,1): (r+0.5)/P → avoid exact 0/1
                z = (ranks + 0.5) / float(P)
                Z[:, j] = z.astype(E.dtype)
            return Z, None

        if self.metric == "identity":
            self._metric_info = None
            self.t_embeds = E
        elif self.metric == "perdim":
            Z, info = _perdim(E)
            self._metric_info = ("perdim", info)
            self.t_embeds = Z.astype(np.float32)
        elif self.metric == "whiten":
            Z, info = _whiten(E)
            self._metric_info = ("whiten", info)
            self.t_embeds = Z.astype(np.float32)
        elif self.metric == "cdf":
            Z, _ = _cdf(E)
            self._metric_info = ("cdf", None)
            self.t_embeds = Z.astype(np.float32)
        elif self.metric in ("cdf+whiten", "whiten+cdf"):
            Z, _ = _cdf(E)
            Z2, info = _whiten(Z)
            self._metric_info = ("cdf+whiten", info)
            self.t_embeds = Z2.astype(np.float32)
        else:
            raise ValueError(f"unknown metric: {self.metric}")

    def _transform_query(self, e):
        """Apply the same transform to a single query embed e[4,]."""
        e = np.asarray(e, np.float32).reshape(1, -1)
        kind = "identity" if self._metric_info is None else self._metric_info[0]
        eps = 1e-8
        if kind == "identity":
            return e
        if kind == "perdim":
            mu, sd = self._metric_info[1]
            sd = np.where(sd < eps, 1.0, sd)
            return (e - mu) / sd
        if kind == "whiten":
            mu, W = self._metric_info[1]
            return (e - mu) @ W
        if kind == "cdf":
            # query単体でのECDFは本来定義が難しいので、近似として
            # 各次元で線形補間：x を E の分位にマッピング（最近傍2点の rank 比）
            # 安価な近似として、学習時の分布が概ね単調なら次で十分
            E = self.embeds  # original
            Z = np.empty_like(e)
            for j in range(E.shape[1]):
                xj = e[0, j]
                col = E[:, j]
                order = np.argsort(col)
                col_sorted = col[order]
                # 挿入位置
                k = np.searchsorted(col_sorted, xj, side="left")
                P = col_sorted.shape[0]
                if k <= 0:
                    zj = (0.5) / P
                elif k >= P:
                    zj = (P - 0.5) / P
                else:
                    x0, x1 = col_sorted[k - 1], col_sorted[k]
                    t = 0.0 if abs(x1 - x0) < eps else (xj - x0) / (x1 - x0)
                    r0 = (k - 0.5) / P
                    r1 = (k + 0.5) / P
                    zj = (1 - t) * r0 + t * r1
                Z[0, j] = zj
            return Z.astype(np.float32)
        if kind == "cdf+whiten":
            # まず CDF 近似を「この場で」明示的に適用（再帰しない）
            E = self.embeds  # 元の埋め込み [P,4]
            Zq = np.empty_like(e)  # e は [1,4]
            P = E.shape[0]
            eps = 1e-8

            for j in range(E.shape[1]):
                xj = float(e[0, j])
                col = E[:, j]
                order = np.argsort(col)
                col_sorted = col[order]

                k = np.searchsorted(col_sorted, xj, side="left")
                if k <= 0:
                    zj = (0.5) / P
                elif k >= P:
                    zj = (P - 0.5) / P
                else:
                    x0, x1 = col_sorted[k - 1], col_sorted[k]
                    t = 0.0 if abs(x1 - x0) < eps else (xj - x0) / (x1 - x0)
                    r0 = (k - 0.5) / P
                    r1 = (k + 0.5) / P
                    zj = (1 - t) * r0 + t * r1
                Zq[0, j] = zj

            # 次に、CDF 後の空間で Whiten（_prepare_metric_transform で学習済みの μ と W を使用）
            mu, W = self._metric_info[1]  # ("cdf+whiten", (mu, W))
            return (Zq - mu) @ W


    def _rebuild_kdt(self):
        """Rebuild KD-tree on transformed embeds."""
        self._kdt = None
        self._use = "brute"
        try:
            from scipy.spatial import cKDTree  # type: ignore
            self._kdt = cKDTree(self.t_embeds)
            self._use = "scipy"
        except Exception:
            try:
                from sklearn.neighbors import KDTree  # type: ignore
                self._kdt = KDTree(self.t_embeds)
                self._use = "sklearn"
            except Exception:
                self._kdt = None
                self._use = "brute"

