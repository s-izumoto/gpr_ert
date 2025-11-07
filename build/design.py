"""
ERT design space utilities
==========================

This module provides a concrete design space for 2‑D/line ERT (Electrical
Resistivity Tomography) electrode configurations and a light‑weight embedding
+ nearest‑neighbour search API that is convenient for active design / BO loops.

Core ideas
----------
1) **Discrete designs** are 4‑tuples ``(A, B, M, N)`` of 0‑based electrode
   indices with constraints ``B - A >= min_gap`` and ``N - M >= min_gap``.
   Overlap between current and potential dipoles can be disabled (default).

2) **Reciprocity collapsing** keeps only one of ``(A,B,M,N)`` and
   ``(M,N,A,B)`` by sorting the two dipoles so that the tuple with the smaller
   dipole comes first. This reduces duplicates globally.

3) **Normalized embeds** map each discrete design to a continuous 4‑vector
   ``[dAB, dMN, mAB, mMN]`` in ``[0,1]^4`` where distances are normalized by
   ``(n_elecs-1)`` and midpoints are in the same normalized coordinate.

4) **Metric transforms** optionally rescale / decorrelate the embed space prior
   to distance computations (``identity``, ``perdim`` z‑score, ``whiten`` via
   covariance inverse square‑root, empirical ``cdf``, or ``cdf+whiten``). The
   same transform is applied to queries to keep distances consistent.

5) **Nearest‑k lookup** is provided either by SciPy's ``cKDTree``, scikit‑learn
   ``KDTree``, or a simple brute‑force fallback, all over the *transformed*
   embed space.

6) **Policy helper** ``map_uv_to_embed`` converts policy samples ``u ∈ (0,1)^4``
   to feasible continuous embeds by enforcing the geometric constraints
   ``d ∈ [d_min, 1]`` and ``m ∈ [d/2, 1-d/2]`` per dipole, where
   ``d_min = min_gap / (n_elecs-1)``. These continuous points can then be
   snapped to a discrete design via ``nearest_1``/``nearest_k``.

Notes
-----
* If ``allowed_pairs`` is provided, it is assumed to be **1‑based** indices and
  will be converted to 0‑based internally. Invalid pairs w.r.t bounds, gaps or
  overlap are filtered out.
* All arrays returned by the class are ``np.float32`` where applicable.
* This file contains **no** ERT physics — only the geometry/embedding utilities.

Example
-------
>>> space = ERTDesignSpace(n_elecs=32, min_gap=1, allow_overlap=False, metric="cdf+whiten")
>>> len(space)  # number of canonical designs
...  
>>> u = torch.rand(4)  # (0,1)^4 sample from a policy
>>> e = map_uv_to_embed(u, n_elecs=32, min_gap=1)
>>> idx = space.nearest_1(e.numpy())
>>> discrete = space.pairs[idx]  # (A,B,M,N)

"""
from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
import torch

# Discrete design: (A, B) is the current dipole, (M, N) the potential dipole
Pair = Tuple[int, int, int, int]

# ---------------------------------------------------------------------------
# Policy output (0,1)^4  →  feasible continuous embed [dAB, dMN, mAB, mMN]
# ---------------------------------------------------------------------------

@torch.no_grad()  # Remove this decorator if gradients through the mapping are needed
def map_uv_to_embed(u: torch.Tensor, n_elecs: int, min_gap: int = 1) -> torch.Tensor:
    """Map ``u ∈ (0,1)^4`` to a feasible continuous embed ``[dAB, dMN, mAB, mMN]``.

    For each dipole (AB or MN):
      - distance ``d ∈ [d_min, 1]``, where ``d_min = min_gap / (n_elecs-1)``
      - midpoint ``m ∈ [d/2, 1 - d/2]`` so that the dipole fits within electrodes.

    The result can be snapped to a discrete design using ``ERTDesignSpace.nearest_k``.

    Args:
        u: Tensor of shape ``(..., 4)`` with values in ``(0,1)``.
        n_elecs: Total number of electrodes.
        min_gap: Minimal discrete separation (``A < B`` and ``B-A >= min_gap``).

    Returns:
        Tensor ``(..., 4)`` with the same dtype/device as ``u``.
    """
    assert u.shape[-1] == 4, "u must be (..., 4)"
    L = float(max(1, n_elecs - 1))
    d_min = min_gap / L

    # keep a safe open interval to avoid exact 0/1 artifacts
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

# ---------------------------------------------------------------------------
# Design space with canonical pair list + transformed embeds + kNN queries
# ---------------------------------------------------------------------------

class ERTDesignSpace:
    """Enumerate valid ERT designs and provide nearest‑neighbour queries.

    Features
    --------
    - Enumerates all ``(A,B,M,N)`` with gap constraints.
    - Optionally disallows overlap between electrodes used by AB and MN.
    - Collapses reciprocity globally.
    - Precomputes normalized embeds ``[dAB, dMN, mAB, mMN]``.
    - Applies an optional metric transform before distance computations.
    - Exposes ``nearest_k`` / ``nearest_1`` over the transformed embed space.
    - Can also be built from a pre‑filtered list of ``allowed_pairs``.
    """

    def __init__(
        self,
        n_elecs: int,
        min_gap: int = 1,
        allow_overlap: bool = False,
        metric: str = "identity",
        allowed_pairs: list | None = None,
        debug_whiten: bool = False,
        feature_names: list[str] | None = None
    ) -> None:
        self.n_elecs = int(n_elecs)
        self.min_gap = int(min_gap)
        self.metric = str(metric).lower()

        # --- enumerate valid discrete pairs or use provided allowed_pairs ---
        if allowed_pairs is None:
            pairs: List[Pair] = []
            for A in range(n_elecs):
                for B in range(A + min_gap, n_elecs):
                    for M in range(n_elecs):
                        for N in range(M + min_gap, n_elecs):
                            if not allow_overlap and len({A, B, M, N}) < 4:
                                # skip if any electrode is reused across AB and MN
                                continue
                            pairs.append((A, B, M, N))
        else:
            # Assume incoming allowed pairs are 1‑based; convert to 0‑based
            pairs = []
            for (A, B, M, N) in allowed_pairs:
                A0, B0, M0, N0 = A - 1, B - 1, M - 1, N - 1
                if not allow_overlap and len({A0, B0, M0, N0}) < 4:
                    continue
                if not (
                    0 <= A0 < n_elecs
                    and 0 <= B0 < n_elecs
                    and 0 <= M0 < n_elecs
                    and 0 <= N0 < n_elecs
                ):
                    continue
                if B0 - A0 < min_gap or N0 - M0 < min_gap:
                    continue
                pairs.append((A0, B0, M0, N0))

        # --- collapse reciprocity globally: keep one canonical orientation ---
        pairs_canon: List[Pair] = []
        seen = set()
        for (A, B, M, N) in pairs:
            dip1 = (A, B)
            dip2 = (M, N)
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
        self.embeds = (
            np.stack([self._pair_to_embed(p) for p in self.pairs], axis=0).astype(np.float32)
        )

        self._debug_whiten = bool(debug_whiten)
        self._feature_names = feature_names or ["dAB","dMN","mAB","mMN"]
        # --- prepare metric transform & transformed embeds ---
        self._prepare_metric_transform()
        self._rebuild_kdt()

    # ---------------------- basic conversions / aliases ---------------------

    def _pair_to_embed(self, p: Pair) -> np.ndarray:
        """Convert ``(A,B,M,N)`` → normalized ``[dAB, dMN, mAB, mMN]`` in ``[0,1]``."""
        A, B, M, N = p
        L = float(self.n_elecs - 1)
        dAB = (B - A) / L
        dMN = (N - M) / L
        mAB = (A + B) / (2.0 * L)
        mMN = (M + N) / (2.0 * L)
        return np.array([dAB, dMN, mAB, mMN], dtype=np.float32)

    def dnorm_from_pair(self, p: Pair) -> np.ndarray:
        """Public alias used elsewhere in codebases to get the normalized embed."""
        return self._pair_to_embed(p)

    # --------------------------- k‑NN search API ----------------------------

    def nearest_k(self, e, k: int) -> np.ndarray:
        """Return indices of the ``k`` nearest discrete designs to ``e``.

        Args:
            e: Query embed of shape ``(4,)`` or ``(1,4)`` (array‑like).
            k: Number of neighbours to return (capped by the number of designs).

        Returns:
            ``np.ndarray`` of shape ``(k,)`` with indices into ``self.pairs``.
        """
        e_t = self._transform_query(e)  # transformed query
        k = min(int(k), len(self.pairs))
        if self._kdt is None or self._use == "brute":
            diffs = self.t_embeds - e_t  # distances in transformed space
            d2 = (diffs * diffs).sum(axis=1)
            return np.argsort(d2)[:k]
        if self._use == "scipy":
            _, idx = self._kdt.query(e_t, k=k)
            return np.asarray(idx).reshape(-1)
        else:  # sklearn KDTree
            idx = self._kdt.query(e_t, k=k, return_distance=False)
            return np.asarray(idx).reshape(-1)

    def nearest_1(self, e) -> int:
        """Return the index of the single nearest design to ``e``."""
        idxs = self.nearest_k(e, 1)
        return int(np.asarray(idxs).reshape(()))

    def __len__(self) -> int:  # convenience
        return len(self.pairs)

    # ------------------------ metric / query transforms ---------------------

    def _prepare_metric_transform(self) -> None:
        """Set up the transform for the chosen metric and transform ``self.embeds``.

        Produces ``self.t_embeds`` (transformed embeds) and stores any parameters
        needed to transform individual queries later in ``self._metric_info``.
        """
        E = self.embeds.copy()  # [P,4]
        eps = 1e-8

        def _perdim(E_: np.ndarray):
            mu = E_.mean(axis=0, keepdims=True)
            sd = E_.std(axis=0, keepdims=True)
            sd = np.where(sd < eps, 1.0, sd)
            return (E_ - mu) / sd, (mu, sd)

        def _whiten(E_: np.ndarray):
            mu = E_.mean(axis=0, keepdims=True)
            X = E_ - mu
            # Cov = U S U^T ⇒ Cov^{-1/2} = U S^{-1/2} U^T
            Cov = (X.T @ X) / max(1, X.shape[0] - 1)
            U, S, _ = np.linalg.svd(
                Cov + eps * np.eye(Cov.shape[0], dtype=Cov.dtype), full_matrices=False
            )
            S_inv_sqrt = np.diag(1.0 / np.sqrt(S + eps)).astype(E_.dtype)
            W = (U @ S_inv_sqrt @ U.T).astype(E_.dtype)  # [4,4]
            return (X @ W), (mu, W)

        def _cdf(E_: np.ndarray):
            # Per‑dimension empirical CDF mapping to (0,1), avoiding exact 0/1.
            Z = np.empty_like(E_)
            P = E_.shape[0]
            for j in range(E_.shape[1]):
                x = E_[:, j]
                order = np.argsort(x)
                ranks = np.empty_like(order)
                ranks[order] = np.arange(P)
                # rank in (0,1): (r+0.5)/P to avoid hard 0/1
                z = (ranks + 0.5) / float(P)
                Z[:, j] = z.astype(E_.dtype)
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
        
        # ================= DEBUG PRINTS (whiten/perdim) =================
        if getattr(self, "_debug_whiten", False):
            kind = "identity" if self._metric_info is None else self._metric_info[0]
            names = getattr(self, "_feature_names", ["f0","f1","f2","f3"])

            print(f"\n[whiten/debug] metric = {kind}")
            if kind == "perdim":
                mu, sd = self._metric_info[1]
                mu = mu.reshape(-1); sd = sd.reshape(-1)
                print("[whiten/debug] per-dimension stats BEFORE standardization:")
                for n, m, s in zip(names, mu, sd):
                    print(f"  {n}: mu={m:.6g}, sigma={s:.6g}")
                print("\n[whiten/debug] Axis construction (z-score; no rotation):")
                for n, m, s in zip(names, mu, sd):
                    print(f"  {n}' = ({n} - {m:.6g}) / {s:.6g}")

            elif kind in ("whiten", "cdf+whiten"):
                mu, W = self._metric_info[1]
                mu = mu.reshape(-1)
                print("[whiten/debug] mean (mu) of the space where whitening is applied:")
                for n, m in zip(names, mu):
                    print(f"  {n}: {m:.6g}")

                # Whitening matrix (new = (x - mu) @ W)
                print("\n[whiten/debug] whitening matrix W (shape {}):".format(W.shape))
                with np.printoptions(precision=4, suppress=True):
                    print(W)

                # Show each new axis as a linear combination of original axes
                print("\n[whiten/debug] Axis mapping (new_k = Σ c_j · original_j):")
                for k in range(W.shape[1]):
                    coeffs = W[:, k]
                    terms = []
                    for n, c in zip(names, coeffs):
                        if abs(c) >= 1e-8:
                            terms.append(f"{c:+.4f}·{n}")
                    rhs = " ".join(terms) if terms else "0"
                    print(f"  new_axis{k+1}(x) = {rhs}")

                # Sanity check: covariance of whitened embeds ~ identity
                Te = self.t_embeds.astype(np.float64)
                C = (Te - Te.mean(0)).T @ (Te - Te.mean(0)) / max(1, Te.shape[0]-1)
                print("\n[whiten/debug] Covariance in whitened space (should be ~I):")
                with np.printoptions(precision=3, suppress=True):
                    print(C)
        # ================= END DEBUG PRINTS =============================


    def _transform_query(self, e):
        """Apply the stored metric transform to a single query ``e`` (shape ``[4]``).

        For ``cdf`` and ``cdf+whiten`` we approximate the query's empirical CDF
        position by inserting ``e`` into the sorted column values of
        ``self.embeds`` (no re‑fitting). This keeps queries and the stored
        distribution consistent without re‑estimating CDF parameters per query.
        """
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
            # Approximate the empirical CDF coordinate of e using the stored E.
            E = self.embeds  # original (untransformed)
            Z = np.empty_like(e)
            for j in range(E.shape[1]):
                xj = e[0, j]
                col = E[:, j]
                order = np.argsort(col)
                col_sorted = col[order]
                # insertion position
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
            # First map to CDF coordinates as above, then apply the learned W.
            E = self.embeds  # original embeds [P,4]
            Zq = np.empty_like(e)  # e is [1,4]
            P = E.shape[0]

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

            # Apply the whitening matrix learned during _prepare_metric_transform
            mu, W = self._metric_info[1]  # ("cdf+whiten", (mu, W))
            return (Zq - mu) @ W

        # If we reached here, the metric kind is unknown (defensive programming)
        raise RuntimeError(f"unhandled metric kind: {kind}")

    # ------------------------------- k‑d tree -------------------------------

    def _rebuild_kdt(self) -> None:
        """Rebuild the KD‑tree on ``self.t_embeds``.

        Uses SciPy's ``cKDTree`` if available, otherwise scikit‑learn's
        ``KDTree``. Falls back to a brute‑force numpy path if neither is
        available. The chosen backend is recorded in ``self._use``.
        """
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
