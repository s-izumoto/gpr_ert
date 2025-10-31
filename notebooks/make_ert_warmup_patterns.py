# -*- coding: utf-8 -*-
"""
16電極(0..15)から、A<B, M<N かつ {A,B}∩{M,N}=∅ の4電極パターンを作り、
(dAB, dMN, mAB, mMN) のピアソン相関オフ対角の絶対値合計を最小化するよう
35パターンを探索します。

使い方:
  python make_ert_warmup_patterns.py

出力:
  - stdout に35行のパターンと相関行列を表示
  - patterns.csv に (A,B,M,N,dAB,dMN,mAB,mMN) を保存
"""

import random
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_ELEC = 16          # 電極数 0..15
N_PAT  = 35          # 取得したいパターン数
MAX_ITERS = 60_000   # 反復回数（増やすともう少し最適化されます）
ANNEAL_T0 = 0.05     # 焼きなましの初期"温度"
ANNEAL_T1 = 0.005    # 焼きなましの最終"温度"

@dataclass(frozen=True)
class Pattern:
    A: int
    B: int
    M: int
    N: int

    def features(self) -> Tuple[float, float, float, float]:
        dAB = self.B - self.A
        dMN = self.N - self.M
        mAB = 0.5 * (self.A + self.B)
        mMN = 0.5 * (self.M + self.N)
        return (float(dAB), float(dMN), float(mAB), float(mMN))

def valid_pattern(A:int,B:int,M:int,N:int) -> bool:
    if not (0 <= A < B < N_ELEC): return False
    if not (0 <= M < N < N_ELEC): return False
    # 4電極がすべて異なる
    if len({A,B,M,N}) != 4: return False
    return True

def random_pattern() -> Pattern:
    # 4つの異なる電極を選ぶ → 2つずつのペアに分ける → どちらをAB/MNにするか決める
    es = random.sample(range(N_ELEC), 4)
    es.sort()
    # 4集合の分割は3通り。等確率に
    partitions = [
        ((es[0], es[1]), (es[2], es[3])),
        ((es[0], es[2]), (es[1], es[3])),
        ((es[0], es[3]), (es[1], es[2]))
    ]
    (p1, p2) = random.choice(partitions)
    # AB/MN をランダムに割り振り、各ペアは小さい方を左に
    if random.random() < 0.5:
        A, B = min(p1), max(p1)
        M, N = min(p2), max(p2)
    else:
        A, B = min(p2), max(p2)
        M, N = min(p1), max(p1)
    assert valid_pattern(A,B,M,N)
    return Pattern(A,B,M,N)

def compute_corr_objective(patterns: List[Pattern]) -> Tuple[float, np.ndarray]:
    """ 相関行列を計算し、オフ対角成分の絶対値合計を返す """
    X = np.array([p.features() for p in patterns], dtype=float)  # [K,4]
    # 特徴を標準化（相関は標準化不要だが、安定のため一応）
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
    C = np.corrcoef(X, rowvar=False)  # [4,4]
    off_diag_sum = np.sum(np.abs(C - np.eye(4)))
    return float(off_diag_sum), C

def neighbor_replace(patterns: List[Pattern], pool: List[Pattern]) -> List[Pattern]:
    """ ランダムに1本を別候補に差し替え（重複は避ける） """
    new_list = patterns.copy()
    idx = random.randrange(len(new_list))
    current_set = set(new_list)
    # 試行してユニークなものを入れる
    for _ in range(100):
        cand = random.choice(pool)
        if cand not in current_set:
            new_list[idx] = cand
            return new_list
    # どうしても見つからなければそのまま
    return new_list

def build_pool(max_pool: int = 6000) -> List[Pattern]:
    """ 候補プールを乱択生成（上限件数まで） """
    pool = set()
    trials = 0
    while len(pool) < max_pool and trials < max_pool * 100:
        p = random_pattern()
        pool.add(p)
        trials += 1
    return list(pool)

def simulated_anneal(initial: List[Pattern], pool: List[Pattern]) -> Tuple[List[Pattern], float, np.ndarray]:
    best = initial
    best_score, bestC = compute_corr_objective(best)
    curr = best
    curr_score, currC = best_score, bestC

    for it in range(1, MAX_ITERS + 1):
        # 温度スケジュール（線形減衰）
        t = ANNEAL_T0 + (ANNEAL_T1 - ANNEAL_T0) * (it / MAX_ITERS)
        cand = neighbor_replace(curr, pool)
        cand_score, candC = compute_corr_objective(cand)
        delta = cand_score - curr_score
        if delta < 0 or math.exp(-delta / max(t, 1e-9)) > random.random():
            curr, curr_score, currC = cand, cand_score, candC
            if curr_score < best_score:
                best, best_score, bestC = curr, curr_score, currC
        # 進捗の軽いログ
        if it % 5000 == 0:
            print(f"[iter {it:6d}] curr={curr_score:.6f}  best={best_score:.6f}")
    return best, best_score, bestC

def main():
    # 候補プール
    pool = build_pool(max_pool=6000)

    # 初期解：重複なしでランダムにN_PAT本
    initial = random.sample(pool, N_PAT)

    # 最適化
    best, best_score, C = simulated_anneal(initial, pool)

    # 出力
    print("\n=== Best 35 patterns (A,B,M,N,dAB,dMN,mAB,mMN) ===")
    rows = []
    for i, p in enumerate(best, 1):
        dAB, dMN, mAB, mMN = p.features()
        rows.append((p.A, p.B, p.M, p.N, dAB, dMN, mAB, mMN))
        print(f"{i:2d}: {p.A:2d} {p.B:2d} {p.M:2d} {p.N:2d} | "
              f"dAB={dAB:2.0f} dMN={dMN:2.0f} mAB={mAB:4.1f} mMN={mMN:4.1f}")

    print("\n=== Correlation matrix (dAB, dMN, mAB, mMN) ===")
    names = ["dAB", "dMN", "mAB", "mMN"]
    header = "      " + " ".join([f"{n:>8s}" for n in names])
    print(header)
    for i in range(4):
        line = f"{names[i]:>5s} " + " ".join([f"{C[i,j]:8.4f}" for j in range(4)])
        print(line)
    print(f"\nObjective (sum of abs off-diagonal): {best_score:.6f}")

    # CSV保存
    with open("patterns.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["A","B","M","N","dAB","dMN","mAB","mMN"])
        w.writerows(rows)
    print("\nSaved to patterns.csv")

if __name__ == "__main__":
    main()
