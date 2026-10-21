"""
NSGA-II (Deb et al. 2002) on a real-coded decision vector in [0,1]^n.

Variation: simulated binary crossover (SBX, eta_c = 15) and polynomial mutation
(eta_m = 20).  An optional problem-specific mutation (the DSS-biased operator of
Definition 5.1) replaces the polynomial mutation with probability p_bias.
All objectives are minimised.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np


def nondominated_sort(F: np.ndarray) -> List[np.ndarray]:
    n = len(F)
    dom = np.all(F[:, None, :] <= F[None, :, :], axis=2) & np.any(F[:, None, :] < F[None, :, :], axis=2)
    n_dom = dom.sum(axis=0)                 # number of solutions dominating j
    fronts = []
    current = np.flatnonzero(n_dom == 0)
    assigned = np.zeros(n, dtype=bool)
    while len(current):
        fronts.append(current)
        assigned[current] = True
        n_dom = n_dom - dom[current].sum(axis=0)
        current = np.flatnonzero((n_dom == 0) & ~assigned)
    return fronts


def crowding(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        o = np.argsort(F[:, j])
        rng = F[o[-1], j] - F[o[0], j]
        d[o[0]] = d[o[-1]] = np.inf
        if rng > 0:
            d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / rng
    return d


def _rank_crowd(F: np.ndarray):
    rank = np.empty(len(F), dtype=int)
    cd = np.empty(len(F))
    for r, fr in enumerate(nondominated_sort(F)):
        rank[fr] = r
        cd[fr] = crowding(F[fr])
    return rank, cd


def sbx(p1, p2, rng, eta=15.0, p_cx=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > p_cx:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-12:
            continue
        u = rng.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        a, b = p1[i], p2[i]
        c1[i] = 0.5 * ((1 + beta) * a + (1 - beta) * b)
        c2[i] = 0.5 * ((1 - beta) * a + (1 + beta) * b)
    return np.clip(c1, 0, 1 - 1e-9), np.clip(c2, 0, 1 - 1e-9)


def poly_mutation(x, rng, p_gene, eta=20.0):
    y = x.copy()
    for i in range(len(x)):
        if rng.random() < p_gene:
            u = rng.random()
            if u < 0.5:
                dq = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                dq = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            y[i] = x[i] + dq
    return np.clip(y, 0, 1 - 1e-9)


def hypervolume3(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact hypervolume for 3 minimised objectives (slicing on the 3rd objective)."""
    P = F[np.all(F < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 2])]
    hv = 0.0
    for i in range(len(P)):
        z0 = P[i, 2]
        z1 = P[i + 1, 2] if i + 1 < len(P) else ref[2]
        if z1 <= z0:
            continue
        hv += _hv2(P[: i + 1, :2], ref[:2]) * (z1 - z0)
    return hv


def _hv2(P, ref):
    P = P[np.argsort(P[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


class NSGA2:
    def __init__(self, n_var: int, evaluate: Callable[[np.ndarray], np.ndarray], pop: int, gens: int,
                 rng: np.random.Generator, p_cx: float = 0.9, p_mut: float = 0.25,
                 biased: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
                 p_bias: float = 0.7, seeds: Optional[List[np.ndarray]] = None,
                 hv_ref: Optional[Callable[[np.ndarray], float]] = None):
        self.n_var, self.evaluate, self.pop, self.gens = n_var, evaluate, pop, gens
        self.rng, self.p_cx, self.p_mut = rng, p_cx, p_mut
        self.biased, self.p_bias = biased, p_bias
        self.seeds = seeds or []
        self.hv_fn = hv_ref
        self.history = []          # hypervolume per generation (if hv_fn given)
        self.n_eval = 0

    def _eval(self, X):
        self.n_eval += len(X)
        return np.array([self.evaluate(x) for x in X])

    def _mutate(self, x):
        if self.biased is not None and self.rng.random() < self.p_bias:
            return self.biased(x, self.rng)
        return poly_mutation(x, self.rng, max(1.0 / self.n_var, self.p_mut / 2))

    def run(self, X0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng
        X = rng.random((self.pop, self.n_var))
        if X0 is not None:
            X[: len(X0)] = X0[: self.pop]
        for i, s in enumerate(self.seeds[: self.pop]):
            X[-(i + 1)] = s
        F = self._eval(X)
        rank, cd = _rank_crowd(F)
        if self.hv_fn:
            self.history.append(self.hv_fn(F[rank == 0]))
        for _ in range(self.gens):
            kids = []
            while len(kids) < self.pop:
                a, b = rng.integers(0, self.pop, 2), rng.integers(0, self.pop, 2)
                pa = a[0] if (rank[a[0]], -cd[a[0]]) < (rank[a[1]], -cd[a[1]]) else a[1]
                pb = b[0] if (rank[b[0]], -cd[b[0]]) < (rank[b[1]], -cd[b[1]]) else b[1]
                c1, c2 = sbx(X[pa], X[pb], rng, p_cx=self.p_cx)
                if rng.random() < self.p_mut * 2:
                    c1 = self._mutate(c1)
                if rng.random() < self.p_mut * 2:
                    c2 = self._mutate(c2)
                kids += [c1, c2]
            K = np.array(kids[: self.pop])
            FK = self._eval(K)
            XA, FA = np.vstack([X, K]), np.vstack([F, FK])
            keep = []
            for fr in nondominated_sort(FA):
                if len(keep) + len(fr) <= self.pop:
                    keep.extend(fr.tolist())
                else:
                    c = crowding(FA[fr])
                    keep.extend(fr[np.argsort(-c)][: self.pop - len(keep)].tolist())
                    break
            X, F = XA[keep], FA[keep]
            rank, cd = _rank_crowd(F)
            if self.hv_fn:
                self.history.append(self.hv_fn(F[rank == 0]))
        return X, F


if __name__ == "__main__":
    # Short demonstration on a two-objective test problem (Schaffer N.1 on [0, 1]).
    def _f(x):
        z = 4 * x[0] - 2
        return np.array([z ** 2, (z - 2) ** 2])
    opt = NSGA2(1, _f, pop=20, gens=30, rng=np.random.default_rng(0))
    X, F = opt.run()
    print(f"dsswaeop.nsga2: {opt.n_eval} evaluations; first front has {len(nondominated_sort(F)[0])} points")
