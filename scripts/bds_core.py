#!/usr/bin/env python3
"""BDS test core: correlation-integral + bootstrap null distribution (verified)."""
import numpy as np
from scipy.spatial import cKDTree

def c_m(x, m, eps):
    x = np.asarray(x, float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    N = len(x) - m + 1
    emb = np.array([x[i:i + m] for i in range(N)])
    tree = cKDTree(emb)
    cnt = tree.count_neighbors(tree, eps, p=np.inf)
    return cnt / (N * N)

def Wm(x, m, eps):
    Cs = {k: c_m(x, k, eps) for k in range(1, 2 * m + 1)}
    return Cs[m] - Cs[1] ** m

def bds_bootstrap(x, m, eps, B=300, rng=None):
    """Return (W_m, p_value) under i.i.d. null via bootstrap resampling of x.
    p-value = fraction of bootstrap W_m with |W*| >= |W_obs|."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, float)
    W_obs = Wm(x, m, eps)
    N = len(x)
    Wstar = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, N, N)
        Wstar[b] = Wm(x[idx], m, eps)
    p = np.mean(np.abs(Wstar) >= np.abs(W_obs))
    return W_obs, p

if __name__ == "__main__":
    # sanity: i.i.d. -> high p (no rejection); chaos -> low p (reject)
    r_rw = np.random.default_rng(1).standard_normal(1500)
    r_chaos = []
    x = 0.4
    for _ in range(1500):
        x = 3.9 * x * (1 - x); r_chaos.append(x - 0.5)
    r_chaos = np.array(r_chaos)
    rng = np.random.default_rng(7)
    for m in [3, 4, 5]:
        W, p = bds_bootstrap(r_rw, m, 0.7, B=300, rng=rng)
        print(f"RW     m={m}: W={W:.5f} p={p:.3f}")
    for m in [3, 4, 5]:
        W, p = bds_bootstrap(r_chaos, m, 0.7, B=300, rng=rng)
        print(f"CHAOS  m={m}: W={W:.5f} p={p:.3f}")
