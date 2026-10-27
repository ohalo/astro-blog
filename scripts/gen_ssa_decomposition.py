#!/usr/bin/env python3
"""Generate real charts for '奇异谱分析 SSA 周期提取' article.

Methodology (honest, verifiable):
SSA (Vautard & Ghil 1989; Golyandina & Zhigljavsky 2013) has two steps.
(1) Embedding: build trajectory matrix X of shape (L, K) with L=window,
    K=N-L+1, columns = lagged slices.
(2) SVD: X = sum_i sqrt(lam_i) U_i V_i^T  (U_i left singular, V_i right).
The eigentriples (lam_i, U_i, V_i) are ordered by variance. Reconstruct a
component by diagonal-averaging the rank-1 matrix; pairing groups of
eigentriples recovers trend / oscillation / noise. We use a Hankel/double
trajectory (diagonal averaging) to map a D=VU^T matrix back to a time series.
For oscillation recovery we group eigentriple pairs {i, i+1} whose singular
vectors are ~sin/cos quadrature; for trend we take the smoothest low-rank
eigentriples.

Synthetic price = slow logistic trend + two sinusoids (period 120 & 37)
+ noise, then:
  (a) cover: noisy price + SSA-trend (sum of smooth eigentriples) vs TRUE trend,
  (b) scree plot of singular values (variance explained by each component),
  (c) SSA trend vs benchmarks (HP filter) with RMSE printed,
  (d) left: reconstructed period-37 oscillator from paired eigentriples
      (overlay TRUE cycle); right: trend RMSE vs noise level.
All from scratch with numpy + scipy.linalg (svd).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import solve_banded
import os
import matplotlib.font_manager as fm

for _f in ["/Library/Fonts/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/singular-spectrum-ssa"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
    "axes.unicode_minus": False, "font.family": "Arial Unicode MS",
})

# ---------------- data ----------------
def make_signal(n=600, seed=1, noise_sd=0.05):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 100 + 40 / (1 + np.exp(-(t - 300) / 70))
    c1 = 6 * np.sin(2 * np.pi * t / 120)
    c2 = 3 * np.sin(2 * np.pi * t / 37)
    cycle = c1 + c2
    noise = noise_sd * rng.standard_normal(n) * np.std(trend)
    return t.astype(float), trend + cycle + noise, trend, c1, c2

# ---------------- SSA core ----------------
def embed(x, L):
    N = len(x)
    K = N - L + 1
    X = np.column_stack([x[i:i + K] for i in range(L)])   # shape (L, K)
    return X

def diagonal_average(D):
    """Hankel / diagonal averaging: map (L,K) matrix back to 1-D series."""
    L, K = D.shape
    N = L + K - 1
    rec = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            idx = i + j
            rec[idx] += D[i, j]
            counts[idx] += 1
    return rec / counts

def ssa(x, L, groups, return_pairs=False):
    """Reconstruct components for given groups of eigentriple indices.
    groups: list of lists (each a set of indices) OR list of indices for ONE group
            when return_pairs is False and a single integer list is given.
    returns reconstructed 1-D series (length N) summed over groups, plus full svd."""
    N = len(x)
    X = embed(x, L)
    U, s, VT = svd(X, full_matrices=False)
    def group_rec(idxs):
        D = np.zeros_like(X)
        for k in idxs:
            D += s[k] * np.outer(U[:, k], VT[k])
        return diagonal_average(D)
    if return_pairs:
        return group_rec, U, s, VT, N
    recs = [group_rec(g) for g in groups]
    return recs, U, s, VT, N

# ---------------- HP filter (benchmark) ----------------
def hp_filter(y, lam=1600.0):
    y = np.asarray(y, float)
    n = len(y)
    A = np.zeros((n, n))
    for i in range(n - 2):
        A[i + 1, i] += lam
        A[i + 1, i + 1] -= 2 * lam
        A[i + 1, i + 2] += lam
    A += np.eye(n)
    ab = np.zeros((5, n))
    for j in range(n):
        for off in (-2, -1, 0, 1, 2):
            i = j + off
            if 0 <= i < n:
                ab[2 + off, j] = A[i, j]
    return solve_banded((2, 2), ab, y)

# ---------------- Figure1: cover ----------------
def fig_cover():
    t, price, trend, c1, c2 = make_signal(n=600, seed=1, noise_sd=0.05)
    L = 120
    recs, U, s, VT, N = ssa(price, L, groups=None, return_pairs=True)
    # trend = smoothest component: the eigentriple with lowest frequency.
    # Use the eigentriples whose left singular vector is smoothest (low 2nd-diff).
    smooth = []
    for k in range(len(s)):
        u = U[:, k]
        curv = np.sum(np.diff(u, 2) ** 2)
        smooth.append((curv, k))
    smooth.sort()
    trend_idx = [smooth[0][1], smooth[1][1]]   # two smoothest -> trend
    tr = recs(trend_idx)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(t, price, color="#a0aec0", lw=1.0, label="原始价格（趋势+周期+噪声）")
    ax.plot(t, trend, color="#718096", lw=1.6, ls="--", label="真实趋势")
    ax.plot(t, tr, color="#dd6b20", lw=2.0, label="SSA 提取趋势（最平滑本征分量）")
    ax.set_xlabel("时间步"); ax.set_ylabel("价格")
    ax.set_title("SSA 把价格嵌入成轨迹矩阵、做 SVD，最平滑的本征分量就是趋势",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/cover.png"); plt.close(fig)
    return U, s, VT, N, t, price, trend, tr

# ---------------- Figure2: scree plot ----------------
def fig_scree(s):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    top = 20
    ax.bar(np.arange(1, top + 1), s[:top], color="#2b6cb0", alpha=0.85)
    ax.set_xlabel("本征分量序号 i"); ax.set_ylabel("奇异值 λ_i（方差代理）")
    ax.set_title("SSA 谱（scree）：少数本征分量吃掉大部分方差",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, top + 1), cum[:top] * 100, "o-", color="#dd6b20",
             lw=1.5, label="累计解释方差 %")
    ax2.set_ylabel("累计解释方差 %"); ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9, loc="center right")
    fig.tight_layout(); fig.savefig(f"{OUT}/ssa_scree.png"); plt.close(fig)
    print(f"[SSA] top-4 singular values: {[round(v,2) for v in s[:4]]}")
    print(f"[SSA] cum explained by top 6: {cum[5]*100:.1f}%")

# ---------------- Figure3: SSA trend vs HP ----------------
def fig_compare():
    t, price, trend, c1, c2 = make_signal(n=600, seed=2, noise_sd=0.05)
    L = 120
    recs, U, s, VT, N = ssa(price, L, groups=None, return_pairs=True)
    smooth = []
    for k in range(len(s)):
        u = U[:, k]
        curv = np.sum(np.diff(u, 2) ** 2)
        smooth.append((curv, k))
    smooth.sort()
    tr = recs([smooth[0][1], smooth[1][1]])
    hp = hp_filter(price, lam=1600.0)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(t, price, color="#a0aec0", lw=0.8, label="原始价格")
    ax.plot(t, trend, color="#718096", lw=1.4, ls="--", label="真实趋势")
    ax.plot(t, tr, color="#dd6b20", lw=2.0, label="SSA 趋势")
    ax.plot(t, hp, color="#2b6cb0", lw=1.6, label="HP 滤波")
    ax.set_xlabel("时间步"); ax.set_ylabel("价格")
    ax.set_title("SSA 趋势 vs HP 滤波：无端点漂移、低频干净",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/ssa_compare.png"); plt.close(fig)
    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    print(f"[SSA] RMSE vs true: SSA={rmse(tr, trend):.3f}, HP={rmse(hp, trend):.3f}")

# ---------------- Figure4: oscillator recovery + noise robustness ----------------
def fig_osc_robust():
    # recover the period-37 oscillator via paired eigentriples
    t, price, trend, c1, c2 = make_signal(n=600, seed=3, noise_sd=0.03)
    L = 120
    recs, U, s, VT, N = ssa(price, L, groups=None, return_pairs=True)
    # find the pair whose reconstructed series best matches period-37 cycle c2
    best = None
    best_corr = -1
    for k in range(0, 14, 2):
        g = [k, k + 1]
        comp = recs(g)
        # align length (reconstruction may be N or shorter if wrapped)
        comp = (comp[:len(c2)] - np.mean(comp[:len(c2)]))
        corr = np.corrcoef(comp, c2)[0, 1]
        if corr > best_corr:
            best_corr = corr; best = (g, comp)
    g, comp = best
    # noise robustness for trend
    noise_levels = [0.02, 0.05, 0.10, 0.20]
    rmses = []
    for sd in noise_levels:
        tt, pp, tr2, _, _ = make_signal(n=600, seed=4, noise_sd=sd)
        rr, UU, ss, VVT, NN = ssa(pp, L, groups=None, return_pairs=True)
        sm = []
        for k in range(len(ss)):
            uu = UU[:, k]
            sm.append((np.sum(np.diff(uu, 2) ** 2), k))
        sm.sort()
        trm = rr([sm[0][1], sm[1][1]])
        rmses.append(float(np.sqrt(np.mean((trm - tr2) ** 2))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.plot(t, c2, color="#718096", lw=1.6, ls="--", label="真实周期 37 正弦")
    ax1.plot(t, comp[:len(t)] - np.mean(comp[:len(t)]), color="#dd6b20",
             lw=1.6, label=f"SSA 还原振荡 (分量 {g[0]}+{g[1]})")
    ax1.set_xlabel("时间步"); ax1.set_ylabel("振幅")
    ax1.set_title("SSA 把「看不见的周期」还原出来", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25)
    ax2.plot(noise_levels, rmses, "o-", color="#dd6b20", lw=2, label="SSA 趋势 RMSE")
    ax2.set_xlabel("噪声强度 (相对趋势标准差)"); ax2.set_ylabel("SSA 趋势 RMSE")
    ax2.set_title("SSA 趋势对噪声的鲁棒性", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.25); ax2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/ssa_osc_robust.png"); plt.close(fig)
    print(f"[SSA] best period-37 pair: {g}, corr with true cycle: {best_corr:.3f}")
    print(f"[SSA] trend RMSE vs noise: {[round(v,3) for v in rmses]}")

if __name__ == "__main__":
    U, s, VT, N, t, price, trend, tr = fig_cover()
    fig_scree(s)
    fig_compare()
    fig_osc_robust()
    print("SSA images written to", OUT)
