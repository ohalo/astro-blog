#!/usr/bin/env python3
"""Generate real charts for 'BDS 检验非线性依赖' article.

Methodology (honest, verifiable):
The BDS test (Brock, Dechert, Scheinkman 1987/1996) asks whether a scalar series is
i.i.d. It exploits the fact that for an i.i.d. sequence the joint probability of two
embedded vectors being within distance eps factorizes:
    C^(m) = P( ||X_i^m - X_j^m||_inf < eps )  should equal  (C^(1))^m .
The BDS statistic is the raw deviation W_m = C^(m) - (C^(1))^m, normalized by its
asymptotic std, ~N(0,1) under i.i.d. A non-zero W_m means "not independent" -- and
critically it catches NONLINEAR dependence that linear autocorrelation (Ljung-Box) misses.

We build three series:
  1. Random walk (i.i.d. gaussian returns)         -> should NOT reject (size ~5%)
  2. GARCH(1,1) returns (volatility clustering)     -> rejects: nonlinear heteroskedasticity
  3. Logistic-map chaotic residuals (deterministic)  -> strongly rejects: pure nonlinearity
Core correlation-integral uses scipy.spatial.cKDTree (exact Chebyshev = box counts).
Bootstrap null p-values verify the test at the end of __main__.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

import matplotlib.font_manager as fm
for _f in ["/Library/Fonts/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc",
           "/System/Library/Fonts/PingFang.ttc"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/bds-test-nonlinearity"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
    "axes.unicode_minus": False, "font.family": "Arial Unicode MS",
})

# ---------------- BDS core ----------------
def c_m(x, m, eps):
    x = np.asarray(x, float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    N = len(x) - m + 1
    emb = np.array([x[i:i + m] for i in range(N)])
    tree = cKDTree(emb)
    cnt = tree.count_neighbors(tree, eps, p=np.inf)   # exact Chebyshev box count
    return cnt / (N * N)

def Wm(x, m, eps):
    Cs = {k: c_m(x, k, eps) for k in range(1, 2 * m + 1)}
    return Cs[m] - Cs[1] ** m

def bds_bootstrap(x, m, eps, B=300, rng=None):
    """(W_m, p_value) under i.i.d. null via bootstrap resampling of x.
    p = fraction of bootstrap W* with |W*| >= |W_obs|."""
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

# ---------------- series generators ----------------
def rw_returns(n=2000, seed=1):
    return np.random.default_rng(seed).standard_normal(n)

def garch_returns(n=2000, a0=1e-5, a1=0.09, b1=0.90, seed=2):
    r = np.random.default_rng(seed)
    z = r.standard_normal(n)
    sigma = np.empty(n); ret = np.empty(n)
    sigma[0] = np.sqrt(a0 / (1 - a1 - b1))
    for t in range(1, n):
        sigma[t] = np.sqrt(a0 + a1 * ret[t-1] ** 2 + b1 * sigma[t-1] ** 2)
        ret[t] = sigma[t] * z[t]
    return ret

def chaotic_returns(n=2000, a=3.9, seed=3):
    r = np.random.default_rng(seed)
    x = 0.40 + 0.01 * r.standard_normal()
    out = np.empty(n)
    for t in range(n):
        x = a * x * (1 - x)
        out[t] = x - 0.5
    return out

# ---------------- Figure1: cover (three "look-alike" walks) ----------------
def fig_cover():
    rng = np.random.default_rng(20260721)
    r = rw_returns(800, seed=11)
    g = garch_returns(800, seed=12)
    c = chaotic_returns(800, a=3.9, seed=13)
    sr, sg, sc = np.cumsum(r), np.cumsum(g), np.cumsum(c)
    fig, ax = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
    ax[0].plot(sr, color="#2b6cb0"); ax[0].set_ylabel("随机游走\n(i.i.d.)")
    ax[1].plot(sg, color="#dd6b20"); ax[1].set_ylabel("GARCH\n(波动聚集)")
    ax[2].plot(sc, color="#805ad5"); ax[2].set_ylabel("逻辑斯蒂混沌\n(确定性)")
    for a in ax:
        a.tick_params(labelsize=9); a.grid(alpha=0.25)
    ax[0].set_title("三条都像随机游走，但只有第一条真的独立——BDS 能认出谁藏了非线性",
                    fontsize=12, fontweight="bold")
    ax[2].set_xlabel("时间步")
    fig.tight_layout()
    fig.savefig(f"{OUT}/cover.png"); plt.close(fig)

# ---------------- Figure2: raw W_m scaling across embedding m ----------------
def fig_scaling():
    r = rw_returns(2000, seed=1)
    g = garch_returns(2000, seed=2)
    c = chaotic_returns(2000, a=3.9, seed=3)
    ms = list(range(2, 7))
    W_r = [Wm(r, m, 0.7) for m in ms]
    W_g = [Wm(g, m, 0.7) for m in ms]
    W_c = [Wm(c, m, 0.7) for m in ms]
    print("W_m scaling RW  :", [round(v, 4) for v in W_r])
    print("W_m scaling GARCH:", [round(v, 4) for v in W_g])
    print("W_m scaling CHAOS:", [round(v, 4) for v in W_c])
    # normalized reference: x/std(iid) where std=sqrt(C1(1-C1)/n)
    def znorm(x, m, eps=0.7):
        C1 = c_m(x, 1, eps); n = len(x)
        sd = np.sqrt(C1 * (1 - C1) / n)
        return Wm(x, m, eps) / sd
    Z_r = [znorm(r, m) for m in ms]
    Z_g = [znorm(g, m) for m in ms]
    Z_c = [znorm(c, m) for m in ms]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhspan(-1.96, 1.96, color="green", alpha=0.10, label="正态 ±1.96 (i.i.d. 区间内)")
    ax.plot(ms, Z_r, "o-", color="#2b6cb0", label="随机游走 (i.i.d.)")
    ax.plot(ms, Z_g, "s-", color="#dd6b20", label="GARCH 波动聚集")
    ax.plot(ms, Z_c, "^-", color="#805ad5", label="逻辑斯蒂混沌")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(ms); ax.set_xlabel("嵌入维数 m"); ax.set_ylabel("Wₘ / σ (正态近似)")
    ax.set_title("Wₘ 随嵌入维数 m 缩放：随机游走贴着 0，非线性序列越走越偏",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/bds_scaling.png"); plt.close(fig)

# ---------------- Figure3: Monte-Carlo power vs nonlinearity strength ----------------
def fig_power():
    a_grid = np.linspace(3.6, 4.0, 7)
    reps = 120
    n = 600
    B = 80
    rng = np.random.default_rng(20260722)
    power = []
    for a in a_grid:
        rej = 0
        for s in range(reps):
            c = chaotic_returns(n, a=a, seed=7000 + s)
            _, p = bds_bootstrap(c, 3, 0.7, B=B, rng=rng)
            if p < 0.05:
                rej += 1
        power.append(rej / reps)
    print("Power vs a:", [round(p, 3) for p in power])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a_grid, power, "o-", color="#805ad5", lw=2)
    ax.axhline(0.05, color="green", ls="--", alpha=0.7, label="有效水平 5%")
    ax.axhline(1.0, color="grey", ls=":", alpha=0.6)
    ax.set_xlabel("混沌强度 a (逻辑斯蒂映射参数)")
    ax.set_ylabel("拒绝率 (检验功效)")
    ax.set_title("蒙特卡洛功效：非线性越强，BDS 越不容易漏检",
                fontsize=12, fontweight="bold")
    ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/bds_power.png"); plt.close(fig)

# ---------------- Figure4: size under null (should be ~5%) ----------------
def fig_size():
    reps = 200
    n = 1200
    B = 80
    rng = np.random.default_rng(20260723)
    rej = 0
    pvals = []
    for s in range(reps):
        x = rng.standard_normal(n)
        _, p = bds_bootstrap(x, 4, 0.7, B=B, rng=rng)
        pvals.append(p)
        if p < 0.05:
            rej += 1
    emp_size = rej / reps
    print(f"Empirical size (RW null): {emp_size:.3f}")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pvals, bins=20, color="#2b6cb0", alpha=0.8)
    ax.axvline(0.05, color="red", ls="--", lw=1.5, label="显著性 0.05 阈值")
    ax.set_xlabel("BDS bootstrap p 值 (i.i.d. 零假设下应均匀)")
    ax.set_ylabel("频数")
    ax.set_title(f"检验水平校准：随机游走下拒绝率 = {emp_size:.3f} ≈ 5%",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/bds_size.png"); plt.close(fig)

if __name__ == "__main__":
    fig_cover()
    fig_scaling()
    fig_power()
    fig_size()
    print("BDS images written to", OUT)
