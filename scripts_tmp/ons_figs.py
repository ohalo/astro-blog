#!/usr/bin/env python3
"""ONS (Online Newton Step) portfolio selection experiment + figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
OUT = "/Users/halo/workspace/astro-blog/public/images/ons-online-newton-step"
os.makedirs(OUT, exist_ok=True)

# ---------- market simulation: 2 assets, 3000 days ----------
T = 3000
# asset A: volatile mean-reverting-ish, asset B: cash-like small drift
muA, sigA = 0.0005, 0.025
muB, sigB = 0.00005, 0.001
zA = rng.standard_normal(T); zA -= zA.mean()  # demean so sample drift == muA
rA = muA + sigA * zA
# symmetric regime episodes (keep b* interior but create adaptation opportunities)
rA[800:1200] -= 0.0012
rA[2000:2400] += 0.0012
zB = rng.standard_normal(T); zB -= zB.mean()
rB = muB + sigB * zB
X = np.column_stack([1.0 + rA, 1.0 + rB])  # price relatives x_t
m = 2

# ---------- best CRP in hindsight (grid) ----------
grid = np.linspace(0, 1, 201)
def crp_logwealth(b, X):
    port = b * X[:, 0] + (1 - b) * X[:, 1]
    return np.log(port).sum()
lw = np.array([crp_logwealth(b, X) for b in grid])
b_star = grid[lw.argmax()]
LW_star_path = np.cumsum(np.log(b_star * X[:, 0] + (1 - b_star) * X[:, 1]))

def simplex_proj(v):
    """Euclidean projection onto probability simplex."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    idx = np.arange(1, len(v) + 1)
    cond = u - (css - 1) / idx > 0
    rho = idx[cond][-1]
    theta = (css[cond][-1] - 1) / rho
    return np.maximum(v - theta, 0)

def gen_simplex_proj(v, Ainv_unused, A):
    """Projection onto simplex in norm induced by A, via iterative method (small m: solve directly)."""
    # minimize (b-v)' A (b-v) s.t. b in simplex. For m=2 do 1-D search.
    bs = np.linspace(0, 1, 2001)
    B = np.column_stack([bs, 1 - bs])
    D = B - v
    # quadratic form
    q = np.einsum("ij,jk,ik->i", D, A, D)
    return B[q.argmin()]

# ---------- ONS ----------
def run_ons(X, eta=None, eps=1.0, delta=0.0):
    T, m = X.shape
    b = np.ones(m) / m
    A = eps * np.eye(m)
    W = []
    Bs = []
    logw = 0.0
    if eta is None:
        # theory: eta = 0.5 * min(1/(4GD), alpha); use practical constant
        eta = 2.0
    for t in range(T):
        x = X[t]
        ret = b @ x
        logw += np.log(ret)
        W.append(logw)
        Bs.append(b.copy())
        g = -x / ret          # gradient of loss -log(b'x)
        A += np.outer(g, g)
        Ainv_g = np.linalg.solve(A, g)
        y = b - (1.0 / eta) * Ainv_g
        b = gen_simplex_proj(y, None, A)
        if delta > 0:
            b = (1 - delta) * b + delta / m
    return np.array(W), np.array(Bs)

# ---------- EG (exponentiated gradient) ----------
def run_eg(X, eta=0.05):
    T, m = X.shape
    b = np.ones(m) / m
    W, Bs = [], []
    logw = 0.0
    for t in range(T):
        x = X[t]
        ret = b @ x
        logw += np.log(ret)
        W.append(logw); Bs.append(b.copy())
        b = b * np.exp(eta * x / ret)
        b = b / b.sum()
    return np.array(W), np.array(Bs)

# ---------- OGD (online gradient descent, sqrt-T regret) ----------
def run_ogd(X, eta0=0.05):
    T, m = X.shape
    b = np.ones(m) / m
    W, Bs = [], []
    logw = 0.0
    for t in range(T):
        x = X[t]
        ret = b @ x
        logw += np.log(ret)
        W.append(logw); Bs.append(b.copy())
        g = -x / ret
        b = simplex_proj(b - eta0 / np.sqrt(t + 1) * g)
    return np.array(W), np.array(Bs)

W_ons, B_ons = run_ons(X, eta=0.5, eps=1.0)
W_eg, B_eg = run_eg(X, eta=0.05)
W_ogd, B_ogd = run_ogd(X, eta0=0.1)
W_unif = np.cumsum(np.log(X.mean(axis=1)))  # uniform CRP (rebalanced 50/50)
# recompute uniform CRP properly
W_unif = np.cumsum(np.log(0.5 * X[:, 0] + 0.5 * X[:, 1]))

print(f"b* = {b_star:.3f}")
print(f"final logwealth: BCRP={LW_star_path[-1]:.4f} ONS={W_ons[-1]:.4f} EG={W_eg[-1]:.4f} OGD={W_ogd[-1]:.4f} 50/50={W_unif[-1]:.4f}")
print(f"final wealth: BCRP={np.exp(LW_star_path[-1]):.4f} ONS={np.exp(W_ons[-1]):.4f} EG={np.exp(W_eg[-1]):.4f} OGD={np.exp(W_ogd[-1]):.4f} 50/50={np.exp(W_unif[-1]):.4f}")

# regrets vs BCRP on the regime market (can be negative for adaptive algos)
R_ons = LW_star_path - W_ons
R_eg = LW_star_path - W_eg
R_ogd = LW_star_path - W_ogd
print(f"final regret (regime mkt): ONS={R_ons[-1]:.4f} EG={R_eg[-1]:.4f} OGD={R_ogd[-1]:.4f}")

# ---------- deterministic Cover market (cash vs double-or-half alternating) ----------
# classic adversarial-flavored benchmark: b*=0.5, per-round losses are exp-concave
xa2 = np.ones(T)
xb2 = np.where(np.arange(T) % 2 == 0, 2.0, 0.5)
X2 = np.column_stack([xa2, xb2])
lw2 = np.array([crp_logwealth(b, X2) for b in grid])
b_star2 = grid[lw2.argmax()]
LW2_star = np.cumsum(np.log(b_star2 * X2[:, 0] + (1 - b_star2) * X2[:, 1]))
W2_ons, _ = run_ons(X2, eta=8.0, eps=1.0)
W2_eg, _ = run_eg(X2, eta=0.05)
W2_ogd, _ = run_ogd(X2, eta0=0.1)
R2_ons = LW2_star - W2_ons
R2_eg = LW2_star - W2_eg
R2_ogd = LW2_star - W2_ogd
print(f"Cover mkt: b*={b_star2:.3f} final regret ONS={R2_ons[-1]:.4f} OGD={R2_ogd[-1]:.4f} EG={R2_eg[-1]:.4f}")
for tt in [30, 100, 300, 1000, 3000]:
    print(f"  T={tt}: ONS={R2_ons[tt-1]:.3f} OGD={R2_ogd[tt-1]:.3f} EG={R2_eg[tt-1]:.3f}")

# ---------- Fig 1: wealth curves ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax = axes[0]
ax.plot(np.exp(LW_star_path), label=f"事后最优 CRP (b*={b_star:.2f})", color="black", lw=1.8)
ax.plot(np.exp(W_ons), label="ONS 在线牛顿步", color="#d62728", lw=1.5)
ax.plot(np.exp(W_eg), label="EG 指数梯度", color="#1f77b4", lw=1.2)
ax.plot(np.exp(W_ogd), label="OGD 在线梯度下降", color="#2ca02c", lw=1.2)
ax.plot(np.exp(W_unif), label="50/50 朴素再平衡", color="gray", lw=1.0, ls="--")
ax.set_ylabel("财富（初始=1）")
ax.set_title("3000 日双资产：ONS vs EG vs OGD vs 事后最优 CRP")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax2 = axes[1]
pA = np.cumprod(X[:, 0]); pB = np.cumprod(X[:, 1])
ax2.plot(pA, label="资产 A（高波动，含趋势段）", color="#ff7f0e", lw=1.2)
ax2.plot(pB, label="资产 B（类现金低波动）", color="#9467bd", lw=1.2)
ax2.set_xlabel("交易日"); ax2.set_ylabel("价格（初始=1）")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/wealth-curves.png", dpi=110)
plt.close()

# ---------- Fig 2: regret curves + log fit ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
t_axis = np.arange(1, T + 1)
ax = axes[0]
ax.plot(t_axis, R2_ons, label="ONS 遗憾", color="#d62728", lw=1.5)
ax.plot(t_axis, R2_ogd, label="OGD 遗憾", color="#2ca02c", lw=1.2)
ax.plot(t_axis, R2_eg, label="EG 遗憾", color="#1f77b4", lw=1.2)
# reference curves scaled
c_log = max(R2_ons[-1], 1e-6) / np.log(T)
c_sqrt = max(R2_ogd[-1], 1e-6) / np.sqrt(T)
ax.plot(t_axis, c_log * np.log(t_axis), "k--", lw=0.9, label=r"c·log T 参考线")
ax.plot(t_axis, c_sqrt * np.sqrt(t_axis), "k:", lw=0.9, label=r"c·√T 参考线")
ax.set_xlabel("交易日 T"); ax.set_ylabel("对数财富遗憾")
ax.set_title("Cover 市场（现金 vs 翻倍/减半交替）：ONS 贴 log T，OGD 贴 √T")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax = axes[1]
ax.semilogx(t_axis, R2_ons, label="ONS", color="#d62728", lw=1.5)
ax.semilogx(t_axis, c_log * np.log(t_axis), "k--", lw=0.9, label="c·log T")
ax.set_xlabel("交易日 T（对数轴）"); ax.set_ylabel("对数财富遗憾")
ax.set_title("对数横轴下 ONS 遗憾近似直线，即 O(log T)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/regret-curves.png", dpi=110)
plt.close()

# ---------- Fig 3: weight trajectories ----------
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(B_ons[:, 0], label="ONS 资产A权重", color="#d62728", lw=1.3)
ax.plot(B_eg[:, 0], label="EG 资产A权重", color="#1f77b4", lw=1.1)
ax.plot(B_ogd[:, 0], label="OGD 资产A权重", color="#2ca02c", lw=1.1)
ax.axhline(b_star, color="black", ls="--", lw=1.0, label=f"事后最优 b*={b_star:.2f}")
ax.axvspan(800, 1200, color="red", alpha=0.07)
ax.axvspan(2000, 2400, color="green", alpha=0.07)
ax.text(1000, 0.92, "A 趋势下跌段", ha="center", fontsize=9, color="darkred")
ax.text(2200, 0.92, "A 趋势上涨段", ha="center", fontsize=9, color="darkgreen")
ax.set_xlabel("交易日"); ax.set_ylabel("资产 A 权重")
ax.set_ylim(-0.02, 1.02)
ax.set_title("权重轨迹：ONS 收敛快且对 regime 变化反应更敏锐")
ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/weight-trajectories.png", dpi=110)
plt.close()

# ---------- Fig 4: eta sensitivity + transaction cost ----------
etas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
finals = []
for e in etas:
    W_, _ = run_ons(X, eta=e)
    finals.append(np.exp(W_[-1]))
# turnover & cost impact for ONS vs EG
def turnover_series(Bs, X):
    # weight drift then rebalance
    to = []
    for t in range(1, len(Bs)):
        x = X[t - 1]
        drift = Bs[t - 1] * x / (Bs[t - 1] @ x)
        to.append(np.abs(Bs[t] - drift).sum())
    return np.array(to)
to_ons = turnover_series(B_ons, X)
to_eg = turnover_series(B_eg, X)
costs_bp = np.array([0, 2, 5, 10, 20])
def net_wealth(W, to, costs_bp):
    out = []
    for c in costs_bp:
        cost_drag = (c / 1e4) * to.sum()
        out.append(np.exp(W[-1] - cost_drag))
    return out
nw_ons = net_wealth(W_ons, to_ons, costs_bp)
nw_eg = net_wealth(W_eg, to_eg, costs_bp)
print(f"turnover total: ONS={to_ons.sum():.2f} EG={to_eg.sum():.2f}")
print(f"mean daily turnover: ONS={to_ons.mean()*1e4:.2f}bp EG={to_eg.mean()*1e4:.2f}bp")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(etas, finals, "o-", color="#d62728")
ax.axhline(np.exp(LW_star_path[-1]), color="black", ls="--", lw=1.0, label="事后最优 CRP 终值")
ax.set_xscale("log", base=2)
ax.set_xlabel("ONS 学习率 η（log2 轴）"); ax.set_ylabel("终值财富")
ax.set_title("η 敏感性：跨 5 个二倍频程终值变化温和")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(costs_bp, nw_ons, "o-", color="#d62728", label="ONS 扣费终值")
ax.plot(costs_bp, nw_eg, "s-", color="#1f77b4", label="EG 扣费终值")
ax.axhline(np.exp(W_unif[-1]), color="gray", ls="--", lw=1.0, label="50/50 终值（零成本）")
ax.set_xlabel("单边成本（bp）"); ax.set_ylabel("扣费后终值财富")
ax.set_title("换手成本压力：日频再平衡的现实税")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/eta-cost-sensitivity.png", dpi=110)
plt.close()

for e, f in zip(etas, finals):
    print(f"eta={e}: final wealth={f:.4f}")
print("DONE ONS")
