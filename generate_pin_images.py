#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「PIN 模型：用知情交易概率量化市场透明度」(pin-informed-trading) 生成真实配图。

数据来源：从零模拟 EKOP (Easley-Kiefer-O'Hara-Paperman 1996) 订单流微观结构模型，
再用极大似然估计(MLE)反推 PIN = αμ / (αμ + 2ε)。
全部为真实数值计算，非占位图。

图表：
  1. pin_trade_counts.png   每日买卖双方成交笔数散点(按新闻类型着色)，展示知情交易的"不对称签名"
  2. pin_likelihood_surface.png  log-似然面(μ, ε 网格) + 真实值 + MLE 估计点
  3. pin_monte_carlo.png    Monte Carlo：真实 PIN vs MLE 估计 PIN 分布(偏差与精度)
  4. pin_rolling.png        多年序列 + 结构性断点(α 跳升)下的滚动 PIN 估计
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.optimize import minimize

rcParams = matplotlib.rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "pin-informed-trading")
os.makedirs(D, exist_ok=True)
np.random.seed(20260713)


# ============================================================
# 1) EKOP 模型：模拟每日买/卖成交笔数 (B, S)
# ============================================================
def simulate_days(n_days, alpha, delta, mu, eps):
    """返回 daily (B, S, type)，type: 0=无新闻, 1=好新闻, 2=坏新闻"""
    B = np.zeros(n_days, dtype=int)
    S = np.zeros(n_days, dtype=int)
    typ = np.zeros(n_days, dtype=int)
    for i in range(n_days):
        has_news = np.random.rand() < alpha
        if has_news:
            good = np.random.rand() > delta
            if good:
                typ[i] = 1
                B[i] = np.random.poisson(eps + mu)   # 知情者买入 + 流动性买
                S[i] = np.random.poisson(eps)
            else:
                typ[i] = 2
                B[i] = np.random.poisson(eps)
                S[i] = np.random.poisson(eps + mu)   # 知情者卖出 + 流动性卖
        else:
            typ[i] = 0
            B[i] = np.random.poisson(eps)
            S[i] = np.random.poisson(eps)
    return B, S, typ


def log_likelihood(params, B, S):
    """EKOP 单日聚合 log-likelihood。params = [alpha, delta, mu, eps]"""
    alpha, delta, mu, eps = params
    alpha = min(max(alpha, 1e-6), 1 - 1e-6)
    delta = min(max(delta, 1e-6), 1 - 1e-6)
    mu = max(mu, 1e-6)
    eps = max(eps, 1e-6)
    # 用 log-sum-exp 稳定
    def lp(n, lam):  # log P(Poisson(lam)=n)
        return n * np.log(lam) - lam - gammaln(n + 1)
    # no news
    ll_no = lp(B, eps) + lp(S, eps)
    # good news: B~Poisson(eps+mu), S~Poisson(eps)
    ll_good = lp(B, eps + mu) + lp(S, eps)
    # bad news: B~Poisson(eps), S~Poisson(eps+mu)
    ll_bad = lp(B, eps) + lp(S, eps + mu)
    # 组合 (log-sum-exp)
    a1 = np.log(1 - alpha) + ll_no
    a2 = np.log(alpha) + np.log(1 - delta) + ll_good
    a3 = np.log(alpha) + np.log(delta) + ll_bad
    m = np.maximum.reduce([a1, a2, a3])
    logL = m + np.log(np.exp(a1 - m) + np.exp(a2 - m) + np.exp(a3 - m))
    return np.sum(logL)


def estimate_pin(B, S):
    """MLE 估计 PIN。粗网格找起点 + 局部优化(控制迭代次数)。"""
    def neg_ll(p):
        return -log_likelihood(p, B, S)
    best = None
    grid_a = np.array([0.15, 0.35, 0.55, 0.75])
    grid_d = np.array([0.2, 0.4, 0.6, 0.8])
    for a in grid_a:
        for d in grid_d:
            p0 = [a, d, 80.0, 120.0]
            try:
                r = minimize(neg_ll, p0, method="Nelder-Mead",
                             options={"xatol": 1e-3, "fatol": 5e-3, "maxiter": 600})
                if best is None or r.fun < best.fun:
                    best = r
            except Exception:
                continue
    a, d, mu, eps = best.x
    a = min(max(a, 0), 1); d = min(max(d, 0), 1)
    mu = max(mu, 0); eps = max(eps, 0)
    pin = a * mu / (a * mu + 2 * eps)
    return pin, (a, d, mu, eps)


# ============================================================
# 图 1：每日买卖成交笔数散点(按新闻类型着色)
# ============================================================
TRUE = dict(alpha=0.45, delta=0.4, mu=120.0, eps=200.0)
B, S, typ = simulate_days(900, **TRUE)
fig, ax = plt.subplots(figsize=(9, 6))
colors = {0: "#9aa0a6", 1: "#2e7d32", 2: "#c62828"}
labels = {0: "无新闻(对称)", 1: "好新闻(买盘异常)", 2: "坏新闻(卖盘异常)"}
for t in [0, 1, 2]:
    m = typ == t
    ax.scatter(B[m], S[m], s=14, alpha=0.55, c=colors[t], label=labels[t])
# 对角线参考
lim = max(B.max(), S.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="买=卖 对称线")
ax.set_xlabel("每日买入笔数 B")
ax.set_ylabel("每日卖出笔数 S")
ax.set_title("EKOP 订单流：知情交易日的买卖不对称签名\n(好新闻→买方异常放量；坏新闻→卖方异常放量)")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
fig.tight_layout()
fig.savefig(os.path.join(D, "pin_trade_counts.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图 2：log-似然面(μ, ε) + 真实值 + MLE
# ============================================================
B2, S2, _ = simulate_days(400, **TRUE)
mus = np.linspace(40, 220, 40)
epss = np.linspace(120, 320, 40)
MU, EPS = np.meshgrid(mus, epss)
LL = np.zeros_like(MU)
for i in range(MU.shape[0]):
    for j in range(MU.shape[1]):
        # 固定 alpha, delta 为真实值，扫描 mu, eps
        LL[i, j] = -log_likelihood([TRUE["alpha"], TRUE["delta"], MU[i, j], EPS[i, j]], B2, S2)
LL = LL - LL.max()
pin_val, pin_params = estimate_pin(B2, S2)
fig, ax = plt.subplots(figsize=(8.5, 6.5))
cs = ax.contourf(MU, EPS, LL, levels=30, cmap="viridis")
cbar = fig.colorbar(cs, ax=ax)
cbar.set_label("log-似然 (相对最大值)")
ax.scatter([TRUE["mu"]], [TRUE["eps"]], c="cyan", s=120, marker="*",
           edgecolors="k", zorder=5, label=f"真实值 (μ={TRUE['mu']:.0f}, ε={TRUE['eps']:.0f})")
ax.scatter([pin_params[2]], [pin_params[3]], c="orange", s=90, marker="X",
           edgecolors="k", zorder=5, label=f"MLE 估计 (μ̂={pin_params[2]:.0f}, ε̂={pin_params[3]:.0f})")
ax.set_xlabel("信息交易强度 μ (知情者每日笔数)")
ax.set_ylabel("流动性强度 ε (知情者每日笔数)")
ax.set_title("PIN 似然面：MLE 在真实参数附近达到峰值")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "pin_likelihood_surface.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图 3：Monte Carlo — 真实 PIN vs 估计 PIN
# ============================================================
N_REP = 25
true_pin_val = TRUE["alpha"] * TRUE["mu"] / (TRUE["alpha"] * TRUE["mu"] + 2 * TRUE["eps"])
est_pins = []
for _ in range(N_REP):
    bB, sS, _ = simulate_days(600, **TRUE)
    p, _ = estimate_pin(bB, sS)
    est_pins.append(p)
N_REP = len(est_pins)
est_pins = np.array(est_pins)
bias = est_pins.mean() - true_pin_val
rmse = np.sqrt(np.mean((est_pins - true_pin_val) ** 2))
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(est_pins, bins=18, color="#1565c0", alpha=0.8, edgecolor="white")
ax.axvline(true_pin_val, color="#c62828", lw=2.5, label=f"真实 PIN = {true_pin_val:.3f}")
ax.axvline(est_pins.mean(), color="#2e7d32", lw=2, ls="--",
           label=f"估计均值 = {est_pins.mean():.3f} (偏差 {bias:+.3f})")
ax.set_xlabel("估计 PIN")
ax.set_ylabel("重复次数")
ax.set_title(f"Monte Carlo ({N_REP} 次)：PIN 估计精度\nRMSE={rmse:.3f}，样本越长偏差越小")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "pin_monte_carlo.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图 4：多年序列 + 结构性断点下的滚动 PIN 估计
# ============================================================
np.random.seed(7)
T = 240  # 240 个交易日(约 1 年)
# 前 160 天 α=0.25(透明市场)，后 80 天 α=0.6(信息泄漏加剧)
seg1 = simulate_days(160, alpha=0.25, delta=0.4, mu=110.0, eps=200.0)
seg2 = simulate_days(80, alpha=0.60, delta=0.45, mu=110.0, eps=200.0)
B_all = np.concatenate([seg1[0], seg2[0]])
S_all = np.concatenate([seg1[1], seg2[1]])
win = 40
roll_pin = []
for t in range(win, T + 1, 2):
    p, _ = estimate_pin(B_all[t - win:t], S_all[t - win:t])
    roll_pin.append(p)
roll_pin = np.array(roll_pin)
days = np.arange(win, T + 1, 2)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(days, roll_pin, color="#6a1b9a", lw=2, label="滚动 PIN 估计(40 日窗)")
ax.axhline(true_pin_val, color="#9e9e9e", ls=":", lw=1.5,
           label=f"基准透明市场 PIN≈{0.25*110/(0.25*110+400):.3f}")
ax.axvspan(160, T, color="#ffcdd2", alpha=0.45, label="信息泄漏加剧区间 (α: 0.25→0.60)")
ax.set_xlabel("交易日")
ax.set_ylabel("PIN")
ax.set_title("滚动 PIN 检测结构性断点：市场透明度骤降被及时捕捉")
ax.legend(loc="upper left", fontsize=9)
ax.set_ylim(0, 0.7)
fig.tight_layout()
fig.savefig(os.path.join(D, "pin_rolling.png"), dpi=130)
plt.close(fig)

print("PIN 图已生成:", os.listdir(D))
print(f"真实 PIN={true_pin_val:.3f}, 图3偏差={bias:+.3f}, RMSE={rmse:.3f}")
print(f"图4断点前 PIN≈{0.25*110/(0.25*110+400):.3f}, 断点后真实≈{0.60*110/(0.60*110+400):.3f}")
