#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 regime-switching-kelly 文章配图（纯 numpy 合成数据）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/regime-switching-kelly"
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

# ---------- 生成两状态市场 ----------
# 状态0=牛市(高均值低波动), 状态1=熊市(负均值高波动)
T = 2000
# 转移矩阵：牛市粘性0.98，熊市粘性0.95
P = np.array([[0.992, 0.008], [0.03, 0.97]])
mu = np.array([0.0009, -0.0016])     # 日均值
sig = np.array([0.009, 0.024])        # 日波动

states = np.zeros(T, dtype=int)
for t in range(1, T):
    states[t] = np.random.choice(2, p=P[states[t-1]])
rets = np.random.normal(mu[states], sig[states])
price = 100 * np.cumprod(1 + rets)

# ---------- 图1：市场状态与价格 ----------
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(price, color="#1f2a44", lw=1.2, label="资产价格")
# 阴影标注熊市
in_bear = states == 1
start = None
for t in range(T):
    if in_bear[t] and start is None:
        start = t
    elif not in_bear[t] and start is not None:
        ax.axvspan(start, t, color="#e74c3c", alpha=0.12)
        start = None
if start is not None:
    ax.axvspan(start, T, color="#e74c3c", alpha=0.12)
ax.set_title("两状态市场：红色阴影为熊市状态", fontsize=13)
ax.set_xlabel("交易日")
ax.set_ylabel("价格")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/regime_price.png", dpi=110)
plt.close(fig)

# ---------- HMM 滤波（前向算法，在线估计状态后验） ----------
def gaussian_pdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

alpha = np.zeros((T, 2))
prior = np.array([0.5, 0.5])
# t=0
emis = gaussian_pdf(rets[0], mu, sig)
alpha[0] = prior * emis
alpha[0] /= alpha[0].sum()
for t in range(1, T):
    pred = alpha[t-1] @ P
    emis = gaussian_pdf(rets[t], mu, sig)
    a = pred * emis
    alpha[t] = a / a.sum()
p_bear = alpha[:, 1]   # 熊市后验概率

# ---------- 图2：熊市后验概率 vs 真实状态 ----------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(p_bear, color="#e74c3c", lw=1.0, label="HMM 熊市后验概率")
ax.fill_between(range(T), 0, in_bear.astype(float), color="#95a5a6",
                alpha=0.25, step="pre", label="真实熊市状态")
ax.set_title("在线状态滤波：后验概率追踪真实 regime", fontsize=13)
ax.set_xlabel("交易日")
ax.set_ylabel("概率 / 状态")
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/regime_posterior.png", dpi=110)
plt.close(fig)

# ---------- 凯利下注 ----------
# 各状态的凯利分数 f* = mu / sig^2 (对数正态近似)
f_bull = mu[0] / sig[0] ** 2
f_bear = mu[1] / sig[1] ** 2
kelly_state = np.array([f_bull, f_bear])

# 半凯利上限，避免过度杠杆
HALF = 0.5
CAP = 1.0
SHORT_FLOOR = -0.5   # 允许适度做空

def clip_f(f, floor=0.0):
    return np.clip(f * HALF, floor, CAP)

# 策略A：全凯利固定（假设永远牛市参数）—— 忽略状态
f_static = clip_f(f_bull)
# 策略B：状态切换凯利（用后验期望的凯利分数，允许适度做空）
f_dynamic = clip_f(p_bear * kelly_state[1] + (1 - p_bear) * kelly_state[0], floor=SHORT_FLOOR)
# 策略C：买入持有
# 权益曲线（次日执行：用 t 的信号，t+1 的收益）
def equity(frac_arr):
    eq = np.ones(T)
    for t in range(1, T):
        f = frac_arr[t-1] if hasattr(frac_arr, "__len__") else frac_arr
        eq[t] = eq[t-1] * (1 + f * rets[t])
    return eq

eq_static = equity(np.full(T, f_static))
eq_dynamic = equity(f_dynamic)
eq_bh = np.cumprod(1 + rets)

# ---------- 图3：下注比例随状态变化 ----------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(f_dynamic, color="#2980b9", lw=1.1, label="状态切换凯利仓位")
ax.axhline(f_static, color="#e67e22", ls="--", lw=1.2, label=f"固定半凯利={f_static:.2f}")
ax2 = ax.twinx()
ax2.fill_between(range(T), 0, in_bear.astype(float), color="#e74c3c",
                 alpha=0.10, step="pre")
ax2.set_yticks([])
ax.set_title("下注比例随熊市概率自适应收缩", fontsize=13)
ax.set_xlabel("交易日")
ax.set_ylabel("凯利仓位比例")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/kelly_fraction.png", dpi=110)
plt.close(fig)

# ---------- 图4：权益曲线对比 ----------
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(eq_dynamic, color="#2980b9", lw=1.6, label="状态切换凯利")
ax.plot(eq_static, color="#e67e22", lw=1.3, label="固定半凯利")
ax.plot(eq_bh, color="#7f8c8d", lw=1.2, ls="--", label="买入持有")
ax.set_title("权益曲线对比（对数轴）", fontsize=13)
ax.set_yscale("log")
ax.set_xlabel("交易日")
ax.set_ylabel("净值（对数）")
ax.legend(loc="upper left")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(f"{OUT}/equity_compare.png", dpi=110)
plt.close(fig)

# ---------- 图5：凯利分数曲线（下注比例 vs 期望增长率） ----------
f_grid = np.linspace(-0.5, 2.5, 200)
# 用牛市参数的长期增长率 g(f)=f*mu - 0.5*f^2*sig^2
g_bull = f_grid * mu[0] - 0.5 * f_grid ** 2 * sig[0] ** 2
g_bear = f_grid * mu[1] - 0.5 * f_grid ** 2 * sig[1] ** 2
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(f_grid, g_bull * 1e4, color="#27ae60", lw=1.8, label="牛市 g(f)")
ax.plot(f_grid, g_bear * 1e4, color="#e74c3c", lw=1.8, label="熊市 g(f)")
ax.axvline(f_bull, color="#27ae60", ls=":", lw=1.2)
ax.axvline(f_bear, color="#e74c3c", ls=":", lw=1.2)
ax.axhline(0, color="#333", lw=0.8)
ax.set_title("凯利曲线：期望对数增长率随下注比例变化", fontsize=13)
ax.set_xlabel("下注比例 f")
ax.set_ylabel("日增长率 g(f)（×1e-4）")
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/kelly_curve.png", dpi=110)
plt.close(fig)

# ---------- 指标 ----------
def metrics(eq):
    r = np.diff(eq) / eq[:-1]
    total = eq[-1] / eq[0] - 1
    sharpe = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(252)
    peak = np.maximum.accumulate(eq)
    mdd = np.min(eq / peak - 1)
    return total, sharpe, mdd

for name, eq in [("状态切换凯利", eq_dynamic), ("固定半凯利", eq_static), ("买入持有", eq_bh)]:
    t, s, m = metrics(eq)
    print(f"{name}: 总收益={t*100:.1f}% Sharpe={s:.2f} 最大回撤={m*100:.1f}%")

print(f"牛市凯利 f*={f_bull:.2f}, 熊市凯利 f*={f_bear:.2f}")
print(f"熊市天数占比={in_bear.mean()*100:.1f}%")
# 状态识别准确率
pred_bear = p_bear > 0.5
acc = (pred_bear == in_bear).mean()
print(f"状态识别准确率={acc*100:.1f}%")
print("图片已生成:", os.listdir(OUT))
