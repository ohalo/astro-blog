#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Newey-West 稳健标准误文章配图（合成数据）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
for cand in ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Heiti SC", "STHeiti"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/newey-west-robust-se"
os.makedirs(OUT, exist_ok=True)

BLUE = "#2c6fbb"
RED = "#d1495b"
GREEN = "#2a9d8f"
GRAY = "#6c757d"
ORANGE = "#e09f3e"

rng = np.random.default_rng(42)

# ---------- 公共：生成一条自相关的策略收益序列 ----------
T = 240  # 20年月度
# 真实月均超额收益 = 0（原假设成立），但注入正自相关（动量型策略残差常见）
phi = 0.35  # AR(1) 系数
eps = rng.normal(0, 0.045, T)
r = np.zeros(T)
for t in range(1, T):
    r[t] = phi * r[t - 1] + eps[t]
r = r + 0.0  # 均值为0，演示假阳性

# ---------- 图1：自相关的收益序列 vs 其 ACF ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.bar(np.arange(T), r * 100, color=np.where(r >= 0, BLUE, RED), width=1.0, alpha=0.8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("策略月度超额收益（真实均值=0，注入AR(1)自相关）", fontsize=11)
ax.set_xlabel("月")
ax.set_ylabel("收益 (%)")

# ACF
def acf(x, nlags):
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x * x)
    return np.array([np.sum(x[k:] * x[:n - k]) / denom for k in range(nlags + 1)])

ax = axes[1]
lags = 20
a = acf(r, lags)
conf = 1.96 / np.sqrt(T)
ax.bar(np.arange(lags + 1), a, color=BLUE, width=0.6, alpha=0.85)
ax.axhline(conf, color=RED, ls="--", lw=1, label="95% 置信带")
ax.axhline(-conf, color=RED, ls="--", lw=1)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("残差自相关函数 ACF：低阶显著为正", fontsize=11)
ax.set_xlabel("滞后阶数 (lag)")
ax.set_ylabel("自相关系数")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/autocorr_series_acf.png", dpi=130, bbox_inches="tight")
plt.close()
print("saved autocorr_series_acf.png")

# ---------- 图2：OLS(iid) vs Newey-West 标准误 与 t 值 对比 ----------
# 蒙特卡洛：反复抽样，记录两种 t 值超过 1.96 的比例（假阳性率）
def nw_se(x, L):
    """Newey-West HAC 标准误：估计均值的标准误。"""
    x = np.asarray(x, float)
    n = len(x)
    u = x - x.mean()
    gamma0 = np.sum(u * u) / n
    s = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)  # Bartlett 核
        gk = np.sum(u[k:] * u[:n - k]) / n
        s += 2 * w * gk
    return np.sqrt(s / n)

def iid_se(x):
    x = np.asarray(x, float)
    return np.std(x, ddof=1) / np.sqrt(len(x))

n_sim = 3000
L = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))  # 经验带宽 ≈ 5
fp_iid, fp_nw = 0, 0
t_iid_list, t_nw_list = [], []
for _ in range(n_sim):
    e = rng.normal(0, 0.045, T)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    m = x.mean()
    t_i = m / iid_se(x)
    t_n = m / nw_se(x, L)
    t_iid_list.append(t_i)
    t_nw_list.append(t_n)
    fp_iid += abs(t_i) > 1.96
    fp_nw += abs(t_n) > 1.96
fp_iid /= n_sim
fp_nw /= n_sim

fig, ax = plt.subplots(figsize=(9, 4.6))
bins = np.linspace(-4.5, 4.5, 60)
ax.hist(t_iid_list, bins=bins, color=RED, alpha=0.55,
        label=f"OLS/iid t 值 (假阳性率={fp_iid:.1%})")
ax.hist(t_nw_list, bins=bins, color=BLUE, alpha=0.55,
        label=f"Newey-West t 值 (假阳性率={fp_nw:.1%})")
ax.axvline(1.96, color="black", ls="--", lw=1)
ax.axvline(-1.96, color="black", ls="--", lw=1)
ax.text(1.98, ax.get_ylim()[1] * 0.9, "±1.96", fontsize=9)
ax.set_title(f"3000次模拟 t 值分布：真实均值=0，名义显著性应为5%（带宽L={L}）", fontsize=11)
ax.set_xlabel("t 统计量")
ax.set_ylabel("频数")
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.savefig(f"{OUT}/tstat_distribution.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"saved tstat_distribution.png  fp_iid={fp_iid:.3f} fp_nw={fp_nw:.3f}")

# ---------- 图3：带宽 L 对标准误的影响 ----------
Ls = np.arange(0, 25)
# 用一条固定的强自相关序列
e = rng.normal(0, 0.045, T)
x = np.zeros(T)
for t in range(1, T):
    x[t] = 0.45 * x[t - 1] + e[t]
ses = [nw_se(x, max(L_, 0)) if L_ > 0 else iid_se(x) for L_ in Ls]
ses = np.array(ses) * 100
opt_L = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))

fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(Ls, ses, "-o", color=BLUE, ms=4, lw=1.6)
ax.axhline(ses[0], color=GRAY, ls=":", lw=1.2, label=f"iid 标准误 (L=0) = {ses[0]:.3f}%")
ax.axvline(opt_L, color=RED, ls="--", lw=1.4, label=f"经验带宽 L*={opt_L}")
ax.set_title("Newey-West 标准误随带宽 L 上升而增大（自相关被逐步计入）", fontsize=11)
ax.set_xlabel("带宽 L（纳入的滞后阶数）")
ax.set_ylabel("均值的标准误 (%)")
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.savefig(f"{OUT}/bandwidth_effect.png", dpi=130, bbox_inches="tight")
plt.close()
print("saved bandwidth_effect.png")

# ---------- 图4：应用——夏普率显著性被高估 ----------
# 展示同一策略在 iid 与 NW 下的 t(Sharpe) 与去膨胀夏普
np.random.seed(1)
strategies = ["动量", "反转", "carry", "价值", "低波"]
phis = [0.40, -0.15, 0.30, 0.10, 0.25]
t_iid_s, t_nw_s = [], []
for ph in phis:
    e = rng.normal(0, 0.04, T)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = ph * x[t - 1] + e[t]
    x = x + 0.006  # 注入一点真实 alpha
    m = x.mean()
    t_iid_s.append(m / iid_se(x))
    t_nw_s.append(m / nw_se(x, opt_L))

xpos = np.arange(len(strategies))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar(xpos - w / 2, t_iid_s, w, color=RED, alpha=0.8, label="iid t 值（高估）")
ax.bar(xpos + w / 2, t_nw_s, w, color=BLUE, alpha=0.85, label="Newey-West t 值")
ax.axhline(1.96, color="black", ls="--", lw=1, label="t=1.96 显著线")
ax.set_xticks(xpos)
ax.set_xticklabels(strategies)
ax.set_title("五个策略的均值显著性：正自相关越强，iid 越乐观", fontsize=11)
ax.set_ylabel("t 统计量")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/sharpe_significance.png", dpi=130, bbox_inches="tight")
plt.close()
print("saved sharpe_significance.png")
print("ALL DONE")
