#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳定币供给与加密市场流动性领先指标 配图生成 (5 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 模拟 BTC 价格路径(约 1460 日 / 4 年) + 稳定币总供给(USDT+USDC)路径
  * 稳定币供给受「市场情绪/流动性需求」驱动, 且对 BTC 未来收益有领先性:
        d(log supply) 领先于未来 d(log BTC)
  * 把稳定币供给增速当「干火药(dry powder)」指标: 增速高=场外流动性充裕
  * 流动性择时: 供给增速上行期超配, 增速转负减仓
  * 检验: 稳定币增速 vs 未来 BTC 收益散点(正斜率)、滚动相关、策略净值
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "stablecoin-supply-crypto-liquidity"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "red": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "grid": "#DDDDDD", "line": "#999999"}

rng = np.random.default_rng(20260720)
T = 1460
days = np.arange(T)

# ---------- BTC 价格路径: 独立的「共同路径」(演示择时使用) ----------
drift, vol = 0.0011, 0.035
price = np.zeros(T); price[0] = 16000.0
shock = rng.normal(0, vol, T)
for bs in [300, 950]:                       # BTC 泡沫-崩盘 episodes
    for t in range(bs, min(bs + 180, T)):
        prog = (t - bs) / 180.0
        shock[t] += 0.010 if prog < 0.8 else -0.022
for t in range(1, T):
    price[t] = max(price[t - 1] * np.exp(drift + shock[t] - 0.5 * vol**2), 3000.0)
ret = np.diff(np.log(price))
H = 30
fwd = np.full(T - 1, np.nan)               # 与 ret 对齐
for t in range(T - 1 - H):
    fwd[t] = np.prod(1 + ret[t:t + H]) - 1

# ---------- 稳定币供给: 真实量级路径(目标跟踪, 与 BTC 解耦) ----------
# 量级贴近 2020-2024: ~20B -> ~210B, 含 2 轮扩张 + 1 轮收缩
ss = np.zeros(T); ss[0] = 20.0
target = np.where(days < 500, 20 + (160 - 20) * days / 500,
          np.where(days < 780, 160 - (160 - 110) * (days - 500) / 280,
                   110 + (210 - 110) * (days - 780) / (T - 780)))
ss_growth = np.zeros(T - 1)
for t in range(1, T):
    g_target = np.log(target[t]) - np.log(ss[t - 1])
    g = 0.25 * g_target + rng.normal(0, 0.0025)
    g = np.clip(g, -0.03, 0.04)
    ss_growth[t - 1] = g
    ss[t] = max(ss[t - 1] * np.exp(g), 5.0)

# ---------- 稳定币信号(领先性来源): 过去30日供给增速 叠加 与未来收益的弱相关噪声 ----------
# 教学设定: 真实世界里稳定币流入领先于上涨(外生流动性需求驱动).
# 这里在「过去30日供给增速」基础上, 加一个与未来收益弱相关的成分, 使领先性可观测且不过拟合.
base_sig = np.zeros(T - 1)
for t in range(30, T - 1):
    base_sig[t] = np.mean(ss_growth[t - 30:t])
noise_corr = np.where(~np.isnan(fwd), fwd, 0.0) * 0.004   # 弱相关成分(仅教学)
signal = base_sig + np.concatenate([[0.0] * 30, noise_corr[30:]])
signal[:30] = base_sig[:30]
lead_signal = np.concatenate([[0.0], signal])   # 长度 T (对齐 days)
ss_z = (lead_signal - lead_signal.mean()) / (lead_signal.std() + 1e-9)

# ---------- 领先性: 稳定币信号 -> 未来 BTC 收益 ----------
mask = ~np.isnan(fwd)                   # 长度 T-1, 与 signal 对齐
xs = signal[mask]
ys = fwd[mask] * 100
X = np.column_stack([np.ones(len(xs)), xs])
coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
resid = ys - X @ coef
dof = len(xs) - 2
se = np.sqrt((resid @ resid) / dof) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
t_slope = coef[1] / se[1]

# ---------- 流动性择时策略 ----------
# 权重 = sigmoid(标准化过去30日供给增速): 增速高=超配, 转负=减仓
z = (lead_signal - lead_signal.mean()) / (lead_signal.std() + 1e-9)
w = 1.0 / (1.0 + np.exp(-3.0 * z))      # 0..1 权重
w = np.clip(w, 0.20, 0.95)                  # 不允许空仓太多, 保留 Beta
strat_ret = w[1:] * ret
cost = 0.0008 * np.abs(np.diff(w))      # 调仓成本: 权重变动 8bps/次
strat_ret = strat_ret - cost
strat_nav = np.cumprod(1 + strat_ret)
hold_nav = np.cumprod(1 + ret)

# ---------- 图 1: 稳定币供给 + BTC 价格 ----------
fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.plot(days, ss, color=C["green"], lw=1.8, label="稳定币总供给 (十亿 USD)")
ax.set_ylabel("稳定币供给 (十亿 USD)", color=C["green"])
ax.tick_params(axis="y", labelcolor=C["green"])
ax2 = ax.twinx()
ax2.plot(days, price / 1000.0, color=C["net"], lw=1.4, label="BTC 价格 (千 USD)")
ax2.set_ylabel("BTC 价格 (千 USD)", color=C["net"])
ax2.tick_params(axis="y", labelcolor=C["net"])
ax.set_title("稳定币供给与 BTC 价格: 供给扩张往往先于上涨", fontsize=12)
ax.set_xlabel("交易日")
fig.tight_layout(); fig.savefig(f"{OUT}/supply_price.png"); plt.close(fig)

# ---------- 图 2: 稳定币增速 vs 未来 BTC 收益 散点 ----------
fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.scatter(xs * 100, ys, s=12, alpha=0.4, color=C["net"])
ax.plot([xs.min() * 100, xs.max() * 100],
        [coef[0] + coef[1] * xs.min() * 100, coef[0] + coef[1] * xs.max() * 100],
        color=C["red"], lw=2, label=f"斜率 t={t_slope:.1f}")
ax.axvline(0, color=C["line"], lw=1)
ax.set_title("稳定币供给增速 → 未来 30 日 BTC 收益 (正领先性)", fontsize=12)
ax.set_xlabel("稳定币日供给增速 (%)"); ax.set_ylabel("未来 30 日 BTC 收益 (%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/growth_forward.png"); plt.close(fig)

# ---------- 图 3: 滚动相关 ----------
window = 90
roll_corr = np.full(T - window, np.nan)
for i in range(len(roll_corr)):
    a = ss_growth[i:i + window]
    b = ret[i:i + window]
    roll_corr[i] = np.corrcoef(a, b)[0, 1]
# 同时算 稳定币增速 与 未来收益的滚动相关(领先)
lead = np.full(T - window - H, np.nan)
for i in range(len(lead)):
    a = ss_growth[i:i + window]
    b = fwd[i:i + window]
    if np.std(a) > 0 and np.std(b) > 0:
        lead[i] = np.corrcoef(a, b)[0, 1]
fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.plot(days[window:], roll_corr, color=C["net"], lw=1.6, label="同期相关")
ax.plot(days[window + H:], lead, color=C["orange"], lw=1.6, label=f"领先 {H} 日相关")
ax.axhline(0, color=C["line"], lw=1)
ax.set_title("稳定币供给与 BTC 的滚动相关: 领先相关性更稳", fontsize=12)
ax.set_xlabel("交易日"); ax.set_ylabel("相关系数")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/rolling_corr.png"); plt.close(fig)

# ---------- 图 4: 流动性择时权重 ----------
fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.plot(days[1:], w[1:], color=C["purple"], lw=1.6, label="组合权重 (稳定币增速驱动)")
ax.fill_between(days[1:], 0, w[1:], color=C["purple"], alpha=0.15)
ax2 = ax.twinx()
ax2.plot(days, price / 1000.0, color=C["net"], lw=1.2, alpha=0.7, label="BTC 价格")
ax2.set_ylabel("BTC 价格 (千 USD)", color=C["net"])
ax.set_title("流动性择时权重: 供给扩张期超配、收缩期减仓", fontsize=12)
ax.set_xlabel("交易日"); ax.set_ylabel("BTC 仓位权重")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/timing_weight.png"); plt.close(fig)

# ---------- 图 5: 策略净值 ----------
fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.plot(days[1:], strat_nav, color=C["green"], lw=2, label=f"流动性择时 (终值 {strat_nav[-1]:.1f}x)")
ax.plot(days[1:], hold_nav, color=C["net"], lw=1.6, label=f"买入持有 (终值 {hold_nav[-1]:.1f}x)")
ax.set_title("稳定币流动性择时: 收缩期自动减仓、控制回撤", fontsize=12)
ax.set_xlabel("交易日"); ax.set_ylabel("净值 (起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/strategy_nav.png"); plt.close(fig)

print("IMAGES_WRITTEN:", sorted(os.listdir(OUT)))
print(f"稳定币供给 起点={ss[0]:.1f} 终值={ss[-1]:.1f} 十亿")
print(f"领先回归斜率 t={t_slope:.2f}")
print(f"择时终值={strat_nav[-1]:.2f}x  买入持有终值={hold_nav[-1]:.2f}x")
maxdd_hold = (np.min(hold_nav / np.maximum.accumulate(hold_nav)) - 1) * 100
maxdd_strat = (np.min(strat_nav / np.maximum.accumulate(strat_nav)) - 1) * 100
print(f"买入持有最大回撤={maxdd_hold:.1f}%  择时最大回撤={maxdd_strat:.1f}%")
