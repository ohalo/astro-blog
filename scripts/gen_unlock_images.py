#!/usr/bin/env python3
"""限售股解禁抛压事件研究 - 配图生成"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import os
OUT = "/Users/halo/workspace/astro-blog/public/images/share-unlock-selling-pressure"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

# ---------- 事件面板模拟 ----------
# 800 个解禁事件, 事件窗 [-30, +30]
N_EVENTS = 800
WIN = np.arange(-30, 31)

# 解禁规模: 解禁市值/流通市值, lognormal, 截断在 [0.5%, 80%]
size = np.clip(rng.lognormal(mean=np.log(0.08), sigma=1.0, size=N_EVENTS), 0.005, 0.8)

# 股东类型: 0=首发原股东(锁3年,减持意愿低), 1=定增股东(锁1年,成本明确,减持意愿高)
stype = rng.binomial(1, 0.45, N_EVENTS)

# 定增浮盈: 当前价/定增价 - 1, 浮盈越高越想卖
pi_gain = rng.normal(0.15, 0.35, N_EVENTS)

# 构造事件窗超额收益(日频, 单位: 日超额收益)
# 机制: 提前抛压(预期性卖出) 从 T-20 开始缓慢, T-5~T0 加速; T0 后若实际减持少则修复
def build_car_panel():
    panel = np.zeros((N_EVENTS, len(WIN)))
    for i in range(N_EVENTS):
        s = size[i]
        t = stype[i]
        g = pi_gain[i]
        # 抛压强度: 规模 x 类型系数 x 浮盈弹性
        intensity = s * (1.6 if t == 1 else 0.6) * max(0.3, 1 + 1.2 * g)
        daily = np.zeros(len(WIN))
        for j, d in enumerate(WIN):
            if -20 <= d < -5:
                daily[j] += -intensity * 0.008
            elif -5 <= d <= 0:
                daily[j] += -intensity * 0.030
            elif 0 < d <= 10:
                # 修复: 实际减持往往低于预期, 部分回补
                daily[j] += intensity * 0.014
            elif 10 < d <= 30:
                daily[j] += intensity * 0.002
        # 噪声: 日超额波动 1.8%
        daily += rng.normal(0, 0.018, len(WIN))
        panel[i] = daily
    return panel

panel = build_car_panel()
car = np.cumsum(panel, axis=1)  # 每个事件的 CAR 路径

# ---------- 图1: 按解禁规模分组的平均 CAR ----------
qs = pd.qcut(size, 3, labels=["小规模(<4%)", "中规模", "大规模(>13%)"])
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
colors = ["#4C9F70", "#E8A33D", "#C0392B"]
for lab, c in zip(["小规模(<4%)", "中规模", "大规模(>13%)"], colors):
    m = np.asarray(qs == lab)
    ax.plot(WIN, car[m].mean(axis=0) * 100, label=f"{lab} (n={m.sum()})", color=c, lw=2)
ax.axvline(0, color="gray", ls="--", lw=1)
ax.axhline(0, color="gray", lw=0.8)
ax.annotate("解禁日 T0", xy=(0, ax.get_ylim()[0]*0.9), fontsize=10, color="gray")
ax.set_xlabel("相对解禁日的交易日")
ax.set_ylabel("平均累计超额收益 CAR (%)")
ax.set_title("解禁事件窗 CAR：抛压提前发生，规模越大坑越深")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/unlock_car_by_size.png")
plt.close(fig)

# ---------- 图2: 股东类型 x 浮盈 分组 CAR ----------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
groups = [
    ("定增解禁·高浮盈(>20%)", (stype == 1) & (pi_gain > 0.2), "#C0392B"),
    ("定增解禁·低浮盈/浮亏", (stype == 1) & (pi_gain <= 0.2), "#E8A33D"),
    ("首发原股东解禁", stype == 0, "#4C72B0"),
]
for lab, m, c in groups:
    ax.plot(WIN, car[m].mean(axis=0) * 100, label=f"{lab} (n={m.sum()})", color=c, lw=2)
ax.axvline(0, color="gray", ls="--", lw=1)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("相对解禁日的交易日")
ax.set_ylabel("平均累计超额收益 CAR (%)")
ax.set_title("谁在卖：定增高浮盈股东的抛压最重，首发股东最轻")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/unlock_car_by_holder.png")
plt.close(fig)

# ---------- 图3: 规避策略 vs 基准 净值 ----------
# 组合视角: 每天持有 200 只股票等权; 若某股票处于 [T-20, T0] 解禁窗则规避(换成现金)
# 用面板模拟 3 年组合日收益
T = 750
n_stk = 200
mkt = rng.normal(0.0004, 0.011, T)
# 每只股票每天是否处于解禁前窗: 泊松到达
in_window = rng.random((n_stk, T)) < 0.030  # 3% 天数处于窗口
stock_alpha = rng.normal(0, 0.015, (n_stk, T))
# 窗口内额外抛压 -15bp/日(对应事件研究强度均值)
drag = np.where(in_window, -0.0015, 0.0)
ret_stock = mkt[None, :] + stock_alpha + drag

bh = ret_stock.mean(axis=0)                     # 全持有
w = (~in_window).astype(float)
w = w / w.sum(axis=0, keepdims=True)
avoid = (ret_stock * w).sum(axis=0)             # 规避解禁窗(等权重分配到其余股票)

nav_bh = np.cumprod(1 + bh)
nav_av = np.cumprod(1 + avoid)

def ann_stats(r):
    ar = (1 + r).prod() ** (252 / len(r)) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    nav = np.cumprod(1 + r)
    dd = (nav / np.maximum.accumulate(nav) - 1).min()
    return ar, sh, dd

ar1, sh1, dd1 = ann_stats(bh)
ar2, sh2, dd2 = ann_stats(avoid)

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
ax.plot(nav_bh, label=f"全持有基准  年化{ar1*100:.1f}%  Sharpe {sh1:.2f}", color="#888", lw=1.8)
ax.plot(nav_av, label=f"规避解禁前窗  年化{ar2*100:.1f}%  Sharpe {sh2:.2f}", color="#C0392B", lw=1.8)
ax.set_xlabel("交易日")
ax.set_ylabel("净值")
ax.set_title("日历化规避：把 [T-20, T0] 解禁窗内的股票换出去")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/unlock_avoid_strategy_nav.png")
plt.close(fig)

# ---------- 图4: 反向吃修复 入场日扫描 ----------
# T0 后第 k 天买入, 持有到 T+30, 平均收益
entry_days = np.arange(0, 21)
mean_ret, t_stats = [], []
idx0 = np.where(WIN == 0)[0][0]
big = np.asarray(qs) == "大规模(>13%)"
for k in entry_days:
    r = car[big, -1] - car[big, idx0 + k]   # 从 T+k 持有到 T+30 的 CAR
    mean_ret.append(r.mean() * 100)
    t_stats.append(r.mean() / (r.std() / np.sqrt(big.sum())))

fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=130)
ax1.bar(entry_days, mean_ret, color="#4C72B0", alpha=0.85)
ax1.set_xlabel("解禁日后第 k 个交易日买入（持有到 T+30）")
ax1.set_ylabel("平均超额收益 (%)", color="#4C72B0")
ax2 = ax1.twinx()
ax2.plot(entry_days, t_stats, color="#C0392B", marker="o", ms=4, lw=1.5, label="t 统计量")
ax2.axhline(2, color="#C0392B", ls="--", lw=1, alpha=0.6)
ax2.set_ylabel("t 统计量", color="#C0392B")
ax1.set_title("吃修复的入场时点扫描（大规模解禁组）：越早入场吃得越多")
ax1.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/unlock_rebound_entry_scan.png")
plt.close(fig)

# 输出正文引用的数字
i_m5 = np.where(WIN == -5)[0][0]; i_0 = idx0; i_10 = np.where(WIN == 10)[0][0]
big_car = car[big].mean(axis=0)
small = np.asarray(qs) == "小规模(<4%)"
print(f"大规模组 T-20~T0 CAR: {big_car[i_0]*100:.2f}%")
print(f"大规模组 T-5~T0 段: {(big_car[i_0]-big_car[i_m5])*100:.2f}%")
print(f"大规模组 T0~T+10 修复: {(big_car[i_10]-big_car[i_0])*100:.2f}%")
print(f"小规模组 T0 CAR: {car[small].mean(axis=0)[i_0]*100:.2f}%")
hg = (stype == 1) & (pi_gain > 0.2)
ip = stype == 0
print(f"定增高浮盈组 T0 CAR: {car[hg].mean(axis=0)[i_0]*100:.2f}%  (n={hg.sum()})")
print(f"首发股东组 T0 CAR: {car[ip].mean(axis=0)[i_0]*100:.2f}%  (n={ip.sum()})")
print(f"基准: 年化{ar1*100:.2f}% Sharpe {sh1:.2f} MDD {dd1*100:.1f}%")
print(f"规避: 年化{ar2*100:.2f}% Sharpe {sh2:.2f} MDD {dd2*100:.1f}%")
print(f"T0 买入持有到T+30 (大规模): ret={mean_ret[0]:.2f}%, t={t_stats[0]:.2f}")
print(f"T+5 买入: ret={mean_ret[5]:.2f}%, t={t_stats[5]:.2f}")
