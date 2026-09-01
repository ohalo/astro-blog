#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「训练大模型预测股市的最大隐形杀手：时间穿越与数据泄漏」生成真实配图与统计数字。

核心逻辑:
  - 合成面板 N=200 股票 x T=180 月。
  - 真实 ex-ante 信号 s_t: 每股 AR(1)(rho=0.9), 横截面标准化, 真实月度 rank-IC 约 0.02。
  - 真实收益: r_{t+1} = 0.02 * s_t + 0.8 * mkt + eps, eps ~ N(0, 0.06^2)。
  - 泄漏信号: s_leak = s_t + gamma * r_{t+1}  (特征被同期/未来收益污染, gamma=0.15)。
    —— 对应真实世界的"标签泄漏": 新闻情绪用抓取时间而非发布时间、全样本标准化、
       用事后修正的财务数据、特征对齐 shift bug 等。
  - 对比: 诚实策略(按 s_t 排名) vs 泄漏策略(按 s_leak 排名) 的
      逐月 rank-IC / 多空 Sharpe / 累计净值 / 年化 / 最大回撤。
  - 结论: 泄漏版 Sharpe 虚高数倍, 一旦时间严格切分立刻打回原形。

图片:
  leak_ic_compare.png      —— 诚实信号 vs 泄漏信号 逐月 rank-IC
  leak_cum_nav.png         —— 诚实 vs 泄漏 多空累计净值
  leak_sharpe_bar.png      —— 年化收益 / Sharpe / 最大回撤 对比柱状图
"""
import os
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "look-ahead-leakage-stock-prediction")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260901)
N, T = 200, 180
rho, alpha, beta, sig_eps, gamma = 0.90, 0.003, 0.8, 0.06, 0.18

# 真实 ex-ante 信号: AR(1) + 创新, 横截面标准化
s = np.zeros((T, N))
innov = rng.normal(0, 0.15, (T, N))
s[0] = innov[0]
for t in range(1, T):
    s[t] = rho * s[t - 1] + np.sqrt(1 - rho**2) * innov[t]

# 横截面标准化(用当期截面, 无未来信息)
def cs_z(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-9
    return (x - mu) / sd

s_z = cs_z(s)

# 市场收益 + 收益过程 r_{t+1} = alpha * s_t + beta * mkt + eps
mkt = rng.normal(0.005, 0.04, T)
eps = rng.normal(0, sig_eps, (T, N))
r = np.zeros((T, N))
for t in range(T - 1):
    r[t + 1] = alpha * s_z[t] + beta * mkt[t] + eps[t + 1]

# 泄漏信号: 特征在 t 时刻却看到了 t+1 的未来收益(典型 look-ahead)
s_leak = np.zeros_like(s_z)
s_leak[:-1] = s_z[:-1] + gamma * r[1:] / 0.06  # 用下一期收益污染当期特征
s_leak_z = cs_z(s_leak)

# 逐月 rank-IC
def rank_ic(sig, ret, start=12):
    ics = []
    for t in range(start, T - 1):
        x = sig[t]; y = ret[t + 1]
        rx = rankdata(x); ry = rankdata(y)
        rx = (rx - rx.mean()) / (rx.std() + 1e-9)
        ry = (ry - ry.mean()) / (ry.std() + 1e-9)
        ics.append((rx * ry).mean())
    return np.array(ics)

ic_true = rank_ic(s_z, r)
ic_leak = rank_ic(s_leak_z, r)

# 多空组合: 每月 top/bottom 20% 等权, 收益 = long - short
def long_short_nav(sig, ret, q=0.2):
    nav = 1.0; rets = []
    for t in range(12, T - 1):
        x = sig[t]; y = ret[t + 1]
        nq = max(1, int(N * q))
        idx = np.argsort(x)
        short_ = idx[:nq]; long_ = idx[-nq:]
        r_ls = y[long_].mean() - y[short_].mean()
        rets.append(r_ls)
        nav *= (1 + r_ls)
    return np.array(rets), nav

ls_true, nav_true = long_short_nav(s_z, r)
ls_leak, nav_leak = long_short_nav(s_leak_z, r)

def stats(rets):
    ann = rets.mean() * 12
    vol = rets.std() * np.sqrt(12)
    sharpe = ann / (vol + 1e-9)
    nav = np.cumprod(1 + rets)
    mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    return ann, sharpe, mdd

a_true, s_true, mdd_true = stats(ls_true)
a_leak, s_leak, mdd_leak = stats(ls_leak)

print(f"TRUE  IC={ic_true.mean():.4f}  ann={a_true:.2%}  sharpe={s_true:.2f}  mdd={mdd_true:.2%}")
print(f"LEAK  IC={ic_leak.mean():.4f}  ann={a_leak:.2%}  sharpe={s_leak:.2f}  mdd={mdd_leak:.2%}")
print(f"IC inflation: {ic_leak.mean()/ic_true.mean():.1f}x  |  Sharpe inflation: {s_leak/s_true:.1f}x")

# ===== 图1: 逐月 rank-IC =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
ax1.plot(ic_true, color="#2F4B7C", lw=1.2, label=f"诚实信号 (均值 {ic_true.mean():.3f})")
ax1.plot(ic_leak, color="#C44E52", lw=1.2, label=f"泄漏信号 (均值 {ic_leak.mean():.3f})")
ax1.axhline(0, color="black", lw=0.7)
ax1.set_title("逐月 rank-IC：泄漏把弱信号放大数倍", fontsize=12, fontweight="bold")
ax1.set_xlabel("月份"); ax1.set_ylabel("rank-IC")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
b = ax2.bar(["诚实信号", "泄漏信号"], [ic_true.mean(), ic_leak.mean()],
            color=["#2F4B7C", "#C44E52"], width=0.5)
ax2.set_title("整体平均 rank-IC", fontsize=12, fontweight="bold")
ax2.set_ylabel("平均 rank-IC"); ax2.grid(True, alpha=0.3, axis="y")
for bb, vv in zip(b, [ic_true.mean(), ic_leak.mean()]):
    ax2.text(bb.get_x() + bb.get_width()/2, vv + 0.002, f"{vv:.3f}", ha="center", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(D, "leak_ic_compare.png"), dpi=150, bbox_inches="tight"); plt.close()

# ===== 图2: 多空累计净值 =====
nav_true_c = np.cumprod(1 + ls_true); nav_leak_c = np.cumprod(1 + ls_leak)
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(nav_leak_c, color="#C44E52", lw=1.8, label=f"泄漏策略 (Sharpe {s_leak:.2f})")
ax.plot(nav_true_c, color="#2F4B7C", lw=1.8, label=f"诚实策略 (Sharpe {s_true:.2f})")
ax.axhline(1, color="black", lw=0.7)
ax.set_yscale("log")
ax.set_title("多空累计净值（log 轴）：泄漏策略的回测曲线只是海市蜃楼", fontsize=13, fontweight="bold")
ax.set_xlabel("月份"); ax.set_ylabel("净值（log，起始=1）")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "leak_cum_nav.png"), dpi=150, bbox_inches="tight"); plt.close()

# ===== 图3: 年化 / Sharpe / 回撤 对比 =====
labels = ["年化收益", "Sharpe", "最大回撤"]
true_vals = [a_true, s_true, mdd_true]
leak_vals = [a_leak, s_leak, mdd_leak]
x = np.arange(3); w = 0.35
fig, ax = plt.subplots(figsize=(11, 5.0))
b1 = ax.bar(x - w/2, true_vals, w, color="#2F4B7C", label="诚实策略")
b2 = ax.bar(x + w/2, leak_vals, w, color="#C44E52", label="泄漏策略")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_title("策略指标对比：泄漏让 Sharpe 虚高、回撤失真", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
for bb, vv in zip(list(b1) + list(b2), true_vals + leak_vals):
    ax.text(bb.get_x() + bb.get_width()/2, vv + (0.01 if vv >= 0 else -0.06),
            f"{vv:.2f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(D, "leak_sharpe_bar.png"), dpi=150, bbox_inches="tight"); plt.close()

print("DONE look-ahead-leakage images")
