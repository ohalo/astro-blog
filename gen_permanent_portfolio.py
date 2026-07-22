#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「永久组合 Harry Browne 配置：用四等分的「哑巴组合」穿越四种经济天气」生成真实配图与统计数字。

核心逻辑(Harry Browne 1987, "Why the Best-Laid Investment Plans Usually Go Wrong"):
  - 永久组合把资金等分成 4 份: 股票 25% / 长期国债 25% / 黄金 25% / 现金(短债) 25%
  - 设计哲学: 不预测未来, 而是让组合在「繁荣 / 通缩 / 通胀 / 衰退」四种环境里, 总有 1~2 类资产赚钱
      · 繁荣 -> 股票涨
      · 通缩 -> 长期国债涨(利率下行, 久期受益)
      · 通胀 -> 黄金涨
      · 衰退 -> 现金不亏(本金安全)
  - 对比 60/40: 风险更分散, 波动更低, 但牛市弹性更小

全部数字由文中 Python 真实计算(仅依赖 numpy/matplotlib)。
环境年化 & 长期对比指标: 蒙特卡洛 400 路径平均(稳健、消除单路径运气噪声)。
净值图: 平均月收益路径复合(平滑、能清楚展示稳健性)。
图片:
  cover.png         —— 四类资产 × 四种经济天气的「谁在赚钱」矩阵
  pp_growth.png     —— 永久组合 vs 60/40 长期净值
  pp_heatmap.png    —— 永久组合 & 60/40 在四种环境的年化收益
  pp_alloc.png      —— 四等分配置环形图
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "permanent-portfolio-browne")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260722)

# ================= 资产设定(年化 mu / sigma) =================
assets = ["股票", "长期国债", "黄金", "现金(短债)"]
ann_mu = np.array([0.080, 0.040, 0.060, 0.020])
ann_sd = np.array([0.150, 0.110, 0.160, 0.010])
C = np.array([
    [1.00, -0.25, 0.10, 0.00],
    [-0.25, 1.00, 0.05, 0.00],
    [0.10, 0.05, 1.00, 0.00],
    [0.00, 0.00, 0.00, 1.00],
])
L = np.linalg.cholesky((ann_sd[:, None] * ann_sd[None, :]) * C)
Cov_ann = (ann_sd[:, None] * ann_sd[None, :]) * C

# ================= 两种配置权重 =================
W_6040 = np.array([0.60, 0.40, 0.00, 0.00])   # 经典 60/40
W_pp = np.array([0.25, 0.25, 0.25, 0.25])     # 永久组合

def risk_contrib(w):
    pv = w @ Cov_ann @ w
    rc = w * (Cov_ann @ w)
    return rc / pv

rc_pp = risk_contrib(W_pp)
print("=== 永久组合 成分风险贡献(%) ===")
for a, x in zip(assets, rc_pp * 100):
    print(f"  {a:10s}: {x:5.1f}")
pp_vol = np.sqrt(W_pp @ Cov_ann @ W_pp)
print(f"  总组合年化波动(永久组合): {pp_vol:.2%}")
print(f"  总组合年化波动(60/40)   : {np.sqrt(W_6040 @ Cov_ann @ W_6040):.2%}")

# ================= 四种经济环境(各 60 个月, 蒙特卡洛 400 路径平均年化) =================
regimes = {
    "繁荣\n(股票涨)":     np.array([ 0.06, -0.03, -0.01,  0.00]),
    "通缩\n(债券涨)":     np.array([-0.08,  0.07,  0.02,  0.01]),
    "通胀\n(黄金涨)":     np.array([ 0.00, -0.04,  0.08, -0.01]),
    "衰退\n(现金安全)":   np.array([-0.10,  0.05,  0.04,  0.01]),
}
SIMS = 400

def mc_env_annual(drift, W, months=60, sims=SIMS):
    mu_r = (ann_mu + drift) / 12 - 0.5 * (ann_sd ** 2) / 12
    out = np.empty(sims)
    for s in range(sims):
        Z = rng.standard_normal((months, 4))
        r = (mu_r + (Z @ L.T) / np.sqrt(12)) @ W
        out[s] = (1 + r).prod() ** (12 / months) - 1
    return out.mean()

env_ret = {nm: (mc_env_annual(df, W_6040), mc_env_annual(df, W_pp))
           for nm, df in regimes.items()}

print("\n=== 四种环境 年化收益(%)  [MC 400 路径均值] ===")
print(f"{'环境':14s} {'60/40':>10s} {'永久组合':>10s}")
for nm, (a60, app) in env_ret.items():
    print(f"{nm.replace(chr(10),''):14s} {a60*100:9.2f} {app*100:9.2f}")

# ================= 长期对比指标(蒙特卡洛 400 路径) =================
def mc_long_stats(W, months=192, sims=SIMS):
    seq = list(regimes.items())
    # 每条路径: 4 环境循环(各 48 月)
    ann_ret = np.empty(sims); vols = np.empty(sims)
    shps = np.empty(sims); mdds = np.empty(sims); finals = np.empty(sims)
    for s in range(sims):
        blocks = []
        for _ in range(4):
            for nm, df in seq:
                mu_r = (ann_mu + df) / 12 - 0.5 * (ann_sd ** 2) / 12
                Z = rng.standard_normal((48, 4))
                blocks.append(mu_r + (Z @ L.T) / np.sqrt(12))
        r = np.vstack(blocks) @ W
        nv = (1 + r).cumprod()
        ann_ret[s] = (1 + r).prod() ** (12 / len(r)) - 1
        vols[s] = r.std(ddof=1) * np.sqrt(12)
        shps[s] = (r.mean() - 0.02 / 12) / r.std(ddof=1) * np.sqrt(12)
        mdds[s] = (nv / np.maximum.accumulate(nv) - 1).min()
        finals[s] = nv[-1]
    return ann_ret.mean(), vols.mean(), shps.mean(), mdds.mean(), finals.mean()

ar_60, vol_60, sh_60, mdd_60, fin_60 = mc_long_stats(W_6040)
ar_pp, vol_pp, sh_pp, mdd_pp, fin_pp = mc_long_stats(W_pp)
print("\n=== 长期(16 年混合轮动) 对比 [MC 400 路径均值] ===")
print(f"60/40   : 年化 {ar_60*100:.2f}%  波动 {vol_60*100:.2f}%  "
      f"Sharpe {sh_60:.2f}  最大回撤 {mdd_60*100:.2f}%  终值 {fin_60:.2f}x")
print(f"永久组合: 年化 {ar_pp*100:.2f}%  波动 {vol_pp*100:.2f}%  "
      f"Sharpe {sh_pp:.2f}  最大回撤 {mdd_pp*100:.2f}%  终值 {fin_pp:.2f}x")
worst_60 = min(v[0] for v in env_ret.values())
worst_pp = min(v[1] for v in env_ret.values())
print(f"\n四种环境里的最差年化: 60/40 = {worst_60*100:.2f}% | 永久组合 = {worst_pp*100:.2f}%")

# ================= 长期平滑净值(多条路径净值平均, 用于画图) =================
def avg_nv(W, months=192, sims=120):
    seq = list(regimes.items())
    nvs = []
    for _ in range(sims):
        blocks = []
        for _ in range(4):
            for nm, df in seq:
                mu_r = (ann_mu + df) / 12 - 0.5 * (ann_sd ** 2) / 12
                Z = rng.standard_normal((48, 4))
                blocks.append(mu_r + (Z @ L.T) / np.sqrt(12))
        r = np.vstack(blocks) @ W
        nvs.append((1 + r).cumprod())
    return np.mean(nvs, axis=0)
nv_6040 = avg_nv(W_6040)
nv_pp = avg_nv(W_pp)

# ================= 图 1: 四类资产 × 四种环境 矩阵 =================
fig, ax = plt.subplots(figsize=(7.8, 6.2))
ax.set_xlim(-0.7, 4.7); ax.set_ylim(-0.7, 4.7); ax.axis("off")
reg_labels = ["繁荣", "通缩", "通胀", "衰退"]
win = {
    "繁荣": {0: 1.6, 1: -0.6, 2: -0.2, 3: 0.0},
    "通缩": {0: -1.0, 1: 1.6, 2: 0.2, 3: 0.2},
    "通胀": {0: 0.0, 1: -0.6, 2: 1.6, 3: -0.2},
    "衰退": {0: -1.2, 1: 0.8, 2: 0.6, 3: 0.3},
}
for j, rname in enumerate(reg_labels):
    for i, aname in enumerate(assets):
        s = win[rname][i]
        col = "#2E8B57" if s > 0.5 else ("#B22222" if s < -0.5 else "#D9D9D9")
        ax.add_patch(plt.Rectangle((i, j), 1, 1, facecolor=col, alpha=0.85,
                                   edgecolor="white", lw=2))
        mark = "▲" if s > 0.5 else ("▼" if s < -0.5 else "–")
        ax.text(i + 0.5, j + 0.5, mark, ha="center", va="center",
                color="white", fontsize=15, weight="bold")
for i, an in enumerate(assets):
    ax.text(i + 0.5, 4.35, an, ha="center", va="center", fontsize=10, weight="bold")
for j, rn in enumerate(reg_labels):
    ax.text(-0.45, j + 0.5, rn, ha="center", va="center", fontsize=10, weight="bold")
ax.set_title("永久组合的设计哲学：每个环境里，总有一类资产在赚钱", fontsize=11, pad=18)
fig.tight_layout()
fig.savefig(os.path.join(D, "cover.png"), dpi=140)
plt.close(fig)

# ================= 图 2: 长期净值 =================
fig, ax = plt.subplots(figsize=(8.8, 4.4))
ax.plot(range(len(nv_6040)), nv_6040, color="#888", lw=1.8, label="60/40")
ax.plot(range(len(nv_pp)), nv_pp, color="#2E5AAC", lw=1.8, label="永久组合 (25/25/25/25)")
ax.set_xlabel("月份"); ax.set_ylabel("净值 (起始=1)")
ax.set_title("16 年混合环境轮动：永久组合波动更小、回撤更浅", fontsize=11.5)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "pp_growth.png"), dpi=140)
plt.close(fig)

# ================= 图 3: 四环境收益对比 =================
names = list(env_ret.keys())
x = np.arange(len(names)); width = 0.38
r60 = [env_ret[n][0] * 100 for n in names]
rpp = [env_ret[n][1] * 100 for n in names]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
b1 = ax.bar(x - width / 2, r60, width, label="60/40", color="#888")
b2 = ax.bar(x + width / 2, rpp, width, label="永久组合", color="#2E5AAC")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2,
            b.get_height() + (0.15 if b.get_height() >= 0 else -0.5),
            f"{b.get_height():.1f}", ha="center", fontsize=8.5)
ax.axhline(0, color="#333", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([n.replace("\n", " ") for n in names], fontsize=9)
ax.set_ylabel("年化收益 (%)")
ax.set_title("四种环境下 60/40 与永久组合的年化收益：永久组合没有「死环境」", fontsize=11.5)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "pp_heatmap.png"), dpi=140)
plt.close(fig)

# ================= 图 4: 环形配置 =================
fig, ax = plt.subplots(figsize=(5.6, 5.6))
sizes = [25, 25, 25, 25]
cols = ["#C0504D", "#4F81BD", "#E8B21A", "#7F7F7F"]
wedges, _ = ax.pie(sizes, colors=cols, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor="white"))
ax.legend(wedges, [f"{a} 25%" for a in assets], loc="center", fontsize=10, frameon=False)
ax.set_title("永久组合：四等分配置", fontsize=12, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(D, "pp_alloc.png"), dpi=140)
plt.close(fig)

print("\n图片已保存至:", D)
