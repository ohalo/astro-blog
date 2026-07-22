#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「桥水全天候风险平价：把「四种经济天气」都装进一个组合里」生成真实配图与统计数字。

核心逻辑(Bridgewater All Weather, Dalio 1996):
  - 传统 60/40: 60% 股票 + 40% 债券, 但风险几乎全压在股票上(股票贡献 ~90% 波动)
  - 风险平价(Risk Parity): 按「对组合风险的贡献」而非资金额分配, 谁波动大谁少配
  - 全天候再进一步: 资产选择覆盖「增长↑/↓ × 通胀↑/↓」四种环境, 让任一种天气都有资产赚钱
  - 经典配置: 股票30% / 长债40% / 中债15% / 黄金7.5% / 商品7.5%
  - 对照 60/40 vs 全天候: 四环境年化收益、成分风险贡献、长期净值

全部数字由文中 Python 真实计算(仅依赖 numpy/matplotlib)。
图片:
  cover.png                  —— 四种经济环境象限 + 全天候资产轮动
  aw_regime.png              —— 60/40 vs 全天候 在四种环境的平均年化收益
  aw_risk_contrib.png        —— 成分风险贡献: 60/40 股票主导 vs 全天候均衡
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
D = os.path.join(BASE, "all-weather-risk-parity")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260722)

# ================= 资产设定(年化 mu / sigma) =================
assets = ["股票", "长债(20Y+)", "中债(7Y)", "黄金", "商品"]
ann_mu = np.array([0.075, 0.045, 0.035, 0.040, 0.050])
ann_sd = np.array([0.150, 0.105, 0.050, 0.160, 0.180])
C = np.array([
    [1.00, -0.20, -0.10, 0.10, 0.25],
    [-0.20, 1.00, 0.80, 0.25, -0.10],
    [-0.10, 0.80, 1.00, 0.15, -0.05],
    [0.10, 0.25, 0.15, 1.00, 0.45],
    [0.25, -0.10, -0.05, 0.45, 1.00],
])
L = np.linalg.cholesky((ann_sd[:, None] * ann_sd[None, :]) * C)

# ================= 两种配置权重 =================
W_6040 = np.array([0.60, 0.0, 0.40, 0.0, 0.0])
W_aw = np.array([0.30, 0.40, 0.15, 0.075, 0.075])

# 成分风险贡献(年化)
Cov_ann = (ann_sd[:, None] * ann_sd[None, :]) * C
def risk_contrib(w):
    pv = w @ Cov_ann @ w
    rc = w * (Cov_ann @ w)
    return rc / pv, np.sqrt(pv)
rc_6040, _ = risk_contrib(W_6040)
rc_aw, _ = risk_contrib(W_aw)
print("=== 成分风险贡献(%) ===")
print("60/40 :", np.round(rc_6040 * 100, 1))
print("全天候:", np.round(rc_aw * 100, 1))

# ================= 四种经济环境(各 60 个月) =================
# 环境对各类资产的超额漂移(年化, 叠加到基础 mu 上)
regimes = {
    "增长↑ 通胀↓\n(黄金期)":   np.array([ 0.04,  0.02,  0.01, -0.01, -0.02]),
    "增长↑ 通胀↑\n(过热)":     np.array([ 0.02, -0.03, -0.01,  0.03,  0.05]),
    "增长↓ 通胀↓\n(衰退)":     np.array([-0.10,  0.06,  0.03,  0.02, -0.03]),
    "增长↓ 通胀↑\n(滞胀)":     np.array([-0.08, -0.02,  0.00,  0.05,  0.06]),
}
env_ret = {}
for name, drift in regimes.items():
    mu_r = (ann_mu + drift) / 12 - 0.5 * (ann_sd**2) / 12
    Z = rng.standard_normal((60, 5))
    monthly = mu_r + (Z @ L.T) / np.sqrt(12)
    env_ret[name] = (monthly @ W_6040, monthly @ W_aw)

print("\n=== 四种环境 年化收益(%) ===")
print(f"{'环境':22s} {'60/40':>10s} {'全天候':>10s}")
for name, (r60, raw) in env_ret.items():
    a60 = (1 + r60).prod() ** (12/60) - 1
    aaw = (1 + raw).prod() ** (12/60) - 1
    print(f"{name.replace(chr(10),' '):22s} {a60*100:9.2f} {aaw*100:9.2f}")

# 长期净值(BB 风格: 混合环境轮动, 每种 48 个月循环)
seq = list(regimes.items())
blocks = []
for _ in range(4):
    for nm, drift in seq:
        mu_r = (ann_mu + drift) / 12 - 0.5 * (ann_sd**2) / 12
        Z = rng.standard_normal((48, 5))
        blocks.append(mu_r + (Z @ L.T) / np.sqrt(12))
monthly_all = np.vstack(blocks)
nv_6040 = (1 + monthly_all @ W_6040).cumprod()
nv_aw = (1 + monthly_all @ W_aw).cumprod()
def ann(r): return (1+r).prod()**(12/len(r)) - 1
def vol(r): return r.std(ddof=1)*np.sqrt(12)
def sharpe(r, rf=0.02): return (r.mean()-rf/12)/(r.std(ddof=1))*np.sqrt(12)
def mdd(c): return (c/np.maximum.accumulate(c)-1).min()
print("\n=== 长期(混合环境轮动 16 年) 对比 ===")
for nm, nv, r in [("60/40", nv_6040, monthly_all@W_6040), ("全天候", nv_aw, monthly_all@W_aw)]:
    print(f"{nm}: 年化 {ann(r)*100:.2f}%  波动 {vol(r)*100:.2f}%  "
          f"Sharpe {sharpe(r):.2f}  最大回撤 {mdd(nv)*100:.2f}%  终值 {nv[-1]:.2f}x")

# ================= 图 1: 四种环境象限 =================
fig, ax = plt.subplots(figsize=(7.6, 6.0))
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.axis("off")
ax.axhline(0, color="#ccc"); ax.axvline(0, color="#ccc")
quad = [
    (1.0, 1.0, "增长↑ 通胀↑\n过热", "#F4A261", "股票↑ 商品/黄金↑ 债券↓"),
    (-1.0, 1.0, "增长↓ 通胀↑\n滞胀", "#E76F51", "商品/黄金↑ 股票↓ 债券↓"),
    (1.0, -1.0, "增长↑ 通胀↓\n黄金期", "#2A9D8F", "股票↑ 债券↑"),
    (-1.0, -1.0, "增长↓ 通胀↓\n衰退", "#577590", "债券↑ 股票↓"),
]
for sx, sy, t, c, sub in quad:
    ax.add_patch(plt.Rectangle((sx-0.95 if sx>0 else sx-0.05-0.9, sy-0.95 if sy>0 else sy-0.05-0.9),
                               0.9, 0.9, fill=True, color=c, alpha=0.85, ec="white", lw=2))
    ax.text(sx - (0.5 if sx>0 else -0.5), sy - (0.5 if sy>0 else -0.5), t,
            ha="center", va="center", color="white", fontsize=11, weight="bold")
    ax.text(sx - (0.5 if sx>0 else -0.5), sy - (0.72 if sy>0 else -0.72), sub,
            ha="center", va="center", color="white", fontsize=7.5)
ax.text(0, 1.35, "增长", ha="center", fontsize=11, weight="bold")
ax.text(1.45, 0, "通胀 →", ha="center", fontsize=11, weight="bold", rotation=90)
ax.set_title("全天候的底层信念：四种经济天气，每种都要有资产赚钱", fontsize=11, pad=14)
fig.tight_layout()
fig.savefig(os.path.join(D, "cover.png"), dpi=140)
plt.close(fig)

# ================= 图 2: 四环境收益 =================
names = list(env_ret.keys())
x = np.arange(len(names)); width = 0.38
r60 = [((1+env_ret[n][0]).prod()**(12/60)-1)*100 for n in names]
raw = [((1+env_ret[n][1]).prod()**(12/60)-1)*100 for n in names]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
b1 = ax.bar(x - width/2, r60, width, label="60/40", color="#888")
b2 = ax.bar(x + width/2, raw, width, label="全天候风险平价", color="#2E5AAC")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+(0.15 if b.get_height()>=0 else -0.5),
            f"{b.get_height():.1f}", ha="center", fontsize=8.5)
ax.axhline(0, color="#333", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([n.replace("\n", " ") for n in names], fontsize=9)
ax.set_ylabel("年化收益 (%)")
ax.set_title("四种环境下 60/40 与全天候的年化收益：全天候把「最坏环境」的坑填平了", fontsize=11.5)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "aw_regime.png"), dpi=140)
plt.close(fig)

# ================= 图 3: 成分风险贡献 =================
x = np.arange(5)
fig, ax = plt.subplots(figsize=(8.6, 4.4))
b1 = ax.bar(x - width/2, rc_6040*100, width, label="60/40", color="#888")
b2 = ax.bar(x + width/2, rc_aw*100, width, label="全天候", color="#2E5AAC")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.0f}", ha="center", fontsize=8.5)
ax.axhline(20, color="#E76F51", ls="--", lw=1, label="理想风险平价(各 20%)")
ax.set_xticks(x); ax.set_xticklabels(assets, fontsize=9)
ax.set_ylabel("对总风险贡献 (%)")
ax.set_title("成分风险贡献：60/40 几乎被股票绑架，全天候压到接近均衡", fontsize=11.5)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "aw_risk_contrib.png"), dpi=140)
plt.close(fig)

print("\n图片已保存至:", D)
