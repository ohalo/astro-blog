#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「跨资产相关性危机：恐慌时相关性冲向 1 的组合含义」
(cross-asset-correlation-crisis) 生成真实配图与核心数值。

机制（自洽合成，仅用于演示方法，固定 seed 可复现）：
  * 5 资产组合：股票 / 商品 / REITs / 新兴市场 / 国债(对冲)。
    用「单因子 + 特异波动」模型：r_i = beta_i * F_t + epsilon_{i,t}。
    - 前 4 个风险资产 beta 为正（0.85~1.10）：危机期共同因子波动爆炸，
      全部被同一因子驱动 → 相关性冲向 1；
    - 国债 beta≈0.05、特异波动低：近似「独立对冲资产」，危机期仍独立。
  * 实证四层：
      ① 相关矩阵热力图：风险资产子集的相关从平静期低/分化 → 危机期全红；
      ② 分散比率 diversification ratio = Σw_i σ_i / σ_p：危机期塌缩；
      ③ 组合波动：风险资产 4 组合 vs 全 5 资产(含国债)对比；
      ④ 蒙特卡洛 2500 路径最大回撤：风险资产组合 vs 含国债组合 vs 60/40。
  * 关键结论：相关性危机 = 相关结构塌缩（不是单纯波动变大）。分散化红利在
    你需要它（危机）时蒸发，且只有真正「beta 独立」的资产（国债）还能对冲。

所有数值与图表均由本脚本 numpy 真实计算生成。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for f in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["figure.autolayout"] = True

SLUG = "cross-asset-correlation-crisis"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"vm": "#4C72B0", "st": "#C44E52", "dd": "#8172B3", "eq": "#DD8452",
     "bd": "#CCB974", "dark": "#333333", "gold": "#DD8452", "blue": "#4C72B0",
     "green": "#55A868", "red": "#C44E52", "purple": "#8172B3", "orange": "#E8A33D"}

NAMES = ["股票", "商品", "REITs", "新兴市场", "国债(对冲)"]
n = len(NAMES)
sigma_idio = np.array([0.14, 0.16, 0.13, 0.20, 0.05])   # 年化特异波动
beta = np.array([1.00, 0.85, 0.90, 1.10, 0.05])          # 前4风险资产正, 国债≈0
mu_ann = np.array([0.08, 0.05, 0.06, 0.09, 0.025])       # 年化漂移
sigma_F_calm, sigma_F_crisis = 0.05, 0.28                # 共同因子年化波动

T_total = 252 * 10
rng = np.random.default_rng(20261007)
regime = np.zeros(T_total, dtype=int)
pos = 0
while pos < T_total:
    if rng.random() < 0.12 and pos < T_total - 40:
        L = int(rng.integers(20, 40))
        regime[pos:pos + L] = 1
        pos += L + int(rng.integers(30, 90))
    else:
        pos += 1
F = np.where(regime == 0,
             rng.normal(0, sigma_F_calm / np.sqrt(252), T_total),
             rng.normal(0, sigma_F_crisis / np.sqrt(252), T_total))
eps = rng.normal(0, 1, (n, T_total)) * (sigma_idio / np.sqrt(252))[:, None]
R = beta[:, None] * F + eps + (mu_ann / 252)[:, None]

calm_mask = regime == 0
crisis_mask = regime == 1
corr_calm = np.corrcoef(R[:, calm_mask])
corr_crisis = np.corrcoef(R[:, crisis_mask])

# ① 风险资产子集(前4个)相关
risky = [0, 1, 2, 3]
off = ~np.eye(4, dtype=bool)
avg_risky_calm = corr_calm[np.ix_(risky, risky)][off].mean()
avg_risky_crisis = corr_crisis[np.ix_(risky, risky)][off].mean()

# ===================== 图1：相关矩阵热力图 =====================
def heat(ax, M, title, only_risky=False):
    idx = risky if only_risky else list(range(n))
    Mm = M[np.ix_(idx, idx)]
    im = ax.imshow(Mm, vmin=-0.2, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(idx))); ax.set_yticks(range(len(idx)))
    ax.set_xticklabels([NAMES[i] for i in idx], fontsize=8)
    ax.set_yticklabels([NAMES[i] for i in idx], fontsize=8)
    for a in range(len(idx)):
        for b in range(len(idx)):
            ax.text(b, a, f"{Mm[a, b]:.2f}", ha="center", va="center",
                    color="white" if abs(Mm[a, b]) > 0.55 else "black", fontsize=8)
    ax.set_title(title, fontsize=10)
    return im

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
heat(axes[0], corr_calm, "平静期：风险资产相关性低且分化")
im = heat(axes[1], corr_crisis, "危机期：风险资产相关性冲高、全红")
fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
fig.suptitle("相关性危机：风险资产间 correlation 冲向 1（仅国债仍独立）", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "corr_matrix.png")); plt.close(fig)

# ===================== 图2：分散比率 & 组合波动 =====================
w5 = np.ones(n) / n
w4 = np.array([0.25, 0.25, 0.25, 0.25, 0.0])
def port_vol(cov, w):
    return np.sqrt(w @ cov @ w)
cov_calm = np.cov(R[:, calm_mask]); cov_crisis = np.cov(R[:, crisis_mask])
sig5_c = np.sqrt(np.diag(cov_calm)) * np.sqrt(252)
sig5_k = np.sqrt(np.diag(cov_crisis)) * np.sqrt(252)
pv5_calm = port_vol(cov_calm, w5) * np.sqrt(252)
pv5_crisis = port_vol(cov_crisis, w5) * np.sqrt(252)
pv4_calm = port_vol(cov_calm, w4) * np.sqrt(252)
pv4_crisis = port_vol(cov_crisis, w4) * np.sqrt(252)
w4r = np.array([0.25, 0.25, 0.25, 0.25])   # 仅风险资产 4 权重
cov_calm4 = cov_calm[np.ix_(risky, risky)]
cov_crisis4 = cov_crisis[np.ix_(risky, risky)]
pv4_calm = np.sqrt(w4r @ cov_calm4 @ w4r) * np.sqrt(252)
pv4_crisis = np.sqrt(w4r @ cov_crisis4 @ w4r) * np.sqrt(252)
div5_calm = (w5 * sig5_c).sum() / pv5_calm
div5_crisis = (w5 * sig5_k).sum() / pv5_crisis
div4_calm = (w4r * sig5_c[risky]).sum() / pv4_calm
div4_crisis = (w4r * sig5_k[risky]).sum() / pv4_crisis

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
xs = np.arange(2)
axes[0].bar(xs - 0.18, [pv4_calm * 100, pv4_crisis * 100], width=0.36,
            color=C["red"], label="风险4资产 组合波动 σ_p")
axes[0].bar(xs + 0.18, [pv5_calm * 100, pv5_crisis * 100], width=0.36,
            color=C["blue"], label="含国债5资产 组合波动 σ_p")
axes[0].set_xticks(xs); axes[0].set_xticklabels(["平静期", "危机期"])
axes[0].set_ylabel("年化波动 (%)")
axes[0].set_title("危机期组合波动翻数倍（风险4资产更惨）")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
xs2 = np.arange(2)
axes[1].bar(xs2 - 0.18, [div4_calm, div5_calm], width=0.36,
            color=C["green"], label="平静期分散比率")
axes[1].bar(xs2 + 0.18, [div4_crisis, div5_crisis], width=0.36,
            color=C["red"], label="危机期分散比率")
axes[1].axhline(1.0, color=C["dark"], ls="--", lw=1.0)
axes[1].set_xticks(xs2); axes[1].set_xticklabels(["风险4资产", "含国债5资产"])
axes[1].set_ylabel("分散比率 = Σwσ / σ_p")
axes[1].set_title(f"分散比率塌缩：风险4 {div4_calm:.2f}→{div4_crisis:.2f}")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "diversification.png")); plt.close(fig)

# ===================== 图3：累计净值（含危机阴影）=====================
eq5 = np.insert(np.cumprod(1 + (R * w5[:, None]).sum(axis=0)), 0, 1.0)
eq4 = np.insert(np.cumprod(1 + (R * w4[:, None]).sum(axis=0)), 0, 1.0)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.plot(eq5, color=C["blue"], lw=1.4, label="含国债 5 资产等权")
ax.plot(eq4, color=C["red"], lw=1.4, label="风险 4 资产等权")
for s in np.where(regime == 1)[0]:
    ax.axvline(s, color=C["red"], alpha=0.05, lw=0.5)
ax.set_xlabel("交易日")
ax.set_ylabel("累计净值")
ax.set_title("危机段（红）风险资产齐跌，国债独自撑住净值")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "equity_curve.png")); plt.close(fig)

# ===================== 蒙特卡洛：最大回撤分布 =====================
M = 2500
days = 252
w6040 = np.array([0.6, 0.0, 0.0, 0.0, 0.4])
def sim_dd(seed, weights):
    rg = np.random.default_rng(seed)
    reg = np.zeros(days, dtype=int)
    p = 0
    while p < days:
        if rg.random() < 0.14 and p < days - 25:
            L = int(rg.integers(20, 35)); reg[p:p + L] = 1
            p += L + int(rg.integers(30, 80))
        else:
            p += 1
    Fs = np.where(reg == 0, rg.normal(0, sigma_F_calm / np.sqrt(252), days),
                  rg.normal(0, sigma_F_crisis / np.sqrt(252), days))
    epss = rg.normal(0, 1, (n, days)) * (sigma_idio / np.sqrt(252))[:, None]
    Rs = beta[:, None] * Fs + epss + (mu_ann / 252)[:, None]
    ret = (Rs * weights[:, None]).sum(axis=0)
    eq = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(eq)
    return ((eq - peak) / peak).min()

dd4 = np.array([sim_dd(5000 + m, w4) for m in range(M)])
dd5 = np.array([sim_dd(8000 + m, w5) for m in range(M)])
dd6040 = np.array([sim_dd(11000 + m, w6040) for m in range(M)])
mean4, p95_4 = dd4.mean(), np.percentile(dd4, 95)
mean5, p95_5 = dd5.mean(), np.percentile(dd5, 95)
mean60, p95_60 = dd6040.mean(), np.percentile(dd6040, 95)

fig, ax = plt.subplots(figsize=(7.6, 4.4))
bins = np.linspace(-0.6, 0.0, 45)
ax.hist(dd4, bins=bins, color=C["red"], alpha=0.6, label=f"风险4资产 P95={p95_4:.0%}")
ax.hist(dd5, bins=bins, color=C["blue"], alpha=0.6, label=f"含国债5资产 P95={p95_5:.0%}")
ax.hist(dd6040, bins=bins, color=C["orange"], alpha=0.55, label=f"60/40 P95={p95_60:.0%}")
ax.set_xlabel("年度最大回撤")
ax.set_ylabel("路径数")
ax.set_title(f"危机相关下：风险4资产 P95 {p95_4:.0%} vs 含国债 {p95_5:.0%} vs 60/40 {p95_60:.0%}")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "drawdown_dist.png")); plt.close(fig)

# ===================== 图5：滚动分散比率时间序列 =====================
win = 60
roll4, roll5 = [], []
for t0 in range(0, T_total - win, 5):
    sub = R[:, t0:t0 + win]
    cov = np.cov(sub)
    s4 = np.sqrt(np.diag(cov))[:4] * np.sqrt(252)
    pv4 = np.sqrt(w4[:4] @ cov[:4, :4] @ w4[:4]) * np.sqrt(252)
    roll4.append((s4 * w4[:4]).sum() / pv4)
    s5 = np.sqrt(np.diag(cov)) * np.sqrt(252)
    pv5 = np.sqrt(w5 @ cov @ w5) * np.sqrt(252)
    roll5.append((s5 * w5).sum() / pv5)
roll4, roll5 = np.array(roll4), np.array(roll5)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.plot(roll4, color=C["red"], lw=0.9, alpha=0.8, label=f"风险4资产 均值 {roll4.mean():.2f}")
ax.plot(roll5, color=C["blue"], lw=0.9, alpha=0.8, label=f"含国债5资产 均值 {roll5.mean():.2f}")
ax.axhline(1.0, color=C["dark"], ls="--", lw=1.0)
ax.set_xlabel("滚动窗口起点（每 5 日）")
ax.set_ylabel("分散比率 (滚动 60 日)")
ax.set_title("滚动分散比率：危机来临时骤降，分散红利蒸发")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "rolling_div.png")); plt.close(fig)

print("=" * 64)
print("ARTICLE_CORR_CRISIS_METRICS")
print(f"资产: {NAMES}")
print(f"[风险资产相关性] 平静={avg_risky_calm:.2f} 危机={avg_risky_crisis:.2f}")
print(f"[组合波动] 风险4资产 平静={pv4_calm*100:.1f}% 危机={pv4_crisis*100:.1f}%")
print(f"[组合波动] 含国债5资产 平静={pv5_calm*100:.1f}% 危机={pv5_crisis*100:.1f}%")
print(f"[分散比率] 风险4资产 {div4_calm:.2f}→{div4_crisis:.2f}  含国债5 {div5_calm:.2f}→{div5_crisis:.2f}")
print(f"[最大回撤 MC{M}] 风险4资产 P95={p95_4:.0%} (均值 {mean4:.0%})")
print(f"[最大回撤 MC{M}] 含国债5资产 P95={p95_5:.0%} (均值 {mean5:.0%})")
print(f"[最大回撤 MC{M}] 60/40 P95={p95_60:.0%} (均值 {mean60:.0%})")
print("=" * 64)
