#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""碳价格因子 配图生成 (5 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 合成 120 家公司面板, 各自碳强度 ci_i = 排放/营收(吨 CO2e / 千美元), lognormal
  * 碳价隐藏过程 P_t: 几何随机游走, 长期上行 + 波动率 + 偶发跳变(政策冲击)
  * 公司月度收益: r = MKT + beta*MKT + s_i*(dP/P) + 再定价项 + eps
        s_i = -lambda*ci_i  -> 碳价上行时, 高碳强度公司被"转型风险"重定价, 跑输
        再定价项  = -theta*ci_i*max(0,(P_t-P_trigger)/P_trigger)  (政策可信后的持续估值压缩)
  * 碳因子组合: 每月按 ci 分 5 组, 多低强度(组1)/空高强度(组5), 等权
  * 碳 beta: 因子收益对 dP/P 回归, 斜率 = 碳敏感度(应为正: 碳价涨, 多低空高赚钱)
  * 对照: 纯多高强度组合 vs 市场, 展示转型风险是"真实亏损"而非纸面
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "carbon-price-factor"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "red": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "grid": "#DDDDDD", "line": "#999999"}

rng = np.random.default_rng(20260720)
N, T = 120, 240                      # 120 公司, 240 月(20 年)

# ---------- 碳强度 ----------
ci = rng.lognormal(mean=-0.5, sigma=0.7, size=N)
ci = np.clip(ci, 0.03, 4.0)

# ---------- 碳价 P_t ----------
P = np.zeros(T)
P[0] = 15.0
shocks = {48: -6.0, 120: 12.0, 175: -4.0}   # 政策冲击(负=回调, 正=收紧跳升)
for t in range(1, T):
    g = rng.normal(0.006, 0.05)             # 长期上行 drift + 波动
    if t in shocks:
        g += shocks[t] / 15.0 * 0.15
    P[t] = max(2.0, P[t-1] * np.exp(g))
dP = np.diff(P) / P[:-1]                     # 月度碳价变化率

# ---------- 市场因子 ----------
MKT = rng.normal(0.008, 0.04, T-1)

# ---------- 公司收益 ----------
beta = rng.normal(1.0, 0.3, N)
lam = 2.2                                   # 碳价敏感度系数(负向)
theta = 0.0009                              # 再定价系数
P_trigger = 45.0                            # 政策可信阈值
rerating = np.array([theta * ci[i] * np.maximum(0.0, (P[1:] - P_trigger) / P_trigger) for i in range(N)])  # (N,T-1)
sig_e = 0.03 + 0.01 * ci / ci.max()         # 高碳强度公司特质波动略高
eps = rng.normal(0, 1, (N, T-1)) * sig_e[:, None]
s_i = -lam * ci                             # 碳价变化敏感度(负)
R = MKT[None, :] + beta[:, None] * MKT[None, :] + s_i[:, None] * dP[None, :] - rerating + eps

# ---------- 碳因子组合(多低空高) ----------
def qtile_portfolio(weights_sort, ret):
    order = np.argsort(ci)
    cs = ci[order]
    n = N // 5
    groups = [order[i*n:(i+1)*n] for i in range(5)]
    grp_ret = np.array([ret[g, :].mean(0) for g in groups])   # (5, T-1)
    return grp_ret, groups, order

grp_ret, groups, order = qtile_portfolio(None, R)
low = grp_ret[0]                            # 低强度组
high = grp_ret[4]                           # 高强度组
carbon_factor = low - high                 # 碳因子(多低空高)
high_only = high.copy()
mkt_only = MKT.copy()

# 净值
nav_factor = np.cumprod(1 + carbon_factor)
nav_high = np.cumprod(1 + high_only)
nav_mkt = np.cumprod(1 + mkt_only)

# ---------- 碳 beta 回归 ----------
X = np.column_stack([np.ones(T-1), dP, MKT])
coef, *_ = np.linalg.lstsq(X, carbon_factor, rcond=None)
carbon_beta = coef[1]
mkt_load = coef[2]

# ---------- 分位年化收益 ----------
def ann(r): return (np.prod(1 + r))**(12/len(r)) - 1
dec_ann = np.array([ann(grp_ret[k]) for k in range(5)])

# ---------- 统计 ----------
summary = {
    "carbon_factor_ann": ann(carbon_factor),
    "carbon_factor_sharpe": carbon_factor.mean() / carbon_factor.std() * np.sqrt(12),
    "high_only_ann": ann(high_only),
    "mkt_ann": ann(mkt_only),
    "carbon_beta": carbon_beta,
    "mkt_load": mkt_load,
    "low_ann": ann(low),
    "P0": float(P[0]), "PT": float(P[-1]),
    "dec_ann": [float(x) for x in dec_ann],
}
print(json.dumps(summary, indent=2))

# ================= 图 1: 碳价路径 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(P, color=C["net"], lw=2)
ax.axhline(P_trigger, color=C["red"], ls="--", lw=1.2, label=f"政策可信阈值 P={P_trigger:.0f}")
ax.set_title("全球碳价路径(合成 EU-ETS 式, 长期上行 + 政策冲击)", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("碳价 ($/吨 CO₂e)")
ax.fill_between(range(T), 0, P, color=C["net"], alpha=0.10)
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/carbon_price.png"); plt.close(fig)

# ================= 图 2: 碳强度 vs 累计收益(再定价) =================
firm_total = np.prod(1 + R, axis=1) - 1
fig, ax = plt.subplots(figsize=(9, 4.2))
sc = ax.scatter(ci, firm_total*100, c=firm_total*100, cmap="RdYlGn", s=28, alpha=0.8)
ax.set_title("碳强度越高, 20 年累计收益越低(转型风险再定价)", fontsize=12)
ax.set_xlabel("碳强度 ci = 排放/营收 (吨 CO₂e / 千美元)")
ax.set_ylabel("累计收益 (%)")
ax.grid(alpha=0.3, color=C["grid"])
cb = fig.colorbar(sc, ax=ax); cb.set_label("累计收益 (%)")
fig.tight_layout(); fig.savefig(f"{OUT}/ci_vs_return.png"); plt.close(fig)

# ================= 图 3: 碳因子净值 vs 高强度纯多 vs 市场 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(nav_factor, color=C["green"], lw=2, label=f"碳因子(多低/空高) {ann(carbon_factor)*100:.1f}%/yr")
ax.plot(nav_high, color=C["red"], lw=1.8, label=f"纯多高强度 {ann(high_only)*100:.1f}%/yr")
ax.plot(nav_mkt, color=C["line"], lw=1.5, ls="--", label=f"市场 {ann(mkt_only)*100:.1f}%/yr")
ax.axhline(1.0, color=C["grid"], lw=1)
ax.set_title("碳因子净值: 把气候风险变成可交易 alpha", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("净值(起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/carbon_factor_nav.png"); plt.close(fig)

# ================= 图 4: 碳 beta 回归 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.scatter(dP, carbon_factor, s=18, alpha=0.5, color=C["net"])
xs = np.linspace(dP.min(), dP.max(), 50)
ax.plot(xs, coef[0] + carbon_beta*xs, color=C["red"], lw=2,
        label=f"碳 beta = {carbon_beta:.2f}")
ax.set_title("碳因子收益 ~ 碳价变化: 斜率为碳敏感度(正)", fontsize=12)
ax.set_xlabel("碳价月度变化 dP/P"); ax.set_ylabel("碳因子月收益")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/carbon_beta_reg.png"); plt.close(fig)

# ================= 图 5: 五分位年化收益 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
colors = [C["green"] if i==0 else (C["red"] if i==4 else C["net"]) for i in range(5)]
ax.bar([f"Q{i+1}\n(ci={np.sort(ci)[i*N//5]:.2f}-{np.sort(ci)[(i+1)*N//5-1]:.2f})" for i in range(5)],
       dec_ann*100, color=colors)
for i, v in enumerate(dec_ann*100):
    ax.text(i, v + (0.3 if v>=0 else -1.0), f"{v:.1f}%", ha="center", fontsize=9)
ax.axhline(0, color=C["line"], lw=1)
ax.set_title("五分位组合年化收益: 碳强度单调(低→高 收益降)", fontsize=12)
ax.set_ylabel("年化收益 (%)")
ax.grid(alpha=0.3, color=C["grid"], axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/decile_returns.png"); plt.close(fig)

print("IMAGES_WRITTEN:", os.listdir(OUT))
