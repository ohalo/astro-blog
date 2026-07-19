#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""供应链集中度因子 配图生成 (5 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 合成 150 家公司面板, 每家有一个"客户集中度" HHI_c 与"供应商集中度" HHI_s (0-1)
        HHI = Σ share^2, 用 Dirichlet 抽样构造份额 -> HHI 越高越集中
  * 隐藏"依赖冲击"事件: 在 T=150 附近注入一次对高客户集中度公司的需求冲击(大客户流失)
        以及对高供应商集中度公司的供给冲击(断供/涨价)
  * 公司月收益 r:
        r = MKT + beta*MKT + shock_beta_c * demand_shock * HHI_c   (大客户流失 -> 收入塌)
                              + shock_beta_s * supply_shock * HHI_s  (断供 -> 成本升/停产)
                              + eps
  * 因子组合: 每月按 HHI_c 分 5 组, 空高集中度(组5)/多低集中度(组1) -> 危机期产生正 alpha
  * 横截面回归: ret ~ HHI_c (+ 控制 MKT, size), t 统计量验证"集中度溢价"
  * 机制分解: 高集中度组在冲击期收益显著为负, 低集中度组免疫 -> 因子收益来自危机期
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

SLUG = "supply-chain-concentration-factor"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "red": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "grid": "#DDDDDD", "line": "#999999"}

rng = np.random.default_rng(20260721)
N, T = 150, 240

# ---------- 集中度 HHI ----------
def make_hhi(n_cust, alpha_conc):
    # 份额: Dirichlet, 越集中 alpha 越小; HHI = Σ share^2 衡量集中
    base = rng.dirichlet([alpha_conc]*n_cust, size=N)
    hhi = (base**2).sum(1)                    # 真实 HHI, n_cust 少->跨度大
    return hhi
HHI_c = make_hhi(4, 1.0)                      # 客户集中度(前 4 大客户份额)
HHI_s = make_hhi(4, 0.9)                      # 供应商集中度(前 4 大供应商份额)
size = rng.lognormal(0, 0.6, N)              # 公司规模(控制变量)

# ---------- 市场 & 冲击 ----------
MKT = rng.normal(0.008, 0.04, T-1)
demand_shock = np.zeros(T-1)                  # 大客户流失需求冲击
supply_shock = np.zeros(T-1)                  # 断供供给冲击
shock_win = range(150, 175)
for t in shock_win:
    demand_shock[t] = rng.normal(-0.05, 0.02)
    supply_shock[t] = rng.normal(0.045, 0.02)

# ---------- 公司收益 ----------
beta = rng.normal(1.0, 0.3, N)
sb_c = 3.0                                    # 大客户流失敏感度(危机期放大)
sb_s = 2.4                                    # 断供敏感度(危机期放大)
kappa = 0.009                                 # 持续脆弱性折价(集中度越高, 每期少赚 kappa*HHI_c)
eps = rng.normal(0, 0.018, (N, T-1))          # 特质噪声
R = (MKT[None, :]
     + beta[:, None] * MKT[None, :]
     - kappa * HHI_c[:, None]                 # 持续脆弱性折价(集中度溢价来源)
     + sb_c * HHI_c[:, None] * demand_shock[None, :]
     - sb_s * HHI_s[:, None] * supply_shock[None, :]
     + eps)

# ---------- 分组组合 ----------
order = np.argsort(HHI_c)
n = N // 5
groups = [order[i*n:(i+1)*n] for i in range(5)]
grp_ret = np.array([R[g, :].mean(0) for g in groups])   # (5, T-1)
low = grp_ret[0]; high = grp_ret[4]
scc_factor = low - high
nav_factor = np.cumprod(1 + scc_factor)
nav_mkt = np.cumprod(1 + MKT)

# ---------- 横截面回归(全样本) ----------
def xs_reg(y, Xcols):
    X = np.column_stack([np.ones(N)] + Xcols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = N - X.shape[1]
    se = np.sqrt((resid @ resid) / dof) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    return coef, se
coef, se = xs_reg(R.mean(1), [HHI_c, size])
t_hhi = coef[1] / se[1]

# ---------- 危机期 vs 平静期因子收益 ----------
calm = list(range(0, 150)) + list(range(175, T-1))
crisis = list(range(150, 175))
def ann(r): return (np.prod(1 + r))**(12/len(r)) - 1
ann_calm = ann(scc_factor[calm])
ann_crisis = ann(scc_factor[crisis])

# ---------- 高/低集中度组在冲击期表现 ----------
high_crisis = high[150:175].mean()
low_crisis = low[150:175].mean()

summary = {
    "factor_ann": ann(scc_factor),
    "factor_sharpe": scc_factor.mean()/scc_factor.std()*np.sqrt(12),
    "t_HHI": float(t_hhi),
    "ann_calm": ann_calm, "ann_crisis": ann_crisis,
    "high_crisis_mean": float(high_crisis), "low_crisis_mean": float(low_crisis),
    "hhi_c_range": [float(HHI_c.min()), float(HHI_c.max())],
}
print(json.dumps(summary, indent=2))

# ================= 图 1: HHI 分布 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.hist(HHI_c, bins=25, color=C["net"], alpha=0.8, edgecolor="white")
ax.axvline(HHI_c.mean(), color=C["red"], ls="--", lw=1.5, label=f"均值 {HHI_c.mean():.2f}")
ax.set_title("客户集中度 HHI 分布(合成 150 家, 份额 Dirichlet 抽样)", fontsize=12)
ax.set_xlabel("客户集中度 HHI_c ∈ [1/8, 1]"); ax.set_ylabel("公司数")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/hhi_dist.png"); plt.close(fig)

# ================= 图 2: 冲击期累计收益(高 vs 低集中度) =================
seg = range(140, 185)
nav_high_seg = np.cumprod(1 + high[list(seg)-1]) if False else np.cumprod(1 + np.concatenate([[0], high[140:185]]))
nav_low_seg = np.cumprod(1 + np.concatenate([[0], low[140:185]]))
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(range(141, 186), nav_high_seg[1:], color=C["red"], lw=2, label="高客户集中度组")
ax.plot(range(141, 186), nav_low_seg[1:], color=C["green"], lw=2, label="低客户集中度组")
ax.axvline(150.5, color=C["orange"], ls="--", lw=1.3, label="大客户流失冲击")
ax.set_title("冲击期(150-175m): 高集中度组被重定价, 低集中度组免疫", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("净值(起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/shock_cumret.png"); plt.close(fig)

# ================= 图 3: 因子净值 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(nav_factor, color=C["green"], lw=2, label=f"供应链集中度因子(空高/多低) {ann(scc_factor)*100:.1f}%/yr")
ax.plot(nav_mkt, color=C["line"], lw=1.5, ls="--", label=f"市场 {ann(MKT)*100:.1f}%/yr")
ax.axhline(1.0, color=C["grid"], lw=1)
ax.axvspan(150, 175, color=C["red"], alpha=0.08, label="依赖冲击期")
ax.set_title("供应链集中度因子净值: alpha 主要集中在危机期", fontsize=12)
ax.set_xlabel("月份"); ax.set_ylabel("净值(起点=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/scc_factor_nav.png"); plt.close(fig)

# ================= 图 4: 横截面回归 HHI 系数 =================
fig, ax = plt.subplots(figsize=(9, 4.2))
xs = np.linspace(0, 1, 50)
ax.scatter(HHI_c, R.mean(1), s=18, alpha=0.45, color=C["net"], label="公司月均收益")
ax.plot(xs, coef[0] + coef[1]*xs + coef[2]*size.mean(), color=C["red"], lw=2,
        label=f"HHI 斜率 = {coef[1]:.3f} (t={t_hhi:.2f})")
ax.set_title("横截面: 客户集中度越高, 收益越低(t 显著)", fontsize=12)
ax.set_xlabel("客户集中度 HHI_c"); ax.set_ylabel("月均收益")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C["grid"])
fig.tight_layout(); fig.savefig(f"{OUT}/xs_regression.png"); plt.close(fig)

# ================= 图 5: 五分位年化收益 =================
dec_ann = np.array([ann(grp_ret[k]) for k in range(5)])
fig, ax = plt.subplots(figsize=(9, 4.2))
colors = [C["green"] if i==0 else (C["red"] if i==4 else C["net"]) for i in range(5)]
hhis = np.sort(HHI_c)
ax.bar([f"Q{i+1}\n(HHI {hhis[i*n]:.2f}-{hhis[(i+1)*n-1]:.2f})" for i in range(5)],
       dec_ann*100, color=colors)
for i, v in enumerate(dec_ann*100):
    ax.text(i, v + (0.3 if v>=0 else -1.0), f"{v:.1f}%", ha="center", fontsize=9)
ax.axhline(0, color=C["line"], lw=1)
ax.set_title("五分位年化收益: 客户集中度越高收益越低", fontsize=12)
ax.set_ylabel("年化收益 (%)")
ax.grid(alpha=0.3, color=C["grid"], axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/decile_returns.png"); plt.close(fig)

print("IMAGES_WRITTEN:", os.listdir(OUT))
