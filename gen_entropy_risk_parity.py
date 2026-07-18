#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""熵风险平价 配图生成 (4 张真实图表, scipy 求解最小交叉熵 + ERC 约束)

机制(自洽合成, 仅用于演示方法):
  * 真实风格资产协方差: 5 资产, 波动差异显著(股票/REITs 高, 债券低, 黄金/商品居中), 相关性有结构
  * 你有一个先验观点 w0(这里用「市值偏重」的集中先验, 股票 0.40)
  * 熵风险平价: min Σ w_i ln(w_i/w0_i)  s.t. 各资产风险贡献(RC)相等 + Σw=1, w≥0
      -> 它把「先验观点」和「风险平价」融成一个可解释权重, 不硬清零先验
  * 对照: 等权 / 传统风险平价(波动倒数) / 最小方差 / 最大熵(逆协方差) / 熵风险平价
  * 图1: 资产历史模拟路径(为什么需要分散)
  * 图2: 各方法「边际风险贡献 %」柱状 -> 越平越「风险平价」
  * 图3: 各方法权重分布 + 香农熵
  * 图4: 样本外滚动波动率对比
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "entropy-risk-parity"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "erp": "#4C72B0", "eq": "#DD8452", "minvar": "#55A868",
     "rp": "#8172B3", "maxent": "#C44E52", "gold": "#CCB974", "bond": "#9AD3BC", "prior": "#999999"}

rng = np.random.default_rng(20260719)
names = ["股票", "债券", "黄金", "商品", "REITs"]
n_assets = 5
ann_vol = np.array([0.22, 0.05, 0.15, 0.18, 0.25])
corr = np.array([
    [1.00, -0.20, 0.10, 0.30, 0.60],
    [-0.20, 1.00, 0.05, -0.10, -0.15],
    [0.10, 0.05, 1.00, 0.20, 0.10],
    [0.30, -0.10, 0.20, 1.00, 0.35],
    [0.60, -0.15, 0.10, 0.35, 1.00],
])
D = np.diag(ann_vol)
cov = D @ corr @ D
T = 1500
L = np.linalg.cholesky(cov)
daily = rng.standard_normal((T, n_assets)) @ L.T / np.sqrt(252)
cum = 100 * np.cumprod(1 + daily, axis=0)

# ---------- 组合求解 ----------
def risk_contrib(w, cov):
    w = np.asarray(w, float)
    pv = w @ cov @ w
    mrc = cov @ w
    rc = w * mrc
    return rc / pv

def solve_min_var(cov):
    n = cov.shape[0]
    ones = np.ones(n)
    inv = np.linalg.inv(cov)
    return inv @ ones / (ones @ inv @ ones)

def solve_rp_inverse_vol(ann_vol):
    inv = 1.0 / ann_vol
    return inv / inv.sum()

def solve_max_entropy(cov):
    inv = np.linalg.inv(cov)
    w = np.diag(inv) / np.trace(inv)
    return np.clip(w, 1e-6, None)

def solve_entropy_rp(cov, w0, lam=50.0):
    """min Σ w_i ln(w_i/w0_i) + λ·Σ(RC_i - 1/n)²   s.t. Σw=1, w≥0"""
    n = cov.shape[0]
    def obj(w):
        w = np.clip(w, 1e-9, None)
        ent = np.sum(w * np.log(w / w0))
        rc = risk_contrib(w, cov)
        conc = np.sum((rc - 1.0 / n) ** 2)
        return ent + lam * conc
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0, 1)] * n
    w0i = w0 / w0.sum()
    res = minimize(obj, w0i, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 500})
    return res.x

def port_vol(w):
    return np.sqrt(w @ cov @ w)

def entropy(w):
    w = np.clip(w, 1e-9, None)
    return -np.sum(w * np.log(w))

# 先验: 市值偏重的集中权重
w_prior = np.array([0.40, 0.10, 0.10, 0.20, 0.20])
w0 = w_prior.copy()
w_eq = np.ones(n_assets) / n_assets
w_mv = solve_min_var(cov)
w_rp = solve_rp_inverse_vol(ann_vol)
w_me = solve_max_entropy(cov)
w_erp = solve_entropy_rp(cov, w0, lam=50.0)

methods = [("先验(市值偏重)", w_prior, C["prior"]), ("等权", w_eq, C["eq"]),
           ("传统风险平价\n(波动倒数)", w_rp, C["rp"]), ("最小方差", w_mv, C["minvar"]),
           ("最大熵\n(逆协方差)", w_me, C["maxent"]), ("熵风险平价", w_erp, C["erp"])]

# 图1
fig, ax = plt.subplots(figsize=(9, 5.2))
cols = [C["erp"], C["bond"], C["gold"], C["minvar"], C["rp"]]
for i in range(n_assets):
    ax.plot(cum[:, i], lw=1.2, label=names[i], color=cols[i])
ax.set_xlabel("交易日")
ax.set_ylabel("累积净值 (初始=100)")
ax.set_title("5 资产历史模拟路径：波动与相关性差异巨大")
ax.legend(fontsize=8, ncol=5, loc="upper left")
ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "erp_price_paths.png"))
plt.close(fig)

# 图2
fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(n_assets)
width = 0.13
for k, (nm, w, col) in enumerate(methods):
    rc = risk_contrib(w, cov)
    ax.bar(x + (k - 2.5) * width, rc * 100, width, label=nm, color=col)
ax.axhline(100 / n_assets, color="#444", ls="--", lw=1, label="理想平价=20%")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("边际风险贡献占比 (%)")
ax.set_title("边际风险贡献对比：越平 = 越「风险平价」")
ax.legend(fontsize=7, ncol=3, loc="upper center")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "erp_risk_contrib.png"))
plt.close(fig)

# 图3
fig, ax = plt.subplots(figsize=(10, 5.2))
for k, (nm, w, col) in enumerate(methods):
    ax.bar(x + (k - 2.5) * width, w * 100, width, label=f"{nm} (H={entropy(w):.2f})", color=col)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("权重 (%)")
ax.set_title("权重分布：熵风险平价在「尊重先验」与「风险均衡」间取折中\n(并非最分散=等权/逆协方差, 而是不过度偏离先验)")
ax.legend(fontsize=7, ncol=3, loc="upper center")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "erp_weights.png"))
plt.close(fig)

# 图4 样本外滚动波动率
half = T // 2
test_daily = daily[half:]
train_cov = np.cov(daily[:half].T) * 252
w_erp_oos = solve_entropy_rp(train_cov, w0, lam=50.0)
w_mv_oos = solve_min_var(train_cov)
port_eq = test_daily @ w_eq
port_mv = test_daily @ w_mv_oos
port_erp = test_daily @ w_erp_oos
port_prior = test_daily @ w_prior
window = 60
def rolling_vol(r, w):
    out = np.full(len(r), np.nan)
    for i in range(w, len(r)):
        out[i] = np.sqrt(np.var(r[i - w:i]) * 252) * 100
    return out
vol_eq = rolling_vol(port_eq, window)
vol_mv = rolling_vol(port_mv, window)
vol_erp = rolling_vol(port_erp, window)
vol_prior = rolling_vol(port_prior, window)
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(vol_prior, color=C["prior"], lw=1.3, label=f"先验/市值偏重 ({np.nanmean(vol_prior):.1f}%)")
ax.plot(vol_eq, color=C["eq"], lw=1.3, label=f"等权 ({np.nanmean(vol_eq):.1f}%)")
ax.plot(vol_mv, color=C["minvar"], lw=1.3, label=f"最小方差 ({np.nanmean(vol_mv):.1f}%)")
ax.plot(vol_erp, color=C["erp"], lw=1.9, label=f"熵风险平价 ({np.nanmean(vol_erp):.1f}%)")
ax.set_xlabel("测试期交易日")
ax.set_ylabel("滚动年化波动率 (%)")
ax.set_title("样本外滚动波动率：熵风险平价在稳定与分散间取平衡")
ax.legend(fontsize=8)
ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "erp_rolling_vol.png"))
plt.close(fig)

print("DONE erp", os.listdir(OUT))
print("w_prior", np.round(w_prior, 3), "entropy", entropy(w_prior))
print("w_erp", np.round(w_erp, 3), "entropy", entropy(w_erp))
print("RC prior", np.round(risk_contrib(w_prior, cov), 3))
print("RC erp", np.round(risk_contrib(w_erp, cov), 3))
print("port_vol: prior", port_vol(w_prior), "eq", port_vol(w_eq), "rp", port_vol(w_rp),
      "mv", port_vol(w_mv), "me", port_vol(w_me), "erp", port_vol(w_erp))
print("oos mean vol: prior", np.nanmean(vol_prior), "eq", np.nanmean(vol_eq),
      "mv", np.nanmean(vol_mv), "erp", np.nanmean(vol_erp))
