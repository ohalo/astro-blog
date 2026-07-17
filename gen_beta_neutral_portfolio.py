#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「Beta 中性组合：用市场对冲把方向风险剥离干净只留 alpha」生成真实配图与统计数字。

核心逻辑(Beta 中性 / Market-Neutral 多空):
  - 资产日收益由「系统性(market) + alpha(特质漂移) + 特质噪声」组成:
        r_i,t = alpha_i + beta_i * r_m,t + eps_i,t
  - 多资产等权多头组合 P: 权重 w_i = 1/N, 组合 beta_B = mean(beta_i)。
  - 用「市场指数」作对冲工具(自身 beta=1), 空头权重 = -beta_B, 则
        组合总 beta = beta_B + (-beta_B)*1 = 0  -> 系统性风险被剥离。
  - Beta 中性组合日收益 = 多头组合收益 - beta_B * 市场收益
        = Σ w_i * (alpha_i + eps_i,t)   -> 只剩 alpha + 特质噪声, 与市场脱钩。
  - 指标口径: 用训练半段估 beta, 在测试半段实盘对冲(避免前视), 报告测试段指标。

全部数字由文中 Python 真实计算(仅依赖 numpy/scipy/matplotlib)。
图片:
  beta_paths.png            —— 累计收益: 市场 / 等权多头 / Beta 中性(与市脱钩)
  beta_exposure.png         —— 中性化前后: 市场 beta / 多 / 空 / 毛 / 净 暴露对比
  beta_hedge_sensitivity.png—— 残差与市场的相关系数 vs 对冲倍数 k(仅 k=1 时归零)
  beta_drawdown.png         —— 多头 vs Beta 中性 回撤曲线对比
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "beta-neutral-portfolio")
os.makedirs(D, exist_ok=True)

C = {"mkt": "#34495E", "long": "#2F4F8F", "neut": "#C0392B", "band": "#C9D8F0",
     "grid": "#DDDDDD", "accent": "#E67E22", "good": "#27AE60", "bad": "#C0392B"}

# ----------------- 模拟参数 -----------------
rng = np.random.default_rng(20260717)
T = 504                       # 约 2 个交易年的交易日
n = 10                        # 资产数
mu_m, sig_m = 0.0003, 0.012   # 市场日收益漂移 / 波动
market = mu_m + sig_m * rng.standard_normal(T)

# 10 个资产的真实 beta 与 alpha(日度漂移), 刻意拉出差异
betas_true = np.array([0.6, 0.8, 0.9, 1.1, 1.2, 0.7, 1.0, 1.35, 0.85, 1.15])
alphas = np.array([0.0004, 0.0002, 0.0006, 0.0001, 0.0003, 0.0005, 0.0002,
                   0.0007, 0.0001, 0.0004])
idio = 0.009 * rng.standard_normal((T, n))   # 特质噪声

# 资产收益矩阵
R = alphas[None, :] + betas_true[None, :] * market[:, None] + idio

# 训练/测试切分(避免前视: 用前半段估 beta, 后半段实盘对冲)
split = T // 2
market_tr, market_te = market[:split], market[split:]
R_tr, R_te = R[:split], R[split:]
T_te = len(market_te)

# 用训练段对每只资产回归估计 beta(OLS: r_i = alpha + beta * r_m)
betas_est = np.zeros(n)
for i in range(n):
    X = np.column_stack([np.ones(split), market_tr])
    coef, *_ = np.linalg.lstsq(X, R_tr[:, i], rcond=None)
    betas_est[i] = coef[1]

# 等权多头组合(权重 1/n)
w = np.ones(n) / n
beta_P = float(np.dot(w, betas_est))          # 组合 beta

# 测试段: 等权多头组合收益
port_ret = R_te @ w
# 测试段: Beta 中性组合收益(空头市场 = -beta_P)
neut_ret = port_ret - beta_P * market_te

# 累计净值(测试段)
def cum(x):
    return np.cumprod(1.0 + x)
eq_mkt = cum(market_te)
eq_long = cum(port_ret)
eq_neut = cum(neut_ret)

# 相关系数(测试段)
corr_long_mkt = float(np.corrcoef(port_ret, market_te)[0, 1])
corr_neut_mkt = float(np.corrcoef(neut_ret, market_te)[0, 1])

# 最大回撤
def mdd(eq):
    peak = np.maximum.accumulate(eq)
    return float(np.min((eq - peak) / peak))
mdd_long = mdd(eq_long)
mdd_neut = mdd(eq_neut)
mdd_mkt = mdd(eq_mkt)

# 年化(测试段约 1 年 = T_te 日)
ann = 252.0 / T_te
def sharpe(x):
    return float(np.mean(x) / (np.std(x, ddof=1) + 1e-12) * np.sqrt(252))
shp_long = sharpe(port_ret)
shp_neut = sharpe(neut_ret)
shp_mkt = sharpe(market_te)

# 暴露口径
long_exp = 1.0
short_exp = beta_P
gross = long_exp + short_exp
net = long_exp - short_exp

# --------------- 图1: 累计净值 ---------------
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(eq_mkt, color=C["mkt"], lw=1.6, label="市场指数")
ax.plot(eq_long, color=C["long"], lw=1.8, label=f"等权多头(β={beta_P:.2f})")
ax.plot(eq_neut, color=C["neut"], lw=1.8, label="Beta 中性组合")
ax.set_title("累计净值：Beta 中性组合与市场脱钩")
ax.set_xlabel("交易日（测试段）")
ax.set_ylabel("净值（起始=1）")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "beta_paths.png"), dpi=130)
plt.close(fig)

# --------------- 图2: 暴露对比 ---------------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
labels = ["市场 Beta", "多头暴露", "空头暴露", "毛敞口", "净敞口"]
before = [beta_P, 1.0, 0.0, 1.0, 1.0]
after = [0.0, 1.0, -short_exp, gross, net]
x = np.arange(len(labels))
wbar = 0.36
b1 = ax.bar(x - wbar/2, before, wbar, label="中性化前", color=C["long"])
b2 = ax.bar(x + wbar/2, after, wbar, label="中性化后", color=C["accent"])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_title("中性化前后风险暴露对比")
ax.set_ylabel("暴露（倍数）")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for bars in (b1, b2):
    for r in bars:
        h = r.get_height()
        ax.annotate(f"{h:.2f}", (r.get_x()+r.get_width()/2, h),
                    ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=7.5, xytext=(0, 2 if h >= 0 else -2),
                    textcoords="offset points")
fig.tight_layout()
fig.savefig(os.path.join(D, "beta_exposure.png"), dpi=130)
plt.close(fig)

# --------------- 图3: 对冲倍数敏感性 ---------------
ks = np.linspace(0.0, 1.5, 31)
corrs_k = []
vols_k = []
for k in ks:
    resid = port_ret - k * beta_P * market_te
    corrs_k.append(np.corrcoef(resid, market_te)[0, 1])
    vols_k.append(np.std(resid) * np.sqrt(252))
corrs_k = np.array(corrs_k)
vols_k = np.array(vols_k)

fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
ax1.plot(ks, corrs_k, color=C["neut"], lw=2, label="残差-市场 相关系数")
ax1.axvline(1.0, color=C["good"], ls="--", lw=1.4, label="k=1（精确 β 中性）")
ax1.axhline(0, color="gray", lw=0.8)
ax1.set_xlabel("对冲倍数 k（空头市场权重 = -k·β_P）")
ax1.set_ylabel("残差与市场的相关系数", color=C["neut"])
ax1.tick_params(axis="y", labelcolor=C["neut"])
ax1.scatter([1.0], [corrs_k[np.argmin(np.abs(ks-1.0))]], color=C["good"], zorder=5)
ax1.legend(loc="upper right", fontsize=9)
ax2 = ax1.twinx()
ax2.plot(ks, vols_k, color=C["accent"], lw=1.6, ls=":", label="残差年化波动")
ax2.set_ylabel("残差年化波动", color=C["accent"])
ax2.tick_params(axis="y", labelcolor=C["accent"])
ax1.set_title("对冲倍数敏感性：只有 k=1 时系统性暴露归零")
fig.tight_layout()
fig.savefig(os.path.join(D, "beta_hedge_sensitivity.png"), dpi=130)
plt.close(fig)

# --------------- 图4: 回撤对比 ---------------
def dd_series(eq):
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / peak * 100.0
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.fill_between(np.arange(T_te), dd_series(eq_long), 0, color=C["long"], alpha=0.35,
                label=f"等权多头 (MDD {mdd_long*100:.1f}%)")
ax.plot(dd_series(eq_long), color=C["long"], lw=1.4)
ax.fill_between(np.arange(T_te), dd_series(eq_neut), 0, color=C["neut"], alpha=0.35,
                label=f"Beta 中性 (MDD {mdd_neut*100:.1f}%)")
ax.plot(dd_series(eq_neut), color=C["neut"], lw=1.4)
ax.set_title("回撤对比：Beta 中性剔除系统性下跌")
ax.set_xlabel("交易日（测试段）")
ax.set_ylabel("回撤 (%)")
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "beta_drawdown.png"), dpi=130)
plt.close(fig)

# ===================== 指标 =====================
metrics = {
    "n_assets": n, "T_total": T, "T_test": T_te,
    "beta_portfolio": round(beta_P, 4),
    "beta_true_mean": round(float(np.mean(betas_true)), 4),
    "corr_long_mkt": round(corr_long_mkt, 4),
    "corr_neut_mkt": round(corr_neut_mkt, 4),
    "mdd_long": round(mdd_long, 4),
    "mdd_neut": round(mdd_neut, 4),
    "mdd_mkt": round(mdd_mkt, 4),
    "sharpe_long": round(shp_long, 2),
    "sharpe_neut": round(shp_neut, 2),
    "sharpe_mkt": round(shp_mkt, 2),
    "long_exp": round(long_exp, 3),
    "short_exp": round(short_exp, 3),
    "gross": round(gross, 3),
    "net": round(net, 3),
    "beta_neutral_residual_beta": round(
        float(np.linalg.lstsq(np.column_stack([np.ones(T_te), market_te]),
                              neut_ret, rcond=None)[0][1]), 5),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("=== BETA NEUTRAL METRICS ===")
for k_, v_ in metrics.items():
    print(f"{k_}: {v_}")
print("done.")
