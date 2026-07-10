#!/usr/bin/env python3
"""
为文章「风险平价与波动率目标的实战融合」(risk-parity-vol-target) 生成真实配图。
数据：单市场因子 + 特异波动（随机波动率）模拟 4 资产日度收益（真实计算，非占位图）。
图表：
  1. rp_weights.png        等权 vs 风险平价权重
  2. rp_risk_contrib.png   等权 vs 风险平价的风险贡献（ERC 应近似相等）
  3. rp_leverage.png       波动率目标杠杆因子时序（危机降杠、平静加杠）
  4. rp_equity.png         四种方案净值对比（等权 / 风险平价 / 等权+波动目标 / 风险平价+波动目标）
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
D = os.path.join(BASE, "risk-parity-vol-target")
os.makedirs(D, exist_ok=True)
np.random.seed(20260711)

# ============================================================
# 1) 模拟 4 资产日度收益：双因子（风险偏好 + 避险）+ 特异波动
# ============================================================
T = 252 * 10
names = ["股票", "国债", "黄金", "商品"]
ann_mu = np.array([0.09, 0.03, 0.05, 0.07])

# 因子1：风险偏好（随机波动率，均值回复 + 偶发尖峰，制造危机聚集）
vm = 0.00015
f1 = np.zeros(T)
for t in range(T):
    vm = max(0.00005, vm + 0.03 * (0.00015 - vm) + np.random.randn() * 0.00004)
    if np.random.rand() < 0.008:
        vm *= 3.0
    f1[t] = np.random.randn() * np.sqrt(vm)
# 因子2：避险（低波动、轻微正漂移，债券与黄金共同受益）
f2 = np.random.randn(T) * np.sqrt(0.00004) + 0.00002

# 因子载荷：股票/商品偏风险偏好，国债/黄金偏避险
L = np.array([
    [1.00, -0.20],   # 股票
    [-0.15, 0.80],   # 国债
    [0.15, 0.50],    # 黄金
    [0.50, -0.10],   # 商品
])
idio_var = np.array([0.00005, 0.00002, 0.00007, 0.00012])
idio_mu = ann_mu / 252.0
R = np.zeros((T, 4))
for i in range(4):
    R[:, i] = idio_mu[i] + L[i, 0] * f1 + L[i, 1] * f2 + np.random.randn(T) * np.sqrt(idio_var[i])

# 年化协方差 / 相关（校验用）
cov_ann = np.cov(R.T) * 252
vol_ann = np.sqrt(np.diag(cov_ann))
corr = cov_ann / np.outer(vol_ann, vol_ann)
print("年化波动:", np.round(vol_ann, 3))
print("相关矩阵:\n", np.round(corr, 2))

# ============================================================
# 2) 等权 vs 风险平价（ERC：等风险贡献）
# ============================================================
def erc_weights(Sigma, n_iter=2000):
    """长仓等风险贡献（ERC）权重：求解 w>=0, Σw=1 且各资产风险贡献相等。
    用约束优化 min Σ(rc_i − 1/n)^2，避免负相关下出现负权重。"""
    from scipy.optimize import minimize
    n = Sigma.shape[0]
    def obj(w):
        pv = w @ Sigma @ w
        rc = w * (Sigma @ w) / pv
        return np.sum((rc - 1.0 / n) ** 2)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, None)] * n
    w0 = np.repeat(1.0 / n, n)
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": n_iter, "ftol": 1e-14})
    w = res.x
    return w / w.sum()

Sigma = np.cov(R.T)                        # 样本协方差（日度）
w_ew = np.repeat(0.25, 4)
w_rp = erc_weights(Sigma)
print("等权:", np.round(w_ew, 3))
print("风险平价:", np.round(w_rp, 3))

# 风险贡献
def risk_contrib(w, Sigma):
    port_var = w @ Sigma @ w
    mrc = Sigma @ w
    rc = w * mrc / port_var
    return rc

rc_ew = risk_contrib(w_ew, Sigma)
rc_rp = risk_contrib(w_rp, Sigma)
print("等权风险贡献%:", np.round(rc_ew * 100, 1))
print("RP 风险贡献%:", np.round(rc_rp * 100, 1))

# ============================================================
# 3) 波动率目标杠杆因子（基于 RP 组合收益的 EWMA 波动）
# ============================================================
r_ew = R @ w_ew
r_rp = R @ w_rp
target_vol = 0.10                          # 目标年化波动 10%

def ewma_vol(r, lam=0.94):
    var = np.zeros(len(r))
    var[0] = r[0] ** 2
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(var) * np.sqrt(252)

vol_rp = ewma_vol(r_rp)
lev = np.clip(target_vol / vol_rp, 0.5, 2.5)   # 杠杆区间 [0.5, 2.5]

# 波动率目标叠加层：用前一日的杠杆作用于当日收益
def apply_vol_target(r, vol, target=0.10, cap=(0.5, 2.5)):
    lev = np.clip(target / vol, cap[0], cap[1])
    out = np.zeros(len(r))
    for t in range(1, len(r)):
        out[t] = lev[t - 1] * r[t]
    return out

r_ew_vt = apply_vol_target(r_ew, ewma_vol(r_ew))
r_rp_vt = apply_vol_target(r_rp, ewma_vol(r_rp))

def nav(r):
    return np.cumprod(1 + r)

nav_ew = nav(r_ew)
nav_rp = nav(r_rp)
nav_ew_vt = nav(r_ew_vt)
nav_rp_vt = nav(r_rp_vt)

def perf(nav):
    rets = np.diff(nav) / nav[:-1]
    cagr = nav[-1] ** (252.0 / len(nav)) - 1
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(252)
    vol = rets.std() * np.sqrt(252)
    mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    return cagr, sharpe, vol, mdd

p_ew = perf(nav_ew); p_rp = perf(nav_rp); p_ew_vt = perf(nav_ew_vt); p_rp_vt = perf(nav_rp_vt)
print("等权:        CAGR={:.2%} Sharpe={:.2f} Vol={:.1%} MDD={:.1%}".format(*p_ew))
print("风险平价:    CAGR={:.2%} Sharpe={:.2f} Vol={:.1%} MDD={:.1%}".format(*p_rp))
print("等权+波动目标: CAGR={:.2%} Sharpe={:.2f} Vol={:.1%} MDD={:.1%}".format(*p_ew_vt))
print("风险平价+波动目标: CAGR={:.2%} Sharpe={:.2f} Vol={:.1%} MDD={:.1%}".format(*p_rp_vt))

# ============================================================
# 图1：等权 vs 风险平价权重
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.4))
x = np.arange(4); wbar = 0.38
ax.bar(x - wbar/2, w_ew * 100, wbar, label="等权", color="#ff7f0e")
ax.bar(x + wbar/2, w_rp * 100, wbar, label="风险平价 (ERC)", color="#1f77b4")
for i in range(4):
    ax.text(i - wbar/2, w_ew[i]*100 + 0.4, f"{w_ew[i]*100:.1f}%", ha="center", fontsize=9)
    ax.text(i + wbar/2, w_rp[i]*100 + 0.4, f"{w_rp[i]*100:.1f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("权重 (%)", fontsize=11)
ax.set_title("风险平价：低波动资产（国债）拿大头，高波动资产（商品/股票）降权", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.5); ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "rp_weights.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图2：风险贡献对比
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.bar(x - wbar/2, rc_ew * 100, wbar, label="等权风险贡献", color="#ff7f0e")
ax.bar(x + wbar/2, rc_rp * 100, wbar, label="风险平价风险贡献", color="#1f77b4")
for i in range(4):
    ax.text(i - wbar/2, rc_ew[i]*100 + 0.5, f"{rc_ew[i]*100:.1f}%", ha="center", fontsize=9)
    ax.text(i + wbar/2, rc_rp[i]*100 + 0.5, f"{rc_rp[i]*100:.1f}%", ha="center", fontsize=9)
ax.axhline(25, color="#d62728", ls="--", lw=1.3, label="等贡献基准 25%")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("风险贡献 (%)", fontsize=11)
ax.set_title("风险平价把「风险饼」切匀：等权则被高波动资产主导", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.5); ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "rp_risk_contrib.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图3：波动率目标杠杆因子
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.plot(lev, color="#2ca02c", lw=1.3, label="波动率目标杠杆因子（目标波动 10%）")
ax.axhline(1.0, color="#444", ls=":", lw=1.2, label="杠杆 = 1（不调整）")
ax.axhline(0.5, color="#d62728", ls="--", lw=1.0, alpha=0.7, label="下限 0.5")
ax.axhline(2.5, color="#1f77b4", ls="--", lw=1.0, alpha=0.7, label="上限 2.5")
ax.set_xlabel("交易日", fontsize=11)
ax.set_ylabel("杠杆倍数", fontsize=11)
ax.set_title("波动率目标：平静期加杠到 2.5，危机期砍杠到 0.5", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "rp_leverage.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图4：四种方案净值
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.plot(nav_ew, color="#ff7f0e", lw=1.5, label=f"等权 (CAGR={p_ew[0]:.1%}, Sharpe={p_ew[1]:.2f})")
ax.plot(nav_rp, color="#1f77b4", lw=1.5, label=f"风险平价 (CAGR={p_rp[0]:.1%}, Sharpe={p_rp[1]:.2f})")
ax.plot(nav_ew_vt, color="#9467bd", lw=1.4, alpha=0.85, label=f"等权+波动目标 (Sharpe={p_ew_vt[1]:.2f}, MDD={p_ew_vt[3]:.0%})")
ax.plot(nav_rp_vt, color="#2ca02c", lw=1.8, label=f"风险平价+波动目标 (CAGR={p_rp_vt[0]:.1%}, Sharpe={p_rp_vt[1]:.2f}, MDD={p_rp_vt[3]:.0%})")
ax.set_xlabel("交易日", fontsize=11)
ax.set_ylabel("净值（起始=1）", fontsize=11)
ax.set_title("融合方案：风险平价打底、波动率目标控险，夏普与回撤双优", fontsize=12.5, fontweight="bold")
ax.set_yscale("log")
ax.legend(loc="upper left", fontsize=8.8)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "rp_equity.png"), dpi=150, bbox_inches="tight")
plt.close()

print("✅ 风险平价+波动目标配图生成完成：", sorted(os.listdir(D)))
