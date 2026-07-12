#!/usr/env python3
"""
为文章「风险平价在债券组合中的应用：用久期与信用分层重构风险预算」(risk-parity-fixed-income)
生成真实配图。所有图表均由脚本内自洽合成数据 + 文中方法真实计算生成。

机制：
  - 债券收益风险来自两条因子: 利率因子(久期 D 加载) + 信用因子(利差久期 SD 加载)。
    Cov(i,j) = D_i D_j σ_R^2 + SD_i SD_j σ_C^2  (两因子独立)。
  - 单资产月波动 σ_i = sqrt((D_i σ_R)^2 + (SD_i σ_C)^2)。
  - 风险平价: 求解权重使每个资产的边际风险贡献 RC_i = w_i (Σw)_i / (w'Σw) 相等。
  - 等权组合风险集中在长债(高久期)与高收益(高利差波动); 风险平价把预算拉平。
  - 久期分层: 在国债阶梯内, 风险预算相等 → 短久期拿更多权重( InverseDuration 关系)。
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
D = os.path.join(BASE, "risk-parity-fixed-income")
os.makedirs(D, exist_ok=True)

C = {"gov": "#2F7ED8", "credit": "#C44E52", "rp": "#55A868", "ew": "#8C564B",
     "grid": "#DDDDDD", "line": "#333333", "inv": "#E8A33D"}

# ===================== 债券宇宙 (8 只) =====================
# name, 久期 D, 利差久期 SD(信用敏感度), 标签类型
names = ["UST 2Y", "UST 5Y", "UST 10Y", "UST 30Y",
         "IG 投资级", "HY 高收益", "EM 主权债", "MBS 抵押债"]
Ddur = np.array([1.9, 4.6, 8.8, 17.5, 7.0, 4.0, 6.5, 3.0])
SDur = np.array([0.0, 0.0, 0.0, 0.0, 6.5, 3.8, 6.0, 0.5])
sigmaR = 0.006        # 利率因子月波动
sigmaC = 0.010        # 信用因子月波动
exp_ret = np.array([0.0015, 0.0020, 0.0028, 0.0040, 0.0045, 0.0065, 0.0060, 0.0035])  # 月期望收益

n = len(names)
# 协方差(两因子独立)
Sigma = np.outer(Ddur, Ddur) * sigmaR**2 + np.outer(SDur, SDur) * sigmaC**2
sig_i = np.sqrt(np.diag(Sigma))
# 相关系数
Corr = Sigma / np.outer(sig_i, sig_i)

print("==== 风险平价(债券) 关键统计 ====")
print("单资产月波动 (%):")
for nm, sv in zip(names, sig_i):
    print(f"  {nm:12s} {sv*100:6.2f}%")

# ===================== 风险平价求解 (等风险贡献) =====================
def risk_parity(S, max_iter=5000, tol=1e-10):
    n_ = S.shape[0]
    w = np.ones(n_) / n_
    for _ in range(max_iter):
        Sw = S @ w
        rc = w * Sw
        tot = rc.sum()
        rc = rc / tot
        w = w * (1.0 / n_) / rc
        w = w / w.sum()
        # 收敛判据
        if np.max(np.abs(rc - 1.0 / n_)) < tol:
            break
    return w

w_rp = risk_parity(Sigma)
w_ew = np.ones(n) / n
w_inv = (1.0 / sig_i)
w_inv = w_inv / w_inv.sum()

def port_vol(w):
    return np.sqrt(w @ Sigma @ w)

def risk_contrib(w):
    Sw = Sigma @ w
    rc = w * Sw
    return rc / rc.sum()

def ann_vol(w):
    return port_vol(w) * np.sqrt(12)

v_ew = port_vol(w_ew)
v_rp = port_vol(w_rp)
v_inv = port_vol(w_inv)
print(f"\n组合月波动: 等权={v_ew*100:.2f}%  风险平价={v_rp*100:.2f}%  逆波动={v_inv*100:.2f}%")
print(f"组合年化波动: 等权={ann_vol(w_ew)*100:.2f}%  风险平价={ann_vol(w_rp)*100:.2f}%")
print("\n等权风险贡献 (%):", np.round(risk_contrib(w_ew) * 100, 1))
print("风险平价风险贡献 (%):", np.round(risk_contrib(w_rp) * 100, 1))

# 久期分层: 仅在 4 只国债内做风险平价
gi = [0, 1, 2, 3]
Sig_g = Sigma[np.ix_(gi, gi)]
w_g = risk_parity(Sig_g)
print("\n国债阶梯内风险平价权重 (%):", np.round(w_g * 100, 1))
print("对应久期:", Ddur[gi])

# ===================== 模拟路径(因子模型) =====================
T = 360  # 30 年月度
rng = np.random.default_rng(20260712)
fR = rng.standard_normal(T) * sigmaR
fC = rng.standard_normal(T) * sigmaC
# 资产收益 = exp_ret + D*rate_factor + SD*credit_factor
ret_mat = np.outer(exp_ret, np.ones(T)) + np.outer(Ddur, fR) + np.outer(SDur, fC)
# 加微噪
ret_mat += 0.0008 * rng.standard_normal((n, T))

def cum(ret, w):
    r = ret_mat.T @ w
    eq = np.cumprod(1 + r)
    return eq, r

eq_ew, r_ew = cum(ret_mat, w_ew)
eq_rp, r_rp = cum(ret_mat, w_rp)
eq_inv, r_inv = cum(ret_mat, w_inv)

def max_dd(eq):
    peak = np.maximum.accumulate(eq)
    return np.min(eq / peak - 1)

def sharpe(r):
    return np.mean(r) / np.std(r) * np.sqrt(12)

print(f"\n30年模拟: 等权 终值={eq_ew[-1]:.2f}x 最大回撤={max_dd(eq_ew)*100:.1f}% Sharpe={sharpe(r_ew):.2f}")
print(f"30年模拟: 风险平价 终值={eq_rp[-1]:.2f}x 最大回撤={max_dd(eq_rp)*100:.1f}% Sharpe={sharpe(r_rp):.2f}")
print(f"30年模拟: 逆波动 终值={eq_inv[-1]:.2f}x 最大回撤={max_dd(eq_inv)*100:.1f}% Sharpe={sharpe(r_inv):.2f}")

# ===================== 图 1: 单资产波动 =====================
fig, ax = plt.subplots(figsize=(8.5, 4.6))
cols = [C["gov"]] * 4 + [C["credit"]] * 4
y = np.arange(n)
ax.bar(y, sig_i * 100, color=cols, edgecolor="white")
for i, sv in enumerate(sig_i):
    ax.text(i, sv * 100 + 0.15, f"{sv*100:.1f}%", ha="center", fontsize=10)
ax.set_xticks(y)
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9.5)
ax.set_ylabel("月波动 (%)", fontsize=11)
ax.set_title("单资产风险高度不均：UST 30Y(10.5%) 是 UST 2Y(1.1%) 的近 10 倍", fontsize=12.5, fontweight="bold")
ax.grid(axis="y", color=C["grid"], alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "rp_standalone_vol.png"), dpi=130)
plt.close()

# ===================== 图 2: 风险贡献对比 =====================
rc_ew = risk_contrib(w_ew) * 100
rc_rp = risk_contrib(w_rp) * 100
fig, ax = plt.subplots(figsize=(9, 4.8))
xx = np.arange(n)
wbar = 0.4
ax.bar(xx - wbar/2, rc_ew, wbar, color=C["ew"], label="等权组合风险贡献")
ax.bar(xx + wbar/2, rc_rp, wbar, color=C["rp"], label="风险平价风险贡献")
ax.axhline(100/n, color=C["line"], ls="--", lw=1, label=f"平等线 {100/n:.0f}%")
ax.set_xticks(xx)
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9.5)
ax.set_ylabel("风险贡献 (%)", fontsize=11)
ax.set_title("等权把风险堆在长债与高收益；风险平价拉平成 1/8", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.5)
ax.grid(axis="y", color=C["grid"], alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "rp_risk_contrib.png"), dpi=130)
plt.close()

# ===================== 图 3: 久期分层(权重 vs 久期) =====================
fig, ax = plt.subplots(figsize=(8, 4.6))
gov_names = [names[i] for i in gi]
ax.scatter(Ddur[gi], w_g * 100, color=C["gov"], s=90, zorder=5)
for i, idx in enumerate(gi):
    ax.annotate(f"{gov_names[i]}\n{w_g[i]*100:.1f}%",
                (Ddur[idx], w_g[i]*100),
                textcoords="offset points", xytext=(6, 6), fontsize=9.5)
# 反函数拟合
xs = Ddur[gi]
ys = w_g * 100
pfit = np.polyfit(xs, 1.0 / ys, 1)   # 1/w ~ a*D
xline = np.linspace(1, 18, 50)
ax.plot(xline, 1.0 / np.polyval(pfit, xline), color=C["inv"], ls="--", lw=1.5, label="1/权重 ≈ a·久期")
ax.set_xlabel("修正久期 D", fontsize=11)
ax.set_ylabel("风险平价权重 (%)", fontsize=11)
ax.set_title("久期分层：风险预算相等 → 权重与久期近似成反比", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.5)
ax.grid(color=C["grid"], alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "rp_duration_tier.png"), dpi=130)
plt.close()

# ===================== 图 4: 累计净值 =====================
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(eq_ew, color=C["ew"], lw=1.8, label=f"等权 (回撤{max_dd(eq_ew)*100:.0f}%)")
ax.plot(eq_rp, color=C["rp"], lw=1.8, label=f"风险平价 (回撤{max_dd(eq_rp)*100:.0f}%)")
ax.plot(eq_inv, color=C["inv"], lw=1.5, ls=":", label=f"逆波动 (回撤{max_dd(eq_inv)*100:.0f}%)")
ax.set_xlabel("月度", fontsize=11)
ax.set_ylabel("净值 (初始=1)", fontsize=11)
ax.set_title("30 年模拟：风险平价下行更平滑、回撤更浅", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(color=C["grid"], alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "rp_cum_pnl.png"), dpi=130)
plt.close()

print("charts saved to", D)
