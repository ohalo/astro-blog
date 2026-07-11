#!/usr/bin/env python3
"""
为文章「分位数回归在风险管理中的应用：不止看均值，更看尾部」(quantile-regression-risk)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

数据模型：y = x*b + 异方差噪声（噪声尺度随 x 增大，制造"喇叭形"尾部扩散），
          x 为某风险因子暴露。分位数回归用 check function 逐 τ 估计条件分位。

图表：
  1. qr_fan.png        多分位回归线（τ=0.05/0.25/0.5/0.75/0.95）呈喇叭形展开
  2. qr_vs_ols.png     OLS 单条均值线 vs QR 分位线：OLS 漏掉尾部不对称
  3. var_backtest.png  条件 VaR：QR-VaR 带 + 真实回撤分布 + 回测突破点
  4. qr_coef_path.png  系数 β_τ 随 τ 的路径：风险因子的尾部效应如何变化

数值校验：QR 在 τ=0.5 应≈OLS（条件中位数≈条件均值，对称噪声时）；
          τ=0.05/0.95 应对称包住大部分点。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import linprog

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "quantile-regression-risk")
os.makedirs(D, exist_ok=True)

C = {"blue": "#4C72B0", "red": "#C44E52", "green": "#55A868",
     "purple": "#8172B3", "orange": "#CCB974", "grid": "#DDDDDD", "band": "#EAEAF2"}

# ============================================================
# 1. 模拟：异方差（噪声尺度随 x 增大）+ 轻微右偏
#    y = 1.0 + 0.8*x + sigma(x)*eps,  sigma(x)=0.3+0.25*x
# ============================================================
rng = np.random.default_rng(2026)
N = 600
x = np.sort(rng.uniform(0, 10, N))
sigma = 0.3 + 0.25 * x
eps = rng.standard_t(df=4, size=N) / np.sqrt(2.0)     # 重尾 t(4) 噪声 -> 尾部更肥
y = 1.0 + 0.8 * x + sigma * eps

XDES = np.column_stack([np.ones(N), x])               # 设计矩阵 [1, x]

# ============================================================
# 2. 分位数回归：最小化 Σ ρ_τ(y_i - x_i'β)，ρ_τ(u)=u(τ - 1{u<0})
#    严谨做法：转成线性规划（simplex / interior-point）
#    min  Σ[ τ u_i + (1-τ) v_i ]
#    s.t. y - Xβ = u - v ,  u≥0, v≥0
# ============================================================
def quantile_regression(X, y, tau):
    n, k = X.shape
    # 目标函数 c = [0..0 (β); τ..τ (u); (1-τ)..(1-τ) (v)]
    c = np.concatenate([np.zeros(k), np.full(n, tau), np.full(n, 1.0 - tau)])
    # 等式约束 A_eq z = b_eq:  -X β - u + v = -y
    A_eq = np.hstack([-X, -np.eye(n), np.eye(n)])
    b_eq = -y
    # β 无界，u,v ≥ 0
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return res.x[:k]

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
betas = {t: quantile_regression(XDES, y, t) for t in TAUS}
beta_ols = np.linalg.lstsq(XDES, y, rcond=None)[0]

# 校验：τ=0.5 应接近 OLS
print("=== 分位数回归系数 ===")
for t in TAUS:
    print(f"τ={t:.2f}: 截距={betas[t][0]:.3f}  斜率={betas[t][1]:.3f}")
print(f"OLS  : 截距={beta_ols[0]:.3f}  斜率={beta_ols[1]:.3f}")

# ============================================================
# 图 1：分位回归喇叭（fan chart）
# ============================================================
xg = np.linspace(0, 10, 100)
Xg = np.column_stack([np.ones(100), xg])
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.scatter(x, y, s=10, color=C["blue"], alpha=0.35, label="样本点 (y, x)")
fan_colors = {0.05: C["red"], 0.25: C["orange"], 0.5: "black", 0.75: C["green"], 0.95: C["purple"]}
for t in TAUS:
    yhat = Xg @ betas[t]
    ax.plot(xg, yhat, color=fan_colors[t], lw=2.2,
            label=f"τ={t:.2f}" + (" (中位数)" if t == 0.5 else ""))
ax.set_xlabel("风险因子暴露 x")
ax.set_ylabel("收益 y")
ax.set_title("分位数回归的喇叭形：条件分位随 x 展开，尾部比均值更分散")
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "qr_fan.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：OLS 单均值线 vs QR 分位线（漏掉尾部）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.scatter(x, y, s=10, color=C["blue"], alpha=0.35, label="样本点")
ax.plot(xg, Xg @ beta_ols, color="black", lw=2.4, label="OLS 条件均值")
y05 = Xg @ betas[0.05]
y95 = Xg @ betas[0.95]
ax.plot(xg, y05, color=C["red"], lw=2.0, ls="--", label="QR τ=0.05 (下行尾部)")
ax.plot(xg, y95, color=C["purple"], lw=2.0, ls="--", label="QR τ=0.95 (上行尾部)")
ax.fill_between(xg, y05, y95, color=C["band"], alpha=0.5, label="5%~95% 分位带")
ax.set_xlabel("风险因子暴露 x")
ax.set_ylabel("收益 y")
ax.set_title("OLS 只给一条均值线；分位数回归把上下尾部都刻画出来")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "qr_vs_ols.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：条件 VaR 回测（用 τ=0.05 的 QR 预测每日 5% VaR）
# ============================================================
# 把 x 当作"前一日波动率proxy"：用 |x| 排序构造时间序列场景
T = 300
rng2 = np.random.default_rng(99)
xv = rng2.uniform(0, 10, T)
sv = 0.3 + 0.25 * xv
ev = rng2.standard_t(df=4, size=T) / np.sqrt(2.0)
yv = 1.0 + 0.8 * xv + sv * ev           # 隔夜/单期收益序列
# 用全样本 QR(τ=0.05) 估计条件分位作为 VaR（负号：损失为正）
Xv = np.column_stack([np.ones(T), xv])
b05 = quantile_regression(Xv, yv, 0.05)
var_hat = Xv @ b05                        # 5% 分位的收益（负数居多）
# 回测：实际收益低于 VaR 的比例应≈5%
breaches = yv < var_hat
breach_rate = breaches.mean()
print(f"\nVaR(5%) 回测突破率: {breach_rate:.3f} (目标 0.05)")

fig, ax = plt.subplots(figsize=(10, 5.2))
tt = np.arange(T)
ax.plot(tt, yv, color=C["blue"], lw=0.9, alpha=0.8, label="真实收益序列")
ax.plot(tt, var_hat, color=C["red"], lw=1.6, label="QR 条件 VaR(5%)")
ax.scatter(tt[breaches], yv[breaches], color="black", s=18, zorder=5, label=f"突破点 ({breach_rate*100:.1f}%)")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("时间")
ax.set_ylabel("收益")
ax.set_title("QR 条件 VaR 回测：突破率应贴近 5%（而非恒定分位）")
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "var_backtest.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：系数路径 β_τ vs τ（风险因子的尾部效应变化）
# ============================================================
fine_taus = np.linspace(0.05, 0.95, 19)
slope_path, intercept_path = [], []
for t in fine_taus:
    b = quantile_regression(XDES, y, t)
    intercept_path.append(b[0])
    slope_path.append(b[1])
slope_path = np.array(slope_path)
intercept_path = np.array(intercept_path)

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(fine_taus, slope_path, "o-", color=C["green"], lw=2, ms=5, label="斜率 β_τ(因子暴露)")
ax.axhline(beta_ols[1], color="black", ls=":", lw=1.8, label=f"OLS 斜率={beta_ols[1]:.2f}")
ax.set_xlabel("分位 τ")
ax.set_ylabel("系数 β_τ（风险因子对收益的边际效应）")
ax.set_title("系数路径：风险因子的效应如何随分位（从下行到上行尾部）变化")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "qr_coef_path.png"), dpi=130)
plt.close()

print("图片已保存到:", D)
