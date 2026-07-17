#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「CIR 利率模型：用平方根扩散守住利率的非负底线」生成真实配图与统计数字。

核心逻辑(Cox-Ingersoll-Ross 1985 "A Theory of the Term Structure of Interest Rates"):
  - 瞬时短期利率 r_t 服从平方根扩散(square-root / CIR 过程):
        dr_t = a(b - r_t) dt + sigma * sqrt(r_t) dW_t
    其中 a=均值回复速度, b=长期中枢(长期均值 = b), sigma=波动。
    关键: 扩散系数是 sqrt(r_t) 而非常数, 于是当 r_t -> 0 时扩散 -> 0 而漂移项
          a(b - r_t) -> a*b > 0(向上托), 使利率**永不穿零**(只要 Feller 条件
          2ab > sigma^2 成立)。这就是 CIR 相对 Vasicek(可负)的核心改进。
  - 债券价格有仿射闭式(风险中性、市场价格为 0):
        P(t,T) = A(tau) * exp(-B(tau) * r_t)
        gamma  = sqrt(a^2 + 2 sigma^2)
        B(tau) = (exp(gamma*tau) - 1) / ((gamma + a)*(exp(gamma*tau)-1) + 2*gamma)
        A(tau) = [ 2*gamma*exp((a+gamma)*tau/2) /
                   ((gamma+a)*(exp(gamma*tau)-1) + 2*gamma) ]^(2*a*b/sigma^2)
        y(t,T) = -log P / tau
    长端收敛到 b(这里用闭式验证: 长利率 = ab(gamma-a)/sigma^2 ≈ b)。
  - 平稳分布: r_infty ~ Gamma(k=2ab/sigma^2, theta=sigma^2/(2a)), 均值 b, 方差 b sigma^2/(2a)。
  - Feller 条件 2ab > sigma^2: 决定利率是严格正(>0)还是能触零。

全部数字由文中 Python 真实计算(仅依赖 numpy/scipy/matplotlib)。
图片:
  cir_paths.png          —— 多条 CIR 模拟路径: 均值回复到 b, 被非负底反射(永不低于 0)
  cir_stationary.png     —— 终端利率直方图 vs 理论 Gamma 平稳分布(均值 b, 方差极小)
  cir_zero_floor.png     —— 平稳定分布下「接近零地板」的尾部质量热图(P(r<1%) over a x sigma)
  cir_yield_curves.png   —— 不同当前短利率 r0 下的即期收益率曲线(上翘/反转切换)
"""
import os
import json
import numpy as np
from scipy import stats
from scipy.special import gammainc  # 正则下不完全 gamma(=Gamma CDF)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "cir-interest-rate-model")
os.makedirs(D, exist_ok=True)

C = {"line": "#2F4F8F", "band": "#C9D8F0", "grid": "#DDDDDD",
     "rw": "#4C72B0", "trend": "#C44E52", "mean": "#55A868",
     "zero": "#C0392B", "accent": "#E67E22"}

# ---------- 模型参数 ----------
a, b, sigma, r0 = 0.30, 0.06, 0.04, 0.03
gamma = np.sqrt(a**2 + 2 * sigma**2)

# ---------- CIR 债券定价闭式 ----------
def cir_bond(r0, tau, a, b, sigma):
    if tau <= 0:
        return 1.0
    g = np.sqrt(a**2 + 2 * sigma**2)
    e = np.exp(g * tau)
    B = 2.0 * (e - 1.0) / ((g + a) * (e - 1.0) + 2 * g)
    A = (2 * g * np.exp((a + g) * tau / 2.0) /
         ((g + a) * (e - 1.0) + 2 * g)) ** (2 * a * b / sigma**2)
    return A * np.exp(-B * r0)

def cir_yield(r0, tau, a, b, sigma):
    if tau <= 0:
        return r0
    P = cir_bond(r0, tau, a, b, sigma)
    return -np.log(P) / tau

# ---------- 1) CIR 路径模拟(精确无偏 Milstein) ----------
rng = np.random.default_rng(20260717)
dt = 1 / 252.0
steps = 252 * 10
n_paths = 6
paths = np.zeros((n_paths, steps + 1))
paths[:, 0] = r0
for i in range(n_paths):
    r = r0
    for s in range(1, steps + 1):
        dW = rng.normal(0, np.sqrt(dt))
        # Milstein 项: sigma^2/4 * (dW^2 - dt) * sqrt(r) 来自 sqrt(r) 的二阶修正
        r = max(r, 0.0)
        drift = a * (b - r) * dt
        diff = sigma * np.sqrt(r) * dW
        milstein = 0.25 * sigma**2 * (dW**2 - dt) * np.sqrt(max(r, 0.0))
        r = r + drift + diff + milstein
        paths[i, s] = max(r, 0.0)
years = np.arange(steps + 1) * dt

# ---------- 平稳分布 ----------
k = 2 * a * b / sigma**2          # Gamma 形状
theta = sigma**2 / (2 * a)        # Gamma 尺度
stat_mean = k * theta             # = b
stat_var = k * theta**2           # = b sigma^2/(2a)
stat_std = np.sqrt(stat_var)

# 终端利率分布(用所有路径最后 2 年作为平稳样本)
burn = steps - 2 * 252
stationary_samples = paths[:, burn:].flatten()
term_min = float(paths.min())
term_max = float(paths.max())
frac_below_01 = float(np.mean(stationary_samples < 0.01))

# ---------- 3) 接近零地板的尾部质量热图 P(r_inf < 1%) over (a, sigma) ----------
As = np.linspace(0.05, 0.6, 16)
Sigs = np.linspace(0.01, 0.12, 16)
tail_grid = np.zeros((len(Sigs), len(As)))
feller_grid = np.zeros_like(tail_grid)
for i, sg in enumerate(Sigs):
    for j, aa in enumerate(As):
        kk = 2 * aa * b / sg**2
        th = sg**2 / (2 * aa)
        # Gamma CDF at 0.01
        tail_grid[i, j] = gammainc(kk, 0.01 / th)   # scipy: 正则化下不完全 gamma = CDF
        feller_grid[i, j] = (2 * aa * b - sg**2)     # Feller: >0 严格正

# ---------- 4) 收益率曲线形状随 r0 ----------
mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
r_low, r_mid, r_high = 0.02, b, 0.10
y_low = np.array([cir_yield(r_low, T, a, b, sigma) for T in mats])
y_mid = np.array([cir_yield(r_mid, T, a, b, sigma) for T in mats])
y_high = np.array([cir_yield(r_high, T, a, b, sigma) for T in mats])
slope_low = cir_yield(r_low, 10, a, b, sigma) - cir_yield(r_low, 2, a, b, sigma)
slope_mid = cir_yield(r_mid, 10, a, b, sigma) - cir_yield(r_mid, 2, a, b, sigma)
slope_high = cir_yield(r_high, 10, a, b, sigma) - cir_yield(r_high, 2, a, b, sigma)
long_rate = a * b * (gamma - a) / sigma**2   # = ab(gamma-a)/sigma^2 理论长端

# ===================== 绘图 =====================
# 图1: 路径
fig, ax = plt.subplots(figsize=(9, 4.6))
for i in range(n_paths):
    ax.plot(years, paths[i], lw=1.1, alpha=0.85, color=C["line"])
ax.axhline(b, color=C["accent"], ls="--", lw=1.6, label=f"长期中枢 b={b*100:.0f}%")
ax.axhline(0, color=C["zero"], lw=1.8)
ax.fill_between(years, 0, 0.004, color=C["zero"], alpha=0.08)
ax.text(years[steps] * 0.99, 0.006, "非负地板 r=0", color=C["zero"],
        ha="right", va="bottom", fontsize=9)
ax.set_xlabel("年")
ax.set_ylabel("瞬时短利率 r")
ax.set_title("CIR 模拟路径：均值回复至 b，被平方根扩散托在零之上")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "cir_paths.png"), dpi=130)
plt.close(fig)

# 图2: 平稳分布
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.linspace(0, 0.14, 400)
pdf = stats.gamma.pdf(x, a=k, scale=theta)
ax.hist(stationary_samples, bins=60, density=True, alpha=0.45,
        color=C["line"], label="模拟终端分布(最后2年)")
ax.plot(x, pdf, color=C["accent"], lw=2.2, label="理论 Gamma 平稳分布")
ax.axvline(b, color=C["trend"], ls="--", lw=1.6, label=f"平稳均值 b={b*100:.0f}%")
ax.set_xlabel("短利率 r")
ax.set_ylabel("概率密度")
ax.set_title(f"CIR 平稳分布：Gamma(k={k:.1f}, θ={theta:.4f})，均值 {stat_mean*100:.1f}%")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "cir_stationary.png"), dpi=130)
plt.close(fig)

# 图3: 接近零地板尾部质量热图
fig, ax = plt.subplots(figsize=(8.2, 5.4))
im = ax.imshow(tail_grid * 100, origin="lower", aspect="auto",
               extent=[As.min(), As.max(), Sigs.min(), Sigs.max()],
               cmap="YlOrRd")
ax.set_xlabel("均值回复速度 a")
ax.set_ylabel("波动 sigma")
ax.set_title("平稳定分布下 P(r < 1%) 的热图（越红越易贴近零地板）")
cb = fig.colorbar(im, ax=ax)
cb.set_label("%")
# 标注 Feller 边界 2ab - sigma^2 = 0
a_feller = sigma**2 / (2 * b)
ax.axvline(a_feller, color="black", ls=":", lw=1.4)
ax.text(a_feller + 0.01, Sigs.max() * 0.9, "Feller 边界 2ab=σ²", fontsize=8, rotation=90,
        va="top", ha="left")
fig.tight_layout()
fig.savefig(os.path.join(D, "cir_zero_floor.png"), dpi=130)
plt.close(fig)

# 图4: 收益率曲线
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(mats, y_low * 100, "-o", color=C["mean"], lw=2, label=f"r₀={r_low*100:.0f}%（低于 b→上翘）")
ax.plot(mats, y_mid * 100, "-s", color=C["line"], lw=2, label=f"r₀={r_mid*100:.0f}%（=b）")
ax.plot(mats, y_high * 100, "-^", color=C["trend"], lw=2, label=f"r₀={r_high*100:.0f}%（高于 b→反转）")
ax.axhline(b * 100, color="gray", ls=":", lw=1.2)
ax.set_xlabel("期限（年）")
ax.set_ylabel("即期收益率 (%)")
ax.set_title("CIR 即期收益率曲线：随当前短利率 r₀ 切换形状")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "cir_yield_curves.png"), dpi=130)
plt.close(fig)

# ===================== 指标 =====================
metrics = {
    "a": a, "b": b, "sigma": sigma, "r0": r0, "gamma": round(float(gamma), 4),
    "feller_2ab_s2": round(float(2 * a * b - sigma**2), 5),
    "stat_mean": round(float(stat_mean), 4),
    "stat_std": round(float(stat_std), 4),
    "path_min": round(term_min, 5),
    "path_max": round(term_max, 4),
    "frac_below_1pct": round(frac_below_01, 5),
    "slope_low_bp": round(float(slope_low * 1e4), 1),
    "slope_mid_bp": round(float(slope_mid * 1e4), 1),
    "slope_high_bp": round(float(slope_high * 1e4), 1),
    "long_rate": round(float(long_rate), 4),
    "y_low_2y": round(float(y_low[mats == 2][0]), 4),
    "y_low_30y": round(float(y_low[mats == 30][0]), 4),
    "y_mid_2y": round(float(y_mid[mats == 2][0]), 4),
    "y_mid_30y": round(float(y_mid[mats == 30][0]), 4),
    "y_high_2y": round(float(y_high[mats == 2][0]), 4),
    "y_high_30y": round(float(y_high[mats == 30][0]), 4),
    "feller_a_boundary": round(float(a_feller), 4),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("=== CIR METRICS ===")
for k_, v_ in metrics.items():
    print(f"{k_}: {v_}")
print("stationary K:", round(k, 3), "theta:", round(theta, 5))
print("done.")
