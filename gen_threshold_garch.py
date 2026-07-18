#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门限GARCH与杠杆效应 配图生成 (3 张真实图表)"""
import os
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "threshold-garch-asymmetry"
OUT = f"public/images/{SLUG}"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260719)

# ---------- 1) 模拟真实 TGARCH(1,1) 数据(含杠杆效应, 合理量纲) ----------
# 日度收益波动 ~2% -> 方差 ~4e-4; 用 omega/(1-persistence)=4e-4 反推
def simulate_tgarch(T, omega=2e-5, alpha=0.08, gamma=0.06, beta=0.85):
    e = np.zeros(T); sigma = np.zeros(T); z = rng.normal(0, 1, T)
    sigma[0] = np.sqrt(max(omega / (1 - alpha - 0.5 * gamma - beta), 1e-4))
    for t in range(1, T):
        g = alpha + (gamma if e[t-1] < 0 else 0.0)
        sigma[t] = np.sqrt(max(omega + g * e[t-1]**2 + beta * sigma[t-1]**2, 1e-10))
        e[t] = sigma[t] * z[t]
    return e, sigma, dict(omega=omega, alpha=alpha, gamma=gamma, beta=beta)

T = 4000
e, sigma_true, TRUE = simulate_tgarch(T)
var_target = np.var(e)

# ---------- 2) MLE 拟合: 对称 GARCH / TGARCH ----------
def garch_ll(params, e, asymmetric=False):
    omega, alpha, beta = params[0], params[1], params[2]
    gamma = params[3] if asymmetric else 0.0
    if omega <= 0 or alpha <= 0 or beta <= 0 or beta >= 0.999:
        return 1e12
    s2 = np.zeros_like(e); s2[0] = np.var(e)
    ll = 0.0
    for t in range(1, len(e)):
        g = alpha + (gamma if e[t-1] < 0 else 0.0)
        s2[t] = omega + g * e[t-1]**2 + beta * s2[t-1]
        s2[t] = max(s2[t], 1e-12)
        ll += -0.5 * (np.log(2*np.pi) + np.log(s2[t]) + e[t]**2 / s2[t])
    return -ll

def fit_garch(e, asymmetric):
    x0 = [2e-5, 0.08, 0.85, 0.06] if asymmetric else [2e-5, 0.10, 0.85]
    bnds = [(1e-8, 1e-3), (1e-4, 0.4), (1e-3, 0.98), (0.0, 0.3)] if asymmetric else [(1e-8, 1e-3), (1e-4, 0.4), (1e-3, 0.98)]
    res = minimize(garch_ll, x0, args=(e, asymmetric), method="L-BFGS-B", bounds=bnds)
    return res.x, -garch_ll(res.x, e, asymmetric)

p_sym, ll_sym = fit_garch(e, False)
p_asym, ll_asym = fit_garch(e, True)
omega_s, alpha_s, beta_s = p_sym
omega_a, alpha_a, beta_a, gamma_a = p_asym

# ---------- 3) 图1: 杠杆效应(同一 TGARCH 模型下, -3% 跌 vs +3% 涨 推高多少条件方差) ----------
sigma_prev2 = np.var(e)  # 用无条件方差作为"昨日条件方差"的代理
shock = 0.03
s2_down = omega_a + (alpha_a + gamma_a) * shock**2 + beta_a * sigma_prev2
s2_up   = omega_a + alpha_a * shock**2 + beta_a * sigma_prev2
leverage_ratio = s2_down / s2_up - 1

# 响应曲线
mags = np.linspace(-0.06, 0.06, 51)
s2_tg = [omega_a + (alpha_a + (gamma_a if m < 0 else 0.0)) * m**2 + beta_a * sigma_prev2 for m in mags]
s2_gs = [omega_s + alpha_s * m**2 + beta_s * sigma_prev2 for m in mags]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
axes[0].plot(np.array(mags)*100, np.array(s2_tg)*1e4, label="TGARCH(含杠杆)", color="#c0392b", lw=2)
axes[0].plot(np.array(mags)*100, np.array(s2_gs)*1e4, label="对称GARCH", color="#2980b9", lw=2, ls="--")
axes[0].axvline(0, color="gray", lw=0.8)
axes[0].set_xlabel("当日收益冲击 (%)"); axes[0].set_ylabel("下一步条件方差 (×1e-4)")
axes[0].set_title("条件方差对正负冲击的响应: 跌弯得更陡", fontweight="bold")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

bars = ["崩盘 -3%", "反弹 +3%"]
tgv = [s2_down, s2_up]
x = np.arange(2); w = 0.5
axes[1].bar(x, np.array(tgv)*1e4, w, color=["#c0392b", "#27ae60"])
axes[1].set_xticks(x); axes[1].set_xticklabels(bars)
axes[1].set_ylabel("下一步条件方差 (×1e-4)")
axes[1].set_title(f"同等 ±3% 冲击: 跌比涨多推高 {leverage_ratio*100:.1f}%", fontweight="bold")
for i, v in enumerate(tgv):
    axes[1].text(i, v*1e4, f"{v*1e4:.1f}", ha="center", va="bottom", fontsize=9)
axes[1].grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/tgarch_leverage_effect.png", bbox_inches="tight"); plt.close(fig)

# ---------- 4) 图2: 非对称响应回归(下行幅度斜率 vs 上行幅度斜率) ----------
neg_ret = -np.minimum(e[:-1], 0)   # 下跌幅度(正数)
pos_ret = np.maximum(e[:-1], 0)    # 上涨幅度(正数)
y_abs = np.abs(e[1:])
X = np.column_stack([neg_ret, pos_ret, np.ones_like(neg_ret)])
coef, *_ = np.linalg.lstsq(X, y_abs, rcond=None)
sl_neg, sl_pos, inter = coef
pred = X @ coef
r2 = 1 - np.sum((y_abs - pred)**2) / np.sum((y_abs - y_abs.mean())**2)

fig, ax = plt.subplots(figsize=(8.6, 5))
ax.scatter(e[:-1]*100, y_abs*100, s=6, alpha=0.25, color="#7f8c8d", label="样本点 |e_t|")
xs = np.linspace(0, 6, 50)
ax.plot(-xs, (sl_neg * xs/100 + inter)*100, color="#c0392b", lw=2.5, label=f"下跌幅度斜率 {sl_neg:.4f}")
ax.plot(xs, (sl_pos * xs/100 + inter)*100, color="#2980b9", lw=2.5, label=f"上涨幅度斜率 {sl_pos:.4f}")
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("前一日收益幅度 |e_{t-1}| (%)"); ax.set_ylabel("当日 |e_t| (×1e-2)")
ax.set_title(f"非对称响应: 跌幅斜率({sl_neg:.4f}) > 涨幅斜率({sl_pos:.4f}), R²={r2:.3f}", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/tgarch_asymmetric_slope.png", bbox_inches="tight"); plt.close(fig)

# ---------- 5) 图3: TGARCH 拟合波动 vs 真实波动 + 波动率聚集 ----------
s2_fit = np.zeros(T); s2_fit[0] = np.var(e)
for t in range(1, T):
    g = alpha_a + (gamma_a if e[t-1] < 0 else 0.0)
    s2_fit[t] = omega_a + g * e[t-1]**2 + beta_a * s2_fit[t-1]
sig_fit = np.sqrt(s2_fit)
rmse_fit = np.sqrt(np.mean((sig_fit - sigma_true)**2))
# 对称GARCH 拟合作为对照
s2_sym = np.zeros(T); s2_sym[0] = np.var(e)
for t in range(1, T):
    s2_sym[t] = omega_s + alpha_s * e[t-1]**2 + beta_s * s2_sym[t-1]
sig_sym = np.sqrt(s2_sym)
rmse_sym = np.sqrt(np.mean((sig_sym - sigma_true)**2))

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
seg = slice(3000, 3200)
axes[0].plot(sigma_true[seg], color="#27ae60", lw=1.4, label="真实 TGARCH 波动")
axes[0].plot(sig_fit[seg], color="#c0392b", lw=1.4, label="TGARCH 拟合波动")
axes[0].plot(sig_sym[seg], color="#2980b9", lw=1.2, ls="--", alpha=0.8, label="对称GARCH 拟合波动")
axes[0].set_ylabel("波动率 σ"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[1].plot(e[seg]*100, color="#7f8c8d", lw=0.7)
axes[1].set_ylabel("收益 (%)"); axes[1].set_xlabel("时间 (采样段)")
axes[1].set_title(f"TGARCH 还原波动更贴真实: 拟合 RMSE {rmse_fit:.4f} < 对称 {rmse_sym:.4f}", fontweight="bold")
axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/tgarch_egarch_fit.png", bbox_inches="tight"); plt.close(fig)

# ---------- 6) 打印真实数字 ----------
print(f"[TRUE  ] ω={TRUE['omega']:.2e} α={TRUE['alpha']:.3f} γ={TRUE['gamma']:.3f} β={TRUE['beta']:.3f}")
print(f"[SYM   ] ω={omega_s:.2e} α={alpha_s:.4f} β={beta_s:.4f}")
print(f"[TGARCH] ω={omega_a:.2e} α={alpha_a:.4f} γ={gamma_a:.4f} β={beta_a:.4f}")
print(f"杠杆效应: 同等 ±3% 冲击, 跌比涨多推高 {leverage_ratio*100:.1f}%")
print(f"分段回归 下跌斜率={sl_neg:.4f} 上涨斜率={sl_pos:.4f} R²={r2:.4f}")
print(f"对称GARCH LL={ll_sym:.1f}  TGARCH LL={ll_asym:.1f}  ΔLL={ll_asym-ll_sym:.1f}")
print(f"拟合 RMSE: TGARCH={rmse_fit:.4f}  对称GARCH={rmse_sym:.4f}")
print("DONE images:", os.listdir(OUT))
