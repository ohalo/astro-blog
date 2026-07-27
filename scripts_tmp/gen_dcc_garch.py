# -*- coding: utf-8 -*-
"""DCC-GARCH 动态相关 配图生成"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUT = '/Users/halo/workspace/astro-blog/public/images/dcc-garch-correlation'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ---------- 模拟：两资产，GARCH(1,1) 波动率 + 时变真实相关 ----------
T = 2500
# 真实相关：慢正弦 + 危机跳升
t = np.arange(T)
rho_true = 0.3 + 0.15*np.sin(2*np.pi*t/1250)
rho_true[1200:1450] = 0.85  # 危机段
rho_true = np.clip(rho_true, -0.95, 0.95)

# GARCH(1,1) 参数（两资产）
omega = np.array([0.05, 0.08]); alpha = np.array([0.08, 0.10]); beta = np.array([0.90, 0.87])
h = omega/(1-alpha-beta)  # 初始
r = np.zeros((T,2)); hs = np.zeros((T,2))
for i in range(T):
    hs[i] = h
    # 相关标准化冲击
    z1 = rng.standard_normal(); z2 = rho_true[i]*z1 + np.sqrt(1-rho_true[i]**2)*rng.standard_normal()
    z = np.array([z1, z2])
    r[i] = np.sqrt(h)*z
    h = omega + alpha*r[i]**2 + beta*h

# ---------- 第一步：单变量 GARCH 拟合（QML） ----------
def garch_nll(params, x):
    w, a, b = params
    if w<=0 or a<0 or b<0 or a+b>=0.999: return 1e10
    h = np.var(x); nll = 0.0
    for xi in x:
        nll += 0.5*(np.log(h) + xi*xi/h)
        h = w + a*xi*xi + b*h
    return nll

def fit_garch(x):
    res = minimize(garch_nll, [0.05, 0.08, 0.88], args=(x,), method='Nelder-Mead',
                   options={'maxiter':2000, 'xatol':1e-6, 'fatol':1e-6})
    return res.x

def garch_filter(x, params):
    w, a, b = params
    h = np.var(x); out = np.zeros(len(x))
    for i, xi in enumerate(x):
        out[i] = h
        h = w + a*xi*xi + b*h
    return out

p1 = fit_garch(r[:,0]); p2 = fit_garch(r[:,1])
h1 = garch_filter(r[:,0], p1); h2 = garch_filter(r[:,1], p2)
eps = np.column_stack([r[:,0]/np.sqrt(h1), r[:,1]/np.sqrt(h2)])  # 标准化残差

# ---------- 第二步：DCC(1,1) 拟合 ----------
Qbar = np.corrcoef(eps.T)

def dcc_filter(eps, a, b, Qbar):
    T = len(eps); Q = Qbar.copy(); rho = np.zeros(T)
    for i in range(T):
        d = 1/np.sqrt(np.diag(Q))
        R = Q * np.outer(d, d)
        rho[i] = R[0,1]
        e = eps[i]
        Q = (1-a-b)*Qbar + a*np.outer(e,e) + b*Q
    return rho

def dcc_nll(params, eps, Qbar):
    a, b = params
    if a<0 or b<0 or a+b>=0.999: return 1e10
    T = len(eps); Q = Qbar.copy(); nll = 0.0
    for i in range(T):
        d = 1/np.sqrt(np.diag(Q))
        R = Q*np.outer(d,d)
        det = R[0,0]*R[1,1]-R[0,1]*R[1,0]
        e = eps[i]
        Rinv_quad = (R[1,1]*e[0]**2 - 2*R[0,1]*e[0]*e[1] + R[0,0]*e[1]**2)/det
        nll += 0.5*(np.log(det) + Rinv_quad - e@e)
        Q = (1-a-b)*Qbar + a*np.outer(e,e) + b*Q
    return nll

res = minimize(dcc_nll, [0.05, 0.90], args=(eps, Qbar), method='Nelder-Mead',
               options={'maxiter':1000})
a_hat, b_hat = res.x
rho_dcc = dcc_filter(eps, a_hat, b_hat, Qbar)
print(f"DCC 估计: a={a_hat:.4f}, b={b_hat:.4f}, a+b={a_hat+b_hat:.4f}")

# 对比估计器
win = 252
rho_roll = np.full(T, np.nan)
for i in range(win, T):
    rho_roll[i] = np.corrcoef(r[i-win:i,0], r[i-win:i,1])[0,1]

lam = 0.94
q11=q22=1.0; q12=Qbar[0,1]
rho_ewma = np.zeros(T)
for i in range(T):
    d = q12/np.sqrt(q11*q22); rho_ewma[i]=d
    q11 = lam*q11 + (1-lam)*eps[i,0]**2
    q22 = lam*q22 + (1-lam)*eps[i,1]**2
    q12 = lam*q12 + (1-lam)*eps[i,0]*eps[i,1]

# ---------- 图1：相关追踪 ----------
fig, ax = plt.subplots(figsize=(11,5.5))
sl = slice(800, 2000)
ax.plot(t[sl], rho_true[sl], 'k-', lw=2.2, label='真实相关 ρ_t', alpha=0.85)
ax.plot(t[sl], rho_roll[sl], color='#d62728', lw=1.4, label='252天滚动相关', alpha=0.8)
ax.plot(t[sl], rho_ewma[sl], color='#2ca02c', lw=1.2, label='EWMA λ=0.94', alpha=0.75)
ax.plot(t[sl], rho_dcc[sl], color='#1f77b4', lw=1.5, label=f'DCC(1,1) a={a_hat:.3f}, b={b_hat:.3f}', alpha=0.9)
ax.axvspan(1200, 1450, color='orange', alpha=0.12, label='危机段（ρ跳至0.85）')
ax.set_xlabel('交易日'); ax.set_ylabel('相关系数')
ax.set_title('时变相关追踪：DCC 贴着真实相关走，滚动窗口慢一拍')
ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/correlation-tracking.png', dpi=110); plt.close()

# 追踪误差统计
mask = ~np.isnan(rho_roll)
mae_roll = np.mean(np.abs(rho_roll[mask]-rho_true[mask]))
mae_ewma = np.mean(np.abs(rho_ewma[mask]-rho_true[mask]))
mae_dcc = np.mean(np.abs(rho_dcc[mask]-rho_true[mask]))
print(f"MAE: roll={mae_roll:.4f}, ewma={mae_ewma:.4f}, dcc={mae_dcc:.4f}")

# 危机段收敛速度
crisis = slice(1200,1450)
def days_to_90(series, start=1200, target=0.85, base=0.3):
    thresh = base + 0.9*(target-base)
    for i in range(start, 1450):
        if series[i] >= thresh: return i-start
    return None
d_roll = days_to_90(rho_roll); d_ewma = days_to_90(rho_ewma); d_dcc = days_to_90(rho_dcc)
print(f"危机收敛到90%天数: roll={d_roll}, ewma={d_ewma}, dcc={d_dcc}")

# ---------- 图2：两步法示意（波动率层 + 相关层） ----------
fig, axes = plt.subplots(2,1, figsize=(11,7), sharex=True)
sl = slice(1000, 1800)
axes[0].plot(t[sl], np.sqrt(hs[sl,0]*252)*1, color='#9467bd', lw=1.2, label='资产1 真实条件波动率(年化)')
axes[0].plot(t[sl], np.sqrt(h1[sl]*252), color='#1f77b4', lw=1.2, ls='--', label='GARCH(1,1) 拟合波动率')
axes[0].axvspan(1200,1450, color='orange', alpha=0.12)
axes[0].set_ylabel('年化波动率'); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[0].set_title('第一步：单变量 GARCH 各自过滤波动率')
axes[1].plot(t[sl], rho_true[sl], 'k-', lw=2, label='真实相关', alpha=0.8)
axes[1].plot(t[sl], rho_dcc[sl], color='#1f77b4', lw=1.4, label='DCC 相关（标准化残差上估计）')
axes[1].axvspan(1200,1450, color='orange', alpha=0.12)
axes[1].set_xlabel('交易日'); axes[1].set_ylabel('相关系数'); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
axes[1].set_title('第二步：DCC 在标准化残差上驱动相关演化')
plt.tight_layout(); plt.savefig(f'{OUT}/two-step.png', dpi=110); plt.close()

# ---------- 图3：组合 VaR 回测 ----------
w = np.array([0.5, 0.5])
port = r @ w
# 三种协方差 -> 组合波动率预测（用前一天信息）
def port_vol_series(rho_series, h1s, h2s):
    return np.sqrt(w[0]**2*h1s + w[1]**2*h2s + 2*w[0]*w[1]*rho_series*np.sqrt(h1s*h2s))

# 滚动：波动率也用滚动
h1_roll = np.full(T, np.nan); h2_roll = np.full(T, np.nan)
for i in range(win, T):
    h1_roll[i] = np.var(r[i-win:i,0]); h2_roll[i] = np.var(r[i-win:i,1])
vol_roll = port_vol_series(rho_roll, h1_roll, h2_roll)
vol_dcc = port_vol_series(rho_dcc, h1, h2)
z99 = 2.326
var_roll = -z99*vol_roll; var_dcc = -z99*vol_dcc
br_roll = np.nanmean(port[win:] < var_roll[win:])
br_dcc = np.nanmean(port[win:] < var_dcc[win:])
print(f"1% VaR 击穿率: roll={br_roll*100:.2f}%, dcc={br_dcc*100:.2f}%")

# 分段击穿率（危机 vs 平静）
idx = np.arange(T)
crisis_mask = (idx>=1200)&(idx<1500)
calm_mask = (idx>=win)&~crisis_mask
br_roll_c = np.nanmean(port[crisis_mask] < var_roll[crisis_mask])
br_dcc_c = np.nanmean(port[crisis_mask] < var_dcc[crisis_mask])
br_roll_q = np.nanmean(port[calm_mask] < var_roll[calm_mask])
br_dcc_q = np.nanmean(port[calm_mask] < var_dcc[calm_mask])
print(f"危机段击穿: roll={br_roll_c*100:.2f}%, dcc={br_dcc_c*100:.2f}%")
print(f"平静段击穿: roll={br_roll_q*100:.2f}%, dcc={br_dcc_q*100:.2f}%")

fig, axes = plt.subplots(1,2, figsize=(11,4.6))
labels = ['全样本', '危机段\n(ρ=0.85)', '平静段']
roll_vals = [br_roll*100, br_roll_c*100, br_roll_q*100]
dcc_vals = [br_dcc*100, br_dcc_c*100, br_dcc_q*100]
x = np.arange(3); wd=0.35
axes[0].bar(x-wd/2, roll_vals, wd, label='滚动252天', color='#d62728', alpha=0.8)
axes[0].bar(x+wd/2, dcc_vals, wd, label='DCC-GARCH', color='#1f77b4', alpha=0.8)
axes[0].axhline(1.0, color='k', ls='--', lw=1, label='目标 1%')
axes[0].set_xticks(x); axes[0].set_xticklabels(labels); axes[0].set_ylabel('1% VaR 击穿率 (%)')
axes[0].set_title('VaR 击穿率：滚动窗口在危机段系统性失守'); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3, axis='y')
sl2 = slice(1150, 1600)
axes[1].plot(t[sl2], port[sl2], color='gray', lw=0.6, alpha=0.6, label='组合日收益')
axes[1].plot(t[sl2], var_roll[sl2], color='#d62728', lw=1.3, label='滚动 VaR(99%)')
axes[1].plot(t[sl2], var_dcc[sl2], color='#1f77b4', lw=1.3, label='DCC VaR(99%)')
breaches = (port < var_roll) & crisis_mask
axes[1].scatter(t[breaches], port[breaches], color='red', s=25, zorder=5, label='滚动VaR击穿点')
axes[1].axvspan(1200,1450,color='orange',alpha=0.1)
axes[1].set_xlabel('交易日'); axes[1].set_title('危机段放大：DCC 的 VaR 线跟着跳下去了')
axes[1].legend(fontsize=8, loc='lower right'); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/var-backtest.png', dpi=110); plt.close()

# ---------- 图4：a+b 持续性 与 半衰期直觉 ----------
fig, axes = plt.subplots(1,2, figsize=(11,4.4))
ab_grid = [(0.02,0.95),(0.05,0.90),(0.10,0.85),(0.20,0.70)]
colors = plt.cm.viridis(np.linspace(0.15,0.85,len(ab_grid)))
for (aa,bb),c in zip(ab_grid, colors):
    rho_g = dcc_filter(eps, aa, bb, Qbar)
    axes[0].plot(t[1100:1700], rho_g[1100:1700], color=c, lw=1.2, label=f'a={aa}, b={bb} (a+b={aa+bb})')
axes[0].plot(t[1100:1700], rho_true[1100:1700], 'k--', lw=1.8, label='真实相关', alpha=0.7)
axes[0].axvspan(1200,1450,color='orange',alpha=0.1)
axes[0].set_xlabel('交易日'); axes[0].set_ylabel('相关系数')
axes[0].set_title('a 控制反应速度，a+b 控制记忆长度')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
maes = []
for aa,bb in ab_grid:
    rho_g = dcc_filter(eps, aa, bb, Qbar)
    maes.append(np.mean(np.abs(rho_g-rho_true)))
labels2 = [f'a={aa}\nb={bb}' for aa,bb in ab_grid]
bars = axes[1].bar(labels2, maes, color=colors, alpha=0.85)
mae_fit = np.mean(np.abs(rho_dcc-rho_true))
axes[1].axhline(mae_fit, color='#1f77b4', ls='--', lw=1.5, label=f'QML估计 (a={a_hat:.3f},b={b_hat:.3f}): {mae_fit:.3f}')
axes[1].set_ylabel('相关追踪 MAE'); axes[1].set_title('追踪误差：QML 估计的参数接近手工网格最优')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/persistence.png', dpi=110); plt.close()

print("figures saved:", os.listdir(OUT))
