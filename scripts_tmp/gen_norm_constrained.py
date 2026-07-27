# -*- coding: utf-8 -*-
"""范数约束组合优化 配图生成"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUT = '/Users/halo/workspace/astro-blog/public/images/norm-constrained-portfolio'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ---------- 市场：N=30 资产，单因子+板块结构 ----------
N = 30
beta = rng.uniform(0.6, 1.4, N)
n_sec = 3
sector = np.repeat(np.arange(n_sec), N//n_sec)
sec_load = rng.uniform(0.4, 0.8, N)
idio = rng.uniform(0.12, 0.30, N)  # 年化特质波动
sig_m = 0.16; sig_s = 0.10

Sigma_true = np.outer(beta,beta)*sig_m**2 + np.diag(idio**2)
for s in range(n_sec):
    m = sector==s
    Sigma_true[np.ix_(m,m)] += np.outer(sec_load[m], sec_load[m])*sig_s**2

Ltrue = np.linalg.cholesky(Sigma_true/252)

def sample_returns(T):
    return (Ltrue @ rng.standard_normal((N,T))).T

def gmv_norm(S, l1_cap=None, l2_cap=None):
    """最小方差，sum w =1，可选 L1/L2 范数约束"""
    w0 = np.ones(N)/N
    cons = [{'type':'eq','fun': lambda w: w.sum()-1}]
    if l1_cap is not None:
        cons.append({'type':'ineq','fun': lambda w: l1_cap - np.abs(w).sum()})
    if l2_cap is not None:
        cons.append({'type':'ineq','fun': lambda w: l2_cap**2 - (w**2).sum()})
    res = minimize(lambda w: w@S@w, w0, jac=lambda w: 2*S@w,
                   constraints=cons, method='SLSQP',
                   options={'maxiter':500, 'ftol':1e-12})
    return res.x

def true_vol(w):
    return np.sqrt(w@Sigma_true@w)

w_opt = gmv_norm(Sigma_true)
vol_opt = true_vol(w_opt)
print(f"理论最优 GMV 真实波动率: {vol_opt*100:.2f}%")

# ---------- 实验1：L1 上限扫描（T=120） ----------
T_est = 120
n_mc = 40
l1_grid = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, None]  # None = 无约束
results = {c: [] for c in l1_grid}
lev = {c: [] for c in l1_grid}
for mc in range(n_mc):
    R = sample_returns(T_est)
    S = np.cov(R.T)*252
    for c in l1_grid:
        w = gmv_norm(S, l1_cap=c)
        results[c].append(true_vol(w))
        lev[c].append(np.abs(w).sum())

med = {c: np.median(results[c])*100 for c in l1_grid}
levm = {c: np.median(lev[c]) for c in l1_grid}
ew_vol = true_vol(np.ones(N)/N)*100
print("L1扫描 (T=120):")
for c in l1_grid:
    print(f"  cap={c}: OOS vol={med[c]:.2f}%, leverage={levm[c]:.2f}")
print(f"  等权: {ew_vol:.2f}%, 理论最优: {vol_opt*100:.2f}%")

fig, axes = plt.subplots(1,2, figsize=(11,4.6))
xs = [1.0,1.2,1.5,2.0,3.0,5.0]
ys = [med[c] for c in xs]
axes[0].plot(xs, ys, 'o-', color='#1f77b4', lw=2, ms=7, label='范数约束 GMV')
axes[0].axhline(med[None], color='#d62728', ls='--', lw=1.5, label=f'无约束样本 GMV：{med[None]:.1f}%')
axes[0].axhline(ew_vol, color='gray', ls=':', lw=1.5, label=f'等权：{ew_vol:.1f}%')
axes[0].axhline(vol_opt*100, color='green', ls='-.', lw=1.5, label=f'理论最优：{vol_opt*100:.1f}%')
axes[0].set_xlabel('L1 范数上限 ‖w‖₁ ≤ c'); axes[0].set_ylabel('样本外真实波动率 (%)')
axes[0].set_title(f'T={T_est} 天：范数上限的 U 形曲线（{n_mc} 次 MC 中位数）')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
ys2 = [levm[c] for c in xs]
axes[1].plot(xs, ys2, 's-', color='#ff7f0e', lw=2, ms=7)
axes[1].plot([1,5],[1,5],'k--', lw=1, alpha=0.5, label='上限本身 (对角线)')
axes[1].axhline(levm[None], color='#d62728', ls='--', lw=1.5, label=f'无约束杠杆：{levm[None]:.1f}')
axes[1].set_xlabel('L1 范数上限'); axes[1].set_ylabel('实际总杠杆 ‖w‖₁')
axes[1].set_title('上限收紧时约束绑定：实际杠杆贴着上限走')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/l1-sweep.png', dpi=110); plt.close()

# ---------- 实验2：权重形态对比 ----------
R = sample_returns(T_est)
S = np.cov(R.T)*252
w_unc = gmv_norm(S)
w_l1 = gmv_norm(S, l1_cap=1.5)
w_lo = gmv_norm(S, l1_cap=1.0)  # L1<=1 且 sum=1 => 等价 long-only
fig, axes = plt.subplots(3,1, figsize=(11,7.5), sharex=True)
xi = np.arange(N)
for ax, w, title, color in [
    (axes[0], w_unc, f'无约束样本 GMV：杠杆 {np.abs(w_unc).sum():.1f}，真实波动率 {true_vol(w_unc)*100:.1f}%', '#d62728'),
    (axes[1], w_l1, f'‖w‖₁ ≤ 1.5：杠杆 {np.abs(w_l1).sum():.1f}，真实波动率 {true_vol(w_l1)*100:.1f}%', '#1f77b4'),
    (axes[2], w_lo, f'‖w‖₁ ≤ 1（等价 long-only）：真实波动率 {true_vol(w_lo)*100:.1f}%', '#2ca02c')]:
    ax.bar(xi, w*100, color=color, alpha=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title(title, fontsize=10); ax.set_ylabel('权重 (%)'); ax.grid(alpha=0.3, axis='y')
axes[2].set_xlabel('资产编号')
plt.tight_layout(); plt.savefig(f'{OUT}/weights-compare.png', dpi=110); plt.close()
print(f"零权重个数 (l1=1.0): {np.sum(np.abs(w_lo)<1e-4)}")

# ---------- 实验3：范数约束 vs Ledoit-Wolf 收缩 等价性 ----------
def lw_shrink(R):
    """Ledoit-Wolf 到单位阵目标（简化版）"""
    Tn, Nn = R.shape
    X = R - R.mean(0)
    S = X.T@X/Tn
    mu = np.trace(S)/Nn
    F = mu*np.eye(Nn)
    d2 = np.sum((S-F)**2)
    b2 = 0.0
    for i in range(Tn):
        xi = X[i][:,None]
        b2 += np.sum((xi@xi.T - S)**2)
    b2 = min(b2/Tn**2, d2)
    delta = b2/d2
    return delta*F + (1-delta)*S, delta

T_grid = [60, 120, 252, 504]
res_unc, res_l1, res_lw, res_both = [], [], [], []
for Tn in T_grid:
    v_unc, v_l1, v_lw, v_both = [], [], [], []
    for mc in range(30):
        R = sample_returns(Tn)
        S = np.cov(R.T)*252
        Slw, delta = lw_shrink(R); Slw *= 252
        v_unc.append(true_vol(gmv_norm(S)))
        v_l1.append(true_vol(gmv_norm(S, l1_cap=1.6)))
        v_lw.append(true_vol(gmv_norm(Slw)))
        v_both.append(true_vol(gmv_norm(Slw, l1_cap=1.6)))
    res_unc.append(np.median(v_unc)*100); res_l1.append(np.median(v_l1)*100)
    res_lw.append(np.median(v_lw)*100); res_both.append(np.median(v_both)*100)
    print(f"T={Tn}: unc={res_unc[-1]:.2f}, l1={res_l1[-1]:.2f}, lw={res_lw[-1]:.2f}, both={res_both[-1]:.2f}")

fig, ax = plt.subplots(figsize=(10,5.2))
ax.plot(T_grid, res_unc, 'o-', color='#d62728', lw=2, label='无约束样本协方差')
ax.plot(T_grid, res_l1, 's-', color='#1f77b4', lw=2, label='范数约束 ‖w‖₁≤1.6')
ax.plot(T_grid, res_lw, '^-', color='#9467bd', lw=2, label='Ledoit-Wolf 收缩')
ax.plot(T_grid, res_both, 'd-', color='#2ca02c', lw=2, label='收缩 + 范数约束')
ax.axhline(vol_opt*100, color='k', ls='-.', lw=1.2, label=f'理论最优 {vol_opt*100:.1f}%')
ax.axhline(ew_vol, color='gray', ls=':', lw=1.2, label=f'等权 {ew_vol:.1f}%')
ax.set_xscale('log'); ax.set_xticks(T_grid); ax.set_xticklabels(T_grid)
ax.set_xlabel('估计窗口 T（天）'); ax.set_ylabel('样本外真实波动率 (%)')
ax.set_title('范数约束 ≈ 隐式收缩：两条药方殊途同归，叠加再赚一点')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/norm-vs-shrinkage.png', dpi=110); plt.close()

# ---------- 实验4：换手率 / 权重稳定性 ----------
n_boot = 40
Ws = {'无约束': [], '‖w‖₁≤1.5': [], 'L2 ‖w‖₂≤0.35': []}
R0 = sample_returns(T_est)
for b in range(n_boot):
    idx = rng.integers(0, T_est, T_est)
    S = np.cov(R0[idx].T)*252
    Ws['无约束'].append(gmv_norm(S))
    Ws['‖w‖₁≤1.5'].append(gmv_norm(S, l1_cap=1.5))
    Ws['L2 ‖w‖₂≤0.35'].append(gmv_norm(S, l2_cap=0.35))
stds = {k: np.mean(np.std(np.array(v), axis=0))*100 for k,v in Ws.items()}
print("Bootstrap 权重标准差(pp):", stds)

fig, axes = plt.subplots(1,2, figsize=(11,4.6))
colors = ['#d62728', '#1f77b4', '#2ca02c']
bars = axes[0].bar(list(stds.keys()), list(stds.values()), color=colors, alpha=0.85)
for b, v in zip(bars, stds.values()):
    axes[0].text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.2f}', ha='center', fontsize=10)
axes[0].set_ylabel('单资产权重 Bootstrap 标准差 (pp)')
axes[0].set_title('权重稳定性：范数约束把估计噪声引起的权重抖动砍掉大半')
axes[0].grid(alpha=0.3, axis='y')
# 抽两个资产画权重散点
W_unc = np.array(Ws['无约束']); W_l1 = np.array(Ws['‖w‖₁≤1.5'])
j1, j2 = 4, 5  # 同板块相邻资产
axes[1].scatter(W_unc[:,j1]*100, W_unc[:,j2]*100, color='#d62728', alpha=0.6, s=30, label='无约束')
axes[1].scatter(W_l1[:,j1]*100, W_l1[:,j2]*100, color='#1f77b4', alpha=0.6, s=30, label='‖w‖₁≤1.5')
axes[1].set_xlabel(f'资产{j1} 权重 (%)'); axes[1].set_ylabel(f'资产{j2} 权重 (%)')
axes[1].set_title('同板块两资产：无约束解在对冲对赌间大幅摇摆')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/weight-stability.png', dpi=110); plt.close()

print("figures saved:", os.listdir(OUT))
