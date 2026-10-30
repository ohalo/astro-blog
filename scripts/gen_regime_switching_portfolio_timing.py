#!/usr/bin/env python3
"""
生成 regime-switching portfolio timing 文章配图（5张PNG）。
输出到 public/images/regime-switching-portfolio-timing/
"""
import json
import os
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from datetime import datetime, timedelta

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'public', 'images', 'regime-switching-portfolio-timing')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 数据生成 ==========
np.random.seed(20261030)
N_DAYS = 2520  # 约10年日数据

# 三状态：牛市(mu=0.0008, vol=0.012), 震荡(mu=0.0001, vol=0.015), 熊市(mu=-0.0005, vol=0.020)
regime_params = [
    (0.0008, 0.012),
    (0.0001, 0.015),
    (-0.0005, 0.020),
]
trans_mat_true = np.array([
    [0.85, 0.12, 0.03],
    [0.15, 0.65, 0.20],
    [0.05, 0.15, 0.80],
])

# 模拟 regime 路径
regime_path_true = np.zeros(N_DAYS, dtype=int)
for t in range(1, N_DAYS):
    regime_path_true[t] = np.random.choice(3, p=trans_mat_true[regime_path_true[t-1]])

# 生成收益率
returns = np.zeros(N_DAYS)
for t in range(N_DAYS):
    returns[t] = np.random.normal(*regime_params[regime_path_true[t]])

# 价格序列
prices = 100 * np.exp(np.cumsum(returns))

# ========== 简化 HMM ( Baum-Welch ) ==========
def train_hmm(y, n_states=3, max_iter=100, tol=1e-5):
    """简化版 HMM，直接输出 Viterbi 路径。"""
    T = len(y)
    N = n_states
    y = np.asarray(y, dtype=float)

    # 初始化
    np.random.seed(20261030)
    A = np.random.dirichlet(np.ones(N) * 10, size=N)
    mus = np.random.randn(N) * 0.001
    sigmas = np.abs(np.random.randn(N)) * 0.005 + 0.008
    pi = np.ones(N) / N

    def log_emit(y_val, k):
        return float(stats.norm.logpdf(y_val, mus[k], sigmas[k]))

    ll_history = []
    for it in range(max_iter):
        # Forward
        log_alpha = np.zeros((T, N))
        log_alpha[0] = np.log(pi) + np.array([log_emit(y[0], k) for k in range(N)])
        for t in range(1, T):
            for n in range(N):
                log_alpha[t, n] = logsumexp(log_alpha[t-1] + np.log(A[:, n])) + log_emit(y[t], n)

        # Backward
        log_beta = np.zeros((T, N))
        for t in range(T-2, -1, -1):
            for n in range(N):
                log_beta[t, n] = logsumexp(np.log(A[n, :]) + np.array([log_emit(y[t+1], m) for m in range(N)]) + log_beta[t+1])

        # Gamma (posterior)
        log_norm = logsumexp(log_alpha[T-1])
        log_gamma = log_alpha + log_beta - log_norm
        gamma = np.exp(log_gamma)

        # 对数似然
        ll = log_norm
        ll_history.append(ll)
        if it > 0 and abs(ll - ll_history[-2]) < tol:
            break

        # M step
        gamma_sum = gamma.sum(axis=0)
        for k in range(N):
            mus[k] = (gamma[:, k] * y).sum() / gamma_sum[k]
            sigmas[k] = float(np.sqrt((gamma[:, k] * (y - mus[k])**2).sum() / gamma_sum[k]))

        # 转移矩阵
        log_alpha_beta = log_alpha + log_beta
        A_new = np.zeros((N, N))
        for t in range(T-1):
            for n in range(N):
                for m in range(N):
                    log_gamma_nm = log_alpha_beta[t, n] + np.log(A[n, m]) + log_emit(y[t+1], m) + log_beta[t+1, m] - log_norm
                    A_new[n, m] += np.exp(log_gamma_nm)
        A_new = A_new / (A_new.sum(axis=1, keepdims=True) + 1e-300)
        A = A_new

    # Viterbi
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = np.log(pi) + np.array([log_emit(y[0], k) for k in range(N)])
    for t in range(1, T):
        for n in range(N):
            cands = delta[t-1] + np.log(A[:, n])
            psi[t, n] = np.argmax(cands)
            delta[t, n] = cands[psi[t, n]] + log_emit(y[t], n)
    path = np.zeros(T, dtype=int)
    path[T-1] = np.argmax(delta[T-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t]]

    return path, mus, sigmas, A, ll_history

viterbi_path, est_mus, est_sigs, est_A, ll_history = train_hmm(returns)

# ========== 回测 ==========
train_end = int(N_DAYS * 0.7)
train_returns = returns[:train_end]
test_returns = returns[train_end:]

# 对每个 regime 训练仓位
regime_weights = {}
for k in range(3):
    regime_data = train_returns[viterbi_path[:train_end] == k]
    if len(regime_data) > 60:
        mu_k = np.mean(regime_data)
        var_k = np.var(regime_data)
        w = mu_k / (5.0 * var_k) if var_k > 0 else 0.5
        regime_weights[k] = float(np.clip(w, 0.1, 1.0))

# OOS 回测
equity_bs = [100.0]
equity_hmm = [100.0]
for t in range(len(test_returns)):
    regime = int(viterbi_path[train_end + t])
    w = regime_weights.get(regime, 0.5)
    equity_bs.append(equity_bs[-1] * (1 + test_returns[t]))
    equity_hmm.append(equity_hmm[-1] * (1 + w * test_returns[t]))

# 统计
def calc_stats(eq_arr):
    eq_arr = np.array(eq_arr)
    daily_ret = np.diff(np.log(eq_arr))
    ann_ret = float(np.mean(daily_ret) * 252 * 100)
    ann_vol = float(np.std(daily_ret) * np.sqrt(252) * 100)
    sharpe = float(np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)) if np.std(daily_ret) > 0 else 0
    max_dd = 0
    peak = eq_arr[0]
    for v in eq_arr:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return ann_ret, ann_vol, max_dd, sharpe

stats_bs = calc_stats(equity_bs)
stats_hmm = calc_stats(equity_hmm)

# 结果 JSON
results = {
    'stats': {
        'equal_weight': {'ann_ret': round(stats_bs[0], 2), 'ann_vol': round(stats_bs[1], 2), 'max_dd': round(stats_bs[2], 2), 'sharpe': round(stats_bs[3], 2)},
        'hmm_timing': {'ann_ret': round(stats_hmm[0], 2), 'ann_vol': round(stats_hmm[1], 2), 'max_dd': round(stats_hmm[2], 2), 'sharpe': round(stats_hmm[3], 2)},
    },
    'regime_mu': [round(float(est_mus[k]), 4) for k in range(3)],
    'regime_sigma': [round(float(est_sigs[k]), 4) for k in range(3)],
}
with open(os.path.join(os.path.dirname(__file__), '..', '_rs_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# ========== 画图 ==========
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

dates = [datetime(2014, 1, 1) + timedelta(days=i) for i in range(N_DAYS)]

# 图1: 价格 + Regime 背景
fig, ax = plt.subplots(figsize=(12, 5))
for k in range(3):
    mask = viterbi_path == k
    ax.fill_between(dates, 0, 500, where=mask, alpha=0.15, color=['green', 'gray', 'red'][k], label=['牛市', '震荡', '熊市'][k])
ax.plot(dates, prices, color='#333333', linewidth=1.2)
ax.set_ylabel('价格', fontsize=11)
ax.set_title('HMM Regime 识别：牛/熊/震荡三步分类', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_regime_detection.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图2: 转移矩阵热图
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(est_A, cmap='YlOrRd', vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{est_A[i, j]:.2f}', ha='center', va='center', fontsize=12, fontweight='bold' if abs(est_A[i,j]-est_A[i,:].max())<0.01 else 'normal')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['牛市', '震荡', '熊市'])
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['牛市', '震荡', '熊市'])
ax.set_xlabel('下一状态', fontsize=11)
ax.set_ylabel('当前状态', fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle('转移矩阵 A：状态间切换概率', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_transition_matrix.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图3: 发射分布对比
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(-0.03, 0.03, 500)
colors = ['green', 'gray', 'red']
labels = ['牛市', '震荡', '熊市']
for k in range(3):
    y = stats.norm.pdf(x, est_mus[k], est_sigs[k])
    ax.fill_between(x, y, alpha=0.3, color=colors[k])
    ax.plot(x, y, color=colors[k], linewidth=2, label=f'{labels[k]} (μ={est_mus[k]*1000:.3f}%, σ={est_sigs[k]*100:.2f}%)')
ax.set_xlabel('日收益率', fontsize=11)
ax.set_ylabel('概率密度', fontsize=11)
ax.set_title('各 Regime 收益率发射分布', fontsize=13, fontweight='bold')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_emission_distributions.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图4: 回测对比
fig, ax = plt.subplots(figsize=(10, 5))
dates_test = dates[train_end:]
eq_bs = np.array(equity_bs)
eq_hmm = np.array(equity_hmm)
ax.plot(dates_test, eq_bs[1:], label='等权基准', color='#999999', linewidth=1.5, linestyle='--')
ax.plot(dates_test, eq_hmm[1:], label='HMM Regime 择时', color='#D32E00', linewidth=1.5)
ax.set_ylabel('净值', fontsize=11)
ax.set_title('Out-of-Sample 回测对比（2022-2024）', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.text(0.02, 0.98, f'等权: 年化{stats_bs[0]:.1f}% | Sharpe{stats_bs[3]:.2f}\nHMM: 年化{stats_hmm[0]:.1f}% | Sharpe{stats_hmm[3]:.2f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_backtest_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图5: 对数似然收敛
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ll_history, color='#D32E00', linewidth=1.5)
ax.set_xlabel('EM 迭代轮次', fontsize=11)
ax.set_ylabel('对数似然', fontsize=11)
ax.set_title('Baum-Welch EM 训练收敛', fontsize=13, fontweight='bold')
ax.axhline(ll_history[-1], color='green', linestyle='--', alpha=0.5, label=f'收敛值: {ll_history[-1]:.1f}')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_ll_convergence.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Done. 5 张图已保存至 {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f} ({os.path.getsize(os.path.join(OUTPUT_DIR, f))} bytes)")
