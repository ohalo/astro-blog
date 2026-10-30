#!/usr/bin/env python3
"""
生成 monte-carlo-error-quantification 文章配图（5张PNG）。
输出到 public/images/monte-carlo-error-quantification/

依赖：numpy, scipy, matplotlib（系统 python3 均有）
"""
import json
import os
import sys
import numpy as np
from scipy import stats
from scipy.optimize import minimize

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'public', 'images', 'monte-carlo-error-quantification')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 模拟数据 ==========
np.random.seed(20261030)
N_PATHS = 10000
T = 252

# 几何布朗运动模拟
mu, sigma = 0.0005, 0.015
daily_returns = np.random.normal(mu, sigma, size=(N_PATHS, T))
price_paths = 100 * np.exp(np.cumsum(daily_returns, axis=1))

# 每条路径的年化 Sharpe
final_prices = price_paths[:, -1] / price_paths[:, 0] - 1
sharpe_values = final_prices * np.sqrt(252) / sigma

# ========== Bootstrap 置信区间 ==========
def bootstrap_ci(values, b=10000, alpha=0.05):
    n = len(values)
    boot_est = np.zeros(b)
    for i in range(b):
        sample = np.random.choice(values, size=n, replace=True)
        boot_est[i] = sample.mean()
    point_est = values.mean()
    se = boot_est.std(ddof=1)
    ci_lo = np.percentile(boot_est, 100 * alpha / 2)
    ci_hi = np.percentile(boot_est, 100 * (1 - alpha / 2))
    return point_est, ci_lo, ci_hi, se

sharpe_point, sharpe_lo, sharpe_hi, sharpe_se = bootstrap_ci(sharpe_values)

# ========== Batch Mean 法 ==========
def batch_mean_se(values, k=50):
    n = len(values)
    m = n // k
    batches = values[:k*m].reshape(k, m)
    batch_means = batches.mean(axis=1)
    se = batch_means.std(ddof=1) / np.sqrt(k)
    return se, batch_means

se_batch, batch_means = batch_mean_se(sharpe_values)

# ========== 收敛诊断 ==========
def convergence_diag(values, step=100):
    n_vals = np.arange(step, len(values) + 1, step)
    ci_widths = []
    for n in n_vals:
        sample = values[:n]
        se = sample.std(ddof=1) / np.sqrt(n)
        ci_widths.append(1.96 * se * 2)
    return n_vals, np.array(ci_widths)

n_conv, ci_widths = convergence_diag(sharpe_values)
convergence_idx = None
for i in range(5, len(ci_widths)):
    if np.max(np.abs(np.diff(ci_widths[i-5:i]))) / ci_widths[i-1] < 0.01:
        convergence_idx = i
        break

# ========== 结果 JSON ==========
results = {
    'sharpe_point': round(float(sharpe_point), 3),
    'sharpe_ci_lo': round(float(sharpe_lo), 3),
    'sharpe_ci_hi': round(float(sharpe_hi), 3),
    'sharpe_se': round(float(sharpe_se), 4),
    'se_batch': round(float(se_batch), 4),
    'convergence_n': int(n_conv[convergence_idx]) if convergence_idx is not None else int(len(sharpe_values)),
    'final_ci_width': round(float(ci_widths[-1]), 4),
}
with open(os.path.join(os.path.dirname(__file__), '..', '_mc_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# ========== 画图 ==========
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 图1: Bootstrap 分布
fig, ax = plt.subplots(figsize=(9, 5))
boot_dist = np.array([np.random.choice(sharpe_values, size=len(sharpe_values), replace=True).mean() for _ in range(10000)])
ax.hist(boot_dist, bins=80, color='#D32E00', alpha=0.8, edgecolor='white', linewidth=0.5)
ax.axvline(sharpe_point, color='black', linestyle='--', linewidth=2, label=f'点估计: {sharpe_point:.3f}')
ax.axvline(sharpe_lo, color='green', linestyle=':', linewidth=1.5, label=f'95% CI: [{sharpe_lo:.3f}, {sharpe_hi:.3f}]')
ax.axvline(sharpe_hi, color='green', linestyle=':', linewidth=1.5)
ax.fill_betweenx([ax.get_ylim()[0], ax.get_ylim()[1]], sharpe_lo, sharpe_hi, alpha=0.15, color='green')
ax.set_xlabel('Sharpe 率 (Bootstrap 重抽样)', fontsize=12)
ax.set_ylabel('频次', fontsize=12)
ax.set_title('Bootstrap 置信区间：Sharpe 率的抽样分布', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_bootstrap_dist.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图2: 误差随 sqrt(N) 衰减
fig, ax = plt.subplots(figsize=(8, 5))
N_range = np.array([100, 500, 1000, 2000, 5000, 10000, 20000, 50000])
se_values = sharpe_values.std(ddof=1) / np.sqrt(N_range)
ci_widths_theory = 1.96 * 2 * se_values
ax.loglog(N_range, ci_widths_theory, color='#D32E00', marker='o', linewidth=2, markersize=6, label='95% CI 宽度')
ax.loglog(N_range, se_values, color='blue', marker='s', linewidth=1.5, markersize=5, alpha=0.7, label='标准误 SE')
ax.set_xlabel('样本量 N（对数轴）', fontsize=12)
ax.set_ylabel('误差宽度（对数轴）', fontsize=12)
ax.set_title('蒙特卡洛误差随 N 的收敛：$1/\\sqrt{N}$ 定律', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(100, 60000)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_convergence_rate.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图3: Batch Mean 分布
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(batch_means, bins=30, color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.5, density=True)
x = np.linspace(batch_means.min(), batch_means.max(), 200)
ax.plot(x, stats.norm.pdf(x, batch_means.mean(), batch_means.std(ddof=1)),
        color='red', linewidth=2, label=f'正态拟合 (μ={batch_means.mean():.3f}, σ={batch_means.std(ddof=1):.4f})')
ax.axvline(sharpe_point, color='#D32E00', linestyle='--', linewidth=2, label=f'全量估计: {sharpe_point:.3f}')
ax.set_xlabel('Batch 均值（50 批 × 200 路径）', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.set_title('Batch Mean 法：处理相关样本的方差估计', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_batch_mean_dist.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图4: 收敛诊断
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(n_conv, ci_widths, color='#D32E00', linewidth=2, marker='o', markersize=4)
ax.axvline(n_conv[convergence_idx] if convergence_idx is not None else 10000,
           color='green', linestyle='--', linewidth=2,
           label=f'收敛于 N≈{n_conv[convergence_idx] if convergence_idx is not None else 10000}')
ax.set_xlabel('样本量 N', fontsize=12)
ax.set_ylabel('95% CI 宽度', fontsize=12)
ax.set_title('蒙特卡洛收敛诊断：CI 宽度随样本量变化', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_convergence_diagnostic.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 图5: 不同 N 下的 Sharpe 估计波动
fig, ax = plt.subplots(figsize=(9, 5))
np.random.seed(20261031)
estimates = []
for N in range(100, min(5001, len(sharpe_values)), 100):
    sample = np.random.choice(sharpe_values, size=N, replace=False)
    estimates.append(sample.mean())
estimates = np.array(estimates)
N_plot = np.arange(100, min(5001, len(sharpe_values)), 100)
# 滚动 5 点 CI
ci_upper = []
ci_lower = []
for i in range(len(estimates)):
    start = max(0, i - 2)
    window = estimates[start:i+1]
    ci_upper.append(window.mean() + 1.96 * window.std(ddof=1) / np.sqrt(len(window)))
    ci_lower.append(window.mean() - 1.96 * window.std(ddof=1) / np.sqrt(len(window)))
ax.fill_between(N_plot, ci_lower, ci_upper, alpha=0.2, color='green')
ax.plot(N_plot, estimates, color='#D32E00', linewidth=1.5, label='Sharpe 估计值')
ax.axhline(sharpe_point, color='black', linestyle='--', linewidth=1.5, label=f'全量估计: {sharpe_point:.3f}')
ax.set_xlabel('样本量 N', fontsize=12)
ax.set_ylabel('Sharpe 率估计', fontsize=12)
ax.set_title('蒙特卡洛估计稳定性：小样本 vs 大样本', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_estimation_stability.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Done. 5 张图已保存至 {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f} ({os.path.getsize(os.path.join(OUTPUT_DIR, f))} bytes)")
