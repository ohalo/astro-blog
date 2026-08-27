#!/usr/bin/env python3
"""
生成第一篇文章配图：扩散模型收益分布建模
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang SC', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# 图1：真实收益分布 vs 正态分布（厚尾特征）
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 合成厚尾收益数据
returns = np.random.standard_t(df=4, size=5000) * 0.015

ax1 = axes[0]
ax1.hist(returns, bins=80, density=True, alpha=0.6, color='steelblue', edgecolor='white', label='实际收益分布')
x_range = np.linspace(returns.min(), returns.max(), 500)
from scipy import stats
ax1.plot(x_range, stats.norm.pdf(x_range, returns.mean(), returns.std()), 'r--', lw=2, label=f'正态近似 (σ={returns.std()*100:.2f}%)')
ax1.set_xlabel('日收益率', fontsize=12)
ax1.set_ylabel('概率密度', fontsize=12)
ax1.set_title('收益分布的厚尾特征：实际 vs 正态', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 图1右：Q-Q图
ax2 = axes[1]
stats.probplot(returns, dist="norm", plot=ax2)
ax2.set_title('Q-Q图：尾部偏离正态的程度', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/score-matching-return-distribution/return_distribution_tail.png', dpi=150, bbox_inches='tight')
plt.close()
print("图1已保存: return_distribution_tail.png")

# 图2：分数匹配估计 vs 真实密度
fig, ax = plt.subplots(figsize=(10, 6))

# 真实密度（t分布）
x = np.linspace(-0.08, 0.08, 400)
true_pdf = stats.t.pdf(x / 0.015, df=4) / (1/0.015)

# 模拟分数匹配估计（用核密度+score估计）
from sklearn.neighbors import KernelDensity
kde = KernelDensity(bandwidth=0.008, kernel='gaussian').fit(returns.reshape(-1, 1))
kde_pdf = np.exp(kde.score_samples(x.reshape(-1, 1)))

# 数值微分估计score
log_pdf = np.log(kde_pdf + 1e-10)
score_est = np.gradient(log_pdf, x)

ax.plot(x, true_pdf, 'b-', lw=2.5, label='真实密度 (t₄, σ=1.5%)')
ax.plot(x, kde_pdf, 'r--', lw=2, label='核密度估计 (KDE)')
ax.set_xlabel('日收益率', fontsize=12)
ax.set_ylabel('概率密度', fontsize=12)
ax.set_title('分数匹配：用 Score 估计整条收益分布', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# 添加注释
ax.annotate('左尾：极端负收益\n概率被正态严重低估', xy=(-0.06, 2), fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.annotate('右尾：极端正收益\n同样厚于正态', xy=(0.045, 1.5), fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/score-matching-return-distribution/score_matching_density.png', dpi=150, bbox_inches='tight')
plt.close()
print("图2已保存: score_matching_density.png")

# 图3：VaR估计对比（正态 vs 分数匹配 vs 历史）
fig, ax = plt.subplots(figsize=(10, 6))

alphas = np.array([0.10, 0.05, 0.025, 0.01, 0.005, 0.001])
# 历史VaR
hist_vars = [np.percentile(returns, a*100) for a in alphas]
# 正态VaR
normal_vars = [stats.norm.ppf(a, returns.mean(), returns.std()) for a in alphas]
# t分布VaR（分数匹配近似）
t_vars = [stats.t.ppf(a, df=4, loc=returns.mean(), scale=returns.std()) for a in alphas]

ax.plot(alphas * 100, np.array(hist_vars) * 100, 'bo-', lw=2, markersize=8, label='历史 VaR')
ax.plot(alphas * 100, np.array(normal_vars) * 100, 'rs--', lw=2, markersize=8, label='正态 VaR')
ax.plot(alphas * 100, np.array(t_vars) * 100, 'g^-.', lw=2, markersize=8, label='分数匹配 / t-分布 VaR')

ax.set_xlabel('置信水平 (%)', fontsize=12)
ax.set_ylabel('VaR (%)', fontsize=12)
ax.set_title('尾部风险估计：正态假设会系统性地低估极端损失', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.invert_xaxis()

# 添加差异标注
for i, (h, n) in enumerate(zip(hist_vars, normal_vars)):
    if i % 2 == 0:
        ax.annotate(f'低估{abs(h-n)*100:.1f}pp', xy=(alphas[i]*100, n*100),
                   xytext=(10, 10), textcoords='offset points', fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/score-matching-return-distribution/var_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("图3已保存: var_comparison.png")

print("所有配图生成完成!")
