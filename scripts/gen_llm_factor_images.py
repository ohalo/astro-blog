#!/usr/bin/env python3
"""
生成第二篇文章配图：LLM作为因子矿工
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang SC', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(2024)

# 图1：文本情感信号 vs 价格走势
fig, ax1 = plt.subplots(figsize=(12, 5))

days = 120
dates = np.arange(days)
# 模拟价格
price = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.008 + 0.0002))
# 模拟LLM情感信号（领先1-2天）
sentiment = np.convolve(np.diff(price, prepend=price[0]) / price[0] + np.random.randn(days)*0.003,
                        np.ones(3)/3, mode='same') * 50 + 0.02
sentiment = np.clip(sentiment, -0.08, 0.08)

ax1.plot(dates, price, 'b-', lw=2, label='股价')
ax1.set_ylabel('股价', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xlabel('交易日', fontsize=12)

ax2 = ax1.twinx()
ax2.bar(dates, sentiment, alpha=0.4, color='green' if np.mean(sentiment) > 0 else 'red', width=0.8, label='LLM情感得分')
ax2.axhline(y=0, color='k', linestyle='-', lw=0.5)
ax2.set_ylabel('LLM情感信号', color='g', fontsize=12)
ax2.tick_params(axis='y', labelcolor='g')
ax2.set_ylim(-0.1, 0.1)

ax1.set_title('LLM情感信号与价格走势的领先滞后关系', fontsize=13, fontweight='bold')
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88), fontsize=10)
ax1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/llm-factor-miner/llm_sentiment_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("图1已保存: llm_sentiment_price.png")

# 图2：Prompt工程策略回测表现
fig, ax = plt.subplots(figsize=(10, 6))

# 模拟三种策略净值
days_bt = 252
bench_ret = np.random.randn(days_bt) * 0.008 + 0.0001
# 基础情感策略
basic_ret = np.sign(np.random.randn(days_bt)) * 0.005 + bench_ret * 0.3
basic_nav = np.cumprod(1 + basic_ret)
# Prompt工程优化策略（更好的信号质量）
prompt_ret = np.sign(np.random.randn(days_bt) * 1.2 + 0.3) * 0.006 + bench_ret * 0.2
prompt_nav = np.cumprod(1 + prompt_ret)
# 买入持有
buyhold_nav = np.cumprod(1 + bench_ret)

ax.plot(buyhold_nav, 'k-', lw=2, label='买入持有', alpha=0.6)
ax.plot(basic_nav, 'b--', lw=2, label='基础情感策略')
ax.plot(prompt_nav, 'r-', lw=2.5, label='Prompt工程优化策略')

ax.set_xlabel('交易日', fontsize=12)
ax.set_ylabel('累计净值', fontsize=12)
ax.set_title('策略回测：Prompt工程对信号质量的提升', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# 添加收益标注
prompt_cagr = (prompt_nav[-1] - 1) * 100
basic_cagr = (basic_nav[-1] - 1) * 100
bh_cagr = (buyhold_nav[-1] - 1) * 100
ax.annotate(f'Prompt策略: +{prompt_cagr:.1f}%', xy=(days_bt-1, prompt_nav[-1]),
           xytext=(10, 0), textcoords='offset points', fontsize=10, color='red', fontweight='bold')
ax.annotate(f'基准: +{bh_cagr:.1f}%', xy=(days_bt-1, buyhold_nav[-1]),
           xytext=(10, 0), textcoords='offset points', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/llm-factor-miner/prompt_backtest.png', dpi=150, bbox_inches='tight')
plt.close()
print("图2已保存: prompt_backtest.png")

# 图3：不同Prompt设计的信息提取效率
fig, ax = plt.subplots(figsize=(10, 6))

prompt_types = ['直接提问\n(零样本)', 'Few-shot\n示例', '链式思考\nCoT', '角色扮演\n(分析师)', '结构化输出\n(JSON)']
# 模拟IC（信息系数）
ics = [0.032, 0.048, 0.061, 0.055, 0.058]
ic_errors = [0.008, 0.009, 0.010, 0.009, 0.009]

bars = ax.bar(prompt_types, ics, yerr=ic_errors, capsize=5,
              color=['lightcoral', 'gold', 'mediumseagreen', 'skyblue', 'plum'],
              edgecolor='black', linewidth=1.2)

ax.axhline(y=0.05, color='r', linestyle='--', lw=1.5, label='可交易阈值 (IC=0.05)')
ax.set_ylabel('信息系数 (IC)', fontsize=12)
ax.set_title('不同Prompt设计的信息提取效率对比', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.09)

# 在柱子上标注数值
for bar, ic in zip(bars, ics):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
            f'{ic:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/llm-factor-miner/prompt_ic_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("图3已保存: prompt_ic_comparison.png")

print("所有配图生成完成!")
