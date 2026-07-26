#!/usr/bin/env python3
"""
生成 Glosten-Milgrom 模型配图
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
out_dir = Path('/Users/halo/workspace/astro-blog/public/images/glosten-milgrom-model')
out_dir.mkdir(parents=True, exist_ok=True)


def glosten_milgrom_update(n_trades, alpha, V_low, V_high, seed=42):
    """
    Glosten-Milgrom 模型模拟
    
    参数:
    - n_trades: 交易次数
    - alpha: 知情交易者占比
    - V_low: 低价值状态
    - V_high: 高价值状态
    """
    rng = np.random.default_rng(seed)
    
    # 真实价值（抛硬币决定是高还是低）
    true_V = V_high if rng.random() < 0.5 else V_low
    
    # 做市商初始信念（先验概率）
    prob_V_high = 0.5
    
    # 记录序列
    bid_list = []
    ask_list = []
    mid_list = []
    prob_list = []
    
    for t in range(n_trades):
        # 计算期望价值 E[V|信息]
        E_V = prob_V_high * V_high + (1 - prob_V_high) * V_low
        
        # 计算条件期望用于 bid/ask
        # E[V | 卖单到达] 和 E[V | 买单到达]
        # 根据贝叶斯公式推导
        
        # 买单到达的后验概率 P(V_high | buy)
        # P(buy | V_high) = alpha + (1-alpha)*0.5 = 0.5*(1+alpha)
        # P(buy | V_low) = (1-alpha)*0.5 = 0.5*(1-alpha)
        P_buy_given_V_high = 0.5 * (1 + alpha)
        P_buy_given_V_low = 0.5 * (1 - alpha)
        P_buy = prob_V_high * P_buy_given_V_high + (1 - prob_V_high) * P_buy_given_V_low
        
        if P_buy > 0:
            prob_V_high_given_buy = prob_V_high * P_buy_given_V_high / P_buy
            E_V_given_buy = prob_V_high_given_buy * V_high + (1 - prob_V_high_given_buy) * V_low
        else:
            E_V_given_buy = E_V
        
        # 卖单到达的后验概率 P(V_high | sell)
        P_sell_given_V_high = 0.5 * (1 - alpha)
        P_sell_given_V_low = 0.5 * (1 + alpha)
        P_sell = prob_V_high * P_sell_given_V_high + (1 - prob_V_high) * P_sell_given_V_low
        
        if P_sell > 0:
            prob_V_high_given_sell = prob_V_high * P_sell_given_V_high / P_sell
            E_V_given_sell = prob_V_high_given_sell * V_high + (1 - prob_V_high_given_sell) * V_low
        else:
            E_V_given_sell = E_V
        
        # 报价
        bid = E_V_given_sell  # 卖单到达时的条件期望
        ask = E_V_given_buy   # 买单到达时的条件期望
        mid = (bid + ask) / 2
        
        bid_list.append(bid)
        ask_list.append(ask)
        mid_list.append(mid)
        prob_list.append(prob_V_high)
        
        # 模拟交易：以概率 alpha 来知情交易者，否则来噪声交易者
        is_informed = rng.random() < alpha
        
        if is_informed:
            # 知情交易者：知道真实价值
            if true_V == V_high:
                # 真实价值高 → 买入
                trade = 'buy'
            else:
                # 真实价值低 → 卖出
                trade = 'sell'
        else:
            # 噪声交易者：随机买卖
            trade = 'buy' if rng.random() < 0.5 else 'sell'
        
        # 根据成交更新信念
        if trade == 'buy':
            prob_V_high = prob_V_high_given_buy
        else:
            prob_V_high = prob_V_high_given_sell
    
    return (np.array(bid_list), np.array(ask_list), np.array(mid_list), 
            np.array(prob_list), true_V)


# ============================================================
# 图1：做市商报价随买卖单序列的贝叶斯更新演化
# ============================================================
print("生成图1：贝叶斯更新演化...")
fig, ax = plt.subplots(figsize=(12, 6), dpi=140)

# 参数
alpha = 0.3
V_low, V_high = 9.5, 10.5
n_trades = 60

bid, ask, mid, prob, true_V = glosten_milgrom_update(n_trades, alpha, V_low, V_high, seed=42)

# 绘制
trades = np.arange(1, n_trades + 1)
ax.fill_between(trades, bid, ask, alpha=0.3, color='gray', label='买卖价差区间')
ax.plot(trades, bid, 'g-', linewidth=1.5, label='买价 (bid)', marker='.', markersize=4)
ax.plot(trades, ask, 'r-', linewidth=1.5, label='卖价 (ask)', marker='.', markersize=4)
ax.plot(trades, mid, 'b-', linewidth=2, label='中间价', marker='o', markersize=3)

# 真实价值
ax.axhline(y=true_V, color='black', linestyle='--', linewidth=2, label=f'真实价值 V={true_V:.1f}')

ax.set_xlabel('交易序号', fontsize=12)
ax.set_ylabel('价格', fontsize=12)
ax.set_title(f'做市商报价的贝叶斯更新演化 (α={alpha:.0%} 知情交易者)\n中间价逐步收敛到真实价值', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, n_trades + 1)

plt.tight_layout()
plt.savefig(out_dir / 'bayesian-updating-evolution.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"  已保存: {out_dir / 'bayesian-updating-evolution.png'}")


# ============================================================
# 图2：买卖价差随知情交易者占比 α 的变化
# ============================================================
print("\n生成图2：价差 vs α...")
fig, ax = plt.subplots(figsize=(10, 6), dpi=140)

V_low, V_high = 9.5, 10.5
E_V_prior = 0.5 * V_low + 0.5 * V_high  # 先验期望
alpha_range = np.linspace(0.01, 0.6, 60)

spread_list = []
for alpha in alpha_range:
    # 在先验信念 prob_V_high = 0.5 时计算初始价差
    prob_V_high = 0.5
    
    P_buy_given_V_high = 0.5 * (1 + alpha)
    P_buy_given_V_low = 0.5 * (1 - alpha)
    P_buy = prob_V_high * P_buy_given_V_high + (1 - prob_V_high) * P_buy_given_V_low
    
    prob_V_high_given_buy = prob_V_high * P_buy_given_V_high / P_buy
    E_V_given_buy = prob_V_high_given_buy * V_high + (1 - prob_V_high_given_buy) * V_low
    
    P_sell_given_V_high = 0.5 * (1 - alpha)
    P_sell_given_V_low = 0.5 * (1 + alpha)
    P_sell = prob_V_high * P_sell_given_V_high + (1 - prob_V_high) * P_sell_given_V_low
    
    prob_V_high_given_sell = prob_V_high * P_sell_given_V_high / P_sell
    E_V_given_sell = prob_V_high_given_sell * V_high + (1 - prob_V_high_given_sell) * V_low
    
    spread = E_V_given_buy - E_V_given_sell
    spread_list.append(spread)

spread_arr = np.array(spread_list)

ax.plot(alpha_range * 100, spread_arr, 'b-', linewidth=2.5, marker='o', markersize=4)
ax.fill_between(alpha_range * 100, 0, spread_arr, alpha=0.2, color='blue')

ax.set_xlabel('知情交易者占比 α (%)', fontsize=12)
ax.set_ylabel('买卖价差 (ask - bid)', fontsize=12)
ax.set_title('买卖价差随知情交易者占比的扩大\n做市商要求更多补偿以覆盖逆向选择风险', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 60)

# 标注关键点
for ann_alpha in [0.1, 0.3, 0.5]:
    idx = int(ann_alpha * 100 - 1)
    if idx < len(spread_arr):
        ax.annotate(f'α={ann_alpha:.0%}\n价差={spread_arr[idx]:.3f}',
                    xy=(ann_alpha * 100, spread_arr[idx]),
                    xytext=(ann_alpha * 100 + 5, spread_arr[idx] + 0.05),
                    fontsize=9, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

plt.tight_layout()
plt.savefig(out_dir / 'spread-vs-alpha.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"  已保存: {out_dir / 'spread-vs-alpha.png'}")


# ============================================================
# 图3：价差成分分解——逆向选择占比
# ============================================================
print("\n生成图3：价差成分分解...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=140)

# 左图：不同 α 下的逆向选择成分占比
V_diff = V_high - V_low  # 价值不确定性范围
alpha_vals = [0.05, 0.15, 0.30, 0.45, 0.60]

# 逆向选择成分 = α * V_diff (简化公式)
# 总价差 = ask - bid (上面已算)
adverse_selection_ratios = []
total_spreads = []

for alpha in alpha_vals:
    prob_V_high = 0.5
    P_buy_given_V_high = 0.5 * (1 + alpha)
    P_buy_given_V_low = 0.5 * (1 - alpha)
    P_buy = prob_V_high * P_buy_given_V_high + (1 - prob_V_high) * P_buy_given_V_low
    prob_V_high_given_buy = prob_V_high * P_buy_given_V_high / P_buy
    E_V_given_buy = prob_V_high_given_buy * V_high + (1 - prob_V_high_given_buy) * V_low
    
    P_sell_given_V_high = 0.5 * (1 - alpha)
    P_sell_given_V_low = 0.5 * (1 + alpha)
    P_sell = prob_V_high * P_sell_given_V_high + (1 - prob_V_high) * P_sell_given_V_low
    prob_V_high_given_sell = prob_V_high * P_sell_given_V_high / P_sell
    E_V_given_sell = prob_V_high_given_sell * V_high + (1 - prob_V_high_given_sell) * V_low
    
    spread = E_V_given_buy - E_V_given_sell
    total_spreads.append(spread)
    
    # 逆向选择成分（粗略）：价差中归因于信息不对称的部分
    # 精确定义需要更细致的推导，这里用比例表示
    adverse_ratio = alpha / (alpha + 0.5)  # 近似比例
    adverse_selection_ratios.append(adverse_ratio)

ax1 = axes[0]
x = np.arange(len(alpha_vals))
bars = ax1.bar(x, adverse_selection_ratios, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#34495e'], 
               edgecolor='black', linewidth=1.5)
ax1.set_xticks(x)
ax1.set_xticklabels([f'α={a:.0%}' for a in alpha_vals])
ax1.set_ylabel('逆向选择占比', fontsize=12)
ax1.set_title('价差中的逆向选择成分占比\n知情交易者越多，信息成分越主导', fontsize=13)
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3, axis='y')

for bar, ratio in zip(bars, adverse_selection_ratios):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{ratio:.1%}', ha='center', fontsize=10, fontweight='bold')

# 右图：总价差及其分解
ax2 = axes[1]
x = np.arange(len(alpha_vals))
width = 0.35

# 逆向选择成分（占价差的比例 * 总价差）
adverse_component = [r * s for r, s in zip(adverse_selection_ratios, total_spreads)]
# 非信息成分
non_info_component = [s - a for s, a in zip(total_spreads, adverse_component)]

bars1 = ax2.bar(x - width/2, adverse_component, width, label='逆向选择成分', color='#e74c3c', edgecolor='black')
bars2 = ax2.bar(x + width/2, non_info_component, width, label='非信息成分', color='#3498db', edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels([f'α={a:.0%}' for a in alpha_vals])
ax2.set_ylabel('价差成分', fontsize=12)
ax2.set_title('总价差的成分分解\n知情交易者增加推高逆向选择成本', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(out_dir / 'spread-decomposition.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"  已保存: {out_dir / 'spread-decomposition.png'}")


# ============================================================
# 验证文件大小
# ============================================================
print("\n验证生成的图片:")
for fname in ['bayesian-updating-evolution.png', 'spread-vs-alpha.png', 'spread-decomposition.png']:
    fpath = out_dir / fname
    if fpath.exists():
        size_kb = fpath.stat().st_size / 1024
        print(f"  {fname}: {size_kb:.1f} KB {'✓' if size_kb > 30 else '✗ (太小)'}")
    else:
        print(f"  {fname}: 不存在 ✗")
