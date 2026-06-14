#!/usr/bin/env python3
"""生成量化文章配图"""
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_order_flow_chart():
    """生成订单流策略图表"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 模拟订单簿数据
    time = np.arange(200)
    bid_size = 500 + 100 * np.sin(time * 0.05) + np.random.randn(200) * 30
    ask_size = 500 + 100 * np.cos(time * 0.05) + np.random.randn(200) * 30
    
    # 计算OBI
    obi = (bid_size - ask_size) / (bid_size + ask_size)
    
    ax1.plot(time, bid_size, label='Bid Size', color='green', alpha=0.7, linewidth=2)
    ax1.plot(time, ask_size, label='Ask Size', color='red', alpha=0.7, linewidth=2)
    ax1.set_ylabel('Order Size', fontsize=12)
    ax1.set_title('Order Book Dynamics (Order Book Imbalance)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time, obi, color='blue', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Long Threshold')
    ax2.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5, label='Short Threshold')
    ax2.set_ylabel('OBI', fontsize=12)
    ax2.set_xlabel('Time (Ticks)', fontsize=12)
    ax2.set_title('Order Flow Imbalance (OFI) Indicator', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'public/images/2026-06-14-hft-order-flow/order_flow_chart.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Generated: {output_path}")
    plt.close()

def generate_risk_management_chart():
    """生成风险管理图表（VaR vs CVaR）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 生成收益率数据（t分布，更厚的尾部）
    np.random.seed(42)
    returns = np.random.standard_t(df=4, size=10000)
    
    # 计算VaR和CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # 左图：收益率分布直方图
    ax1.hist(returns, bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=var_95, color='red', linestyle='--', linewidth=2, label=f'VaR 95% = {var_95:.3f}')
    ax1.axvline(x=cvar_95, color='darkred', linestyle='--', linewidth=2, label=f'CVaR 95% = {cvar_95:.3f}')
    ax1.fill_betweenx([0, 0.4], var_95, returns.min(), alpha=0.3, color='red', label='Tail Risk')
    ax1.set_xlabel('Return', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Return Distribution with VaR and CVaR', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：累积损失
    sorted_returns = np.sort(returns)
    cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    
    ax2.plot(sorted_returns, cumulative_prob, linewidth=2, color='blue')
    ax2.axvline(x=var_95, color='red', linestyle='--', linewidth=2, label='VaR 95%')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Return', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.1])
    
    plt.tight_layout()
    output_path = 'public/images/2026-06-14-var-cvar-risk-management/var_cvar_chart.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Generated: {output_path}")
    plt.close()

def generate_portfolio_risk_chart():
    """生成投资组合风险分解图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 模拟投资组合成分
    assets = ['Stock A', 'Stock B', 'Stock C', 'Stock D', 'Stock E']
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    risks = np.array([0.15, 0.12, 0.18, 0.10, 0.14])
    contributions = weights * risks  # 简化计算风险贡献
    
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    wedges, texts, autotexts = ax.pie(contributions, labels=assets, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    
    ax.set_title('Portfolio Risk Contribution Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = 'public/images/2026-06-14-var-cvar-risk-management/portfolio_risk.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Generated: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("Generating images for quantitative articles...")
    generate_order_flow_chart()
    generate_risk_management_chart()
    generate_portfolio_risk_chart()
    print("All images generated successfully!")
