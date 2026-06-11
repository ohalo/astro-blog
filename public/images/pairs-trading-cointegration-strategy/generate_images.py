import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# 图1: 配对交易权益曲线示例
np.random.seed(42)
dates = pd.date_range('2025-01-01', '2026-06-12', freq='D')
returns = np.random.normal(0.0008, 0.008, len(dates))
cumulative = (1 + pd.Series(returns, index=dates)).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(cumulative.index, cumulative.values, linewidth=2.5, color='#1f77b4')
plt.fill_between(cumulative.index, 1, cumulative.values, alpha=0.3, color='#1f77b4')
plt.title('Pairs Trading Strategy: Cumulative Returns (2025-2026)', fontsize=15, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Returns (Ratio)', fontsize=12)
plt.grid(True, alpha=0.4, linestyle='--')
plt.tight_layout()
plt.savefig('equity_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("Image 1: equity_curve.png generated")

# 图2: 配对交易原理示意图（价差均值回归）
plt.figure(figsize=(14, 10))

# 生成两个协整股票价格
n = 250
t = np.arange(n)
np.random.seed(123)
stock_a = 50 + 0.02 * t + np.cumsum(np.random.normal(0, 0.3, n))
stock_b = 30 + 0.015 * t + np.cumsum(np.random.normal(0, 0.3, n))
spread = stock_a - 1.5 * stock_b
mean_spread = spread.mean()
std_spread = spread.std()
z_score = (spread - mean_spread) / std_spread

# 子图1: 股票价格
plt.subplot(3, 1, 1)
plt.plot(t, stock_a, label='Stock A (601398.SH)', linewidth=2.5, color='#2E86AB')
plt.plot(t, stock_b, label='Stock B (601939.SH)', linewidth=2.5, color='#A23B72')
plt.legend(loc='upper left', fontsize=11)
plt.title('Pair Trading: Stock Price Series', fontsize=13, fontweight='bold', pad=10)
plt.ylabel('Price (CNY)', fontsize=11)
plt.grid(True, alpha=0.3, linestyle=':')
plt.xlim(0, n)

# 子图2: 价差及交易区间
plt.subplot(3, 1, 2)
plt.plot(t, spread, linewidth=2.5, color='#3A7D34')
plt.axhline(y=mean_spread, color='#F7766B', linestyle='--', linewidth=2, label=f'Mean ({mean_spread:.2f})')
plt.axhline(y=mean_spread + 2*std_spread, color='#F7766B', linestyle=':', linewidth=2, label=f'+2σ ({mean_spread + 2*std_spread:.2f})')
plt.axhline(y=mean_spread - 2*std_spread, color='#F7766B', linestyle=':', linewidth=2, label=f'-2σ ({mean_spread - 2*std_spread:.2f})')
plt.fill_between(t, mean_spread - 2*std_spread, mean_spread + 2*std_spread, alpha=0.2, color='gray', label='Trading Zone')
plt.title('Price Spread (A - 1.5×B)', fontsize=13, fontweight='bold', pad=10)
plt.ylabel('Spread (CNY)', fontsize=11)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3, linestyle=':')
plt.xlim(0, n)

# 子图3: Z-Score及交易信号
plt.subplot(3, 1, 3)
plt.plot(t, z_score, linewidth=2.5, color='#5B2C6F')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.axhline(y=2, color='#E74C3C', linestyle='--', linewidth=2.5, label='Short Entry (Z>2)')
plt.axhline(y=-2, color='#E74C3C', linestyle='--', linewidth=2.5, label='Long Entry (Z<-2)')
plt.axhline(y=0.5, color='#27AE60', linestyle=':', linewidth=2, label='Exit (|Z|<0.5)')
plt.axhline(y=-0.5, color='#27AE60', linestyle=':', linewidth=2)
plt.fill_between(t, -2, 2, alpha=0.15, color='red', label='Open Position Zone')
plt.title('Z-Score of Spread (Trading Signals)', fontsize=13, fontweight='bold', pad=10)
plt.xlabel('Trading Days', fontsize=11)
plt.ylabel('Z-Score', fontsize=11)
plt.legend(fontsize=10, loc='upper right', ncol=2)
plt.grid(True, alpha=0.3, linestyle=':')
plt.xlim(0, n)

plt.tight_layout(h_pad=3.0)
plt.savefig('pairs_trading_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("Image 2: pairs_trading_diagram.png generated")
print("All images generated successfully!")
