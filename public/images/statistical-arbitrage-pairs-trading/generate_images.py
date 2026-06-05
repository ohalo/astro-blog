import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Image 1: Cointegration test chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Generate cointegrated series
n = 200
t = np.arange(n)
eps = np.cumsum(np.random.randn(n) * 0.5)  # Random walk as spread
x = 50 + 0.5 * t + np.cumsum(np.random.randn(n) * 2)  # Stock 1
y = 30 + 0.5 * t + eps + np.random.randn(n) * 2  # Stock 2 (cointegrated with x)

ax1.plot(t, x, 'b-', linewidth=2, label='Stock 1 (中信证券)')
ax1.plot(t, y, 'r-', linewidth=2, label='Stock 2 (华泰证券)')
ax1.set_ylabel('Price', fontsize=12)
ax1.set_title('Cointegrated Stock Prices', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Spread (should be stationary)
spread = x - 1.2 * y + 20
ax2.plot(t, spread, 'g-', linewidth=2)
ax2.axhline(y=spread.mean(), color='black', linestyle='--', label='Mean')
ax2.fill_between(t, spread.mean() - 2*spread.std(), 
                  spread.mean() + 2*spread.std(), 
                  alpha=0.2, color='green', label='±2 Std')
ax2.set_xlabel('Trading Days', fontsize=12)
ax2.set_ylabel('Spread', fontsize=12)
ax2.set_title('Spread (Mean-Reverting)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cointegration-test.jpg', dpi=150)
print("Generated cointegration-test.jpg")

# Image 2: Pairs trading spread chart with Z-score
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Calculate Z-score
zscore = (spread - spread.mean()) / spread.std()

ax1.plot(t, spread, 'b-', linewidth=2, label='Spread')
ax1.axhline(y=spread.mean(), color='black', linestyle='--', label='Mean')
ax1.axhline(y=spread.mean() + 2*spread.std(), color='red', 
             linestyle='--', alpha=0.7, label='+2 Std (Sell Signal)')
ax1.axhline(y=spread.mean() - 2*spread.std(), color='green', 
             linestyle='--', alpha=0.7, label='-2 Std (Buy Signal)')
ax1.fill_between(t, spread.mean() - 2*spread.std(), 
                  spread.mean() + 2*spread.std(), 
                  alpha=0.1, color='gray')
ax1.set_ylabel('Spread', fontsize=12)
ax1.set_title('Pairs Trading: Spread with Entry/Exit Thresholds', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t, zscore, 'purple', linewidth=2, label='Z-Score')
ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Entry (2.0)')
ax2.axhline(y=-2, color='green', linestyle='--', alpha=0.7, label='Entry (-2.0)')
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Exit (0.5)')
ax2.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.7)
ax2.fill_between(t, -2, 2, alpha=0.1, color='gray')
ax2.set_xlabel('Trading Days', fontsize=12)
ax2.set_ylabel('Z-Score', fontsize=12)
ax2.set_title('Z-Score of Spread', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pairs-spread-chart.jpg', dpi=150)
print("Generated pairs-spread-chart.jpg")

plt.close('all')
print("All images generated successfully!")
