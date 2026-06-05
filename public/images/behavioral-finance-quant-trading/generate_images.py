import matplotlib.pyplot as plt
import numpy as np

# Image 1: Behavioral bias types
fig, ax = plt.subplots(figsize=(10, 6))
bias_types = ['Overreaction', 'Herding', 'Disposition', 'Limited Attention']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
ax.barh(bias_types, [85, 72, 68, 55], color=colors)
ax.set_xlabel('Impact Score', fontsize=12)
ax.set_title('Behavioral Bias Types in Quantitative Trading', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('behavioral-bias-types.jpg', dpi=150)
print("Generated behavioral-bias-types.jpg")

# Image 2: Herding effect chart
fig, ax = plt.subplots(figsize=(10, 6))
days = np.arange(100)
np.random.seed(42)
price = 100 + np.cumsum(np.random.randn(100) * 0.5)
herding_signal = np.where(np.abs(np.diff(price, prepend=price[0])) > 1.5, 1, 0)
herding_signal = np.convolve(herding_signal, np.ones(5)/5, mode='same')

ax.plot(days, price, 'b-', linewidth=2, label='Stock Price')
ax2 = ax.twinx()
ax2.fill_between(days, herding_signal*100, alpha=0.3, color='red', label='Herding Intensity')
ax.set_xlabel('Trading Days', fontsize=12)
ax.set_ylabel('Price', fontsize=12)
ax2.set_ylabel('Herding Intensity (%)', fontsize=12)
ax.set_title('Herding Effect in A-Share Market', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('herding-effect-chart.jpg', dpi=150)
print("Generated herding-effect-chart.jpg")

plt.close('all')
print("All images generated successfully!")
