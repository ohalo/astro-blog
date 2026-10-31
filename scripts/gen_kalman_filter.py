#!/usr/bin/env python3
"""Generate 5 real computation images for kalman-filter-state-space article."""
import numpy as np
import json
import os

np.random.seed(20261031)
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'public', 'images', 'kalman-filter-state-space')
os.makedirs(output_dir, exist_ok=True)

# ============ Generate data ============
T = 500
mu_theta = 0.015
phi = 0.95
q = 0.002  # 过程噪声

theta = np.zeros(T)
theta[0] = mu_theta
for t in range(1, T):
    theta[t] = mu_theta + phi * (theta[t-1] - mu_theta) + np.random.normal(0, q)

# 观测：收益率 = 真实波动率 * 标准正态噪声（噪声与信号相关）
noise_std = 0.01  # 固定观测噪声标准差
y = theta + np.random.normal(0, noise_std, T)

# KF 参数
Q = q ** 2
R = noise_std ** 2  # 已知观测噪声方差

x_pred = np.zeros(T)
x_post = np.zeros(T)
P_pred = np.zeros(T)
P_post = np.zeros(T)
K_series = np.zeros(T)

x_pred[0] = mu_theta
P_pred[0] = 0.0001
x_post[0] = mu_theta
P_post[0] = 0.0001

for t in range(1, T):
    # Predict
    x_pred[t] = phi * x_post[t-1]
    P_pred[t] = phi * P_post[t-1] * phi + Q

    # Kalman gain
    K = P_pred[t] / (P_pred[t] + R)
    K_series[t] = K

    # Update
    innovation = y[t] - x_pred[t]
    x_post[t] = x_pred[t] + K * innovation
    P_post[t] = (1 - K) * P_pred[t]

mse_kf = float(np.mean((x_post - theta) ** 2))
mse_ma = float(np.mean((np.convolve(y, np.ones(20)/20, mode='same') - theta) ** 2))

# 动态因子暴露数据
T2 = 500
K_factors = 3
np.random.seed(20261032)
beta_true = np.zeros((T2, K_factors))
beta_true[0] = np.array([1.0, 0.5, -0.3])
for t in range(1, T2):
    beta_true[t] = beta_true[t-1] + np.random.normal(0, 0.01, K_factors)

f = np.random.normal(0, 0.02, (T2, K_factors))
eps = np.random.normal(0, 0.01, T2)
r = beta_true @ f.T + eps

# KF for factor exposure
Q_factor = np.eye(K_factors) * 0.0001
x_factor_pred = np.zeros((T2, K_factors))
x_factor_post = np.zeros((T2, K_factors))
P_factor = np.eye(K_factors) * 0.1

for t in range(T2):
    if t > 0:
        x_factor_pred[t] = x_factor_post[t-1]
        P_pred_f = P_factor + Q_factor
    else:
        x_factor_pred[t] = np.zeros(K_factors)
        P_pred_f = np.eye(K_factors) * 0.1

    H_t = f[t]  # 1 x K
    S = H_t @ P_pred_f @ H_t.T + 0.0001
    K_f = (P_pred_f @ H_t.T) / S

    innovation = r[t] - H_t @ x_factor_pred[t]
    x_factor_post[t] = x_factor_pred[t] + K_f * innovation
    P_factor = (np.eye(K_factors) - np.outer(K_f, H_t)) @ P_pred_f

results = {
    'mse_kf': round(mse_kf, 6),
    'mse_ma': round(mse_ma, 6),
    'improvement_pct': round(100 * (1 - mse_kf / mse_ma), 1),
    'T': T, 'phi': phi, 'q': q,
    'factor_rmse': [round(float(np.sqrt(np.mean((x_factor_post[:, i] - beta_true[:, i]) ** 2))), 4)
                    for i in range(K_factors)]
}

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ Fig 1: KF denoising comparison ============
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(T), theta, 'b-', lw=1.5, label='True volatility', alpha=0.8)
ax.plot(range(T), x_post, 'r-', lw=1.2, label='Kalman Filter estimate', alpha=0.9)
ax.fill_between(range(T), y - noise_std, y + noise_std, alpha=0.15, color='gray', label='Observation noise band')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Volatility')
ax.set_title('Kalman Filter Denoising: True Signal vs Estimate')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig01.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 2: Kalman gain ============
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(T), K_series, 'r-', lw=1.2)
ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Kalman Gain K_t')
ax.set_title('Kalman Gain Over Time')
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig02.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 3: Posterior variance convergence ============
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(T), P_post, 'b-', lw=1.2, label='Posterior variance')
ax.plot(range(T), P_pred, 'r--', lw=1, label='Prior variance', alpha=0.7)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Variance')
ax.set_title('Posterior Variance Convergence')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig03.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 4: Residual distribution ============
residuals_kf = x_post - theta
residuals_ma = np.convolve(y, np.ones(20)/20, mode='same') - theta
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(residuals_kf, bins=40, alpha=0.5, label='KF residual', color='red', density=True)
ax.hist(residuals_ma, bins=40, alpha=0.5, label='MA(20) residual', color='blue', density=True)
ax.set_xlabel('Residual')
ax.set_ylabel('Density')
ax.set_title('KF vs MA Residual Distribution')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig04.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 5: Kalman gain vs SNR ============
snr = Q / (R + 1e-10)  # inverse SNR for scalar case
fig, ax = plt.subplots(figsize=(10, 5))
scatter = ax.scatter(snr, K_series, c=range(T), cmap='viridis', alpha=0.6, s=15)
ax.set_xlabel('Process/obs noise ratio (Q/R)')
ax.set_ylabel('Kalman Gain K_t')
ax.set_title('Kalman Gain vs Process-to-Observation Noise Ratio')
plt.colorbar(scatter, label='Time')
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig05.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 6: Factor exposure tracking ============
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['blue', 'green', 'purple']
names = ['Market', 'Momentum', 'Value']
for i in range(K_factors):
    ax.plot(range(T2), beta_true[:, i], colors[i], lw=1.5, alpha=0.6, label=f'True {names[i]}')
    ax.plot(range(T2), x_factor_post[:, i], colors[i], lw=1.0, ls='--', alpha=0.8, label=f'KF {names[i]}')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Factor Exposure')
ax.set_title('Dynamic Factor Exposure Tracking')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig06.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 7: Factor exposure error ============
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(K_factors):
    error = np.sqrt(np.cumsum((x_factor_post[:, i] - beta_true[:, i]) ** 2) / np.arange(1, T2 + 1))
    ax.plot(range(1, T2 + 1), error, lw=1.5, label=names[i])
ax.set_xlabel('Time (days)')
ax.set_ylabel('Cumulative RMSE')
ax.set_title('Factor Exposure Estimation Error Over Time')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig07.png'), dpi=150, bbox_inches='tight')
plt.close()

# Save results JSON
json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         '_kfs_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Generated 7 images in {output_dir}")
print(f"Results: {json.dumps(results, indent=2)}")
