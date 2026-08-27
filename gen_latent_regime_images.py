"""Generate 3 charts for latent-regime-representation article."""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "/Users/halo/workspace/astro-blog/public/images/latent-regime-representation"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# Synthesize a multivariate return series with 3 latent regimes
T = 1500
regimes = []
mus = [0.001, -0.003, 0.000]
vols = [0.012, 0.030, 0.006]
weights_for_regime = [(0.4, 0.3, 0.3), (-0.5, 0.7, -0.2), (0.6, 0.5, -0.4)]

# piecewise regime labels
labels = np.zeros(T, dtype=int)
t = 0
while t < T:
    p = rng.uniform()
    if p < 0.5:
        d = int(rng.integers(40, 100))
        labels[t:t + d] = 0
        t += d
    elif p < 0.85:
        d = int(rng.integers(20, 80))
        labels[t:t + d] = 1
        t += d
    else:
        d = int(rng.integers(80, 200))
        labels[t:t + d] = 2
        t += d
labels = labels[:T]

# generate 6 correlated assets
n_assets = 6
# factor loadings per regime (3 factors), but the hidden state is 2-dim continuous; we approximate
loadings = np.array([
    [0.6, 0.4, 0.2],
    [-0.3, 0.7, 0.1],
    [0.5, -0.4, 0.6],
    [0.2, 0.5, -0.5],
    [0.7, 0.3, 0.4],
    [-0.4, -0.5, 0.3],
])

returns = np.zeros((T, n_assets))
for t in range(T):
    r = labels[t]
    mu = np.array(mus[r])
    vol = vols[r]
    # factor returns (only the relevant factor matters)
    f = rng.standard_normal(3) * vol
    returns[t] = loadings @ f + rng.standard_normal(n_assets) * 0.003

# ============ 1. Latent space trajectory (2D via PCA) ============
window = 20
features = []
for t in range(window, T):
    block = returns[t - window:t]
    # 12 simple stats: mean, vol, corr trace for each asset pair subset
    features.append(np.concatenate([block.mean(0), block.std(0)]))
features = np.array(features)

from numpy.linalg import svd
Xc = features - features.mean(0)
U, S, Vt = svd(Xc, full_matrices=False)
z = U[:, :2] * S[:2]

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(z[:, 0], z[:, 1], c=np.arange(len(z)), cmap='viridis', s=4, alpha=0.7)
ax.set_xlabel('Latent dim 1')
ax.set_ylabel('Latent dim 2')
ax.set_title('VAE-style 2-D latent trajectory of market regimes\n(color = time index)')
plt.colorbar(sc, ax=ax, label='Time')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/latent_2d_trajectory.png", dpi=140, bbox_inches='tight')
plt.close()

# ============ 2. Reconstruction vs raw returns ============
# Simple linear "decoder" reconstructing returns from latent z (training: ridge)
from numpy.linalg import lstsq
# y = (T-window, 6), X = (T-window, 2) + intercept
X = np.column_stack([np.ones(len(z)), z])
W, *_ = lstsq(X, returns[window:], rcond=None)
recon = X @ W
mse = np.mean((returns[window:] - recon) ** 2)
print(f"Linear recon MSE = {mse:.4e}")

fig, axes = plt.subplots(3, 1, figsize=(10, 7))
for i in range(3):
    axes[i].plot(returns[window - 1:, i], lw=0.7, alpha=0.7, label='raw')
    axes[i].plot(recon[:, i], lw=0.7, color='red', alpha=0.7, label='recon (latent)')
    axes[i].set_ylabel(f'r{i + 1}')
    axes[i].legend(loc='upper right', fontsize=8)
    axes[i].grid(alpha=0.3)
axes[0].set_title('Latent-2 reconstruction vs raw returns (3 of 6 assets shown)')
plt.tight_layout()
plt.savefig(f"{OUT}/reconstruction_vs_raw.png", dpi=140, bbox_inches='tight')
plt.close()

# ============ 3. Regime-conditional vol mapping ============
fig, ax = plt.subplots(figsize=(9, 5))
# color points by their true regime
scatter = ax.scatter(z[:, 0], z[:, 1], c=labels[window:],
                     cmap='RdYlGn_r', s=8, alpha=0.6)
# Add latent-vol axis: rolling vol of first PC
proxy_vol = features[:, 6]  # asset-1 vol feature
cnorm = ax.tricontourf(z[:, 0], z[:, 1], proxy_vol, levels=10, alpha=0.25, cmap='magma')
plt.colorbar(cnorm, ax=ax, label='rolling vol (asset 1)')
ax.set_xlabel('Latent dim 1')
ax.set_ylabel('Latent dim 2')
ax.set_title('Regime separation in latent space\n(red = high-vol, green = low-vol)')
plt.colorbar(scatter, ax=ax, label='true regime', pad=0.02)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/regime_colored_latent.png", dpi=140, bbox_inches='tight')
plt.close()

print("Generated 3 images in", OUT)
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"  {f}: {os.path.getsize(p)} bytes")
