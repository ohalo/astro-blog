"""Generate 3 charts for sparse-expert-routing-factor article."""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "/Users/halo/workspace/astro-blog/public/images/sparse-expert-routing-factor"
os.makedirs(OUT, exist_ok=True)
np.random.seed(20260828)
rng = np.random.default_rng(42)

# ============ 1. Routing weights heatmap (top-K sparsity) ============
n_assets, n_experts = 30, 8
logits = rng.standard_normal((n_assets, n_experts)) * 1.5
W = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax per asset

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(W, aspect='auto', cmap='viridis', interpolation='nearest')
ax.set_xlabel('Expert index')
ax.set_ylabel('Asset index')
ax.set_title('Routing weights: each asset activates only a few experts')
plt.colorbar(im, ax=ax, label='Routing weight')
dom = np.argmax(W, axis=1)
for i, d in enumerate(dom):
    ax.text(d, i, '*', ha='center', va='center', color='white', fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/routing_weights_heatmap.png", dpi=140, bbox_inches='tight')
plt.close()

# ============ 2. Load concentration: top-K prob mass curve ============
sorted_W = np.sort(W, axis=1)[:, ::-1]
cum = sorted_W.cumsum(axis=1)
k_vals = np.arange(1, n_experts + 1)

fig, ax = plt.subplots(figsize=(8, 5))
mean_curve = cum.mean(axis=0)
std_curve = cum.std(axis=0)
ax.plot(k_vals, mean_curve, 'o-', lw=2, label='Mean cumulative routing mass')
ax.fill_between(k_vals, mean_curve - std_curve, mean_curve + std_curve,
                alpha=0.25, label='+/- 1 std')
for K in [2, 3, 5]:
    mass = cum[:, K - 1].mean()
    ax.axhline(mass, ls='--', lw=1, alpha=0.6)
    ax.annotate(f"top-{K} = {mass:.1%}", xy=(K + 0.3, mass + 0.02),
                fontsize=9, color='darkred')
ax.set_xlabel('Number of activated experts (K)')
ax.set_ylabel('Cumulative routing probability mass')
ax.set_title('Top-K routing captures most of the signal')
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/topk_concentration_curve.png", dpi=140, bbox_inches='tight')
plt.close()

# ============ 3. Out-of-sample IC: dense vs top-K sparse ============
# Cross-sectional factor model with rolling-window ridge regression.
# Simulate a setting where only 3 of 8 experts truly carry signal.
T = 200   # periods
N = 80    # assets
K = 8     # experts
n_active = 3
true_idx = np.array([1, 4, 6])

expert_signals = rng.standard_normal((T, K)) * 0.02
true_loadings = np.zeros((N, K))
true_loadings[:, true_idx] = rng.standard_normal((N, n_active)) * 0.6
asset_returns = true_loadings @ expert_signals.T + rng.standard_normal((N, T)) * 0.012

# rolling: train on [0, t), evaluate information coefficient at time t
def rolling_ic(stride_window, top_k=None, lam=1e-3):
    ics = []
    for t in range(30, T):
        X_tr = expert_signals[:t]
        y_tr = asset_returns[:, :t]  # (N, t)
        # panel regression: stack, solve per-asset
        # y_i = X_tr @ beta_i + e, OLS for all assets together: B = (X'X)^-1 X' Y
        XtX = X_tr.T @ X_tr + lam * np.eye(K)
        XtY = X_tr.T @ y_tr.T   # (K, N)
        B = np.linalg.solve(XtX, XtY)  # (K, N)
        if top_k is not None:
            mask = np.zeros_like(B)
            for i in range(N):
                idx = np.argsort(np.abs(B[:, i]))[::-1][:top_k]
                mask[idx, i] = B[idx, i]
            B = mask
        # IC: at period t, score per-asset using factors at t and per-asset loadings
        scores = B.T @ expert_signals[t]  # (N,)
        ics.append(np.corrcoef(scores, asset_returns[:, t])[0, 1])
    return np.array(ics)

ic_dense = rolling_ic(30, top_k=None)
ic_topk5 = rolling_ic(30, top_k=5)
ic_topk3 = rolling_ic(30, top_k=3)
ic_topk2 = rolling_ic(30, top_k=2)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ic_dense, label='Dense (8 experts)', lw=1.2, alpha=0.85)
ax.plot(ic_topk5, label='Top-5 sparse', lw=1.2, alpha=0.85)
ax.plot(ic_topk3, label='Top-3 sparse (truth K=3)', lw=1.6, alpha=0.9)
ax.plot(ic_topk2, label='Top-2 sparse', lw=1.2, alpha=0.85)
ax.axhline(ic_dense.mean(), color='C0', ls='--', lw=1, alpha=0.6,
           label=f'mean IC dense = {ic_dense.mean():.3f}')
ax.axhline(ic_topk3.mean(), color='C2', ls='--', lw=1, alpha=0.6,
           label=f'mean IC top-3 = {ic_topk3.mean():.3f}')
ax.axhline(ic_topk2.mean(), color='C3', ls='--', lw=1, alpha=0.6,
           label=f'mean IC top-2 = {ic_topk2.mean():.3f}')
ax.set_xlabel('Time index (rolling window)')
ax.set_ylabel('Cross-sectional IC')
ax.set_title('OOS IC: top-3 sparse ~= dense; aggressive sparsifying hurts')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/oos_ic_comparison.png", dpi=140, bbox_inches='tight')
plt.close()

print("Generated 3 images in", OUT)
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"  {f}: {os.path.getsize(p)} bytes")
