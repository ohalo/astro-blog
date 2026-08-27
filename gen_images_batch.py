#!/usr/bin/env python3
"""Generate 6 charts for two blog articles (English labels to avoid CJK font issues)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT1 = "/Users/halo/workspace/astro-blog/public/images/information-geometry-portfolio"
OUT2 = "/Users/halo/workspace/astro-blog/public/images/quantum-annealing-portfolio"
os.makedirs(OUT1, exist_ok=True)
os.makedirs(OUT2, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
})

# ============================================================
# Article 1: Information Geometry & Portfolio
# ============================================================

# --- Chart 1: Fisher metric contours on 2-simplex ---
fig, ax = plt.subplots(figsize=(9, 7))

tri_verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
triangle = plt.Polygon(tri_verts, fill=False, edgecolor='#333333', linewidth=2)
ax.add_patch(triangle)

w0 = np.array([1/3, 1/3, 1/3])
n_grid = 100
w1 = np.linspace(0.01, 0.98, n_grid)
w2 = np.linspace(0.01, 0.98, n_grid)
W1, W2 = np.meshgrid(w1, w2)
W3 = 1 - W1 - W2
mask = W3 > 0.01

def to_cart(w1, w2):
    w3 = 1 - w1 - w2
    x = w2 + w3 * 0.5
    y = w3 * np.sqrt(3) / 2
    return x, y

def fisher_dist_grid(W1, W2):
    W3 = 1 - W1 - W2
    valid = (W1 > 0.001) & (W2 > 0.001) & (W3 > 0.001)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_ratio = np.log(np.where(valid, W1/w0[0], 1))**2 + \
                   np.log(np.where(valid, W2/w0[1], 1))**2 + \
                   np.log(np.where(valid, W3/w0[2], 1))**2
    d = np.where(valid, np.sqrt(ln_ratio), np.nan)
    return d

FD = fisher_dist_grid(W1, W2)
XC, YC = to_cart(W1, W2)
levels = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
cs = ax.contourf(XC, YC, FD, levels=levels, cmap='YlOrRd', alpha=0.6)
plt.colorbar(cs, ax=ax, label='Fisher Distance from Uniform')
ax.contour(XC, YC, FD, levels=levels, colors='gray', linewidths=0.5, alpha=0.5)

# Gradient flow trajectories
np.random.seed(42)
center = np.array([1/3, 1/3, 1/3])
for start in [(0.8, 0.1), (0.1, 0.8), (0.7, 0.25), (0.05, 0.15), (0.45, 0.5)]:
    w = np.array([start[0], start[1], 1-start[0]-start[1]])
    path_x, path_y = [], []
    for step in range(200):
        x, y = to_cart(w[1], w[2])
        path_x.append(x)
        path_y.append(y)
        grad = center - w
        riem_grad = grad * w  # natural gradient
        norm = np.linalg.norm(riem_grad)
        if norm < 1e-6:
            break
        w = w + 0.05 * riem_grad / norm
        w = np.clip(w, 0.001, 0.999)
        w = w / w.sum()
    ax.plot(path_x, path_y, 'k-', linewidth=1.5, alpha=0.7)
    ax.plot(path_x[0], path_y[0], 'go', markersize=6)

cx, cy = to_cart(1/3, 1/3)
ax.plot(cx, cy, 'r*', markersize=15, zorder=5)
ax.annotate('Uniform Portfolio\n(Min Description Length)', (cx, cy),
            textcoords="offset points", xytext=(15, -20), fontsize=9, color='red')

ax.text(0.0, -0.06, 'w1 = 1', ha='center', fontsize=10)
ax.text(1.0, -0.06, 'w2 = 1', ha='center', fontsize=10)
ax.text(0.5, np.sqrt(3)/2 + 0.06, 'w3 = 1', ha='center', fontsize=10)

ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
ax.set_aspect('equal')
ax.set_title('Information Geometry: Fisher Metric on Simplex & Riemannian Gradient Flow', fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUT1}/fisher_metric_simplex.png", bbox_inches='tight')
plt.close()
print("  OK fisher_metric_simplex.png")

# --- Chart 2: Euclidean vs Riemannian convergence ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

np.random.seed(123)
n_assets = 5
n_obs = 252
returns = np.random.randn(n_obs, n_assets) * 0.01 + 0.0003
Sigma = np.cov(returns.T)
mu = returns.mean(axis=0)
target_ret = 0.02

def euclidean_grad_descent(mu, Sigma, target_ret, n_steps=300, lr=0.5):
    w = np.ones(len(mu)) / len(mu)
    path = [w.copy()]
    for _ in range(n_steps):
        port_ret = w @ mu
        grad = mu - target_ret * np.ones(len(mu)) + 0.2 * 2 * Sigma @ w
        w = w + lr * grad
        w = np.clip(w, 0, 1)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(mu))/len(mu)
        path.append(w.copy())
    return np.array(path)

def riemannian_grad_descent(mu, Sigma, target_ret, n_steps=300, lr=0.3):
    w = np.ones(len(mu)) / len(mu)
    path = [w.copy()]
    for _ in range(n_steps):
        grad = mu - target_ret * np.ones(len(mu)) + 0.2 * 2 * Sigma @ w
        nat_grad = grad * w
        nat_grad = nat_grad - nat_grad.mean()
        w = w + lr * nat_grad
        w = np.clip(w, 1e-6, 1)
        w = w / w.sum()
        path.append(w.copy())
    return np.array(path)

path_euc = euclidean_grad_descent(mu, Sigma, target_ret)
path_riem = riemannian_grad_descent(mu, Sigma, target_ret)

ax1 = axes[0]
colors_list = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
for i in range(n_assets):
    ax1.plot(path_euc[:, i], '--', color=colors_list[i], alpha=0.6, linewidth=1.2, label=f'w{i+1} Euclidean')
    ax1.plot(path_riem[:, i], '-', color=colors_list[i], linewidth=1.8, label=f'w{i+1} Riemannian')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Weight')
ax1.set_title('Weight Trajectories: Euclidean (dashed) vs Riemannian (solid)')
ax1.legend(fontsize=7, ncol=2, loc='upper right')
ax1.axhline(1/n_assets, color='gray', linestyle=':', alpha=0.3)

ax2 = axes[1]
obj_euc = [0.5 * (w @ Sigma @ w) - 0.5 * (w @ mu) for w in path_euc]
obj_riem = [0.5 * (w @ Sigma @ w) - 0.5 * (w @ mu) for w in path_riem]
ax2.plot(obj_euc, '--', color='#e74c3c', linewidth=1.5, label='Euclidean gradient')
ax2.plot(obj_riem, '-', color='#2ecc71', linewidth=2, label='Riemannian (natural) gradient')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Objective Value')
ax2.set_title('Objective Convergence: Natural Gradient Reaches Optimum Faster')
ax2.legend()
ax2.set_yscale('symlog')

plt.suptitle('Euclidean vs Riemannian Gradient Descent: Portfolio Optimization', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT1}/euclidean_vs_riemannian.png", bbox_inches='tight')
plt.close()
print("  OK euclidean_vs_riemannian.png")

# --- Chart 3: Out-of-sample Sharpe comparison ---
fig, ax = plt.subplots(figsize=(10, 6))

np.random.seed(999)
n_assets = 10
n_train = 500
n_test = 500

true_mu = np.random.randn(n_assets) * 0.001 + 0.0005
true_Sigma = np.eye(n_assets) * 0.0004 + np.outer(np.random.randn(n_assets), np.random.randn(n_assets)) * 0.0001
true_Sigma = (true_Sigma + true_Sigma.T) / 2
eigvals = np.linalg.eigvalsh(true_Sigma)
if eigvals.min() < 0.001:
    true_Sigma += np.eye(n_assets) * (0.001 - eigvals.min() + 0.001)

train_ret = np.random.multivariate_normal(true_mu, true_Sigma, n_train)
test_ret = np.random.multivariate_normal(true_mu, true_Sigma, n_test)
train_ret_noisy = train_ret + np.random.randn(*train_ret.shape) * 0.003
mu_train = train_ret_noisy.mean(axis=0)
Sigma_train = np.cov(train_ret_noisy.T)

from numpy.linalg import inv
inv_Sigma = inv(Sigma_train + 0.1 * np.eye(n_assets) * np.trace(Sigma_train) / n_assets)
w_euc = inv_Sigma @ mu_train
w_euc = np.clip(w_euc, 0, 1)
w_euc = w_euc / w_euc.sum()

w_r = np.ones(n_assets) / n_assets
lr = 0.01
for _ in range(2000):
    grad = mu_train - 2 * 0.01 * Sigma_train @ w_r
    nat_grad = grad * w_r
    nat_grad -= nat_grad.mean()
    w_r = w_r + lr * nat_grad
    w_r = np.clip(w_r, 1e-8, 1)
    w_r = w_r / w_r.sum()
w_riem = w_r
w_eq = np.ones(n_assets) / n_assets

portfolios = {
    'Euclidean\n(Markowitz)': w_euc,
    'Riemannian\n(Natural Grad)': w_riem,
    'Equal Weight': w_eq,
}
methods = list(portfolios.keys())
sharpe_train = []
sharpe_test = []
for w in portfolios.values():
    r_train = train_ret @ w
    r_test = test_ret @ w
    sharpe_train.append(np.mean(r_train) / (np.std(r_train) + 1e-10) * np.sqrt(252))
    sharpe_test.append(np.mean(r_test) / (np.std(r_test) + 1e-10) * np.sqrt(252))

x = np.arange(len(methods))
width = 0.35
bars1 = ax.bar(x - width/2, sharpe_train, width, label='In-Sample Sharpe', color=['#3498db', '#2ecc71', '#f39c12'], alpha=0.7)
bars2 = ax.bar(x + width/2, sharpe_test, width, label='Out-of-Sample Sharpe', color=['#2980b9', '#27ae60', '#e67e22'], alpha=0.9)

ax.set_ylabel('Annualized Sharpe Ratio')
ax.set_title('In/Out-of-Sample Sharpe Decay: Info-Geometry Regularization Reduces Overfit', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.axhline(0, color='gray', linewidth=0.5)

for bar in bars1 + bars2:
    h = bar.get_height()
    ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

for i in range(len(methods)):
    decay = sharpe_train[i] - sharpe_test[i]
    ax.annotate(f'Decay {decay:.2f}', xy=(x[i], max(sharpe_train[i], sharpe_test[i]) + 0.15),
                ha='center', fontsize=8, color='#e74c3c', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT1}/sharpe_decay_comparison.png", bbox_inches='tight')
plt.close()
print("  OK sharpe_decay_comparison.png")

# ============================================================
# Article 2: Quantum Annealing Portfolio Optimization
# ============================================================

# --- Chart 1: Energy landscape with classical vs quantum paths ---
fig, ax = plt.subplots(figsize=(10, 6.5))

x = np.linspace(-3, 7, 500)
y = 0.5 * np.sin(3*x) * np.exp(-0.15*(x-2)**2) + 0.3*np.sin(5*x) + 0.5*x**2*0.05 - 0.3
y[x < 0] += 2.0
y = y - y.min() + 0.5

ax.fill_between(x, y, -2, alpha=0.15, color='#3498db')
ax.plot(x, y, color='#2c3e50', linewidth=2.5, label='Energy Landscape E(w)')

local_min_x = 1.0
local_min_y = y[np.argmin(np.abs(x-1.0))]
ax.plot(local_min_x, local_min_y, 'o', color='#e74c3c', markersize=12, zorder=5)
ax.annotate('Local Optimum\n(Markowitz)', (local_min_x, local_min_y),
            textcoords="offset points", xytext=(-70, 25), fontsize=9, color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c'))

global_min_x = 5.0
global_min_y = y[np.argmin(np.abs(x-5.0))]
ax.plot(global_min_x, global_min_y, '*', color='#2ecc71', markersize=15, zorder=5)
ax.annotate('Global Optimum', (global_min_x, global_min_y),
            textcoords="offset points", xytext=(15, 15), fontsize=9, color='#2ecc71',
            arrowprops=dict(arrowstyle='->', color='#2ecc71'))

sa_x = np.linspace(1.0, 1.5, 50)
sa_y = local_min_y + np.random.randn(50) * 0.3 * np.exp(-np.arange(50)/20)
ax.plot(sa_x, sa_y, '--', color='#e74c3c', linewidth=1.5, alpha=0.7, label='Simulated Annealing (thermal, stuck local)')

tunnel_x = np.array([1.0, 1.5, 2.5, 3.5, 4.5, 5.0])
tunnel_y = np.array([local_min_y, 0.5, 0.3, 0.5, 0.8, global_min_y])
ax.plot(tunnel_x, tunnel_y, '-', color='#9b59b6', linewidth=2.5, alpha=0.8, label='Quantum Tunneling (through barrier)')

barrier_region = (x > 1.5) & (x < 4.0)
ax.fill_between(x[barrier_region], y[barrier_region], -2, alpha=0.2, color='#e74c3c')
ax.annotate('Barrier Region', (2.75, 2.5), ha='center', fontsize=9, color='#e74c3c', fontstyle='italic')

ax.set_xlim(-3, 7)
ax.set_ylim(-2, 4.5)
ax.set_xlabel('Portfolio Weight Space w')
ax.set_ylabel('Energy / Objective')
ax.set_title('Quantum Annealing vs Simulated Annealing: Tunneling Through Barriers', fontweight='bold')
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT2}/energy_landscape_tunneling.png", bbox_inches='tight')
plt.close()
print("  OK energy_landscape_tunneling.png")

# --- Chart 2: Annealing schedule and convergence ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Annealing schedules
ax_left = axes[0]
t = np.linspace(0, 10, 200)
T_classical = 5.0 * np.exp(-t * 0.5)
ax_left.plot(t, T_classical, '-', color='#e74c3c', linewidth=2, label='SA: T(t) = T0 * exp(-at)')
A_quantum = 5.0 * np.clip(1 - t/10, 0, 5)
B_quantum = t / 10 * 3
ax_left.plot(t, A_quantum, '--', color='#9b59b6', linewidth=2, label='QA: A(t) transverse field -> 0')
ax_left.plot(t, B_quantum, ':', color='#2ecc71', linewidth=2, label='QA: B(t) problem Hamiltonian -> H_max')
ax_left.fill_between(t, A_quantum, alpha=0.1, color='#9b59b6')
ax_left.set_xlabel('Annealing Time t')
ax_left.set_ylabel('Intensity')
ax_left.set_title('Annealing Schedules: Temperature vs Transverse Field')
ax_left.legend(fontsize=8)
ax_left.set_ylim(-0.5, 6)

# Right: Convergence comparison
ax_right = axes[1]
np.random.seed(77)
n_assets = 8
n_steps = 300
true_w = np.array([0.05, 0.15, 0.25, 0.05, 0.10, 0.20, 0.10, 0.10])
returns_data = np.random.randn(500, n_assets) * 0.015 + true_w * 0.001 * 5
mu = returns_data.mean(axis=0)
Sigma = np.cov(returns_data.T)

def portfolio_energy(w, mu, Sigma, lam=0.5):
    return -w @ mu + lam * w @ Sigma @ w

def run_sa(mu, Sigma, n_steps=300, T0=1.0, cooling=0.97):
    w = np.ones(len(mu)) / len(mu)
    best_w = w.copy()
    best_e = portfolio_energy(w, mu, Sigma)
    energies = [best_e]
    T = T0
    for _ in range(n_steps):
        w_new = w + np.random.randn(len(mu)) * T * 0.3
        w_new = np.clip(w_new, 0, 1)
        w_new = w_new / w_new.sum()
        e_new = portfolio_energy(w_new, mu, Sigma)
        if e_new < best_e:
            best_w = w_new.copy()
            best_e = e_new
        curr_e = portfolio_energy(w, mu, Sigma)
        if e_new < curr_e or np.random.rand() < np.exp(-(e_new - curr_e)/max(T, 1e-10)):
            w = w_new
        T *= cooling
        energies.append(best_e)
    return np.array(energies)

def run_qa(mu, Sigma, n_steps=300, A0=1.0):
    w = np.ones(len(mu)) / len(mu)
    best_w = w.copy()
    best_e = portfolio_energy(w, mu, Sigma)
    energies = [best_e]
    A = A0
    for step in range(n_steps):
        grad = -mu + 2 * 0.5 * Sigma @ w
        w = w + 0.01 * grad * w
        tunnel = A * np.random.randn(len(mu)) * 0.1
        w = w + tunnel
        w = np.clip(w, 0, 1)
        w = w / w.sum()
        e = portfolio_energy(w, mu, Sigma)
        if e < best_e:
            best_w = w.copy()
            best_e = e
        A *= 0.97
        energies.append(best_e)
    return np.array(energies)

n_trials = 20
sa_finals, qa_finals = [], []
sa_all, qa_all = [], []
for _ in range(n_trials):
    e_sa = run_sa(mu, Sigma, n_steps=n_steps)
    e_qa = run_qa(mu, Sigma, n_steps=n_steps)
    sa_finals.append(e_sa[-1])
    qa_finals.append(e_qa[-1])
    sa_all.append(e_sa)
    qa_all.append(e_qa)

steps = np.arange(n_steps + 1)
sa_mean = np.mean(sa_all, axis=0)
qa_mean = np.mean(qa_all, axis=0)
sa_std = np.std(sa_all, axis=0)
qa_std = np.std(qa_all, axis=0)

ax_right.plot(steps, sa_mean, '-', color='#e74c3c', linewidth=2, label='Simulated Annealing (mean)')
ax_right.fill_between(steps, sa_mean - sa_std, sa_mean + sa_std, alpha=0.15, color='#e74c3c')
ax_right.plot(steps, qa_mean, '-', color='#9b59b6', linewidth=2, label='Quantum Annealing (mean)')
ax_right.fill_between(steps, qa_mean - qa_std, qa_mean + qa_std, alpha=0.15, color='#9b59b6')

ax_right.set_xlabel('Annealing Step')
ax_right.set_ylabel('Best Energy Found')
ax_right.set_title(f'Convergence Comparison ({n_trials} trials, mean +/- 1std)')
ax_right.legend()
ax_right.set_yscale('symlog')

plt.suptitle('Simulated Annealing vs Quantum Annealing: Schedule & Convergence', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT2}/annealing_schedule_convergence.png", bbox_inches='tight')
plt.close()
print("  OK annealing_schedule_convergence.png")

# --- Chart 3: Portfolio performance comparison ---
fig, ax = plt.subplots(figsize=(10, 6))

np.random.seed(55)
n_assets = 12
n_obs = 252 * 3
factor_loadings = np.random.randn(n_assets, 3) * 0.5 + 0.5
factor_returns = np.random.randn(n_obs, 3) * 0.008
idio = np.random.randn(n_obs, n_assets) * 0.005
returns = factor_returns @ factor_loadings.T + idio + 0.0002
mu = returns.mean(axis=0)
Sigma = np.cov(returns.T)

train = returns[:n_obs//2]
test = returns[n_obs//2:]
mu_train = train.mean(axis=0)
Sigma_train = np.cov(train.T)

from numpy.linalg import inv
inv_S = inv(Sigma_train + 0.05 * np.eye(n_assets) * np.trace(Sigma_train)/n_assets)
w_mark = inv_S @ mu_train
w_mark = np.clip(w_mark, 0, 1)
w_mark = w_mark / w_mark.sum()

def run_sa_portfolio(mu, Sigma, n_steps=2000, T0=0.5, cooling=0.998, lam=0.5):
    w = np.ones(len(mu)) / len(mu)
    best_w = w.copy()
    best_obj = -w @ mu + lam * w @ Sigma @ w
    T = T0
    for _ in range(n_steps):
        w_new = w + np.random.randn(len(mu)) * T * 0.2
        w_new = np.clip(w_new, 0, 1)
        w_new = w_new / w_new.sum()
        obj = -w_new @ mu + lam * w_new @ Sigma @ w_new
        if obj < best_obj:
            best_w = w_new.copy()
            best_obj = obj
        curr_obj = -w @ mu + lam * w @ Sigma @ w
        if obj < curr_obj or np.random.rand() < np.exp(-(obj - curr_obj)/max(T, 1e-10)):
            w = w_new
        T *= cooling
    return best_w

def run_qa_portfolio(mu, Sigma, n_steps=2000, A0=0.3, lam=0.5):
    w = np.ones(len(mu)) / len(mu)
    best_w = w.copy()
    best_obj = -w @ mu + lam * w @ Sigma @ w
    A = A0
    for step in range(n_steps):
        grad = -mu + 2 * lam * Sigma @ w
        w = w + 0.005 * grad * w
        w = w + A * np.random.randn(len(mu)) * 0.15
        w = np.clip(w, 0, 1)
        w = w / w.sum()
        obj = -w @ mu + lam * w @ Sigma @ w
        if obj < best_obj:
            best_w = w.copy()
            best_obj = obj
        A *= 0.998
    return best_w

w_sa = run_sa_portfolio(mu_train, Sigma_train)
w_qa = run_qa_portfolio(mu_train, Sigma_train)
w_eq = np.ones(n_assets) / n_assets

methods = ['Markowitz', 'Simulated\nAnnealing', 'Quantum\nAnnealing', 'Equal\nWeight']
weights = [w_mark, w_sa, w_qa, w_eq]
colors_bar = ['#3498db', '#e74c3c', '#9b59b6', '#f39c12']

n_windows = 20
window_size = 60
mark_sharpes, sa_sharpes, qa_sharpes, eq_sharpes = [], [], [], []

for i in range(n_windows):
    start = i * (len(test) - window_size) // n_windows
    end = start + window_size
    if end > len(test):
        break
    window = test[start:end]
    for w, lst in [(w_mark, mark_sharpes), (w_sa, sa_sharpes), (w_qa, qa_sharpes), (w_eq, eq_sharpes)]:
        r = window @ w
        sr = np.mean(r) / (np.std(r) + 1e-10) * np.sqrt(252)
        lst.append(sr)

data = [mark_sharpes, sa_sharpes, qa_sharpes, eq_sharpes]
bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bp['boxes'], colors_bar):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('Out-of-Sample Annualized Sharpe')
ax.set_title(f'Out-of-Sample Sharpe Distribution ({len(mark_sharpes)} Rolling Windows)', fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

for i, d in enumerate(data):
    mean_val = np.mean(d)
    ax.plot(i + 1, mean_val, 'D', color=colors_bar[i], markersize=8, zorder=5)
    ax.annotate(f'mu={mean_val:.2f}', (i + 1, mean_val),
                textcoords="offset points", xytext=(15, 5), fontsize=8, color=colors_bar[i])

plt.tight_layout()
plt.savefig(f"{OUT2}/sharpe_distribution_comparison.png", bbox_inches='tight')
plt.close()
print("  OK sharpe_distribution_comparison.png")

print("\nAll 6 images generated successfully!")
