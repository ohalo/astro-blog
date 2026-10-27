#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 5 real-computation figures for two quant articles:
  A) random-matrix-covariance-cleaning  (RMT denoising of noisy covariance)
  B) kalman-timevarying-beta             (Kalman filter for time-varying beta)
Prints key metrics so the markdown text can quote exact numbers.
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(20261026)
ROOT = "/Users/halo123456/workspace/astro-blog"
IMG_A = os.path.join(ROOT, "public/images/random-matrix-covariance-cleaning")
IMG_B = os.path.join(ROOT, "public/images/kalman-timevarying-beta")
os.makedirs(IMG_A, exist_ok=True)
os.makedirs(IMG_B, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})

# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def mp_edges(q, sigma2=1.0):
    return sigma2 * (1 - math.sqrt(q)) ** 2, sigma2 * (1 + math.sqrt(q)) ** 2

def marchenko_pastur_pdf(x, q, sigma2=1.0):
    lam_p = sigma2 * (1 + math.sqrt(q)) ** 2
    lam_m = sigma2 * (1 - math.sqrt(q)) ** 2
    out = np.zeros_like(x)
    mask = (x > lam_m) & (x < lam_p)
    out[mask] = (1.0 / (2 * math.pi * q * x[mask] * sigma2)) * np.sqrt(
        (lam_p - x[mask]) * (x[mask] - lam_m))
    return out

def ledoit_wolf(S, R):
    """Constant-correlation Ledoit-Wolf (2004). S: cov, R: (N,T) mean-adjusted returns."""
    N, T = R.shape
    s = (R @ R.T) / T
    d = np.sqrt(np.diag(s))
    cor = s / np.outer(d, d)
    rbar = (2.0 / (N * (N - 1))) * np.sum(np.triu(cor, 1))
    F = np.diag(np.diag(s)).astype(float).copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                F[i, j] = rbar * math.sqrt(s[i, i] * s[j, j])
    pi = np.zeros((N, N)); rho = 0.0; gamma = 0.0
    for t in range(T):
        diff = np.outer(R[:, t], R[:, t]) - s
        pi += diff ** 2
        gamma += np.sum((s - F) ** 2)
        rho += np.sum((s - F) * diff)
    pi /= T; rho /= T; gamma /= T
    delta = min(max((np.sum(pi) - rho) / gamma, 0.0), 1.0)
    return delta * F + (1 - delta) * S, delta

def minvar_weights(Sigma):
    try:
        inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sigma)
    ones = np.ones(Sigma.shape[0])
    w = inv @ ones
    return w / (ones @ w)

def rmt_denoise(S, q):
    """Replace noise-bulk eigenvalues (below MP upper edge) with their bulk mean."""
    eigvals, eigvecs = np.linalg.eigh(S)
    _, lam_p = mp_edges(q)
    is_noise = eigvals < lam_p
    lam_tilde = eigvals.copy()
    lam_tilde[is_noise] = eigvals[is_noise].mean()
    C = (eigvecs * lam_tilde) @ eigvecs.T
    C *= np.trace(S) / np.trace(C)
    return C, int(np.sum(~is_noise))

# ============================================================================
# ARTICLE A: Random Matrix Theory covariance cleaning
# ============================================================================
print("=" * 60)
print("ARTICLE A: Random Matrix Theory covariance cleaning")
N = 100; K = 3; T_train = 250; T_test = 250; n_mc = 40

B = np.random.randn(N, K) * 0.5
D = np.diag(np.random.uniform(0.4, 1.0, N))
Sigma_true = B @ B.T + D
sd = np.sqrt(np.diag(Sigma_true))
Sigma_true_corr = Sigma_true / np.outer(sd, sd)
signal_eigs = np.sort(np.linalg.eigvalsh(Sigma_true_corr))[::-1][:K]

ooss = {"sample": [], "rmt": [], "lw": [], "equal": []}
n_above_list = []
for trial in range(n_mc):
    X = np.random.multivariate_normal(np.zeros(N), Sigma_true_corr, size=T_train + T_test).T
    Xtr = X[:, :T_train]; Xte = X[:, T_train:]
    mu = Xtr.mean(axis=1, keepdims=True); Xtr_c = Xtr - mu; Xte_c = Xte - mu
    sdtr = Xtr_c.std(axis=1, keepdims=True)
    Rtr = Xtr_c / sdtr; Rte = Xte_c / sdtr
    S = (Rtr @ Rtr.T) / (T_train - 1)
    Ste = (Rte @ Rte.T) / (T_test - 1)
    q = N / T_train
    C_rmt, n_above = rmt_denoise(S, q)
    S_lw, _ = ledoit_wolf(S, Rtr)
    w_s = minvar_weights(S); w_r = minvar_weights(C_rmt)
    w_l = minvar_weights(S_lw); w_e = np.ones(N) / N
    ooss["sample"].append(float(w_s @ Ste @ w_s))
    ooss["rmt"].append(float(w_r @ Ste @ w_r))
    ooss["lw"].append(float(w_l @ Ste @ w_l))
    ooss["equal"].append(float(w_e @ Ste @ w_e))
    n_above_list.append(n_above)

mean_oos = {k: np.mean(v) for k, v in ooss.items()}
q = N / T_train
lam_m, lam_p = mp_edges(q)
print(f"  N={N}, T={T_train}, q=N/T={q:.3f}")
print(f"  MP support: [{lam_m:.3f}, {lam_p:.3f}]")
print(f"  true signal eigenvalues (top {K}): {np.round(signal_eigs,3)}")
print(f"  avg eigenvalues above MP upper edge: {np.mean(n_above_list):.1f}  (true K={K})")
print("  OOS portfolio variance (avg over %d MC):" % n_mc)
for k in ("sample", "rmt", "lw", "equal"):
    print(f"    {k:8s}: {mean_oos[k]:.5f}")
print(f"  RMT vs sample OOS reduction: {100*(mean_oos['sample']-mean_oos['rmt'])/mean_oos['sample']:.1f}%")
print(f"  LW  vs sample OOS reduction: {100*(mean_oos['sample']-mean_oos['lw'])/mean_oos['sample']:.1f}%")
print(f"  naive sample min-var vs 1/N: {100*(mean_oos['equal']-mean_oos['sample'])/mean_oos['equal']:.1f}% (negative => sample worse than 1/N)")

# one trial matrices for structure figure
X = np.random.multivariate_normal(np.zeros(N), Sigma_true_corr, size=T_train + T_test).T
Xtr = X[:, :T_train]; mu = Xtr.mean(axis=1, keepdims=True); Xtr_c = Xtr - mu
sdtr = Xtr_c.std(axis=1, keepdims=True); Rtr = Xtr_c / sdtr
S = (Rtr @ Rtr.T) / (T_train - 1)
C_rmt, _ = rmt_denoise(S, q)
eig_S = np.linalg.eigvalsh(S)[::-1]

# ---- FIG A1: eigenvalue spectrum vs MP ----
fig, ax = plt.subplots(figsize=(7.2, 4.3))
xs = np.linspace(0.01, max(eig_S) * 1.05, 400)
ax.hist(eig_S, bins=45, density=True, alpha=0.55, color="#4C72B0", label="Sample covariance eigenvalues")
ax.plot(xs, marchenko_pastur_pdf(xs, q), "r-", lw=2, label="Marchenko-Pastur density")
ax.axvline(lam_p, color="black", ls="--", lw=1.3, label=f"MP upper edge {lam_p:.2f}")
ax.set_xlabel("Eigenvalue"); ax.set_ylabel("Density")
ax.set_title("Eigenvalue spectrum vs Random Matrix Theory (N=100, T=250)")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(IMG_A, "fig1_spectrum.png")); plt.close(fig)

# ---- FIG A2: correlation structure recovery ----
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
im0 = axes[0].imshow(Sigma_true_corr, cmap="RdBu_r", vmin=-1, vmax=1); axes[0].set_title("True structure")
im1 = axes[1].imshow(S, cmap="RdBu_r", vmin=-1, vmax=1); axes[1].set_title("Sample (noisy)")
im2 = axes[2].imshow(C_rmt, cmap="RdBu_r", vmin=-1, vmax=1); axes[2].set_title("RMT denoised")
for a in axes: a.set_xticks([]); a.set_yticks([])
fig.colorbar(im2, ax=axes, fraction=0.025, pad=0.02)
fig.suptitle("True / Sample / RMT-denoised correlation matrices (K=3 factors)", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(IMG_A, "fig2_corr_recover.png"), bbox_inches="tight"); plt.close(fig)

# ---- FIG A3: OOS portfolio variance boxplot ----
fig, ax = plt.subplots(figsize=(7.2, 4.3))
data = [ooss["sample"], ooss["rmt"], ooss["lw"], ooss["equal"]]
labels = ["Sample cov", "RMT denoised", "Ledoit-Wolf", "1/N equal"]
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], ["#C44E52", "#55A868", "#4C72B0", "#8172B3"]):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax.set_ylabel("Out-of-sample portfolio variance")
ax.set_title("OOS min-variance risk: RMT vs alternatives")
fig.tight_layout(); fig.savefig(os.path.join(IMG_A, "fig3_oos_var.png")); plt.close(fig)

# ---- FIG A4: eigenvalue shrinkage map ----
fig, ax = plt.subplots(figsize=(7.2, 4.3))
idx = np.arange(1, N + 1)
ev = np.sort(np.linalg.eigvalsh(S))[::-1]
C2, _ = rmt_denoise(S, q)
lt = np.sort(np.linalg.eigvalsh(C2))[::-1]
ax.plot(idx, ev, "o-", ms=3, color="#C44E52", label="Sample eigenvalues")
ax.plot(idx, lt, "s-", ms=3, color="#55A868", label="Denoised eigenvalues")
ax.axhline(lam_p, color="black", ls="--", label=f"MP upper edge {lam_p:.2f}")
ax.set_xlabel("Eigenvalue rank"); ax.set_ylabel("Eigenvalue")
ax.set_title("Eigenvalue shrinkage: noise bulk pinned to its mean")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(IMG_A, "fig4_eig_shrink.png")); plt.close(fig)

# ---- FIG A5: OOS variance reduction vs N/T ratio ----
ratios = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
red_rmt, red_lw = [], []
N0 = 100; nrep = 12
for rr in ratios:
    Tt = max(40, min(int(round(N0 / rr)), 400))
    vt = max(20, Tt * 2 // 3); te = Tt - vt
    rs = []; rl = []; rsamp = []
    for _ in range(nrep):
        X = np.random.multivariate_normal(np.zeros(N0), Sigma_true_corr, size=Tt).T
        mu = X[:, :vt].mean(axis=1, keepdims=True); Xc = X - mu
        sdt = Xc[:, :vt].std(axis=1, keepdims=True)
        Rtr = Xc[:, :vt] / sdt; Rte = Xc[:, vt:] / sdt
        S = (Rtr @ Rtr.T) / (vt - 1); Ste = (Rte @ Rte.T) / (te - 1)
        ql = N0 / vt; Crmt, _ = rmt_denoise(S, ql); Slw, _ = ledoit_wolf(S, Rtr)
        ws = minvar_weights(S); wr = minvar_weights(Crmt); wl = minvar_weights(Slw)
        rsamp.append(ws @ Ste @ ws); rs.append(wr @ Ste @ wr); rl.append(wl @ Ste @ wl)
    red_rmt.append(100 * (np.mean(rsamp) - np.mean(rs)) / np.mean(rsamp))
    red_lw.append(100 * (np.mean(rsamp) - np.mean(rl)) / np.mean(rsamp))
fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.plot(ratios, red_rmt, "o-", color="#55A868", label="RMT denoised")
ax.plot(ratios, red_lw, "s-", color="#4C72B0", label="Ledoit-Wolf")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("q = N / T (assets / samples)")
ax.set_ylabel("OOS variance reduction vs sample (%)")
ax.set_title("RMT advantage grows as q rises (density regime)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(IMG_A, "fig5_nt_ratio.png")); plt.close(fig)
print("  A figures written to", IMG_A)

# ============================================================================
# ARTICLE B: Kalman filter for time-varying beta
# ============================================================================
print("=" * 60)
print("ARTICLE B: Kalman filter time-varying beta")
T = 600; alpha = 0.0003
true_beta = np.ones(T) * 1.0
true_beta[300:] += np.linspace(0, 0.6, T - 300)
rw = np.cumsum(np.random.randn(T) * 0.01)
true_beta += rw - rw.mean()
r_mkt = np.random.randn(T) * 0.012
r_ast = alpha + true_beta * r_mkt + np.random.randn(T) * 0.008

Q = 1e-4; R_var = 8e-5
beta_filt = np.zeros(T); beta_pred = np.zeros(T); beta_smooth = np.zeros(T)
innov = np.zeros(T); innov_var = np.zeros(T); P = np.zeros(T)
beta_est = 0.0; P_est = 1.0
for t in range(T):
    P_pred = P_est + Q
    H = r_mkt[t]; S = H * P_pred * H + R_var; K = P_pred * H / S
    innov[t] = r_ast[t] - (alpha + beta_est * H); innov_var[t] = S
    beta_filt[t] = beta_est + K * innov[t]; P[t] = (1 - K * H) * P_pred
    beta_pred[t] = beta_est; beta_est = beta_filt[t]; P_est = P[t]
beta_smooth[T - 1] = beta_filt[T - 1]
for t in range(T - 2, -1, -1):
    C = P[t] / (P[t] + Q)
    beta_smooth[t] = beta_filt[t] + C * (beta_smooth[t + 1] - beta_filt[t])

static_beta = np.sum(r_mkt * (r_ast - alpha)) / np.sum(r_mkt ** 2)
w = 60; roll = np.zeros(T)
for t in range(T):
    if t >= w:
        y = r_ast[t - w + 1:t + 1] - alpha; x = r_mkt[t - w + 1:t + 1]
        roll[t] = np.sum(x * y) / np.sum(x ** 2)

rmse_kalman = math.sqrt(np.mean((beta_filt - true_beta) ** 2))
rmse_smooth = math.sqrt(np.mean((beta_smooth - true_beta) ** 2))
rmse_roll = math.sqrt(np.nanmean((roll - true_beta) ** 2))
rmse_static = math.sqrt(np.mean((static_beta - true_beta) ** 2))
rmse_kalman_seg = math.sqrt(np.mean((beta_filt[300:] - true_beta[300:]) ** 2))
rmse_roll_seg = math.sqrt(np.mean((roll[300:] - true_beta[300:]) ** 2))
print(f"  static OLS beta (full sample): {static_beta:.4f}  (true mean {true_beta.mean():.4f})")
print(f"  RMSE vs true beta: Kalman {rmse_kalman:.4f} | Smooth {rmse_smooth:.4f} | Rolling {rmse_roll:.4f} | Static {rmse_static:.4f}")
print(f"  drift segment (t>=300) RMSE: Kalman {rmse_kalman_seg:.4f} vs Rolling {rmse_roll_seg:.4f}")
print(f"  final filtered beta: {beta_filt[-1]:.3f}  true: {true_beta[-1]:.3f}")
print(f"  innovation +/-2sigma covers {100*np.mean(np.abs(innov) < 2*np.sqrt(innov_var)):.1f}% of points")

# ---- FIG B1 ----
fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.plot(true_beta, "k-", lw=2, label="True beta_t")
ax.plot(beta_filt, color="#C44E52", lw=1.4, label="Kalman filter")
ax.plot(beta_smooth, color="#55A868", lw=1.4, label="Kalman smoother")
ax.plot(roll, color="#4C72B0", lw=1.2, ls="--", label="Rolling 60d OLS")
ax.axhline(static_beta, color="#8172B3", ls=":", lw=1.5, label=f"Full-sample OLS ({static_beta:.2f})")
ax.axvline(300, color="gray", ls="--", alpha=0.6)
ax.text(305, 1.05, "beta structural lift", fontsize=9)
ax.set_xlabel("Trading day t"); ax.set_ylabel("beta")
ax.set_title("State-space Kalman vs rolling/static OLS for time-varying beta")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(IMG_B, "fig1_beta_track.png")); plt.close(fig)

# ---- FIG B2 ----
pred = alpha + beta_pred * r_mkt
fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.plot(r_ast, color="#C44E52", lw=1.0, label="Asset realized return")
ax.plot(pred, color="#4C72B0", lw=1.0, label="Kalman one-step prediction")
ax.set_xlabel("Trading day t"); ax.set_ylabel("Daily return")
ax.set_title("Asset return vs Kalman filter one-step prediction")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(IMG_B, "fig2_fit.png")); plt.close(fig)

# ---- FIG B3: Q effect ----
def kalman_run(QQ):
    be = 0.0; Pe = 1.0; out = np.zeros(T)
    for t in range(T):
        Pp = Pe + QQ; H = r_mkt[t]; S = H * Pp * H + R_var; K = Pp * H / S
        out[t] = be + K * (r_ast[t] - (alpha + be * H)); Pe = (1 - K * H) * Pp; be = out[t]
    return out
fig, ax = plt.subplots(figsize=(7.6, 4.3))
for QQ, col, lab in [(1e-5, "#C44E52", "Q=1e-5 (smooth)"),
                     (1e-4, "#55A868", "Q=1e-4 (this paper)"),
                     (1e-3, "#4C72B0", "Q=1e-3 (aggressive)")]:
    ax.plot(kalman_run(QQ), lw=1.3, color=col, label=lab)
ax.plot(true_beta, "k--", lw=1.2, label="True beta_t")
ax.set_xlabel("Trading day t"); ax.set_ylabel("beta")
ax.set_title("State-noise Q controls tracking aggressiveness")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(IMG_B, "fig3_Q_effect.png")); plt.close(fig)

# ---- FIG B4 ----
fig, ax = plt.subplots(figsize=(7.2, 4.3))
names = ["Kalman filter", "Kalman smoother", "Rolling 60d", "Full-sample OLS"]
vals = [rmse_kalman, rmse_smooth, rmse_roll, rmse_static]
for b, v, c in zip(ax.bar(names, vals, color=["#C44E52", "#55A868", "#4C72B0", "#8172B3"], alpha=0.85),
                   vals, range(4)):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("RMSE of beta estimate")
ax.set_title("RMSE of four estimators vs true beta_t")
fig.tight_layout(); fig.savefig(os.path.join(IMG_B, "fig4_rmse.png")); plt.close(fig)

# ---- FIG B5 ----
sig = np.sqrt(innov_var)
fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.plot(innov, color="#4C72B0", lw=0.8)
ax.plot(2 * sig, "r--", lw=1.0, label="+2 sigma"); ax.plot(-2 * sig, "r--", lw=1.0, label="-2 sigma")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Trading day t"); ax.set_ylabel("Innovation (residual)")
ax.set_title("Kalman innovations with +/-2 sigma control band")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(IMG_B, "fig5_innovation.png")); plt.close(fig)
print("  B figures written to", IMG_B)
print("DONE")
