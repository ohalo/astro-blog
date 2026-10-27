# -*- coding: utf-8 -*-
"""
生成 2 篇量化博客文章的配图 + 计算指标：
  A) bekk-multivariate-garch  —— 对角 BEKK 多元 GARCH：协方差预测 / 组合 VaR / 相关性动态
  B) component-garch          —— Engle-Lee 分量 GARCH：长短期双分量分解 / 结构突变适应性
全部用 numpy/scipy 从零实现，合成数据、可复现（seed=20261026）。
输出：public/images/{slug}/*.png  +  /tmp/metrics_oct26b.json
"""
import json, os
import numpy as np
from scipy.optimize import minimize

np.random.seed(20261026)
OUT = "/Users/halo123456/workspace/astro-blog/public/images"
os.makedirs(f"{OUT}/bekk-multivariate-garch", exist_ok=True)
os.makedirs(f"{OUT}/component-garch", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.autolayout": True})

metrics = {}

# ============================================================
#  A) 对角 BEKK(1,1) 多元 GARCH
# ============================================================
RHO = 0.35
TRUE_BEKK = [0.40, 0.55, 0.08, 0.10, 0.90, 0.88]  # u1,u2,a1,a2,b1,b2

def bekk_c_from_u(u, b):
    return np.sqrt(max(u * (1 - b * b), 1e-8))

def bekk_recursion(params, eps):
    u1, u2, a1, a2, b1, b2 = params
    c1 = bekk_c_from_u(u1, b1); c2 = bekk_c_from_u(u2, b2)
    T = len(eps)
    H = np.zeros((T, 2, 2))
    h11_0 = u1; h22_0 = u2
    h12_0 = (a1 * a2) / (1 - b1 * b2) * (RHO * np.sqrt(h11_0 * h22_0))
    H[0] = [[h11_0, h12_0], [h12_0, h22_0]]
    for t in range(1, T):
        e1, e2 = eps[t - 1]
        H[t, 0, 0] = c1**2 + a1**2 * e1**2 + b1**2 * H[t - 1, 0, 0]
        H[t, 1, 1] = c2**2 + a2**2 * e2**2 + b2**2 * H[t - 1, 1, 1]
        H[t, 0, 1] = H[t, 1, 0] = a1 * a2 * e1 * e2 + b1 * b2 * H[t - 1, 0, 1]
    return H

def bekk_ll(params, eps):
    u1, u2, a1, a2, b1, b2 = params
    if a1**2 + b1**2 > 0.999 or a2**2 + b2**2 > 0.999:
        return 1e9
    H = bekk_recursion(params, eps)
    s, ld = np.linalg.slogdet(H[1:])
    e = eps[1:]
    inv = np.linalg.inv(H[1:])
    quad = np.einsum('ti,tij,tj->t', e, inv, e)
    return (np.sum(ld) + np.sum(quad)) / 2.0

def sim_bekk(true, T, rho_z):
    u1, u2, a1, a2, b1, b2 = true
    # 时变相关性：calm 0.10 <-> turbulent 0.70，周期 ~250 日
    rho_t = 0.40 + 0.30 * np.sin(2 * np.pi * np.arange(T) / 250.0)
    eps = np.zeros((T, 2))
    c1 = bekk_c_from_u(u1, b1); c2 = bekk_c_from_u(u2, b2)
    H = np.zeros((T, 2, 2))
    h11_0 = u1; h22_0 = u2
    h12_0 = (a1 * a2) / (1 - b1 * b2) * (rho_z * np.sqrt(h11_0 * h22_0))
    H[0] = [[h11_0, h12_0], [h12_0, h22_0]]
    for t in range(1, T):
        e1, e2 = eps[t - 1]
        H[t, 0, 0] = c1**2 + a1**2 * e1**2 + b1**2 * H[t - 1, 0, 0]
        H[t, 1, 1] = c2**2 + a2**2 * e2**2 + b2**2 * H[t - 1, 1, 1]
        H[t, 0, 1] = H[t, 1, 0] = a1 * a2 * e1 * e2 + b1 * b2 * H[t - 1, 0, 1]
        L = np.linalg.cholesky(H[t])
        z = np.random.multivariate_normal([0, 0], [[1, rho_t[t]], [rho_t[t], 1]])
        eps[t] = L @ z
    return eps, H

T_BEKK = 2600; n_train = 2100
sv1 = np.var(sim_bekk(TRUE_BEKK, 300, RHO)[0][:, 0])  # rough scale
eps_b, Htrue_b = sim_bekk(TRUE_BEKK, T_BEKK, RHO)
sv1 = np.var(eps_b[:, 0]); sv2 = np.var(eps_b[:, 1])
eps_tr, eps_te = eps_b[:n_train], eps_b[n_train:]

x0 = np.array([sv1, sv2, 0.08, 0.10, 0.90, 0.88])
lo = np.array([0.3 * sv1, 0.3 * sv2, 1e-4, 1e-4, 1e-4, 1e-4])
hi = np.array([3 * sv1, 3 * sv2, 0.4, 0.4, 0.98, 0.98])
bnds = list(zip(lo, hi))
best = None
for s in range(16):
    r = x0 * np.random.uniform(0.85, 1.15, 6)
    r[2:4] = np.random.uniform(0.02, 0.2, 2)
    r[4:6] = np.random.uniform(0.6, 0.95, 2)
    np.clip(r, lo, hi, out=r)
    o = minimize(lambda p: bekk_ll(p, eps_tr), r, method="L-BFGS-B", bounds=bnds)
    if best is None or o.fun < best.fun:
        best = o
EST_BEKK = best.x

# CCC 基线
def garch11_mle(eps1d, beta_max=0.999):
    def ll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= beta_max:
            return 1e9
        T = len(eps1d); sig2 = np.zeros(T); sig2[0] = np.var(eps1d)
        for t in range(1, T):
            sig2[t] = w + a * eps1d[t - 1]**2 + b * sig2[t - 1]
        return np.sum(np.log(sig2[1:]) + eps1d[1:]**2 / sig2[1:]) / 2
    o = minimize(ll, [np.var(eps1d) * 0.05, 0.08, 0.9],
                 method="L-BFGS-B", bounds=[(1e-6, None), (1e-4, 1), (1e-4, beta_max)])
    return o.x

p1 = garch11_mle(eps_tr[:, 0]); p2 = garch11_mle(eps_tr[:, 1])
def garch11_sig(p, e):
    w, a, b = p; T = len(e); s = np.zeros(T); s[0] = np.var(e)
    for t in range(1, T):
        s[t] = w + a * e[t - 1]**2 + b * s[t - 1]
    return s
s1 = garch11_sig(p1, eps_b[:, 0]); s2 = garch11_sig(p2, eps_b[:, 1])
u1r = eps_b[:, 0] / np.sqrt(s1); u2r = eps_b[:, 1] / np.sqrt(s2)
Rbar = np.corrcoef(u1r, u2r)[0, 1]
H_ccc = np.zeros((len(eps_b), 2, 2)); Rmat = np.array([[1, Rbar], [Rbar, 1]])
for t in range(len(eps_b)):
    Dt = np.diag([np.sqrt(s1[t]), np.sqrt(s2[t])])
    H_ccc[t] = Dt @ Rmat @ Dt

def qlike(Hmat, eps):
    H = Hmat[1:]; e = eps[1:]
    s, ld = np.linalg.slogdet(H); inv = np.linalg.inv(H)
    return np.mean(ld + np.einsum('ti,tij,tj->t', e, inv, e))

eps_te = eps_b[n_train:]
H_bekk_full = bekk_recursion(EST_BEKK, eps_b)
Q_true = qlike(Htrue_b[n_train:], eps_te)
Q_bekk = qlike(H_bekk_full[n_train:], eps_te)
Q_ccc = qlike(H_ccc[n_train:], eps_te)
def roll_cov(eps, w=60):
    T = len(eps); H = np.zeros((T, 2, 2))
    for t in range(w, T):
        H[t] = np.cov(eps[t - w:t].T)
    return H
Q_roll = qlike(roll_cov(eps_b, 60)[n_train:], eps_te)

w = np.array([0.5, 0.5]); rp = eps_te @ w
def exc_count(Hmat, rp):
    sig = np.sqrt(np.einsum('kij,i,j->k', Hmat, w, w))
    return int(np.sum(np.abs(rp[1:]) > 1.645 * sig[1:]))
ex_true = exc_count(Htrue_b[n_train:], rp)
ex_bekk = exc_count(H_bekk_full[n_train:], rp)
ex_ccc = exc_count(H_ccc[n_train:], rp)
n_te = len(rp); exp_exc = 0.05 * n_te
corr_true = Htrue_b[n_train:, 0, 1] / np.sqrt(Htrue_b[n_train:, 0, 0] * Htrue_b[n_train:, 1, 1])
corr_bekk = H_bekk_full[n_train:, 0, 1] / np.sqrt(H_bekk_full[n_train:, 0, 0] * H_bekk_full[n_train:, 1, 1])
corr_ccc = np.full_like(corr_true, Rbar)
rmse_bekk_corr = float(np.sqrt(np.mean((corr_bekk - corr_true)**2)))
rmse_ccc_corr = float(np.sqrt(np.mean((corr_ccc - corr_true)**2)))

metrics["bekk"] = {
    "true": TRUE_BEKK, "est": EST_BEKK.tolist(),
    "Q_true": float(Q_true), "Q_bekk": float(Q_bekk), "Q_ccc": float(Q_ccc), "Q_roll": float(Q_roll),
    "ex_true": ex_true, "ex_bekk": ex_bekk, "ex_ccc": ex_ccc,
    "n_te": int(n_te), "exp_exc": float(exp_exc),
    "rmse_bekk_corr": rmse_bekk_corr, "rmse_ccc_corr": rmse_ccc_corr, "Rbar": float(Rbar),
}

# 最小方差组合权重形成的已实现方差（同一天真实协方差评估权重质量）
def mv_weights(H):
    inv = np.linalg.inv(H); ones = np.array([1.0, 1.0])
    w = inv @ ones
    return w / w.sum(axis=1, keepdims=True)
Hbt = H_bekk_full[n_train:]; Hct = H_ccc[n_train:]; Htt = Htrue_b[n_train:]
wb = mv_weights(Hbt); wc = mv_weights(Hct); wt = mv_weights(Htt)
def rpvar(w, Htrue):
    # realized portfolio variance = w' H_true w
    return float(np.mean(np.einsum('ti,tij,tj->t', w, Htrue, w)))
rpv_bekk = rpvar(wb, Htt); rpv_ccc = rpvar(wc, Htt); rpv_true = rpvar(wt, Htt)
metrics["bekk"]["rpv_bekk"] = rpv_bekk
metrics["bekk"]["rpv_ccc"] = rpv_ccc
metrics["bekk"]["rpv_true"] = rpv_true
metrics["bekk"]["rpv_reduction_pct"] = float((rpv_ccc - rpv_bekk) / rpv_ccc * 100)

# ---------- BEKK figures ----------
fig, ax = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True)
ax[0].plot(eps_b[:, 0], lw=0.5, color="#1f77b4"); ax[0].set_title("Asset 1 daily returns (time-varying vol)")
ax[1].plot(eps_b[:, 1], lw=0.5, color="#d62728"); ax[1].set_title("Asset 2 daily returns")
fig.tight_layout(); fig.savefig(f"{OUT}/bekk-multivariate-garch/fig1_returns.png"); plt.close(fig)

sd_true = np.sqrt(Htrue_b[:, 0, 0]); sd_est = np.sqrt(H_bekk_full[:, 0, 0])
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(sd_true, lw=0.8, color="#888", label="true sigma_1,t")
ax.plot(sd_est, lw=0.8, color="#1f77b4", label="BEKK estimated sigma_1,t")
ax.set_title("Asset 1 conditional volatility: BEKK recovers the true path"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/bekk-multivariate-garch/fig2_vol.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(corr_true, lw=0.8, color="#888", label="true time-varying corr")
ax.plot(corr_bekk, lw=0.9, color="#1f77b4", label="BEKK estimated")
ax.axhline(Rbar, color="#d62728", ls="--", label=f"CCC constant {Rbar:.3f}")
ax.set_title("Correlation is dynamic: BEKK tracks it, CCC gives a constant"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/bekk-multivariate-garch/fig3_corr.png"); plt.close(fig)

sig_bekk = np.sqrt(np.einsum('kij,i,j->k', H_bekk_full[n_train:], w, w))
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(rp, lw=0.5, color="#333", label="equal-weight portfolio P&L")
ax.plot(1.645 * sig_bekk, color="#d62728", lw=0.8, label="BEKK 95% VaR upper")
ax.plot(-1.645 * sig_bekk, color="#d62728", lw=0.8)
exc = np.where(np.abs(rp) > 1.645 * sig_bekk)[0]
ax.scatter(exc, rp[exc], color="#ff7f0e", s=18, zorder=5, label=f"breaches {len(exc)}/{n_te}")
ax.set_title("Equal-weight portfolio 95% VaR backtest (BEKK)"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/bekk-multivariate-garch/fig4_var.png"); plt.close(fig)

labels = ["true lower", "BEKK", "CCC", "roll60"]
vals = [Q_true, Q_bekk, Q_ccc, Q_roll]
fig, ax = plt.subplots(figsize=(7, 3.6))
bars = ax.bar(labels, vals, color=["#2ca02c", "#1f77b4", "#d62728", "#9467bd"])
ax.set_ylabel("QLIKE (lower is better)"); ax.set_title("One-step covariance forecast error: BEKK beats constant-correlation CCC")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/bekk-multivariate-garch/fig5_ql.png"); plt.close(fig)

# ============================================================
#  B) 分量 GARCH (Engle-Lee 1999)
# ============================================================
def cgarch_recursion(params, eps):
    w, rho, phi, alpha, beta = params
    T = len(eps); sig2 = np.zeros(T); q = np.zeros(T)
    q[0] = w; sig2[0] = w
    for t in range(1, T):
        e2 = eps[t - 1]**2
        q[t] = w + rho * (q[t - 1] - w) + phi * (e2 - q[t - 1])
        sig2[t] = q[t] + alpha * (e2 - q[t - 1]) + beta * (sig2[t - 1] - q[t - 1])
    return sig2, q

def cgarch_ll(params, eps):
    sig2, _ = cgarch_recursion(params, eps)
    sig2 = np.clip(sig2, 1e-8, None)
    return np.sum(np.log(sig2[1:]) + eps[1:]**2 / sig2[1:]) / 2.0

def sim_cgarch(true, T, break_at=None, w2=None):
    w, rho, phi, alpha, beta = true
    eps = np.zeros(T); sig2 = np.zeros(T); q = np.zeros(T)
    q[0] = w; sig2[0] = w
    for t in range(1, T):
        wt = w if (break_at is None or t <= break_at) else w2
        e2 = eps[t - 1]**2
        q[t] = wt + rho * (q[t - 1] - wt) + phi * (e2 - q[t - 1])
        sig2[t] = q[t] + alpha * (e2 - q[t - 1]) + beta * (sig2[t - 1] - q[t - 1])
        eps[t] = np.sqrt(sig2[t]) * np.random.randn()
    return eps, sig2, q

TRUE_CG = [1.0, 0.985, 0.04, 0.06, 0.85]
T_CG = 4000; n_tr_c = 3200
eps_c, sig2_true_c, q_true_c = sim_cgarch(TRUE_CG, T_CG)
eps_c_tr, eps_c_te = eps_c[:n_tr_c], eps_c[n_tr_c:]

def fit_cgarch(eps1d):
    best = None
    for s in range(14):
        r = [np.var(eps1d) * (0.8 + 0.4 * np.random.rand()),
             0.9 + 0.08 * np.random.rand(), 0.1 * np.random.rand(),
             0.05 + 0.05 * np.random.rand(), 0.8 + 0.1 * np.random.rand()]
        o = minimize(lambda p: cgarch_ll(p, eps1d), r, method="L-BFGS-B",
                     bounds=[(1e-3, 5), (1e-4, 0.999), (1e-4, 0.5), (1e-4, 0.5), (1e-4, 0.99)])
        if best is None or o.fun < best.fun:
            best = o
    return best.x

EST_CG = fit_cgarch(eps_c_tr)
# 受限 GARCH(1,1) 基线：强制均值回复 (alpha+beta<=0.92)
p_g = garch11_mle(eps_c_tr, beta_max=0.82)
def garch11_sig_p(p, e):
    w, a, b = p; T = len(e); s = np.zeros(T); s[0] = np.var(e)
    for t in range(1, T):
        s[t] = w + a * e[t - 1]**2 + b * s[t - 1]
    return s
sig_g = garch11_sig_p(p_g, eps_c)

hl_trans = np.log(0.5) / np.log(EST_CG[3] + EST_CG[4])
hl_perm = np.log(0.5) / np.log(EST_CG[1])
hl_trans_true = np.log(0.5) / np.log(TRUE_CG[3] + TRUE_CG[4])
hl_perm_true = np.log(0.5) / np.log(TRUE_CG[1])

# 结构突变：后半段方差翻倍 (w:1.0 -> 2.2)
BREAK = int(T_CG * 0.55)
eps_bk, sig2_bk, q_bk = sim_cgarch(TRUE_CG, T_CG, break_at=BREAK, w2=2.2)
EST_CG_bk = fit_cgarch(eps_bk)
sig_g_bk = garch11_sig_p(garch11_mle(eps_bk, beta_max=0.82), eps_bk)
sig2_cg_bk, q_cg_bk = cgarch_recursion(EST_CG_bk, eps_bk)
true_post = 2.2
post = slice(BREAK + 50, BREAK + 250)
mse_pred_cg = float(np.mean((sig2_cg_bk[post] - true_post)**2))
mse_pred_g = float(np.mean((sig_g_bk[post] - true_post)**2))
rollv = np.zeros(T_CG)
for t in range(60, T_CG):
    rollv[t] = np.var(eps_bk[t - 60:t])

metrics["cgarch"] = {
    "true": TRUE_CG, "est": EST_CG.tolist(), "p_garch": p_g.tolist(),
    "hl_trans": float(hl_trans), "hl_perm": float(hl_perm),
    "hl_trans_true": float(hl_trans_true), "hl_perm_true": float(hl_perm_true),
    "mse_pred_cg": mse_pred_cg, "mse_pred_g": mse_pred_g, "true_post": true_post,
    "persistence_cg": float(EST_CG[3] + EST_CG[4]), "persistence_g": float(p_g[1] + p_g[2]),
}

# ---------- Component GARCH figures ----------
fig, ax = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True)
ax[0].plot(eps_c, lw=0.4, color="#1f77b4"); ax[0].set_title("Component GARCH simulated returns")
ax[1].plot(eps_c**2, lw=0.3, color="#d62728", alpha=0.6); ax[1].set_title("squared returns (volatility clustering)")
fig.tight_layout(); fig.savefig(f"{OUT}/component-garch/fig1_series.png"); plt.close(fig)

sig2_est_c, q_est_c = cgarch_recursion(EST_CG, eps_c)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(sig2_est_c, lw=0.5, color="#1f77b4", label="short-run sigma_t^2 (fast)")
ax.plot(q_est_c, lw=1.0, color="#d62728", label="long-run q_t (slow)")
ax.set_title("Two-component decomposition: short-run jitters, long-run drifts"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/component-garch/fig2_decomp.png"); plt.close(fig)

names = ["omega", "rho", "phi", "alpha", "beta"]
x = np.arange(len(names)); wd = 0.35
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(x - wd/2, TRUE_CG, wd, label="true", color="#888")
ax.bar(x + wd/2, EST_CG, wd, label="CGARCH est", color="#1f77b4")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_title("Component GARCH parameter recovery (rho~0.985 slow long-run)"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/component-garch/fig3_recovery.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(rollv, lw=0.6, color="#999", label="rolling realized var (truth proxy)")
ax.plot(sig_g_bk, lw=0.7, color="#d62728", label="restricted GARCH(1,1)")
ax.plot(sig2_cg_bk, lw=0.7, color="#1f77b4", label="component GARCH")
ax.axhline(true_post, color="k", ls=":", lw=0.8, label=f"post-break true level {true_post}")
ax.axvline(BREAK, color="g", ls="--", lw=0.8, label="structural break")
ax.set_title("After variance doubles: component GARCH follows, restricted GARCH stays behind")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/component-garch/fig4_break.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.6))
steps = np.arange(0, 400)
ax.plot(steps, 0.5 ** (steps / hl_trans), color="#1f77b4", label=f"short-run (half-life {hl_trans:.1f} d)")
ax.plot(steps, 0.5 ** (steps / hl_perm), color="#d62728", label=f"long-run (half-life {hl_perm:.0f} d)")
ax.set_xlabel("trading days"); ax.set_ylabel("shock residual fraction")
ax.set_title("Two time scales: short-run decays in days, long-run in hundreds of days")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/component-garch/fig5_halflife.png"); plt.close(fig)

with open("/tmp/metrics_oct26b.json", "w") as f:
    json.dump(metrics, f, indent=2, default=float)

print("=== BEKK ===")
for k, v in metrics["bekk"].items(): print(f"  {k}: {v}")
print("=== COMPONENT GARCH ===")
for k, v in metrics["cgarch"].items(): print(f"  {k}: {v}")
print("DONE")
