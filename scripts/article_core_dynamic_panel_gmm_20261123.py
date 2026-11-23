# -*- coding: utf-8 -*-
"""Dynamic-panel GMM: Nickell bias, difference GMM, system GMM.

Single source of truth for every number and figure in the article.

DGP (true parameters known):
    y_it = rho * y_{i,t-1} + beta * x_it + alpha_i + eps_it
    x_it ~ N(0,1) exogenous,  alpha_i ~ N(0, sigma_a^2),  eps_it ~ N(0,1)
with a stationary initial condition. Four estimators of rho:
    pooled OLS (ignores alpha_i), within/FE (demeans -> Nickell bias),
    Arellano-Bond difference GMM (levels lagged 2+ as instruments),
    Blundell-Bond system GMM (adds the level equation with d.y instruments).

One-step GMM with the standard instrument cross-product weighting. Everything
is vectorised over N individuals; fixed seeds make exec-replay identical to
results.json. We report the honest failures: FE biases down at small T,
difference GMM is weak-instrument biased as rho -> 1, and too many instruments
overfit (Hansen p-values rise).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261123
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "dynamic-panel-arellano-bond-gmm")
os.makedirs(OUT, exist_ok=True)

RHO_TRUE = 0.8
BETA_TRUE = 0.5
SIG_ALPHA = 0.5
SIG_EPS = 1.0


def gen_panel(N, T, rho, rng, beta=BETA_TRUE):
    """Return Y, X of shape (N,T) with a stationary start."""
    alpha = rng.normal(0.0, SIG_ALPHA, N)
    y0 = alpha / (1.0 - rho) + rng.normal(0.0, SIG_EPS / np.sqrt(1.0 - rho ** 2), N)
    X = rng.standard_normal((N, T))
    E = rng.standard_normal((N, T)) * SIG_EPS
    Y = np.empty((N, T))
    Y[:, 0] = rho * y0 + beta * X[:, 0] + alpha + E[:, 0]
    for t in range(1, T):
        Y[:, t] = rho * Y[:, t - 1] + beta * X[:, t] + alpha + E[:, t]
    return Y, X


def ols_estimators(Y, X):
    """Pooled OLS and within/FE, each returns [rho_hat, beta_hat]."""
    y = Y[:, 1:]
    lag = Y[:, :-1]
    xx = X[:, 1:]
    A = np.stack([lag, xx], axis=-1).reshape(-1, 2)
    b = y.reshape(-1)
    pooled = np.linalg.lstsq(A, b, rcond=None)[0]
    yd = y - y.mean(axis=1, keepdims=True)
    ld = lag - lag.mean(axis=1, keepdims=True)
    xd = xx - xx.mean(axis=1, keepdims=True)
    Aw = np.stack([ld, xd], axis=-1).reshape(-1, 2)
    fe = np.linalg.lstsq(Aw, yd.reshape(-1), rcond=None)[0]
    return pooled, fe


def diff_gmm(Y, X, collapse=False):
    """One-step Arellano-Bond difference GMM. Returns theta, J-stat, J p-value."""
    N, T = Y.shape
    M = T - 2
    dY = np.diff(Y, axis=1)          # (N, T-1), column m ~ period m+2
    dX = np.diff(X, axis=1)
    dep = dY[:, 1:M + 1]             # (N, M)  delta y_it
    reg = np.stack([dY[:, 0:M], dX[:, 1:M + 1]], axis=-1)   # (N, M, 2)
    # instrument Z[n,m,k] = y_{i,k+1} if k<=m  (levels lagged >=2)
    levels = Y[:, :M]
    mask = np.arange(M)[None, :, None] >= np.arange(M)[None, None, :]
    Z = levels[:, None, :] * mask
    if collapse:
        # Roodman-style collapse: each equation gets ONE instrument, the sum of
        # its available lags, placed on the diagonal -> still M instrument cols.
        row_sum = Z.sum(axis=2)               # (N, M)
        Zc = np.zeros((N, M, M))
        idx = np.arange(M)
        Zc[:, idx, idx] = row_sum
        Z = Zc
    Szy = np.einsum("nmk,nm->k", Z, dep)
    Szx = np.einsum("nmk,nml->kl", Z, reg)
    Szz = np.einsum("nmk,nmj->kj", Z, Z)
    K = Z.shape[2]
    W = np.linalg.inv(Szz / N + 1e-8 * np.eye(K))
    A = Szx.T @ W @ Szx
    b = Szx.T @ W @ Szy
    theta = np.linalg.solve(A, b)
    resid = dep - reg @ theta
    g = np.einsum("nmk,nm->k", Z, resid)
    # g is a sum over N individuals; W=(Szz/N)^-1, so J = N*g'Wg/N = g'(Szz/N)^-1 g/N
    J = float(g @ W @ g) / N
    dof = max(K - 2, 1)
    return theta, J, float(stats.chi2.sf(J, dof))


def sys_gmm(Y, X):
    """One-step Blundell-Bond system GMM (difference + level equations)."""
    N, T = Y.shape
    M = T - 2
    dY = np.diff(Y, axis=1)
    dX = np.diff(X, axis=1)
    dep_d = dY[:, 1:M + 1]
    reg_d = np.stack([dY[:, 0:M], dX[:, 1:M + 1]], axis=-1)   # (N,M,2)
    mask = np.arange(M)[None, :, None] >= np.arange(M)[None, None, :]
    Zd = Y[:, None, :M] * mask                                 # (N,M,M)
    # level equation for period t=m+2: y_it = rho y_{i,t-1} + beta x_it + a_i + e
    dep_l = Y[:, 2:M + 2]
    reg_l = np.stack([Y[:, 1:M + 1], X[:, 2:M + 2]], axis=-1)  # (N,M,2)
    Zl = np.stack([dY[:, 0:M], X[:, 2:M + 2]], axis=-1)        # (N,M,2)
    Szy = np.concatenate([np.einsum("nmk,nm->k", Zd, dep_d),
                          np.einsum("nmi,nm->i", Zl, dep_l)])
    Szx = np.vstack([np.einsum("nmk,nml->kl", Zd, reg_d),
                     np.einsum("nmi,nmj->ij", Zl, reg_l)])
    Szz = np.zeros((M + 2, M + 2))
    Szz[:M, :M] = np.einsum("nmk,nmj->kj", Zd, Zd)
    Szz[M:, M:] = np.einsum("nmi,nmj->ij", Zl, Zl)
    W = np.linalg.inv(Szz / N + 1e-8 * np.eye(M + 2))
    theta = np.linalg.solve(Szx.T @ W @ Szx, Szx.T @ W @ Szy)
    return theta


def one_rep(N, T, rho, rng, collapse=False):
    Y, X = gen_panel(N, T, rho, rng)
    pooled, fe = ols_estimators(Y, X)
    dgm, J, jp = diff_gmm(Y, X, collapse=collapse)
    sg = sys_gmm(Y, X)
    return np.array([pooled[0], fe[0], dgm[0], sg[0]]), \
           np.array([pooled[1], fe[1], dgm[1], sg[1]]), J, jp


def summarize(rho, T, N, R, collapse=False, seed=SEED):
    rng = np.random.default_rng(seed)
    rs = np.empty((R, 4))
    bs = np.empty((R, 4))
    js = np.empty(R)
    jps = np.empty(R)
    for r in range(R):
        rs[r], bs[r], js[r], jps[r] = one_rep(N, T, rho, rng, collapse=collapse)
    bias = rs.mean(axis=0) - rho
    rmse = np.sqrt(((rs - rho) ** 2).mean(axis=0))
    return {
        "rho": rho, "T": T, "N": N, "R": R, "collapse": collapse,
        "mean": [float(v) for v in rs.mean(axis=0)],
        "bias": [float(v) for v in bias],
        "rmse": [float(v) for v in rmse],
        "beta_mean": [float(v) for v in bs.mean(axis=0)],
        "J_mean": float(js.mean()), "j_p_mean": float(jps.mean()),
        "j_reject": float(np.mean(jps < 0.05)),
    }


def exp_nickell():
    """FE bias vs T, the four estimators side by side."""
    return [summarize(RHO_TRUE, T, 200, 300, seed=SEED + T)
            for T in (4, 5, 8, 12, 20, 40)]


def exp_rho_persistence():
    """As rho -> 1 difference GMM goes weak-instrument biased."""
    return [summarize(rho, 6, 200, 300, seed=SEED + int(rho * 1000))
            for rho in (0.3, 0.5, 0.7, 0.85, 0.95, 0.98)]


def exp_instruments():
    """More instruments -> overfit: bias worsens and Hansen p rises."""
    return [summarize(RHO_TRUE, T, 150, 300, seed=SEED + 5000 + T)
            for T in (5, 8, 12, 20, 30, 40)]


def exp_collapse():
    """Collapsing the instrument set at large T recovers some of the damage."""
    out = []
    for T in (12, 20, 40):
        a = summarize(RHO_TRUE, T, 150, 300, collapse=False, seed=SEED + 7000 + T)
        b = summarize(RHO_TRUE, T, 150, 300, collapse=True, seed=SEED + 7000 + T)
        out.append({"T": T, "full": a, "collapsed": b})
    return out


def exp_beta():
    """Beta (time-varying regressor) averaged over reps for each estimator."""
    rows = []
    for T in (5, 12, 40):
        rng = np.random.default_rng(SEED + 9000 + T)
        bs = np.empty((300, 4))
        for i in range(300):
            _, b, _, _ = one_rep(200, T, RHO_TRUE, rng)
            bs[i] = b
        rows.append({"T": T, "beta_mean_DGMBB": [float(v) for v in bs.mean(axis=0)]})
    return rows


# ------------------------------------------------------------------- figures
COL = ["#888888", "#e8684a", "#5b8ff9", "#2ca25f"]
LAB = ["Pooled OLS", "Within / FE", "Diff GMM (AB)", "System GMM (BB)"]


def fig1_nickell(nick):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    Ts = [r["T"] for r in nick]
    for k in range(4):
        ax[0].plot(Ts, [r["mean"][k] for r in nick], "o-", color=COL[k], label=LAB[k])
    ax[0].axhline(RHO_TRUE, color="black", ls="--", lw=1, label="true rho=0.8")
    ax[0].set_xlabel("T (time periods)"); ax[0].set_ylabel("mean rho-hat")
    ax[0].set_title("Nickell bias: FE catches up as T grows")
    ax[0].legend(fontsize=8)
    for k in (1, 2, 3):
        ax[1].plot(Ts, [r["bias"][k] for r in nick], "o-", color=COL[k], label=LAB[k])
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xlabel("T"); ax[1].set_ylabel("bias of rho-hat")
    ax[1].set_title("Bias: FE down, GMM near zero at moderate T")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); p = "fig1_nickell.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig2_persistence(pers):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    rhos = [r["rho"] for r in pers]
    for k in (2, 3):
        ax.plot(rhos, [r["mean"][k] for r in pers], "o-", color=COL[k], label=LAB[k])
    ax.plot(rhos, rhos, color="black", ls="--", lw=1, label="true = estimate")
    ax.set_xlabel("true rho"); ax.set_ylabel("mean rho-hat")
    ax.set_title(" rho -> 1: difference GMM weak-instrument bias, system GMM holds up")
    ax.legend(fontsize=9)
    fig.tight_layout(); p = "fig2_persistence.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig3_instruments(ins):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    Ts = [r["T"] for r in ins]
    ax[0].plot(Ts, [r["bias"][2] for r in ins], "o-", color=COL[2], label="Diff GMM")
    ax[0].plot(Ts, [r["bias"][3] for r in ins], "o-", color=COL[3], label="System GMM")
    ax[0].axhline(0, color="black", lw=1)
    ax[0].set_xlabel("T (more instruments)"); ax[0].set_ylabel("bias of rho-hat")
    ax[0].set_title("Too many instruments -> overfit bias")
    ax[0].legend(fontsize=9)
    ax[1].plot(Ts, [r["j_p_mean"] for r in ins], "o-", color="#7b3294")
    ax[1].axhline(0.05, color="black", ls="--", lw=1)
    ax[1].set_xlabel("T"); ax[1].set_ylabel("Hansen J p-value (mean)")
    ax[1].set_title("Hansen p rises: instruments fail to reject as T grows")
    fig.tight_layout(); p = "fig3_instruments.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig4_collapse(col):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    Ts = [r["T"] for r in col]
    ax.plot(Ts, [r["full"]["bias"][2] for r in col], "o-", color="#e8684a", label="AB full instruments")
    ax.plot(Ts, [r["collapsed"]["bias"][2] for r in col], "s--", color="#5b8ff9", label="AB collapsed")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("T"); ax.set_ylabel("difference-GMM rho bias")
    ax.set_title("Collapsing instruments reduces the overfit at large T")
    ax.legend(fontsize=9)
    fig.tight_layout(); p = "fig4_collapse.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig5_panel():
    """One panel: the four rho estimates on a single small-T sample."""
    rng = np.random.default_rng(SEED + 12345)
    Y, X = gen_panel(150, 6, RHO_TRUE, rng)
    pooled, fe = ols_estimators(Y, X)
    dgm, _, jp = diff_gmm(Y, X)
    sg = sys_gmm(Y, X)
    vals = [pooled[0], fe[0], dgm[0], sg[0]]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(range(4), vals, color=COL)
    ax.axhline(RHO_TRUE, color="black", ls="--", lw=1.5, label="true rho=0.8")
    ax.set_xticks(range(4)); ax.set_xticklabels(LAB, fontsize=8)
    ax.set_ylabel("rho-hat"); ax.set_title("One T=6 panel, N=150 (Hansen p=%.3f)" % jp)
    ax.legend(fontsize=8)
    fig.tight_layout(); p = "fig5_one_panel.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p, [float(v) for v in vals]


def main():
    nick = exp_nickell()
    pers = exp_rho_persistence()
    ins = exp_instruments()
    col = exp_collapse()
    beta = exp_beta()
    f5, f5vals = fig5_panel()
    r = {
        "setup": {"seed": SEED, "rho_true": RHO_TRUE, "beta_true": BETA_TRUE,
                  "sigma_alpha": SIG_ALPHA, "sigma_eps": SIG_EPS,
                  "estimators": LAB},
        "nickell": nick,
        "persistence": pers,
        "instruments": ins,
        "collapse": col,
        "beta_recovery": beta,
        "one_panel": f5vals,
    }
    r["figs"] = [fig1_nickell(nick), fig2_persistence(pers),
                 fig3_instruments(ins), fig4_collapse(col), f5]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
