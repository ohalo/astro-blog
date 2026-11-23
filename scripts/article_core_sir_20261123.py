# -*- coding: utf-8 -*-
"""Sliced Inverse Regression (SIR): when inverse regression finds the index.

Single source of truth for every number and figure in the article.

Single-index model  y = f(beta'x) + eps, x ~ N(0, I_p). The goal is the
direction beta (f is a nuisance). SIR (Li 1991) standardises x, slices y into
H bins, and takes the eigenvectors of the sliced means' covariance under the
whitening metric. It is y-aware, unlike PCA which just finds the largest
variance direction of x.

    z = Sigma_x^{-1/2}(x - xbar)
    M = sum_h (n_h/n) m_h m_h' ,  m_h = mean(z | y in slice h)
    beta_hat = Sigma_x^{-1/2} v_1 ,  v_1 = top eigenvector of M

The catch: SIR relies on the linear-design condition E[x | beta'x] being
linear in beta'x (true for elliptical x). For a symmetric link like f(t)=t^2,
E[z | y] cancels to zero and SIR collapses -- we show this honestly. We also
sweep slices H, sample size n, dimension p, and a two-index model, always
measuring |cos(angle)| with the truth (sign unidentifiable). Fixed seeds keep
exec-replay identical to results.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261123
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "sliced-inverse-regression-sir")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------- SIR core
def sir_direction(X, y, H, d=1, ridge=0.0):
    """Return d estimated directions in the ORIGINAL x space (columns)."""
    n, p = X.shape
    xc = X - X.mean(axis=0)
    cov = xc.T @ xc / n
    # whiten with optional ridge for n<=p
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-10, None)
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(w + ridge)) @ V.T
    Z = xc @ inv_sqrt

    order = np.argsort(y)
    zs = Z[order]
    ys = y[order]
    cuts = np.linspace(0, n, H + 1).astype(int)
    M = np.zeros((p, p))
    for h in range(H):
        lo, hi = cuts[h], cuts[h + 1]
        if hi - lo < 2:
            continue
        mh = zs[lo:hi].mean(axis=0)
        ph = (hi - lo) / n
        M += ph * np.outer(mh, mh)

    ev, evec = np.linalg.eigh(M)
    top = evec[:, ::-1][:, :d]
    return inv_sqrt @ top          # back to original scale


def abs_cos(u, v):
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    den = np.linalg.norm(u) * np.linalg.norm(v)
    return float(abs(u @ v) / den) if den > 0 else 0.0


def principal_angles(U, V):
    """cos of principal angles between two subspaces (columns orthonormal)."""
    Q, _ = np.linalg.qr(U)
    P, _ = np.linalg.qr(V)
    s = np.linalg.svd(Q.T @ P, compute_uv=False)
    return np.clip(s, 0, 1)


def sample_single(n, p, beta, link, sigma, rng):
    X = rng.standard_normal((n, p))
    eta = X @ beta
    y = link(eta) + sigma * rng.standard_normal(n)
    return X, y


def ols_direction(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return b[1:]


def pca_direction(X):
    xc = X - X.mean(axis=0)
    ev, evec = np.linalg.eigh(xc.T @ xc / len(X))
    return evec[:, -1]


def dir_quality(X, y, beta, H, ridge=0.0):
    b_sir = sir_direction(X, y, H, d=1, ridge=ridge)
    b_ols = ols_direction(X, y)
    b_pca = pca_direction(X)
    return {
        "SIR": abs_cos(b_sir, beta),
        "OLS": abs_cos(b_ols, beta),
        "PCA": abs_cos(b_pca, beta),
    }


# ---------------------------------------------------------------- experiments
def exp_single_index():
    """Linear / monotone / symmetric / oscillatory link functions."""
    rng = np.random.default_rng(6001)
    n, p, R, H, sigma = 400, 10, 200, 10, 0.5
    beta = np.zeros(p); beta[0] = 1.0; beta[1] = 0.5
    beta /= np.linalg.norm(beta)
    links = {
        "linear t": lambda t: t,
        "monotone t^3": lambda t: t ** 3,
        "symmetric t^2": lambda t: t ** 2,
        "oscillatory sin(2t)": lambda t: np.sin(2 * t),
    }
    out = {}
    for name, f in links.items():
        acc = {"SIR": [], "OLS": [], "PCA": []}
        for _ in range(R):
            X, y = sample_single(n, p, beta, f, sigma, rng)
            q = dir_quality(X, y, beta, H)
            for k in acc:
                acc[k].append(q[k])
        out[name] = {
            "R": R, "n": n, "p": p, "H": H,
            "mean": {k: float(np.mean(v)) for k, v in acc.items()},
            "sd": {k: float(np.std(v)) for k, v in acc.items()},
            "q05": {k: float(np.quantile(v, 0.05)) for k, v in acc.items()},
        }
    return out


def exp_slices():
    rng = np.random.default_rng(7001)
    n, p, R, sigma = 400, 10, 160, 0.5
    beta = np.zeros(p); beta[0] = 1.0; beta[1] = 0.5
    beta /= np.linalg.norm(beta)
    f = lambda t: t ** 3
    out = {}
    for H in [2, 4, 8, 12, 20, 40]:
        vals = []
        for _ in range(R):
            X, y = sample_single(n, p, beta, f, sigma, rng)
            vals.append(dir_quality(X, y, beta, H)["SIR"])
        out[str(H)] = {"H": H, "R": R, "mean": float(np.mean(vals)),
                       "sd": float(np.std(vals))}
    return out


def exp_sample_size():
    rng = np.random.default_rng(8001)
    p, R, H, sigma = 10, 160, 10, 0.5
    beta = np.zeros(p); beta[0] = 1.0; beta[1] = 0.5
    beta /= np.linalg.norm(beta)
    f = lambda t: t ** 3
    out = {}
    for n in [50, 100, 200, 400, 800]:
        vals = []
        for _ in range(R):
            X, y = sample_single(n, p, beta, f, sigma, rng)
            vals.append(dir_quality(X, y, beta, H)["SIR"])
        out[str(n)] = {"n": n, "R": R, "mean": float(np.mean(vals)),
                       "sd": float(np.std(vals))}
    return out


def exp_dimension():
    rng = np.random.default_rng(9001)
    n, R, H, sigma = 400, 120, 10, 0.5
    f = lambda t: t ** 3
    out = {}
    for p in [5, 10, 20, 40, 80]:
        beta = np.zeros(p); beta[0] = 1.0; beta[1] = 0.5
        beta /= np.linalg.norm(beta)
        vals = []
        for _ in range(R):
            X, y = sample_single(n, p, beta, f, sigma, rng)
            vals.append(dir_quality(X, y, beta, H)["SIR"])
        out[str(p)] = {"p": p, "n": n, "R": R, "mean": float(np.mean(vals)),
                       "sd": float(np.std(vals))}
    return out


def exp_two_index():
    rng = np.random.default_rng(10001)
    n, p, R, H, sigma = 600, 10, 120, 12, 0.5
    b1 = np.zeros(p); b1[0] = 1.0
    b2 = np.zeros(p); b2[1] = 1.0
    B_true = np.column_stack([b1, b2])
    out = {"R": R, "n": n, "p": p, "H": H, "sir": [], "pca": []}
    for _ in range(R):
        X = rng.standard_normal((n, p))
        y = (X @ b1) ** 3 + 0.8 * (X @ b2) + sigma * rng.standard_normal(n)
        U = sir_direction(X, y, H, d=2)
        out["sir"].append(principal_angles(U, B_true).tolist())
        out["pca"].append(principal_angles(pca_direction(X).reshape(-1, 1),
                                           B_true[:, :1]).tolist())
    return out


# ------------------------------------------------------------------- figures
def fig1_links(res):
    names = list(res)
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 5))
    for j, (m, color) in enumerate([("SIR", "#b23a2b"), ("OLS", "#5b8ff9"),
                                    ("PCA", "#9aa6b2")]):
        vals = [res[k]["mean"][m] for k in names]
        err = [res[k]["sd"][m] for k in names]
        ax.bar(x + (j - 1) * w, vals, w, yerr=err, capsize=3,
               color=color, label=m)
    ax.axhline(1 / np.sqrt(10), color="#333", ls=":", lw=1.2,
               label="random direction 1/sqrt(p)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("|cos| with true beta")
    ax.set_title("SIR recovers monotone index directions; symmetric t^2 breaks it")
    ax.legend(fontsize=9)
    fig.tight_layout()
    pth = os.path.join(OUT, "fig1_links.png")
    fig.savefig(pth, dpi=115); plt.close(fig)
    return "fig1_links.png"


def fig2_slices(sl, ss):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    Hs = sorted(int(k) for k in sl)
    a0 = ax[0]
    a0.errorbar(Hs, [sl[str(h)]["mean"] for h in Hs],
                yerr=[sl[str(h)]["sd"] for h in Hs],
                fmt="o-", color="#b23a2b", capsize=3)
    a0.axhline(1 / np.sqrt(10), color="#333", ls=":", lw=1.2)
    a0.set_xlabel("slices H (n=400, p=10)")
    a0.set_ylabel("|cos| with beta")
    a0.set_title("More slices help up to a point, then plateau")
    ns = sorted(int(k) for k in ss)
    ax[1].errorbar(ns, [ss[str(n)]["mean"] for n in ns],
                   yerr=[ss[str(n)]["sd"] for n in ns],
                   fmt="s-", color="#2e7d55", capsize=3)
    ax[1].axhline(1 / np.sqrt(10), color="#333", ls=":", lw=1.2)
    ax[1].set_xlabel("sample size n (p=10)")
    ax[1].set_ylabel("|cos| with beta")
    ax[1].set_title("Direction quality improves with n")
    fig.tight_layout()
    pth = os.path.join(OUT, "fig2_slices_n.png")
    fig.savefig(pth, dpi=115); plt.close(fig)
    return "fig2_slices_n.png"


def fig3_dimension(dim):
    ps = sorted(int(k) for k in dim)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.errorbar(ps, [dim[str(p)]["mean"] for p in ps],
                yerr=[dim[str(p)]["sd"] for p in ps],
                fmt="o-", color="#8a5a2b", capsize=3)
    ax.plot(ps, [1 / np.sqrt(p) for p in ps], "^--", color="#333",
            label="random direction 1/sqrt(p)")
    ax.set_xlabel("ambient dimension p (n=400)")
    ax.set_ylabel("|cos| with beta")
    ax.set_title("SIR degrades as p grows; linear-design noise accumulates")
    ax.legend(fontsize=9)
    fig.tight_layout()
    pth = os.path.join(OUT, "fig3_dimension.png")
    fig.savefig(pth, dpi=115); plt.close(fig)
    return "fig3_dimension.png"


def fig4_two_index(two):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    sir = np.array(two["sir"])
    for j, lab in enumerate(["dir 1", "dir 2"]):
        ax.hist(sir[:, j], bins=30, alpha=0.7, density=True,
                label="SIR " + lab)
    ax.axvline(1 / np.sqrt(two["p"]), color="#333", ls="--", lw=1.4,
               label="random 1/sqrt(p)")
    ax.set_xlabel("principal-angle cosine with true 2-D space")
    ax.set_ylabel("density")
    ax.set_title("Two-index model: SIR finds both directions (d=2)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    pth = os.path.join(OUT, "fig4_two_index.png")
    fig.savefig(pth, dpi=115); plt.close(fig)
    return "fig4_two_index.png"


def fig5_demo():
    """A single monotone-nonlinear sample: y vs beta'x and SIR weight curve."""
    rng = np.random.default_rng(11001)
    n, p, H = 400, 10, 10
    beta = np.zeros(p); beta[0] = 1.0; beta[1] = 0.5
    beta /= np.linalg.norm(beta)
    X = rng.standard_normal((n, p))
    y = (X @ beta) ** 3 + 0.5 * rng.standard_normal(n)
    eta = X @ beta
    bidx = np.argsort(np.argsort(y))
    slice_id = np.minimum((bidx * H) // n, H - 1)
    b_sir = sir_direction(X, y, H, d=1)
    scores = X @ (b_sir / np.linalg.norm(b_sir))
    slice_mean = [float(scores[slice_id == h].mean()) for h in range(H)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].scatter(eta, y, s=10, color="#5b8ff9", alpha=0.5)
    ax[0].set_xlabel("true index beta'x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Single-index data: y = (beta'x)^3 + noise")
    ax[1].plot(range(H), slice_mean, "o-", color="#b23a2b")
    ax[1].set_xlabel("y slice (low to high)")
    ax[1].set_ylabel("mean of SIR score")
    ax[1].set_title("Sliced means trace the index: |cos|=%.3f"
                    % abs_cos(b_sir, beta))
    fig.tight_layout()
    pth = os.path.join(OUT, "fig5_demo.png")
    fig.savefig(pth, dpi=115); plt.close(fig)
    return "fig5_demo.png", abs_cos(b_sir, beta)


def main():
    links = exp_single_index()
    sl = exp_slices()
    ss = exp_sample_size()
    dim = exp_dimension()
    two = exp_two_index()
    f5, f5cos = fig5_demo()
    r = {
        "setup": {"seed": SEED, "model": "y=f(beta'x)+eps", "sigma": 0.5,
                  "random_dir": 1 / np.sqrt(10)},
        "single_index": links,
        "slices": sl,
        "sample_size": ss,
        "dimension": dim,
        "two_index": {
            "R": two["R"], "n": two["n"], "p": two["p"], "H": two["H"],
            "sir_dir1_mean": float(np.mean(np.array(two["sir"])[:, 0])),
            "sir_dir2_mean": float(np.mean(np.array(two["sir"])[:, 1])),
            "random_dir": 1 / np.sqrt(two["p"]),
        },
        "demo_cos": f5cos,
    }
    r["figs"] = [fig1_links(links), fig2_slices(sl, ss), fig3_dimension(dim),
                 fig4_two_index(two), f5]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
