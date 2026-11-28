# -*- coding: utf-8 -*-
"""Signed graph balance and long-short hedge experiment (deterministic)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261128
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "signed-graph-portfolio-balance")
os.makedirs(OUT, exist_ok=True)

N = 40
HALF = N // 2
TRUE_SIGN = np.r_[np.ones(HALF), -np.ones(HALF)]
FACTOR_VAR = np.array([1.0, 0.3, 0.3])
IDIO = 0.7


def loadings(rng, idio=IDIO):
    b0 = TRUE_SIGN * 1.8
    b1 = rng.normal(0.0, 0.5, N)
    b2 = rng.normal(0.0, 0.5, N)
    return np.column_stack((b0, b1, b2))


def make_returns(seed, n, idio=IDIO, B=None):
    rng = np.random.default_rng(seed)
    if B is None:
        B = loadings(rng, idio)
    f = rng.normal(size=(n, 3)) * np.sqrt(FACTOR_VAR)
    e = rng.normal(0.0, idio, size=(n, N))
    return f @ B.T + e, B


def corr_matrix(X):
    Z = X - X.mean(axis=0)
    C = Z.T @ Z
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def upper_pairs():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


PAIRS = upper_pairs()


def signed_graph(C, edge_fraction=0.25, sign_flip=0.0, rng_seed=SEED):
    order = np.argsort(-np.abs(np.array([C[i, j] for i, j in PAIRS])))
    keep = set(order[: int(round(edge_fraction * len(PAIRS)))])
    A = np.zeros((N, N))
    rng = np.random.default_rng(rng_seed)
    for idx, (i, j) in enumerate(PAIRS):
        if idx not in keep:
            continue
        s = 1.0 if C[i, j] >= 0 else -1.0
        if sign_flip > 0 and rng.random() < sign_flip:
            s = -s
        A[i, j] = A[j, i] = s
    return A


def connected_components(A):
    seen = set()
    count = 0
    for i in range(N):
        if i in seen:
            continue
        count += 1
        stack = [i]
        seen.add(i)
        while stack:
            u = stack.pop()
            for v in np.where(np.abs(A[u]) > 0)[0]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    return count


def signed_laplacian(A):
    return np.diag(np.sum(np.abs(A), axis=1)) - A


def frustration(A):
    binary = (np.abs(A) > 0).astype(float)
    tri = np.trace(binary @ binary @ binary) / 6.0
    if tri == 0:
        return 0.0
    prod = np.trace(A @ A @ A) / (6.0 * tri)
    return float(np.clip((1.0 - prod) / 2.0, 0.0, 1.0))


def fiedler_sign(A):
    L = signed_laplacian(A)
    vals, vecs = np.linalg.eigh(L)
    v = vecs[:, 0]
    s = np.sign(v)
    if s @ TRUE_SIGN < 0:
        s = -s
    return s, float(vals[0]), vals


def weights_from_sign(s):
    w = np.zeros(N)
    w[s > 0] = -1.0 / max(1, np.sum(s > 0))
    w[s < 0] = 1.0 / max(1, np.sum(s < 0))
    return w


def portfolio_vol(R, w):
    return float(np.std(R @ w, ddof=1))


def sign_accuracy(s):
    return float(np.mean(s == TRUE_SIGN))


# Baseline: population graph, sample graph, and a graph with 25% signs flipped.
B = loadings(np.random.default_rng(SEED))
train, _ = make_returns(SEED, 500, B=B)
test, _ = make_returns(SEED + 1, 1000, B=B)
B_pop = loadings(np.random.default_rng(SEED))
cov_pop = B_pop @ np.diag(FACTOR_VAR) @ B_pop.T + np.eye(N) * IDIO ** 2
C_pop = cov_pop / np.outer(np.sqrt(np.diag(cov_pop)), np.sqrt(np.diag(cov_pop)))
C_train = corr_matrix(train)

A_pop = signed_graph(C_pop, 0.25)
A_train = signed_graph(C_train, 0.25)
A_flip = signed_graph(C_train, 0.25, sign_flip=0.25, rng_seed=SEED + 2)

s_pop, lam_pop, eig_pop = fiedler_sign(A_pop)
s_train, lam_train, eig_train = fiedler_sign(A_train)
s_flip, lam_flip, eig_flip = fiedler_sign(A_flip)

w_true = weights_from_sign(TRUE_SIGN)
w_train = weights_from_sign(s_train)
w_flip = weights_from_sign(s_flip)
rng_rand = np.random.default_rng(SEED + 3)
s_rand = TRUE_SIGN * rng_rand.choice([-1.0, 1.0], size=N)
w_rand = weights_from_sign(s_rand)

vol_true = portfolio_vol(test, w_true)
vol_train = portfolio_vol(test, w_train)
vol_flip = portfolio_vol(test, w_flip)
vol_rand = portfolio_vol(test, w_rand)
vol_equal = portfolio_vol(test, weights_from_sign(np.where(TRUE_SIGN > 0, 1.0, 0.0)))

cross_corr = float(np.mean(C_pop[np.ix_(TRUE_SIGN > 0, TRUE_SIGN < 0)]))
within_corr = float(np.mean(C_pop[np.ix_(TRUE_SIGN > 0, TRUE_SIGN > 0)][np.triu_indices(HALF, 1)]))

# Fig 1: sample signed correlation heatmap.
fig, ax = plt.subplots(figsize=(7.2, 5.8))
im = ax.imshow(C_train, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("Asset index")
ax.set_ylabel("Asset index")
ax.set_title("Sample signed correlation, n=500")
fig.colorbar(im, ax=ax, label="Correlation")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_signed_correlation.png"), dpi=150)
plt.close(fig)

# Fig 2: signed Laplacian spectra.
fig, ax = plt.subplots(figsize=(8, 4.8))
idx = np.arange(1, N + 1)
ax.plot(idx, eig_pop, "o-", color="#1a7f4b", lw=1.6, label="Population signs")
ax.plot(idx, eig_train, "s-", color="#1764ab", lw=1.6, label="Sample signs")
ax.plot(idx, eig_flip, "^-", color="#c23b22", lw=1.6, label="25% signs flipped")
ax.set_xlabel("Eigenvalue rank")
ax.set_ylabel("Signed Laplacian eigenvalue")
ax.set_title("Balance shows up in the smallest eigenvalue")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_laplacian_spectrum.png"), dpi=150)
plt.close(fig)

# Fig 3: recovered partition and resulting long-short weights.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
axes[0].plot(TRUE_SIGN, color="#333333", lw=1.6, label="True camp")
axes[0].plot(s_train, color="#1764ab", lw=1.2, alpha=0.85, label="Fiedler camp")
axes[0].set_xlabel("Asset index")
axes[0].set_ylabel("Camp sign")
axes[0].set_title("Recovered signed partition")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].bar(np.arange(N), w_train, color=np.where(w_train > 0, "#c23b22", "#1764ab"))
axes[1].set_xlabel("Asset index")
axes[1].set_ylabel("Portfolio weight")
axes[1].set_title("Dollar-neutral long-short legs")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_partition_weights.png"), dpi=150)
plt.close(fig)

# Fig 4: edge sparsity and sign noise.
edge_fractions = [0.05, 0.10, 0.15, 0.25, 0.40, 0.60]
sparse_lam, sparse_acc, sparse_vol, sparse_comp = [], [], [], []
for frac in edge_fractions:
    A = signed_graph(C_train, frac)
    s, lam, _ = fiedler_sign(A)
    sparse_lam.append(lam)
    sparse_acc.append(sign_accuracy(s))
    sparse_vol.append(portfolio_vol(test, weights_from_sign(s)))
    sparse_comp.append(connected_components(A))

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(edge_fractions, sparse_vol, "o-", color="#c23b22", lw=2, label="Test vol")
ax2 = ax.twinx()
ax2.plot(edge_fractions, sparse_lam, "s--", color="#1764ab", lw=1.7, label="Smallest Laplacian eigenvalue")
ax2.plot(edge_fractions, sparse_acc, "^--", color="#1a7f4b", lw=1.7, label="Camp accuracy")
ax.set_xlabel("Fraction of asset pairs kept as edges")
ax.set_ylabel("Daily portfolio volatility")
ax2.set_ylabel("Eigenvalue / accuracy")
ax.set_title("Too few signed edges lose the partition")
ax.legend(loc="upper right")
ax2.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_edge_sparsity.png"), dpi=150)
plt.close(fig)

# Fig 5: sample length and idiosyncratic noise.
sample_sizes = [60, 120, 250, 500, 1000, 2000]
size_acc, size_vol, size_lam = [], [], []
for n in sample_sizes:
    Xn, _ = make_returns(SEED + 10, n, B=B)
    An = signed_graph(corr_matrix(Xn), 0.25)
    sn, ln, _ = fiedler_sign(An)
    size_acc.append(sign_accuracy(sn))
    size_vol.append(portfolio_vol(test, weights_from_sign(sn)))
    size_lam.append(ln)

idio_levels = [0.7, 2.5, 5.0, 8.0, 12.0, 20.0]
idio_acc,idio_vol, idio_corr = [], [], []
for level in idio_levels:
    Xi, _ = make_returns(SEED + 20, 500, idio=level, B=B)
    Cx = corr_matrix(Xi)
    Ai = signed_graph(Cx, 0.25)
    si, _, _ = fiedler_sign(Ai)
    idio_acc.append(sign_accuracy(si))
    idio_vol.append(portfolio_vol(test, weights_from_sign(si)))
    idio_corr.append(float(np.mean(Cx[np.ix_(TRUE_SIGN > 0, TRUE_SIGN < 0)])))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
axes[0].plot(sample_sizes, size_acc, "o-", color="#1a7f4b", lw=2, label="Camp accuracy")
axes[0].plot(sample_sizes, size_vol, "s--", color="#c23b22", lw=1.7, label="Test vol")
axes[0].set_xlabel("Training sample length")
axes[0].set_ylabel("Accuracy / volatility")
axes[0].set_title("Correlation sign needs enough history")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot(idio_levels, idio_corr, "o-", color="#7b3fa1", lw=2, label="Cross-camp correlation")
axes[1].plot(idio_levels, idio_acc, "s--", color="#1a7f4b", lw=1.7, label="Camp accuracy")
axes[1].plot(idio_levels, idio_vol, "^--", color="#c23b22", lw=1.7, label="Test vol")
axes[1].set_xlabel("Idiosyncratic volatility")
axes[1].set_ylabel("Correlation / accuracy / volatility")
axes[1].set_title("Idiosyncratic noise erases signed edges")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_sample_noise.png"), dpi=150)
plt.close(fig)

results = {
    "assets": N,
    "true_cross_camp_correlation": cross_corr,
    "true_within_camp_correlation": within_corr,
    "edge_fraction": 0.25,
    "population_lambda_min": lam_pop,
    "sample_lambda_min": lam_train,
    "flipped_lambda_min": lam_flip,
    "population_frustration": frustration(A_pop),
    "sample_frustration": frustration(A_train),
    "flipped_frustration": frustration(A_flip),
    "sample_camp_accuracy": sign_accuracy(s_train),
    "flipped_camp_accuracy": sign_accuracy(s_flip),
    "vol_long_camp": vol_equal,
    "vol_true_partition": vol_true,
    "vol_sample_partition": vol_train,
    "vol_flipped_partition": vol_flip,
    "vol_random_partition": vol_rand,
    "factor_exposure_true": float(w_true @ B[:, 0]),
    "factor_exposure_sample": float(w_train @ B[:, 0]),
    "factor_exposure_flipped": float(w_flip @ B[:, 0]),
    "factor_exposure_random": float(w_rand @ B[:, 0]),
    "factor_purity_true": abs(float(w_true @ B[:, 0])) / vol_true,
    "factor_purity_sample": abs(float(w_train @ B[:, 0])) / vol_train,
    "factor_purity_flipped": abs(float(w_flip @ B[:, 0])) / vol_flip,
    "vol_reduction_pct": 100.0 * (1.0 - vol_train / vol_equal),
    "vol_gap_vs_oracle_pct": 100.0 * (vol_train / vol_true - 1.0),
    "edge_fractions": edge_fractions,
    "sparse_lambdas": [float(x) for x in sparse_lam],
    "sparse_accuracies": [float(x) for x in sparse_acc],
    "sparse_vols": [float(x) for x in sparse_vol],
    "sparse_components": sparse_comp,
    "sample_sizes": sample_sizes,
    "size_accuracies": [float(x) for x in size_acc],
    "size_vols": [float(x) for x in size_vol],
    "idio_levels": idio_levels,
    "idio_cross_correlations": idio_corr,
    "idio_accuracies": [float(x) for x in idio_acc],
    "idio_vols": [float(x) for x in idio_vol],
}
with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(results, fh, ensure_ascii=False, indent=2, sort_keys=True)
print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    pass
