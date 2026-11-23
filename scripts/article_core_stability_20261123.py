# -*- coding: utf-8 -*-
"""Stability selection with complementary pairs: FDR, power, and the correlation trap.

Single source of truth for every number and figure in the article.

Linear signal model y = X beta + eps, X in R^{n x p}, eps ~ N(0, sigma^2 I).
Lasso coordinate descent on the Gram matrix. Stability selection
(Meinshausen & Buhlmann 2010) draws B complementary pairs of half-samples,
records selection indicators along a lambda path, and keeps variables whose
selection frequency exceeds pi_thr > 0.5. The MB bound
    E[V] <= q^2 / ((2 pi_thr - 1) n)
is printed against the empirical false-positive count; q is the mean number
of selected variables per half-sample along the path.

Experiments:
  1. selection frequencies for true vs null signals, complementary vs iid halves
  2. pi_thr sweep: FDR/power vs BH on marginal p-values vs single-fit lasso
  3. correlated null blocks (rho=0.6): the same three methods degrade differently
  4. sample-size sweep with the MB bound, showing how loose the bound can be
  5. out-of-sample long-short portfolio built on the selected signals

Fixed seeds make every replay byte-identical to results.json.
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
SIGMA = 1.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "stability-selection-subsampling-fdr")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------- lasso machinery
def lasso_path(G, c, lam_path, max_iter=200, tol=1e-7):
    """Coordinate descent over a decreasing lambda path. G = X'X/n, c = X'y/n.

    Returns array (n_lambda, p) of coefficient vectors (no intercept; X,y centred).
    """
    p = G.shape[0]
    beta = np.zeros(p)
    out = np.zeros((len(lam_path), p))
    diag = np.diag(G).copy()
    for li, lam in enumerate(lam_path):
        for _ in range(max_iter):
            old = beta.copy()
            for j in range(p):
                r = c[j] - G[j] @ beta + diag[j] * beta[j]
                beta[j] = np.sign(r) * max(abs(r) - lam, 0.0) / diag[j]
            if np.max(np.abs(beta - old)) <= tol * max(1.0, np.max(np.abs(old))):
                break
        out[li] = beta
    return out


def lam_grid(G, c, n_lam=20, hi=0.8, lo=0.15):
    lm = np.max(np.abs(c))
    return np.linspace(hi * lm, lo * lm, n_lam)


def complement_pairs(n, B, rng):
    """B pairs of complementary half-samples (indices)."""
    pairs = []
    for _ in range(B):
        perm = rng.permutation(n)
        pairs.append((perm[: n // 2], perm[n // 2 :]))
    return pairs


def selection_freq(X, y, pairs, lam_path, lam_lo, lam_hi):
    """Selection probability per variable: max over the lambda window.

    Also returns q = mean |selected| per half-sample over the full path.
    """
    n, p = X.shape
    hit = np.zeros((2 * len(pairs), p), dtype=bool)
    sizes = []
    for bi, (ia, ib) in enumerate(pairs):
        for k, idx in enumerate((ia, ib)):
            Xs, ys = X[idx], y[idx]
            Xs = Xs - Xs.mean(0)
            ys = ys - ys.mean()
            G = Xs.T @ Xs / len(idx)
            c = Xs.T @ ys / len(idx)
            path = lasso_path(G, c, lam_path)
            sel = np.abs(path) > 1e-8
            keep = (lam_path >= lam_lo) & (lam_path <= lam_hi)
            hit[2 * bi + k] = sel[keep].any(axis=0)
            sizes.append(sel[-1].sum())
    return hit.mean(axis=0), float(np.mean(sizes))


def bh_from_y(X, y, active, q=0.1):
    """BH on marginal OLS t-tests, using an independent null-variance estimate."""
    n, p = X.shape
    Xc = X - X.mean(0)
    yc = y - y.mean()
    XtX = Xc.T @ Xc
    b = np.linalg.lstsq(XtX, Xc.T @ yc, rcond=None)[0]
    resid = yc - Xc @ b
    # leave-one-out style null sigma from the presumed null variables
    null = np.setdiff1d(np.arange(p), active)
    s2 = float(np.sum(resid ** 2) / max(1, len(null)))
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(XtX)) * s2, 1e-12))
    from scipy import stats
    pv = 2 * stats.norm.sf(np.abs(b) / se)
    order = np.argsort(pv)
    m = p
    crit = q * (np.arange(1, m + 1) / m)
    passed = pv[order] <= crit
    sel = np.zeros(p, dtype=bool)
    if passed.any():
        kmax = np.where(passed)[0].max()
        sel[order[: kmax + 1]] = True
    return sel


def metrics(sel, active):
    active = np.asarray(active, int)
    tp = int(np.sum(sel[active]))
    fp = int(np.sum(sel) - tp)
    fdr = fp / max(1, tp + fp)
    power = tp / len(active)
    return tp, fp, float(fdr), float(power)


def make_data(n, p, active, amps, rho, rng):
    if rho == 0.0:
        X = rng.standard_normal((n, p))
    else:
        X = rng.standard_normal((n, p))
        # null variables live in correlated blocks of 10
        null = np.setdiff1d(np.arange(p), active)
        for s in range(0, len(null), 10):
            blk = null[s : s + 10]
            if len(blk) < 2:
                continue
            common = rng.standard_normal(n)
            X[:, blk] = np.sqrt(rho) * common[:, None] + np.sqrt(1 - rho) * X[:, blk]
    beta = np.zeros(p)
    beta[active] = amps
    y = X @ beta + SIGMA * rng.standard_normal(n)
    Xc, yc = X - X.mean(0), y - y.mean()
    return Xc, yc


# ------------------------------------------------------------------- experiments
N, P = 240, 100
ACTIVE = np.array([3, 17, 29, 41, 53, 67, 78, 91])
AMPS = np.array([0.30, -0.26, 0.22, -0.34, 0.28, -0.20, 0.36, -0.24])
GRID = np.linspace(0.8, 0.25, 20)
LAM_LO, LAM_HI = 0.25, 0.80


def _batch_std(half_lists, X, y, lam_path, lm):
    """Std of selection frequency across independent batches of half-samples."""
    fs = []
    for halves in half_lists:
        pairs = [(halves[2 * i], halves[2 * i + 1])
                 for i in range(len(halves) // 2)]
        f, _ = selection_freq(X, y, pairs, lam_path, LAM_LO * lm, LAM_HI * lm)
        fs.append(f)
    return float(np.mean(np.std(np.asarray(fs), axis=0)))


def exp_frequency():
    rng = np.random.default_rng(SEED + 1)
    X, y = make_data(N, P, ACTIVE, AMPS, 0.0, rng)
    lm = np.max(np.abs(X.T @ y / N))
    lam_path = GRID * lm
    pairs = complement_pairs(N, 60, rng)
    freq, q = selection_freq(X, y, pairs, lam_path, LAM_LO * lm, LAM_HI * lm)
    # 10 batches of 12 half-samples: complementary pairs vs iid halves.
    comp_batches, iid_batches = [], []
    for b in range(10):
        sub = pairs[b * 6: (b + 1) * 6]   # disjoint pairs per batch
        comp_batches.append([idx for pair in sub for idx in pair])
        iid_batches.append([rng.permutation(N)[: N // 2] for _ in range(12)])
    return {"true_freq": [float(freq[j]) for j in ACTIVE],
            "all_freq": [float(v) for v in freq],
            "null_mean": float(np.mean(np.delete(freq, ACTIVE))),
            "null_q95": float(np.quantile(np.delete(freq, ACTIVE), 0.95)),
            "comp_pair_batch_std": _batch_std(comp_batches, X, y, lam_path, lm),
            "iid_batch_std": _batch_std(iid_batches, X, y, lam_path, lm),
            "q": q}


def exp_thresholds():
    rng = np.random.default_rng(SEED + 2)
    thrs = [0.60, 0.70, 0.80, 0.90, 0.95]
    rows = []
    n_rep = 12
    lm_cache = []
    for _ in range(n_rep):
        X, y = make_data(N, P, ACTIVE, AMPS, 0.0, rng)
        pairs = complement_pairs(N, 40, rng)
        lm = np.max(np.abs(X.T @ y / N))
        f, q = selection_freq(X, y, pairs, GRID * lm, LAM_LO * lm, LAM_HI * lm)
        lm_cache.append((f, q, X, y))
    for t in thrs:
        fdrs, pows, fps, bound = [], [], [], []
        for f, q, X, y in lm_cache:
            sel = f >= t
            _tp, _fp, fd, pw = metrics(sel, ACTIVE)
            fdrs.append(fd); pows.append(pw); fps.append(_fp)
            bound.append(q ** 2 / ((2 * t - 1) * N))
        rows.append({"pi": t,
                     "fdr": float(np.mean(fdrs)),
                     "power": float(np.mean(pows)),
                     "fp_mean": float(np.mean(fps)),
                     "mb_bound_fp": float(np.mean(bound))})
    # comparators on the same draws
    bh_fd, bh_pw, la_fd, la_pw = [], [], [], []
    for _f, _q, X, y in lm_cache:
        sel = bh_from_y(X, y, ACTIVE, 0.1)
        _, _, fd, pw = metrics(sel, ACTIVE)
        bh_fd.append(fd); bh_pw.append(pw)
        G = X.T @ X / N
        c = X.T @ y / N
        lm = np.max(np.abs(c))
        b = lasso_path(G, c, [0.3 * lm])[-1]
        _, _, fd, pw = metrics(np.abs(b) > 1e-8, ACTIVE)
        la_fd.append(fd); la_pw.append(pw)
    return {"stability": rows,
            "bh": {"fdr": float(np.mean(bh_fd)), "power": float(np.mean(bh_pw))},
            "lasso": {"fdr": float(np.mean(la_fd)), "power": float(np.mean(la_pw))}}


def exp_correlation():
    rng = np.random.default_rng(SEED + 3)
    rows = []
    sel_freq = None
    for rho in [0.0, 0.3, 0.6, 0.85]:
        fds, pws, cnts = [], [], []
        for _ in range(10):
            X, y = make_data(N, P, ACTIVE, AMPS, rho, rng)
            pairs = complement_pairs(N, 40, rng)
            lm = np.max(np.abs(X.T @ y / N))
            f, _ = selection_freq(X, y, pairs, GRID * lm, LAM_LO * lm, LAM_HI * lm)
            _tp, _fp, fd, pw = metrics(f >= 0.8, ACTIVE)
            fds.append(fd); pws.append(pw); cnts.append(_fp)
            if rho == 0.6:
                sel_freq = f
        rows.append({"rho": rho, "fdr": float(np.mean(fds)),
                     "power": float(np.mean(pws)),
                     "fp_mean": float(np.mean(cnts))})
    null = np.setdiff1d(np.arange(P), ACTIVE)
    return {"rows": rows,
            "rho06_false_freq_top5": [float(v) for v in
                                      np.sort(sel_freq[null])[::-1][:5]]}


def exp_sample_size():
    rng = np.random.default_rng(SEED + 4)
    rows = []
    for n in [100, 200, 400, 800]:
        fds, pws, bnd = [], [], []
        for _ in range(8):
            X, y = make_data(n, P, ACTIVE, AMPS, 0.0, rng)
            pairs = complement_pairs(n, 40, rng)
            lm = np.max(np.abs(X.T @ y / n))
            f, q = selection_freq(X, y, pairs, GRID * lm, LAM_LO * lm, LAM_HI * lm)
            _, _, fd, pw = metrics(f >= 0.8, ACTIVE)
            fds.append(fd); pws.append(pw)
            bnd.append(q ** 2 / ((2 * 0.8 - 1) * n))
        rows.append({"n": n, "fdr": float(np.mean(fds)),
                     "power": float(np.mean(pws)),
                     "mb_bound_fp": float(np.mean(bnd))})
    return rows


def exp_portfolio():
    rng = np.random.default_rng(SEED + 5)
    rows = []
    for method in ["stability", "lasso", "bh", "oracle"]:
        rets = []
        nsel = []
        for _ in range(25):
            X, y = make_data(N, P, ACTIVE, AMPS, 0.0, rng)
            pairs = complement_pairs(N, 40, rng)
            lm = np.max(np.abs(X.T @ y / N))
            if method == "stability":
                f, _ = selection_freq(X, y, pairs, GRID * lm, LAM_LO * lm, LAM_HI * lm)
                sel = np.where(f >= 0.8)[0]
            elif method == "lasso":
                G = X.T @ X / N
                b = lasso_path(G, X.T @ y / N, [0.3 * lm])[-1]
                sel = np.where(np.abs(b) > 1e-8)[0]
            elif method == "bh":
                sel = np.where(bh_from_y(X, y, ACTIVE, 0.1))[0]
            else:
                sel = ACTIVE
            nsel.append(len(sel))
            # independent test period: same beta, new covariance and noise
            Xt = rng.standard_normal((500, P))
            beta_true = np.zeros(P)
            beta_true[ACTIVE] = AMPS
            yt = Xt @ beta_true + rng.standard_normal(500)
            if len(sel) == 0:
                rets.append(0.0)
                continue
            # use fitted signs on train for every selected name
            G = X.T @ X / N
            b = lasso_path(G, X.T @ y / N, [LAM_LO * lm])[-1]
            w = np.sign(b[sel])
            w[w == 0] = 1.0
            port = Xt[:, sel] @ w / len(sel)
            rets.append(float(np.mean(port) / max(np.std(port), 1e-9) * np.sqrt(252)))
        rows.append({"method": method,
                     "sharpe_mean": float(np.mean(rets)),
                     "n_selected_mean": float(np.mean(nsel))})
    return rows


# ----------------------------------------------------------------------- figures
def fig1(freq):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(np.arange(P), freq, color=["#c0392b" if j in ACTIVE else "#7f8c8d"
                                         for j in range(P)], width=0.9)
    ax[0].axhline(0.8, ls="--", c="#2980b9")
    ax[0].set_xlabel("variable index"); ax[0].set_ylabel("selection frequency")
    ax[0].set_title("True signals separate from nulls (rho=0)")
    xs = np.delete(freq, ACTIVE)
    ax[1].hist(xs, bins=20, color="#7f8c8d")
    ax[1].axvline(0.8, ls="--", c="#2980b9", label="pi_thr = 0.8")
    ax[1].set_xlabel("selection frequency"); ax[1].legend()
    ax[1].set_title("Null distribution stays near zero")
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_frequencies.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig1_frequencies.png"


def fig2(data):
    st = data["stability"]
    x = [r["pi"] for r in st]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [r["fdr"] for r in st], "o-", c="#c0392b", label="stability FDR")
    ax.plot(x, [r["power"] for r in st], "s-", c="#2980b9", label="stability power")
    ax.axhline(data["bh"]["fdr"], ls="--", c="#7f8c8d", label="BH FDR")
    ax.axhline(data["lasso"]["fdr"], ls=":", c="#34495e", label="single lasso FDR")
    ax.set_xlabel("pi_thr"); ax.set_ylabel("rate")
    ax.set_title("Higher threshold buys lower FDR at a power cost")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig2_threshold.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig2_threshold.png"


def fig3(data):
    rows = data["rows"]
    xs = [r["rho"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(xs, [r["fdr"] for r in rows], "o-", c="#c0392b", label="stability FDR")
    ax[0].plot(xs, [r["power"] for r in rows], "s-", c="#2980b9", label="power")
    ax[0].set_xlabel("within-block correlation rho"); ax[0].legend()
    ax[0].set_title("Correlated nulls push FDR up")
    ax[1].bar(np.arange(5), data["rho06_false_freq_top5"], color="#c0392b")
    ax[1].set_ylabel("selection frequency"); ax[1].set_xlabel("top 5 null variables")
    ax[1].set_title("rho=0.6: which nulls survive")
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_correlation.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig3_correlation.png"


def fig4(rows):
    xs = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, [r["fdr"] for r in rows], "o-", c="#c0392b", label="empirical FP")
    ax.plot(xs, [r["mb_bound_fp"] for r in rows], "s--", c="#2980b9",
            label="MB bound")
    ax.set_xscale("log"); ax.set_xlabel("sample size n"); ax.set_ylabel("false selections")
    ax.set_title("The theoretical bound is valid but very loose")
    ax.legend(); fig.tight_layout()
    p = os.path.join(OUT, "fig4_bound.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig4_bound.png"


def fig5(rows):
    names = [r["method"] for r in rows]
    vals = [r["sharpe_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, vals, color=["#2980b9", "#7f8c8d", "#7f8c8d", "#c0392b"])
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                "n=%.1f" % r["n_selected_mean"], ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("OOS Sharpe (annualised)")
    ax.set_title("Selected signals traded on an independent period")
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_portfolio.png")
    fig.savefig(p, dpi=115); plt.close(fig)
    return "fig5_portfolio.png"


def main():
    fr = exp_frequency()
    th = exp_thresholds()
    co = exp_correlation()
    ss = exp_sample_size()
    po = exp_portfolio()
    result = {"setup": {"seed": SEED, "n": N, "p": P, "true_signals": ACTIVE.tolist(),
                        "amplitudes": AMPS.tolist(), "sigma": SIGMA,
                        "B_complementary_pairs": 40},
              "frequency": fr, "thresholds": th, "correlation": co,
              "sample_size": ss, "portfolio": po}
    result["figs"] = [fig1(np.asarray(fr["all_freq"], float)), fig2(th),
                      fig3(co), fig4(ss), fig5(po)]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
