"""Second-order stochastic dominance portfolio: in-sample vs out-of-sample.

This file IS the runnable code block embedded in the article. Run it directly:

    python3 article_core_ssd_portfolio_20261110.py

It writes five real PNG figures and results.json into ARTICLE_OUT (default: cwd).
The LP is solved with scipy.optimize.linprog (HiGHS), not faked or hardcoded.
All numbers quoted in the article come from results.json.

Dependencies: numpy, scipy, matplotlib.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

OUT = Path(os.environ.get("ARTICLE_OUT", ".")).resolve()
SEED = 20261110
ANN = 252.0
N_ASSETS = 30
POOL = 6000
K_CONSTRAINT = 20
K_EVAL = 200


# ---------- synthetic return panel ----------
def build_panel():
    """Risk and expected return are positively linked; crashes are rare (1.5%).

    This is the setting where maximising the *sample* mean under sample shortfall
    constraints is genuinely dangerous: a small training window can contain zero
    or one crash, so the LP happily loads on high-beta high-alpha names whose
    left tail only shows up out of sample.
    """
    rng = np.random.default_rng([SEED, 0, 0])
    beta = rng.uniform(0.35, 1.65, size=N_ASSETS)
    beta[:6] = rng.uniform(-0.45, 0.10, size=6)          # defensive names
    alpha = 0.00008 + 0.00028 * beta + rng.normal(0.0, 0.00015, size=N_ASSETS)
    idio_vol = rng.uniform(0.004, 0.010, size=N_ASSETS)
    S = POOL
    crisis = rng.random(S) < 0.015
    f = 0.0006 + 0.008 * rng.standard_normal(S) - crisis * (0.050 + 0.015 * rng.standard_normal(S))
    noise = rng.standard_normal((S, N_ASSETS)) * idio_vol[None, :]
    R = alpha[None, :] + f[:, None] * beta[None, :] + noise
    return R, beta, alpha, idio_vol


# ---------- SSD linear program ----------
def ssd_lp(R_train, b, eps=0.0, K=K_CONSTRAINT):
    """max mu'w  s.t. w in simplex and E[(t_k - r_w)_+] <= E[(t_k - r_b)_+] + eps.

    Epigraph variables z[s,k] >= t_k - r_w[s] linearise the shortfall.
    w = b is always feasible, so the LP is never infeasible for eps >= 0.
    """
    S, N = R_train.shape
    rb = R_train @ b
    mu = R_train.mean(axis=0)
    t = np.quantile(rb, np.linspace(0.02, 0.98, K))
    cap = np.maximum(t[None, :] - rb[:, None], 0.0).sum(axis=0) + eps * S

    M = S * K
    nv = N + M
    s_idx = np.repeat(np.arange(S), K)
    k_idx = np.tile(np.arange(K), S)
    row_base = np.arange(M)
    var_z = N + s_idx * K + k_idx
    col_w = np.broadcast_to(np.arange(N), (M, N)).ravel()
    rows = np.concatenate([row_base, np.repeat(row_base, N),
                           M + np.repeat(np.arange(K), S)])
    cols = np.concatenate([var_z, col_w,
                           (N + np.arange(S)[None, :] * K + np.arange(K)[:, None]).ravel()])
    data = np.concatenate([-np.ones(M), -R_train[s_idx].ravel(),
                           np.ones(M)])
    A_ub = coo_matrix((data, (rows, cols)), shape=(M + K, nv)).tocsr()
    b_ub = np.concatenate([-t[k_idx], cap])

    A_eq = np.zeros((1, nv))
    A_eq[0, :N] = 1.0
    c = np.zeros(nv)
    c[:N] = -mu
    bounds = [(0.0, 1.0)] * N + [(0.0, None)] * M
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x[:N], t


def simplex_opt(R_train, kind, gamma=5.0):
    """Long-only min-variance / mean-variance via SLSQP, same simplex as the LP."""
    S, N = R_train.shape
    mu = R_train.mean(axis=0)
    cov = np.cov(R_train, rowvar=False) + 1e-9 * np.eye(N)
    if kind == "minvar":
        fun = lambda w: float(w @ cov @ w)
    else:
        fun = lambda w: float(-mu @ w + gamma * w @ cov @ w)
    cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    x0 = np.ones(N) / N
    r = minimize(fun, x0, method="SLSQP", bounds=[(0.0, 1.0)] * N,
                 constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    w = np.clip(r.x, 0.0, None)
    return w / w.sum()


# ---------- evaluation ----------
def shortfall_curve(r, thresholds):
    return np.maximum(thresholds[None, :] - r[:, None], 0.0).mean(axis=0)


def ssd_violation(rw, rb, thresholds):
    """Positive value means the portfolio FAILS to dominate the benchmark."""
    return float(np.max(shortfall_curve(rw, thresholds) - shortfall_curve(rb, thresholds)))


def perf(r):
    sr = float(r.mean() / r.std(ddof=1) * np.sqrt(ANN))
    ordered = np.sort(r)
    k = max(1, int(np.ceil(0.05 * len(r))))
    cvar = float(-ordered[:k].mean())          # positive number = 5% tail loss
    equity = np.concatenate([[0.0], np.cumsum(r)])
    mdd = float(-np.min(equity - np.maximum.accumulate(equity)))
    return {"sharpe": round(sr, 4), "cvar5": round(cvar, 6),
            "max_dd": round(mdd, 6), "ann_mean": round(float(r.mean() * ANN), 6)}


def evaluate_round(R, S_train, S_test, round_id, eps=0.0, K=K_CONSTRAINT):
    rng = np.random.default_rng([SEED, 1, round_id])
    pick = rng.choice(len(R), size=S_train + S_test, replace=False)
    tr, te = R[pick[:S_train]], R[pick[S_train:]]
    b = np.ones(N_ASSETS) / N_ASSETS
    w_ssd, t_constraint = ssd_lp(tr, b, eps=eps, K=K)
    w_minvar = simplex_opt(tr, "minvar")
    w_mv = simplex_opt(tr, "meanvar")
    weights = {"benchmark": b, "ssd": w_ssd, "minvar": w_minvar, "meanvar": w_mv}

    rb_tr, rb_te = tr @ b, te @ b
    # Fixed diagnostic grid from the WHOLE scenario pool (the "population"),
    # never from the training window. Otherwise a training sample with no crash
    # would produce a truncated grid and hide the out-of-sample left tail.
    rb_pool = R @ b
    eval_t = np.quantile(rb_pool, np.linspace(0.001, 0.999, K_EVAL))
    out = {"weights": {k: [round(float(v), 6) for v in w] for k, w in weights.items()},
           "in_sample_violation": {}, "oos_violation": {}, "perf": {},
           "train_mean": {}, "l1_to_benchmark": {}, "n_active": {},
           "shortfall_train": {}, "shortfall_test": {}, "eval_t": [round(float(v), 8) for v in eval_t],
           "in_sample_violation_on_constraint_grid": round(
               ssd_violation(tr @ w_ssd, rb_tr, t_constraint), 12),
           "train_benchmark_min": round(float(rb_tr.min()), 8),
           "pool_benchmark_min": round(float(rb_pool.min()), 8),
           "train_crash_days": int((rb_tr < -0.03).sum()),
           "test_crash_days": int((rb_te < -0.03).sum())}
    for name, w in weights.items():
        rw_tr, rw_te = tr @ w, te @ w
        out["in_sample_violation"][name] = round(ssd_violation(rw_tr, rb_tr, eval_t), 8)
        out["oos_violation"][name] = round(ssd_violation(rw_te, rb_te, eval_t), 8)
        out["perf"][name] = perf(rw_te)
        out["train_mean"][name] = round(float(rw_tr.mean() * ANN), 6)
        out["l1_to_benchmark"][name] = round(float(np.abs(w - b).sum()), 4)
        out["n_active"][name] = int((w > 1e-3).sum())
        out["shortfall_train"][name] = [round(float(v), 8) for v in shortfall_curve(rw_tr, eval_t)]
        out["shortfall_test"][name] = [round(float(v), 8) for v in shortfall_curve(rw_te, eval_t)]
    out["shortfall_train"]["benchmark"] = [round(float(v), 8) for v in shortfall_curve(rb_tr, eval_t)]
    out["shortfall_test"]["benchmark"] = [round(float(v), 8) for v in shortfall_curve(rb_te, eval_t)]
    out["constraint_thresholds"] = [round(float(v), 8) for v in t_constraint]
    return out


def summarise(rounds):
    keys = ["benchmark", "ssd", "minvar", "meanvar"]
    out = {}
    for name in keys:
        oos = [r["oos_violation"][name] for r in rounds]
        inv = [r["in_sample_violation"][name] for r in rounds]
        out[name] = {
            "oos_violation_mean": round(float(np.mean(oos)), 8),
            "oos_violation_median": round(float(np.median(oos)), 8),
            "oos_violation_positive_frac": round(float(np.mean(np.array(oos) > 1e-8)), 4),
            "in_sample_violation_mean": round(float(np.mean(inv)), 8),
            "sharpe_mean": round(float(np.mean([r["perf"][name]["sharpe"] for r in rounds])), 4),
            "cvar5_mean": round(float(np.mean([r["perf"][name]["cvar5"] for r in rounds])), 6),
            "max_dd_mean": round(float(np.mean([r["perf"][name]["max_dd"] for r in rounds])), 6),
            "ann_mean_mean": round(float(np.mean([r["perf"][name]["ann_mean"] for r in rounds])), 6),
            "train_mean_mean": round(float(np.mean([r["train_mean"][name] for r in rounds])), 6),
            "l1_to_benchmark_mean": round(float(np.mean([r["l1_to_benchmark"][name] for r in rounds])), 4),
            "n_active_mean": round(float(np.mean([r["n_active"][name] for r in rounds])), 2),
        }
    return out


def run_all(verbose=True):
    R, beta, alpha, idio_vol = build_panel()
    stats = {"meta": {
        "seed": SEED, "n_assets": N_ASSETS, "scenario_pool": POOL,
        "K_constraint": K_CONSTRAINT, "K_eval": K_EVAL,
        "main": {"rounds": 60, "S_train": 240, "S_test": 1500},
        "size_scan": {"rounds": 25, "S_train_values": [60, 120, 240, 480, 960]},
        "eps_scan": {"rounds": 25, "eps_values": [0.0, 0.0005, 0.001, 0.002, 0.004]},
        "beta_first6": [round(float(v), 4) for v in beta[:6]],
    }}

    main_rounds = []
    for i in range(60):
        main_rounds.append(evaluate_round(R, 240, 1500, i))
    stats["main"] = summarise(main_rounds)
    stats["representative"] = main_rounds[0]
    stats["main_diagnostics"] = {
        "in_sample_violation_on_constraint_grid_mean": round(float(np.mean(
            [r["in_sample_violation_on_constraint_grid"] for r in main_rounds])), 12),
        "eval_grid_in_sample_violation_mean": round(float(np.mean(
            [r["in_sample_violation"]["ssd"] for r in main_rounds])), 10),
        "train_crash_days_mean": round(float(np.mean([r["train_crash_days"] for r in main_rounds])), 2),
        "test_crash_days_mean": round(float(np.mean([r["test_crash_days"] for r in main_rounds])), 2),
        "pool_benchmark_min": main_rounds[0]["pool_benchmark_min"],
        "train_benchmark_min_mean": round(float(np.mean([r["train_benchmark_min"] for r in main_rounds])), 6),
    }
    if verbose:
        print("[1/3] main 60 rounds done; diagnostics:", stats["main_diagnostics"])
        for k, v in stats["main"].items():
            print("   ", k, "oos viol", v["oos_violation_mean"],
                  "frac>0", v["oos_violation_positive_frac"],
                  "sharpe", v["sharpe_mean"], "cvar", v["cvar5_mean"])

    size_scan = {}
    for S in [60, 120, 240, 480, 960]:
        rr = [evaluate_round(R, S, 1500, 1000 + S * 10 + j) for j in range(25)]
        s = summarise(rr)
        size_scan[str(S)] = {
            "ssd_oos_violation": s["ssd"]["oos_violation_mean"],
            "ssd_oos_positive_frac": s["ssd"]["oos_violation_positive_frac"],
            "ssd_train_mean": s["ssd"]["train_mean_mean"],
            "benchmark_train_mean": s["benchmark"]["train_mean_mean"],
            "ssd_sharpe": s["ssd"]["sharpe_mean"],
            "benchmark_sharpe": s["benchmark"]["sharpe_mean"],
            "l1_to_benchmark": s["ssd"]["l1_to_benchmark_mean"],
        }
    stats["size_scan"] = size_scan
    if verbose:
        print("[2/3] training-size scan done:", size_scan)

    eps_scan = {}
    for j, eps in enumerate([0.0, 0.0005, 0.001, 0.002, 0.004]):
        rr = [evaluate_round(R, 240, 1500, 5000 + j * 100 + kk, eps=eps) for kk in range(25)]
        s = summarise(rr)
        eps_scan[f"{eps:.4f}"] = {
            "ssd_in_sample_violation": s["ssd"]["in_sample_violation_mean"],
            "ssd_oos_violation": s["ssd"]["oos_violation_mean"],
            "ssd_train_mean": s["ssd"]["train_mean_mean"],
            "ssd_ann_mean_oos": s["ssd"]["ann_mean_mean"],
            "l1_to_benchmark": s["ssd"]["l1_to_benchmark_mean"],
        }
    stats["eps_scan"] = eps_scan
    if verbose:
        print("[3/3] epsilon scan done:", eps_scan)

    figs = {"main": stats["main"], "rep": main_rounds[0],
            "size_scan": size_scan, "eps_scan": eps_scan,
            "oos_violations": [r["oos_violation"]["ssd"] for r in main_rounds],
            "all_rounds": main_rounds,
            "beta": beta.tolist()}
    return stats, figs


# ---------- figures ----------
def make_figures(figs):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
    rep = figs["rep"]
    t = np.array(rep["eval_t"])
    names = ["benchmark", "ssd", "minvar", "meanvar"]
    colors = {"benchmark": "#6b7280", "ssd": "#d62728",
              "minvar": "#1f77b4", "meanvar": "#f59e0b"}

    # Fig 1: absolute shortfall curves (left) and their differences (right).
    # The violation is ~1e-4, invisible on the absolute scale, so the
    # difference panels with a zero line are the honest way to show it.
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
    for row, (key, tag) in enumerate([("shortfall_train", "In sample"),
                                      ("shortfall_test", "Out of sample")]):
        bench = np.array(rep[key]["benchmark"])
        a_abs, a_dif = ax[row, 0], ax[row, 1]
        a_abs.plot(t, bench, color=colors["benchmark"], lw=2.2, label="equal-weight benchmark")
        for nm in ["ssd", "minvar", "meanvar"]:
            d = np.array(rep[key][nm]) - bench
            a_abs.plot(t, np.array(rep[key][nm]), color=colors[nm], lw=1.6, label=nm)
            a_dif.plot(t, d, color=colors[nm], lw=1.7, label=nm)
        a_dif.axhline(0, color="#111827", lw=1.4, ls="--", label="dominance boundary")
        a_dif.fill_between(t, 0, np.array(rep[key]["ssd"]) - bench,
                           where=(np.array(rep[key]["ssd"]) - bench) > 0,
                           color=colors["ssd"], alpha=0.25)
        a_abs.set_ylabel("E[(t - r)+]  shortfall")
        a_dif.set_ylabel("shortfall  minus benchmark")
        a_abs.set_title(f"{tag}: absolute shortfall")
        a_dif.set_title(f"{tag}: violation becomes visible")
        a_abs.legend(frameon=False, fontsize=8)
        a_dif.legend(frameon=False, fontsize=8)
        viol = rep["in_sample_violation"]["ssd"] if row == 0 else rep["oos_violation"]["ssd"]
        a_dif.text(0.03, 0.90, f"max SSD violation = {viol:.2e}", transform=a_dif.transAxes,
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                         fc="#fef2f2" if row else "#ecfdf5",
                                         ec="#fecaca" if row else "#a7f3d0"))
    ax[1, 0].set_xlabel("threshold t (daily return)")
    ax[1, 1].set_xlabel("threshold t (daily return)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_shortfall_in_vs_out.png")
    plt.close(fig)

    # Fig 2: weights
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(N_ASSETS)
    w_ssd = np.array(rep["weights"]["ssd"])
    w_b = np.array(rep["weights"]["benchmark"])
    ax[0].bar(x - 0.2, w_b, width=0.4, color=colors["benchmark"], label="equal weight")
    ax[0].bar(x + 0.2, w_ssd, width=0.4, color=colors["ssd"], label="SSD LP")
    ax[0].set_xlabel("asset id")
    ax[0].set_ylabel("portfolio weight")
    ax[0].set_title(f"SSD weights: L1 distance to benchmark = {rep['l1_to_benchmark']['ssd']:.3f}")
    ax[0].legend(frameon=False, fontsize=8)
    lb = np.array(figs["beta"])
    ax[1].scatter(lb, w_ssd, s=32, color=colors["ssd"], edgecolor="white", zorder=3)
    for i in range(N_ASSETS):
        if w_ssd[i] > 0.02:
            ax[1].annotate(str(i), (lb[i], w_ssd[i]), fontsize=7,
                           xytext=(3, 3), textcoords="offset points")
    ax[1].axhline(1 / N_ASSETS, color=colors["benchmark"], ls="--", lw=1.2, label="equal weight")
    ax[1].set_xlabel("market beta (first 6 assets are defensive)")
    ax[1].set_ylabel("SSD LP weight")
    ax[1].set_title("The LP concentrates on a handful of names")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_ssd_weights.png")
    plt.close(fig)

    # Fig 3: distribution of OOS violations over 60 rounds
    v = np.array(figs["oos_violations"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].hist(v, bins=22, color="#d62728", alpha=0.8, edgecolor="white")
    ax[0].axvline(0, color="#111827", lw=1.5, ls="--", label="dominance boundary")
    ax[0].axvline(v.mean(), color="#1f77b4", lw=1.5, label=f"mean = {v.mean():.2e}")
    ax[0].set_xlabel("max OOS SSD violation (positive = dominance fails)")
    ax[0].set_ylabel("number of rounds")
    ax[0].set_title(f"60 rounds: {(v > 1e-8).mean() * 100:.1f}% of rounds break dominance")
    ax[0].legend(frameon=False, fontsize=8)
    main = figs["main"]
    ax[1].bar(range(4), [main[n]["oos_violation_mean"] for n in names],
              color=[colors[n] for n in names], edgecolor="white")
    ax[1].axhline(0, color="#111827", lw=1)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("mean max OOS violation")
    ax[1].set_title("SSD is not uniquely bad: fitted distributions all leak")
    for i, n in enumerate(names):
        ax[1].text(i, main[n]["oos_violation_mean"], f"{main[n]['oos_violation_mean']:.1e}",
                   ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_oos_violation_distribution.png")
    plt.close(fig)

    # Fig 4: OOS performance across rounds (boxplots)
    rounds = figs["all_rounds"]
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))
    metrics = [("sharpe", "Out-of-sample Sharpe (annualised)"),
               ("cvar5", "5% CVaR (tail loss, lower is better)"),
               ("max_dd", "Max cumulative drawdown (lower is better)"),
               ("ann_mean", "Out-of-sample annualised mean")]
    for a, (m, title) in zip(ax.ravel(), metrics):
        data = [[r["perf"][n][m] for r in rounds] for n in names]
        bp = a.boxplot(data, tick_labels=names, patch_artist=True, widths=0.6,
                       medianprops=dict(color="#111827", lw=1.4))
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(colors[n])
            patch.set_alpha(0.75)
        a.set_title(title, fontsize=10)
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_oos_performance_boxplots.png")
    plt.close(fig)

    # Fig 5: two sensitivity scans
    ss = figs["size_scan"]
    sizes = [60, 120, 240, 480, 960]
    es = figs["eps_scan"]
    eps_keys = [f"{e:.4f}" for e in [0.0, 0.0005, 0.001, 0.002, 0.004]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    axA = ax[0]
    axA.plot(sizes, [ss[str(s)]["ssd_oos_violation"] for s in sizes], "o-",
             color="#d62728", label="OOS violation (left)")
    axA.set_xscale("log", base=2)
    axA.set_xticks(sizes)
    axA.set_xticklabels([str(s) for s in sizes])
    axA.set_xlabel("training scenarios S")
    axA.set_ylabel("mean max OOS SSD violation", color="#d62728")
    axA2 = axA.twinx()
    axA2.plot(sizes, [ss[str(s)]["ssd_train_mean"] - ss[str(s)]["benchmark_train_mean"] for s in sizes],
              "s--", color="#1f77b4", label="in-sample mean edge (right)")
    axA2.set_ylabel("in-sample annualised mean edge", color="#1f77b4")
    axA2.grid(False)
    axA.set_title("More training data shrinks both the leak and the edge")
    lines = axA.get_lines() + axA2.get_lines()
    axA.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc="upper right")
    ax[1].plot(range(len(eps_keys)), [es[k]["ssd_oos_violation"] for k in eps_keys], "o-",
               color="#d62728", label="OOS violation")
    ax[1].set_xticks(range(len(eps_keys)))
    ax[1].set_xticklabels(eps_keys)
    ax[1].set_xlabel("slack epsilon in the shortfall constraint")
    ax[1].set_ylabel("mean max OOS SSD violation", color="#d62728")
    axB2 = ax[1].twinx()
    axB2.plot(range(len(eps_keys)), [es[k]["ssd_train_mean"] for k in eps_keys], "s--",
              color="#1f77b4", label="in-sample annualised mean")
    axB2.set_ylabel("in-sample annualised mean", color="#1f77b4")
    axB2.grid(False)
    ax[1].set_title("Slack buys violation, not a reliable return edge")
    lines2 = ax[1].get_lines() + axB2.get_lines()
    ax[1].legend(lines2, [l.get_label() for l in lines2], frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_sensitivity_scans.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats, figs = run_all()
    make_figures(figs)
    (OUT / "results.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nMain summary (60 independent train/test splits):")
    for k, v in stats["main"].items():
        print(f"  {k:10s} in-sample viol={v['in_sample_violation_mean']:.2e}  "
              f"OOS viol={v['oos_violation_mean']:.2e}  "
              f"frac>0={v['oos_violation_positive_frac']:.3f}  "
              f"Sharpe={v['sharpe_mean']:.3f}  CVaR5={v['cvar5_mean']:.5f}  "
              f"L1={v['l1_to_benchmark_mean']:.3f}")
    print("Wrote 5 PNGs + results.json to", OUT)


if __name__ == "__main__":
    main()
