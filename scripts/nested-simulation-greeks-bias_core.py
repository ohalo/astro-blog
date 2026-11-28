# -*- coding: utf-8 -*-
"""Nested simulation Greeks: inner-sample bias, variance trade-off and Heston cost."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

SEED = 20261128
S0 = 100.0
R = 0.03
SIG = 0.20
T = 1.0
T1 = 0.25
TAU = T - T1
K1 = 100.0
K2 = 6.0
N_OUT = 20000
M_GRID = [2, 5, 10, 20, 50, 100, 200, 500]
BUDGET = 400000
BUDGET_M = [5, 20, 50, 100, 200, 1000]
BUDGET_REPS = 10
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "public", "images", "nested-simulation-greeks-bias")

V0 = 0.04
KAPPA = 1.5
THETA = 0.04
XI = 0.60
RHO = -0.50
HN_OUT = 2000
H_M_MAX = 1000
H_INNER_STEPS = 4
H_OUTER_STEPS = 6


def bs_call(s, k, tt):
    d1 = (np.log(s / k) + (R + 0.5 * SIG ** 2) * tt) / (SIG * np.sqrt(tt))
    d2 = d1 - SIG * np.sqrt(tt)
    return s * norm.cdf(d1) - k * np.exp(-R * tt) * norm.cdf(d2)


def geske_price(s):
    rho = np.sqrt(T1 / T)
    cov = [[1.0, rho], [rho, 1.0]]
    a2 = (np.log(s / S_STAR) + (R - 0.5 * SIG ** 2) * T1) / (SIG * np.sqrt(T1))
    a1 = a2 + SIG * np.sqrt(T1)
    b1 = (np.log(s / K1) + (R + 0.5 * SIG ** 2) * T) / (SIG * np.sqrt(T))
    b2 = b1 - SIG * np.sqrt(T)
    p = multivariate_normal.cdf([a1, b1], cov=cov)
    q = multivariate_normal.cdf([a2, b2], cov=cov)
    return s * p - K1 * np.exp(-R * T) * q - K2 * np.exp(-R * T1) * norm.cdf(a2)


def bs_nested(rng, n_out, m, want_exact=False):
    z1 = rng.standard_normal(n_out)
    s1 = S0 * np.exp((R - 0.5 * SIG ** 2) * T1 + SIG * np.sqrt(T1) * z1)
    zi = rng.standard_normal((n_out, m))
    st = s1[:, None] * np.exp((R - 0.5 * SIG ** 2) * TAU + SIG * np.sqrt(TAU) * zi)
    inner_pay = np.maximum(st - K1, 0.0)
    inner_val = np.exp(-R * TAU) * inner_pay.mean(axis=1)
    inner_dlt = np.exp(-R * TAU) * ((st > K1).astype(float) * st / S0).mean(axis=1)
    take = (inner_val > K2).astype(float)
    price = np.exp(-R * T1) * np.maximum(inner_val - K2, 0.0).mean()
    delta = np.exp(-R * T1) * (take * inner_dlt).mean()
    result = {
        "price": float(price),
        "delta": float(delta),
        "delta_se": float(np.exp(-R * T1) * np.std(take * inner_dlt, ddof=1) / np.sqrt(n_out)),
        "price_se": float(np.exp(-R * T1) * np.std(np.maximum(inner_val - K2, 0.0), ddof=1) / np.sqrt(n_out)),
    }
    if want_exact:
        exact_val = bs_call(s1, K1, TAU)
        exact_take = (exact_val > K2)
        mis = float(np.mean(take.astype(bool) != exact_take))
        result["misclassification"] = mis
        result["near_boundary_share"] = float(np.mean(np.abs(exact_val - K2) < 0.5))
    return result


def heston_nested(rng, n_out, m_grid, dt_out, dt_in, want_inner_paths=False):
    z2o = rng.standard_normal((n_out, H_OUTER_STEPS))
    zio = rng.standard_normal((n_out, H_OUTER_STEPS))
    z1o = RHO * z2o + np.sqrt(1.0 - RHO ** 2) * zio
    s = np.full(n_out, S0)
    v = np.full(n_out, V0)
    for k in range(H_OUTER_STEPS):
        vp = np.maximum(v, 0.0)
        s = s * np.exp((R - 0.5 * vp) * dt_out + np.sqrt(vp * dt_out) * z1o[:, k])
        v = v + KAPPA * (THETA - vp) * dt_out + XI * np.sqrt(vp) * np.sqrt(dt_out) * z2o[:, k]
    s1 = s.copy()
    v1 = np.maximum(v, 0.0)

    mmax = max(m_grid)
    z2i = rng.standard_normal((n_out, mmax, H_INNER_STEPS))
    zii = rng.standard_normal((n_out, mmax, H_INNER_STEPS))
    z1i = RHO * z2i + np.sqrt(1.0 - RHO ** 2) * zii
    si = np.broadcast_to(s1[:, None], (n_out, mmax)).copy()
    vi = np.broadcast_to(v1[:, None], (n_out, mmax)).copy()
    growth = np.ones((n_out, mmax))
    for k in range(H_INNER_STEPS):
        vp = np.maximum(vi, 0.0)
        factor = np.exp((R - 0.5 * vp) * dt_in + np.sqrt(vp * dt_in) * z1i[:, :, k])
        si = si * factor
        growth = growth * factor
        vi = vi + KAPPA * (THETA - vp) * dt_in + XI * np.sqrt(vp) * np.sqrt(dt_in) * z2i[:, :, k]
    pay = np.maximum(si - K1, 0.0)
    disc_in = np.exp(-R * TAU)
    counts = np.arange(1, mmax + 1, dtype=float)[None, :]
    inner_val = disc_in * np.cumsum(pay, axis=1) / counts
    inner_dlt = disc_in * np.cumsum((si > K1).astype(float) * growth * s1[:, None] / S0, axis=1) / counts
    ref_take = (inner_val[:, mmax - 1] > K2)
    out = {}
    for m in m_grid:
        iv = inner_val[:, m - 1]
        idl = inner_dlt[:, m - 1]
        tk = (iv > K2)
        out[str(m)] = {
            "price": float(np.exp(-R * T1) * np.maximum(iv - K2, 0.0).mean()),
            "delta": float(np.exp(-R * T1) * (tk.astype(float) * idl).mean()),
            "misclassification_vs_max": float(np.mean(tk != ref_take)),
        }
    return out


def main():
    global S_STAR
    S_STAR = brentq(lambda s: bs_call(s, K1, TAU) - K2, 1.0, 300.0)
    analytic_price = float(geske_price(S0))
    h = 1e-4
    analytic_delta = float((geske_price(S0 + h) - geske_price(S0 - h)) / (2.0 * h))

    rng = np.random.default_rng(SEED)
    z1 = rng.standard_normal(N_OUT)
    s1 = S0 * np.exp((R - 0.5 * SIG ** 2) * T1 + SIG * np.sqrt(T1) * z1)
    zi = rng.standard_normal((N_OUT, max(M_GRID)))
    st = s1[:, None] * np.exp((R - 0.5 * SIG ** 2) * TAU + SIG * np.sqrt(TAU) * zi)
    pay = np.maximum(st - K1, 0.0)
    counts = np.arange(1, max(M_GRID) + 1, dtype=float)[None, :]
    disc_in = np.exp(-R * TAU)
    inner_vals = disc_in * np.cumsum(pay, axis=1) / counts
    inner_dlts = disc_in * np.cumsum((st > K1).astype(float) * st / S0, axis=1) / counts
    exact_val = bs_call(s1, K1, TAU)
    exact_take = exact_val > K2
    study = {}
    for m in M_GRID:
        iv = inner_vals[:, m - 1]
        idl = inner_dlts[:, m - 1]
        tk = iv > K2
        delta_path = np.exp(-R * T1) * tk.astype(float) * idl
        study[str(m)] = {
            "price": float(np.exp(-R * T1) * np.maximum(iv - K2, 0.0).mean()),
            "delta": float(delta_path.mean()),
            "delta_se": float(np.std(delta_path, ddof=1) / np.sqrt(N_OUT)),
            "misclassification": float(np.mean(tk != exact_take)),
            "near_boundary_share": float(np.mean(np.abs(exact_val - K2) < 0.5)),
        }

    budget = {}
    for m in BUDGET_M:
        n_out = BUDGET // m
        ds = []
        ps = []
        for rep in range(BUDGET_REPS):
            res = bs_nested(rng, n_out, m)
            ds.append(res["delta"] - analytic_delta)
            ps.append(res["price"] - analytic_price)
        budget[str(m)] = {
            "n_outer": int(n_out),
            "delta_bias": round(float(np.mean(ds)), 6),
            "delta_rmse": round(float(np.sqrt(np.mean(np.square(ds)))), 6),
            "price_bias": round(float(np.mean(ps)), 6),
        }

    h_rng = np.random.default_rng(SEED + 31)
    heston = heston_nested(h_rng, HN_OUT, [5, 20, 50, 200, H_M_MAX],
                           T1 / H_OUTER_STEPS, TAU / H_INNER_STEPS)
    ref_delta = heston[str(H_M_MAX)]["delta"]

    # Figure 1: convergence of nested price and delta with inner sample size
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(M_GRID, [study[str(m)]["price"] - analytic_price for m in M_GRID], "o-", color="#c0392b")
    axes[0].axhline(0.0, color="#555555", ls="--")
    axes[0].set_xscale("log")
    axes[0].set_title("Nested price bias (BS compound call)")
    axes[0].set_xlabel("inner paths M")
    axes[0].set_ylabel("price minus analytic")
    axes[1].plot(M_GRID, [study[str(m)]["delta"] - analytic_delta for m in M_GRID], "o-", color="#3b5b8c")
    axes[1].axhline(0.0, color="#555555", ls="--")
    axes[1].set_xscale("log")
    axes[1].set_title("Nested delta bias (BS compound call)")
    axes[1].set_xlabel("inner paths M")
    axes[1].set_ylabel("delta minus analytic")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig1_bs_convergence.png"), dpi=150)
    plt.close(fig)

    # Figure 2: exercise-boundary misclassification drives the delta bias
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(M_GRID, [study[str(m)]["misclassification"] * 100.0 for m in M_GRID],
                 "o-", color="#8e6cbb")
    axes[0].set_xscale("log")
    axes[0].set_title("Wrong exercise decisions")
    axes[0].set_xlabel("inner paths M")
    axes[0].set_ylabel("misclassified outer paths (%)")
    axes[1].plot(M_GRID,
                 [study[str(m)]["delta"] / analytic_delta - 1.0 for m in M_GRID],
                 "o-", color="#e07b39", label="delta")
    axes[1].plot(M_GRID,
                 [study[str(m)]["price"] / analytic_price - 1.0 for m in M_GRID],
                 "s-", color="#3b5b8c", label="price")
    axes[1].axhline(0.0, color="#555555", ls="--")
    axes[1].set_xscale("log")
    axes[1].set_title("Relative bias")
    axes[1].set_xlabel("inner paths M")
    axes[1].set_ylabel("relative error")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig2_misclassification.png"), dpi=150)
    plt.close(fig)

    # Figure 3: fixed budget variance penalty
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(BUDGET_M, [budget[str(m)]["delta_bias"] for m in BUDGET_M], "o-", color="#c0392b")
    axes[0].axhline(0.0, color="#555555", ls="--")
    axes[0].set_xscale("log")
    axes[0].set_title("Bias at fixed inner-draw budget")
    axes[0].set_xlabel("inner paths M")
    axes[0].set_ylabel("delta bias")
    axes[1].plot(BUDGET_M, [budget[str(m)]["delta_rmse"] for m in BUDGET_M], "o-", color="#3b5b8c")
    axes[1].set_xscale("log")
    axes[1].set_title("RMSE at fixed inner-draw budget")
    axes[1].set_xlabel("inner paths M")
    axes[1].set_ylabel("delta RMSE")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig3_fixed_budget.png"), dpi=150)
    plt.close(fig)

    # Figure 4: Heston nested delta versus inner sample size
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    hm = [5, 20, 50, 200]
    axes[0].plot(hm, [heston[str(m)]["delta"] - ref_delta for m in hm], "o-", color="#c0392b")
    axes[0].axhline(0.0, color="#555555", ls="--")
    axes[0].set_xscale("log")
    axes[0].set_title("Heston delta bias versus M=1000 reference")
    axes[0].set_xlabel("inner paths M")
    axes[0].set_ylabel("delta difference")
    axes[1].plot(hm, [heston[str(m)]["misclassification_vs_max"] * 100.0 for m in hm],
                 "o-", color="#8e6cbb")
    axes[1].set_xscale("log")
    axes[1].set_title("Exercise decisions versus reference")
    axes[1].set_xlabel("inner paths M")
    axes[1].set_ylabel("disagreeing paths (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig4_heston_bias.png"), dpi=150)
    plt.close(fig)

    # Figure 5: inner-draw cost explosion
    fig, ax = plt.subplots(figsize=(9, 4))
    ms = [5, 20, 50, 200, H_M_MAX]
    ax.plot(ms, [HN_OUT * m for m in ms], "o-", color="#e07b39")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Inner path evaluations in the Heston experiment")
    ax.set_xlabel("inner paths M")
    ax.set_ylabel("outer x inner draws")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig5_cost_explosion.png"), dpi=150)
    plt.close(fig)

    out = {
        "seed": SEED,
        "s0": S0,
        "strike_inner": K1,
        "strike_outer": K2,
        "t1": T1,
        "maturity": T,
        "sigma": SIG,
        "rate": R,
        "critical_s_star": round(S_STAR, 6),
        "analytic_price": round(analytic_price, 6),
        "analytic_delta": round(analytic_delta, 6),
        "n_outer": N_OUT,
        "m_grid": M_GRID,
        "nested_price": {k: round(v["price"], 6) for k, v in study.items()},
        "nested_delta": {k: round(v["delta"], 6) for k, v in study.items()},
        "nested_delta_se": {k: round(v["delta_se"], 6) for k, v in study.items()},
        "price_bias": {k: round(v["price"] - analytic_price, 6) for k, v in study.items()},
        "delta_bias": {k: round(v["delta"] - analytic_delta, 6) for k, v in study.items()},
        "misclassification": {k: round(v["misclassification"], 6) for k, v in study.items()},
        "near_boundary_share": round(study["500"]["near_boundary_share"], 6),
        "budget": budget,
        "budget_total": BUDGET,
        "budget_reps": BUDGET_REPS,
        "heston_ref_m": H_M_MAX,
        "heston_ref_delta": round(ref_delta, 6),
        "heston_delta": {k: round(v["delta"], 6) for k, v in heston.items()},
        "heston_misclassification": {k: round(v["misclassification_vs_max"], 6) for k, v in heston.items()},
        "heston_inner_draws": {k: HN_OUT * int(k) for k in heston},
    }
    with open(os.path.join(IMG, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
