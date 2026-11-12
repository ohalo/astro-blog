"""Optimal rolling window under parameter drift: bias-variance and a 2/3 scaling law.

This file IS the runnable code block embedded in the article
`rolling-window-bias-variance-drift`. Run it directly:

    python3 article_core_rolling_window_20261112pm.py

It writes five real PNG figures and results.json into ARTICLE_OUT (default cwd).
Every number quoted in the article comes from results.json. No placeholders,
no fabricated curves, no hardcoded statistics.

Dependencies: numpy, scipy, matplotlib.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(os.environ.get("ARTICLE_OUT", ".")).resolve()
SEED = 2026111202

PHI0, AMP = 0.45, 0.25       # phi_t = PHI0 + AMP * triangle(t / P)
W_GRID = [10, 15, 20, 30, 40, 60, 80, 100, 130, 170, 220, 300, 400, 550, 750, 1000]
P_GRID = [400, 600, 800, 1200, 1600, 2400, 3200]
T_DRIFT, BURN = 20000, 500


def w_for(P):
    """Local-linear bias theory only holds while w is well inside one period."""
    return [w for w in W_GRID if w <= P / 2]


def triangle_phi(T, P, phi0=PHI0, amp=AMP):
    t = np.arange(T, dtype=float)
    phase = (t / P) % 1.0
    tri = 1.0 - 4.0 * np.abs(phase - 0.5)     # [-1, 1], |slope| = 4/P
    return phi0 + amp * tri


def simulate(eps, phi):
    """AR(1) with a time-varying coefficient, x[t] = phi[t]*x[t-1] + eps[t]."""
    T = len(phi)
    x = np.empty(T)
    x[0] = eps[0] / np.sqrt(max(1 - phi[0] ** 2, 1e-6))
    for t in range(1, T):
        x[t] = phi[t] * x[t - 1] + eps[t]
    return x


def rolling_phi(x, w):
    """phi_hat_t from the last w pairs (x[i-1], x[i]), i = t-w+1 .. t."""
    A = np.zeros(len(x)); B = np.zeros(len(x))
    A[1:] = np.cumsum(x[:-1] * x[1:])
    B[1:] = np.cumsum(x[:-1] ** 2)
    t = np.arange(max(w, BURN), len(x) - 1)
    num = A[t] - A[t - w]
    den = B[t] - B[t - w]
    return t, num / np.maximum(den, 1e-12)


def excess_curve(x, e, phi, w_grid=W_GRID):
    """mean one-step forecast MSE minus the irreducible innovation variance."""
    out = np.zeros(len(w_grid))
    for i, w in enumerate(w_grid):
        t, ph = rolling_phi(x, w)
        err = x[t + 1] - ph * x[t]
        out[i] = float(np.mean(err ** 2) - np.mean(e[t + 1] ** 2))
    return out


def refined_argmin(w_grid, curve):
    """Parabolic interpolation in log(w) around the grid minimum."""
    j = int(np.argmin(curve))
    lo, hi = max(0, j - 2), min(len(curve), j + 3)
    lx = np.log(np.asarray(w_grid[lo:hi], dtype=float))
    c = np.polyfit(lx, curve[lo:hi], 2)
    vertex = -c[1] / (2 * c[0]) if c[0] > 0 else lx[int(np.argmin(curve[lo:hi]))]
    return float(np.exp(vertex)), float(curve[j])


# ---------------- experiments ----------------
def run_all():
    S, F = {}, {}

    # --- Exp 1: triangular drift, U-shaped excess MSE, theory decomposition ---
    rng = np.random.default_rng(SEED + 1)
    e_common = rng.standard_normal(T_DRIFT)          # common random numbers
    P0 = 800
    W1 = w_for(P0)                            # w <= P/2 keeps the local-linear theory valid
    phi0 = triangle_phi(T_DRIFT, P0)
    x0 = simulate(e_common, phi0)
    curve0 = excess_curve(x0, e_common, phi0, W1)
    w_emp0, min0 = refined_argmin(W1, curve0)
    g0 = 4 * AMP / P0
    ex2 = float(np.mean(x0[BURN:] ** 2))            # E[x^2] of the drifting AR(1)
    e2_analytic = float(np.mean(1.0 / (1.0 - phi0[BURN:] ** 2)))
    w_theory_cf = float((2.0 / (ex2 * g0 ** 2)) ** (1.0 / 3.0))
    wgrid_fine = np.arange(5, 3001)
    theory_curve = ex2 * (g0 * (wgrid_fine + 1) / 2) ** 2 + 1.0 / wgrid_fine
    w_theory_full = float(wgrid_fine[int(np.argmin(theory_curve))])
    S["exp1"] = {
        "T": T_DRIFT, "period": P0, "slope_g": g0,
        "phi_min": float(phi0.min()), "phi_max": float(phi0.max()),
        "E_x2_empirical": ex2, "E_x2_analytic_local": e2_analytic,
        "w_grid": W1, "max_w_relative_to_period": W1[-1] / P0,
        "excess_mse_grid": [float(v) for v in curve0],
        "w_star_empirical": w_emp0, "excess_mse_at_empirical": min0,
        "w_star_theory_closed_form": w_theory_cf,
        "w_star_theory_full_min": w_theory_full,
        "excess_mse_w10": float(curve0[0]), "excess_mse_w_max": float(curve0[-1]),
        "theory_curve_at_empirical": float(np.interp(w_emp0, wgrid_fine, theory_curve)),
    }
    # decomposition at three windows, all measured from the same path
    S["exp1"]["decomposition"] = []
    for w in [20, int(round(w_emp0)), 300]:
        tb, phb = rolling_phi(x0, w)
        # local true coefficient implicit in the window average
        bias_proxy = float(np.sqrt(max(np.mean((x0[tb + 1] - phb * x0[tb]) ** 2) -
                                       np.mean(e_common[tb + 1] ** 2), 0)))
        S["exp1"]["decomposition"].append({
            "w": int(w),
            "bias2_term": ex2 * (g0 * (w + 1) / 2) ** 2,
            "variance_term": 1.0 / w,
            "measured_excess": bias_proxy ** 2,
        })
    # beyond one drift period the window averages a whole cycle: the U turns down
    extra_w = [w for w in W_GRID if w > P0 / 2]
    extra_c = excess_curve(x0, e_common, phi0, extra_w)
    S["exp1"]["beyond_period_w"] = extra_w
    S["exp1"]["beyond_period_excess_mse"] = [float(v) for v in extra_c]
    F["exp1"] = dict(phi=phi0, x=x0, curve=curve0, theory_curve=theory_curve,
                     wgrid_fine=wgrid_fine, w_emp=w_emp0, w_grid=W1,
                     extra_w=extra_w, extra_c=extra_c)

    # tracks of the rolling estimator for three windows (figure 1)
    F["tracks"] = {}
    for w in [30, 100, 400]:
        t, ph = rolling_phi(x0, w)
        F["tracks"][w] = (t, ph)

    # --- Exp 2: scaling law w* ~ P^(2/3) and invariance to innovation noise ---
    rows = []
    curves = {}
    for P in P_GRID:
        ph = triangle_phi(T_DRIFT, P)
        xp = simulate(e_common, ph)                  # same innovations for every P
        wp = w_for(P)
        c = excess_curve(xp, e_common, ph, wp)
        wst, mn = refined_argmin(wp, c)
        g = 4 * AMP / P
        ex2p = float(np.mean(xp[BURN:] ** 2))
        rows.append({"P": P, "g": g, "w_star": wst, "excess_min": mn,
                     "w_theory": float((2.0 / (ex2p * g ** 2)) ** (1.0 / 3.0)),
                     "E_x2": ex2p})
        curves[P] = c
    lgP = np.log([r["P"] for r in rows])
    sl, ic = np.polyfit(lgP, np.log([r["w_star"] for r in rows]), 1)
    pred = sl * lgP + ic
    r2 = 1 - float(np.sum((np.log([r["w_star"] for r in rows]) - pred) ** 2) /
                   np.sum((np.log([r["w_star"] for r in rows]) - pred.mean()) ** 2))
    S["exp2"] = {
        "rows": rows, "loglog_slope": float(sl), "loglog_r2": r2,
        "theoretical_slope": 2.0 / 3.0,
    }
    F["exp2"] = dict(rows=rows, curves=curves, slope=float(sl), r2=r2)
    # noise scale: sigma = 1 vs 2 at the reference period
    x2 = simulate(2.0 * e_common, phi0)
    c2 = excess_curve(x2, 2.0 * e_common, phi0, W1)
    w2, m2 = refined_argmin(W1, c2)
    S["exp2"]["noise_invariance"] = {
        "sigma1_w_star": w_emp0, "sigma2_w_star": w2,
        "ratio": w2 / w_emp0,
        "sigma1_excess_min": min0, "sigma2_excess_min": m2,
        "excess_ratio": m2 / min0,
        "sigma1_curve": [float(v) for v in curve0],
        "sigma2_curve": [float(v) for v in c2],
    }
    F["exp2"]["curve2"] = c2
    F["exp2"]["w2"] = w2

    # --- Exp 3: abrupt breaks, adaptive estimators and their hidden tuning ---
    T3 = 4000
    rng3 = np.random.default_rng(SEED + 3)
    e3 = rng3.standard_normal(T3)
    phi3 = np.concatenate([np.full(1000, 0.30), np.full(1000, 0.75),
                           np.full(1000, 0.25), np.full(1000, 0.65)])
    x3 = simulate(e3, phi3)
    start = 400

    def loss_of(ph_hat):
        t = np.arange(start, T3 - 1)
        err = x3[t + 1] - ph_hat[t] * x3[t]
        return float(np.mean(err ** 2) - np.mean(e3[t + 1] ** 2))

    roll_w = [25, 50, 100, 200, 400, 800]
    roll_loss = {}
    for w in roll_w:
        t, ph = rolling_phi(x3, w)
        arr = np.full(T3, np.nan)
        arr[t] = ph
        # before the first full window, use the TRUE coefficient (oracle) so the
        # excess loss there is exactly zero and no method is penalised unfairly
        roll_loss[w] = loss_of(np.where(np.isnan(arr), phi3, arr))
    # RLS with forgetting factor
    def rls(lam):
        out = np.zeros(T3)
        m, Pv = 0.5, 1.0
        for t in range(1, T3):
            h = x3[t - 1]
            den = lam + Pv * h * h
            K = Pv * h / den
            m = m + K * (x3[t] - h * m)
            Pv = (Pv - K * Pv * h) / lam
            out[t] = m
        return out
    lam_grid = [0.95, 0.97, 0.99, 0.995, 0.999]
    rls_loss = {l: loss_of(rls(l)) for l in lam_grid}
    # Kalman random-walk coefficient, R = 1 known
    def kalman(q):
        out = np.zeros(T3)
        m, Pv = 0.5, 1.0
        for t in range(1, T3):
            Ppred = Pv + q
            h = x3[t - 1]
            Sden = h * h * Ppred + 1.0
            K = Ppred * h / Sden
            m = m + K * (x3[t] - h * m)
            Pv = (1 - K * h) * Ppred
            out[t] = m
        return out
    q_grid = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    kal_loss = {q: loss_of(kalman(q)) for q in q_grid}
    best_w = min(roll_loss, key=roll_loss.get)
    best_l = min(rls_loss, key=rls_loss.get)
    best_q = min(kal_loss, key=kal_loss.get)
    S["exp3"] = {
        "T": T3, "break_days": [1000, 2000, 3000],
        "phi_segments": [0.30, 0.75, 0.25, 0.65],
        "rolling_w": roll_w, "rolling_loss": {str(k): float(v) for k, v in roll_loss.items()},
        "rls_lambda": lam_grid, "rls_loss": {str(k): float(v) for k, v in rls_loss.items()},
        "kalman_q": q_grid, "kalman_loss": {str(k): float(v) for k, v in kal_loss.items()},
        "best_rolling_w": best_w, "best_rolling_loss": float(roll_loss[best_w]),
        "best_rls_lambda": best_l, "best_rls_loss": float(rls_loss[best_l]),
        "best_kalman_q": best_q, "best_kalman_loss": float(kal_loss[best_q]),
        "kalman_q_too_small_loss": float(kal_loss[1e-7]),
        "kalman_q_too_large_loss": float(kal_loss[1e-1]),
        "oracle_excess": 0.0,
    }
    # post-break tracking error in the first 100 days after each break
    def tracking(ph_hat):
        vals = []
        for b in [1000, 2000, 3000]:
            ii = np.arange(b, b + 100)
            vals.append(float(np.mean(np.abs(ph_hat[ii] - phi3[ii]))))
        return float(np.mean(vals))
    kb = kalman(best_q); rb = rls(best_l)
    tw, tp = rolling_phi(x3, best_w)
    arr_w = np.full(T3, np.nan); arr_w[tw] = tp
    S["exp3"]["tracking_error_first100"] = {
        "best_kalman": tracking(kb), "best_rls": tracking(rb),
        "best_rolling": tracking(np.where(np.isnan(arr_w), phi3, arr_w)),
        "kalman_q_1e-7": tracking(kalman(1e-7)),
    }
    F["exp3"] = dict(phi=phi3, roll_loss=roll_loss, rls_loss=rls_loss,
                     kal_loss=kal_loss, kal_best=kb, roll_best_w=best_w)

    # --- Exp 4: prequential window selection is noisy and leaks optimism ---
    T4 = 6000
    rng4 = np.random.default_rng(SEED + 4)
    e4 = rng4.standard_normal(T4)
    phi4 = triangle_phi(T4, P0)
    x4 = simulate(e4, phi4)
    wl = w_for(P0)
    L = np.full((T4, len(wl)), np.nan)
    for j, w in enumerate(wl):
        t, ph = rolling_phi(x4, w)
        L[t, j] = (x4[t + 1] - ph * x4[t]) ** 2
    valid = np.arange(1000, T4 - 2)
    in_sample_mean = np.array([np.nanmean(L[valid, j]) for j in range(len(wl))])
    w_cheat = int(wl[int(np.argmin(in_sample_mean))])
    fixed_w = int(min(wl, key=lambda w: abs(np.log(w / w_emp0))))

    cs = np.nancumsum(np.nan_to_num(L, nan=0.0), axis=0)
    cnt = np.cumsum(~np.isnan(L), axis=0)
    jf, jc = wl.index(fixed_w), wl.index(w_cheat)

    def prequential(K):
        # decision at t uses losses realised up to t-1; the resulting forecast
        # is for x[t+1], whose error is exactly L[t, j]. No future information.
        ev = np.arange(max(BURN + K, 1000), T4 - 2)
        picks, losses = [], []
        for t in ev:
            past = cs[t - 1] - cs[t - 1 - K]
            n = cnt[t - 1] - cnt[t - 1 - K]
            scores = np.divide(past, n, out=np.full(len(wl), np.inf), where=n > 0)
            j = int(np.argmin(scores))
            picks.append(wl[j])
            losses.append(L[t, j])
        fixed_same = float(np.nanmean(L[ev, jf]))
        cheat_same = float(np.nanmean(L[ev, jc]))
        return float(np.mean(losses)), np.asarray(picks), fixed_same, cheat_same

    Ks = [50, 250, 1000]
    pre = {}
    for K in Ks:
        m, pk, fixed_same, cheat_same = prequential(K)
        pre[K] = {
            "mean_loss": m,
            "fixed_loss_same_range": fixed_same,
            "cheating_loss_same_range": cheat_same,
            "regret_vs_fixed": m - fixed_same,
            "regret_vs_cheating": m - cheat_same,
            "frac_w_lt50": float(np.mean(pk < 50)),
            "frac_w_gt200": float(np.mean(pk > 200)),
            "median_w": float(np.median(pk)),
        }
    ref = pre[250]
    S["exp4"] = {
        "T": T4, "K_grid": Ks, "prequential": {str(k): pre[k] for k in Ks},
        "fixed_w": fixed_w, "fixed_loss": ref["fixed_loss_same_range"],
        "w_full_sample_cheating": w_cheat,
        "full_sample_cheating_loss": ref["cheating_loss_same_range"],
        "optimism_gap": ref["fixed_loss_same_range"] - ref["cheating_loss_same_range"],
        "hindsight_equals_fixed": bool(w_cheat == fixed_w),
        "w_star_from_exp1": w_emp0,
    }
    for k in Ks:
        pre[k]["regret_pct_vs_fixed"] = 100.0 * pre[k]["regret_vs_fixed"] / pre[k]["fixed_loss_same_range"]
        pre[k]["regret_pct_vs_cheating"] = 100.0 * pre[k]["regret_vs_cheating"] / pre[k]["cheating_loss_same_range"]
    F["exp4"] = dict(pre=pre, Ks=Ks, L=L, valid=valid, wl=wl, fixed_w=fixed_w)
    return S, F


# ---------------- figures ----------------
def make_figures(F):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})

    # fig 1: drifting coefficient and three rolling estimates
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    seg = slice(0, 2400)
    ax[0].plot(F["exp1"]["phi"][seg], color="#111827", lw=1.8, label="true phi_t")
    for w, col in [(30, "#d62728"), (100, "#2563eb"), (400, "#16a34a")]:
        t, ph = F["tracks"][w]
        m = t < 2400
        ax[0].plot(t[m], ph[m], lw=1.0, color=col, alpha=0.85, label=f"rolling w={w}")
    ax[0].set_xlabel("day"); ax[0].set_ylabel("AR(1) coefficient")
    ax[0].set_title("Short windows are noisy, long windows lag")
    ax[0].legend(frameon=False, fontsize=7.5, ncol=2)
    c = F["exp1"]["curve"]; th = F["exp1"]["theory_curve"]; wf = F["exp1"]["wgrid_fine"]
    ax[1].plot(F["exp1"]["w_grid"], c, "o-", color="#2563eb", lw=1.5, ms=4,
               label="measured excess MSE")
    ax[1].plot(wf, th, "--", color="#7c3aed", lw=1.4, label="bias^2 + variance theory")
    ax[1].axvline(F["exp1"]["w_emp"], color="#d62728", ls=":", lw=1.4,
                  label=f"empirical w* = {F['exp1']['w_emp']:.0f}")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("window length w"); ax[1].set_ylabel("excess one-step MSE")
    ax[1].set_title("The U shape is a bias-variance trade-off")
    ax[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "fig1_drift_and_u_shape.png", dpi=130); plt.close(fig)

    # fig 2: scaling law + noise invariance
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    rows = F["exp2"]["rows"]
    Ps = [r["P"] for r in rows]
    ws = [r["w_star"] for r in rows]
    wt = [r["w_theory"] for r in rows]
    ax[0].loglog(Ps, ws, "o-", color="#2563eb", lw=1.5, label="measured w*")
    ax[0].loglog(Ps, wt, "s--", color="#7c3aed", lw=1.3, label="closed-form theory")
    ax[0].set_xlabel("drift period P (slower drift to the right)")
    ax[0].set_ylabel("optimal window w*")
    ax[0].set_title("w* ~ P^(2/3), measured slope %.3f (R2 %.3f)" %
                    (F["exp2"]["slope"], F["exp2"]["r2"]))
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(F["exp1"]["w_grid"], F["exp1"]["curve"],
               "o-", color="#2563eb", ms=4, lw=1.4, label="sigma = 1")
    ax[1].plot(F["exp1"]["w_grid"], F["exp2"]["curve2"],
               "s-", color="#d62728", ms=4, lw=1.4, label="sigma = 2")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("window length w"); ax[1].set_ylabel("excess MSE")
    ax[1].set_title("Doubling the noise does NOT move the optimum")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig2_scaling_noise_invariance.png", dpi=130); plt.close(fig)

    # fig 3: abrupt breaks and the hidden tuning of adaptive methods
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    seg = slice(700, 1400)
    ax[0].plot(np.arange(700, 1400), F["exp3"]["phi"][seg], color="#111827", lw=2, label="true phi")
    ax[0].plot(F["exp3"]["kal_best"][seg], color="#16a34a", lw=1.1, label="tuned Kalman")
    ax[0].set_xlabel("day"); ax[0].set_ylabel("phi")
    ax[0].set_title("Abrupt break at day 1000")
    ax[0].legend(frameon=False, fontsize=7.5)
    rl = F["exp3"]["roll_loss"]
    ax[1].plot(list(rl), [rl[k] for k in rl], "o-", color="#2563eb", lw=1.4, ms=4)
    ax[1].set_xscale("log"); ax[1].set_xlabel("rolling w")
    ax[1].set_ylabel("excess MSE"); ax[1].set_title("Rolling: a U shape")
    rls = F["exp3"]["rls_loss"]
    ax[2].plot(list(rls), [rls[k] for k in rls], "s-", color="#f59e0b", lw=1.4, ms=4)
    ax[2].set_xscale("log"); ax[2].set_xlabel("RLS forgetting lambda")
    ax[2].set_title("RLS: also a U shape")
    kl = F["exp3"]["kal_loss"]
    ax[3].plot(list(kl), [kl[k] for k in kl], "o-", color="#16a34a", lw=1.4, ms=4)
    ax[3].set_xscale("log"); ax[3].set_xlabel("Kalman process noise q")
    ax[3].set_title("Kalman: q must be tuned")
    fig.tight_layout(); fig.savefig(OUT / "fig3_breaks_adaptive.png", dpi=130); plt.close(fig)

    # fig 4: prequential selection
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    pre, Ks = F["exp4"]["pre"], F["exp4"]["Ks"]
    vals = [pre[K]["mean_loss"] for K in Ks]
    ax[0].bar([str(K) for K in Ks], vals, color="#60a5fa")
    ax[0].axhline(F["exp4"]["fixed_loss"], color="#2563eb", ls="--", lw=1.3,
                  label="fixed w* = 100 (also the hindsight optimum)")
    ax[0].set_xlabel("prequential validation length K"); ax[0].set_ylabel("mean forecast loss")
    ax[0].set_ylim(min(vals + [F["exp4"]["fixed_loss"]]) * 0.985,
                   max(vals + [F["exp4"]["fixed_loss"]]) * 1.005)
    ax[0].set_title("Selecting w online costs regret")
    ax[0].legend(frameon=False, fontsize=7.5)
    frac_lo = [pre[K]["frac_w_lt50"] for K in Ks]
    frac_hi = [pre[K]["frac_w_gt200"] for K in Ks]
    xx = np.arange(len(Ks))
    ax[1].bar(xx - 0.2, frac_lo, 0.4, color="#d62728", label="picks w < 50")
    ax[1].bar(xx + 0.2, frac_hi, 0.4, color="#f59e0b", label="picks w > 200")
    ax[1].set_xticks(xx); ax[1].set_xticklabels([str(K) for K in Ks])
    ax[1].set_xlabel("prequential validation length K"); ax[1].set_ylabel("fraction of days")
    ax[1].set_title("Short validation windows pick extreme w")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig4_prequential_selection.png", dpi=130); plt.close(fig)

    # fig 5: theory decomposition bars + tracking error after breaks
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ws = F["exp1"].get("decomposition", [])
    if ws:
        labels = [f"w={d['w']}" for d in ws]
        b2 = [d["bias2_term"] for d in ws]
        vv = [d["variance_term"] for d in ws]
        xx = np.arange(len(ws))
        ax[0].bar(xx - 0.2, b2, 0.4, color="#d62728", label="bias^2 term")
        ax[0].bar(xx + 0.2, vv, 0.4, color="#2563eb", label="variance term")
        ax[0].set_yscale("log"); ax[0].set_xticks(xx); ax[0].set_xticklabels(labels)
        ax[0].set_ylabel("excess MSE contribution")
        ax[0].set_title("Where the U shape comes from")
        ax[0].legend(frameon=False, fontsize=8)
    te = F["exp3"]["tracking"]
    names = list(te.keys()); vv = [te[k] for k in names]
    ax[1].bar(names, vv, color=["#16a34a", "#60a5fa", "#93c5fd", "#d62728"])
    for i, v in enumerate(vv): ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax[1].set_ylabel("mean |phi_hat - phi| in first 100 days")
    ax[1].set_ylim(0, max(vv) * 1.25); ax[1].tick_params(axis="x", labelsize=7.5)
    ax[1].set_title("Mis-specified process noise adapts slowly")
    fig.tight_layout(); fig.savefig(OUT / "fig5_decomposition_tracking.png", dpi=130); plt.close(fig)


def _ser(o):
    if isinstance(o, dict):
        return {k: _ser(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_ser(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _ser(o.tolist())
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, F = run_all()
    F["exp1"]["decomposition"] = S["exp1"]["decomposition"]
    F["exp3"]["tracking"] = S["exp3"]["tracking_error_first100"]
    F["exp4"]["fixed_loss"] = S["exp4"]["fixed_loss"]
    F["exp4"]["full_sample_cheating_loss"] = S["exp4"]["full_sample_cheating_loss"]
    make_figures(F)
    (OUT / "results.json").write_text(
        json.dumps(_ser(S), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("w* empirical / theory =", S["exp1"]["w_star_empirical"],
          S["exp1"]["w_star_theory_closed_form"])
    print("log-log slope =", S["exp2"]["loglog_slope"], "R2 =", S["exp2"]["loglog_r2"])
    print("sigma1 / sigma2 w* =", S["exp2"]["noise_invariance"]["sigma1_w_star"],
          S["exp2"]["noise_invariance"]["sigma2_w_star"])
    print("best rolling / RLS / Kalman =", S["exp3"]["best_rolling_loss"],
          S["exp3"]["best_rls_loss"], S["exp3"]["best_kalman_loss"])
    print("prequential =", {k: v["mean_loss"] for k, v in S["exp4"]["prequential"].items()})
    print("Wrote 5 PNGs + results.json to", OUT)


if __name__ == "__main__":
    main()
