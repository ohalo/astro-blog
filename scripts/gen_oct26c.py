#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 2 quant blog articles (figures + numbers) for the automation run.
  - granger-causality  : Granger causality test (from-scratch + statsmodels)
  - hurst-exponent     : Hurst exponent via rescaled-range (R/S) analysis
Produces 5 figures per article under public/images/{slug}/ and dumps all
key numbers to _oct26c_numbers.json for the markdown authoring step.
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/halo123456/workspace/astro-blog"
PUB = os.path.join(ROOT, "public", "images")

def save(fig, slug, name):
    d = os.path.join(PUB, slug)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {slug}/{name}")
    return p

# --------------------------------------------------------------------------
# ARTICLE 1 : GRANGER CAUSALITY
# --------------------------------------------------------------------------
def granger_block():
    from numpy.linalg import lstsq
    rng = np.random.default_rng(20261026)
    N = 2000
    e = rng.normal(0, 1, N)
    u = rng.normal(0, 1, N)
    x = np.zeros(N); y = np.zeros(N)
    # DGP: x_t = 0.5 x_{t-1} + u_t   (own AR)
    #      y_t = 0.5 y_{t-1} + 0.3 x_{t-1} + e_t   (x Granger-causes y)
    for t in range(1, N):
        x[t] = 0.5 * x[t-1] + u[t]
        y[t] = 0.5 * y[t-1] + 0.3 * x[t-1] + e[t]
    x = x[200:]; y = y[200:]          # burn-in
    n = len(x)

    # ---- manual Granger F-test (from scratch) ----
    def ols(X, z):
        beta, *_ = lstsq(X, z, rcond=None)
        resid = z - X @ beta
        return beta, resid
    def granger_ftest(y, x, p=1):
        # unrestricted: y_t ~ y_{t-1..t-p} + x_{t-1..t-p}
        # restricted:   y_t ~ y_{t-1..t-p}
        T = len(y)
        Y = y[p:T]
        Xr = np.column_stack([np.ones(T-p)] + [y[p-k:T-k] for k in range(1, p+1)])
        Xu = np.column_stack([Xr,] + [x[p-k:T-k] for k in range(1, p+1)])
        br, rr = ols(Xr, Y)
        bu, ru = ols(Xu, Y)
        ssr_r = rr @ rr; ssr_u = ru @ ru
        k = Xu.shape[1] - Xr.shape[1]
        df1 = k; df2 = T - p - Xu.shape[1]
        F = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        # p-value from F(df1,df2)
        from scipy.stats import f as fdist
        pval = 1 - fdist.cdf(F, df1, df2)
        return F, pval, ssr_r, ssr_u

    F_xy, p_xy, ssr_r_xy, ssr_u_xy = granger_ftest(y, x, 1)   # x -> y
    F_yx, p_yx, ssr_r_yx, ssr_u_yx = granger_ftest(x, y, 1)   # y -> x

    # ---- statsmodels cross check ----
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        g_xy = grangercausalitytests(np.column_stack([y, x]), maxlag=1, verbose=False)
        g_yx = grangercausalitytests(np.column_stack([x, y]), maxlag=1, verbose=False)
        sm_F_xy = float(g_xy[1][0]["ssr_ftest"][0]); sm_p_xy = float(g_xy[1][0]["ssr_ftest"][1])
        sm_F_yx = float(g_yx[1][0]["ssr_ftest"][0]); sm_p_yx = float(g_yx[1][0]["ssr_ftest"][1])
    except Exception as ex:
        sm_F_xy = sm_p_xy = sm_F_yx = sm_p_yx = float("nan")
        print("statsmodels unavailable:", ex)

    # ---- out-of-sample 1-step conditional forecast ----
    tr = int(n * 0.7)
    xtr, ytr = x[:tr], y[:tr]
    xte, yte = x[tr:], y[tr:]
    # restricted model (no x): y_t ~ c + y_{t-1}
    br, _ = ols(np.column_stack([np.ones(tr-1), ytr[:-1]]), ytr[1:])
    # unrestricted (with x): y_t ~ c + y_{t-1} + x_{t-1}
    bu, _ = ols(np.column_stack([np.ones(tr-1), ytr[:-1], xtr[:-1]]), ytr[1:])
    # evaluate on test (use actual realized lags -> conditional 1-step)
    yhat_r = br[0] + br[1] * yte[:-1]
    yhat_u = bu[0] + bu[1] * yte[:-1] + bu[2] * xte[:-1]
    mse_r = np.mean((yte[1:] - yhat_r) ** 2)
    mse_u = np.mean((yte[1:] - yhat_u) ** 2)
    mse_red = (mse_r - mse_u) / mse_r * 100.0

    # ---- spurious example: high correlation from a shared CONTEMPORANEOUS
    #      shock, with NO lead-lag (predictive) structure ----
    sp_rng = np.random.default_rng(777)
    w = sp_rng.normal(0, 1, 900); e = sp_rng.normal(0, 1, 900); u = sp_rng.normal(0, 1, 900)
    sx = np.zeros(900); sy = np.zeros(900)
    for t in range(1, 900):
        sx[t] = 0.5 * sx[t-1] + 0.7 * w[t] + e[t]
        sy[t] = 0.5 * sy[t-1] + 0.7 * w[t] + u[t]
    sx = sx[100:]; sy = sy[100:]
    corr_rw = float(np.corrcoef(sx, sy)[0, 1])
    F_sp_12, p_sp_12, *_ = granger_ftest(sy, sx, 1)   # sx -> sy ?
    F_sp_21, p_sp_21, *_ = granger_ftest(sx, sy, 1)   # sy -> sx ?

    slug = "granger-causality"
    # Fig 1: simulated series
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax[0].plot(x, color="#2563eb", lw=0.8); ax[0].set_title("X series:  $x_t = 0.5 x_{t-1} + u_t$")
    ax[0].set_ylabel("x"); ax[1].plot(y, color="#dc2626", lw=0.8)
    ax[1].set_title("Y series:  $y_t = 0.5 y_{t-1} + 0.3 x_{t-1} + e_t$")
    ax[1].set_ylabel("y"); ax[1].set_xlabel("time")
    fig.tight_layout(); save(fig, slug, "fig1_sim.png")

    # Fig 2: y_t vs x_{t-1} predictive scatter
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(x[1:], y[1:], s=6, alpha=0.35, color="#2563eb")
    b1, b0 = np.polyfit(x[1:], y[1:], 1)
    xs = np.linspace(x[1:].min(), x[1:].max(), 50)
    ax.plot(xs, b0 + b1 * xs, color="#dc2626", lw=2)
    ax.set_xlabel(r"$x_{t-1}$ (lagged X)"); ax.set_ylabel(r"$y_t$")
    ax.set_title("Lagged X predicts Y  (slope = %.3f)" % b1)
    fig.tight_layout(); save(fig, slug, "fig2_scatter.png")

    # Fig 3: Granger F-stat both directions
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    dirs = ["X → Y\n(causal)", "Y → X\n(not causal)"]
    fs = [F_xy, F_yx]; ps = [p_xy, p_yx]
    bars = ax.bar(dirs, fs, color=["#16a34a", "#9ca3af"])
    for b, f, p in zip(bars, fs, ps):
        ax.text(b.get_x()+b.get_width()/2, f, f"F={f:.1f}\np={p:.3g}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Granger F-statistic"); ax.set_title("Granger causality test (lag 1)")
    fig.tight_layout(); save(fig, slug, "fig3_granger.png")

    # Fig 4: OOS forecast comparison
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tt = np.arange(len(yte[1:]))
    ax.plot(tt, yte[1:], color="#111827", lw=0.9, label="actual y")
    ax.plot(tt, yhat_u, color="#16a34a", lw=0.9, label="with X (unrestricted)")
    ax.plot(tt, yhat_r, color="#dc2626", lw=0.9, alpha=0.8, label="without X (restricted)")
    ax.set_xlabel("out-of-sample time"); ax.set_ylabel("y"); ax.legend()
    ax.set_title("1-step forecast: MSE reduction %.1f%%" % mse_red)
    fig.tight_layout(); save(fig, slug, "fig4_oos.png")

    # Fig 5: spurious correlation (shared contemporaneous shock)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(sx, color="#2563eb", lw=0.9, label="X (own AR + shared W)")
    ax.plot(sy, color="#dc2626", lw=0.9, label="Y (own AR + shared W)")
    ax.set_title("Shared contemporaneous shock\ncorr=%.2f, but no Granger causality" % corr_rw)
    ax.set_xlabel("time"); ax.legend()
    fig.tight_layout(); save(fig, slug, "fig5_spurious.png")

    nums = dict(
        N=N, n=int(n), F_xy=float(F_xy), p_xy=float(p_xy),
        F_yx=float(F_yx), p_yx=float(p_yx),
        sm_F_xy=float(sm_F_xy), sm_p_xy=float(sm_p_xy),
        sm_F_yx=float(sm_F_yx), sm_p_yx=float(sm_p_yx),
        mse_r=float(mse_r), mse_u=float(mse_u), mse_red=float(mse_red),
        tr=int(tr), corr_rw=float(corr_rw), p_sp_12=float(p_sp_12), p_sp_21=float(p_sp_21),
        b1=float(b1),
    )
    return nums

# --------------------------------------------------------------------------
# ARTICLE 2 : HURST EXPONENT (R/S analysis)
# --------------------------------------------------------------------------
def hosking_fgn(N, H, rng):
    def rho(k):
        if k == 0: return 1.0
        return 0.5 * ((k+1)**(2*H) - 2*k**(2*H) + (k-1)**(2*H))
    phi_prev = np.zeros(1)
    v_prev = 1.0
    X = np.zeros(N)
    eps = rng.normal(size=N)
    for nn in range(1, N+1):
        s1 = 0.0; s2 = 0.0
        for k in range(1, nn):
            s1 += phi_prev[k-1] * rho(nn - k)
            s2 += phi_prev[k-1] * rho(k)
        phi_nn = (rho(nn) - s1) / (1 - s2)
        phi_new = np.zeros(nn)
        for k in range(1, nn):
            phi_new[k-1] = phi_prev[k-1] - phi_nn * phi_prev[nn - 1 - k]
        phi_new[nn-1] = phi_nn
        v_n = v_prev * (1 - phi_nn**2)
        val = phi_nn * X[nn-1]
        for k in range(1, nn):
            val += phi_new[k-1] * X[nn - 1 - k]
        val += math.sqrt(v_n) * eps[nn-1]
        X[nn-1] = val
        phi_prev = phi_new
        v_prev = v_n
    return X

def rs_stat(series, ns, corrected=False):
    """rescaled range: mean R/S over non-overlapping blocks of size n.
    If corrected, subtract the Anis-Lloyd expected R/S under H=0.5 to
    remove the well-known finite-sample upward bias of naive R/S."""
    from scipy.special import gammaln
    out = []
    for nlen in ns:
        k = len(series) // nlen
        if k < 1: continue
        # Anis-Lloyd correction constant under H=0.5 (computed in log-space)
        logc = gammaln((nlen-1)/2.0) - 0.5*math.log(math.pi) - gammaln(nlen/2.0)
        E0 = (nlen - 0.5) / math.sqrt(nlen) * math.exp(logc)
        rs = 0.0
        for j in range(k):
            blk = series[j*nlen:(j+1)*nlen]
            dev = blk - blk.mean()
            Z = np.cumsum(dev)
            R = Z.max() - Z.min()
            S = blk.std(ddof=1)
            if S > 0:
                val = R / S
                if corrected:
                    val = val - E0
                rs += val
        out.append((nlen, rs / k))
    return np.array(out)

def agg_var(series, ms):
    """variogram / aggregated-variance method on the *increments* (fGn):
    Var(sum of m consecutive increments) ~ m^{2H}  ->  slope(ln Var, ln m) = 2H."""
    out = []
    for m in ms:
        k = len(series) // m
        if k < 2: continue
        blocks = series[:k*m].reshape(k, m)
        sums = blocks.sum(axis=1)
        out.append((m, sums.var(ddof=1)))
    return np.array(out)

def slope_h(logx, logy):
    A = np.vstack([logx, np.ones_like(logx)]).T
    beta, *_ = np.linalg.lstsq(A, logy, rcond=None)
    return beta[0]

def hurst_block():
    rng = np.random.default_rng(20261027)
    N = 2000
    Hs = [0.3, 0.5, 0.7]
    ns = [10, 20, 40, 80, 160, 320, 640]
    ms = [4, 8, 16, 32, 64, 128, 256]
    fbm = {}; fgn_series = {}; rs_h = {}; true_h = {}
    for H in Hs:
        fgn = hosking_fgn(N, H, rng)
        fb = np.cumsum(fgn)                       # fractional Brownian motion
        fbm[H] = fb
        fgn_series[H] = fgn
        # R/S is applied to the STATIONARY increments (fGn); on the integrated
        # path it degenerates to slope ~1.  slope(log R/S, log n) = H.
        rs = rs_stat(fgn, ns, corrected=False)
        rs_h[H] = slope_h(np.log(rs[:,0]), np.log(rs[:,1]))
        true_h[H] = H

    # ---- applied: returns vs |returns| with persistent volatility ----
    M = 4000
    v = np.zeros(M); v[0] = 1.0
    for t in range(1, M):
        v[t] = 0.98 * v[t-1] + 0.05 * rng.normal()
    v = np.abs(v) + 0.2
    ret = np.sqrt(v) * rng.normal(size=M)
    abs_ret = np.abs(ret)
    r_ret = rs_stat(ret, ns, corrected=False); r_abs = rs_stat(abs_ret, ns, corrected=False)
    h_ret = slope_h(np.log(r_ret[:,0]), np.log(r_ret[:,1]))
    h_abs = slope_h(np.log(r_abs[:,0]), np.log(r_abs[:,1]))

    slug = "hurst-exponent"
    # Fig 1: three fBm paths
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {0.3:"#dc2626", 0.5:"#111827", 0.7:"#2563eb"}
    for H in Hs:
        ax.plot(fbm[H][:800], lw=0.8, color=colors[H], label="H = %.1f" % H)
    ax.set_xlabel("time"); ax.set_ylabel("fBm path")
    ax.set_title("Fractional Brownian motion: rough (H<0.5) vs smooth (H>0.5)")
    ax.legend(); fig.tight_layout(); save(fig, slug, "fig1_paths.png")

    # Fig 2: R/S log-log fit on increments (slope = H)
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for H in Hs:
        rs = rs_stat(fgn_series[H], ns, corrected=False)
        ax.loglog(rs[:,0], rs[:,1], "o", color=colors[H], ms=4, label="H=%.1f sim" % H)
        b = slope_h(np.log(rs[:,0]), np.log(rs[:,1]))
        xs = np.logspace(np.log10(rs[:,0].min()), np.log10(rs[:,0].max()), 50)
        ax.loglog(xs, xs**b, "-", color=colors[H], alpha=0.8, lw=2, label="fit H=%.2f" % b)
    ax.set_xlabel("block size n"); ax.set_ylabel("R/S(n)")
    ax.set_title("Rescaled-range log-log: slope = Hurst H"); ax.legend(fontsize=8)
    fig.tight_layout(); save(fig, slug, "fig2_rsfit.png")

    # Fig 3: estimated vs true H
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = np.arange(len(Hs))
    ax.bar(xs-0.18, [true_h[H] for H in Hs], width=0.36, color="#9ca3af", label="true H")
    ax.bar(xs+0.18, [rs_h[H] for H in Hs], width=0.36, color="#16a34a", label="R/S estimate")
    ax.set_xticks(xs); ax.set_xticklabels(["H=0.3","H=0.5","H=0.7"])
    ax.set_ylabel("Hurst exponent"); ax.set_ylim(0, 0.95)
    ax.set_title("R/S recovers H (finite-sample bias noted in text)"); ax.legend()
    fig.tight_layout(); save(fig, slug, "fig3_est.png")

    # Fig 4: applied returns vs |returns|
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["returns\n(efficient)", "|returns|\n(vol proxy)"], [h_ret, h_abs],
           color=["#9ca3af","#dc2626"])
    ax.axhline(0.5, color="#111827", ls="--", lw=1, label="H=0.5 (no memory)")
    ax.set_ylabel("Hurst H"); ax.set_title("Applied: H(returns)=%.2f vs H(|r|)=%.2f" % (h_ret, h_abs))
    ax.legend(); fig.tight_layout(); save(fig, slug, "fig4_applied.png")

    # Fig 5: V-statistic diagnostic (R/S / sqrt(n)) on increments
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for H in Hs:
        rs = rs_stat(fgn_series[H], ns)
        V = rs[:,1] / np.sqrt(rs[:,0])
        ax.plot(rs[:,0], V, "o-", color=colors[H], label="H=%.1f" % H)
    ax.set_xscale("log"); ax.set_xlabel("block size n"); ax.set_ylabel("V(n)=R/S / sqrt(n)")
    ax.set_title("V-statistic: departs from flat when H≠0.5"); ax.legend()
    fig.tight_layout(); save(fig, slug, "fig5_vstat.png")

    nums = dict(
        N=N, Hs=Hs, true_h=true_h, rs_h=rs_h,
        M=M, h_ret=float(h_ret), h_abs=float(h_abs),
        ns=[int(x) for x in ns],
    )
    return nums

if __name__ == "__main__":
    g = granger_block()
    h = hurst_block()
    allnums = {"granger": g, "hurst": h}
    with open(os.path.join(ROOT, "_oct26c_numbers.json"), "w") as f:
        json.dump(allnums, f, indent=2)
    print("\n===== GRANGER =====")
    for k, v in g.items(): print(f"  {k} = {v}")
    print("\n===== HURST =====")
    for k, v in h.items():
        if isinstance(v, dict):
            print(f"  {k}: " + ", ".join(f"{kk}={vv:.4f}" for kk, vv in v.items()))
        else:
            print(f"  {k} = {v}")
    print("\nDONE")
