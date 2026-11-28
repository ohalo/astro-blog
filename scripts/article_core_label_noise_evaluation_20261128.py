"""Noisy close-price labels: direction-strategy evaluation, confusion correction and audit cost."""

def main():
    import json
    import math
    import os

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    SEED = 20261128
    OUT = "/Users/halo/workspace/astro-blog/public/images/label-noise-direction-strategy-evaluation"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    N_TRAIN = 12000
    N_TEST = 6000
    B = 400
    SIG_R = 0.0062
    TRUE_BETA = 0.30
    SIG_GRID = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008]
    SIG_MAIN = 0.003

    def make_base(n):
        x = rng.standard_normal((n, 3))
        eps = rng.standard_normal(n)
        r = SIG_R * (TRUE_BETA * x[:, 0] + eps)
        return x, r

    def add_noise(r, sig_eta):
        eta = rng.standard_normal(len(r) + 1) * sig_eta
        return r + eta[1:] - eta[:-1]

    def fit_predict(x_train, lab_train, x_test):
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
        model.fit(x_train, lab_train)
        return model, model.predict_proba(x_test)[:, 1]

    def sharpe(pnl):
        s = pnl.std(ddof=1)
        return float(pnl.mean() / s * math.sqrt(252)) if s > 0 else 0.0

    xt, rt = make_base(N_TRAIN)
    xv, rv = make_base(N_TEST)
    grid = {}
    data_store = {}
    for sig_eta in SIG_GRID:
        yt = add_noise(rt, sig_eta)
        yv = add_noise(rv, sig_eta)
        yl_t = (yt > 0).astype(int)
        yl_v = (yv > 0).astype(int)
        rl_v = (rv > 0).astype(int)
        model_noisy, p_noisy = fit_predict(xt, yl_t, xv)
        model_clean, p_clean = fit_predict(xt, (rt > 0).astype(int), xv)
        pred = (p_noisy > 0.5).astype(int)
        pos = 2 * pred - 1
        pnl_true = pos * rv
        pnl_obs = pos * yv
        pnl_proxy = pos * (2 * yl_v - 1)
        grid[str(sig_eta)] = {
            "flip_rate": float(np.mean(yl_v != rl_v)),
            "beta_noisy": float(model_noisy.coef_[0][0]),
            "acc_noisy": float(np.mean(pred == yl_v)),
            "acc_true": float(np.mean(pred == rl_v)),
            "acc_clean_model_true": float(np.mean((p_clean > 0.5).astype(int) == rl_v)),
            "sharpe_true": sharpe(pnl_true),
            "sharpe_obs": sharpe(pnl_obs),
            "sharpe_proxy": sharpe(pnl_proxy),
        }
        data_store[sig_eta] = (xv, rv, yv, yl_v, rl_v, p_noisy)

    # Main scenario: threshold sweep, confusion correction, audit-size sensitivity.
    xv, rv, yv, yl_v, rl_v, p_noisy = data_store[SIG_MAIN]
    pred = (p_noisy > 0.5).astype(int)
    thresholds = [0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60]
    sweep = []
    for th in thresholds:
        pnl = (2 * (p_noisy > th).astype(int) - 1) * rv
        hit = (p_noisy > th).astype(int) == yl_v
        sweep.append({"threshold": th, "sharpe_true": sharpe(pnl),
                      "acc_noisy": float(hit.mean())})

    rho_true = grid[str(SIG_MAIN)]["flip_rate"]
    acc_obs = grid[str(SIG_MAIN)]["acc_noisy"]
    acc_true = grid[str(SIG_MAIN)]["acc_true"]
    corrected_point = (acc_obs - rho_true) / (1 - 2 * rho_true)
    flip_flat = yl_v != rl_v
    true_sign = 2 * rl_v - 1
    obs_sign = 2 * yl_v - 1
    flip_given_true_up = float(flip_flat[true_sign > 0].mean())
    flip_given_true_down = float(flip_flat[true_sign < 0].mean())

    up = pred == 1
    q_up = float(np.mean(yl_v[up] == 1))
    rho_up = float(flip_flat[up].mean())
    t_up = (q_up - rho_up) / max(1 - 2 * rho_up, 0.02)
    dn = ~up
    q_dn = float(np.mean(yl_v[dn] == 0))
    rho_dn = float(flip_flat[dn].mean())
    t_dn = (q_dn - rho_dn) / max(1 - 2 * rho_dn, 0.02)
    cond_corrected = float(up.mean() * t_up + dn.mean() * t_dn)

    agreement = pred == rl_v
    rho_correct = float(flip_flat[agreement].mean())
    rho_wrong = float(flip_flat[~agreement].mean())
    agreement_corrected = (acc_obs - rho_wrong) / (1 - rho_correct - rho_wrong)

    audit_sizes = [20, 50, 100, 200, 500, 1000, 3000]
    audit_stats = []
    for m in audit_sizes:
        rho_hats = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, N_TEST, size=m)
            rho_hats[b] = flip_flat[idx].mean()
        corrected = (acc_obs - rho_hats) / np.maximum(1 - 2 * rho_hats, 0.02)
        audit_stats.append({
            "m": m,
            "rho_bias": float(rho_hats.mean() - rho_true),
            "rho_sd": float(rho_hats.std(ddof=1)),
            "acc_sd": float(corrected.std(ddof=1)),
            "absurd_frac": float(np.mean((corrected > 1) | (corrected < 0))),
            "p10": float(np.quantile(corrected, 0.10)),
            "p90": float(np.quantile(corrected, 0.90)),
        })

    # High-noise regime: correction itself becomes unusable.
    high = {}
    for sig_eta in [0.006, 0.010, 0.014, 0.018, 0.022]:
        yt = add_noise(rt, sig_eta)
        yv2 = add_noise(rv, sig_eta)
        yl2 = (yv2 > 0).astype(int)
        rl2 = (rv > 0).astype(int)
        _, p2 = fit_predict(xt, (yt > 0).astype(int), xv)
        pred2 = (p2 > 0.5).astype(int)
        acc_o = float(np.mean(pred2 == yl2))
        rho = float(np.mean(yl2 != rl2))
        corrected = (acc_o - rho) / max(1 - 2 * rho, 0.02)
        high[str(sig_eta)] = {"rho": rho, "acc_obs": acc_o,
                              "corrected": float(corrected),
                              "acc_true": float(np.mean(pred2 == rl2))}

    results = {
        "params": {"seed": SEED, "n_train": N_TRAIN, "n_test": N_TEST,
                   "sig_r": SIG_R, "true_beta": TRUE_BETA, "sig_main": SIG_MAIN,
                   "audit_draws": B},
        "grid": grid,
        "main": {"rho": rho_true, "acc_obs": acc_obs, "acc_true": acc_true,
                 "corrected_point": float(corrected_point),
                 "conditional_corrected": cond_corrected,
                 "agreement_corrected": float(agreement_corrected),
                 "rho_correct": rho_correct, "rho_wrong": rho_wrong,
                 "flip_given_true_up": flip_given_true_up,
                 "flip_given_true_down": flip_given_true_down,
                 "rho_pred_up": rho_up, "rho_pred_down": rho_dn,
                 "sweep": sweep},
        "audit": audit_stats,
        "high_noise": high,
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig1: true vs observed return and the confusion matrix.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.scatter(rv[:1800] * 100, yv[:1800] * 100, s=5, alpha=0.28, color="#4c78a8")
    lim = 2.5
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axvline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("true return (percent)"); ax.set_ylabel("observed return (percent)")
    ax.set_title("Close-price noise rotates signs near zero")
    cm = np.array([[np.mean((rl_v == 0) & (yl_v == 0)), np.mean((rl_v == 0) & (yl_v == 1))],
                   [np.mean((rl_v == 1) & (yl_v == 0)), np.mean((rl_v == 1) & (yl_v == 1))]])
    ax = axes[1]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1], labels=["obs down", "obs up"])
    ax.set_yticks([0, 1], labels=["true down", "true up"])
    ax.set_title(f"Confusion matrix, flip rate {rho_true:.3f}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_noise_confusion.png"), dpi=150)
    plt.close(fig)

    # fig2: flip probability by true-return size.
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    edges = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 4.0])
    centers, probs = [], []
    absr = np.abs(rv) * 100
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (absr >= lo) & (absr < hi)
        if sel.sum() > 20:
            centers.append(0.5 * (lo + hi))
            probs.append(float(flip_flat[sel].mean()))
    ax.plot(centers, probs, "o-", color="#e45756", label="empirical flip rate")
    ax.axhline(rho_true, color="black", ls="--", lw=1.0, label="overall flip rate")
    ax.set_xscale("log")
    ax.set_xlabel("absolute true return (percent)")
    ax.set_ylabel("probability of sign flip")
    ax.set_title("Label errors concentrate on small true moves")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_flip_vs_return.png"), dpi=150)
    plt.close(fig)

    # fig3: attenuation of fitted signal and accuracy gap.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sg = np.array(SIG_GRID)
    ax = axes[0]
    ax.plot(sg * 100, [grid[str(s)]["beta_noisy"] for s in SIG_GRID], "o-", color="#4c78a8")
    ax.set_xlabel("close noise sd (percent)")
    ax.set_ylabel("fitted coefficient on true signal")
    ax.set_title("Label noise attenuates the learned signal")
    ax = axes[1]
    ax.plot(sg * 100, [grid[str(s)]["acc_noisy"] for s in SIG_GRID], "o-", color="#e45756",
            label="accuracy vs noisy labels")
    ax.plot(sg * 100, [grid[str(s)]["acc_true"] for s in SIG_GRID], "s--", color="#2ca02c",
            label="accuracy vs true labels")
    ax.set_xlabel("close noise sd (percent)")
    ax.set_ylabel("directional accuracy")
    ax.set_title("Two accuracies drift apart")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_attenuation.png"), dpi=150)
    plt.close(fig)

    # fig4: Sharpe under three evaluation conventions and threshold sweep.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.plot(sg * 100, [grid[str(s)]["sharpe_true"] for s in SIG_GRID], "o-", color="#2ca02c",
            label="true-return Sharpe")
    ax.plot(sg * 100, [grid[str(s)]["sharpe_obs"] for s in SIG_GRID], "s--", color="#4c78a8",
            label="noisy-return Sharpe")
    ax.plot(sg * 100, [grid[str(s)]["sharpe_proxy"] for s in SIG_GRID], "^:", color="#e45756",
            label="sign-proxy Sharpe")
    ax.set_xlabel("close noise sd (percent)")
    ax.set_ylabel("annualized Sharpe")
    ax.set_title("Sign proxy overstates performance")
    ax.legend(fontsize=8)
    ax = axes[1]
    ths = [r["threshold"] for r in sweep]
    ax.plot(ths, [r["sharpe_true"] for r in sweep], "o-", color="#2ca02c", label="true Sharpe")
    ax.plot(ths, [r["acc_noisy"] for r in sweep], "s--", color="#e45756", label="noisy accuracy")
    ax.set_xlabel("probability threshold")
    ax.set_ylabel("Sharpe / accuracy")
    ax.set_title("Optimizing the noisy metric picks a bad threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_sharpe_threshold.png"), dpi=150)
    plt.close(fig)

    # fig5: audit uncertainty and the high-noise breakdown.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ms = [a["m"] for a in audit_stats]
    ax.plot(ms, [a["acc_sd"] for a in audit_stats], "o-", color="#4c78a8")
    ax.set_xscale("log")
    ax.set_xlabel("clean audit sample size")
    ax.set_ylabel("sd of corrected accuracy")
    ax.set_title("Small audits make the correction unstable")
    ax = axes[1]
    hn = np.array([float(k) for k in high.keys()]) * 100
    ax.plot(hn, [high[k]["rho"] for k in high], "o-", color="#e45756", label="flip rate")
    ax.plot(hn, [high[k]["corrected"] for k in high], "s--", color="#4c78a8",
            label="corrected accuracy")
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.set_xlabel("close noise sd (percent)")
    ax.set_ylabel("rate")
    ax.set_title("Correction diverges as flip rate approaches 50 percent")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_audit_breakdown.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
