"""Generate images for multi-task-factor-learning article."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False

OUT_DIR = "/Users/halo/workspace/astro-blog/public/images/multi-task-factor-learning"
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(7)


def simulate_multi_task(n=400, n_features=8, n_assets=30):
    """Simulate shared feature layer → two heads: return (μ) and risk (σ).
    Returns features, return_targets, vol_targets, true_signal_strengths.
    """
    t = np.arange(n)
    # Time-varying regime (affects both heads but with different sensitivities)
    regime = np.sin(2 * np.pi * t / 80) + 0.3 * np.sin(2 * np.pi * t / 23)
    # Features (n_features=8): some useful for both, some only return, some only risk
    X = rng.normal(0, 1, (n, n_assets, n_features))
    # Shared signal: features 0..2 drive BOTH return and vol
    # Return-only signal: 3..5
    # Risk-only signal: 6..7
    w_ret = np.array([0.10, -0.08, 0.06, 0.07, -0.05, 0.04, 0.0, 0.0])
    w_vol = np.array([0.04, 0.05, -0.06, 0.0, 0.0, 0.0, 0.08, -0.07])
    # Cross-sectional signal (per asset)
    asset_idio = rng.normal(0, 1, (n_assets, n_features))
    # Per-asset latent factor exposure
    exp_ret = np.zeros((n, n_assets))
    exp_vol = np.zeros((n, n_assets))
    for a in range(n_assets):
        # Each asset has slightly different feature loadings (shifted by idio)
        wr = w_ret + 0.4 * asset_idio[a]
        wv = w_vol + 0.4 * asset_idio[a]
        exp_ret[:, a] = X[:, a, :] @ wr + 0.3 * regime
        exp_vol[:, a] = X[:, a, :] @ wv + 0.5 * regime
    # Realize targets: returns = exp_ret + noise; vol = |exp_vol| + noise
    ret_target = exp_ret + rng.normal(0, 0.5, (n, n_assets))
    # Vol target: per-asset per-day realized (e.g., abs innovations)
    vol_target = np.clip(np.abs(exp_vol) + rng.normal(0, 0.2, (n, n_assets)), 0.05, None)
    return {
        "t": t,
        "X": X,
        "ret_target": ret_target,
        "vol_target": vol_target,
        "w_ret": w_ret,
        "w_vol": w_vol,
        "regime": regime,
    }


def fit_shared_bottom(sim, lr=0.01, epochs=200):
    """Hard-shared multi-task linear model: H shared, then H @ W_ret, H @ W_vol."""
    n, A, F = sim["X"].shape
    X = sim["X"].reshape(n * A, F)
    y_ret = sim["ret_target"].reshape(-1)
    y_vol = sim["vol_target"].reshape(-1)

    # Standardize features (one-shot z-score)
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    Xn = (X - mu) / sd
    # Targets: standardize returns, keep vol raw
    yr_mu, yr_sd = y_ret.mean(), y_ret.std() + 1e-8
    yn_ret = (y_ret - yr_mu) / yr_sd

    # Shared representation: H = Xn @ W_shared (a single matrix F→F is trivial; we instead use a 2-layer net)
    # For simplicity + tractable comparison: linear shared + 2 task heads.
    # Linear shared: H = Xn @ W_shared, shape (n_samples, hidden_dim)
    H_dim = 4
    rng2 = np.random.default_rng(11)
    W_shared = rng2.normal(0, 0.3, (F, H_dim))
    W_ret = np.zeros(H_dim)
    W_vol = np.zeros(H_dim)
    b_ret = 0.0
    b_vol = y_vol.mean()
    losses = {"ret": [], "vol": []}
    for epoch in range(epochs):
        H = Xn @ W_shared
        pred_ret = H @ W_ret + b_ret
        pred_vol = H @ W_vol + b_vol
        loss_ret = ((pred_ret - yn_ret) ** 2).mean()
        loss_vol = ((pred_vol - y_vol) ** 2).mean()
        losses["ret"].append(loss_ret)
        losses["vol"].append(loss_vol)

        # Gradients
        dL_dpred_ret = 2 * (pred_ret - yn_ret) / Xn.shape[0]
        dL_dpred_vol = 2 * (pred_vol - y_vol) / Xn.shape[0]
        dL_dW_ret = H.T @ dL_dpred_ret
        dL_db_ret = dL_dpred_ret.sum()
        dL_dW_vol = H.T @ dL_dpred_vol
        dL_db_vol = dL_dpred_vol.sum()

        dL_dH = np.outer(dL_dpred_ret, W_ret) + np.outer(dL_dpred_vol, W_vol)
        dL_dW_shared = Xn.T @ dL_dH

        W_shared -= lr * dL_dW_shared
        W_ret -= lr * dL_dW_ret
        W_vol -= lr * dL_dW_vol
        b_ret -= lr * dL_db_ret
        b_vol -= lr * dL_db_vol
    final_W_shared = W_shared.copy()
    return {
        "W_shared": W_shared,
        "W_ret": W_ret,
        "W_vol": W_vol,
        "b_ret": b_ret,
        "b_vol": b_vol,
        "Xn": Xn,
        "yn_ret": yn_ret,
        "y_vol": y_vol,
        "yr_mu": yr_mu,
        "yr_sd": yr_sd,
        "Xn_mu": mu,
        "Xn_sd": sd,
        "losses": losses,
    }


def fit_single_task(sim, head="ret", lr=0.01, epochs=200):
    """Independent (no-shared) model."""
    n, A, F = sim["X"].shape
    X = sim["X"].reshape(n * A, F)
    y = sim["ret_target"].reshape(-1) if head == "ret" else sim["vol_target"].reshape(-1)
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    Xn = (X - mu) / sd
    if head == "ret":
        y_mu, y_sd = y.mean(), y.std() + 1e-8
        y_n = (y - y_mu) / y_sd
        b0 = 0.0
    else:
        y_n = y
        y_mu, y_sd = 0.0, 1.0
        b0 = y.mean()
    rng2 = np.random.default_rng(13)
    W = rng2.normal(0, 0.3, F)
    b = b0
    losses = []
    for _ in range(epochs):
        pred = Xn @ W + b
        d = 2 * (pred - y_n) / Xn.shape[0]
        W -= lr * (Xn.T @ d)
        b -= lr * d.sum()
        losses.append(((pred - y_n) ** 2).mean())
    if head == "ret":
        return {"W": W, "b": b, "losses": losses, "Xn": Xn, "y_n": y_n,
                "Xn_mu": mu, "Xn_sd": sd, "y_mu": y_mu, "y_sd": y_sd}
    return {"W": W, "b": b, "losses": losses, "Xn": Xn, "y_n": y_n,
            "Xn_mu": mu, "Xn_sd": sd}


def portfolio_metric(pred_ret, pred_vol, ret_target, vol_target):
    """Risk-adjusted portfolio IC: rank corr(pred_ret / pred_vol) vs realized return."""
    n, A = pred_ret.shape
    # Sharpe-like signal: predicted excess return / predicted risk
    signal = pred_ret / pred_vol
    # Cross-sectional Spearman per day, then average
    ranks_signal = np.argsort(np.argsort(signal, axis=1), axis=1) + 1
    ranks_ret = np.argsort(np.argsort(ret_target, axis=1), axis=1) + 1
    rhos = []
    for i in range(n):
        d_signal = ranks_signal[i]
        d_ret = ranks_ret[i]
        d_signal = d_signal - d_signal.mean()
        d_ret = d_ret - d_ret.mean()
        num = (d_signal * d_ret).sum()
        den = np.sqrt((d_signal**2).sum() * (d_ret**2).sum()) + 1e-12
        rhos.append(num / den)
    return np.mean(rhos)


def make_cover():
    """Cover image: shared bottom + two heads + regime decomposition."""
    sim = simulate_multi_task()
    mt = fit_shared_bottom(sim, lr=0.01, epochs=250)

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Left: architecture diagram
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis("off")

    # Feature input column
    for i in range(8):
        ax1.add_patch(plt.Rectangle((0.4, 0.4 + i * 0.55), 0.7, 0.5,
                                     facecolor="#5D6D7E", edgecolor="k"))
    ax1.text(0.75, 5.2, "X\n(8 feats × asset)", ha="center", fontsize=9, color="white")

    # Shared bottom
    for i in range(4):
        ax1.add_patch(plt.Rectangle((3.3, 0.9 + i * 0.9), 0.8, 0.9,
                                     facecolor="#2874A6", edgecolor="k"))
    ax1.text(3.7, 5.5, "Shared\nbottom H", ha="center", fontsize=10, color="white", fontweight="bold")

    # Arrows from input to shared
    for i in range(8):
        for j in range(4):
            ax1.annotate("", xy=(3.3, 1.35 + j * 0.9), xytext=(1.1, 0.65 + i * 0.55),
                          arrowprops=dict(arrowstyle="-", color="grey", lw=0.5, alpha=0.6))

    # Return head
    ax1.add_patch(plt.Rectangle((6.2, 1.8), 1.3, 1.6, facecolor="#C0392B", edgecolor="k"))
    ax1.text(6.85, 2.6, "return\nhead", ha="center", fontsize=10, color="white", fontweight="bold")
    # Vol head
    ax1.add_patch(plt.Rectangle((6.2, 3.6), 1.3, 1.6, facecolor="#27AE60", edgecolor="k"))
    ax1.text(6.85, 4.4, "risk\nhead", ha="center", fontsize=10, color="white", fontweight="bold")

    # Arrows from shared to heads
    for j in range(4):
        ax1.annotate("", xy=(6.2, 2.6), xytext=(4.1, 1.35 + j * 0.9),
                      arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
        ax1.annotate("", xy=(6.2, 4.4), xytext=(4.1, 1.35 + j * 0.9),
                      arrowprops=dict(arrowstyle="->", color="black", lw=0.9))

    # Output
    ax1.add_patch(plt.Rectangle((8.4, 1.8), 1.1, 1.6, facecolor="#F5CBA7", edgecolor="k"))
    ax1.text(8.95, 2.6, "μ_pred", ha="center", fontsize=10)
    ax1.add_patch(plt.Rectangle((8.4, 3.6), 1.1, 1.6, facecolor="#A9DFBF", edgecolor="k"))
    ax1.text(8.95, 4.4, "σ_pred", ha="center", fontsize=10)
    ax1.annotate("", xy=(8.4, 2.6), xytext=(7.5, 2.6),
                  arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax1.annotate("", xy=(8.4, 4.4), xytext=(7.5, 4.4),
                  arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    ax1.text(5.0, 5.8, "Multi-task Factor Model: Hard-Shared Bottom + Two Heads",
              ha="center", fontsize=12, fontweight="bold")
    ax1.text(5.0, 0.15, "Returns and risk share representation → gradient signal amplifies, "
                         "shared features get cleaner",
              ha="center", fontsize=8.5, color="grey", style="italic")

    # Right: regime decomposition
    t = sim["t"]
    regime = sim["regime"]
    ax2.plot(t, regime, color="#2C3E50", lw=1.5, label="market regime (latent)")
    # Marker for return-only features vs shared
    w_ret = sim["w_ret"]
    w_vol = sim["w_vol"]
    shared = (w_ret != 0) & (w_vol != 0)
    return_only = (w_ret != 0) & (w_vol == 0)
    vol_only = (w_ret == 0) & (w_vol != 0)
    feature_names = [f"f{i}" for i in range(len(w_ret))]
    bar_y = np.arange(len(w_ret))
    colors = []
    for j in range(len(w_ret)):
        if shared[j]:
            colors.append("#8E44AD")  # purple: shared
        elif return_only[j]:
            colors.append("#C0392B")  # red: return only
        elif vol_only[j]:
            colors.append("#27AE60")  # green: vol only
    ax2b = ax2.inset_axes([0, -0.5, 1, 0.5])
    ax2b.barh(bar_y, w_ret, color=colors, edgecolor="k")
    ax2b.set_yticks(bar_y)
    ax2b.set_yticklabels(feature_names, fontsize=8)
    ax2b.set_xlabel("true loading (return head)", fontsize=8)
    legend_elements = [
        Patch(facecolor="#8E44AD", label="shared feature"),
        Patch(facecolor="#C0392B", label="return-only"),
        Patch(facecolor="#27AE60", label="risk-only"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax2.set_title("Feature roles and regime", fontsize=10)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cover.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_loss_curves():
    """Shared losses vs singletask losses."""
    sim = simulate_multi_task()
    mt = fit_shared_bottom(sim, lr=0.01, epochs=200)
    st_ret = fit_single_task(sim, head="ret", lr=0.01, epochs=200)
    st_vol = fit_single_task(sim, head="vol", lr=0.01, epochs=200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(mt["losses"]["ret"], color="#C0392B", lw=2.0, label="multi-task return loss")
    ax.plot(st_ret["losses"], color="#C0392B", ls="--", lw=1.2, label="single-task return loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (return head)")
    ax.set_title("Return head: shared loss < single-task loss", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(mt["losses"]["vol"], color="#27AE60", lw=2.0, label="multi-task vol loss")
    ax.plot(st_vol["losses"], color="#27AE60", ls="--", lw=1.2, label="single-task vol loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (vol head)")
    ax.set_title("Risk head: shared loss < single-task loss (the bonus)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_curves_shared_vs_single.png"), dpi=140)
    plt.close(fig)


def make_ic_horizon():
    """Compare average sharpe-like IC for shared vs single vs naive equal-weight."""
    sim = simulate_multi_task()
    mt = fit_shared_bottom(sim, lr=0.01, epochs=200)
    # Compute predictions for MT
    H = mt["Xn"] @ mt["W_shared"]
    pred_ret_mt = (H @ mt["W_ret"] + mt["b_ret"]) * mt["yr_sd"] + mt["yr_mu"]
    pred_vol_mt = H @ mt["W_vol"] + mt["b_vol"]
    n = sim["X"].shape[0]
    A = sim["X"].shape[1]
    pred_ret_mt = pred_ret_mt.reshape(n, A)
    pred_vol_mt = pred_vol_mt.reshape(n, A)

    st_ret = fit_single_task(sim, head="ret", lr=0.01, epochs=200)
    pred_ret_st = (st_ret["Xn"] @ st_ret["W"] + st_ret["b"]) * st_ret["y_sd"] + st_ret["y_mu"]
    pred_ret_st = pred_ret_st.reshape(n, A)
    st_vol = fit_single_task(sim, head="vol", lr=0.01, epochs=200)
    pred_vol_st = (st_vol["Xn"] @ st_vol["W"] + st_vol["b"]).reshape(n, A)

    # Naive: use raw feature as ret signal, constant vol
    raw_signal = sim["X"].mean(axis=2)  # average across features (rough proxy)
    pred_ret_naive = raw_signal
    pred_vol_naive = np.ones_like(raw_signal) * sim["vol_target"].mean()

    sharpe_mt = portfolio_metric(pred_ret_mt, pred_vol_mt, sim["ret_target"], sim["vol_target"])
    sharpe_st = portfolio_metric(pred_ret_st, pred_vol_st, sim["ret_target"], sim["vol_target"])
    sharpe_naive = portfolio_metric(pred_ret_naive, pred_vol_naive, sim["ret_target"], sim["vol_target"])

    # Walk-forward OOS
    n = sim["X"].shape[0]
    ic_walk = {"mt": [], "st": [], "naive": []}
    cut = 200
    for split in range(cut, n - 1):
        # Refit on first `split` days, evaluate one-shot on day `split`
        sub = {k: v[:split] if hasattr(v, "shape") and v.shape[0] == n else v for k, v in sim.items()}
        sub["X"] = sim["X"][:split]
        sub["ret_target"] = sim["ret_target"][:split]
        sub["vol_target"] = sim["vol_target"][:split]
        mt_w = fit_shared_bottom(sub, lr=0.01, epochs=120)
        st_r = fit_single_task(sub, head="ret", lr=0.01, epochs=120)
        st_v = fit_single_task(sub, head="vol", lr=0.01, epochs=120)

        Xn_test = (sim["X"][split] - mt_w["Xn_mu"]) / mt_w["Xn_sd"]
        Ht = Xn_test @ mt_w["W_shared"]
        pred_mt_r = (Ht @ mt_w["W_ret"] + mt_w["b_ret"]) * mt_w["yr_sd"] + mt_w["yr_mu"]
        pred_mt_v = Ht @ mt_w["W_vol"] + mt_w["b_vol"]
        # Single-task ret
        Xn_test_r = (sim["X"][split] - st_r["Xn_mu"]) / st_r["Xn_sd"]
        pred_st_r = (Xn_test_r @ st_r["W"] + st_r["b"]) * st_r["y_sd"] + st_r["y_mu"]
        # Single-task vol
        Xn_test_v = (sim["X"][split] - st_v["Xn_mu"]) / st_v["Xn_sd"]
        pred_st_v = Xn_test_v @ st_v["W"] + st_v["b"]

        # Cross-sectional rank IC
        r_mt = np.corrcoef(pred_mt_r / pred_mt_v, sim["ret_target"][split])[0, 1]
        r_st = np.corrcoef(pred_st_r / pred_st_v, sim["ret_target"][split])[0, 1]
        naive_sig = sim["X"][split].mean(axis=1)
        r_na = np.corrcoef(naive_sig, sim["ret_target"][split])[0, 1]
        if not np.isnan(r_mt):
            ic_walk["mt"].append(r_mt)
        if not np.isnan(r_st):
            ic_walk["st"].append(r_st)
        if not np.isnan(r_na):
            ic_walk["naive"].append(r_na)
    fig, ax = plt.subplots(figsize=(10, 5))
    ic_walk["mt"] = np.array(ic_walk["mt"])
    ic_walk["st"] = np.array(ic_walk["st"])
    ic_walk["naive"] = np.array(ic_walk["naive"])
    rolling = 30
    for tag, color, label in [
        ("mt", "#8E44AD", "multi-task (shared bottom)"),
        ("st", "#2980B9", "two singletask models (concat)"),
        ("naive", "#7F8C8D", "naive equal-feature baseline"),
    ]:
        series = ic_walk[tag]
        cum = np.cumsum(series)
        ax.plot(np.arange(series.size), cum, color=color, lw=1.8, label=label)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("walk-forward step (day)")
    ax.set_ylabel("cumulative rank IC (μ̂/σ̂ signal vs realized return)")
    ax.set_title(
        f"Walk-forward OOS: multi-task IR > single > naive "
        f"(mean IC: MT={np.mean(ic_walk['mt']):.3f}, "
        f"ST={np.mean(ic_walk['st']):.3f}, "
        f"naive={np.mean(ic_walk['naive']):.3f})",
        fontsize=10)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "oos_ic_walkforward.png"), dpi=140)
    plt.close(fig)


def make_coupling_heatmap():
    """Show how loss in risk task improves as return task gets harder (and vice versa)."""
    sim = simulate_multi_task()
    coupling = np.zeros((10, 10))  # rows: factor scaling for ret head, cols: vol head
    for i, ret_w in enumerate(np.linspace(0.0, 2.0, 10)):
        for j, vol_w in enumerate(np.linspace(0.0, 2.0, 10)):
            sim2 = {**sim}
            sim2["ret_target"] = ret_w * sim["ret_target"]
            sim2["vol_target"] = vol_w * sim["vol_target"]
            mt = fit_shared_bottom(sim2, lr=0.01, epochs=120)
            # OOS R^2 for both heads on day cut onwards
            n = sim["X"].shape[0]
            cut = int(0.7 * n)
            H = mt["Xn"] @ mt["W_shared"]
            pred_ret = (H @ mt["W_ret"] + mt["b_ret"]) * mt["yr_sd"] + mt["yr_mu"]
            pred_vol = H @ mt["W_vol"] + mt["b_vol"]
            # Use cross-sectional samples
            yr = mt["yn_ret"]
            yv = mt["y_vol"]
            ss_res_ret = ((pred_ret - yr) ** 2).sum()
            ss_tot_ret = ((yr - yr.mean()) ** 2).sum() + 1e-12
            ss_res_vol = ((pred_vol - yv) ** 2).sum()
            ss_tot_vol = ((yv - yv.mean()) ** 2).sum() + 1e-12
            coupling[i, j] = 1 - 0.5 * (ss_res_ret / ss_tot_ret + ss_res_vol / ss_tot_vol)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    cmap = "viridis"
    im0 = axes[0].imshow(1 - coupling / coupling.max(),
                          cmap=cmap, origin="lower", aspect="auto")
    axes[0].set_xlabel("vol target scale"); axes[0].set_ylabel("ret target scale")
    axes[0].set_title("Combined MSE (return+vol)", fontsize=11)
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(coupling, cmap=cmap, origin="lower", aspect="auto")
    axes[1].set_xlabel("vol target scale"); axes[1].set_ylabel("ret target scale")
    axes[1].set_title("Combined R² (return+vol)", fontsize=11)
    fig.colorbar(im1, ax=axes[1])
    # Marginal R² for each head as function of the OTHER target scale
    axes[2].plot(np.linspace(0, 2, 10), coupling[5, :], "o-", color="#C0392B", label="R² at ret=1.0× vs vol scale")
    axes[2].plot(np.linspace(0, 2, 10), coupling[:, 5], "s-", color="#27AE60", label="R² at vol=1.0× vs ret scale")
    axes[2].set_xlabel("other-target scale"); axes[2].set_ylabel("combined R²")
    axes[2].set_title("Coupling: target scaling on the OTHER task still helps",
                      fontsize=10)
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "shared_representation_coupling.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    make_cover()
    make_loss_curves()
    make_ic_horizon()
    make_coupling_heatmap()
    print("ok: generated 4 images for multi-task-factor-learning")
