"""Generate images for online-concept-drift-adapt article."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False

OUT_DIR = "/Users/halo/workspace/astro-blog/public/images/online-concept-drift-adapt"
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)


def simulate_drift_series(n=600, n_factors=3):
    """Simulate 3 factors with drifting IC over time, returns dict."""
    t = np.arange(n)
    # Effective IC of each factor decays starting at different points
    drift_onset = np.array([120, 280, 420])
    # Different decay speeds
    decay = np.array([0.012, 0.008, 0.020])
    base_ic = np.array([0.08, 0.06, 0.05])
    ics = np.zeros((n, n_factors))
    for j in range(n_factors):
        active = t >= drift_onset[j]
        ics[:, j] = base_ic[j] * np.exp(-decay[j] * np.maximum(0, t - drift_onset[j]))
        # Add small noise
        ics[:, j] += rng.normal(0, 0.005, n)
    # Returns
    returns = ics.mean(axis=1) * 1.0 + rng.normal(0, 0.02, n)
    # Raw factor IC observation (rolling 60-day)
    raw_ic = np.zeros((n, n_factors))
    for j in range(n_factors):
        # Sample-based rolling correlation
        f_signal = ics[:, j] + rng.normal(0, 0.3, n)
        for i in range(60, n):
            window = slice(i - 60, i)
            corr = np.corrcoef(f_signal[window], returns[window])[0, 1]
            raw_ic[i, j] = corr if not np.isnan(corr) else 0
    return {
        "t": t,
        "true_ic": ics,
        "raw_ic": raw_ic,
        "drift_onset": drift_onset,
        "decay": decay,
        "returns": returns,
        "names": ["momentum", "value", "liquidity"],
    }


def run_adp_oracle(raw_ic, true_ic, drift_onset, decay, lr=0.02):
    """Run online-learning weight adaptation with Page-Hinkley style drift detector."""
    n, k = raw_ic.shape
    weights = np.ones(k) / k
    weight_hist = np.zeros((n, k))
    cumsum_drift = np.zeros(k)
    threshold = 0.05
    detected = np.full(k, np.nan)
    # PH detection state
    mean = np.zeros(k)
    cum_dev = np.zeros(k)
    drift_score = np.zeros((n, k))
    for i in range(n):
        if i < 60:
            weight_hist[i] = weights
            continue
        # Update per-factor IC sample (use most recent rolling)
        ic_now = raw_ic[i]
        # PH update
        for j in range(k):
            if np.isnan(detected[j]):
                mean[j] = 0.95 * mean[j] + 0.05 * ic_now[j]
                cum_dev[j] = max(0, cum_dev[j] + ic_now[j] - mean[j] - 0.001)
                if cum_dev[j] > threshold and i > 100:
                    detected[j] = i
        # Weight adaptation: lower weight if observed IC drops
        for j in range(k):
            target = max(ic_now[j], 0.0)
            weights[j] += lr * (target - weights[j])
        # Renormalize + small floor
        weights = np.maximum(weights, 0.02)
        weights /= weights.sum()
        weight_hist[i] = weights
    return weights, weight_hist, detected


def run_static(raw_ic, n):
    """Static equal-weighted baseline."""
    n_, k = raw_ic.shape
    w = np.ones(k) / k
    hist = np.tile(w, (n_, 1))
    return w, hist


def make_cover():
    """Cover image: factor weight drift timeline + drift detection."""
    sim = simulate_drift_series()
    weights, wh, detected = run_adp_oracle(sim["raw_ic"], sim["true_ic"], sim["drift_onset"], sim["decay"])
    n = sim["t"].size
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for j in range(3):
        c = cmap(j)
        ax.plot(sim["t"], wh[:, j], color=c, lw=2.0, label=f"w[{sim['names'][j]}]")
        # Mark detection time
        if not np.isnan(detected[j]):
            ax.axvline(detected[j], color=c, ls="--", alpha=0.5, lw=1.2)
            ax.annotate(f"detect@{int(detected[j])}",
                        xy=(detected[j], wh[int(detected[j]), j]),
                        xytext=(detected[j]+20, 0.55-j*0.12),
                        fontsize=8, color=c)
    ax.set_ylabel("adaptive weight")
    ax.set_title("Adaptive Reweighting under Concept Drift (3 factors, drift onsets at 120/280/420)",
                 fontsize=11)
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, n)

    ax = axes[1]
    ax.plot(sim["t"], sim["true_ic"][:, 0], color=cmap(0), lw=1.5, label="momentum true IC")
    ax.plot(sim["t"], sim["true_ic"][:, 1], color=cmap(1), lw=1.5, label="value true IC")
    ax.plot(sim["t"], sim["true_ic"][:, 2], color=cmap(2), lw=1.5, label="liquidity true IC")
    ax.set_xlabel("day")
    ax.set_ylabel("true factor IC")
    ax.set_xlim(0, n)
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cover.png"), dpi=140)
    plt.close(fig)


def make_adaptive_learning_rate():
    """Compare fixed lr vs adaptive lr under drift."""
    sim = simulate_drift_series()
    n = sim["t"].size
    results = {}
    for tag, lr in [("static=1/3", None), ("lr=0.005", 0.005), ("lr=0.02", 0.02), ("lr=0.05", 0.05)]:
        if lr is None:
            w, h = run_static(sim["raw_ic"], n)
        else:
            w, h, _ = run_adp_oracle(sim["raw_ic"], sim["true_ic"], sim["drift_onset"], sim["decay"], lr=lr)
        # Compute rolling portfolio IC (weight @ factor IC vectors)
        port_ic = (h * sim["true_ic"]).sum(axis=1)
        # Cumulative IC excess over static
        cum = np.cumsum(port_ic[60:])
        results[tag] = cum
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("viridis")
    for i, (tag, series) in enumerate(results.items()):
        ax.plot(np.arange(series.size) + 60, series, color=cmap(i / max(len(results) - 1, 1)), lw=1.8, label=tag)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("day")
    ax.set_ylabel("cumulative portfolio IC sum")
    ax.set_title("Adaptive Learning Rate Ablation: how fast should we reweight when IC drifts?",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "adaptive_lr_ablation.png"), dpi=140)
    plt.close(fig)


def make_detection_delay_vs_threshold():
    """Sweep PH threshold → plot detection delay vs false alarm rate."""
    sim = simulate_drift_series()
    n = sim["t"].size
    true_drift_onset = sim["drift_onset"]
    true_decay = sim["decay"]
    raw_ic = sim["raw_ic"]

    thresholds = np.linspace(0.01, 0.20, 30)
    delays = []
    false_alarms = []
    for thr in thresholds:
        delays_t = []
        false_alarm_count = 0
        for j in range(3):
            detected = np.nan
            mean = 0.0
            cum_dev = 0.0
            for i in range(60, n):
                ic = raw_ic[i, j]
                mean = 0.95 * mean + 0.05 * ic
                cum_dev = max(0, cum_dev + ic - mean - 0.001)
                if np.isnan(detected) and cum_dev > thr and i > 100:
                    detected = i
                elif not np.isnan(detected):
                    pass
            if np.isnan(detected):
                delays_t.append(n)  # never detected
            else:
                delays_t.append(max(detected - true_drift_onset[j], 0))
            # False alarm: detected before true onset
            if not np.isnan(detected) and detected < true_drift_onset[j]:
                false_alarm_count += 1
        delays.append(np.mean(delays_t))
        false_alarms.append(false_alarm_count / 3)
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(thresholds, delays, c=false_alarms, cmap="YlOrRd", s=60, edgecolor="k")
    ax.set_xlabel("Page-Hinkley threshold")
    ax.set_ylabel("avg detection delay (days)")
    ax.set_title("Drift detection: delay vs false-alarm rate trade-off (averaged over 3 factors)",
                 fontsize=11)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("false-alarm rate")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "detection_delay_tradeoff.png"), dpi=140)
    plt.close(fig)


def make_rolling_portfolio_ic():
    """Show adaptive weights minimize drawdown vs static after drift onset."""
    sim = simulate_drift_series()
    n = sim["t"].size
    w_static, h_static = run_static(sim["raw_ic"], n)
    w_adp, h_adp, _ = run_adp_oracle(sim["raw_ic"], sim["true_ic"], sim["drift_onset"], sim["decay"], lr=0.02)
    port_ic_static = (h_static * sim["true_ic"]).sum(axis=1)
    port_ic_adp = (h_adp * sim["true_ic"]).sum(axis=1)
    cum_static = np.cumsum(port_ic_static)
    cum_adp = np.cumsum(port_ic_adp)
    # Drift regions
    drift_onset = sim["drift_onset"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(sim["t"], cum_static, color="#888", lw=2.0, label="equal-weight baseline (cumulative IC)")
    ax.plot(sim["t"], cum_adp, color="#c0392b", lw=2.0, label="adaptive reweighting (cumulative IC)")
    for d in drift_onset:
        ax.axvline(d, color="orange", ls="--", alpha=0.6)
        ax.text(d+5, ax.get_ylim()[1]*0.92, f"drift@{d}", fontsize=8, color="orange")
    ax.set_ylabel("cumulative IC")
    ax.set_title("Cumulative portfolio IC: adaptive reweighting recovers after each drift onset",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    ax = axes[1]
    # Daily PnL delta
    diff = port_ic_adp - port_ic_static
    ax.bar(sim["t"], diff, color=np.where(diff >= 0, "#27ae60", "#c0392b"), width=1.0)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("day")
    ax.set_ylabel("Δ IC (adaptive - static)")
    ax.set_xlim(0, n)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rolling_ic_comparison.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    make_cover()
    make_adaptive_learning_rate()
    make_detection_delay_vs_threshold()
    make_rolling_portfolio_ic()
    print("ok: generated 4 images for online-concept-drift-adapt")
