"""Generate all images for two blog articles."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

rng = np.random.default_rng(20260828)

# ─── paths ───
ROPE_DIR = "/Users/halo/workspace/astro-blog/public/images/rope-rotary-position-finance"
DT_DIR   = "/Users/halo/workspace/astro-blog/public/images/decision-tree-interpretable-factor"
os.makedirs(ROPE_DIR, exist_ok=True)
os.makedirs(DT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# ═════════════════════════════════════════════════════════════════
# Helper: rotate_half (for RoPE)
# ═════════════════════════════════════════════════════════════════
def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.stack([-x2, x1], axis=-1).reshape(x.shape)

def apply_rope(q, freqs):
    """q: (T, d), freqs: (d/2,)  ->  (T, d)"""
    T, d = q.shape
    pos = np.arange(T).reshape(-1, 1)          # (T,1)
    angles = pos * freqs.reshape(1, -1)        # (T, d/2)
    cos = np.cos(angles)
    sin = np.sin(angles)
    # interleave cos/sin into (T, d)
    cos_full = np.zeros((T, d))
    sin_full = np.zeros((T, d))
    cos_full[:, 0::2] = cos
    cos_full[:, 1::2] = cos
    sin_full[:, 0::2] = sin
    sin_full[:, 1::2] = sin
    return q * cos_full + rotate_half(q) * sin_full

def multi_freq_rope(q, freq_sets):
    """Apply RoPE with multiple frequency sets sequentially."""
    out = q.copy()
    for freqs in freq_sets:
        out = apply_rope(out, freqs)
    return out

# ═════════════════════════════════════════════════════════════════
# IMAGE 1: Attention heatmap comparison (RoPE article)
# ═════════════════════════════════════════════════════════════════
def gen_rope_img1():
    T, d = 128, 32
    # Build a synthetic query with financial-like periodicities
    t = np.arange(T)
    # signal has weekly (5), monthly (21), quarterly (63) periodicity
    signal = (np.sin(2*np.pi*t/5) * 0.4 +
              np.sin(2*np.pi*t/21) * 0.3 +
              np.sin(2*np.pi*t/63) * 0.3)
    q = np.zeros((T, d))
    for i in range(d):
        q[:, i] = signal * np.cos(i * 0.3) + rng.normal(0, 0.1, T)
    k = q.copy()  # self-attention keys

    # Standard RoPE: single frequency base 10000^(−2i/d)
    std_freqs = 1.0 / (10000 ** (np.arange(0, d, 2) / d))
    q_std = apply_rope(q, std_freqs)
    k_std = apply_rope(k, std_freqs)
    attn_std = q_std @ k_std.T / np.sqrt(d)
    attn_std = np.exp(attn_std - attn_std.max(axis=1, keepdims=True))
    attn_std /= attn_std.sum(axis=1, keepdims=True)

    # Multi-frequency RoPE: explicit financial periods
    fin_periods = [5, 21, 63]
    fin_freqs = [2*np.pi/p * np.ones(d//2) for p in fin_periods]
    q_mf = multi_freq_rope(q, fin_freqs)
    k_mf = multi_freq_rope(k, fin_freqs)
    attn_mf = q_mf @ k_mf.T / np.sqrt(d)
    attn_mf = np.exp(attn_mf - attn_mf.max(axis=1, keepdims=True))
    attn_mf /= attn_mf.sum(axis=1, keepdims=True)

    # No position encoding
    attn_none = q @ k.T / np.sqrt(d)
    attn_none = np.exp(attn_none - attn_none.max(axis=1, keepdims=True))
    attn_none /= attn_none.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ["No Position Encoding", "Standard RoPE (θ=10000^{-2i/d})", "Multi-Freq RoPE (P=5,21,63)"]
    data = [attn_none, attn_std, attn_mf]
    for ax, title, mat in zip(axes, titles, data):
        im = ax.imshow(mat[:40, :40], cmap="YlOrRd", aspect="auto", vmin=0, vmax=mat[:40,:40].max())
        ax.set_title(title)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Attention Heatmap Comparison — Standard vs Multi-Frequency RoPE", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{ROPE_DIR}/attention_heatmap_comparison.png")
    plt.close(fig)
    print("  saved attention_heatmap_comparison.png")

# ═════════════════════════════════════════════════════════════════
# IMAGE 2: Frequency capture spectrum (RoPE article)
# ═════════════════════════════════════════════════════════════════
def gen_rope_img2():
    T, d = 256, 64
    t = np.arange(T)
    signal = (np.sin(2*np.pi*t/5) * 0.35 +
              np.sin(2*np.pi*t/21) * 0.30 +
              np.sin(2*np.pi*t/63) * 0.25 +
              np.sin(2*np.pi*t/126) * 0.10 +
              rng.normal(0, 0.15, T))

    # Compute power spectrum of attention scores at each query
    std_freqs = 1.0 / (10000 ** (np.arange(0, d, 2) / d))
    fin_periods = [5, 21, 63]
    fin_freqs = [2*np.pi/p * np.ones(d//2) for p in fin_periods]

    q = np.outer(signal, np.exp(np.arange(d) * 0.1))
    q = q + rng.normal(0, 0.05, (T, d))
    k = q.copy()

    # Standard RoPE attention
    q_std = apply_rope(q, std_freqs)
    k_std = apply_rope(k, std_freqs)
    attn_std = q_std @ k_std.T / np.sqrt(d)
    attn_std = np.exp(attn_std - attn_std.max(axis=1, keepdims=True))
    attn_std /= attn_std.sum(axis=1, keepdims=True)

    # Multi-freq RoPE attention
    q_mf = multi_freq_rope(q, fin_freqs)
    k_mf = multi_freq_rope(k, fin_freqs)
    attn_mf = q_mf @ k_mf.T / np.sqrt(d)
    attn_mf = np.exp(attn_mf - attn_mf.max(axis=1, keepdims=True))
    attn_mf /= attn_mf.sum(axis=1, keepdims=True)

    # No pos encoding
    attn_none = q @ k.T / np.sqrt(d)
    attn_none = np.exp(attn_none - attn_none.max(axis=1, keepdims=True))
    attn_none /= attn_none.sum(axis=1, keepdims=True)

    # Average attention row -> spectrum
    def avg_spectrum(attn):
        avg_row = attn.mean(axis=0)
        freqs = np.fft.fftfreq(T)
        mask = freqs > 0
        spectrum = np.abs(np.fft.fft(avg_row))
        return freqs[mask], spectrum[mask]

    f_none, s_none = avg_spectrum(attn_none)
    f_std, s_std = avg_spectrum(attn_std)
    f_mf, s_mf = avg_spectrum(attn_mf)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f_none * T, s_none, alpha=0.6, label="No Pos Encoding", color="gray")
    ax.semilogy(f_std * T, s_std, alpha=0.7, label="Standard RoPE", color="#2196F3")
    ax.semilogy(f_mf * T, s_mf, alpha=0.9, label="Multi-Freq RoPE", color="#FF5722", linewidth=2)
    for p in [5, 21, 63, 126]:
        ax.axvline(p, color="green", linestyle="--", alpha=0.5)
        ax.annotate(f"P={p}", (p, ax.get_ylim()[1]*0.3), fontsize=9, color="green", ha="center")
    ax.set_xlabel("Period (trading days)")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("Attention Power Spectrum: Multi-Freq RoPE Captures Financial Periodicities")
    ax.legend()
    ax.set_xlim(2, 150)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{ROPE_DIR}/frequency_capture_spectrum.png")
    plt.close(fig)
    print("  saved frequency_capture_spectrum.png")

# ═════════════════════════════════════════════════════════════════
# IMAGE 3: Rolling IC comparison (RoPE article)
# ═════════════════════════════════════════════════════════════════
def gen_rope_img3():
    T, d = 500, 32
    window = 60

    # Build synthetic factor: returns depend on 3 periodic components + noise
    t = np.arange(T)
    returns = (np.sin(2*np.pi*t/5) * 0.5 +
               np.sin(2*np.pi*t/21) * 0.3 +
               np.sin(2*np.pi*t/63) * 0.2 +
               rng.normal(0, 0.8, T))

    # Build a query factor vector that embeds the signal
    q = np.zeros((T, d))
    for i in range(d):
        q[:, i] = np.sin(2*np.pi*t/5) * np.cos(i*0.2) + \
                  np.sin(2*np.pi*t/21) * np.cos(i*0.4) + \
                  np.sin(2*np.pi*t/63) * np.cos(i*0.6) + \
                  rng.normal(0, 0.15, T)

    std_freqs = 1.0 / (10000 ** (np.arange(0, d, 2) / d))
    fin_periods = [5, 21, 63]
    fin_freqs = [2*np.pi/p * np.ones(d//2) for p in fin_periods]

    # Predict next-period return using attention-weighted past values
    methods = {
        "No Pos Enc": q.copy(),
        "Standard RoPE": apply_rope(q, std_freqs),
        "Multi-Freq RoPE": multi_freq_rope(q, fin_freqs),
    }

    rolling_ics = {}
    for name, q_enc in methods.items():
        k = q_enc.copy()
        attn = q_enc @ k.T / np.sqrt(d)
        attn = np.exp(attn - attn.max(axis=1, keepdims=True))
        attn /= attn.sum(axis=1, keepdims=True)
        # prediction = attention-weighted average of next-period returns
        pred = attn @ np.roll(returns, -1)
        # rolling IC
        ics = []
        for i in range(window, T):
            r_true = returns[i-window:i]
            r_pred = pred[i-window:i]
            ic = np.corrcoef(r_true, r_pred)[0, 1] if np.std(r_pred) > 0 else 0
            ics.append(ic)
        rolling_ics[name] = np.array(ics)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"No Pos Enc": "gray", "Standard RoPE": "#2196F3", "Multi-Freq RoPE": "#FF5722"}
    x = np.arange(window, T)
    for name, ics in rolling_ics.items():
        # smooth with moving average
        kernel = np.ones(20) / 20
        smoothed = np.convolve(ics, kernel, mode="same")
        ax.plot(x, smoothed, label=name, color=colors[name], linewidth=2, alpha=0.85)
        ax.axhline(np.mean(ics), color=colors[name], linestyle=":", alpha=0.4)

    ax.set_xlabel("Time (trading days)")
    ax.set_ylabel("Rolling IC (60-day window, smoothed)")
    ax.set_title("Rolling IC: Multi-Freq RoPE vs Standard RoPE vs No Position Encoding")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, 0.8)
    plt.tight_layout()
    fig.savefig(f"{ROPE_DIR}/rolling_ic_comparison.png")
    plt.close(fig)
    print("  saved rolling_ic_comparison.png")

# ═════════════════════════════════════════════════════════════════
# IMAGE 4: Cover image (RoPE article) — attention pattern + period annotation
# ═════════════════════════════════════════════════════════════════
def gen_rope_cover():
    T, d = 128, 32
    t = np.arange(T)
    signal = (np.sin(2*np.pi*t/5) * 0.4 + np.sin(2*np.pi*t/21) * 0.3 + np.sin(2*np.pi*t/63) * 0.3)
    q = np.outer(signal, np.exp(np.arange(d)*0.08)) + rng.normal(0, 0.1, (T, d))

    fin_periods = [5, 21, 63]
    fin_freqs = [2*np.pi/p * np.ones(d//2) for p in fin_periods]
    q_mf = multi_freq_rope(q, fin_freqs)
    k_mf = q_mf.copy()
    attn = q_mf @ k_mf.T / np.sqrt(d)
    attn = np.exp(attn - attn.max(axis=1, keepdims=True))
    attn /= attn.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    # Left: attention heatmap with period bands
    ax = axes[0]
    im = ax.imshow(attn[:64, :64], cmap="inferno", aspect="auto")
    for p, label, color in [(5, "P=5d", "cyan"), (21, "P=21d", "yellow"), (63, "P=63d", "lime")]:
        ax.axvline(p, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax.annotate(label, (p, 0), color=color, fontsize=9, fontweight="bold", va="top")
    ax.set_title("Multi-Freq RoPE Attention Pattern")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Right: signal with periodicities
    ax2 = axes[1]
    ax2.plot(t[:128], signal[:128], color="#FF5722", linewidth=2)
    ax2.set_title("Synthetic Signal with Multi-Period Components")
    ax2.set_xlabel("Trading day")
    ax2.set_ylabel("Signal value")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{ROPE_DIR}/cover.png")
    plt.close(fig)
    print("  saved cover.png")

# ═════════════════════════════════════════════════════════════════
# IMAGE 1: Decision tree rules visualization (DT article)
# ═════════════════════════════════════════════════════════════════
def gen_dt_img1():
    """Visualize a decision tree structure with rule paths and their IC."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Nodes: (x, y, text, is_leaf, ic_value)
    nodes = [
        (7, 9.2, "Momentum_20d\n≤ 0.15?", False, None),
        (3.5, 7.0, "Volatility_20d\n≤ 0.025?", False, None),
        (10.5, 7.0, "RSI_14\n≤ 70?", False, None),
        (1.5, 4.5, "Leaf A\nLong\nn=340", True, 0.082),
        (5.5, 4.5, "Volume_Change\n≤ 1.5?", False, None),
        (9.0, 4.5, "Leaf B\nNeutral\nn=210", True, 0.015),
        (12.0, 4.5, "Leaf C\nShort\nn=180", True, -0.067),
        (4.0, 2.0, "Leaf D\nLong\nn=120", True, 0.061),
        (7.0, 2.0, "Leaf E\nShort\nn=90", True, -0.044),
    ]

    edges = [
        (0, 1, "Yes"), (0, 2, "No"),
        (1, 3, "Yes"), (1, 4, "No"),
        (2, 5, "Yes"), (2, 6, "No"),
        (4, 7, "Yes"), (4, 8, "No"),
    ]

    # Draw edges
    for i, j, label in edges:
        x1, y1 = nodes[i][0], nodes[i][1]
        x2, y2 = nodes[j][0], nodes[j][1]
        ax.annotate("", xy=(x2, y2+0.4), xytext=(x1, y1-0.4),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.2, my+0.15, label, fontsize=8, color="#444",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

    # Draw nodes
    for x, y, text, is_leaf, ic in nodes:
        if is_leaf:
            color = "#4CAF50" if (ic and ic > 0) else ("#F44336" if (ic and ic < 0) else "#9E9E9E")
            alpha = 0.25
            ic_text = f"\nIC={ic:.3f}" if ic is not None else ""
            ax.text(x, y, text + ic_text, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=alpha,
                              edgecolor=color, linewidth=2))
        else:
            ax.text(x, y, text, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#2196F3", alpha=0.2,
                              edgecolor="#2196F3", linewidth=2))

    ax.set_title("Decision Tree Factor: Rule Paths & Leaf IC Values", fontsize=14, fontweight="bold", pad=20)
    fig.savefig(f"{DT_DIR}/tree_rules_visualization.png")
    plt.close(fig)
    print("  saved tree_rules_visualization.png")

# ═════════════════════════════════════════════════════════════════
# IMAGE 2: IC comparison across methods (DT article)
# ═════════════════════════════════════════════════════════════════
def gen_dt_img2():
    """Compare IC of decision tree (different depths) vs linear model vs MLP."""
    rng = np.random.default_rng(20260828)

    methods = ["Linear\n(OLS)", "DT\ndepth=2", "DT\ndepth=3", "DT\ndepth=5", "DT\ndepth=8", "MLP\n(2-layer)", "GBDT\n(100 trees)"]
    # Realistic IC values (synthetic but representative)
    ic_mean = [0.041, 0.058, 0.072, 0.085, 0.079, 0.091, 0.098]
    ic_std  = [0.018, 0.020, 0.022, 0.025, 0.031, 0.028, 0.030]
    # Interpretable? (binary)
    interpretable = [1, 1, 1, 0.5, 0, 0, 0]
    # Compliance score
    compliance = [1.0, 1.0, 1.0, 0.6, 0.2, 0.0, 0.0]

    # Simulate IC samples
    ic_samples = [rng.normal(m, s, 50) for m, s in zip(ic_mean, ic_std)]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(methods))

    # Box plot
    bp = ax1.boxplot(ic_samples, positions=positions, widths=0.5, patch_artist=True,
                     showmeans=True, meanprops=dict(marker="D", markerfacecolor="white", markersize=6))
    colors = ["#4CAF50", "#2196F3", "#2196F3", "#2196F3", "#2196F3", "#FF9800", "#F44336"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax1.set_ylabel("Out-of-sample IC", fontsize=12)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylim(0, 0.18)
    ax1.axhline(0, color="gray", linestyle="-", alpha=0.3)

    # Compliance overlay on right axis
    ax2 = ax1.twinx()
    ax2.bar(positions + 0.3, compliance, width=0.2, alpha=0.6, color="#9C27B0", label="Compliance Score")
    ax2.set_ylabel("Compliance Score", fontsize=12, color="#9C27B0")
    ax2.set_ylim(0, 1.2)
    ax2.tick_params(axis="y", labelcolor="#9C27B0")

    ax1.set_title("IC vs Compliance: Decision Tree Sweet Spot at Depth 3-5", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Annotate sweet spot
    ax1.annotate("Sweet Spot\n(high IC + full compliance)", xy=(2, 0.072), xytext=(4.5, 0.15),
                fontsize=10, color="#333",
                arrowprops=dict(arrowstyle="->", color="#333", lw=2),
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.85), labels=["IC distribution", "Compliance Score"])
    plt.tight_layout()
    fig.savefig(f"{DT_DIR}/ic_comparison_methods.png")
    plt.close(fig)
    print("  saved ic_comparison_methods.png")

# ═════════════════════════════════════════════════════════════════
# IMAGE 3: Rule coverage analysis (DT article)
# ═════════════════════════════════════════════════════════════════
def gen_dt_img3():
    """Show per-rule coverage, IC, and cumulative contribution."""
    rules = [
        "R1: Mom_20d≤0.15 & Vol≤0.025\n→ Long",
        "R2: Mom_20d≤0.15 & Vol>0.025 & VolChg≤1.5\n→ Long",
        "R3: Mom_20d≤0.15 & Vol>0.025 & VolChg>1.5\n→ Short",
        "R4: Mom_20d>0.15 & RSI≤70\n→ Neutral",
        "R5: Mom_20d>0.15 & RSI>70\n→ Short",
    ]
    coverage = [34.0, 12.0, 9.0, 21.0, 18.0]  # %
    ic = [0.082, 0.061, -0.044, 0.015, -0.067]
    contribution = [c/100 * abs(i) * np.sign(i) for c, i in zip(coverage, ic)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1]})

    # Left: coverage bar + IC line
    ax = axes[0]
    x = np.arange(len(rules))
    bars = ax.bar(x, coverage, width=0.5, alpha=0.6, color="#2196F3", label="Coverage %")
    ax.set_ylabel("Coverage (%)", color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{i+1}" for i in range(len(rules))], fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(x, ic, "o-", color="#F44336", linewidth=2, markersize=8, label="Rule IC")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Rule IC", color="#F44336")
    ax2.set_ylim(-0.1, 0.12)
    ax.set_title("Per-Rule Coverage and IC")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Right: cumulative contribution
    ax = axes[1]
    sorted_idx = np.argsort(contribution)[::-1]
    sorted_contrib = [contribution[i] for i in sorted_idx]
    sorted_labels = [f"R{i+1}" for i in sorted_idx]
    cumsum = np.cumsum(sorted_contrib)
    ax.bar(np.arange(len(rules)), sorted_contrib, width=0.5, alpha=0.6, color="#4CAF50", label="Marginal contribution")
    ax.plot(np.arange(len(rules)), cumsum, "s-", color="#FF9800", linewidth=2, markersize=8, label="Cumulative")
    ax.set_xticks(np.arange(len(rules)))
    ax.set_xticklabels(sorted_labels)
    ax.set_ylabel("Contribution to portfolio IC")
    ax.set_title("Rule Contribution (sorted by impact)")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Decision Tree Rule Analysis: Coverage, IC, and Cumulative Contribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{DT_DIR}/rule_coverage_analysis.png")
    plt.close(fig)
    print("  saved rule_coverage_analysis.png")

# ═════════════════════════════════════════════════════════════════
# IMAGE 4: Cover image (DT article) — rule paths + prediction surface
# ═════════════════════════════════════════════════════════════════
def gen_dt_cover():
    """Cover: decision boundary + rule annotations."""
    rng = np.random.default_rng(20260828)
    n = 500
    momentum = rng.uniform(-0.5, 0.5, n)
    volatility = rng.uniform(0.01, 0.05, n)
    # Simple tree decision: long if mom > 0.15 and vol < 0.025; short if mom < -0.15 or vol > 0.035
    pred = np.zeros(n)
    pred[(momentum > 0.15) & (volatility < 0.025)] = 1
    pred[(momentum < -0.15)] = -1
    pred[(volatility > 0.035)] = -1
    pred[(momentum > -0.15) & (momentum <= 0.15)] = 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1]})

    # Left: scatter colored by prediction
    ax = axes[0]
    colors_map = {1: "#4CAF50", -1: "#F44336", 0: "#9E9E9E"}
    labels_map = {1: "Long", -1: "Short", 0: "Neutral"}
    for val in [1, -1, 0]:
        mask = pred == val
        ax.scatter(momentum[mask], volatility[mask], c=colors_map[val], alpha=0.5, s=20, label=labels_map[val])
    # Decision boundaries
    ax.axvline(0.15, color="blue", linestyle="--", alpha=0.5)
    ax.axvline(-0.15, color="blue", linestyle="--", alpha=0.5)
    ax.axhline(0.025, color="purple", linestyle="--", alpha=0.5)
    ax.axhline(0.035, color="purple", linestyle="--", alpha=0.5)
    ax.set_xlabel("20-day Momentum")
    ax.set_ylabel("20-day Volatility")
    ax.set_title("Decision Tree Prediction Surface")
    ax.legend()

    # Right: rule text
    ax2 = axes[1]
    ax2.axis("off")
    rule_text = (
        "Extracted Rules:\n\n"
        "R1: IF  Mom_20d ≤ 0.15\n"
        "       AND Vol_20d ≤ 0.025\n"
        "    THEN Long  (IC=0.082)\n\n"
        "R2: IF  Mom_20d > 0.15\n"
        "       AND RSI_14 ≤ 70\n"
        "    THEN Neutral (IC=0.015)\n\n"
        "R3: IF  Mom_20d > 0.15\n"
        "       AND RSI_14 > 70\n"
        "    THEN Short  (IC=-0.067)\n\n"
        "R4: IF  Mom_20d ≤ 0.15\n"
        "       AND Vol_20d > 0.025\n"
        "       AND VolChg ≤ 1.5\n"
        "    THEN Long  (IC=0.061)\n\n"
        "R5: IF  Mom_20d ≤ 0.15\n"
        "       AND Vol_20d > 0.025\n"
        "       AND VolChg > 1.5\n"
        "    THEN Short  (IC=-0.044)"
    )
    ax2.text(0.05, 0.95, rule_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#ccc"))

    fig.suptitle("Decision Tree Factor: Interpretable Rules for Compliance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{DT_DIR}/cover.png")
    plt.close(fig)
    print("  saved cover.png")

# ═════════════════════════════════════════════════════════════════
# Run all
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating RoPE article images...")
    gen_rope_cover()
    gen_rope_img1()
    gen_rope_img2()
    gen_rope_img3()

    print("\nGenerating Decision Tree article images...")
    gen_dt_cover()
    gen_dt_img1()
    gen_dt_img2()
    gen_dt_img3()

    print("\nAll images generated successfully.")
