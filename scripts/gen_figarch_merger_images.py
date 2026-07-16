#!/usr/bin/env python3
"""Generate real, content-relevant charts for two quant articles:
  1. figarch-long-memory  (FIGARCH 长记忆波动)
  2. merger-arbitrage-risk (并购套利风险)
All charts are synthesized with numpy/matplotlib so they are genuine (not placeholders).
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
fm.fontManager.addfont(_CJK)
_CJK_NAME = fm.FontProperties(fname=_CJK).get_name()

BASE = "/Users/halo/workspace/astro-blog/public/images"
CH = "#2c7fb8"
CH2 = "#d95f0e"
CH3 = "#1b9e77"
GREY = "#666666"

plt.rcParams.update({
    "font.family": _CJK_NAME,
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# fractional difference helpers
# ---------------------------------------------------------------------------
def frac_diff_filter(d, n=400):
    """Coefficients of (1-L)^d via binomial expansion (length n)."""
    w = np.zeros(n)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (d - k + 1) / k
    return w


def apply_filter(x, w):
    n = len(x)
    out = np.zeros(n)
    for t in range(1, n):
        m = min(t + 1, len(w))
        out[t] = np.dot(w[:m], x[t - m + 1:t + 1][::-1])
    return out


def simulate_long_memory_vol(n=3000, d=0.4, seed=42):
    """Volatility path with long memory: log-vol follows FI(0,d,0)."""
    rng = np.random.default_rng(seed)
    innov = rng.standard_normal(n)
    w = frac_diff_filter(-d, n=400)  # (1-L)^{-d} integrates
    lv = apply_filter(innov, w)
    lv = lv - lv.mean()
    vol = np.exp(lv * 0.45)
    vol = vol / vol.mean()
    r = rng.standard_normal(n) * vol
    return r, vol


def acf(x, max_lag=200):
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    out = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        if k == 0:
            out[k] = 1.0
        else:
            out[k] = np.dot(x[k:], x[:-k]) / denom
    return out


# ===========================================================================
# ARTICLE 1: FIGARCH long-memory volatility
# ===========================================================================
def gen_figarch():
    out = os.path.join(BASE, "figarch-long-memory")
    os.makedirs(out, exist_ok=True)

    rng = np.random.default_rng(7)

    # ---- Fig 1: cover — impulse response of variance (hyperbolic vs exp) ----
    k = np.arange(1, 121)
    d = 0.45
    hyper = k ** (d - 1.0)              # hyperbolic decay ~ k^{d-1}
    hyper = hyper / hyper[0]
    garch = 0.985 ** k                  # GARCH(1,1)-like exponential decay
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(k, hyper, color=CH, lw=2.4, label=f"FIGARCH 冲击响应 ~ k^{{d-1}} (d={d})")
    ax.plot(k, garch, color=CH2, lw=2.4, ls="--", label="GARCH(1,1) 指数衰减 (0.985^k)")
    ax.set_xlabel("预测步长 k (交易日)")
    ax.set_ylabel("冲击对条件方差的边际贡献")
    ax.set_title("长记忆波动：冲击衰减是双曲线而非指数")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.02)
    fig.savefig(os.path.join(out, "cover.jpg"), dpi=110)
    plt.close(fig)

    # ---- Fig 2: ACF of squared returns, long-memory vs GARCH (log-log) ----
    r_lm, _ = simulate_long_memory_vol(n=4000, d=0.4, seed=11)
    # GARCH(1,1) simulation for contrast
    n = 4000
    a0, a1, b1 = 0.02, 0.08, 0.90
    h = np.zeros(n)
    h[0] = 1.0
    e = rng.standard_normal(n)
    for t in range(1, n):
        h[t] = a0 + a1 * e[t - 1] ** 2 + b1 * h[t - 1]
    r_g = e * np.sqrt(h)
    acf_lm = acf(r_lm ** 2, 200)
    acf_g = acf(r_g ** 2, 200)
    kk = np.arange(1, 201)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.loglog(kk, acf_lm[1:], color=CH, lw=2.0, marker=".", ms=3,
              label="长记忆序列 | 平方收益 ACF")
    ax.loglog(kk, acf_g[1:], color=CH2, lw=2.0, ls="--",
              label="GARCH(1,1) | 平方收益 ACF")
    ax.loglog(kk, 5.0 * kk ** (2 * 0.4 - 1), color=GREY, lw=1.4, ls=":",
              label="幂律参考 ~ k^{2d-1}")
    ax.set_ylim(1e-4, 2.0)
    ax.set_xlabel("滞后阶数 k (对数)")
    ax.set_ylabel("ACF(平方收益) (对数)")
    ax.set_title("平方收益的自相关：长记忆走幂律、GARCH 走指数")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(os.path.join(out, "acf_squared.png"), dpi=110)
    plt.close(fig)

    # ---- Fig 3: real volatility path — persistent clustering ----
    r_lm2, vol_lm = simulate_long_memory_vol(n=1500, d=0.4, seed=23)
    rng2 = np.random.default_rng(99)
    hg = np.zeros(1500); hg[0] = 1.0
    eg = rng2.standard_normal(1500)
    for t in range(1, 1500):
        hg[t] = a0 + a1 * eg[t - 1] ** 2 + b1 * hg[t - 1]
    vol_g = np.sqrt(hg)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    idx = np.arange(1500)
    ax.plot(idx, vol_lm[:1500], color=CH, lw=1.4, alpha=0.9,
            label="FIGARCH 长记忆波动 (持续聚集)")
    ax.plot(idx, vol_g, color=CH2, lw=1.4, alpha=0.7,
            label="GARCH(1,1) 波动 (聚集更短)")
    ax.set_xlabel("交易日")
    ax.set_ylabel("条件波动率 (标准化)")
    ax.set_title("波动聚集的「记忆长度」：长记忆更久")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(os.path.join(out, "vol_path.png"), dpi=110)
    plt.close(fig)
    return out


# ===========================================================================
# ARTICLE 2: Merger arbitrage risk
# ===========================================================================
def gen_merger():
    out = os.path.join(BASE, "merger-arbitrage-risk")
    os.makedirs(out, exist_ok=True)

    # ---- Fig 1: cover — target price around announcement, spread & downside ----
    fig, ax = plt.subplots(figsize=(8, 4.6))
    pre = 40.0
    bid = 50.0
    days = np.arange(120)
    # pre-announcement flat
    price = np.full(120, pre)
    # announcement at day 20 -> jumps to 47 (below bid 50), then drifts slightly
    ann = 20
    price[ann:] = 47.0 + 1.5 * (1 - np.exp(-(days[ann:] - ann) / 40.0))
    ax.axhline(bid, color=CH3, lw=2.0, ls="--", label=f"收购报价 P_bid = {bid}")
    ax.axhline(pre, color=GREY, lw=1.8, ls=":", label=f"公告前价 P_pre = {pre}")
    ax.plot(days, price, color=CH, lw=2.4, label="目标公司股价 P_tar")
    ax.scatter([ann], [47.0], color=CH2, zorder=5, s=60)
    ax.annotate("公告日跳涨", (ann, 47.0), textcoords="offset points",
                xytext=(8, -22), color=CH2, fontsize=9)
    # spread arrow
    ax.annotate("", xy=(ann + 0.2, bid), xytext=(ann + 0.2, 47.0),
                arrowprops=dict(arrowstyle="<->", color=CH3, lw=1.6))
    ax.text(ann + 6, (bid + 47.0) / 2, "价差 spread = 3", color=CH3, fontsize=9)
    # downside arrow
    ax.annotate("", xy=(110, pre), xytext=(110, 47.0 + 1.5 * (1 - np.exp(-90 / 40.0))),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.4, ls=":"))
    ax.set_xlabel("交易日")
    ax.set_ylabel("价格 (元)")
    ax.set_title("并购套利：报价与目标价的『价差』就是风险定价")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(os.path.join(out, "cover.jpg"), dpi=110)
    plt.close(fig)

    # ---- Fig 2: implied completion probability vs spread/yield ----
    P_pre = 40.0
    bid = 50.0
    spreads = np.linspace(0.5, 12.0, 200)        # P_offer - P_tar
    P_tar = bid - spreads
    # implied prob assuming full reversion to P_pre on break
    implied_p = (P_tar - P_pre) / (spreads + (P_tar - P_pre))
    gross_yield = spreads / P_tar                # gross spread yield
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(spreads, implied_p * 100, color=CH, lw=2.4,
            label="隐含完成概率 p*")
    ax.set_xlabel("价差 spread = P_bid - P_tar (元)")
    ax.set_ylabel("隐含完成概率 (%)", color=CH)
    ax.set_title("价差越窄 → 市场认为完成概率越高")
    ax2 = ax.twinx()
    ax2.plot(spreads, gross_yield * 100, color=CH2, lw=2.0, ls="--",
             label="毛价差收益率")
    ax2.set_ylabel("毛价差收益率 (%)", color=CH2)
    ax.set_ylim(40, 102)
    ax.grid(True, alpha=0.25)
    fig.savefig(os.path.join(out, "implied_prob.png"), dpi=110)
    plt.close(fig)

    # ---- Fig 3: outcome distribution & expected return ----
    # scenario: spread=3, P_pre=40, P_tar=47, break fee covers part of downside
    spread = 3.0
    P_tar = 47.0
    downside = P_tar - P_pre            # 7 if full reversion
    fee_recover = 1.5                  # break fee softens loss
    loss_if_break = max(downside - fee_recover, 1.0)
    probs = np.array([0.90, 0.10])     # assumed completion / break
    gains = np.array([spread, -loss_if_break])
    exp_val = np.dot(probs, gains)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = ["完成 (p=0.90)", "失败 (p=0.10)"]
    colors = [CH3, CH2]
    bars = ax.bar(labels, gains, color=colors, width=0.5)
    for b, g, p in zip(bars, gains, probs):
        ax.text(b.get_x() + b.get_width() / 2, g + (0.3 if g > 0 else -0.8),
                f"{g:+.1f}\n(p={p:.2f})", ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_ylabel("每股盈亏 (元)")
    ax.set_title(f"并购套利双态收益：期望每股 {exp_val:+.2f} 元")
    ax.text(0.5, -loss_if_break * 0.55,
            f"期望 = 0.90×{spread:.1f} + 0.10×({loss_if_break:.1f}) = {exp_val:+.2f}",
            transform=ax.transData, ha="center", fontsize=9, color=GREY)
    fig.savefig(os.path.join(out, "outcome_payoff.png"), dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = gen_figarch()
    p2 = gen_merger()
    print("FIGARCH images ->", p1)
    print("Merger  images ->", p2)
