#!/usr/bin/env python3
"""Generate real charts for 'Kyle lambda 与信息含量度量' article.

Methodology (honest, verifiable):
Kyle (1985) models a market with a single informed trader, noise traders, and a competitive
market maker. The market maker cannot tell informed order flow from noise, so they set price
as a linear function of the net order flow (signed volume) Q:
    Delta P = lambda * Q
lambda (Kyle's lambda) is the PRICE IMPACT coefficient: how much the price moves per unit of
net signed volume. A HIGH lambda = an illiquid, information-heavy market (each trade moves
price a lot). A LOW lambda = a deep, liquid market.

Empirically lambda is estimated by regressing price changes on signed order flow:
    r_t = alpha + lambda * OF_t + e_t
where OF_t is signed volume (buys positive, sells negative) via a trade-classification rule
(we simulate the sign directly). We:
  1. simulate a Kyle market and recover lambda by OLS (scatter + fit line),
  2. show lambda rises with the fraction of informed trading (info content),
  3. build a rolling lambda time series that spikes around an information event,
  4. show the cross-sectional relation: illiquid (small/low-volume) names have higher lambda.
All from scratch with numpy.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

import matplotlib.font_manager as fm
for _f in ["/Library/Fonts/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc",
           "/System/Library/Fonts/Supplemental/Songti.ttc"]:
    try:
        fm.fontManager.addfont(_f); break
    except Exception:
        continue
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/kyle-lambda-information"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight", "font.family": "Arial Unicode MS",
    "axes.unicode_minus": False,
})

rng = np.random.default_rng(20260721)

# ---------------- Kyle market simulator ----------------
def simulate_kyle(n=2000, lam_true=0.8, sigma_noise=1.0, seed=None, event=None):
    """Simulate signed order flow OF and resulting price changes under Delta P = lambda*OF + eps.
    OF = informed component (persistent, information-driven) + noise-trader component (mean-zero).
    Returns OF (signed volume) and r (price change)."""
    r = np.random.default_rng(seed)
    # noise-trader order flow: mean-zero
    noise_of = r.normal(0, sigma_noise, n)
    # informed order flow: driven by a latent signal v (fundamental innovation)
    v = r.normal(0, 0.6, n)
    informed_of = v  # informed trade in direction of signal
    of = noise_of + informed_of
    lam = np.full(n, lam_true)
    if event is not None:
        # around the event, informational asymmetry (and thus lambda) spikes
        e0, e1, mult = event
        lam[e0:e1] = lam_true * mult
        informed_of[e0:e1] *= 2.0
        of = noise_of + informed_of
    micro = r.normal(0, 0.15, n)   # residual microstructure
    price_change = lam * of + micro
    return of, price_change

def estimate_lambda(of, r):
    """OLS: r = alpha + lambda*of. Return lambda, alpha, R^2."""
    X = np.column_stack([np.ones_like(of), of])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    alpha, lam = beta
    pred = X @ beta
    ss_res = np.sum((r - pred)**2)
    ss_tot = np.sum((r - r.mean())**2)
    r2 = 1 - ss_res/ss_tot
    return lam, alpha, r2

# ---------- 1. Scatter: price change vs signed order flow, with fitted lambda ----------
def scatter_lambda():
    of, r = simulate_kyle(n=1500, lam_true=0.8, seed=11)
    lam, alpha, r2 = estimate_lambda(of, r)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(of, r, s=10, alpha=0.35, color="#1f77b4", label="逐笔观测（净签名成交量, 价格变化）")
    xs = np.linspace(of.min(), of.max(), 100)
    ax.plot(xs, alpha + lam*xs, color="#d62728", lw=2.5,
            label=f"OLS 拟合：λ = {lam:.3f}  (R²={r2:.2f})")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("净签名成交量 OF（买为正 / 卖为负）")
    ax.set_ylabel("价格变化 ΔP")
    ax.set_title("Kyle λ 的本质：价格变化对净订单流的回归斜率")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.savefig(f"{OUT}/lambda_scatter.png"); plt.close(fig)
    print(f"[1] lambda_scatter.png  lam_hat={lam:.3f} (true 0.8)  R2={r2:.3f}")

# ---------- 2. lambda rises with informed fraction (information content) ----------
def lambda_vs_info():
    fracs = np.linspace(0.05, 0.95, 12)   # fraction of order flow that is informed
    lams = []
    for f in fracs:
        # build order flow with controlled informed share
        n = 3000
        v = rng.normal(0, 1.0, n)
        informed = v
        noise = rng.normal(0, 1.0, n)
        of = f*informed + (1-f)*noise
        # true price move loads only on the informed (信息) part
        r = 1.0*(f*informed) + rng.normal(0, 0.2, n)
        lam, _, _ = estimate_lambda(of, r)
        lams.append(lam)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fracs*100, lams, "o-", color="#9467bd", lw=2.2, ms=7)
    ax.set_xlabel("订单流中知情交易占比 (%)")
    ax.set_ylabel("估计的 Kyle λ（价格冲击）")
    ax.set_title("信息含量越高，价格冲击 λ 越大")
    ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/lambda_vs_info.png"); plt.close(fig)
    print(f"[2] lambda_vs_info.png  lam@5%={lams[0]:.3f}  lam@95%={lams[-1]:.3f}")

# ---------- 3. Rolling lambda spikes around an information event ----------
def rolling_lambda_event():
    n = 4000
    event = (2000, 2300, 3.0)  # info shock at t=2000..2300, lambda x3
    of, r = simulate_kyle(n=n, lam_true=0.5, seed=77, event=event)
    win = 200
    xs, lams = [], []
    for t in range(win, n):
        l, _, _ = estimate_lambda(of[t-win:t], r[t-win:t])
        xs.append(t); lams.append(l)
    xs = np.array(xs); lams = np.array(lams)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(xs, lams, color="#1f77b4", lw=1.6, label=f"滚动 λ（{win} 笔窗口）")
    ax.axvspan(event[0], event[1], color="#d62728", alpha=0.15, label="信息事件窗口")
    ax.axhline(0.5, color="#2ca02c", ls="--", lw=1.3, label="常态 λ ≈ 0.5")
    ax.set_xlabel("成交序号（时间）"); ax.set_ylabel("滚动估计 λ")
    ax.set_title("滚动 Kyle λ 在信息事件期间显著抬升")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/rolling_lambda_event.png"); plt.close(fig)
    peak = lams[(xs>=event[0]) & (xs<=event[1]+win)].max()
    print(f"[3] rolling_lambda_event.png  base~0.5  peak={peak:.2f}")

# ---------- 4. Cross-section: illiquid names have higher lambda ----------
def cross_section_liquidity():
    # simulate 40 stocks with different daily volumes; lambda ~ inversely related to depth
    n_stocks = 40
    volume = np.exp(rng.normal(14, 1.2, n_stocks))   # daily volume (shares), lognormal
    # true lambda inversely proportional to sqrt(volume) (Amihud-like), plus noise
    lam_true = 5000.0 / np.sqrt(volume) * np.exp(rng.normal(0, 0.25, n_stocks))
    # estimate each via a short regression
    lam_hat = []
    for i in range(n_stocks):
        of, r = simulate_kyle(n=800, lam_true=lam_true[i], seed=500+i)
        l, _, _ = estimate_lambda(of, r)
        lam_hat.append(l)
    lam_hat = np.array(lam_hat)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(volume, lam_hat, c=np.log(volume), cmap="viridis_r", s=70, edgecolor="k", lw=0.5)
    ax.set_xscale("log")
    # fit line in log-log
    lv = np.log(volume); ll = np.log(np.maximum(lam_hat, 1e-6))
    b = np.polyfit(lv, ll, 1)
    xs = np.linspace(lv.min(), lv.max(), 50)
    ax.plot(np.exp(xs), np.exp(b[1]+b[0]*xs), color="#d62728", lw=2,
            label=f"幂律拟合：λ ∝ 成交量^{b[0]:.2f}")
    ax.set_xlabel("日成交量（对数轴）"); ax.set_ylabel("估计的 Kyle λ")
    ax.set_title("横截面：成交量越低（越不流动），λ 越高")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="log(成交量)")
    fig.savefig(f"{OUT}/cross_section_liquidity.png"); plt.close(fig)
    print(f"[4] cross_section_liquidity.png  slope={b[0]:.2f} (expect ~ -0.5)")

if __name__ == "__main__":
    scatter_lambda()
    lambda_vs_info()
    rolling_lambda_event()
    cross_section_liquidity()
    print("DONE kyle-lambda charts ->", OUT)
