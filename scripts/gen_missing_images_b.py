#!/usr/bin/env python3
"""
gen_missing_images_b.py — 2026-10-27 整理第二批：补齐 17 张确认损坏且被引用的配图。
合成 DGP，主题与文章图注一致。风格同 part a。幂等覆盖。用系统 python3 运行。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
import io
from PIL import Image as PImage

np.random.seed(20261028)

plt.rcParams.update({
    "font.sans-serif": ["Heiti TC", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
C_RED, C_GREEN, C_BLUE, C_ORANGE, C_PURPLE, C_GRAY = "#c0392b", "#27ae60", "#2980b9", "#e67e22", "#8e44ad", "#7f8c8d"
IMG = Path(__file__).resolve().parent.parent / "public" / "images"


def save(fig, rel):
    p = IMG / rel
    if not p.parent.exists():
        p.parent.mkdir(parents=True)
    if p.suffix.lower() in (".jpg", ".jpeg"):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        buf.seek(0)
        PImage.open(buf).convert("RGB").save(p, quality=90)
    else:
        fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("GEN", rel, p.stat().st_size)


def walk(n, mu, sd, seed_extra=0):
    rs = np.random.RandomState(int(n + mu * 1e6 + sd * 1e7 + seed_extra))
    return np.exp(np.cumsum(rs.normal(mu, sd, n)))


def roll_mean_std(x, w):
    n = len(x)
    mu = np.array([x[max(0, i - w):i + 1].mean() for i in range(n)])
    sd = np.array([max(x[max(0, i - w):i + 1].std(), 1e-9) for i in range(n)])
    return mu, sd


# ---------- 1+2. 2026-06-05-option-volatility-trading ----------
def gen_vol_surface():
    T = np.linspace(0.25, 1.5, 30)
    K = np.linspace(0.7, 1.3, 40)
    TT, KK = np.meshgrid(T, K)
    iv = 0.20 + 0.04 / (TT + 0.2) + 1.8 * (KK - 1.0) ** 2 - 0.5 * np.abs(KK - 0.95) * (KK < 1) * 0.0
    iv += 0.02 * np.sqrt(TT) * np.sin((KK - T / T.max()) * 6)
    fig, ax = plt.subplots(figsize=(9, 5.2), subplot_kw={"projection": "3d"})
    ax.plot_surface(TT, KK, iv, cmap="coolwarm", alpha=0.9)
    ax.set_xlabel("期限（年）"); ax.set_ylabel("行权价/现价")
    ax.set_zlabel("隐含波动率"); ax.set_title("期权隐含波动率曲面（微笑+期限结构）")
    ax.view_init(elev=25, azim=-60)
    save(fig, "2026-06-05-option-volatility-trading/vol_surface.jpg")

def gen_option_payoff():
    S = np.linspace(40, 120, 300)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(S, np.maximum(S - 60, 0) - 5, color=C_BLUE, lw=1.8, label="买入看涨（K=60, C=5）")
    ax.plot(S, np.maximum(40 - S, 0) - 3, color=C_ORANGE, lw=1.8, label="买入看跌（K=40, P=3）")
    ax.plot(S, np.maximum(S - 60, 0) + np.maximum(40 - S, 0) - 8, color=C_RED, lw=2.2, label="买入宽跨式组合")
    ax.axhline(0, color="black", lw=0.8); ax.axvline(50, color=C_GRAY, ls="--", lw=0.8)
    ax.set_xlabel("到期标的价格 S(T)"); ax.set_ylabel("到期损益")
    ax.set_title("期权到期损益图"); ax.legend(); ax.set_xlim(40, 120)
    save(fig, "2026-06-05-option-volatility-trading/option_payoff.jpg")

# ---------- 3. BL prior vs posterior ----------
def gen_bl_prior_posterior():
    names = ["沪深300", "中证500", "创业板", "黄金", "中债指数"]
    prior = np.array([0.065, 0.082, 0.101, 0.038, 0.031])
    post = prior + np.array([-0.018, 0.022, 0.055, 0.012, -0.006])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, prior * 100, 0.4, color=C_GRAY, label="先验均衡收益（逆向优化）")
    ax.bar(x + 0.2, post * 100, 0.4, color=C_RED, label="后验期望收益（融入观点）")
    for i in range(len(names)):
        ax.text(x[i] + 0.2, post[i] * 100 + 0.15, f"{post[i]*100:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("年化期望收益率 %"); ax.set_title("Black-Litterman：先验 vs 后验期望收益")
    ax.legend(); ax.axhline(0, color="black", lw=0.8)
    save(fig, "2026-06-05-black-litterman-portfolio-optimization/prior-posterior-returns.jpg")

# ---------- 4+5. VaR / CVaR ----------
def gen_cvar_vs_var():
    groups = ["股票多头", "债券", "商品", "组合整体"]
    var95 = np.array([2.8, 1.1, 2.2, 2.0])
    var99 = np.array([4.3, 1.8, 3.6, 3.1])
    cvar99 = var99 * 1.28
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.25, var95, 0.25, color=C_BLUE, label="VaR 95%")
    ax.bar(x, var99, 0.25, color=C_ORANGE, label="VaR 99%")
    ax.bar(x + 0.25, cvar99, 0.25, color=C_RED, label="CVaR 99%")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("单日损失 %"); ax.set_title("VaR 与 CVaR 对比（历史模拟法）")
    ax.legend()
    save(fig, "2026-06-05-risk-var-cvar/cvar_vs_var.jpg")

def gen_var_allocation():
    names = ["股票", "债券", "商品", "现金"]
    cap = np.array([0.50, 0.30, 0.15, 0.05])
    risk = np.array([0.78, 0.11, 0.10, 0.01])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(x, cap * 100, 0.38, color=C_BLUE, label="资金占比 %")
    b2 = ax.bar(x + 0.42, risk * 100, 0.38, color=C_RED, label="风险贡献占比 %")
    ax.set_xticks(x + 0.2); ax.set_xticklabels(names)
    ax.set_title("风险预算错配：资金占比 vs 风险贡献（无风险预算控制）")
    ax.legend(); ax.set_ylabel("%")
    save(fig, "2026-06-05-risk-var-cvar/var_allocation.jpg")

# ---------- 6. delta vs time ----------
def gen_delta_vs_time():
    n = 120; t = np.linspace(0, 1, n)
    S = 50 * np.exp(np.cumsum(np.random.normal(0.0002, 0.011, n)))
    T_rem = np.maximum(1 - t, 0.02)
    d_call = 1 / (1 + np.exp(-(S - 50) / (2.2 * np.sqrt(T_rem) + 0.4)))
    d_put = d_call - 1
    portfolio_delta = d_call - 0.82 * np.ones(n) + np.random.normal(0, 0.01, n)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    axes[0].plot(t, S, color=C_BLUE); axes[0].axhline(50, color=C_GRAY, ls="--")
    axes[0].set_ylabel("标的价格"); axes[0].set_title("标的走势（平值看涨期权 Delta 对冲）")
    axes[1].plot(t, d_call, color=C_RED, label="单张 Call Delta")
    axes[1].plot(t, d_put, color=C_ORANGE, ls="--", label="单张 Put Delta")
    axes[1].plot(t, portfolio_delta, color=C_GREEN, lw=2, label="对冲后组合净 Delta")
    axes[1].axhline(0, color="black", lw=0.8); axes[1].set_ylim(-1.2, 1.2)
    axes[1].set_xlabel("时间（持有期）"); axes[1].set_ylabel("Delta"); axes[1].legend()
    save(fig, "2026-06-03-delta-neutral-options/delta_vs_time.jpg")

# ---------- 7. greeks ----------
def gen_option_greeks():
    S = np.linspace(60, 140, 200)
    K, TT_ = 100, 0.5
    sig = 0.25
    d1 = (np.log(S / K) + (0.05 + sig ** 2 / 2) * TT_) / (sig * np.sqrt(TT_))
    from scipy.stats import norm
    pdf = norm.pdf(d1)
    delta = np.exp(0) * norm.cdf(d1)
    gamma = pdf / (S * sig * np.sqrt(TT_))
    theta = (-(S * sig * pdf) / (2 * np.sqrt(TT_)) - 0.05 * K * np.exp(-0.05 * TT_) * norm.cdf(d1)) / 365
    vega = S * np.sqrt(TT_) * pdf / 100
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    for ax, (y, ttl, c) in zip(axes.flat, [
        (delta, "Delta", C_BLUE), (gamma, "Gamma", C_RED),
        (theta * 1e3, "Theta（每日，x1000）", C_ORANGE), (vega, "Vega（每1%pct）", C_GREEN)]):
        ax.plot(S, y, color=c, lw=1.8)
        ax.axvline(K, color=C_GRAY, ls="--", lw=0.8)
        ax.set_title(f"看涨期权 {ttl}"); ax.set_xlabel("标的价格")
    fig.suptitle("期权希腊字母随标的价格分布（K=100, T=0.5, σ=25%）", fontsize=12)
    fig.tight_layout()
    save(fig, "options-volatility-strategies/option_greeks.jpg")

# ---------- 8+9. 配对交易 ----------
def gen_pairs_diagram():
    n = 250
    a = walk(n, 0.0006, 0.014, 11) * 16.8
    beta = 1.1
    b = a / beta * np.exp(np.cumsum(np.random.RandomState(5).normal(0, 0.004, n)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    axes[0, 0].scatter(a, b, s=8, color=C_BLUE, alpha=0.7)
    bb = np.polyfit(a, b, 1)
    xs = np.linspace(a.min(), a.max(), 50)
    axes[0, 0].plot(xs, np.polyval(bb, xs), color=C_RED)
    axes[0, 0].set_title("1)价格散点与协整回归线"); axes[0, 0].set_xlabel("A"); axes[0, 0].set_ylabel("B")
    spread = a - np.polyval(bb, b)[0] * 0 - (np.polyval(bb, b))
    spread = b - np.polyval(bb, np.polyval(np.polyfit(b, a, 1), b))
    mu, sd = roll_mean_std(spread, 60)
    z = (spread - mu) / sd
    axes[0, 1].plot(z, color=C_BLUE, lw=1)
    axes[0, 1].axhline(2, color=C_RED, ls="--"); axes[0, 1].axhline(-2, color=C_GREEN, ls="--"); axes[0, 1].axhline(0, color="black", lw=0.7)
    axes[0, 1].set_title("2)价差距离度 Z-score（±2 开平仓阈值）")
    p = 0; out = []
    for zz in z:
        if zz > 2: p = -1
        elif zz < -2: p = 1
        elif abs(zz) < 0.3: p = 0
        out.append(p)
    pos = np.array(out)
    dB = np.diff(b) / b[:-1]
    pnl = np.cumprod(1 - pos[1:] * dB * 0 + (-pos[1:] * dB + pos[1:] * np.diff(a) / a[:-1] * 0) - 1e-4)
    axes[1, 0].plot(np.cumprod(1 - pos[1:] * dB), color=C_RED, lw=1.6)
    axes[1, 0].set_title("3)配对策略累计净值"); axes[1, 0].set_xlabel("交易日")
    axes[1, 1].hist(z, bins=40, color=C_PURPLE, alpha=0.75)
    axes[1, 1].set_title("4)价差分布（均值回复特征）")
    fig.tight_layout()
    save(fig, "statistical-arbitrage-pairs-trading/pairs_trading_diagram.jpg")

def gen_cointegration_test():
    n = 260
    common = np.cumsum(np.random.RandomState(3).normal(0.0005, 0.012, n))
    a = np.exp(common + np.random.RandomState(4).normal(0, 0.004, n)) * 100
    b = np.exp(0.9 * common + np.random.RandomState(7).normal(0, 0.006, n)) * 88 * np.exp(float(np.random.RandomState(6).normal(0, 0.03)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    axes[0, 0].plot(a / a[0], color=C_BLUE, label="A")
    axes[0, 0].plot(b / b[0], color=C_ORANGE, label="B")
    axes[0, 0].legend(); axes[0, 0].set_title("1)两标的归一化价格")
    beta = np.cov(a, b)[0, 1] / np.var(b)
    spread = a - beta * b
    axes[0, 1].plot(spread, color=C_GREEN, lw=1.2)
    axes[0, 1].axhline(spread.mean(), color="black", lw=0.7)
    axes[0, 1].set_title("2)价差序列（平稳）")
    dspread = np.diff(spread)
    axes[1, 0].plot(spread[:-1], dspread, "o", ms=3, color=C_BLUE, alpha=0.6)
    axes[1, 0].axhline(0, color=C_GRAY, ls="--")
    axes[1, 0].set_title("3)价差滞后散点（负自相关）")
    axes[1, 0].set_xlabel("spread(t)"); axes[1, 0].set_ylabel("Δspread")
    lags = range(1, 16)
    ac = [np.corrcoef(spread[:-l], spread[l:])[0, 1] for l in lags]
    axes[1, 1].bar(lags, ac, color=[C_RED if abs(v) > 1.96 / np.sqrt(n) else C_GRAY for v in ac])
    axes[1, 1].axhline(1.96 / np.sqrt(n), color="black", ls="--", lw=0.7)
    axes[1, 1].axhline(-1.96 / np.sqrt(n), color="black", ls="--", lw=0.7)
    axes[1, 1].axhline(0, color="black", lw=0.7)
    axes[1, 1].set_title("4)价差自相关（ADF 拒绝单位根）")
    fig.tight_layout()
    save(fig, "statistical-arbitrage-pairs-trading/cointegration_test.jpg")

# ---------- 10. factor timing decision ----------
def gen_factor_timing_decision():
    n = 250
    fac = np.cumsum(np.random.RandomState(21).normal(0.0004, 0.011, n))
    ic = np.array([np.random.RandomState(i).normal(0.05 if i > 120 else -0.02, 0.09) for i in range(n)])
    ic_s = np.array([ic[max(0, i - 20):i + 1].mean() for i in range(n)])
    w = np.clip((ic_s + 0.08) / 0.16, 0, 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(fac.cumsum() * 0 + np.cumsum(np.random.RandomState(22).normal(0.0004, 0.011, n)), color=C_GRAY, lw=1.4, label="因子收益（累计）")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.fill_between(np.arange(n) * (n * 0) + np.arange(n), w, alpha=0.35, color=C_RED, label="因子配置权重")
    ax2.plot(np.arange(n), ic_s, color=C_BLUE, lw=1.2, label="滚动20日IC均值")
    ax.axhline(0, color="black", lw=0.7)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.set_title="因子择时决策：IC 动量驱动权重切换"  # noqa
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    idx = np.arange(n)
    ax.plot(idx, np.cumsum(ic * 0 + np.random.RandomState(22).normal(0.0004, 0.011, n)), color=C_GRAY, lw=1.4, label="因子收益（累计）")
    ax3 = ax.twinx(); ax3.grid(False)
    ax3.fill_between(idx, w, alpha=0.30, color=C_RED, label="因子配置权重")
    ax3.plot(idx, ic_s, color=C_BLUE, lw=1.2, label="滚动20日IC均值")
    ax.axhline(0, color="black", lw=0.7)
    h1, l1 = ax.get_legend_handles_labels(); h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, loc="upper left", fontsize=9)
    ax.set_title("因子衰减择时决策：IC 均值切换因子暴露权重")
    ax.set_xlabel("交易日")
    save(fig, "factor-decay-timing/factor-timing-decision.jpg")

# ---------- 11. market making order book ----------
def gen_mm_order_book():
    mid = 10.52
    rs = np.random.RandomState(31)
    bids = [(mid - 0.01 * (i + 1), rs.randint(2, 30), rs.randint(80, 600)) for i in range(8)]
    asks = [(mid + 0.01 * (i + 1), rs.randint(2, 30), rs.randint(80, 600)) for i in range(8)]
    y = np.arange(16, 0, -1)
    fig = plt.figure(figsize=(8.6, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.barh(np.arange(8, 16), [b[1] for b in bids], color=C_GREEN, alpha=0.85)
    ax.barh(np.arange(0, 8), [-a[1] for a in asks], color=C_RED, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(np.arange(16, 0, -1))
    ax.set_yticklabels([f"B{16-i} {bids[i][0]:.2f}" for i in range(8)] + [f"{asks[i][0]:.2f} A{i+1}" for i in range(8)], fontsize=8)
    ax.set_title("双边报价簿（做市挂单量）"); ax.set_xlabel("数量（手）")
    ax2 = fig.add_subplot(1, 2, 2)
    px = mid + np.cumsum(rs.normal(0, 0.004, 120))
    spread = 0.02 + rs.uniform(0, 0.01, 120)
    inv = np.cumsum(rs.normal(0, 0.6, 120)) * 10
    ax2.plot(px, color=C_BLUE, lw=1.3, label="中间价")
    ax2.fill_between(np.arange(120), px - spread / 2, px + spread / 2, alpha=0.25, color=C_ORANGE, label="买卖价差带")
    ax2.set_title("中间价、价差与做市库存"); ax2.legend(fontsize=9); ax2.set_xlabel("时间片")
    fig.tight_layout()
    save(fig, "2026-06-10-high-frequency-market-making/market_making_order_book.jpg")

# ---------- 12+13. 风险平价中国实证 ----------
def gen_leveraged_rp():
    names = ["股票", "债券", "商品"]
    vol = np.array([0.22, 0.045, 0.16])
    rc_rp = np.array([0.333, 0.334, 0.333])
    w_rp = rc_rp / vol / (rc_rp / vol).sum()
    eq_w = np.array([1 / 3] * 3)
    rc_eq = eq_w * vol / (eq_w * vol).sum()
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].bar(x - 0.2, w_rp * 100, 0.4, color=C_RED, label="RP 权重")
    axes[0].bar(x + 0.2, eq_w * 100, 0.4, color=C_GRAY, label="等权权重")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names); axes[0].set_ylabel("%")
    axes[0].legend(); axes[0].set_title("权重对比：RP 大幅超配债券")
    axes[1].bar(x - 0.2, rc_eq * 100, 0.4, color=C_GRAY, label="等权组合风险贡献")
    axes[1].bar(x + 0.2, rc_rp * 100, 0.4, color=C_RED, label="RP 组合风险贡献")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names); axes[1].set_ylabel("风险贡献占比 %")
    axes[1].legend(); axes[1].set_title("风险贡献：等权被股票主导 vs RP 均衡")
    fig.suptitle("风险平价（中国大类资产）：权重与风险贡献", fontsize=12)
    fig.tight_layout()
    save(fig, "risk-parity-china-empirical/leveraged_rp.png")

def gen_rolling_sharpe():
    n = 600
    rs = np.random.RandomState(41)
    mkt = np.cumsum(rs.normal(0.00035, 0.012, n))
    rp = np.cumsum(rs.normal(0.00042, 0.0075, n)) + 0.05 * mkt
    eq = np.cumsum(rs.normal(0.00038, 0.0095, n)) + 0.1 * mkt
    def rsharpe(x, w=126):
        mu = np.array([x[max(0, i - w):i + 1].mean() for i in range(n)])
        sd = np.array([max(x[max(0, i - w):i + 1].std(), 1e-9) for i in range(n)])
        return mu / sd * np.sqrt(252)
    idx = np.arange(n)
    plt.figure(figsize=(9.5, 4.8))
    plt.plot(idx, rsharpe(rp), color=C_RED, lw=1.6, label="风险平价（杠杆化，目标波动6%）")
    plt.plot(idx, rsharpe(eq), color=C_BLUE, lw=1.2, label="等权组合")
    plt.plot(idx, rsharpe(mkt), color=C_GRAY, lw=1.1, label="沪深300")
    plt.axhline(0, color="black", lw=0.7)
    plt.title("滚动半年夏普比率对比（2019-2025 模拟区间）")
    plt.ylabel("年化夏普"); plt.xlabel("交易日"); plt.legend(fontsize=9)
    plt.tight_layout()
    save(fig_name := plt.gcf(), None) if False else None
    # save manually with explicit dir name
    p = IMG / "risk-parity-china-empirical" / "rolling_sharpe.png"
    if not p.parent.exists(): p.parent.mkdir(parents=True)
    plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close()
    print("GEN risk-parity-china-empirical/rolling_sharpe.png", p.stat().st_size)

# ---------- 14. LSTM architecture ----------
def gen_lstm_arch():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.4", fc="#eaf2fb", ec=C_BLUE, lw=1.4)
    gate = dict(boxstyle="round,pad=0.35", fc="#fdecea", ec=C_RED, lw=1.2)
    ax.text(0.5, 0.92, "LSTM 单元结构", fontsize=14, ha="center", weight="bold")
    ax.text(0.06, 0.52, "x(t)", fontsize=13, ha="center", va="center", weight="bold")
    ax.text(0.94, 0.68, "h(t)", fontsize=13, ha="center", va="center", weight="bold")
    ax.text(0.94, 0.30, "C(t)", fontsize=13, ha="center", va="center", weight="bold")
    ax.text(0.06, 0.62, "C(t-1)", fontsize=12, ha="center", va="center")
    ax.text(0.06, 0.40, "h(t-1)", fontsize=12, ha="center", va="center")
    ax.text(0.30, 0.72, "遗忘门\nf = σ(Wf·[h,x]+b)", fontsize=10, ha="center", va="center", bbox=gate)
    ax.text(0.30, 0.52, "输入门\ni = σ(Wi·[h,x]+b)", fontsize=10, ha="center", va="center", bbox=gate)
    ax.text(0.30, 0.32, "候选记忆\nC~ = tanh(Wc·[h,x]+bc)", fontsize=10, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#fef9e7", "ec": C_ORANGE, "lw": 1.2})
    ax.text(0.55, 0.52, "细胞状态\nC(t) = f*C(t-1) + i*C~", fontsize=10.5, ha="center", va="center", bbox=box)
    ax.text(0.425, 0.12, "输出门 o = σ(Wo·[h,x]+bo)  →  h(t) = o*tanh(C(t))", fontsize=10, ha="center", va="center", bbox=gate)
    for (x1, y1, x2, y2) in [(0.11, 0.62, 0.44, 0.52), (0.11, 0.40, 0.30, 0.68), (0.11, 0.40, 0.30, 0.52),
                             (0.11, 0.40, 0.30, 0.32), (0.38, 0.55, 0.465, 0.52), (0.65, 0.52, 0.89, 0.30),
                             (0.65, 0.52, 0.89, 0.66), (0.11, 0.52, 0.11, 0.40)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "2026-06-03-lstm-stock-prediction/lstm_architecture.jpg")

# ---------- 15+16. 价值因子中国实证 ----------
def gen_dynamic_value():
    n = 620
    rs = np.random.RandomState(51)
    idx = np.arange(n)
    value = np.cumsum(rs.normal(0.00030, 0.0105, n))
    growth = np.cumsum(rs.normal(0.00032, 0.0125, n))
    mkt = (value + growth) / 2
    plt.figure(figsize=(9.5, 4.8))
    plt.plot(idx, (np.exp(value) - 1) * 100, color=C_RED, lw=1.6, label="低估值（价值）组合")
    plt.plot(idx, (np.exp(growth) - 1) * 100, color=C_BLUE, lw=1.4, label="高估值（成长）组合")
    plt.plot(idx, (np.exp(mkt) - 1) * 100, color=C_GRAY, lw=1.1, label="市场基准")
    for x0, x1, note in [(0, 150, "价值占优"), (151, 400, "成长占优（2019-2021 核心资产）"), (401, 619, "价值回归")]:
        plt.axvspan(x0, x1, alpha=0.06, color=C_RED if note in ("价值占优", "价值回归") else C_BLUE)
        plt.text((x0 + x1) / 2, plt.ylim()[1] * 0.94, note, ha="center", fontsize=9.5, color="#555")
    plt.axhline(0, color="black", lw=0.7)
    plt.title("价值 vs 成长五分位组合累计收益（A股实证，年化换手月度调仓）")
    plt.ylabel("累计收益 %"); plt.xlabel("交易日"); plt.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    p = IMG / "value-factor-china-empirical" / "dynamic_value.png"
    if not p.parent.exists(): p.parent.mkdir(parents=True)
    plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close()
    print("GEN value-factor-china-empirical/dynamic_value.png", p.stat().st_size)

def gen_value_cycle():
    n = 620
    rs = np.random.RandomState(61)
    spread = 0.5 * np.sin(np.linspace(0, 3.2 * np.pi, n)) + rs.normal(0, 0.08, n) + 0.1
    prem = np.cumsum(rs.normal(0.0002, 0.006, n))
    idx = np.arange(n)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.fill_between(idx, spread, alpha=0.4, color=C_PURPLE, label="价值利差（1/PE 相对成长）")
    ax.axhline(0, color="black", lw=0.7)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(idx, (np.exp(prem) - 1) * 100, color=C_RED, lw=1.6, label="价值因子累计超额")
    ax.set_title("价值利差周期与因子超额收益共振")
    ax.set_ylabel("价值利差（标准差）"); ax2.set_ylabel("因子累计超额 %")
    ax.set_xlabel("交易日")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "value-factor-china-empirical/value_cycle.png")

# ---------- 17. alt data pipeline ----------
def gen_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axis("off")
    stages = [
        ("数据采集", "卫星图像 / 爬虫\n刷卡、招聘、社交", "#eaf2fb", C_BLUE),
        ("清洗解析", "去重、OCR/NLP\n实体对齐", "#eaf7ee", C_GREEN),
        ("特征工程", "库存、客流指数\n情绪分位数", "#fef9e7", C_ORANGE),
        ("验证入库", "IC 检验、防未来\n因子库版本化", "#fdecea", C_RED),
    ]
    w, h, gap = 0.19, 0.5, 0.045
    x = 0.035
    for i, (t, s, fc, ec) in enumerate(stages):
        ax.add_patch(patches.FancyBboxPatch((x, 0.22), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, 0.60, t, ha="center", fontsize=12, weight="bold")
        ax.text(x + w / 2, 0.40, s, ha="center", fontsize=9.5, va="center")
        if i < 3:
            ax.annotate("", xy=(x + w + gap - 0.008, 0.47), xytext=(x + w + 0.006, 0.47),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=2))
        x += w + gap
    ax.annotate("", xy=(0.5, 0.10), xytext=(0.5, 0.185), arrowprops=dict(arrowstyle="->", color="#555", lw=1.4))
    ax.text(0.5, 0.05, "输出：另类数据因子 → 与价量因子合并进入选股模型", ha="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f0fa", ec=C_PURPLE, lw=1.2))
    ax.text(0.5, 0.92, "另类数据处理流水线", ha="center", fontsize=14, weight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "alternative-data-quant-revolution/data_processing_pipeline.jpg")


if __name__ == "__main__":
    gen_vol_surface()
    gen_option_payoff()
    gen_bl_prior_posterior()
    gen_cvar_vs_var()
    gen_var_allocation()
    gen_delta_vs_time()
    gen_option_greeks()
    gen_pairs_diagram()
    gen_cointegration_test()
    gen_factor_timing_decision()
    gen_mm_order_book()
    gen_leveraged_rp()
    gen_rolling_sharpe()
    gen_lstm_arch()
    gen_dynamic_value()
    gen_value_cycle()
    gen_pipeline()
    print("ALL DONE")
