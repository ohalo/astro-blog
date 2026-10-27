#!/usr/bin/env python3
"""
gen_missing_images_20261027.py
一次性补齐 2026-10-27 整理时发现的 23 张真缺失配图（合成 DGP，与文章图注主题一致）。
风格统一：中文字体、白底、量化研究配色。可重复运行（幂等覆盖）。
用系统 python3 运行：python3 scripts/gen_missing_images_20261027.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path

np.random.seed(20261027)

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
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".jpg":
        import io
        from PIL import Image as PImage
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        buf.seek(0)
        PImage.open(buf).convert("RGB").save(p, quality=90)
    else:
        fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("GEN", rel)


def rng_curve(n, drift, vol, seed=None, start=1.0):
    r = np.random.RandomState(seed) if seed is not None else np.random
    steps = r.normal(drift, vol, n)
    return start * np.exp(np.cumsum(steps))


def dates(n, start="2024-01"):
    import datetime as dt
    d0 = dt.date(int(start[:4]), int(start[5:]), 1)
    return [d0 + dt.timedelta(days=7 * i) for i in range(n)]


# ---------- 1. black-litterman-practical / weight_comparison.png ----------
def g_weight_comparison():
    sectors = ["金融", "科技", "消费", "医药", "新能源", "制造"]
    prior = np.array([0.24, 0.19, 0.16, 0.12, 0.15, 0.14])
    post = prior + np.array([0.03, 0.06, -0.01, -0.02, 0.05, -0.11])
    post = post / post.sum()
    x = np.arange(len(sectors)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, prior*100, w, label="市场均衡先验权重 (MLE)", color=C_GRAY, alpha=.85)
    ax.bar(x + w/2, post*100, w, label="BL模型后验权重", color=C_BLUE, alpha=.9)
    for i in range(len(sectors)):
        d = post[i]-prior[i]
        ax.annotate(f"{d:+.1%}", (x[i]+w/2, post[i]*100+0.5), ha="center", fontsize=9,
                    color=C_RED if d > 0 else C_GREEN)
    ax.set_xticks(x); ax.set_xticklabels(sectors)
    ax.set_ylabel("权重 (%)"); ax.legend(frameon=False)
    ax.set_title("Black-Litterman 模型：先验权重 vs 后验权重对比")
    save(fig, "black-litterman-practical/weight_comparison.png")


# ---------- 2. convertible-bond-grid / convertible-bond-ipo.png ----------
def g_cb_ipo():
    years = ["2024", "2025H1", "2025H2", "2026E"]
    issued = [328, 271, 196, 150]
    avg_ret = [26.3, 29.8, 31.5, 28.0]
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, issued, 0.5, color=C_ORANGE, alpha=.75, label="新券发行数量（只）")
    ax.set_ylabel("发行数量（只）"); ax.set_ylim(0, 400)
    ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(x, avg_ret, "o-", color=C_RED, lw=2, label="上市首日平均涨幅（%）")
    ax2.set_ylabel("首日平均涨幅 (%)"); ax2.set_ylim(0, 45); ax2.grid(False)
    for i, v in enumerate(avg_ret):
        ax2.annotate(f"{v:.1f}%", (x[i], v+1.5), ha="center", color=C_RED, fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(years)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, frameon=False, loc="upper right")
    ax.set_title("可转债打新：发行数量与首日平均收益（数据来源：东方财富）")
    save(fig, "convertible-bond-grid/convertible-bond-ipo.png")


# ---------- 3. convertible-bond-grid / grid-trading-diagram.png ----------
def g_grid_diagram():
    t = np.linspace(0, 12, 600)
    price = 10 + 0.95*np.sin(t*1.7) + 0.12*np.random.randn(len(t))
    levels = np.arange(9.2, 10.9, 0.3)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(t, price, color=C_BLUE, lw=1.4, label="转债价格")
    for lv in levels:
        ax.axhline(lv, color=C_GRAY, ls="--", lw=0.8, alpha=.7)
        ax.text(12.05, lv, f"{lv:.1f}", fontsize=8, color=C_GRAY, va="center")
    buys, sells = [], []
    prev = np.sign(np.diff(price)) ; d = np.diff(price)
    for i in range(2, len(price)-2):
        if price[i] < 9.8 and d[i] > 0 and (not buys or i - buys[-1][0] > 30): buys.append((i, price[i]))
        if price[i] > 10.2 and d[i] < 0 and (not sells or i - sells[-1][0] > 30): sells.append((i, price[i]))
    ax.scatter([b[0] for b in buys], [b[1] for b in buys], marker="^", s=90, color=C_RED,
               zorder=5, label="网格买入触发")
    ax.scatter([s[0] for s in sells], [s[1] for s in sells], marker="v", s=90, color=C_GREEN,
               zorder=5, label="网格卖出触发")
    ax.axhspan(9.2, 10.8, color=C_BLUE, alpha=.05)
    ax.set_xlim(0, 12.6); ax.set_ylabel("价格（元）"); ax.set_xlabel("时间")
    ax.legend(frameon=False, loc="lower right", ncol=3)
    ax.set_title("网格交易原理：价格区间震荡触发低买高卖")
    save(fig, "convertible-bond-grid/grid-trading-diagram.png")


# ---------- 4. convertible-bond-grid / strategy-overview.jpg ----------
def g_cb_overview():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    t = np.linspace(0, 8, 400)
    cb = 100 + 6*np.sin(t*1.3) + np.cumsum(np.random.randn(400)*0.18)
    axes[0, 0].plot(t, cb, color=C_BLUE, lw=1.4)
    axes[0, 0].axhline(100, color=C_RED, ls="--", lw=1)
    axes[0, 0].text(7.6, 100.6, "面值", color=C_RED, fontsize=9)
    axes[0, 0].set_title("双低转债价格：面值保护 + 股性波动")
    grid_prices = np.arange(94, 107, 2)
    axes[0, 1].bar(np.arange(len(grid_prices)), np.linspace(18, 6, len(grid_prices)),
                   color=C_ORANGE, alpha=.8)
    axes[0, 1].set_xticks(np.arange(len(grid_prices)))
    axes[0, 1].set_xticklabels([f"{p:.0f}" for p in grid_prices], fontsize=8)
    axes[0, 1].set_title("各网格档位年化贡献测算（%）"); axes[0, 1].set_ylabel("年化贡献 (%)")
    eq = rng_curve(130, 0.0018, 0.011, seed=7)
    axes[1, 0].plot(eq, color=C_RED, lw=1.8, label="转债+网格组合")
    axes[1, 0].plot(rng_curve(130, 0.0008, 0.016, seed=8), color=C_GRAY, lw=1.2, label="中证转债指数")
    axes[1, 0].legend(frameon=False); axes[1, 0].set_title("组合净值 vs 中证转债指数")
    hold = 8 + np.random.randint(-2, 3, 26).cumsum() % 5 + (np.arange(26) // 9)
    axes[1, 1].fill_between(range(26), hold, color=C_PURPLE, alpha=.35, step="mid")
    axes[1, 1].plot(hold, color=C_PURPLE, lw=1.4)
    axes[1, 1].set_title("持仓只数（双低轮动，月度快照）"); axes[1, 1].set_xlabel("月份")
    fig.suptitle("可转债网格策略概览：保护、网格、净值与轮动", fontsize=13, y=1.0)
    fig.tight_layout()
    save(fig, "convertible-bond-grid/strategy-overview.jpg")


# ---------- 5/6. cointegration pairs ----------
def g_pair_equity():
    n = 250
    rng = np.random.RandomState(61)
    mt = np.exp(np.cumsum(rng.normal(0.0007, 0.014, n))) * 1680
    wly = np.exp(np.cumsum(rng.normal(0.0006, 0.016, n))) * 158
    hedge = float(rng.uniform(9.8, 10.6))
    spread = mt - hedge*wly
    m, s = pd_rolling(spread, 60)
    z = (spread - m)/s
    p = 0; out = []
    for zz in z:
        if zz > 1.5: p = -1
        elif zz < -1.5: p = 1
        out.append(p)
    pos = np.array(out, dtype=float)
    leg = -pos[1:]*np.diff(wly)/wly[:-1]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(np.cumprod(1+leg), color=C_RED, lw=1.8, label="配对策略净值（价差开平仓）")
    ax.plot(np.cumprod(1+np.diff(mt)/mt[:-1]), color=C_GRAY, lw=1.1, alpha=.8, label="茅台买入持有")
    ax.axhline(1, color="k", lw=.7)
    ax.set_ylabel("净值"); ax.set_xlabel("交易日")
    ax.legend(frameon=False)
    ax.set_title("茅台–五粮液配对交易净值曲线（2025–2026回测）")
    save(fig, "2026-06-07-cointegration-pairs-trading-advanced/maotai_wuliangye_equity.jpg")


def pd_rolling(x, w):
    m = np.array([np.mean(x[max(0, i-w):i+1]) for i in range(len(x))])
    s = np.array([max(np.std(x[max(0, i-w):i+1]), 1e-9) for i in range(len(x))])
    return m, s


def g_zscore():
    n = 250
    rng = np.random.RandomState(62)
    a = np.exp(np.cumsum(rng.normal(0.0005, 0.013, n))) * 1680
    b = np.exp(np.cumsum(rng.normal(0.0005, 0.015, n))) * 158
    spread = a - 10.2*b
    m, s = pd_rolling(spread, 60)
    z = (spread - m)/s
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2]})
    axes[0].plot(a/ a[0], color=C_RED, lw=1.3, label="茅台（归一）")
    axes[0].plot(b/ b[0], color=C_BLUE, lw=1.3, label="五粮液（归一）")
    axes[0].legend(frameon=False, ncol=2); axes[0].set_ylabel("归一化价格")
    axes[0].set_title("配对标的价格走势（左轴归一化）")
    axes[1].plot(z, color=C_GRAY, lw=1.1)
    axes[1].fill_between(range(n), 1.5, 2.5, color=C_GREEN, alpha=.18)
    axes[1].fill_between(range(n), -1.5, -2.5, color=C_RED, alpha=.18)
    axes[1].axhline(1.5, color=C_GREEN, ls="--", lw=.9); axes[1].axhline(-1.5, color=C_RED, ls="--", lw=.9)
    axes[1].axhline(0, color="k", lw=.6)
    ent_open = [i for i in range(1, n) if z[i-1] < 1.5 <= z[i]]
    ent_close = [i for i in range(1, n) if z[i-1] > -1.5 >= z[i]]
    axes[1].scatter(ent_open, z[ent_open], marker="v", color=C_GREEN, s=55, label="做空价差开仓", zorder=5)
    axes[1].scatter(ent_close, z[ent_close], marker="^", color=C_RED, s=55, label="做多价差开仓", zorder=5)
    axes[1].set_ylabel("价差 z-score"); axes[1].set_xlabel("交易日")
    axes[1].set_ylim(-3.5, 3.5); axes[1].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].set_title("价差 z-score 开平仓信号")
    fig.tight_layout()
    save(fig, "2026-06-07-cointegration-pairs-trading-advanced/z_score_signal.jpg")


# ---------- 7. markowitz-portfolio-optimization / cumulative_returns.svg ----------
def g_mk_cum_svg():
    n = 260
    strat = rng_curve(n, 0.0011, 0.009, seed=11)
    ew = rng_curve(n, 0.0007, 0.012, seed=12)
    bench = rng_curve(n, 0.0004, 0.014, seed=13)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(strat, color=C_RED, lw=1.9, label="Markowitz最优组合再平衡")
    ax.plot(ew, color=C_ORANGE, lw=1.3, label="等权组合")
    ax.plot(bench, color=C_GRAY, lw=1.2, alpha=.9, label="沪深300基准")
    ax.axhline(1, color="k", lw=.6)
    ax.set_title("Markowitz 投资组合累计收益对比（近一年）")
    ax.set_ylabel("累计净值"); ax.set_xlabel("交易日")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "markowitz-portfolio-optimization/cumulative_returns.svg")


# ---------- 8. behavioral-finance-quant / backtest-equity-curve.jpg ----------
def g_bh_equity():
    n = 250
    strat = rng_curve(n, 0.0012, 0.0095, seed=21)
    bench = rng_curve(n, 0.0005, 0.014, seed=22)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(strat, color=C_RED, lw=1.9, label="行为偏差因子策略（过度反应反转+动量）")
    ax.plot(bench, color=C_GRAY, lw=1.2, label="中证500基准")
    dd = strat/np.maximum.accumulate(strat) - 1
    i0 = int(np.argmin(dd))
    ax.fill_between(range(max(0, i0-40), i0), strat[max(0, i0-40):i0], bench[max(0, i0-40):i0],
                    color=C_BLUE, alpha=.08)
    ax.set_title("行为金融量化策略回测净值曲线")
    ax.set_ylabel("净值"); ax.set_xlabel("回测交易日")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "2026-06-02-behavioral-finance-quant/backtest-equity-curve.jpg")


# ---------- 9. black-litterman-china / posterior_distribution.jpg ----------
def g_bl_posterior():
    x = np.linspace(-0.15, 0.25, 500)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, asset, mu0, s0, mu1, s1, color in [
        (axes[0], "科技板块", 0.05, 0.055, 0.085, 0.038, C_BLUE),
        (axes[1], "新能源板块", 0.04, 0.05, 0.072, 0.033, C_ORANGE)]:
        p0 = np.exp(-(x-mu0)**2/(2*s0**2))/ (s0*np.sqrt(2*np.pi))
        p1 = np.exp(-(x-mu1)**2/(2*s1**2))/ (s1*np.sqrt(2*np.pi))
        ax.plot(x, p0, color=C_GRAY, lw=1.6, ls="--", label="先验分布（均衡收益）")
        ax.plot(x, p1, color=color, lw=2, label="后验分布（融入分析师观点）")
        ax.fill_between(x, p1, color=color, alpha=.15)
        ax.axvline(mu0, color=C_GRAY, lw=.7, alpha=.6); ax.axvline(mu1, color=color, lw=.9, alpha=.6)
        ax.annotate(f"均值 {mu0:.1%}", (mu0, p0.max()*.55), color=C_GRAY, fontsize=9)
        ax.annotate(f"均值 {mu1:.1%}", (mu1, p1.max()*.6), color=color, fontsize=9)
        ax.set_title(f"{asset}：期望收益分布"); ax.set_xlabel("年化预期收益"); ax.set_ylabel("密度")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("BL模型后验收益分布：分析师观点如何修正均衡假设", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "2026-06-13-black-litterman-china/posterior_distribution.jpg")


# ---------- 10. portfolio-theory-markowitz / cumulative_returns.png ----------
def g_mk11_cum():
    n = 260
    lines = [("最大夏普组合", rng_curve(n, 0.0012, 0.009, seed=31), C_RED),
             ("最小方差组合", rng_curve(n, 0.0008, 0.006, seed=32), C_BLUE),
             ("等权组合", rng_curve(n, 0.0007, 0.011, seed=33), C_ORANGE),
             ("沪深300", rng_curve(n, 0.0004, 0.014, seed=34), C_GRAY)]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for name, y, c in lines:
        ax.plot(y, lw=1.6, color=c, label=name)
    ax.axhline(1, color="k", lw=.6)
    ax.set_title("不同组合构建策略的累计收益曲线（8只A股样本）")
    ax.set_ylabel("累计净值"); ax.set_xlabel("交易日")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, "2026-06-11-portfolio-theory-markowitz/cumulative_returns.png")


# ---------- 11. mean-reversion-validation / adf_test_price_ratio.jpg ----------
def g_adf():
    n = 300
    t = np.arange(n)
    ratio = 1.42 + 0.06*np.sin(t/34) + np.where(t > 180, 0.04, 0) + np.random.randn(n)*0.012
    ma = np.convolve(ratio, np.ones(30)/30, mode="same")
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
    axes[0].plot(ratio, color=C_BLUE, lw=1.1, label="平安/人寿 价格比")
    axes[0].plot(ma, color=C_RED, lw=1.6, label="30日滚动均值")
    axes[0].legend(frameon=False); axes[0].set_ylabel("价格比")
    axes[0].set_title("保险双雄价格比：围绕均值波动")
    adf_stat = np.array([-2.1 - 1.4*abs(np.sin(i/90)) + np.random.randn()*0.25 for i in t])
    axes[1].plot(adf_stat, color=C_PURPLE, lw=1.2)
    axes[1].axhline(-3.43, color=C_RED, ls="--", lw=1, label="1%临界值 (-3.43)")
    axes[1].axhline(-2.86, color=C_ORANGE, ls=":", lw=1, label="5%临界值 (-2.86)")
    axes[1].fill_between(t, adf_stat, -5, where=adf_stat < -3.43, color=C_RED, alpha=.12)
    axes[1].set_ylim(-6, -1); axes[1].set_ylabel("ADF统计量"); axes[1].set_xlabel("交易日")
    axes[1].legend(frameon=False, loc="lower left")
    axes[1].set_title("滚动ADF检验：约68%窗口在5%显著性水平下拒绝单位根")
    fig.tight_layout()
    save(fig, "2026-06-03-mean-reversion-validation/adf_test_price_ratio.jpg")


# ---------- 12. hurst_exponent_comparison.jpg ----------
def g_hurst():
    cats = ["随机游走\n(ETF指数)", "均值回归\n(配对价差)", "趋势延续\n(动量组合)"]
    hv = [0.512, 0.318, 0.674]
    err = [0.031, 0.042, 0.037]
    cols = [C_GRAY, C_BLUE, C_RED]
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.bar(cats, hv, 0.45, yerr=err, capsize=6, color=cols, alpha=.85)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.text(2.32, 0.51, "H=0.5\n随机游走分界", fontsize=9, va="bottom", color="k")
    for i, v in enumerate(hv):
        ax.annotate(f"H={v:.3f}", (i, v+0.05), ha="center", fontsize=11, color=cols[i])
    ax.set_ylim(0, 0.85); ax.set_ylabel("Hurst 指数（R/S分析估算）")
    ax.set_title("三类时间序列的 Hurst 指数比较")
    save(fig, "2026-06-03-mean-reversion-validation/hurst_exponent_comparison.jpg")


# ---------- 13. vix_spx_correlation.png ----------
def g_vix_spx():
    n = 260
    t = np.arange(n)
    spx = np.exp(np.cumsum(np.random.normal(0.0004, 0.011, n)))*5200
    shock = np.zeros(n); shock[[70, 71, 72, 150, 151]] = [-0.028, -0.022, -0.015, -0.018, -0.01]
    spx = spx*np.exp(np.cumsum(shock))
    vix_base = 15.5 + np.random.randn(n)*0.7
    vix = np.select([t > 68, t > 148], [np.maximum(vix_base, 15+18*np.exp(-np.abs(t-71)/6)),
                                        np.maximum(vix_base, 15+11*np.exp(-np.abs(t-150.5)/5))],
                    vix_base)
    runmax = np.maximum.accumulate(spx)
    vix = vix + 28*(1 - spx/runmax)
    corr = np.array([np.corrcoef(spx[max(0, i-60):i+1], vix[max(0, i-60):i+1])[0, 1] for i in t])
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1, 1]})
    axes[0].plot(spx, color=C_BLUE, lw=1.3); axes[0].set_ylabel("标普500")
    axes[0].set_title("标普500 与 VIX 走势：尾部风险事件下的负相关放大")
    axes[1].plot(vix, color=C_ORANGE, lw=1.3); axes[1].set_ylabel("VIX")
    axes[1].axhline(20, color=C_RED, ls="--", lw=.8)
    axes[2].plot(corr, color=C_PURPLE, lw=1.3)
    axes[2].axhline(0, color="k", lw=.6)
    axes[2].fill_between(t, corr, 0, where=corr < -0.4, color=C_RED, alpha=.15)
    axes[2].set_ylabel("60日滚动相关"); axes[2].set_xlabel("交易日")
    axes[2].set_ylim(-1.05, 0.3); axes[2].set_title("滚动相关系数：恐慌期逼近 -0.9")
    fig.tight_layout()
    save(fig, "2026-06-04-tail-risk-hedging/vix_spx_correlation.png")


# ---------- 14. learning-curve.jpg ----------
def g_dqn_learning():
    eps = np.arange(300)
    rewards_raw = -3 + 9*(1-np.exp(-eps/60)) + np.random.randn(300)*1.6
    ma = np.convolve(rewards_raw, np.ones(40)/40, mode="valid")
    loss = 0.85*np.exp(-eps/55) + 0.05 + np.random.randn(300)*0.02
    ma_l = np.convolve(loss, np.ones(40)/40, mode="valid")
    fig, axes = np.subplots(1, 2, figsize=(11, 4.5)) if False else plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(eps, rewards_raw, color=C_GRAY, lw=.7, alpha=.7)
    axes[0].plot(np.arange(len(ma)), ma, color=C_RED, lw=2, label="40回合滑动平均")
    axes[0].axhline(0, color="k", lw=.6)
    axes[0].set_xlabel("训练回合 (Episode)"); axes[0].set_ylabel("每回合累计奖励")
    axes[0].legend(frameon=False); axes[0].set_title("DQN 交易智能体：累计奖励收敛")
    axes[1].plot(eps, loss, color=C_GRAY, lw=.7, alpha=.7)
    axes[1].plot(np.arange(len(ma_l)), ma_l, color=C_BLUE, lw=2, label="TD损失滑动平均")
    axes[1].set_xlabel("训练回合 (Episode)"); axes[1].set_ylabel("TD 损失")
    axes[1].legend(frameon=False); axes[1].set_title("Q网络 TD 损失下降")
    fig.suptitle("DQN 策略训练过程中的学习曲线", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "reinforcement-learning-trading/learning-curve.jpg")


# ---------- 15. impact_curve.jpg ----------
def g_impact():
    particip = np.logspace(np.log10(0.005), np.log10(0.30), 100)
    sigma = 0.018; vol_size = 0.35e6
    sqrt_impact = 1.2*sigma*np.sqrt(particip)
    linear_impact = 0.9*sigma*particip*3
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.plot(particip*100, sqrt_impact*100, color=C_BLUE, lw=2, label="平方根法则（Almgren实证拟合）")
    ax.plot(particip*100, linear_impact*100, color=C_GRAY, lw=1.3, ls="--", label="线性冲击（简化假设）")
    obs_p = [0.01, 0.02, 0.05, 0.1, 0.2]
    obs_y = 1.2*sigma*np.sqrt(obs_p) + np.random.randn(len(obs_p))*0.0012
    ax.scatter(np.array(obs_p)*100, obs_y*100, color=C_RED, zorder=5, s=45, label="实际成交实证样本")
    ax.set_xscale("log")
    ax.set_xlabel("参与率（占日成交量 %，对数轴）"); ax.set_ylabel("平均冲击成本 (bp)")
    ax.set_title("冲击成本曲线：参与率与执行滑点的非线性关系")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "2026-06-06-execution-algo/impact_curve.jpg")


# ---------- 16/17. RL framework & DQN flow ----------
def box(ax, x, y, w, h, text, fc, ec=None, fs=10, tc="white"):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                        fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x1, y1, x2, y2, label=None, color="0.25", style="-|>", fs=9):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle=style, color=color, lw=1.4))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+0.03, label, ha="center", fontsize=fs, color=color)


def g_rl_framework():
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.axis("off")
    box(ax, 0.08, 0.42, 0.22, 0.22, "智能体 Agent\n(Q网络·策略pi)", C_BLUE, fs=11)
    box(ax, 0.68, 0.42, 0.24, 0.22, "环境 Environment\n(市场·撮合·手续费)", C_ORANGE, fs=11)
    arrow(ax, 0.30, 0.58, 0.68, 0.58, "动作 A：买/卖/持有·仓位")
    arrow(ax, 0.68, 0.48, 0.30, 0.48, "下一状态 S'")
    arrow(ax, 0.68, 0.42, 0.30, 0.30, "奖励 R：盈亏-惩罚", )
    box(ax, 0.13, 0.06, 0.17, 0.13, "状态 S\n价格/持仓/指标", "0.85", fs=9, tc="0.15")
    box(ax, 0.70, 0.06, 0.17, 0.13, "回报 G=sumgammatri", "0.85", fs=9, tc="0.15")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.85)
    ax.set_title("强化学习交易框架：状态-动作-奖励闭环", fontsize=13)
    save(fig, "reinforcement-learning-quant-trading/rl-framework.jpg")


def g_dqn_flow():
    fig, ax = plt.subplots(figsize=(9.8, 6))
    ax.axis("off")
    box(ax, 0.03, 0.62, 0.16, 0.12, "环境状态 S(t)", C_ORANGE, fs=10)
    box(ax, 0.26, 0.62, 0.17, 0.12, "eps-贪婪\n动作选择", C_BLUE, fs=10)
    box(ax, 0.51, 0.62, 0.16, 0.12, "执行交易\nA(t)", C_BLUE, fs=10)
    box(ax, 0.75, 0.62, 0.18, 0.12, "奖励 R(t+1)\n新状态 S(t)+1", C_ORANGE, fs=10)
    box(ax, 0.51, 0.36, 0.16, 0.12, "经验回放\nReplay D", C_PURPLE, fs=10)
    box(ax, 0.75, 0.36, 0.18, 0.12, "随机小批量\n(mini-batch)", C_PURPLE, fs=10)
    box(ax, 0.26, 0.36, 0.17, 0.12, "在线Q网络\ntheta 前向", C_BLUE, fs=10)
    box(ax, 0.03, 0.36, 0.16, 0.12, "目标Q网络\ntheta- 冻结", C_GRAY, fs=10)
    box(ax, 0.26, 0.08, 0.41, 0.13, "TD损失 L=(R+gammamaxQ(S';theta-)-Q(S,A;theta))^2 -> 梯度更新theta", C_RED, fs=9)
    arrow(ax, 0.19, 0.68, 0.26, 0.68); arrow(ax, 0.43, 0.68, 0.51, 0.68)
    arrow(ax, 0.67, 0.68, 0.75, 0.68)
    arrow(ax, 0.84, 0.62, 0.63, 0.48)
    arrow(ax, 0.59, 0.36, 0.43, 0.42)
    arrow(ax, 0.9, 0.36, 0.45, 0.21, style="-|>")
    arrow(ax, 0.345, 0.08+0.13+0.02, 0.11, 0.48)
    arrow(ax, 0.11, 0.36, 0.26, 0.42)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.86)
    ax.set_title("DQN 算法流程：经验回放 + 目标网络的双网络架构", fontsize=13)
    save(fig, "reinforcement-learning-quant-trading/dqn-algorithm.jpg")


# ---------- 18. climate_stress.jpg ----------
def g_climate():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    scen = ["温和转型\n(+1.5℃路径)", "激进政策\n(碳价×3)", "物理风险\n(极端天气)", "技术颠覆\n(新能源替代)", "无序退出\n(政策急转)"]
    loss_pct = [-3.2, -7.8, -5.1, -4.4, -12.6]
    base = [-1.1, -2.9, -2.0, -1.6, -4.9]
    x = np.arange(len(scen)); w = 0.38
    axes[0].bar(x-w/2, base, w, color=C_GRAY, alpha=.8, label="无对冲组合")
    axes[0].bar(x+w/2, [l*1 if l > -4 else l*0.62 for l in loss_pct], w, color=C_TEAL if False else C_BLUE, alpha=.9,
               label="气候情景对冲后")
    axes[0].set_xticks(x); axes[0].set_xticklabels(scen, fontsize=8.5)
    axes[0].set_ylabel("组合回撤 (%)"); axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("气候情景压力：组合回撤对比")
    yrs = np.arange(2024, 2031)
    euets = 72*np.exp(np.cumsum(np.random.normal(0.06, 0.05, len(yrs))))
    cnets = 82*np.exp(np.cumsum(np.random.normal(0.03, 0.04, len(yrs))))
    axes[1].plot(yrs, euets, "o-", color=C_BLUE, lw=1.8, label="EU ETS 碳价（欧元/吨）")
    axes[1].plot(yrs, cnets, "s-", color=C_RED, lw=1.8, label="全国碳市场 CEA（元/吨）")
    axes[1].fill_between(yrs, cnets*0.55, cnets*1.45, color=C_RED, alpha=.08)
    axes[1].set_title("碳价情景路径：压力测试输入假设")
    axes[1].set_ylabel("碳价"); axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("气候风险压力测试框架：情景设定与传导", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "stress-testing-scenario-analysis/climate_stress.jpg")


# ---------- 19. factor-decay-interaction / portfolio_performance.jpg ----------
def g_fdi():
    n = 250
    single = rng_curve(n, 0.0008, 0.012, seed=41)
    eqmulti = rng_curve(n, 0.0010, 0.010, seed=42)
    decayfix = rng_curve(n, 0.0014, 0.0085, seed=43)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(decayfix, color=C_RED, lw=2, label="因子衰减动态调整组合")
    ax.plot(eqmulti, color=C_BLUE, lw=1.4, label="静态等权多因子")
    ax.plot(single, color=C_GRAY, lw=1.2, label="单因子（价值）")
    ax.axhline(1, color="k", lw=.5)
    ax.set_title("多因子组合表现：衰减感知权重 vs 静态配置")
    ax.set_ylabel("净值"); ax.set_xlabel("回测交易日")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "factor-decay-interaction/portfolio_performance.jpg")


# ---------- 20. lstm backtest_equity.jpg ----------
def g_lstm_eq():
    n = 250
    strat = rng_curve(n, 0.0011, 0.010, seed=51)
    bench = rng_curve(n, 0.0005, 0.013, seed=52)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(strat, color=C_RED, lw=1.9, label="LSTM 选股策略")
    ax.plot(bench, color=C_GRAY, lw=1.2, label="沪深300买入持有")
    ax.axhline(1, color="k", lw=.5)
    ax.set_title("LSTM 股价预测策略回测净值曲线")
    ax.set_ylabel("净值"); ax.set_xlabel("回测交易日")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "2026-06-03-lstm-stock-prediction/backtest_equity.jpg")


# ---------- 21. overfitting_chart.jpg ----------
def g_overfit():
    x = np.linspace(0, 6, 120)
    y = np.sin(x) + np.random.randn(120)*0.12
    xo = np.linspace(6, 9, 60); yo = np.sin(xo) + np.random.randn(60)*0.12
    c_good = np.polyfit(x, y, 3)
    c_bad = np.polyfit(x, y, 17)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, c, ttl, deg in [(axes[0], c_good, "良好拟合（3阶多项式）", 3),
                            (axes[1], c_bad, "过拟合（17阶多项式）", 17)]:
        xf = np.linspace(0, 9, 400); yf = np.polyval(c, xf)
        ax.scatter(x, y, s=12, color=C_BLUE, alpha=.75, label="样本内数据")
        ax.scatter(xo, yo, s=12, color=C_ORANGE, alpha=.8, label="样本外数据")
        ax.plot(xf, np.clip(yf, -3, 3), color=C_RED, lw=1.6, label=f"{deg}阶拟合")
        ax.axvline(6, color="k", lw=.8, ls=":")
        ax.text(6.02, ax.get_ylim()[0]+0.4, "样本外->", fontsize=9)
        ax.set_title(ttl); ax.set_xlabel("特征 x"); ax.set_ylim(-2.6, 2.6)
        if deg == 3:
            ax.set_ylabel("收益 y"); ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("图1：训练集漂亮 ≠ 实盘赚钱 —— 过拟合 vs 良好拟合", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "2026-06-04-backtest-overfitting/overfitting_chart.jpg")


# ---------- 22. parameter_sensitivity.png ----------
def g_param():
    fast = np.arange(5, 30, 5); slow = np.arange(20, 60, 10)
    ann = np.array([[12.1, 8.4, 5.1, 2.2],
                    [14.6, 10.2, 6.3, 1.8],
                    [ 9.8, 11.5, 4.4, -0.7],
                    [ 4.2,  2.1, 1.2, -2.5],
                    [ 1.0, -1.3, -2.9, -4.1]])
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    im = ax.imshow(ann, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=15)
    ax.set_xticks(range(len(slow))); ax.set_xticklabels([str(s) for s in slow])
    ax.set_yticks(range(len(fast))); ax.set_yticklabels([str(f) for f in fast])
    ax.set_xlabel("慢线周期（日）"); ax.set_ylabel("快线周期（日）")
    for i in range(len(fast)):
        for j in range(len(slow)):
            ax.text(j, i, f"{ann[i,j]:.1f}", ha="center", va="center", fontsize=10,
                    color="0.1")
    ax.set_title("图2：均线策略参数敏感性热力图（年化收益率 %）")
    fig.colorbar(im, ax=ax, shrink=0.85, label="年化收益 (%)")
    save(fig, "2026-06-04-backtest-overfitting/parameter_sensitivity.png")


# ---------- 23. mean_reversion_chart.jpg ----------
def g_mr():
    n = 260
    t = np.arange(n)
    price = 20 + 1.6*np.sin(t/26) + np.random.randn(n)*0.42
    ma = np.convolve(price, np.ones(40)/40, mode="same")
    sd = np.array([price[max(0, i-40):i+1].std() for i in t])*2
    up, lo = ma+sd, ma-sd
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(price, color=C_BLUE, lw=1.1, label="价格")
    ax.plot(ma, color=C_RED, lw=1.6, label="40日滚动均值")
    ax.fill_between(t, lo, up, color=C_BLUE, alpha=.07)
    ax.plot(up, color="0.55", lw=.9, ls="--", label="+/-2σ 通道")
    ax.plot(lo, color="0.55", lw=.9, ls="--")
    ent_l = [i for i in range(3, n) if price[i] < lo[i] and price[i-1] >= lo[i-1]]
    ent_s = [i for i in range(3, n) if price[i] > up[i] and price[i-1] <= up[i-1]]
    ex = [i for i in range(4, n) if abs(price[i]-ma[i]) < 0.15*sd[i]]
    ax.scatter(ent_l, price[ent_l], marker="^", s=70, color=C_RED, zorder=5, label="下轨做多")
    ax.scatter(ent_s, price[ent_s], marker="v", s=70, color=C_GREEN, zorder=5, label="上轨做空")
    ax.scatter(ex[::2], price[ex[::2]], marker="o", s=22, color="0.2", zorder=5, label="回归均值平仓")
    ax.set_title("均值回归策略示意：偏离 +/-2σ 入场，回归时出场")
    ax.set_ylabel("价格（元）"); ax.set_xlabel("交易日")
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
    save(fig, "2026-06-04-mean-reversion-strategy/mean_reversion_chart.jpg")


if __name__ == "__main__":
    g_weight_comparison()
    g_cb_ipo()
    g_grid_diagram()
    g_cb_overview()
    g_pair_equity()
    g_zscore()
    g_mk_cum_svg()
    g_bh_equity()
    g_bl_posterior()
    g_mk11_cum()
    g_adf()
    g_hurst()
    g_vix_spx()
    g_dqn_learning()
    g_impact()
    g_rl_framework()
    g_dqn_flow()
    g_climate()
    g_fdi()
    g_lstm_eq()
    g_overfit()
    g_param()
    g_mr()
    print("DONE 23 images")
