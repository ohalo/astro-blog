#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oct25g (v2): 10 figures + key stats for
A) smart-beta-dividend-lowvol  因子投资 Smart Beta 红利低波：用质量与分红重构宽基
B) realized-skewness-kurtosis  已实现偏度与峰度：用日内高频矩预测下一周的收益与波动

v2 修正:
- A: 特征时变化（63日已实现特质波动 + 月度慢随机游走的 DY/质量），月度调仓有真实换手；
     低波通道强度随制度放大（熊市 1.6x），牛市植入彩票需求通道（高波被买高）→ 分块不对称。
- B: 植入两条机制（高 RSK 周随后被压回 / 高 RTK 随后波动抬升）+ 跳跃聚集；
     周频多空改用「本周信号→下周持仓」无前视结构。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["figure.facecolor"] = "white"

BASE = "/Users/halo123456/workspace/astro-blog/public/images"
OUT_A = os.path.join(BASE, "smart-beta-dividend-lowvol")
OUT_B = os.path.join(BASE, "realized-skewness-kurtosis")
os.makedirs(OUT_A, exist_ok=True)
os.makedirs(OUT_B, exist_ok=True)

RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
DAYS = 252.0


def ann_stats(r):
    cum = np.prod(1 + r)
    cagr = cum ** (DAYS / len(r)) - 1
    vol = r.std() * np.sqrt(DAYS)
    sr = (r.mean() / r.std() * np.sqrt(DAYS)) if r.std() > 0 else 0.0
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return cagr, vol, sr, dd


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


# ============================================================
# A) smart-beta-dividend-lowvol
# ============================================================
def article_a():
    rng = np.random.default_rng(20261025)
    n_stocks, n_days = 500, 2500
    blocks = [(500, 0.0006, 0.010), (400, -0.0004, 0.016), (500, 0.0004, 0.012),
              (500, -0.0005, 0.018), (600, 0.0007, 0.011)]
    mkt = np.concatenate([rng.normal(mu, sig, n) for n, mu, sig in blocks])
    block_id = np.concatenate([np.full(n, i) for i, (n, _, _) in enumerate(blocks)])
    is_bear = block_id % 2 == 1  # 块2、块4为熊市

    beta = rng.uniform(0.6, 1.7, n_stocks)
    base_vol = np.clip(0.003 + 0.002 * np.abs(rng.normal(0, 1, n_stocks)), 0.0015, 0.012)
    base_dy = np.clip(rng.normal(0.025, 0.02, n_stocks), 0.0, 0.08)
    base_q = rng.normal(0, 1, n_stocks)
    mktcap = np.exp(rng.normal(0, 1.2, n_stocks))

    # 特质波动过程: 每只股票 vol 随时间缓慢演化（GARCH-ish），使「已实现特质波动」成为时变信号
    g = 0.08
    vol_path = np.zeros((n_days, n_stocks))
    vol_path[0] = base_vol
    innov = rng.normal(0, 1, (n_days, n_stocks)) * 0.25
    for d in range(1, n_days):
        vol_path[d] = np.clip(0.94 * vol_path[d - 1] + 0.06 * base_vol * (1 + g * innov[d]),
                              0.0012, 0.015)
    idio = rng.normal(0, 1, (n_days, n_stocks)) * vol_path

    # alpha 通道（基于静态真实特征 + 制度依赖）
    vol_z = (base_vol - base_vol.mean()) / base_vol.std()
    dy_z = (base_dy - base_dy.mean()) / base_dy.std()
    q_z = base_q / base_q.std()
    coef_lv = np.where(is_bear, 0.000352, 0.000154)          # 杠杆约束熊市更紧
    lottery = np.where(is_bear, 0.0, 0.00010)                # 牛市彩票需求把高波买高
    alpha_lv = 0.00020 - coef_lv[:, None] * vol_z[None, :]
    alpha_lot = lottery[:, None] * vol_z[None, :]
    alpha_dy = 0.00010 + 0.00012 * dy_z
    alpha_q = 0.00008 + 0.00010 * q_z
    R = alpha_lv + alpha_lot + alpha_dy[None, :] + alpha_q[None, :] \
        + beta[None, :] * mkt[:, None] + idio

    # 时变可观测特征（调仓信号）:
    # - 特质波动: 用 beta*mkt 回归残差的 63 日已实现波动
    resid = R - (beta[None, :] * mkt[:, None] + alpha_lv + alpha_lot)
    def roll_std(x, w):
        c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
        m = (c[w:] - c[:-w]) / w
        c2 = np.cumsum(np.insert(x ** 2, 0, 0.0, axis=0), axis=0)
        m2 = (c2[w:] - c2[:-w]) / w
        return np.sqrt(np.maximum(m2 - m ** 2, 1e-12))
    rv63 = np.vstack([np.full((63, n_stocks), np.nan), roll_std(resid, 63)])
    # - DY / 质量: 月度慢随机游走（季报节奏 + 测量噪声）
    reb = np.arange(0, n_days, 21)
    dy_t = np.zeros((n_days, n_stocks)); q_t = np.zeros((n_days, n_stocks))
    cur_dy, cur_q = base_dy.copy(), base_q.copy()
    k = 0
    for d0 in reb:
        d1 = min(d0 + 21, n_days)
        if k > 0:
            cur_dy = np.clip(cur_dy + rng.normal(0, 0.002, n_stocks), 0.0, 0.09)
            cur_q = cur_q + rng.normal(0, 0.15, n_stocks)
        dy_t[d0:d1] = cur_dy; q_t[d0:d1] = cur_q
        k += 1
    meas_dy = dy_t + rng.normal(0, 0.003, (n_days, n_stocks))   # 测量噪声
    meas_q = q_t + rng.normal(0, 0.2, (n_days, n_stocks))

    def pct_rank(x):
        return np.argsort(np.argsort(x)) / (len(x) - 1)

    # ---- 图1: 综合分五分组（月度调仓）----
    q_rets = np.zeros((n_days, 5))
    monthly_masks = []
    for ki, d0 in enumerate(reb):
        d1 = min(d0 + 21, n_days)
        if d0 < 63:
            mask = np.ones(n_stocks, dtype=bool)  # 预热期全市场等权
            monthly_masks.append((d0, d1, mask, None))
        else:
            vol_m = rv63[d0]
            sc = (pct_rank(meas_dy[d0]) + pct_rank(-vol_m) + pct_rank(meas_q[d0])) / 3.0
            q_edges = np.quantile(sc, [0, .2, .4, .6, .8, 1.0])
            mask = sc >= q_edges[4]
            for qi in range(5):
                lo, hi = q_edges[qi], q_edges[qi + 1]
                m = (sc >= lo) & (sc <= hi if qi == 4 else sc < hi)
                q_rets[d0:d1, qi] = R[d0:d1, m].mean(axis=1)
            monthly_masks.append((d0, d1, mask, sc))
    q_cum = np.ones((len(reb), 5))
    for kk in range(len(reb) - 1):
        d0, d1, _, _ = monthly_masks[kk]
        for qi in range(5):
            q_cum[kk + 1, qi] = q_cum[kk, qi] * np.prod(1 + q_rets[d0:d1, qi])
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    colors5 = [BLUE, GREEN, ORANGE, GRAY, RED]
    q_stats = []
    for qi in range(5):
        s = ann_stats(q_rets[63:, qi]); q_stats.append(s)
        ax.plot(np.arange(len(reb)), q_cum[:, qi], color=colors5[qi], lw=1.8,
                label=f"Q{qi+1}（{'最低分' if qi==0 else '最高分' if qi==4 else ''}）CAGR {s[0]*100:+.1f}% / SR {s[2]:.2f}")
    ax.set_xlabel("月（21 日/期，前 63 日为预热期）")
    ax.set_ylabel("累计净值（组内等权、月度调仓）")
    ax.set_title("综合分五分组：红利 + 低波 + 质量，年化收益单调递增")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_A, "quintile_cumulative.png")); plt.close(fig)

    # ---- 图2: 三个单因子截面 ----
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    specs = [("股息率 DY", base_dy, "高红利组年化收益更高"),
             ("特质波动（低者更优）", -base_vol, "低波组年化收益更高"),
             ("质量分", base_q, "高质量组年化收益更高")]
    comp_anns = []
    for ax, (name, x, ttl) in zip(axes, specs):
        qs = np.quantile(x, [0, .2, .4, .6, .8, 1.0])
        anns = []
        for qi in range(5):
            lo, hi = qs[qi], qs[qi + 1]
            m = (x >= lo) & (x <= hi if qi == 4 else x < hi)
            anns.append(ann_stats(R[63:, m].mean(axis=1))[0] * 100)
        comp_anns.append(anns)
        bars = ax.bar([f"Q{i+1}" for i in range(5)], anns, color=[RED, GRAY, GRAY, GRAY, GREEN])
        ax.axhline(0, color=BLACK, lw=.8)
        for b, v in zip(bars, anns):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.3 if v >= 0 else -0.9),
                    f"{v:+.1f}%", ha="center", fontsize=8)
        ax.set_title(f"{name}\n{ttl}", fontsize=9.5)
        ax.set_ylabel("年化收益 %"); ax.grid(alpha=.3, axis="y")
    fig.suptitle("三条植入通道的截面单调性（预热期后、组内等权）", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_A, "component_spreads.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- 三个因子价差序列（时序归因用）----
    def factor_spread(x, hi_good=True):
        hi = x >= np.quantile(x, 0.8); lo = x <= np.quantile(x, 0.2)
        return R[:, hi].mean(axis=1) - R[:, lo].mean(axis=1)
    s_lv = factor_spread(-base_vol); s_dy = factor_spread(base_dy); s_q = factor_spread(base_q)

    # ---- 组合: SB vs EW vs Cap ----
    sb = np.zeros(n_days); ew = R.mean(axis=1)
    cap_w = mktcap / mktcap.sum(); cap = R @ cap_w
    prev_w = None; turnover_m = []
    for d0, d1, mask, sc in monthly_masks:
        if sc is None:
            w = np.ones(n_stocks) / n_stocks
        else:
            w = np.where(mask, sc, 0.0)
            w = w / w.sum()
        sb[d0:d1] = R[d0:d1] @ w
        if prev_w is not None and d0 >= 63:
            turnover_m.append(0.5 * np.abs(w - prev_w).sum())
        prev_w = w
    turnover_m = np.array(turnover_m)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for r, name, c in [(sb, "红利低波 Smart Beta（分数加权）", RED),
                       (ew, "等权基准（500 只）", GRAY), (cap, "市值加权基准", BLUE)]:
        s = ann_stats(r[63:])
        ax.plot(np.cumprod(1 + r[63:]), lw=1.7, color=c,
                label=f"{name}  CAGR {s[0]*100:+.1f}% / SR {s[2]:.2f} / DD {s[3]*100:.1f}%")
    ax.set_title("Smart Beta 组合 vs 基准（月度调仓、无成本）")
    ax.set_xlabel("交易日（预热期后）"); ax.set_ylabel("累计净值")
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_A, "sb_vs_benchmark.png")); plt.close(fig)
    sb_stats, ew_stats, cap_stats = ann_stats(sb[63:]), ann_stats(ew[63:]), ann_stats(cap[63:])

    # ---- 图4: 归因 ----
    X = np.column_stack([s_lv, s_dy, s_q, np.ones(n_days)])
    y = sb - cap
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_a = y - X @ coef
    r2 = 1 - resid_a.var() / y.var()
    contrib = coef[:3] * np.array([s_lv.mean(), s_dy.mean(), s_q.mean()])
    names = ["低波价差", "红利价差", "质量价差"]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    vals = list(contrib) + [y[63:].mean() - contrib.sum()]
    bars = ax.bar(names + ["残差"], vals, color=[BLUE, GREEN, ORANGE, GRAY])
    ax.axhline(0, color=BLACK, lw=.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.07),
                f"{v*1e4:+.1f}bp/日", ha="center", fontsize=9)
    ax.set_ylabel("日超额收益贡献 bp")
    ax.set_title(f"Smart Beta 日超额归因于三条价差（R²={r2:.2f}）\n"
                 f"低波 {contrib[0]/y[63:].mean()*100:.0f}% / 红利 {contrib[1]/y[63:].mean()*100:.0f}% / "
                 f"质量 {contrib[2]/y[63:].mean()*100:.0f}% / 残差 {vals[3]/y[63:].mean()*100:.0f}%")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_A, "attribution.png")); plt.close(fig)

    # ---- 图5: 成本 + 分块 ----
    costs = [0, 10, 30, 50]
    cagrs = []
    for cbps in costs:
        cost_d = turnover_m.mean() * cbps / 1e4 / 21
        cagrs.append(ann_stats(sb[63:] - cost_d)[0] * 100)
    blk_excess = []
    for i, (n, mu, sig) in enumerate(blocks):
        m = block_id == i
        blk_excess.append(ann_stats(sb[m & (np.arange(n_days) >= 63)])[0] * 100
                          - ann_stats(cap[m & (np.arange(n_days) >= 63)])[0] * 100)
    blk_names = [f"块{i+1}\n{'牛' if mu > 0 else '熊'}" for i, (n, mu, sig) in enumerate(blocks)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    ax1.plot(costs, cagrs, "o-", color=RED, lw=1.8)
    for c, v in zip(costs, cagrs):
        ax1.text(c, v + 0.15, f"{v:+.1f}%", ha="center", fontsize=8.5)
    ax1.set_xlabel("单边成本（bp/单位换手）"); ax1.set_ylabel("SB 年化 %")
    ax1.set_title(f"成本敏感性（月均单边换手 {turnover_m.mean()*100:.0f}%）")
    ax1.grid(alpha=.3)
    bars = ax2.bar(blk_names, blk_excess, color=[GREEN if v >= 0 else RED for v in blk_excess])
    ax2.axhline(0, color=BLACK, lw=.8)
    for b, v in zip(bars, blk_excess):
        ax2.text(b.get_x() + b.get_width() / 2, v + (0.15 if v >= 0 else -0.45),
                 f"{v:+.1f}pp", ha="center", fontsize=8.5)
    ax2.set_ylabel("SB−市值基准 年化 pp")
    ax2.set_title("分制度块超额：熊市块放大，牛市块被彩票需求对冲")
    ax2.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_A, "cost_subperiod.png")); plt.close(fig)

    print("=== A smart-beta-dividend-lowvol ===")
    for qi in range(5):
        s = q_stats[qi]
        print(f"Q{qi+1}: CAGR {s[0]*100:+.2f}% vol {s[1]*100:.2f}% SR {s[2]:.2f} DD {s[3]*100:.1f}%")
    print(f"SB : CAGR {sb_stats[0]*100:+.2f}% vol {sb_stats[1]*100:.2f}% SR {sb_stats[2]:.2f} DD {sb_stats[3]*100:.1f}%")
    print(f"EW : CAGR {ew_stats[0]*100:+.2f}% vol {ew_stats[1]*100:.2f}% SR {ew_stats[2]:.2f} DD {ew_stats[3]*100:.1f}%")
    print(f"Cap: CAGR {cap_stats[0]*100:+.2f}% vol {cap_stats[1]*100:.2f}% SR {cap_stats[2]:.2f} DD {cap_stats[3]*100:.1f}%")
    print(f"SB-Cap 日超额 {y[63:].mean()*1e4:+.2f}bp, R2={r2:.3f}")
    print(f"contrib bp/day: lv {contrib[0]*1e4:+.2f} dy {contrib[1]*1e4:+.2f} q {contrib[2]*1e4:+.2f} resid {vals[3]*1e4:+.2f}")
    print(f"月均单边换手 {turnover_m.mean()*100:.1f}%  成本后CAGR {[round(v,2) for v in cagrs]}")
    print(f"分块超额 pp: {[round(v,2) for v in blk_excess]}")
    print(f"单因子Q1->Q5 年化: DY {[round(v,1) for v in comp_anns[0]]}")
    print(f"                VOL {[round(v,1) for v in comp_anns[1]]}")
    print(f"                Q  {[round(v,1) for v in comp_anns[2]]}")
    return sb, cap, ew, s_lv, s_dy, s_q, turnover_m, q_stats, blk_excess


# ============================================================
# B) realized-skewness-kurtosis
# ============================================================
def article_b():
    rng = np.random.default_rng(4242)
    n_days, bars = 1250, 390
    t = np.linspace(0, 1, bars)
    seasonal = 1.0 + 0.9 * (np.exp(-((t - 0.02) / 0.06) ** 2) + np.exp(-((t - 0.98) / 0.06) ** 2)) - 0.15
    seasonal = seasonal / seasonal.mean()

    phi, mu_lv, sd_eta = 0.98, np.log(0.0006), 0.09   # 1min bar 波动 ~0.06% → 日波动 ~1.2%

    # 机制参数（植入）。注意 RTK 的无条件均值被日内 U 型波动机械抬到 ~4.7
    # （异质波动下 E[n·Σr⁴/RV²] > 3），机制中心必须用这个经验基准而不是 3。
    LAM_RSK = -0.004    # 每单位 RSK 对未来5日总收益的漂移（彩票需求→反转）
    LAM_RTK = 0.025     # 每单位 (RTK−基准) 对 log 波动的即时冲击（AR 衰减，不复合）
    RHO_S = 0.88        # 波动信号衰减系数（半衰期 ~5.4 日，覆盖 5 日预测窗）
    J0 = 0.02           # 基础跳跃强度
    J1 = 0.08           # 跳跃聚集

    def simulate(lam_rsk, lam_rtk, rtk_base, seed):
        """日内 1min bar 模拟。lam_* 为植入机制强度（0=无机制对照世界）。
        RSK 机制: 当日 RSK 给未来 5 日播种收益漂移（彩票需求→反转）。
        RTK 机制: 当日 RTK 冲击 log 波动信号，信号按 AR(ρ=0.7) 衰减（不复合爆炸）。"""
        rng2 = np.random.default_rng(seed)
        b_rets = np.zeros((n_days, bars))
        d_ret = np.zeros(n_days)
        rv_a = np.zeros(n_days); rsk_a = np.zeros(n_days); rtk_a = np.zeros(n_days)
        j_day = np.zeros(n_days, dtype=bool)
        drift_f = np.zeros(n_days)
        sig = 0.0                          # 当前 log 波动信号
        sig_path = np.zeros(n_days)
        lv_cur = mu_lv
        for d in range(n_days):
            lv_cur = mu_lv + phi * (lv_cur - mu_lv) + rng2.normal(0, sd_eta)
            sd = np.exp(lv_cur + sig)
            br = sd * seasonal * rng2.standard_normal(bars)
            jp = J1 if (d > 0 and j_day[d - 1]) else J0
            if rng2.random() < jp:
                j_day[d] = True
                br[rng2.integers(0, bars)] += -(0.003 + 0.002 * np.abs(rng2.standard_normal()))
            b_rets[d] = br
            rv_d = (br ** 2).sum()
            rv_a[d] = rv_d
            rsk_a[d] = np.sqrt(bars) * (br ** 3).sum() / rv_d ** 1.5
            rtk_a[d] = bars * (br ** 4).sum() / rv_d ** 2
            d_ret[d] = br.sum() + drift_f[d]
            # 播种未来 5 日收益漂移 + 更新波动信号（AR，不复合）
            for k in range(1, 6):
                if d + k < n_days:
                    drift_f[d + k] += lam_rsk * rsk_a[d] / 5
            sig = RHO_S * sig + lam_rtk * (rtk_a[d] - rtk_base)
            sig_path[d] = sig
        return b_rets, d_ret, rv_a, rsk_a, rtk_a, j_day, sig_path

    # 第一遍：无机制世界 → 测 RTK 经验基准 + 对照相关性
    _, dr0, rv0, rsk0, rtk0, jd0, _ = simulate(0.0, 0.0, 0.0, 4242)
    RTK_BASE = float(rtk0.mean())
    nw_ret0 = np.array([dr0[d + 1:d + 6].sum() for d in range(n_days - 5)])
    nw_rv0 = np.array([rv0[d + 1:d + 6].mean() for d in range(n_days - 5)])
    print(f"[对照世界·无机制] corr(RSK,nw_ret)={corr(rsk0[:-5], nw_ret0):+.4f} "
          f"corr(RTK,nw_rv)={corr(rtk0[:-5], nw_rv0):+.4f}  RTK经验基准={RTK_BASE:.2f}")

    # 第二遍：植入机制（同一随机种子 → 同一条基础路径，只有机制项不同）
    bar_rets, daily_ret, rv, rsk, rtk, jump_day, sig_path = simulate(LAM_RSK, LAM_RTK, RTK_BASE, 4242)
    nw_ret = np.array([daily_ret[d + 1:d + 6].sum() for d in range(n_days - 5)])
    nw_rv = np.array([rv[d + 1:d + 6].mean() for d in range(n_days - 5)])
    L = len(nw_ret)
    rsk_l, rtk_l, rv_l = rsk[:-5], rtk[:-5], rv[:-5]

    def ols(X, y):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
        n, k = X.shape
        s2 = resid @ resid / (n - k)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        return b, se

    print("=== B realized-skewness-kurtosis ===")
    print(f"corr(RSK, 未来5日收益) = {corr(rsk_l, nw_ret):+.4f}  (t={corr(rsk_l,nw_ret)*np.sqrt(L):+.2f})")
    print(f"corr(RTK, 未来5日RV)  = {corr(rtk_l, nw_rv):+.4f}")
    print(f"corr(RV,  未来5日RV)  = {corr(rv_l, nw_rv):+.4f}")
    print(f"corr(RSK, RV) = {corr(rsk_l, rv_l):+.4f}, corr(RTK, RV) = {corr(rtk_l, rv_l):+.4f}")
    print(f"RSK mean {rsk_l.mean():+.3f} std {rsk_l.std():.3f} min {rsk_l.min():.2f} max {rsk_l.max():.2f}")
    print(f"RTK mean {rtk_l.mean():.3f} std {rtk_l.std():.3f}")
    print(f"RSK自相关 lag1 {corr(rsk_l[1:], rsk_l[:-1]):+.3f}, RTK自相关 lag1 {corr(rtk_l[1:], rtk_l[:-1]):+.3f}")
    print(f"跳跃日 {jump_day.sum()} 天 ({jump_day.mean()*100:.1f}%)")
    jmask = jump_day[:L]
    print(f"跳跃日 RSK 均值 {rsk_l[jmask].mean():+.2f} vs 非跳跃日 {rsk_l[~jmask].mean():+.2f}")

    # 回归: nw_ret ~ RSK + RV
    X = np.column_stack([rsk_l, rv_l, np.ones(L)])
    b_reg, se_reg = ols(X, nw_ret)
    print(f"回归 nw_ret ~ RSK+RV: RSK {b_reg[0]*1e4:+.1f}bp/单位 (t={b_reg[0]/se_reg[0]:+.1f}), RV {b_reg[1]:+.4f} (t={b_reg[1]/se_reg[1]:+.1f})")
    X2 = np.column_stack([rtk_l, rv_l, np.ones(L)])
    b2, se2 = ols(X2, nw_rv)
    print(f"回归 nw_rv ~ RTK+RV: RTK {b2[0]:+.6f} (t={b2[0]/se2[0]:+.1f}), RV {b2[1]:+.3f} (t={b2[1]/se2[1]:+.1f})")
    print(f"[信号诊断] sig_path std {sig_path.std():.4f}; corr(sig, nw_rv) {corr(sig_path[:-5], nw_rv):+.4f}; log(nw_rv) std {np.log(nw_rv).std():.3f}; RTK十分位未来日波动%: "
          f"{[round(float(np.sqrt(nw_rv[(rtk_l>=np.quantile(rtk_l,i/10))&(rtk_l<np.quantile(rtk_l,(i+1)/10) if i<9 else rtk_l<=np.quantile(rtk_l,1.0))].mean())*100),3) for i in range(10)]}")

    # ---- 图1 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 5.2), sharex=True)
    jd = np.where(jump_day[:-5])[0]
    ax1.plot(rsk_l, color=BLUE, lw=.7)
    ax1.scatter(jd, rsk_l[jd], color=RED, s=14, zorder=5, label=f"真实跳跃日（{len(jd)} 天）")
    ax1.axhline(0, color=BLACK, lw=.6)
    ax1.set_ylabel("RSK（已实现偏度）")
    ax1.set_title(f"日度已实现偏度：均值 {rsk_l.mean():+.2f}；最深的负偏尖峰几乎全部来自跳跃日")
    ax1.legend(fontsize=8); ax1.grid(alpha=.3)
    ax2.plot(rtk_l, color=PURPLE, lw=.7)
    ax2.scatter(jd, rtk_l[jd], color=RED, s=14, zorder=5)
    ax2.axhline(RTK_BASE, color=GRAY, lw=.8, ls="--", label=f"经验基准 {RTK_BASE:.2f}（日内U型波动机械膨胀）")
    ax2.set_ylabel("RTK（已实现峰度）"); ax2.set_xlabel("交易日")
    ax2.set_title(f"日度已实现峰度：均值 {rtk_l.mean():.2f}、std {rtk_l.std():.2f}")
    ax2.legend(fontsize=8); ax2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_B, "rsk_rtk_daily.png")); plt.close(fig)

    # ---- 图2: 十分位预测 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    qs = np.quantile(rsk_l, np.linspace(0, 1, 11))
    def dec_stats(x, y, qs):
        vals, ses = [], []
        for i in range(10):
            m = (x >= qs[i]) & (x < qs[i + 1] if i < 9 else x <= qs[i + 1])
            vals.append(y[m].mean()); ses.append(y[m].std() / np.sqrt(max(m.sum(), 1)))
        return np.array(vals), np.array(ses)
    v, se = dec_stats(rsk_l, nw_ret, qs)
    ax1.bar(range(1, 11), v * 100, yerr=se * 100, color=BLUE, alpha=.85, capsize=2)
    ax1.axhline(0, color=BLACK, lw=.8)
    p = np.polyfit(range(1, 11), v, 1)
    ax1.plot(range(1, 11), np.polyval(p, range(1, 11)) * 100, color=RED, lw=1.5, ls="--",
             label=f"斜率 {p[0]*100*100:+.1f}bp/十分位")
    ax1.set_xlabel("当日 RSK 十分位（1=最低）"); ax1.set_ylabel("未来5日收益均值 %")
    ax1.set_title(f"RSK 十分位 vs 未来一周收益\n整体 corr {corr(rsk_l, nw_ret):+.3f}")
    ax1.legend(fontsize=8.5); ax1.grid(alpha=.3, axis="y")
    qs2 = np.quantile(rtk_l, np.linspace(0, 1, 11))
    # 剔除「当前 RV 可预测部分」后的未来波动（隔离 RTK 的增量信息）
    Xrv = np.column_stack([rv_l, np.ones(L)])
    b_rv, _ = ols(Xrv, nw_rv)
    nw_rv_resid = nw_rv - Xrv @ b_rv
    v2, se2 = dec_stats(rtk_l, nw_rv_resid, qs2)
    ax2.bar(range(1, 11), np.sqrt(v2 + nw_rv.mean()) * 100,
            yerr=np.sqrt(np.abs(se2)) * 100, color=PURPLE, alpha=.85, capsize=2)
    p2 = np.polyfit(range(1, 11), v2, 1)
    ax2.plot(range(1, 11), np.sqrt(np.polyval(p2, range(1, 11)) + nw_rv.mean()) * 100,
             color=RED, lw=1.5, ls="--")
    ax2.set_xlabel("当日 RTK 十分位（1=最低）")
    ax2.set_ylabel("未来5日平均日波动率 %")
    ax2.set_title(f"RTK 十分位 vs 未来一周波动（已剔除当前 RV 可预测部分）\n原始 corr {corr(rtk_l, nw_rv):+.3f} → 控制 RV 后 t={b2[0]/se2[0]:+.1f}")
    ax2.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_B, "predictive_bins.png")); plt.close(fig)

    # ---- 图3: 抽样频率（从 1min 聚合）----
    freqs = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "60min": 60}
    f_names, ac1s, ac2s, preds, mean_abs, tvals, sig_corr = [], [], [], [], [], [], []
    for name, step in freqs.items():
        if step == 1:
            agg = bar_rets
        else:
            nb = bars // step
            agg = bar_rets[:, :nb * step].reshape(n_days, nb, step).sum(axis=2)
        rv_a = (agg ** 2).sum(axis=1)
        rsk_a = np.sqrt(agg.shape[1]) * (agg ** 3).sum(axis=1) / rv_a ** 1.5
        rtk_a = agg.shape[1] * (agg ** 4).sum(axis=1) / rv_a ** 2
        rsk_a, rtk_a, rv_a = rsk_a[:-5], rtk_a[:-5], rv_a[:-5]
        f_names.append(name)
        ac1s.append(corr(rsk_a[1:], rsk_a[:-1])); ac2s.append(corr(rtk_a[1:], rtk_a[:-1]))
        c = corr(rsk_a, nw_ret); preds.append(c); tvals.append(c * np.sqrt(L))
        mean_abs.append(np.abs(rsk_a).mean())
        sig_corr.append(corr(rsk_a, rsk_l))   # 与 1min「真信号」的相关（信号衰减通道）
    print("频率:", {n: f"RSK_ac1={a:+.3f} pred={p:+.3f}(t={tt:+.1f}) |RSK|={m:.2f} vs1min={s:+.2f}"
                  for n, a, p, tt, m, s in zip(f_names, ac1s, preds, tvals, mean_abs, sig_corr)})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    ax1.plot(f_names, sig_corr, "o-", color=RED, lw=1.8, label="corr(粗抽样 RSK, 1min RSK)")
    ax1.plot(f_names, ac1s, "s--", color=BLUE, lw=1.5, label="RSK lag-1 自相关")
    ax1.plot(f_names, ac2s, "^--", color=PURPLE, lw=1.5, label="RTK lag-1 自相关")
    ax1.set_ylabel("相关系数")
    ax1.set_title("粗抽样 = 真信号衰减（红）+ 高阶矩近似白噪声（蓝/紫）")
    ax1.legend(fontsize=8); ax1.grid(alpha=.3)
    ax2.plot(f_names, preds, "o-", color=RED, lw=1.8, label="corr(RSK, 未来5日收益)")
    ax2.axhline(0, color=BLACK, lw=.7)
    ax2b = ax2.twinx()
    ax2b.plot(f_names, mean_abs, "s-", color=GRAY, lw=1.5, label="mean|RSK|（噪声尺度）")
    ax2b.set_ylabel("mean|RSK|", color=GRAY); ax2b.tick_params(axis="y", colors=GRAY)
    ax2.set_ylabel("预测相关", color=RED); ax2.tick_params(axis="y", colors=RED)
    ax2.set_title("预测力（红）随抽样变粗而衰减；噪声尺度（灰）上升")
    ax2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_B, "sampling_frequency.png")); plt.close(fig)

    # ---- 图4: 五分位 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    q_edges = [np.quantile(rsk_l, [0, .2, .4, .6, .8, 1.0]), np.quantile(rtk_l, [0, .2, .4, .6, .8, 1.0])]
    for ax, x, name, c, qe in [(axes[0], rsk_l, "RSK", BLUE, q_edges[0]),
                                (axes[1], rtk_l, "RTK", PURPLE, q_edges[1])]:
        vals, ses = [], []
        for i in range(5):
            m = (x >= qe[i]) & (x <= qe[i + 1] if i == 4 else x < qe[i + 1])
            vals.append(nw_ret[m].mean() * 100); ses.append(nw_ret[m].std() / np.sqrt(max(m.sum(), 1)) * 100)
        ax.bar([f"Q{i+1}" for i in range(5)], vals, yerr=ses, color=c, alpha=.85, capsize=3)
        ax.axhline(0, color=BLACK, lw=.8)
        ax.set_xlabel(f"{name} 五分位（Q1 最低）"); ax.set_ylabel("未来5日收益均值 %")
        ax.set_title(f"按当日 {name} 五分位\nQ1 {vals[0]:+.3f}% vs Q5 {vals[4]:+.3f}%，差 {vals[0]-vals[4]:+.3f}%")
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_B, "quintile_spread.png")); plt.close(fig)

    # ---- 图5: 周频多空（本周信号 → 下周持仓，无前视）----
    wk = 5
    n_wk = (n_days - 5) // wk
    sig = np.array([rsk[5 * w:5 * w + 5].mean() for w in range(n_wk)])          # 第 w 周信号
    fr = np.array([daily_ret[5 * w + 5:5 * w + 10].sum() for w in range(n_wk)])  # 第 w+1 周收益
    pos = np.zeros(n_wk)
    for w in range(n_wk):
        hist = sig[:w + 1]
        if w < 26:
            continue
        hi, lo = np.quantile(hist[-104:], 0.8), np.quantile(hist[-104:], 0.2)
        if sig[w] <= lo: pos[w] = 1.0
        elif sig[w] >= hi: pos[w] = -1.0
    pnl = pos * fr
    turn_w = np.abs(np.diff(pos, prepend=0)).sum() / 2 / n_wk
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(np.cumprod(1 + pnl), color=RED, lw=1.7,
            label=f"无成本：CAGR {((1+pnl).prod()**(52/n_wk)-1)*100:+.1f}% / SR {pnl.mean()/pnl.std()*np.sqrt(52):.2f}")
    for cbps, c in [(10, GRAY), (30, BLUE)]:
        cost = cbps / 1e4
        pnl_c = pnl - cost * np.abs(np.diff(pos, prepend=0))
        ax.plot(np.cumprod(1 + pnl_c), lw=1.4, ls="--", color=c,
                label=f"成本 {cbps}bp/次：CAGR {((1+pnl_c).prod()**(52/n_wk)-1)*100:+.1f}% / SR {pnl_c.mean()/pnl_c.std()*np.sqrt(52):.2f}")
    ax.axhline(1, color=BLACK, lw=.7)
    ax.set_xlabel("周"); ax.set_ylabel("累计净值（周频多空）")
    ax.set_title(f"RSK 反转多空：做多低偏度周、做空高偏度周（周均换手 {turn_w*100:.0f}%）")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_B, "strategy_equity.png")); plt.close(fig)

    q1r = nw_ret[rsk_l <= np.quantile(rsk_l, .2)].mean() * 100
    q5r = nw_ret[rsk_l >= np.quantile(rsk_l, .8)].mean() * 100
    print(f"五分位: Q1 {q1r:+.3f}% vs Q5 {q5r:+.3f}% 差 {q1r-q5r:+.3f}%")
    print(f"策略: 无成本CAGR {((1+pnl).prod()**(52/n_wk)-1)*100:+.1f}% SR {pnl.mean()/pnl.std()*np.sqrt(52):.2f}, 周换手 {turn_w*100:.0f}%")
    for cbps in (10, 30):
        cost = cbps / 1e4
        pnl_c = pnl - cost * np.abs(np.diff(pos, prepend=0))
        print(f"  成本{cbps}bp: CAGR {((1+pnl_c).prod()**(52/n_wk)-1)*100:+.1f}% SR {pnl_c.mean()/pnl_c.std()*np.sqrt(52):.2f}")


if __name__ == "__main__":
    article_a()
    article_b()
    print("DONE")
