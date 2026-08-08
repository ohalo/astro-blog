#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
尾部对冲的成本拖累：保险费到底吃掉多少长期收益
所有图表与数字由真实蒙特卡洛计算生成，固定随机种子可复现。
按月分块向量化：外层 240 个月，内层 21 天一次性生成。
SEED = 20260807
"""
import json, os, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import erf, sqrt, log, exp

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.facecolor"] = "white"

IMG_DIR = "/Users/halo/workspace/astro-blog/public/images/tail-hedge-cost-drag"
os.makedirs(IMG_DIR, exist_ok=True)

SEED = 20260807

C_HEDGE = "#1e3a5f"
C_NAKED = "#dc2626"
C_OK    = "#16a34a"
C_WARN  = "#e05c2a"
C_PUR   = "#9333ea"
C_CASH  = "#2563eb"
C_GREY  = "#9ca3af"

# ============================================================
# 真值（由构造给定）
# ============================================================
TD       = 252
TENOR    = 21                 # 每月滚动一次
N_MONTHS = 240                # 20 年
N_YEARS  = N_MONTHS * TENOR / TD
N_PATHS  = 20000
N_SCAN   = 6000

MU_TRUE  = 0.07               # 标的年化对数漂移（含跳跃补偿后）
SIG_TRUE = 0.16               # 扩散年化波动
RF       = 0.02

JLAM     = 0.25               # 危机跳跃：年均 0.25 次
JMU      = -0.18
JSD      = 0.10

HEDGE_MONEY = 0.90            # 执行价 = 90% 现价
LOAD        = 0.35            # 保险加成：保费 = 精算公平值 × (1+LOAD)

_FAIR_CACHE = {}


def fair_premium(mu, sig, jlam, jmu, jsd, moneyness, tenor, n=4_000_000):
    """精算公平保费 = 真实测度下的期望赔付（月度，已贴现）。
    这样 累计赔付/累计保费 应精确收敛到 1/(1+LOAD)。"""
    key = (round(mu, 8), round(sig, 8), round(jlam, 8), round(jmu, 8),
           round(jsd, 8), round(moneyness, 8), tenor)
    if key in _FAIR_CACHE:
        return _FAIR_CACHE[key]
    rng = np.random.default_rng(987654321)
    dt = 1.0 / TD
    drift = (mu - jlam * jmu - 0.5 * sig ** 2) * dt
    sq = sig * sqrt(dt)
    tot = 0.0
    done = 0
    block = 200_000
    while done < n:
        b = min(block, n - done)
        z = rng.standard_normal((tenor, b))
        jn = rng.random((tenor, b)) < jlam * dt
        jz = jmu + jsd * rng.standard_normal((tenor, b))
        lr = drift + sq * z + np.where(jn, jz, 0.0)
        st = np.exp(lr.sum(axis=0))
        tot += float(np.maximum(moneyness - st, 0.0).sum())
        done += b
    val = tot / n * exp(-RF * tenor / TD)
    _FAIR_CACHE[key] = val
    return val


def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def simulate(n_paths=N_PATHS, n_months=N_MONTHS, mu=MU_TRUE, sig=SIG_TRUE,
             jlam=JLAM, jmu=JMU, jsd=JSD, load=LOAD,
             moneyness=HEDGE_MONEY, hedge_ratio=1.0, tenor=TENOR,
             cash_weights=None, seed=SEED, keep_monthly=False):
    """
    对冲组合 vs 不对冲组合，共用同一组随机数（精确配对）。
    cash_weights: 额外同步跟踪的「减仓+现金」对照组权重列表。
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TD
    T_opt = tenor / TD
    k_jump = jlam * jmu
    drift = (mu - k_jump - 0.5 * sig ** 2) * dt
    sq = sig * sqrt(dt)
    rf_d = RF / TD

    fair = fair_premium(mu, sig, jlam, jmu, jsd, moneyness, tenor)
    put_px = fair * (1.0 + load)
    prem = hedge_ratio * put_px

    cw = list(cash_weights) if cash_weights else []
    n_cw = len(cw)

    eq_n = np.ones(n_paths)
    eq_h = np.ones(n_paths)
    eq_c = np.ones((n_cw, n_paths)) if n_cw else None

    peak_n = np.ones(n_paths); mdd_n = np.zeros(n_paths)
    peak_h = np.ones(n_paths); mdd_h = np.zeros(n_paths)
    peak_c = np.ones((n_cw, n_paths)) if n_cw else None
    mdd_c = np.zeros((n_cw, n_paths)) if n_cw else None

    tot_prem = 0.0
    tot_pay = 0.0
    n_pay = 0
    rel_list = []          # 每月 (对冲 − 不对冲) 相对收益
    mon_n = np.zeros((n_months + 1, n_paths), dtype=np.float32) if keep_monthly else None
    mon_h = np.zeros((n_months + 1, n_paths), dtype=np.float32) if keep_monthly else None
    if keep_monthly:
        mon_n[0] = 1.0; mon_h[0] = 1.0

    worst_day = 0.0
    day_samples = []

    for m in range(n_months):
        # ---- 月初买保险：保费从对冲组合净值中扣除 ----
        eq_h_open = eq_h * (1.0 - prem)
        eq_h = eq_h_open
        tot_prem += prem

        # ---- 生成本月 21 天 ----
        z = rng.standard_normal((tenor, n_paths))
        jn = rng.random((tenor, n_paths)) < jlam * dt
        jz = jmu + jsd * rng.standard_normal((tenor, n_paths))
        logret = drift + sq * z + np.where(jn, jz, 0.0)
        gross = np.exp(logret)
        cum = np.cumprod(gross, axis=0)          # 月内累计标的因子

        if m < 120:
            day_samples.append((gross[:, :800] - 1.0).ravel())
        worst_day = min(worst_day, float((gross - 1.0).min()))

        # ---- 月内逐日净值（用于回撤）----
        path_n = eq_n * cum
        path_h = eq_h * cum
        # running peak = max(历史峰值, 月内累计最高)，向量化
        run_pk_n = np.maximum(peak_n[None, :], np.maximum.accumulate(path_n, axis=0))
        run_pk_h = np.maximum(peak_h[None, :], np.maximum.accumulate(path_h, axis=0))
        mdd_n = np.maximum(mdd_n, (1.0 - path_n / run_pk_n).max(axis=0))
        mdd_h = np.maximum(mdd_h, (1.0 - path_h / run_pk_h).max(axis=0))
        peak_n = run_pk_n[-1]
        peak_h = run_pk_h[-1]

        eq_n_new = path_n[-1]
        eq_h_new = path_h[-1]

        # ---- 月末期权结算：执行价按月初现价的 moneyness ----
        underlying_ret = cum[-1]                       # 月度标的总因子
        intrinsic = np.maximum(moneyness - underlying_ret, 0.0)
        pay = hedge_ratio * intrinsic
        # 赔付按月初对冲组合规模计（名义 = 月初市值）
        eq_h_new = eq_h_new + pay * eq_h_open
        tot_pay += float(pay.mean())
        n_pay += int((pay > 0).sum())

        # ---- 现金对照组 ----
        if n_cw:
            for i, w in enumerate(cw):
                blend = w * (gross - 1.0) + (1.0 - w) * rf_d
                cpath = eq_c[i] * np.cumprod(1.0 + blend, axis=0)
                run_pk_c = np.maximum(peak_c[i][None, :], np.maximum.accumulate(cpath, axis=0))
                mdd_c[i] = np.maximum(mdd_c[i], (1.0 - cpath / run_pk_c).max(axis=0))
                peak_c[i] = run_pk_c[-1]
                eq_c[i] = cpath[-1]

        # 月度收益：分母都用「扣保费之前」的月初净值，两者口径一致
        eq_h_prev = eq_h_open / (1.0 - prem) if prem < 1.0 else eq_h_open
        r_n = eq_n_new / np.maximum(eq_n, 1e-12) - 1.0
        r_h = eq_h_new / np.maximum(eq_h_prev, 1e-12) - 1.0
        rel_list.append(r_h - r_n)

        eq_n = eq_n_new
        eq_h = eq_h_new
        if keep_monthly:
            mon_n[m + 1] = eq_n
            mon_h[m + 1] = eq_h

    yrs = n_months * tenor / TD
    cagr_n = eq_n ** (1.0 / yrs) - 1.0
    cagr_h = np.where(eq_h > 0, eq_h ** (1.0 / yrs) - 1.0, -1.0)
    rel = np.concatenate(rel_list)

    out = dict(
        cagr_n_med=float(np.median(cagr_n)), cagr_h_med=float(np.median(cagr_h)),
        cagr_n_mean=float(cagr_n.mean()), cagr_h_mean=float(cagr_h.mean()),
        cagr_n_p05=float(np.percentile(cagr_n, 5)), cagr_h_p05=float(np.percentile(cagr_h, 5)),
        cagr_n_p95=float(np.percentile(cagr_n, 95)), cagr_h_p95=float(np.percentile(cagr_h, 95)),
        drag_med=float(np.median(cagr_h) - np.median(cagr_n)),
        drag_mean=float(cagr_h.mean() - cagr_n.mean()),
        prem_per_year=float(tot_prem / yrs),
        payout_per_year=float(tot_pay / yrs),
        payout_ratio=float(tot_pay / max(tot_prem, 1e-12)),
        n_payout_per_year=float(n_pay / n_paths / yrs),
        put_price=float(put_px), fair_price=float(fair), load=float(load),
        beat_rate=float((cagr_h > cagr_n).mean()),
        mdd_n_med=float(np.median(mdd_n)), mdd_h_med=float(np.median(mdd_h)),
        mdd_n_p95=float(np.percentile(mdd_n, 95)), mdd_h_p95=float(np.percentile(mdd_h, 95)),
        mdd_n_p99=float(np.percentile(mdd_n, 99)), mdd_h_p99=float(np.percentile(mdd_h, 99)),
        worst_day=float(worst_day),
        eq_n=eq_n, eq_h=eq_h, cagr_n=cagr_n, cagr_h=cagr_h,
        mdd_n=mdd_n, mdd_h=mdd_h, rel=rel,
    )
    if n_cw:
        out["cash"] = [dict(w=float(cw[i]),
                            cagr_med=float(np.median(eq_c[i] ** (1.0 / yrs) - 1.0)),
                            mdd_med=float(np.median(mdd_c[i])),
                            mdd_p95=float(np.percentile(mdd_c[i], 95)),
                            mdd_p99=float(np.percentile(mdd_c[i], 99)))
                       for i in range(n_cw)]
    if keep_monthly:
        out["mon_n"] = mon_n; out["mon_h"] = mon_h
    if day_samples:
        ds = np.concatenate(day_samples)
        out["daily_skew"] = float(((ds - ds.mean()) ** 3).mean() / ds.std() ** 3)
        out["daily_kurt"] = float(((ds - ds.mean()) ** 4).mean() / ds.std() ** 4)
    return out


R = {}
print("=" * 70)
CASH_W = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
base = simulate(cash_weights=CASH_W, keep_monthly=True)
R["base"] = {k: v for k, v in base.items()
             if not isinstance(v, (np.ndarray, list))}
print(json.dumps(R["base"], ensure_ascii=False, indent=2))
R["base"]["mdd_improve_med"] = base["mdd_n_med"] - base["mdd_h_med"]
R["base"]["mdd_improve_p99"] = base["mdd_n_p99"] - base["mdd_h_p99"]

# ---- 保费解析核对 ----
n_per_yr = TD / TENOR
R["premium_check"] = dict(
    put_price=base["put_price"], fair_price=base["fair_price"],
    load=base["load"],
    payout_ratio_theory=float(1.0 / (1.0 + LOAD)),
    payout_ratio_empirical=base["payout_ratio"],
    payout_ratio_err=float(abs(base["payout_ratio"] - 1.0 / (1.0 + LOAD))),
    per_year_analytic=float(base["put_price"] * n_per_yr),
    per_year_empirical=base["prem_per_year"],
    abs_err=float(abs(base["put_price"] * n_per_yr - base["prem_per_year"])),
    rolls_per_year=float(n_per_yr),
)
print("保费解析核对：", R["premium_check"])

# ============================================================
# 安慰剂
# ============================================================
pa = simulate(n_paths=N_SCAN, load=0.0)
R["placebo_fairiv"] = dict(drag_med=pa["drag_med"], prem_per_year=pa["prem_per_year"],
                           payout_ratio=pa["payout_ratio"], beat_rate=pa["beat_rate"],
                           mdd_h_p99=pa["mdd_h_p99"], put_price=pa["put_price"],
                           payout_ratio_err=float(abs(pa["payout_ratio"]-1.0)))
print("安慰剂A 公平定价(load=0)：", R["placebo_fairiv"])

pb = simulate(n_paths=N_SCAN, jlam=0.0)
R["placebo_nojump"] = dict(drag_med=pb["drag_med"], payout_ratio=pb["payout_ratio"],
                           prem_per_year=pb["prem_per_year"], beat_rate=pb["beat_rate"],
                           mdd_n_p99=pb["mdd_n_p99"], mdd_h_p99=pb["mdd_h_p99"])
print("安慰剂B 无跳跃：", R["placebo_nojump"])

pc = simulate(n_paths=N_SCAN, hedge_ratio=0.0)
R["placebo_zerohedge"] = dict(
    drag_med=pc["drag_med"], drag_mean=pc["drag_mean"],
    max_abs_dev=float(np.abs(pc["eq_h"] - pc["eq_n"]).max()),
    mdd_dev=float(np.abs(pc["mdd_h"] - pc["mdd_n"]).max()),
    prem_per_year=pc["prem_per_year"],
)
print("安慰剂C 零对冲：", R["placebo_zerohedge"])

# ============================================================
# 现金对照：插值找到匹配回撤的权重
# ============================================================
cash = base["cash"]
ws = np.array([d["w"] for d in cash])
p99s = np.array([d["mdd_p99"] for d in cash])
meds = np.array([d["mdd_med"] for d in cash])
cgs = np.array([d["cagr_med"] for d in cash])

w_p99 = float(np.interp(base["mdd_h_p99"], p99s, ws))
cg_p99 = float(np.interp(w_p99, ws, cgs))
w_med = float(np.interp(base["mdd_h_med"], meds, ws))
cg_med = float(np.interp(w_med, ws, cgs))

R["cash_match"] = dict(
    match_p99=dict(weight=w_p99, target=base["mdd_h_p99"], cash_cagr=cg_p99,
                   hedge_cagr=base["cagr_h_med"],
                   hedge_edge_pp=float((base["cagr_h_med"] - cg_p99) * 100)),
    match_med=dict(weight=w_med, target=base["mdd_h_med"], cash_cagr=cg_med,
                   hedge_cagr=base["cagr_h_med"],
                   hedge_edge_pp=float((base["cagr_h_med"] - cg_med) * 100)),
    grid=cash,
)
print("现金对照：", json.dumps(R["cash_match"]["match_p99"], ensure_ascii=False))
print("           ", json.dumps(R["cash_match"]["match_med"], ensure_ascii=False))

# ============================================================
# 剂量反应
# ============================================================
iv_scan = []
for v in [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50]:
    r = simulate(n_paths=N_SCAN, load=v)
    iv_scan.append(dict(load=v, drag_med=r["drag_med"], prem=r["prem_per_year"],
                        payout_ratio=r["payout_ratio"], beat=r["beat_rate"]))
    print(f"  加成={v:.2f}  保费/年={r['prem_per_year']*100:5.2f}%  拖累={r['drag_med']*100:+6.3f}pp  赔付率={r['payout_ratio']*100:6.1f}%")
R["iv_scan"] = iv_scan
ivv = np.array([d["load"] for d in iv_scan]); ivd = np.array([d["drag_med"] for d in iv_scan])
be_iv = float(np.interp(0.0, ivd[::-1], ivv[::-1])) if ivd.min() < 0 < ivd.max() else float("nan")
R["iv_breakeven"] = be_iv
print(f"  拖累过零的保险加成 = {be_iv:.4f}（基准 {LOAD}）")

jl_scan = []
for j in [0.0, 0.1, 0.2, 0.25, 0.4, 0.6, 0.9, 1.3]:
    r = simulate(n_paths=N_SCAN, jlam=j)
    jl_scan.append(dict(jlam=j, drag_med=r["drag_med"], payout_ratio=r["payout_ratio"],
                        beat=r["beat"] if "beat" in r else r["beat_rate"],
                        mdd_n_p99=r["mdd_n_p99"], mdd_h_p99=r["mdd_h_p99"]))
    print(f"  λ={j:.2f}/年  拖累={r['drag_med']*100:+6.3f}pp  赔付率={r['payout_ratio']*100:5.1f}%  跑赢率={r['beat_rate']*100:5.1f}%")
R["jl_scan"] = jl_scan
jj = np.array([d["jlam"] for d in jl_scan]); jd = np.array([d["drag_med"] for d in jl_scan])
be_jl = float(np.interp(0.0, jd, jj)) if jd.min() < 0 < jd.max() else float("nan")
R["jl_breakeven"] = be_jl
print(f"  拖累过零跳跃强度 = {be_jl:.4f} 次/年（基准 {JLAM}）")

mn_scan = []
for mm in [0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80, 0.70]:
    r = simulate(n_paths=N_SCAN, moneyness=mm)
    imp = base["mdd_n_p99"] - r["mdd_h_p99"]
    mn_scan.append(dict(moneyness=mm, drag_med=r["drag_med"], prem=r["prem_per_year"],
                        payout_ratio=r["payout_ratio"], mdd_h_p99=r["mdd_h_p99"],
                        mdd_h_med=r["mdd_h_med"], n_payout=r["n_payout_per_year"],
                        cost_per_pp=float(abs(r["drag_med"]) * 100 / imp / 100) if imp > 0 else float("nan")))
    print(f"  K={mm*100:.0f}%  保费/年={r['prem_per_year']*100:6.2f}%  拖累={r['drag_med']*100:+6.3f}pp  "
          f"99%回撤={r['mdd_h_p99']*100:5.1f}%  年赔付={r['n_payout_per_year']:.2f}次")
R["mn_scan"] = mn_scan
# 只在「确实带来 >=2pp 尾部改善」的档位里比效率，否则分母趋零会选出退化点
valid = [d for d in mn_scan
         if np.isfinite(d["cost_per_pp"]) and (base["mdd_n_p99"] - d["mdd_h_p99"]) >= 0.02]
R["mn_best_efficiency"] = min(valid, key=lambda d: d["cost_per_pp"]) if valid else None
R["mn_best_protection"] = min(mn_scan, key=lambda d: d["mdd_h_p99"])
print("  单位回撤改善最便宜：", R["mn_best_efficiency"])
print("  尾部保护最强：", R["mn_best_protection"])
# 非单调性：最贵的档位反而保护更差（保费本身制造回撤）
R["mn_nonmonotone"] = dict(
    k99=[d for d in mn_scan if d["moneyness"] == 0.99][0],
    k97=[d for d in mn_scan if d["moneyness"] == 0.97][0],
)
_k99 = R["mn_nonmonotone"]["k99"]; _k97 = R["mn_nonmonotone"]["k97"]
print(f"  非单调：K=99% 年保费 {_k99['prem']*100:.2f}% 但 99分位回撤 {_k99['mdd_h_p99']*100:.1f}%"
      f" > K=97% 的 {_k97['mdd_h_p99']*100:.1f}%")

# ============================================================
# 拖累分解：公平定价收益 vs 加成成本
# ============================================================
R["decomposition"] = dict(
    fair_price_benefit_pp=float(pa["drag_med"] * 100),
    total_drag_pp=float(base["drag_med"] * 100),
    load_cost_pp=float((base["drag_med"] - pa["drag_med"]) * 100),
    load_prem_gap_pp=float((base["prem_per_year"] - pa["prem_per_year"]) * 100),
    breakeven_load=be_iv,
    actual_load=LOAD,
    margin_pct=float((be_iv / LOAD - 1) * 100),
)
print("分解：", R["decomposition"])

hr_scan = []
for h in [0.0, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5]:
    r = simulate(n_paths=N_SCAN, hedge_ratio=h)
    hr_scan.append(dict(hr=h, drag_med=r["drag_med"], cagr_h_med=r["cagr_h_med"],
                        mdd_h_med=r["mdd_h_med"], mdd_h_p99=r["mdd_h_p99"],
                        prem=r["prem_per_year"]))
    print(f"  对冲比例={h:.2f}  拖累={r['drag_med']*100:+6.3f}pp  中位回撤={r['mdd_h_med']*100:5.1f}%  99%回撤={r['mdd_h_p99']*100:5.1f}%")
R["hr_scan"] = hr_scan

# ============================================================
# 赔付集中度
# ============================================================
rel = base["rel"]
srt = np.sort(rel)
n1 = max(1, int(0.01 * srt.size)); n5 = max(1, int(0.05 * srt.size))
R["payoff_concentration"] = dict(
    total_mean=float(rel.mean()),
    best1pct_mean=float(srt[-n1:].mean()),
    best5pct_mean=float(srt[-n5:].mean()),
    pct_months_negative=float((rel < 0).mean()),
    pct_months_positive=float((rel > 0).mean()),
    median_month=float(np.median(rel)),
    best_month=float(srt[-1]),
    mean_ex_best1pct=float(srt[:-n1].mean()),
    ann_ex_best1pct=float(srt[:-n1].mean() * 12 * 100),
)
print("赔付集中度：", R["payoff_concentration"])

mu_r, sd_r = rel.mean(), rel.std()
mo_need = (2.0 * sd_r / abs(mu_r)) ** 2
R["sample_size"] = dict(monthly_mean=float(mu_r), monthly_sd=float(sd_r),
                        months_needed=float(mo_need), years_needed=float(mo_need / 12.0),
                        signal_noise=float(abs(mu_r) / sd_r))
print("样本量：", R["sample_size"])

# 5 年滚动窗口下的跑赢率
mn_, mh_ = base["mon_n"], base["mon_h"]
w5 = 60
beats = []
for s in range(0, N_MONTHS - w5 + 1, 6):
    gn = mn_[s + w5] / np.maximum(mn_[s], 1e-12)
    gh = mh_[s + w5] / np.maximum(mh_[s], 1e-12)
    beats.append((gh > gn).astype(float))
beats = np.concatenate(beats)
R["window_5y_beat"] = float(beats.mean())
w10 = 120
b10 = []
for s in range(0, N_MONTHS - w10 + 1, 6):
    gn = mn_[s + w10] / np.maximum(mn_[s], 1e-12)
    gh = mh_[s + w10] / np.maximum(mh_[s], 1e-12)
    b10.append((gh > gn).astype(float))
R["window_10y_beat"] = float(np.concatenate(b10).mean())
print(f"5 年窗口跑赢率 {R['window_5y_beat']*100:.2f}%，10 年 {R['window_10y_beat']*100:.2f}%")

uni = []
for i in range(12):
    r = simulate(n_paths=3000, seed=SEED + 1000 * (i + 1))
    uni.append(r["drag_med"])
uni = np.array(uni)
R["universe"] = dict(values=[float(v) for v in uni], mean=float(uni.mean()),
                     sd=float(uni.std()), cv=float(abs(uni.std() / uni.mean())))
print("12 宇宙 CV =", R["universe"]["cv"])

# ============================================================
# 绘图
# ============================================================
xs = np.arange(N_MONTHS + 1) / 12.0

fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
a = ax[0]
a.plot(xs, np.median(mn_, axis=1), color=C_NAKED, lw=2.4, label=f"不对冲（CAGR {base['cagr_n_med']*100:.2f}%）")
a.plot(xs, np.median(mh_, axis=1), color=C_HEDGE, lw=2.4, label=f"每月买保险（CAGR {base['cagr_h_med']*100:.2f}%）")
a.fill_between(xs, np.percentile(mn_, 5, axis=1), np.percentile(mn_, 95, axis=1), color=C_NAKED, alpha=0.10)
a.fill_between(xs, np.percentile(mh_, 5, axis=1), np.percentile(mh_, 95, axis=1), color=C_HEDGE, alpha=0.10)
a.set_yscale("log"); a.set_xlabel("年"); a.set_ylabel("净值（对数轴）")
a.set_title(f"20 年中位拖累 {base['drag_med']*100:+.3f} pp/年，{base['beat_rate']*100:.1f}% 的路径跑赢", fontsize=12.5)
a.legend(fontsize=9, loc="upper left")

a = ax[1]
labels = ["不对冲", "每月买保险", f"减仓到 {w_p99*100:.0f}%\n+现金", f"减仓到 {w_med*100:.0f}%\n+现金"]
cg = [base["cagr_n_med"] * 100, base["cagr_h_med"] * 100, cg_p99 * 100, cg_med * 100]
d99 = [base["mdd_n_p99"] * 100, base["mdd_h_p99"] * 100, base["mdd_h_p99"] * 100,
       float(np.interp(w_med, ws, p99s)) * 100]
xp = np.arange(4)
a.bar(xp - 0.2, cg, 0.4, color=C_OK, label="中位 CAGR %")
a.bar(xp + 0.2, d99, 0.4, color=C_NAKED, label="99 分位最大回撤 %")
for i, (c_, d_) in enumerate(zip(cg, d99)):
    a.text(i - 0.2, c_ + 0.7, f"{c_:.2f}", ha="center", fontsize=9)
    a.text(i + 0.2, d_ + 0.7, f"{d_:.1f}", ha="center", fontsize=9)
a.set_xticks(xp); a.set_xticklabels(labels, fontsize=9); a.set_ylabel("%")
a.set_title(f"同样压到 {base['mdd_h_p99']*100:.1f}% 尾部回撤，对冲比减仓多赚 "
            f"{R['cash_match']['match_p99']['hedge_edge_pp']:+.3f} pp", fontsize=12.5)
a.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/cover.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
a = ax[0]
a.hist(np.clip(rel * 100, -3, 30), bins=180, color=C_HEDGE, alpha=0.85)
a.axvline(0, color="k", lw=1.2)
a.axvline(rel.mean() * 100, color=C_OK, lw=2.2, label=f"均值 {rel.mean()*100:+.4f}%")
a.axvline(np.median(rel) * 100, color=C_NAKED, lw=2.2, ls="--", label=f"中位 {np.median(rel)*100:+.4f}%")
a.set_yscale("log"); a.set_xlabel("对冲 − 不对冲（单月，%）"); a.set_ylabel("月份数（对数）")
a.set_title(f"{R['payoff_concentration']['pct_months_negative']*100:.1f}% 的月份在付钱", fontsize=12.5)
a.legend(fontsize=9)

a = ax[1]
vals = [R["payoff_concentration"]["best1pct_mean"] * 100,
        R["payoff_concentration"]["best5pct_mean"] * 100,
        R["payoff_concentration"]["median_month"] * 100,
        R["payoff_concentration"]["mean_ex_best1pct"] * 100]
lb = ["最好 1%\n月份", "最好 5%\n月份", "中位\n月份", "剔除最好 1%\n后均值"]
b = a.bar(lb, vals, color=[C_OK, C_CASH, C_GREY, C_NAKED], alpha=0.9)
for r_, v in zip(b, vals):
    a.text(r_.get_x() + r_.get_width() / 2, v + (0.08 if v >= 0 else -0.2),
           f"{v:+.4f}%", ha="center", fontsize=10, fontweight="bold")
a.axhline(0, color="k", lw=1.0); a.set_ylabel("单月相对收益 %")
a.set_title(f"剔除最好 1% 月份后，年化 {R['payoff_concentration']['ann_ex_best1pct']:+.2f}%", fontsize=12.5)
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/mechanism.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
a = ax[0]
lbs = [f"基准\n加成 {LOAD:.0%}", "安慰剂A\n公平定价", "安慰剂B\n无跳跃", "安慰剂C\n零对冲"]
vs = [base["drag_med"] * 100, pa["drag_med"] * 100, pb["drag_med"] * 100, pc["drag_med"] * 100]
b = a.bar(lbs, vs, color=[C_HEDGE, C_WARN, C_PUR, C_OK], alpha=0.9)
for r_, v in zip(b, vs):
    a.text(r_.get_x() + r_.get_width() / 2, v + (0.05 if v >= 0 else -0.18),
           f"{v:+.3f}", ha="center", fontsize=10.5, fontweight="bold")
a.axhline(0, color="k", lw=1.0); a.set_ylabel("中位 CAGR 拖累 pp/年")
a.set_title(f"零对冲安慰剂最大逐路径偏差 {R['placebo_zerohedge']['max_abs_dev']:.2e}", fontsize=12.5)

a = ax[1]
lbs2 = ["基准", "公平定价", "无跳跃"]
pr = [base["payout_ratio"] * 100, pa["payout_ratio"] * 100, pb["payout_ratio"] * 100]
pm = [base["prem_per_year"] * 100, pa["prem_per_year"] * 100, pb["prem_per_year"] * 100]
xp = np.arange(3)
a.bar(xp - 0.2, pm, 0.4, color=C_NAKED, label="年保费 % NAV")
a.bar(xp + 0.2, pr, 0.4, color=C_OK, label="累计赔付 / 累计保费 %")
for i, (mv, rv) in enumerate(zip(pm, pr)):
    a.text(i - 0.2, mv + 1.5, f"{mv:.2f}", ha="center", fontsize=9)
    a.text(i + 0.2, rv + 1.5, f"{rv:.1f}", ha="center", fontsize=9)
a.axhline(100, color=C_PUR, lw=1.4, ls="--", label="赔付率 100%（公平保险）")
a.set_xticks(xp); a.set_xticklabels(lbs2); a.set_ylabel("%")
a.set_title(f"基准赔付率 {base['payout_ratio']*100:.1f}%（理论 {100/(1+LOAD):.1f}%），公平定价下 {pa['payout_ratio']*100:.2f}%", fontsize=12.5)
a.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/placebo.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.0))
a = ax[0]
a.plot([d["load"] for d in iv_scan], [d["drag_med"] * 100 for d in iv_scan], "o-", color=C_HEDGE, lw=2.2)
a.axhline(0, color="k", lw=1.0)
a.axvline(LOAD, color=C_GREY, lw=1.4, ls=":", label=f"基准 {LOAD:.0%}")
if np.isfinite(be_iv):
    a.axvline(be_iv, color=C_OK, lw=1.6, ls="--", label=f"过零 {be_iv:.3f}")
a.set_xlabel("保险加成（保费/精算公平值 − 1）"); a.set_ylabel("中位 CAGR 拖累 pp")
a.set_title("保费贵多少", fontsize=12); a.legend(fontsize=8.5)

a = ax[1]
a.plot([d["jlam"] for d in jl_scan], [d["drag_med"] * 100 for d in jl_scan], "o-", color=C_PUR, lw=2.2)
a.axhline(0, color="k", lw=1.0)
a.axvline(JLAM, color=C_GREY, lw=1.4, ls=":", label=f"基准 {JLAM}/年")
if np.isfinite(be_jl):
    a.axvline(be_jl, color=C_OK, lw=1.6, ls="--", label=f"过零 {be_jl:.3f}/年")
a.set_xlabel("危机跳跃强度（次/年）"); a.set_ylabel("中位 CAGR 拖累 pp")
a.set_title("危机要多频繁保险才回本", fontsize=12); a.legend(fontsize=8.5)

a = ax[2]
xv = [d["moneyness"] * 100 for d in mn_scan]
a.plot(xv, [d["prem"] * 100 for d in mn_scan], "o-", color=C_NAKED, lw=2.2)
a.set_xlabel("执行价 / 现价（%）"); a.set_ylabel("年保费 % NAV", color=C_NAKED)
a.tick_params(axis="y", labelcolor=C_NAKED); a.invert_xaxis()
a2 = a.twinx(); a2.grid(False)
a2.plot(xv, [d["mdd_h_p99"] * 100 for d in mn_scan], "^--", color=C_HEDGE, lw=2.0)
a2.axhline(base["mdd_n_p99"] * 100, color=C_GREY, lw=1.4, ls=":")
a2.set_ylabel("99 分位最大回撤 %", color=C_HEDGE)
a2.tick_params(axis="y", labelcolor=C_HEDGE)
a.set_title("越虚值越便宜，保护也越少", fontsize=12)
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/sensitivity.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
a = ax[0]
a.hist(base["cagr_n"] * 100, bins=110, color=C_NAKED, alpha=0.55, label="不对冲")
a.hist(base["cagr_h"] * 100, bins=110, color=C_HEDGE, alpha=0.55, label="每月买保险")
a.axvline(base["cagr_n_med"] * 100, color=C_NAKED, lw=2.2, ls="--")
a.axvline(base["cagr_h_med"] * 100, color=C_HEDGE, lw=2.2, ls="--")
a.set_xlabel("20 年年化收益 %"); a.set_ylabel("路径数")
a.set_title(f"5 分位 {base['cagr_n_p05']*100:.2f}%→{base['cagr_h_p05']*100:.2f}%，"
            f"95 分位 {base['cagr_n_p95']*100:.2f}%→{base['cagr_h_p95']*100:.2f}%", fontsize=11.5)
a.legend(fontsize=9)

a = ax[1]
qs = [50, 75, 90, 95, 99]
vn = [np.percentile(base["mdd_n"], q) * 100 for q in qs]
vh = [np.percentile(base["mdd_h"], q) * 100 for q in qs]
xp = np.arange(len(qs))
a.bar(xp - 0.2, vn, 0.4, color=C_NAKED, label="不对冲")
a.bar(xp + 0.2, vh, 0.4, color=C_HEDGE, label="每月买保险")
for i, (n_, h_) in enumerate(zip(vn, vh)):
    a.text(i - 0.2, n_ + 0.7, f"{n_:.1f}", ha="center", fontsize=9)
    a.text(i + 0.2, h_ + 0.7, f"{h_:.1f}", ha="center", fontsize=9)
    a.text(i, -3.6, f"{n_-h_:+.1f}pp", ha="center", fontsize=8.5, color=C_OK)
a.set_xticks(xp); a.set_xticklabels([f"{q} 分位" for q in qs])
a.set_ylim(-5, max(vn) * 1.16); a.set_ylabel("最大回撤 %")
a.set_title("保护随分位数单调增强——买的确实是尾部", fontsize=12.5)
a.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/distribution.png"); plt.close()

R["params"] = dict(N_MONTHS=N_MONTHS, N_YEARS=float(N_YEARS), N_PATHS=N_PATHS,
                   MU_TRUE=MU_TRUE, SIG_TRUE=SIG_TRUE, RF=RF, JLAM=JLAM,
                   JMU=JMU, JSD=JSD, LOAD=LOAD,
                   HEDGE_MONEY=HEDGE_MONEY, TENOR=TENOR, SEED=SEED)

with open(f"{IMG_DIR}/stats.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=float)
print("\n完成 ->", IMG_DIR)
