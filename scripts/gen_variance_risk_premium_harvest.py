#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方差风险溢价收割：卖波动为什么长期赚钱又偶尔归零
所有图表与数字由真实蒙特卡洛计算生成，固定随机种子可复现。
SEED = 20260807
"""
import json, os, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.facecolor"] = "white"

IMG_DIR = "/Users/halo/workspace/astro-blog/public/images/variance-risk-premium-harvest"
os.makedirs(IMG_DIR, exist_ok=True)

SEED = 20260807

C_SELL = "#dc2626"
C_FAIR = "#1e3a5f"
C_CAP  = "#2563eb"
C_OK   = "#16a34a"
C_WARN = "#e05c2a"
C_PUR  = "#9333ea"
C_GREY = "#9ca3af"

# ============================================================
# 引擎参数（真值，由构造给定）
# ============================================================
N_YEARS  = 30
N_MONTHS = N_YEARS * 12
N_PATHS  = 20000

PHI      = 0.75          # 月度对数方差 AR(1)
SIG_X    = 0.45          # 对数方差无条件标准差
S_INNOV  = SIG_X * np.sqrt(1.0 - PHI ** 2)

LAM      = 0.04          # 每月跳跃概率
JRATE    = 1.4           # 对数跳跃幅度 ~ Exp(JRATE)，E[e^J] = JRATE/(JRATE-1)
JMEAN    = JRATE / (JRATE - 1.0)          # = 3.5
JFACTOR  = (1.0 - LAM) + LAM * JMEAN      # 跳跃对期望方差的乘数

TARGET_RV = 0.18 ** 2    # 无条件期望已实现方差（18% 年化波动）
# E[RV] = exp(mu_x + SIG_X^2/2) * JFACTOR  =>  解 mu_x
MU_X = np.log(TARGET_RV / JFACTOR) - SIG_X ** 2 / 2.0

VRP_BASE = 0.15          # 隐含方差比公允方差高 15%
LEV_BASE = 25.0          # 方差名义 / 资本

CAP_MULT = 6.25          # 上限型方差互换：赔付封顶在 6.25 倍执行方差（= 2.5 倍波动）


def simulate(n_paths=N_PATHS, n_months=N_MONTHS, vrp=VRP_BASE, lev=LEV_BASE,
             lam=LAM, jrate=JRATE, sig_x=SIG_X, phi=PHI,
             mode="constant", cap_mult=CAP_MULT, seed=SEED,
             stop_dd=None, ret_paths=False):
    """
    mode:
      constant   —— 恒定相对杠杆（名义随净值缩放）
      fixed      —— 名义固定在初始资本（不随净值缩放）
      capped     —— 恒定相对杠杆 + 赔付封顶
      voltarget  —— 名义反比于当前公允方差（恒定 vega）
      stoploss   —— 恒定相对杠杆 + 单月巨亏后停做 3 个月
    """
    rng = np.random.default_rng(seed)
    s_innov = sig_x * np.sqrt(1.0 - phi ** 2)
    jmean = jrate / (jrate - 1.0) if jrate > 1.0 else 1.0
    jfactor = (1.0 - lam) + lam * jmean
    mu_x = np.log(TARGET_RV / jfactor) - sig_x ** 2 / 2.0

    x = mu_x + sig_x * rng.standard_normal(n_paths)   # 从无条件分布起步
    equity = np.ones(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    peak = np.ones(n_paths)
    maxdd = np.zeros(n_paths)
    cooldown = np.zeros(n_paths, dtype=int)

    pnl_all = np.zeros((n_months, n_paths)) if ret_paths else None
    eq_track = np.zeros((n_months + 1, n_paths)) if ret_paths else None
    if ret_paths:
        eq_track[0] = 1.0

    sum_k = 0.0
    sum_rv = 0.0
    sum_shortfall = 0.0
    n_shortfall = 0
    rv_samples = []
    pnl_samples = []
    log_growth = np.zeros(n_paths)

    for m in range(n_months):
        # 公允执行方差：以 m-1 期状态为条件的真实期望（解析）
        fair = np.exp(mu_x + phi * (x - mu_x) + s_innov ** 2 / 2.0) * jfactor
        strike = fair * (1.0 + vrp)

        # 状态推进
        x = mu_x + phi * (x - mu_x) + s_innov * rng.standard_normal(n_paths)
        jump = (rng.random(n_paths) < lam) * rng.exponential(1.0 / jrate, n_paths)
        rv = np.exp(x + jump)

        sum_k += strike.mean()
        sum_rv += rv.mean()
        if m < 120:
            rv_samples.append(rv[:2000])

        rv_eff = np.minimum(rv, strike * cap_mult) if mode == "capped" else rv

        if mode == "voltarget":
            notional = lev * (TARGET_RV / fair)
        else:
            notional = np.full(n_paths, lev)

        if mode == "stoploss":
            notional = np.where(cooldown > 0, 0.0, notional)

        if mode == "fixed":
            # 名义固定在初始资本：名义不随净值缩放，等价于相对杠杆 = lev / equity
            eff_notional = notional / np.maximum(equity, 1e-9)
        else:
            eff_notional = notional

        pnl_raw = eff_notional * (strike - rv_eff) / 12.0   # 相对当前资本的月度收益率
        # 亏损不可能超过全部资本：超出部分是违约缺口，单独记账
        shortfall = np.where(pnl_raw < -1.0, -1.0 - pnl_raw, 0.0)
        pnl = np.maximum(pnl_raw, -1.0)

        pnl = np.where(alive, pnl, 0.0)
        shortfall = np.where(alive, shortfall, 0.0)
        sum_shortfall += float(shortfall.sum())
        n_shortfall += int((shortfall > 0).sum())
        if m < 120:
            pnl_samples.append(pnl[:2000])

        if mode == "stoploss":
            cooldown = np.maximum(cooldown - 1, 0)
            cooldown = np.where(pnl < -0.20, 3, cooldown)

        log_growth += np.where(alive & (pnl > -1.0), np.log(np.maximum(1.0 + pnl, 1e-300)), 0.0)
        equity = equity * (1.0 + pnl)
        wiped = equity <= 1e-12
        equity = np.where(wiped, 0.0, equity)
        alive = alive & (~wiped)

        peak = np.maximum(peak, equity)
        dd = 1.0 - equity / np.maximum(peak, 1e-12)
        maxdd = np.maximum(maxdd, dd)

        if ret_paths:
            pnl_all[m] = pnl
            eq_track[m + 1] = equity

    yrs = n_months / 12.0
    yrs_pre = yrs
    cagr = np.where(equity > 0, equity ** (1.0 / yrs) - 1.0, -1.0)

    rv_samples = np.concatenate(rv_samples) if rv_samples else np.array([])
    pnl_samples = np.concatenate(pnl_samples) if pnl_samples else np.array([])

    out = dict(
        equity=equity, cagr=cagr, maxdd=maxdd,
        terminal_mean=float(equity.mean()),
        terminal_med=float(np.median(equity)),
        shortfall_per_path=float(sum_shortfall / n_paths),
        shortfall_events=int(n_shortfall),
        wipeout=float((equity <= 0).mean()),
        below10=float((equity < 0.10).mean()),
        cagr_med=float(np.median(cagr)),
        cagr_mean=float(cagr.mean()),
        cagr_p05=float(np.percentile(cagr, 5)),
        cagr_p95=float(np.percentile(cagr, 95)),
        maxdd_med=float(np.median(maxdd)),
        maxdd_p95=float(np.percentile(maxdd, 95)),
        pos_rate=float((cagr > 0).mean()),
        mean_log_growth=float(log_growth.mean() / yrs_pre),
        mean_strike=float(sum_k / n_months),
        mean_rv=float(sum_rv / n_months),
        rv_samples=rv_samples, pnl_samples=pnl_samples,
    )
    if pnl_samples.size:
        mu_m, sd_m = pnl_samples.mean(), pnl_samples.std()
        out["sharpe"] = float(mu_m / sd_m * np.sqrt(12)) if sd_m > 0 else float("nan")
        out["pnl_mean_m"] = float(mu_m)
        out["pnl_sd_m"] = float(sd_m)
        out["pnl_skew"] = float(((pnl_samples - mu_m) ** 3).mean() / sd_m ** 3)
        out["pnl_kurt"] = float(((pnl_samples - mu_m) ** 4).mean() / sd_m ** 4)
        out["worst_m"] = float(pnl_samples.min())
        out["win_rate"] = float((pnl_samples > 0).mean())
        # 剔除最差 1% 月份后的 Sharpe
        thr = np.percentile(pnl_samples, 1)
        keep = pnl_samples[pnl_samples > thr]
        out["sharpe_ex1"] = float(keep.mean() / keep.std() * np.sqrt(12))
    if ret_paths:
        out["eq_track"] = eq_track
        out["pnl_all"] = pnl_all
    return out


R = {}
print("=" * 70)
print("基准情景")
base = simulate(ret_paths=True)
R["base"] = {k: v for k, v in base.items()
             if k not in ("equity", "cagr", "maxdd", "rv_samples", "pnl_samples", "eq_track", "pnl_all")}
print(json.dumps(R["base"], ensure_ascii=False, indent=2))

# ============================================================
# 恒等式检验：E[K - RV] 应精确等于 VRP * E[RV]
# ============================================================
emp_gap = base["mean_strike"] - base["mean_rv"]
ana_gap = VRP_BASE * TARGET_RV
R["identity"] = dict(
    empirical_gap=float(emp_gap),
    analytic_gap=float(ana_gap),
    abs_err=float(abs(emp_gap - ana_gap)),
    rel_err=float(abs(emp_gap - ana_gap) / ana_gap),
    mean_rv_emp=float(base["mean_rv"]),
    mean_rv_ana=float(TARGET_RV),
    rv_err=float(abs(base["mean_rv"] - TARGET_RV)),
)
print("恒等式 E[K-RV] = VRP*E[RV]：", R["identity"])

# ============================================================
# 安慰剂 A：VRP = 0（保险公平定价）
# ============================================================
pa = simulate(vrp=0.0)
R["placebo_vrp0"] = dict(cagr_med=pa["cagr_med"], cagr_mean=pa["cagr_mean"],
                         wipeout=pa["wipeout"], pos_rate=pa["pos_rate"],
                         pnl_mean_m=pa["pnl_mean_m"], sharpe=pa["sharpe"],
                         gap=float(pa["mean_strike"] - pa["mean_rv"]))
print("安慰剂A VRP=0：", R["placebo_vrp0"])

# ============================================================
# 安慰剂 B：零波动率不确定性（RV 恒等于公允值）
# ============================================================
pb = simulate(sig_x=1e-9, lam=0.0, jrate=1e9)
det_month = LEV_BASE * VRP_BASE * TARGET_RV / 12.0
det_cagr = (1.0 + det_month) ** 12 - 1.0
R["placebo_zerovol"] = dict(cagr_med=pb["cagr_med"], cagr_analytic=float(det_cagr),
                            abs_err=float(abs(pb["cagr_med"] - det_cagr)),
                            pnl_sd_m=pb["pnl_sd_m"], wipeout=pb["wipeout"],
                            maxdd_med=pb["maxdd_med"])
print("安慰剂B 零波动：", R["placebo_zerovol"])

# ============================================================
# 安慰剂 C：无跳跃（只保留扩散型波动聚集）
# ============================================================
pc = simulate(lam=0.0, jrate=1e9)
R["placebo_nojump"] = dict(cagr_med=pc["cagr_med"], wipeout=pc["wipeout"],
                           below10=pc["below10"], maxdd_med=pc["maxdd_med"],
                           maxdd_p95=pc["maxdd_p95"], sharpe=pc["sharpe"],
                           worst_m=pc["worst_m"], pnl_skew=pc["pnl_skew"])
print("安慰剂C 无跳跃：", R["placebo_nojump"])

# ============================================================
# 剂量反应 1：VRP 扫描
# ============================================================
vrp_grid = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40]
vrp_scan = []
for v in vrp_grid:
    r = simulate(vrp=v)
    vrp_scan.append(dict(vrp=v, cagr_med=r["cagr_med"], cagr_mean=r["cagr_mean"],
                         wipeout=r["wipeout"], pos_rate=r["pos_rate"],
                         sharpe=r["sharpe"], maxdd_med=r["maxdd_med"],
                         terminal_mean=r["terminal_mean"]))
    print(f"  VRP={v:.3f}  中位CAGR={r['cagr_med']*100:+7.3f}%  均值CAGR={r['cagr_mean']*100:+8.3f}%  归零={r['wipeout']*100:5.2f}%  Sharpe={r['sharpe']:.3f}")
R["vrp_scan"] = vrp_scan

# 均值 CAGR 过零点（线性插值，均值口径计入归零路径）
vv = np.array([d["vrp"] for d in vrp_scan]); cm = np.array([d["cagr_mean"] for d in vrp_scan])
be_vrp = float(np.interp(0.0, cm, vv)) if cm.min() < 0 < cm.max() else float("nan")
R["vrp_breakeven_mean"] = be_vrp
cc = np.array([d["cagr_med"] for d in vrp_scan])
be_vrp_med = float(np.interp(0.0, cc, vv)) if cc.min() < 0 < cc.max() else float("nan")
R["vrp_breakeven_med"] = be_vrp_med
print(f"  均值 CAGR 盈亏平衡 VRP = {be_vrp:.4f}（基准 {VRP_BASE}）；中位口径 = {be_vrp_med}")

# ============================================================
# 剂量反应 2：杠杆扫描
# ============================================================
lev_grid = [5, 10, 15, 20, 25, 30, 40, 50]
lev_scan = []
for L in lev_grid:
    r = simulate(lev=float(L))
    lev_scan.append(dict(lev=L, cagr_med=r["cagr_med"], cagr_mean=r["cagr_mean"],
                         wipeout=r["wipeout"], below10=r["below10"],
                         maxdd_med=r["maxdd_med"], sharpe=r["sharpe"]))
    print(f"  L={L:3d}  中位CAGR={r['cagr_med']*100:+7.3f}%  均值={r['cagr_mean']*100:+7.3f}%  归零={r['wipeout']*100:5.2f}%  中位回撤={r['maxdd_med']*100:5.1f}%")
R["lev_scan"] = lev_scan
best = max(lev_scan, key=lambda d: d["cagr_med"])
R["lev_best"] = best
print(f"  中位 CAGR 最优杠杆 = {best['lev']}（{best['cagr_med']*100:.3f}%，归零 {best['wipeout']*100:.2f}%）")

# ============================================================
# 剂量反应 3：跳跃强度扫描
# ============================================================
lam_grid = [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12]
lam_scan = []
for lm in lam_grid:
    r = simulate(lam=lm, jrate=JRATE if lm > 0 else 1e9)
    lam_scan.append(dict(lam=lm, lam_yr=lm * 12, cagr_med=r["cagr_med"],
                         cagr_mean=r["cagr_mean"],
                         wipeout=r["wipeout"], sharpe=r["sharpe"],
                         maxdd_med=r["maxdd_med"]))
    print(f"  λ={lm*12:.2f}/年  中位CAGR={r['cagr_med']*100:+7.3f}%  均值={r['cagr_mean']*100:+8.3f}%  归零={r['wipeout']*100:5.2f}%")
R["lam_scan"] = lam_scan
ll = np.array([d["lam_yr"] for d in lam_scan]); lc = np.array([d["cagr_mean"] for d in lam_scan])
# lc 单调递减，反转后递增才能用 np.interp
be_lam = float(np.interp(0.0, lc[::-1], ll[::-1])) if lc.min() < 0 < lc.max() else float("nan")
R["lam_breakeven_mean"] = be_lam
print(f"  均值 CAGR 过零跳跃强度 = {be_lam:.4f} 次/年（基准 {LAM*12:.2f}，安全边际 {(be_lam/(LAM*12)-1)*100:+.1f}%）")

# 归零概率对跳跃强度的斜率
wz = np.array([d["wipeout"] for d in lam_scan])
R["lam_wipeout_slope"] = float(np.polyfit(ll[1:], wz[1:], 1)[0])
print(f"  归零概率 ~ 跳跃强度 斜率 = {R['lam_wipeout_slope']:.4f} /（次/年）")

# ============================================================
# 中位幻觉：公平定价下中位月份依然赚钱
# ============================================================
median_rv_ana = float(np.exp(MU_X))            # 无跳跃时的 RV 中位数
mean_rv_ana = float(TARGET_RV)
fair_strike = mean_rv_ana                       # VRP=0 时的执行方差
mirage_month = LEV_BASE * (fair_strike - median_rv_ana) / 12.0
R["median_mirage"] = dict(
    mean_rv=mean_rv_ana, median_rv=median_rv_ana,
    mean_over_median=float(mean_rv_ana / median_rv_ana),
    typical_month_gain_pct=float(mirage_month * 100),
    vrp0_median_cagr=pa["cagr_med"], vrp0_pos_rate=pa["pos_rate"],
    vrp0_mean_cagr=pa["cagr_mean"], vrp0_wipeout=pa["wipeout"],
    vrp0_win_rate=pa["win_rate"],
)
print("中位幻觉：", R["median_mirage"])

# ============================================================
# 5 年滚动窗口：业绩报告看到的东西
# ============================================================
eqb = base["eq_track"]
win = 60
seg_ret, seg_sharpe = [], []
for s in range(0, N_MONTHS - win + 1, 12):
    e0, e1 = eqb[s], eqb[s + win]
    ok = e0 > 1e-9
    gr = np.where(ok, e1 / np.maximum(e0, 1e-12), 0.0)
    seg_ret.append(np.where(ok, np.where(gr > 0, gr ** (12.0 / win) - 1.0, -1.0), np.nan))
seg_ret = np.concatenate([s[~np.isnan(s)] for s in seg_ret])
R["window_5y"] = dict(
    n=int(seg_ret.size),
    pos_rate=float((seg_ret > 0).mean()),
    med=float(np.median(seg_ret)),
    p05=float(np.percentile(seg_ret, 5)),
    p25=float(np.percentile(seg_ret, 25)),
    p95=float(np.percentile(seg_ret, 95)),
    ruin_rate=float((seg_ret <= -0.999).mean()),
)
print("5 年滚动窗口：", R["window_5y"])

# ============================================================
# 仓位口径对照
# ============================================================
modes = [("constant", "恒定相对杠杆"), ("fixed", "名义固定"),
         ("capped", "赔付封顶"), ("voltarget", "恒定 vega"),
         ("stoploss", "巨亏后停做 3 月")]
mode_res = []
for mk, mn in modes:
    r = simulate(mode=mk)
    mode_res.append(dict(key=mk, name=mn, cagr_med=r["cagr_med"], cagr_mean=r["cagr_mean"],
                         sharpe=r["sharpe"], sharpe_ex1=r["sharpe_ex1"],
                         wipeout=r["wipeout"], below10=r["below10"],
                         maxdd_med=r["maxdd_med"], maxdd_p95=r["maxdd_p95"],
                         pos_rate=r["pos_rate"], worst_m=r["worst_m"],
                         terminal_mean=r["terminal_mean"]))
    print(f"  {mn:16s} 中位CAGR={r['cagr_med']*100:+7.3f}%  Sharpe={r['sharpe']:6.3f}  "
          f"归零={r['wipeout']*100:5.2f}%  95%回撤={r['maxdd_p95']*100:5.1f}%")
R["modes"] = mode_res
sh = [d["sharpe"] for d in mode_res]
wp = [max(d["wipeout"], 1e-9) for d in mode_res]
R["mode_spread"] = dict(sharpe_range=float(max(sh) - min(sh)),
                        wipeout_ratio=float(max(wp) / min(wp)))

# ============================================================
# 时间维度：前 N 年赚钱的路径后来怎样了
# ============================================================
eq = base["eq_track"]
e5, e10, e20, e30 = eq[60], eq[120], eq[240], eq[360]
prof5 = e5 > 1.0
R["persistence"] = dict(
    profit_at_5y=float(prof5.mean()),
    of_those_neg_at_30y=float((e30[prof5] <= 1.0).mean()),
    of_those_wiped_at_30y=float((e30[prof5] <= 0.0).mean()),
    profit_at_10y=float((e10 > 1.0).mean()),
    of_10y_profit_neg_at_30y=float((e30[e10 > 1.0] <= 1.0).mean()),
    med_eq_5y=float(np.median(e5)), med_eq_10y=float(np.median(e10)),
    med_eq_20y=float(np.median(e20)), med_eq_30y=float(np.median(e30)),
)
print("持续性：", R["persistence"])

# 首次跌破 50% 的时间
pnl_all = base["pnl_all"]
eqp = eq[1:]
below = eqp < 0.5
first = np.where(below.any(axis=0), below.argmax(axis=0) + 1, -1)
hit = first[first > 0]
R["first_half_loss"] = dict(prob=float((first > 0).mean()),
                            median_month=float(np.median(hit)) if hit.size else float("nan"))
print("首次腰斩：", R["first_half_loss"])

# 单月损益分布特征
pmm = base["pnl_samples"]
R["monthly"] = dict(mean=float(pmm.mean()), sd=float(pmm.std()),
                    skew=base["pnl_skew"], kurt=base["pnl_kurt"],
                    win_rate=base["win_rate"], worst=float(pmm.min()),
                    p01=float(np.percentile(pmm, 1)), p05=float(np.percentile(pmm, 5)),
                    p50=float(np.percentile(pmm, 50)), p99=float(np.percentile(pmm, 99)),
                    sharpe=base["sharpe"], sharpe_ex1=base["sharpe_ex1"])
# 最差 1% 月份贡献了多少累计损益
srt = np.sort(pmm)
n1 = max(1, int(0.01 * srt.size))
R["monthly"]["worst1pct_share_of_total"] = float(srt[:n1].sum() / srt.sum()) if srt.sum() != 0 else float("nan")
R["monthly"]["mean_ex_worst1pct"] = float(srt[n1:].mean())
print("月度分布：", R["monthly"])

# ============================================================
# 样本量红线
# ============================================================
mu_m, sd_m = pmm.mean(), pmm.std()
months_needed = (2.0 * sd_m / mu_m) ** 2 if mu_m > 0 else float("nan")
R["sample_size"] = dict(months_needed=float(months_needed),
                        years_needed=float(months_needed / 12.0),
                        monthly_sharpe=float(mu_m / sd_m))
# 30 年样本里，估出的 Sharpe 有多散
n_boot, n_obs = 4000, 360
rng_b = np.random.default_rng(SEED + 7)
idx = rng_b.integers(0, pmm.size, size=(n_boot, n_obs))
smp = pmm[idx]
sh_boot = smp.mean(axis=1) / smp.std(axis=1) * np.sqrt(12)
R["sample_size"]["sharpe_30y_p05"] = float(np.percentile(sh_boot, 5))
R["sample_size"]["sharpe_30y_p50"] = float(np.percentile(sh_boot, 50))
R["sample_size"]["sharpe_30y_p95"] = float(np.percentile(sh_boot, 95))
print("样本量：", R["sample_size"])

# ============================================================
# 12 宇宙稳健性
# ============================================================
uni = []
for i in range(12):
    r = simulate(n_paths=6000, seed=SEED + 1000 * (i + 1))
    uni.append(r["cagr_med"])
uni = np.array(uni)
R["universe"] = dict(values=[float(v) for v in uni], mean=float(uni.mean()),
                     sd=float(uni.std()), cv=float(abs(uni.std() / uni.mean())))
print("12 宇宙 CV =", R["universe"]["cv"])

# ============================================================
# 绘图
# ============================================================
def pct(x):
    return x * 100.0

# ---------- cover ----------
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
a = ax[0]
sub = eq[:, :400]
for j in range(400):
    a.plot(np.arange(361) / 12.0, sub[:, j], color=C_GREY, lw=0.4, alpha=0.25)
med = np.median(eq, axis=1)
a.plot(np.arange(361) / 12.0, med, color=C_SELL, lw=2.6, label=f"中位路径（30年 CAGR {pct(base['cagr_med']):.2f}%）")
a.plot(np.arange(361) / 12.0, np.percentile(eq, 95, axis=1), color=C_OK, lw=1.8, ls="--", label="95 分位")
a.plot(np.arange(361) / 12.0, np.percentile(eq, 5, axis=1), color=C_PUR, lw=1.8, ls="--", label="5 分位")
a.axhline(1.0, color="k", lw=0.9, ls=":")
a.set_yscale("log"); a.set_ylim(1e-3, 1e3)
a.set_xlabel("年"); a.set_ylabel("净值（对数轴，初始 = 1）")
a.set_title(f"卖方差 30 年净值：{pct(base['wipeout']):.2f}% 的路径归零", fontsize=12.5)
a.legend(fontsize=9, loc="upper left")

a = ax[1]
names = [d["name"] for d in mode_res]
shv = [d["sharpe"] for d in mode_res]
wpv = [d["wipeout"] * 100 for d in mode_res]
xp = np.arange(len(names))
a.bar(xp - 0.2, shv, 0.4, color=C_FAIR, label="Sharpe（左轴）")
a.set_ylabel("Sharpe", color=C_FAIR)
a.tick_params(axis="y", labelcolor=C_FAIR)
a2 = a.twinx(); a2.grid(False)
a2.bar(xp + 0.2, wpv, 0.4, color=C_SELL, label="归零概率（右轴）")
a2.set_ylabel("30 年归零概率 %", color=C_SELL)
a2.tick_params(axis="y", labelcolor=C_SELL)
for i, (s, w) in enumerate(zip(shv, wpv)):
    a.text(i - 0.2, s + 0.008, f"{s:.3f}", ha="center", fontsize=8.5, color=C_FAIR)
    a2.text(i + 0.2, w + 0.4, f"{w:.2f}%", ha="center", fontsize=8.5, color=C_SELL)
a.set_xticks(xp); a.set_xticklabels(names, fontsize=9, rotation=12)
a.set_title(f"Sharpe 极差仅 {max(shv)-min(shv):.3f}，归零概率差 {max(wpv)/max(min(wpv),1e-9):.0f} 倍", fontsize=12.5)
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/cover.png"); plt.close()

# ---------- mechanism ----------
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
a = ax[0]
rvs = base["rv_samples"]
vol = np.sqrt(rvs) * 100
a.hist(vol, bins=140, range=(0, 120), color=C_FAIR, alpha=0.8, density=True)
a.axvline(np.sqrt(TARGET_RV) * 100, color=C_SELL, lw=2.2,
          label=f"无条件期望方差对应 {np.sqrt(TARGET_RV)*100:.1f}% 波动")
a.axvline(np.median(vol), color=C_OK, lw=2.2, ls="--",
          label=f"中位已实现波动 {np.median(vol):.1f}%")
a.set_xlabel("月度已实现波动（年化 %）"); a.set_ylabel("密度")
a.set_title("已实现波动分布：均值被右尾拖高，中位数远低于均值", fontsize=12.5)
a.legend(fontsize=9)

a = ax[1]
p = base["pnl_samples"] * 100
a.hist(p, bins=200, range=(-120, 20), color=C_SELL, alpha=0.85, density=True)
a.axvline(p.mean(), color=C_OK, lw=2.2, label=f"均值 {p.mean():+.3f}%")
a.axvline(np.median(p), color=C_FAIR, lw=2.2, ls="--", label=f"中位 {np.median(p):+.3f}%")
a.axvline(-100, color="k", lw=1.4, ls=":", label="单月归零线")
a.set_yscale("log")
a.set_xlabel("单月损益（占资本 %）"); a.set_ylabel("密度（对数）")
a.set_title(f"月度损益：胜率 {pct(base['win_rate']):.1f}%，偏度 {base['pnl_skew']:.2f}，峰度 {base['pnl_kurt']:.1f}", fontsize=12.5)
a.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/mechanism.png"); plt.close()

# ---------- placebo ----------
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
a = ax[0]
labels = ["基准\nVRP=15%", "安慰剂A\nVRP=0", "安慰剂B\n零波动不确定", "安慰剂C\n无跳跃"]
vals = [base["cagr_med"] * 100, pa["cagr_med"] * 100, pb["cagr_med"] * 100, pc["cagr_med"] * 100]
cols = [C_SELL, C_WARN, C_OK, C_CAP]
b = a.bar(labels, vals, color=cols, alpha=0.9)
for r_, v in zip(b, vals):
    a.text(r_.get_x() + r_.get_width() / 2, v + (0.4 if v >= 0 else -1.4),
           f"{v:+.2f}%", ha="center", fontsize=10.5, fontweight="bold")
a.axhline(0, color="k", lw=1.0)
a.set_ylabel("30 年中位 CAGR %")
a.set_title(f"安慰剂B 解析值 {det_cagr*100:.4f}%，实测偏差 {abs(pb['cagr_med']-det_cagr):.2e}", fontsize=12.5)

a = ax[1]
labels2 = ["基准", "VRP=0", "零波动", "无跳跃"]
w = [base["wipeout"] * 100, pa["wipeout"] * 100, pb["wipeout"] * 100, pc["wipeout"] * 100]
d = [base["maxdd_med"] * 100, pa["maxdd_med"] * 100, pb["maxdd_med"] * 100, pc["maxdd_med"] * 100]
xp = np.arange(4)
a.bar(xp - 0.2, w, 0.4, color=C_SELL, label="归零概率 %")
a.bar(xp + 0.2, d, 0.4, color=C_FAIR, label="中位最大回撤 %")
for i, (wv, dv) in enumerate(zip(w, d)):
    a.text(i - 0.2, wv + 1.0, f"{wv:.2f}", ha="center", fontsize=9)
    a.text(i + 0.2, dv + 1.0, f"{dv:.1f}", ha="center", fontsize=9)
a.set_xticks(xp); a.set_xticklabels(labels2)
a.set_ylabel("%")
a.set_title(f"无跳跃时归零概率 {pc['wipeout']*100:.2f}%（基准 {base['wipeout']*100:.2f}%）", fontsize=12.5)
a.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/placebo.png"); plt.close()

# ---------- sensitivity ----------
fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.0))
a = ax[0]
xv = [d["vrp"] * 100 for d in vrp_scan]
a.plot(xv, [d["cagr_med"] * 100 for d in vrp_scan], "o-", color=C_SELL, lw=2.2, label="中位 CAGR")
a.plot(xv, [d["cagr_mean"] * 100 for d in vrp_scan], "s--", color=C_OK, lw=1.8, label="均值 CAGR")
a.axhline(0, color="k", lw=1.0)
a.axvline(VRP_BASE * 100, color=C_GREY, lw=1.4, ls=":", label=f"基准 {VRP_BASE*100:.0f}%")
if np.isfinite(be_vrp):
    a.axvline(be_vrp * 100, color=C_PUR, lw=1.6, ls="--", label=f"均值口径盈亏平衡 {be_vrp*100:.2f}%")
a.set_xlabel("方差风险溢价 VRP（%）"); a.set_ylabel("30 年 CAGR %")
a.set_title("VRP 剂量反应", fontsize=12)
a.legend(fontsize=8.5)

a = ax[1]
xv = [d["lev"] for d in lev_scan]
a.plot(xv, [d["cagr_med"] * 100 for d in lev_scan], "o-", color=C_FAIR, lw=2.2, label="中位 CAGR（左）")
a.axhline(0, color="k", lw=1.0)
a.set_xlabel("方差名义 / 资本"); a.set_ylabel("中位 CAGR %", color=C_FAIR)
a.tick_params(axis="y", labelcolor=C_FAIR)
a2 = a.twinx(); a2.grid(False)
a2.plot(xv, [d["wipeout"] * 100 for d in lev_scan], "^--", color=C_SELL, lw=2.0, label="归零概率（右）")
a2.set_ylabel("30 年归零概率 %", color=C_SELL)
a2.tick_params(axis="y", labelcolor=C_SELL)
a.set_title(f"杠杆扫描：中位最优 L={best['lev']}", fontsize=12)

a = ax[2]
xv = [d["lam_yr"] for d in lam_scan]
a.plot(xv, [d["cagr_med"] * 100 for d in lam_scan], "o-", color=C_PUR, lw=2.2, label="中位 CAGR")
a.plot(xv, [d["cagr_mean"] * 100 for d in lam_scan], "s--", color=C_SELL, lw=1.8, label="均值 CAGR")
a.axhline(0, color="k", lw=1.0)
a.axvline(LAM * 12, color=C_GREY, lw=1.4, ls=":", label=f"基准 {LAM*12:.2f} 次/年")
if np.isfinite(be_lam):
    a.axvline(be_lam, color=C_OK, lw=1.6, ls="--", label=f"均值过零 {be_lam:.3f} 次/年")
a.set_xlabel("波动率跳跃强度（次/年）"); a.set_ylabel("中位 CAGR %")
a.set_title("跳跃强度：唯一无法估计的参数", fontsize=12)
a.legend(fontsize=8.5)
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/sensitivity.png"); plt.close()

# ---------- distribution ----------
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
a = ax[0]
c = np.clip(base["cagr"] * 100, -100, 60)
a.hist(c, bins=140, color=C_FAIR, alpha=0.85)
a.axvline(base["cagr_med"] * 100, color=C_SELL, lw=2.2, label=f"中位 {base['cagr_med']*100:+.2f}%")
a.axvline(base["cagr_mean"] * 100, color=C_OK, lw=2.2, ls="--", label=f"均值 {base['cagr_mean']*100:+.2f}%")
a.axvline(0, color="k", lw=1.2, ls=":")
a.set_yscale("log")
a.set_xlabel("30 年年化收益 %"); a.set_ylabel("路径数（对数）")
a.set_title(f"{pct(base['pos_rate']):.1f}% 的路径 30 年后赚钱", fontsize=12.5)
a.legend(fontsize=9)

a = ax[1]
buckets = ["5 年", "10 年", "20 年", "30 年"]
snaps = [e5, e10, e20, e30]
posr = [float((s > 1).mean()) * 100 for s in snaps]
wipr = [float((s <= 0).mean()) * 100 for s in snaps]
medv = [float(np.median(s)) for s in snaps]
xp = np.arange(4)
a.bar(xp - 0.2, posr, 0.4, color=C_OK, label="赚钱路径占比 %")
a.bar(xp + 0.2, wipr, 0.4, color=C_SELL, label="已归零占比 %")
for i, (pv, wv, mv) in enumerate(zip(posr, wipr, medv)):
    a.text(i - 0.2, pv + 1.0, f"{pv:.1f}", ha="center", fontsize=9)
    a.text(i + 0.2, wv + 1.0, f"{wv:.2f}", ha="center", fontsize=9)
    a.text(i, -6.0, f"中位净值 {mv:.2f}", ha="center", fontsize=8.5, color=C_FAIR)
a.set_xticks(xp); a.set_xticklabels(buckets)
a.set_ylim(-10, 105); a.set_ylabel("%")
a.set_title(f"前 5 年赚钱的路径中，{pct(R['persistence']['of_those_neg_at_30y']):.1f}% 到 30 年翻负", fontsize=12.5)
a.legend(fontsize=9, loc="center right")
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/distribution.png"); plt.close()

R["params"] = dict(N_YEARS=N_YEARS, N_PATHS=N_PATHS, PHI=PHI, SIG_X=SIG_X,
                   LAM=LAM, JRATE=JRATE, JMEAN=float(JMEAN),
                   TARGET_RV=float(TARGET_RV), TARGET_VOL=float(np.sqrt(TARGET_RV)),
                   MU_X=float(MU_X), VRP_BASE=VRP_BASE, LEV_BASE=LEV_BASE,
                   CAP_MULT=CAP_MULT, SEED=SEED)

with open(f"{IMG_DIR}/stats.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=float)
print("\n完成 ->", IMG_DIR)
