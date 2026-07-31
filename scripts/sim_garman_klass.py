#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Garman-Klass 波动率估计量：受控蒙特卡洛验证
固定随机种子 20260801。生成 4 张 PNG 并打印全部关键数字。

估计量对比：
  CC   : close-to-close      r^2
  PARK : Parkinson (1980)    (1/(4ln2)) (ln H/L)^2
  GK   : Garman-Klass (1980) 0.5(ln H/L)^2 - (2ln2-1)(ln C/O)^2
  RS   : Rogers-Satchell(1991) ln(H/C)ln(H/O) + ln(L/C)ln(L/O)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 中文字体 ----------
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
]
for fp in FONT_CANDIDATES:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "STHeiti", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

OUT = "/Users/halo/workspace/astro-blog/public/images/garman-klass-volatility"
SEED = 20260801
rng = np.random.default_rng(SEED)

LN2 = np.log(2.0)
PARK_C = 1.0 / (4.0 * LN2)
GK_C2 = 2.0 * LN2 - 1.0   # ≈ 0.386294


# ---------- 日内 OHLC 生成器 ----------
def gen_ohlc(n_days, steps_per_day, sigma_day, mu_day=0.0,
             overnight_sigma=0.0, S0=100.0, rng=None):
    """
    生成 n_days 天的 OHLC。
    sigma_day        : 每日日内扩散波动率（标准差，log 尺度）
    mu_day           : 每日日内漂移（log 尺度）
    overnight_sigma  : 隔夜跳空波动率（log 尺度），加在开盘价上
    steps_per_day    : 日内步数
    返回 dict of arrays: open, high, low, close, true_var(日内真实方差=sigma_day^2)
    """
    dt = 1.0 / steps_per_day
    sig_step = sigma_day * np.sqrt(dt)
    mu_step = mu_day * dt
    O = np.empty(n_days); H = np.empty(n_days)
    L = np.empty(n_days); C = np.empty(n_days)
    logS = np.log(S0)
    prev_close_log = logS
    for d in range(n_days):
        # 隔夜跳空
        if overnight_sigma > 0:
            gap = rng.normal(0.0, overnight_sigma)
        else:
            gap = 0.0
        open_log = prev_close_log + gap
        # 日内路径
        incr = rng.normal(mu_step, sig_step, steps_per_day)
        path = open_log + np.cumsum(incr)
        path = np.concatenate([[open_log], path])
        O[d] = open_log
        H[d] = path.max()
        L[d] = path.min()
        C[d] = path[-1]
        prev_close_log = C[d]
    return {
        "open": np.exp(O), "high": np.exp(H),
        "low": np.exp(L), "close": np.exp(C),
        "open_log": O, "close_log": C,
    }


# ---------- 估计量（单日方差） ----------
def cc_var(close_log, prev_close_log):
    return (close_log - prev_close_log) ** 2

def parkinson_var(high, low):
    return PARK_C * np.log(high / low) ** 2

def gk_var(o, h, l, c):
    hl = np.log(h / l)
    co = np.log(c / o)
    return 0.5 * hl ** 2 - GK_C2 * co ** 2

def rs_var(o, h, l, c):
    hc = np.log(h / c); ho = np.log(h / o)
    lc = np.log(l / c); lo = np.log(l / o)
    return hc * ho + lc * lo


# ==================================================================
# 实验 1：效率对比（多路径分布）
# ==================================================================
def exp_efficiency(n_paths=4000, n_days=200, steps=234, sigma_day=0.02):
    true_var = sigma_day ** 2
    cc_all, pk_all, gk_all, rs_all = [], [], [], []
    for _ in range(n_paths):
        d = gen_ohlc(n_days, steps, sigma_day, mu_day=0.0,
                     overnight_sigma=0.0, rng=rng)
        # CC 用相邻收盘（第 0 天用 open 作为 prev）
        prev_c = np.concatenate([[d["open_log"][0]], d["close_log"][:-1]])
        cc = cc_var(d["close_log"], prev_c)
        pk = parkinson_var(d["high"], d["low"])
        gk = gk_var(d["open"], d["high"], d["low"], d["close"])
        rs = rs_var(d["open"], d["high"], d["low"], d["close"])
        cc_all.append(cc.mean()); pk_all.append(pk.mean())
        gk_all.append(gk.mean()); rs_all.append(rs.mean())
    cc_all = np.array(cc_all); pk_all = np.array(pk_all)
    gk_all = np.array(gk_all); rs_all = np.array(rs_all)

    stats = {}
    for name, arr in [("CC", cc_all), ("PARK", pk_all), ("GK", gk_all), ("RS", rs_all)]:
        stats[name] = dict(mean=arr.mean(), std=arr.std(), var=arr.var(),
                           bias=(arr.mean() - true_var) / true_var)
    eff_park = stats["CC"]["var"] / stats["PARK"]["var"]
    eff_gk = stats["CC"]["var"] / stats["GK"]["var"]
    eff_rs = stats["CC"]["var"] / stats["RS"]["var"]

    print("\n=== 实验1 效率对比 (n_paths=%d, n_days=%d, steps=%d) ===" % (n_paths, n_days, steps))
    print("真实日方差 = %.6g" % true_var)
    for name in ["CC", "PARK", "GK", "RS"]:
        s = stats[name]
        print("  %-5s mean=%.6g std=%.4g var=%.4g bias=%+.2f%%" %
              (name, s["mean"], s["std"], s["var"], 100 * s["bias"]))
    print("  效率比 CC/PARK = %.2fx" % eff_park)
    print("  效率比 CC/GK   = %.2fx" % eff_gk)
    print("  效率比 CC/RS   = %.2fx" % eff_rs)
    print("  效率比 GK/PARK = %.2fx (GK 比 Parkinson 再高多少)" % (stats["PARK"]["var"] / stats["GK"]["var"]))

    # ---- 画图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    bins = np.linspace(min(cc_all.min(), gk_all.min()),
                       max(cc_all.max(), gk_all.max()), 60)
    ax.hist(cc_all, bins=bins, alpha=0.45, label="Close-to-Close", color="#d62728")
    ax.hist(pk_all, bins=bins, alpha=0.45, label="Parkinson", color="#1f77b4")
    ax.hist(gk_all, bins=bins, alpha=0.55, label="Garman-Klass", color="#2ca02c")
    ax.axvline(true_var, color="k", ls="--", lw=1.5, label="真实 σ²")
    ax.set_title("方差估计量分布（%d 条路径 × %d 天）" % (n_paths, n_days))
    ax.set_xlabel("日均方差估计"); ax.set_ylabel("频数")
    ax.legend()

    ax = axes[1]
    data = [cc_all, pk_all, gk_all, rs_all]
    labels = ["CC", "Parkinson", "Garman-Klass", "Rogers-Satchell"]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.axhline(true_var, color="k", ls="--", lw=1.5, label="真实 σ²")
    ax.set_title("四种估计量集中度对比（箱线图）")
    ax.set_ylabel("日均方差估计")
    ax.legend()
    plt.tight_layout()
    plt.savefig("%s/efficiency-distribution.png" % OUT)
    plt.close()
    return stats, dict(park=eff_park, gk=eff_gk, rs=eff_rs)


# ==================================================================
# 实验 2：效率比随样本量
# ==================================================================
def exp_efficiency_vs_n(n_paths=3000, day_grid=(5, 10, 20, 40, 80, 160, 320),
                        steps=234, sigma_day=0.02):
    true_var = sigma_day ** 2
    eff_gk, eff_pk, rmse_cc, rmse_gk = [], [], [], []
    print("\n=== 实验2 效率比随样本量 ===")
    for nd in day_grid:
        cc_all, pk_all, gk_all = [], [], []
        for _ in range(n_paths):
            d = gen_ohlc(nd, steps, sigma_day, rng=rng)
            prev_c = np.concatenate([[d["open_log"][0]], d["close_log"][:-1]])
            cc_all.append(cc_var(d["close_log"], prev_c).mean())
            pk_all.append(parkinson_var(d["high"], d["low"]).mean())
            gk_all.append(gk_var(d["open"], d["high"], d["low"], d["close"]).mean())
        cc_all = np.array(cc_all); pk_all = np.array(pk_all); gk_all = np.array(gk_all)
        e_gk = cc_all.var() / gk_all.var()
        e_pk = cc_all.var() / pk_all.var()
        eff_gk.append(e_gk); eff_pk.append(e_pk)
        rmse_cc.append(np.sqrt(np.mean((cc_all - true_var) ** 2)))
        rmse_gk.append(np.sqrt(np.mean((gk_all - true_var) ** 2)))
        print("  n_days=%4d  eff CC/GK=%.2fx  eff CC/PARK=%.2fx  rmse_cc=%.3g rmse_gk=%.3g"
              % (nd, e_gk, e_pk, rmse_cc[-1], rmse_gk[-1]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(day_grid, rmse_cc, "o-", color="#d62728", label="Close-to-Close RMSE")
    ax.plot(day_grid, rmse_gk, "s-", color="#2ca02c", label="Garman-Klass RMSE")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("样本天数"); ax.set_ylabel("RMSE (对真实 σ²)")
    ax.set_title("估计误差随样本量下降"); ax.legend(); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(day_grid, eff_gk, "s-", color="#2ca02c", label="CC/GK 效率比")
    ax.plot(day_grid, eff_pk, "^-", color="#1f77b4", label="CC/Parkinson 效率比")
    ax.axhline(7.4, color="gray", ls=":", label="GK 理论上界 ≈7.4")
    ax.set_xscale("log")
    ax.set_xlabel("样本天数"); ax.set_ylabel("方差效率比 (CC / X)")
    ax.set_title("效率比对样本量稳定"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("%s/efficiency-vs-sample-size.png" % OUT)
    plt.close()
    return day_grid, eff_gk


# ==================================================================
# 实验 3：漂移敏感性（GK vs RS 分水岭）
# ==================================================================
def exp_drift(n_paths=4000, n_days=250, steps=234, sigma_day=0.02,
              drift_grid=(0.0, 0.005, 0.01, 0.02, 0.04)):
    true_var = sigma_day ** 2
    print("\n=== 实验3 漂移敏感性 (真实日方差=%.6g) ===" % true_var)
    print("  日漂移 |  GK偏差  |  RS偏差  |  PARK偏差 | CC偏差")
    gk_bias, rs_bias, pk_bias, cc_bias = [], [], [], []
    for mu in drift_grid:
        gk_v, rs_v, pk_v, cc_v = [], [], [], []
        for _ in range(n_paths):
            d = gen_ohlc(n_days, steps, sigma_day, mu_day=mu, rng=rng)
            prev_c = np.concatenate([[d["open_log"][0]], d["close_log"][:-1]])
            cc_v.append(cc_var(d["close_log"], prev_c).mean())
            pk_v.append(parkinson_var(d["high"], d["low"]).mean())
            gk_v.append(gk_var(d["open"], d["high"], d["low"], d["close"]).mean())
            rs_v.append(rs_var(d["open"], d["high"], d["low"], d["close"]).mean())
        gb = (np.mean(gk_v) - true_var) / true_var
        rb = (np.mean(rs_v) - true_var) / true_var
        pb = (np.mean(pk_v) - true_var) / true_var
        cb = (np.mean(cc_v) - true_var) / true_var
        gk_bias.append(gb); rs_bias.append(rb); pk_bias.append(pb); cc_bias.append(cb)
        print("  %6.3f |  %+.2f%% | %+.2f%% | %+.2f%% | %+.2f%%"
              % (mu, 100*gb, 100*rb, 100*pb, 100*cb))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.array(drift_grid)
    ax.plot(x, 100*np.array(gk_bias), "o-", color="#2ca02c", lw=2, label="Garman-Klass")
    ax.plot(x, 100*np.array(rs_bias), "s-", color="#9467bd", lw=2, label="Rogers-Satchell")
    ax.plot(x, 100*np.array(pk_bias), "^-", color="#1f77b4", lw=2, label="Parkinson")
    ax.plot(x, 100*np.array(cc_bias), "d-", color="#d62728", lw=2, label="Close-to-Close")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("每日漂移 μ (log 尺度)")
    ax.set_ylabel("估计偏差 (%)")
    ax.set_title("漂移污染：GK 上偏而 Rogers-Satchell 免疫")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("%s/drift-sensitivity.png" % OUT)
    plt.close()
    return drift_grid, gk_bias, rs_bias


# ==================================================================
# 实验 4：隔夜跳空盲区
# ==================================================================
def exp_overnight(n_paths=4000, n_days=250, steps=234, sigma_day=0.02,
                  ratio_grid=(0.0, 0.3, 0.5, 1.0, 1.5)):
    """
    隔夜/日内波动比。总真实方差 = 日内方差 + 隔夜方差。
    GK/Parkinson 只测日内，CC 吸收隔夜。以「总方差」为基准衡量偏差。
    """
    intraday_var = sigma_day ** 2
    print("\n=== 实验4 隔夜跳空盲区 ===")
    print("  overnight/intraday | GK偏差(对总方差) | CC偏差 | PARK偏差")
    gk_b, cc_b, pk_b = [], [], []
    for ratio in ratio_grid:
        ov_sigma = sigma_day * ratio
        total_var = intraday_var + ov_sigma ** 2
        gk_v, cc_v, pk_v = [], [], []
        for _ in range(n_paths):
            d = gen_ohlc(n_days, steps, sigma_day, mu_day=0.0,
                         overnight_sigma=ov_sigma, rng=rng)
            prev_c = np.concatenate([[d["open_log"][0]], d["close_log"][:-1]])
            cc_v.append(cc_var(d["close_log"], prev_c).mean())
            pk_v.append(parkinson_var(d["high"], d["low"]).mean())
            gk_v.append(gk_var(d["open"], d["high"], d["low"], d["close"]).mean())
        gb = (np.mean(gk_v) - total_var) / total_var
        cb = (np.mean(cc_v) - total_var) / total_var
        pb = (np.mean(pk_v) - total_var) / total_var
        gk_b.append(gb); cc_b.append(cb); pk_b.append(pb)
        print("  %5.2f | %+.2f%% | %+.2f%% | %+.2f%%" % (ratio, 100*gb, 100*cb, 100*pb))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.array(ratio_grid)
    ax.plot(x, 100*np.array(cc_b), "d-", color="#d62728", lw=2, label="Close-to-Close")
    ax.plot(x, 100*np.array(gk_b), "o-", color="#2ca02c", lw=2, label="Garman-Klass")
    ax.plot(x, 100*np.array(pk_b), "^-", color="#1f77b4", lw=2, label="Parkinson")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("隔夜波动 / 日内波动 比值")
    ax.set_ylabel("对『总方差』的估计偏差 (%)")
    ax.set_title("隔夜跳空：CC 上偏发散，GK 锚定在负偏区")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("%s/overnight-gap-blindspot.png" % OUT)
    plt.close()
    return ratio_grid, gk_b, cc_b


if __name__ == "__main__":
    stats, eff = exp_efficiency()
    exp_efficiency_vs_n()
    exp_drift()
    exp_overnight()
    print("\n全部图表已保存到:", OUT)
