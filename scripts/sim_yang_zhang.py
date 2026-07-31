#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yang-Zhang 波动率估计量：受控蒙特卡洛验证
固定随机种子 20260801。生成 4 张 PNG 并打印全部关键数字。

Yang-Zhang (2000) 把总波动分成三块：
  overnight variance  : 隔夜跳空方差  (前收 -> 次开)
  open-to-close var    : 日内开收方差
  Rogers-Satchell var  : 日内极差方差（对漂移免疫）
  YZ = overnight + k * open_close + (1-k) * RS
  k  = 0.34 / (1.34 + (n+1)/(n-1))
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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

OUT = "/Users/halo/workspace/astro-blog/public/images/yang-zhang-volatility-estimator"
SEED = 20260801
rng = np.random.default_rng(SEED)

LN2 = np.log(2.0)
PARK_C = 1.0 / (4.0 * LN2)
GK_C2 = 2.0 * LN2 - 1.0


# ---------- 日内 OHLC 生成器 ----------
def gen_ohlc(n_days, steps_per_day, sigma_day, mu_day=0.0,
             overnight_sigma=0.0, S0=100.0, rng=None):
    dt = 1.0 / steps_per_day
    sig_step = sigma_day * np.sqrt(dt)
    mu_step = mu_day * dt
    O = np.empty(n_days); H = np.empty(n_days)
    L = np.empty(n_days); C = np.empty(n_days)
    prev_close_log = np.log(S0)
    for d in range(n_days):
        gap = rng.normal(0.0, overnight_sigma) if overnight_sigma > 0 else 0.0
        open_log = prev_close_log + gap
        incr = rng.normal(mu_step, sig_step, steps_per_day)
        path = np.concatenate([[open_log], open_log + np.cumsum(incr)])
        O[d] = open_log; H[d] = path.max(); L[d] = path.min(); C[d] = path[-1]
        prev_close_log = C[d]
    return {"open": np.exp(O), "high": np.exp(H), "low": np.exp(L), "close": np.exp(C),
            "open_log": O, "close_log": C}


# ---------- 估计量（对一整段序列返回单一年化前的日方差） ----------
def cc_var_series(close_log, first_prev):
    prev = np.concatenate([[first_prev], close_log[:-1]])
    return (close_log - prev) ** 2  # per-day

def parkinson_var(h, l):
    return PARK_C * np.log(h / l) ** 2

def gk_var(o, h, l, c):
    return 0.5 * np.log(h / l) ** 2 - GK_C2 * np.log(c / o) ** 2

def rs_var(o, h, l, c):
    hc, ho = np.log(h / c), np.log(h / o)
    lc, lo = np.log(l / c), np.log(l / o)
    return hc * ho + lc * lo


def yang_zhang_var(d, first_prev_close_log):
    """
    返回整段窗口的 Yang-Zhang 日方差（标量）。
    d: gen_ohlc 输出
    first_prev_close_log: 第一天的前收（log）
    """
    o = d["open"]; h = d["high"]; l = d["low"]; c = d["close"]
    o_log = d["open_log"]; c_log = d["close_log"]
    n = len(c)
    prev_c_log = np.concatenate([[first_prev_close_log], c_log[:-1]])
    # 隔夜收益 o_i - c_{i-1}
    ov = o_log - prev_c_log
    ov_var = np.sum((ov - ov.mean()) ** 2) / (n - 1)
    # 开收收益 c_i - o_i
    oc = c_log - o_log
    oc_var = np.sum((oc - oc.mean()) ** 2) / (n - 1)
    # RS
    rs = rs_var(o, h, l, c)
    rs_var_val = rs.mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz = ov_var + k * oc_var + (1 - k) * rs_var_val
    return yz, ov_var, oc_var, rs_var_val, k


# ==================================================================
# 实验 1：隔夜场景下的总波动率还原
# ==================================================================
def exp_total_recovery(n_paths=4000, n_days=120, steps=234,
                       sigma_day=0.02, overnight_ratio=1.0):
    """真实总方差 = 日内方差 + 隔夜方差。看谁能还原总方差。"""
    intraday_var = sigma_day ** 2
    ov_sigma = sigma_day * overnight_ratio
    total_var = intraday_var + ov_sigma ** 2
    yz_all, gk_all, cc_all, pk_all = [], [], [], []
    for _ in range(n_paths):
        d = gen_ohlc(n_days, steps, sigma_day, mu_day=0.0,
                     overnight_sigma=ov_sigma, rng=rng)
        yz, *_ = yang_zhang_var(d, d["open_log"][0] - ov_sigma * 0)  # first prev≈open
        # 更严谨：第一天前收设为 open（无隔夜），影响可忽略（n_days大）
        yz_all.append(yz)
        gk_all.append(gk_var(d["open"], d["high"], d["low"], d["close"]).mean())
        cc = cc_var_series(d["close_log"], d["open_log"][0]).mean()
        cc_all.append(cc)
        pk_all.append(parkinson_var(d["high"], d["low"]).mean())
    yz_all = np.array(yz_all); gk_all = np.array(gk_all)
    cc_all = np.array(cc_all); pk_all = np.array(pk_all)

    print("\n=== 实验1 总波动率还原 (隔夜/日内比=%.1f) ===" % overnight_ratio)
    print("真实总方差=%.6g (日内=%.6g + 隔夜=%.6g)" % (total_var, intraday_var, ov_sigma**2))
    stats = {}
    for name, arr in [("YZ", yz_all), ("GK", gk_all), ("CC", cc_all), ("PARK", pk_all)]:
        bias = (arr.mean() - total_var) / total_var
        stats[name] = dict(mean=arr.mean(), std=arr.std(), var=arr.var(), bias=bias)
        print("  %-5s mean=%.6g std=%.4g bias=%+.2f%%" % (name, arr.mean(), arr.std(), 100*bias))
    print("  效率比 CC/YZ = %.2fx" % (cc_all.var() / yz_all.var()))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(min(cc_all.min(), yz_all.min()),
                       max(cc_all.max(), yz_all.max()), 70)
    ax.hist(cc_all, bins=bins, alpha=0.45, color="#d62728", label="Close-to-Close")
    ax.hist(gk_all, bins=bins, alpha=0.45, color="#2ca02c", label="Garman-Klass (只测日内)")
    ax.hist(yz_all, bins=bins, alpha=0.6, color="#ff7f0e", label="Yang-Zhang")
    ax.axvline(total_var, color="k", ls="--", lw=1.6, label="真实总方差")
    ax.axvline(intraday_var, color="gray", ls=":", lw=1.4, label="仅日内方差")
    ax.set_title("隔夜=日内时的总波动率还原（%d 路径 × %d 天）" % (n_paths, n_days))
    ax.set_xlabel("日均方差估计"); ax.set_ylabel("频数"); ax.legend()
    plt.tight_layout()
    plt.savefig("%s/total-variance-recovery.png" % OUT)
    plt.close()
    return stats


# ==================================================================
# 实验 2：三块方差分解
# ==================================================================
def exp_decomposition(n_paths=3000, n_days=120, steps=234, sigma_day=0.02,
                      ratio_grid=(0.0, 0.3, 0.5, 0.8, 1.2)):
    intraday_var = sigma_day ** 2
    print("\n=== 实验2 YZ 三块分解 ===")
    print("  隔夜比 | 隔夜项 | 开收项(权重k) | RS项(权重1-k) | YZ合计 | 真实总")
    ov_list, oc_list, rs_list, yz_list, total_list = [], [], [], [], []
    for ratio in ratio_grid:
        ov_sig = sigma_day * ratio
        total = intraday_var + ov_sig ** 2
        ov_a, oc_a, rs_a, yz_a, k_a = [], [], [], [], []
        for _ in range(n_paths):
            d = gen_ohlc(n_days, steps, sigma_day, overnight_sigma=ov_sig, rng=rng)
            yz, ov, oc, rs, k = yang_zhang_var(d, d["open_log"][0])
            ov_a.append(ov); oc_a.append(oc); rs_a.append(rs); yz_a.append(yz); k_a.append(k)
        ov_m, oc_m, rs_m, yz_m = np.mean(ov_a), np.mean(oc_a), np.mean(rs_a), np.mean(yz_a)
        k_m = np.mean(k_a)
        ov_list.append(ov_m); oc_list.append(k_m*oc_m); rs_list.append((1-k_m)*rs_m)
        yz_list.append(yz_m); total_list.append(total)
        print("  %5.2f | %.3g | %.3g | %.3g | %.3g | %.3g"
              % (ratio, ov_m, k_m*oc_m, (1-k_m)*rs_m, yz_m, total))

    x = np.arange(len(ratio_grid))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ov_arr = np.array(ov_list); oc_arr = np.array(oc_list); rs_arr = np.array(rs_list)
    ax.bar(x, ov_arr, label="隔夜项", color="#8c564b")
    ax.bar(x, oc_arr, bottom=ov_arr, label="开收项 (k)", color="#1f77b4")
    ax.bar(x, rs_arr, bottom=ov_arr+oc_arr, label="RS 日内项 (1-k)", color="#2ca02c")
    ax.plot(x, total_list, "k*--", ms=13, label="真实总方差")
    ax.set_xticks(x); ax.set_xticklabels(["%.1f" % r for r in ratio_grid])
    ax.set_xlabel("隔夜波动 / 日内波动 比值")
    ax.set_ylabel("方差贡献")
    ax.set_title("Yang-Zhang 三块方差分解：随隔夜比重构")
    ax.legend()
    plt.tight_layout()
    plt.savefig("%s/variance-decomposition.png" % OUT)
    plt.close()


# ==================================================================
# 实验 3：漂移免疫（YZ 继承 RS 的抗漂移能力）
# ==================================================================
def exp_drift(n_paths=3000, n_days=150, steps=234, sigma_day=0.02,
              overnight_ratio=0.5, drift_grid=(0.0, 0.005, 0.01, 0.02, 0.04)):
    intraday_var = sigma_day ** 2
    ov_sig = sigma_day * overnight_ratio
    total = intraday_var + ov_sig ** 2
    print("\n=== 实验3 漂移免疫 (隔夜比=%.1f, 真实总=%.6g) ===" % (overnight_ratio, total))
    print("  日漂移 | YZ偏差 | GK偏差 | CC偏差")
    yz_b, gk_b, cc_b = [], [], []
    for mu in drift_grid:
        yz_a, gk_a, cc_a = [], [], []
        for _ in range(n_paths):
            d = gen_ohlc(n_days, steps, sigma_day, mu_day=mu, overnight_sigma=ov_sig, rng=rng)
            yz, *_ = yang_zhang_var(d, d["open_log"][0])
            yz_a.append(yz)
            gk_a.append(gk_var(d["open"], d["high"], d["low"], d["close"]).mean())
            cc_a.append(cc_var_series(d["close_log"], d["open_log"][0]).mean())
        yb = (np.mean(yz_a)-total)/total
        gb = (np.mean(gk_a)-total)/total
        cb = (np.mean(cc_a)-total)/total
        yz_b.append(yb); gk_b.append(gb); cc_b.append(cb)
        print("  %6.3f | %+.2f%% | %+.2f%% | %+.2f%%" % (mu, 100*yb, 100*gb, 100*cb))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.array(drift_grid)
    ax.plot(x, 100*np.array(cc_b), "d-", color="#d62728", lw=2, label="Close-to-Close")
    ax.plot(x, 100*np.array(gk_b), "o-", color="#2ca02c", lw=2, label="Garman-Klass")
    ax.plot(x, 100*np.array(yz_b), "s-", color="#ff7f0e", lw=2.4, label="Yang-Zhang")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("每日漂移 μ (log 尺度)")
    ax.set_ylabel("对总方差的估计偏差 (%)")
    ax.set_title("漂移免疫：YZ 与 GK 稳定，CC 发散")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("%s/drift-immunity.png" % OUT)
    plt.close()


# ==================================================================
# 实验 4：效率比随样本量 + 综合评分
# ==================================================================
def exp_efficiency(n_paths=3000, day_grid=(10, 20, 40, 80, 160),
                   steps=234, sigma_day=0.02, overnight_ratio=0.5):
    intraday_var = sigma_day ** 2
    ov_sig = sigma_day * overnight_ratio
    total = intraday_var + ov_sig ** 2
    print("\n=== 实验4 效率比随样本量 (隔夜比=%.1f) ===" % overnight_ratio)
    eff_yz, rmse_cc, rmse_yz, rmse_gk = [], [], [], []
    for nd in day_grid:
        yz_a, cc_a, gk_a = [], [], []
        for _ in range(n_paths):
            d = gen_ohlc(nd, steps, sigma_day, overnight_sigma=ov_sig, rng=rng)
            yz, *_ = yang_zhang_var(d, d["open_log"][0])
            yz_a.append(yz)
            cc_a.append(cc_var_series(d["close_log"], d["open_log"][0]).mean())
            gk_a.append(gk_var(d["open"], d["high"], d["low"], d["close"]).mean())
        yz_a = np.array(yz_a); cc_a = np.array(cc_a); gk_a = np.array(gk_a)
        e = cc_a.var() / yz_a.var()
        eff_yz.append(e)
        rmse_cc.append(np.sqrt(np.mean((cc_a-total)**2)))
        rmse_yz.append(np.sqrt(np.mean((yz_a-total)**2)))
        rmse_gk.append(np.sqrt(np.mean((gk_a-total)**2)))
        print("  n_days=%4d eff CC/YZ=%.2fx rmse_cc=%.3g rmse_yz=%.3g rmse_gk=%.3g"
              % (nd, e, rmse_cc[-1], rmse_yz[-1], rmse_gk[-1]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(day_grid, rmse_cc, "d-", color="#d62728", label="CC")
    ax.plot(day_grid, rmse_gk, "o-", color="#2ca02c", label="GK (漏隔夜)")
    ax.plot(day_grid, rmse_yz, "s-", color="#ff7f0e", label="Yang-Zhang")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("样本天数"); ax.set_ylabel("RMSE (对总方差)")
    ax.set_title("RMSE：YZ 兼顾无偏与低方差"); ax.legend(); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(day_grid, eff_yz, "s-", color="#ff7f0e", lw=2, label="CC/YZ 效率比")
    ax.set_xscale("log")
    ax.set_xlabel("样本天数"); ax.set_ylabel("方差效率比 CC/YZ")
    ax.set_title("YZ 对 CC 的效率优势"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("%s/efficiency-vs-sample-size.png" % OUT)
    plt.close()


if __name__ == "__main__":
    exp_total_recovery()
    exp_decomposition()
    exp_drift()
    exp_efficiency()
    print("\n全部图表已保存到:", OUT)
