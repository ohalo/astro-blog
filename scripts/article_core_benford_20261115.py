# -*- coding: utf-8 -*-
"""本福特定律（Benford's Law）第一数字审计：什么数据真的服从、什么会翻车。

本文全部数字的唯一来源。运行本脚本会：
1. 现场生成多组已知分布（对数正态不同 log 宽度、均匀、GBM 价格与日收益）；
2. 计算第一数字经验分布、与理论 Benford 的 MAD（Nigrini 准则）与 KS 距离；
3. 画 5 张图到 public/images/benford-law-first-digit-audit/；
4. 把全部统计量写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261115
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "benford-law-first-digit-audit"
)
os.makedirs(_OUT, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


# 第一有效数字：取 |x| > 0，d = floor(10^{frac(log10|x|)})
def first_digit(x):
    x = np.asarray(x, dtype=float)
    x = np.abs(x)
    x = x[x > 0]
    lx = np.log10(x)
    frac = lx - np.floor(lx)
    return np.floor(10.0 ** frac + 1e-12).astype(int)


def empirical_first_digit(x, digits=range(1, 10)):
    d = first_digit(x)
    counts = np.array([float(np.sum(d == k)) for k in digits])
    return counts / counts.sum()


BENFORD = np.log10(1.0 + 1.0 / np.arange(1, 10))


def mad_emp(x):
    """经验分布与 Benford 的平均绝对偏差（Nigrini 准则）。"""
    emp = empirical_first_digit(x)
    return float(np.mean(np.abs(emp - BENFORD)))


def ks_discrete(x):
    """离散 KS 距离：累积分布的最大偏差。"""
    emp = empirical_first_digit(x)
    cum_emp = np.cumsum(emp)
    cum_ben = np.cumsum(BENFORD)
    return float(np.max(np.abs(cum_emp - cum_ben)))


def benford_sample(n, rng):
    """从精确 Benford 分布抽 n 个「第一数字」（逆 CDF）。"""
    u = rng.random(n)
    cum = np.cumsum(BENFORD)
    return np.array([int(np.searchsorted(cum, uu)) + 1 for uu in u])


def main():
    rng = np.random.default_rng(SEED)
    N = 5000

    # ---------- 实验 1：log 宽度越大，越逼近 Benford ----------
    sigmas = np.round(np.arange(0.1, 2.01, 0.1), 2)
    mads = []
    for s in sigmas:
        # 对数正态 LN(0, s^2)：log 宽度 = s
        samples = rng.lognormal(mean=0.0, sigma=float(s), size=N)
        mads.append(mad_emp(samples))
    mads = np.array(mads)

    # ---------- 实验 2：三种宽度的第一数字直方图 ----------
    widths = [0.2, 1.0, 2.0]
    hist_data = {}
    for w in widths:
        hist_data[w] = empirical_first_digit(
            rng.lognormal(mean=0.0, sigma=w, size=N))

    # ---------- 实验 3：价格服从、收益翻车 ----------
    # GBM：价格横跨多个数量级 -> 近似尺度不变 -> 接近 Benford
    # 日收益集中在 0 附近 -> 第一数字塌在 1 -> 大幅偏离
    mu, sigma, ndays = 0.0007, 0.025, 6000
    z = rng.standard_normal(ndays)
    price = 100.0 * np.cumprod(np.exp(mu + sigma * z))
    ret = price[1:] / price[:-1] - 1.0
    mad_price = mad_emp(price)
    mad_ret = mad_emp(np.abs(ret))
    ks_price = ks_discrete(price)
    ks_ret = ks_discrete(np.abs(ret))
    # 对照：均匀 [1,9] 应明显不服从
    mad_unif = mad_emp(rng.uniform(1.0, 9.0, N))
    ks_unif = ks_discrete(rng.uniform(1.0, 9.0, N))

    # ---------- 实验 4：KS 检验的「尺寸」与「功效」 ----------
    # 真实 Benford 下重抽 -> KS 临界值（α=0.05）与误报率
    N_KS = 2000
    n_trials = 500
    ds_benford = []
    for _ in range(n_trials):
        samp = benford_sample(N_KS, rng)
        cum = np.cumsum([np.mean(samp == k) for k in range(1, 10)])
        ds_benford.append(float(np.max(np.abs(cum - np.cumsum(BENFORD)))))
    crit_05 = float(np.quantile(ds_benford, 0.95))
    # 真实 Benford 下用临界值判定的误报率（应≈0.05）
    false_alarm = float(np.mean(np.array(ds_benford) > crit_05))
    # 功效：对 log 正态不同 σ，有多大比例被判「不服从」
    pow_sigmas = np.array([0.4, 0.8, 1.2, 1.6])
    power = {}
    for s in pow_sigmas:
        rej = 0
        for _ in range(n_trials):
            samp = first_digit(rng.lognormal(0.0, float(s), N_KS))
            cum = np.cumsum([np.mean(samp == k) for k in range(1, 10)])
            d = np.max(np.abs(cum - np.cumsum(BENFORD)))
            rej += int(d > crit_05)
        power[float(s)] = rej / n_trials

    # ---------- 实验 5：样本量敏感性 ----------
    sizes = [100, 250, 500, 1000, 2000, 5000]
    mad_vs_n = {}
    for nsize in sizes:
        reps = []
        for _ in range(60):
            reps.append(mad_emp(rng.lognormal(0.0, 1.0, nsize)))
        mad_vs_n[int(nsize)] = {
            "mean": float(np.mean(reps)),
            "lo": float(np.percentile(reps, 5)),
            "hi": float(np.percentile(reps, 95)),
        }

    # ===================== 画图 =====================
    # 图1：三种 log 宽度的第一数字 vs Benford
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, w in zip(axes, widths):
        emp = hist_data[w]
        ax.bar(range(1, 10), emp, 0.6, color="#1f6feb", alpha=0.8,
               label="经验")
        ax.plot(range(1, 10), BENFORD, "o-", color="#d1242f", lw=1.4,
                label="Benford")
        ax.set_xticks(range(1, 10))
        ax.set_title(f"σ_log={w}: MAD={float(np.mean(np.abs(hist_data[w] - BENFORD))):.4f}",
                     fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("第一数字概率")
    fig.suptitle("图1 对数正态的 log 宽度越大，第一数字越逼近 Benford",
                 fontsize=11)
    _save(fig, "fig1_width_histograms.png")

    # 图2：MAD 随 log 宽度的收敛曲线 + Nigrini 阈值
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(sigmas, mads, "o-", color="#1f6feb", lw=1.6, ms=4)
    ax.axhline(0.006, color="#2da44e", ls="--", lw=1.4,
               label="紧密符合 (<0.006)")
    ax.axhline(0.012, color="#bf8700", ls="--", lw=1.4,
               label="可接受的临界 (0.012)")
    ax.axhline(0.015, color="#d1242f", ls="--", lw=1.4,
               label="不合规 (>0.015)")
    ax.set_xlabel("对数正态的 log 宽度 σ")
    ax.set_ylabel("MAD（与 Benford 的平均绝对偏差）")
    ax.set_title("图2 MAD 随分布「跨度」单调下降：Benford 要求数据横跨多个数量级")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    _save(fig, "fig2_mad_convergence.png")

    # 图3：价格 vs 收益 vs 均匀 的第一数字对比
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    emp_price = empirical_first_digit(price)
    emp_ret = empirical_first_digit(np.abs(ret))
    emp_unif = empirical_first_digit(rng.uniform(1.0, 9.0, N))
    x = np.arange(1, 10)
    w = 0.27
    ax.bar(x - w, emp_price, w, color="#2da44e", label=f"GBM 价格 MAD={mad_price:.4f}")
    ax.bar(x, emp_ret, w, color="#d1242f", label=f"日收益|r| MAD={mad_ret:.4f}")
    ax.bar(x + w, emp_unif, w, color="#8250df", label=f"均匀[1,9] MAD={mad_unif:.4f}")
    ax.plot(x, BENFORD, "ko-", lw=1.4, label="Benford")
    ax.set_xticks(x)
    ax.set_ylabel("第一数字概率")
    ax.set_title("图3 价格更靠近合规、收益彻底翻车、均匀完全不匹配")
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig3_price_vs_return.png")

    # 图4：KS 临界值分布 + 功效曲线
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].hist(ds_benford, bins=40, color="#1f6feb", alpha=0.8)
    axes[0].axvline(crit_05, color="#d1242f", ls="--", lw=1.6,
                    label=f"临界值 D*={crit_05:.4f}")
    axes[0].set_title(f"真实 Benford 下 KS 距离分布\n误报率={false_alarm:.3f}",
                      fontsize=10)
    axes[0].set_xlabel("KS 距离 D")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3)
    pw = [power[float(s)] for s in pow_sigmas]
    axes[1].plot(pow_sigmas, pw, "s-", color="#bf8700", lw=1.6, ms=6)
    axes[1].set_xlabel("对数正态 log 宽度 σ")
    axes[1].set_ylabel("被 KS 判为「不服从」的比例 (功效)")
    axes[1].set_title("图4 KS 检验的功效：σ 越大越容易抓到偏离", fontsize=10)
    axes[1].grid(alpha=0.3)
    _save(fig, "fig4_ks_power.png")

    # 图5：MAD 随样本量的波动
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    xs = [int(s) for s in sizes]
    means = [mad_vs_n[s]["mean"] for s in xs]
    los = [mad_vs_n[s]["lo"] for s in xs]
    his = [mad_vs_n[s]["hi"] for s in xs]
    ax.fill_between(xs, los, his, color="#1f6feb", alpha=0.2, label="5%-95% 区间")
    ax.plot(xs, means, "o-", color="#1f6feb", lw=1.8, label="平均 MAD")
    ax.set_xscale("log")
    ax.set_xlabel("样本量 N（对数轴）")
    ax.set_ylabel("对 LN(0,1) 抽样的 MAD")
    ax.set_title("图5 小样本下 MAD 波动巨大：N<500 时单点估计不可信")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "fig5_sample_size.png")

    results = {
        "dgp": {"N": N, "seed": SEED},
        "benford_mass": BENFORD.tolist(),
        "width_convergence": {
            "sigmas": sigmas.tolist(),
            "mads": mads.tolist(),
            "mad_at_0.2": float(mads[np.where(np.isclose(sigmas, 0.2))[0][0]]),
            "mad_at_1.0": float(mads[np.where(np.isclose(sigmas, 1.0))[0][0]]),
            "mad_at_2.0": float(mads[np.where(np.isclose(sigmas, 2.0))[0][0]]),
        },
        "price_vs_return": {
            "mad_price": mad_price,
            "mad_return_abs": mad_ret,
            "ks_price": ks_price,
            "ks_return_abs": ks_ret,
            "mad_uniform": mad_unif,
            "ks_uniform": ks_unif,
            "gbm_params": {"mu": mu, "sigma": sigma, "ndays": ndays},
        },
        "ks_test": {
            "N_KS": N_KS,
            "n_trials": n_trials,
            "critical_05": crit_05,
            "false_alarm_rate": false_alarm,
            "power": {str(float(s)): float(power[float(s)]) for s in pow_sigmas},
        },
        "sample_size": {str(s): mad_vs_n[int(s)] for s in sizes},
    }
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
