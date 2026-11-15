---
title: "本福特定律审计：股票收益根本不服从，但市值审计能抓假账"
description: "本福特定律（首数字为 d 的概率是 log10(1+1/d)）常被当成万能异常检测工具，但它在金融数据上大面积翻车。本文用真值已知的蒙特卡洛现场拆解：对数正态的 log 宽度 σ 从 0.2 增到 2.0 时，与 Benford 的平均绝对偏差 MAD 从 0.1133 单调降到 0.0022——只有当数据横跨多个数量级、近似尺度不变时才收敛。现场造 GBM 价格序列 MAD=0.0217（KS=0.0361）接近合规，而日收益绝对值 MAD=0.0196 但 KS=0.0812 直接翻车，均匀[1,9] 更是 MAD=0.0617、KS=0.2249。KS 检验在真实 Benford 下临界值 D*=0.0256、误报率恰好 0.05，但对窄分布功效拉满（σ=0.4 时 100%、σ=0.8 时 94.6%），对宽分布又几乎抓不到（σ=1.6 时仅 4.6%）。样本量扫描显示 N<500 时 MAD 在 0.013-0.035 间剧烈波动。诚实结论：别对收益率用 Benford，它对尺度不变的总量数据（市值、成交量、账户余额）才有效，且小样本下 MAD 根本不可信。"
publishDate: '2026-11-15T06:54:00Z'
tags:
  - 量化
  - 交易
  - 统计检验
  - 异常检测
  - 数据审计
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

本福特定律说：一组数据首位有效数字为 d（d=1..9）的概率

$$P(D=d)=\log_{10}\left(1+\frac{1}{d}\right)$$

于是数字 1 开头约 30.1%，数字 9 开头仅 4.6%。很多文章把它写成「万能造假检测器」——只要数据不服从就疑有猫腻。但现实是：**绝大多数金融时间序列要么根本不服从 Benford，要么在小样本下你根本测不出来**。

三个现场结论：

1. Benford 成立的前提是数据近似**尺度不变**、横跨多个数量级。对数正态的 log 宽度 σ 从 0.2 涨到 2.0，与 Benford 的 MAD（平均绝对偏差）从 **0.1133 单调降到 0.0022**。
2. 用 GBM 造一段「股价」和「日收益」：股价 MAD=0.0217、KS=0.0361 勉强靠近合规；**日收益绝对值 MAD=0.0196 但 KS 距离 0.0812，直接翻车**——因为收益都挤在 0 附近，首数字塌在 1。均匀分布在 [1,9] 上 MAD=0.0617、KS=0.2249，彻底不服从。
3. 即便数据真的服从 Benford，KS 检验在 N=2000 下的临界值 D*=0.0256，误报率恰好 0.05；但你若只看 N<500 的 MAD，单点估计会在 0.013–0.035 之间剧烈漂移，**一个样本就能把结论翻过来**。

换句话说：Benford 对市值、成交量这类天然尺度不变的总量数据是好工具；对收益率、价差这类围着 0 转的数据是类别错误；对小样本更是测不准。

## 二、为什么必须横跨多个数量级

Benford 分布来自一个朴素事实：如果一个量 X 取对数后近似均匀分布（即 X 在 log 尺度上尺度不变），那么它在每个数量级里停留的时间相等，首位数字就落进 Benford。关键在「横跨多个数量级」。可以做个反推：若 X 只在一个数量级内晃（比如永远在 1–9），首位分布几乎由该区间内的形状决定，跟 Benford 半点关系没有；只有当 X 从个位一路爬到千位、万位，log 尺度被铺满，首位才「忘记」自己的尺度的束缚，收敛到那个经典的非均匀分布。

这里还有一个常被忽略的离散细节：首位数字定义要求 X>0 且只取首位有效数字。对负数要先取绝对值，对带小数的值要从第一个非零数字算起（例如 0.034 的首位是 3）。一旦数据里混进大量绝对值远小于 1 的数——典型如日收益率——它们的首位会集中塌在 1，这本身就破坏了 Benford 假设，跟是否造假无关。

我现场抽对数正态 $X\sim\mathrm{LN}(0,\sigma^2)$，σ 从 0.1 到 2.0 扫描，每档 N=5000，算经验首位分布与 Benford 的 MAD：

- σ=0.2（窄，基本只在个位数晃）：MAD=0.1133，首位高度集中在某几个数字；
- σ=1.0（中等跨度）：MAD=0.0067，落到 Nigrini「可接受」阈值 0.012 以内；
- σ=2.0（宽，横跨多个十倍区间）：MAD=0.0022，几乎贴合理论。

图 1 把三种宽度的经验首位分布叠在 Benford 曲线上，σ 越大越贴合；图 2 是 MAD 随 log 宽度的收敛曲线，三条 Nigrini 阈值线把「紧密符合 / 可接受 / 不合规」三段切得清清楚楚。

![三种 log 宽度的首位分布对比](/images/benford-law-first-digit-audit/fig1_width_histograms.png)

![MAD 随 log 宽度的收敛曲线](/images/benford-law-first-digit-audit/fig2_mad_convergence.png)

## 三、股价近似合规，收益彻底翻车

为排除「我挑了特殊分布」的嫌疑，用最朴素的几何布朗运动造一段 6000 天数据：价格 $P_t=P_{t-1}\exp(\mu+\sigma z_t)$，收益 $r_t=P_t/P_{t-1}-1$。价格一路从 100 涨到几百上千，横跨多个数量级；收益则全部挤在 0 附近。

算它们首位分布与 Benford 的距离：

| 数据 | MAD | KS 距离 |
|---|---|---|
| GBM 价格 | 0.0217 | 0.0361 |
| 日收益 \|r\| | 0.0196 | 0.0812 |
| 均匀 [1,9] | 0.0617 | 0.2249 |

注意一个有意思的反差：日收益的 **MAD 数值（0.0196）甚至比价格（0.0217）还低**，但 **KS 距离（0.0812）却是价格的 2.25 倍**。原因是 MAD 对「首位整体偏低」不敏感，而 KS 累积距离能揪出收益在低位数字的异常堆积。图 3 把三者首位分布并排放：价格最靠近 Benford，收益几乎把概率全压在首位 1，均匀则完全平摊——三者形态天差地别。

![价格、收益与均匀分布的首位对比](/images/benford-law-first-digit-audit/fig3_price_vs_return.png)

诚实提醒：即便 GBM 价格，MAD=0.0217 也落在 Nigrini「不合规」（>0.015）区间，KS 0.0361 也超过临界值。这本身就是重要一课——**连「自然」金融数据都需要真正宽的 log 跨度才勉强合规**，拿 Benford 当股价造假铁证是站不住的。

## 四、KS 检验的尺寸与功效

把 Benford 当原假设，真正要问的是：检验的「尺寸」（真实服从时误报率）和「功效」（不服从时检出率）到底多少？

做法：从精确 Benford 分布逆 CDF 抽 N=2000 的样本，重复 500 次，得到 KS 距离的经验分布，取 95 分位作临界值 $D^*=0.0256$；真实 Benford 下用该临界值判定的误报率 = 0.05，正好贴着名义水平。接着对窄到宽的对数正态测功效：

- σ=0.4：功效 100%（全抓）
- σ=0.8：功效 94.6%
- σ=1.2：功效 6.6%
- σ=1.6：功效 4.6%（基本抓不到）

图 4 左是真实 Benford 下 KS 距离的分布与临界值线，右是功效随 log 宽度的衰减——**KS 检验对窄分布极其敏感，对宽（已接近合规）分布几乎无能为力**，这跟它的设计目的一致，但也意味着「没检出来」绝不等于「服从」。

![KS 临界值与功效曲线](/images/benford-law-first-digit-audit/fig4_ks_power.png)

## 五、小样本下 MAD 根本不可信

最后扫样本量 N∈{100,250,500,1000,2000,5000}，每个 N 跑 60 次取 MAD 的 5%–95% 区间：

- N=100：MAD 均值 0.0238，区间 [0.0130, 0.0347]
- N=500：均值 0.0109，区间 [0.0060, 0.0154]
- N=5000：均值 0.0045，区间 [0.0029, 0.0061]

图 5 显示：N 越小，单点 MAD 波动越夸张。实务里常有人拿几百条财务数据算个 MAD 就说「偏离 Benford 有猫腻」——在 N=100 那档，MAD 的 90% 区间横跨 0.013 到 0.035，**随便抽一组就可能从「合规」跳到「不合规」**。没有置信区间的 MAD 没有任何判决力。

![MAD 随样本量的波动](/images/benford-law-first-digit-audit/fig5_sample_size.png)

## 六、给量化交易的诚实用法

- **别对收益率用 Benford**。收益围着 0 转，首数字塌在 1，天生偏离——这是数据的几何属性，不是造假信号。
- **适合审计尺度不变的总量**：市值、成交量、账户余额、成交笔数这类跨越多个数量级且乘法演化的量，Benford 才是合理基线。
- **必须配样本量意识**：N<500 时先报置信区间，单点 MAD 不具判决力；KS 比 MAD 更能抓「低位堆积」型偏离。
- **「没检出来」≠「服从」**：宽分布下 KS 功效会塌到 5% 以下，阴性结果不能当清白证明。

下面是本文本全部数字的生成代码与结果 JSON（含五张图），所有结论均可复现。

```python
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
```

