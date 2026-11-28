---
title: "电力现货尖峰价格与峰负荷期权：用带状态切换的跳扩散复现极端电价，为什么正态模型系统性低估尖峰、又在负电价时段彻底失灵"
description: "1095 天合成现货电价中，99.9% 分位 96.07 而按稳健主体尺度拟合的正态只给 82.24。22 日均价看涨在 K=45/55/65 的跳扩散价为 8.6100/1.2166/0.0477，正态模型只有 1.3434/0.0985/0.0021，低估 540.91%/1135.43%/2129.77%。跳跃强度从 0.0190 错估到 0.0066 时 K=65 价格少 61.37%。高新能源场景下 20% 价格为负，零执行价看跌真值 1.6793 而对数正态报 0。附完整代码与五张真实计算图。"
publishDate: '2026-11-28T19:30:00Z'
tags:
  - 量化
  - 衍生品定价
  - 电力市场
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

电力现货和股票最大的差别，是价格会在几小时内翻几倍，也会被新能源出力压到零以下。这两个现象都不在这样的教科书假设里：对数价格服从布朗运动、收益独立同分布、负价格不可能出现。

本文用一个可复现的合成市场把问题量化。1095 天日度电价里，均值 40.2664、标准差 15.9014，99.9% 分位达到 96.0726；如果只用主体数据做稳健正态拟合，同一分位只有 82.2373，低估 16.8%。更关键的是定价：22 日均价看涨在 K=45、55、65 三档，跳扩散模型给出 8.6100、1.2166、0.0477，正态模型只有 1.3434、0.0985、0.0021，分别低估 540.91%、1135.43% 和 2129.77%。把跳跃强度从真实的 0.0190 错估成 0.0066，K=65 的价格又会少 61.37%。当新能源占比高到 20% 的交易日出现负电价，零执行价看跌真值 1.6793，对数正态却只能报 0。

结论不是「跳跃模型永远更贵」，而是尾部与负价这两件事必须由模型结构承载，不能靠调波动率参数补救。

![电价路径与尖峰](/images/electricity-spike-jump-pricing/fig1_price_path.png)

## 二、均值回复加状态切换跳跃

价格写成季节项、新能源压价项与随机残差之和。季节项取 45 加上年周期与半年周期；新能源状态以 0.18 的概率出现，出现时把价格中枢下压 30 元。残差服从一阶均值回复，系数 0.72，扩散波动 4.0，再叠加泊松跳跃。

市场有两态。平静态跳跃概率 0.012，压力态 0.080；状态以 0.008 的概率进入压力、以 0.070 的概率退出。两条链的稳态压力占比是 0.1026，因此全样本平均跳跃强度约 0.0190，也就是每 53 天一次。跳跃幅度不是对称的：72% 为向上跳，均值 38、标准差 12；其余向下跳，均值 -28、标准差 9。这个结构直接对应电力市场的供给冲击：机组跳闸把价格顶上去，新能源大发把它按下去。

为什么不用几何布朗运动？因为对数变换要求价格恒正，而本文正文第五节的场景里有 20% 的价格小于零。算术过程没有这个限制，代价是模型不保证价格非负——对电力现货而言这反而是特性而非缺陷。

## 三、正态校准漏掉的不是波动，是尾部结构

把 1095 天样本的均值与标准差直接代入正态，99.5% 分位是 81.2256，实测样本是 82.5020，看起来只差一点点；99.9% 分位正态给 89.4053，样本给 96.0726，差 7.5%。这个对比容易让人以为「正态其实还行」。

问题在于标准差本身被尖峰和负价撑大了。改用中位数与 MAD 估计主体尺度，得到 13.4421，相应正态 99.9% 分位只有 82.2373；真实分位除以它是 1.1682，即尾部比主体尺度预测的高 16.8%。这才是尖峰的真实代价：它不是把整体波动抬高，而是在主体分布之外多出一块正态根本没见过的质量。

稳健尺度还有操作价值。用 13.4421 而不是 15.9014 做日常风险限额，等于承认尖峰是独立风险因子，需要单独的跳变限额管理；把两者混在一个标准差里，限额会被几个尖峰永久抬高，随后在平静期又过于宽松。

跳跃识别也要小心。直接用相邻价格差减季节差去阈值，会把新能源状态切换造成的 30 元阶跃当成跳跃，阈值 18 时能误报 343 次，占总天数 31.4%。先把可观测的新能源状态从差分里剔除，阈值 12、18、25 分别只保留 26、11、8 次，检出率 2.38%、1.01%、0.73%。这说明强度估计对阈值极敏感：取 18 会漏掉近一半真跳跃，而取 12 又把一部分扩散噪声算进来。

![分布与尾部对拍](/images/electricity-spike-jump-pricing/fig2_tail_distribution.png)

## 四、峰负荷期权：越过执行价才暴露差异

交易上最直接的检验是 22 日均价看涨，折现率 3%。用 6000 条路径对拍跳扩散与正态模型，正态模型保留同样的均值回复系数，只把创新波动按样本标准差反解，使两者的边际方差一致。

在 K=45，跳扩散价 8.6100，正态价 1.3434；K=55 是 1.2166 对 0.0985；K=65 是 0.0477 对 0.0021。相对低估分别为 540.91%、1135.43%、2129.77%。请注意最后这个百分比：分母只有两厘，百分比看似夸张，绝对差只有 4.55 分。真正值得关注的是前两档——它们是能成交的合约，差价达到 7.27 元和 1.12 元。

跳跃强度误差的影响集中在虚值端。把强度从 0.0190 降到 0.0066，K=45 只少 4.88%，K=55 少 21.09%，K=65 少 61.37%，K=75 直接归零。近月平值合约主要由扩散决定，远端虚值合约几乎全靠跳跃，因此历史上平静期估出来的强度拿去给压力期定价，会系统性地把尾部白送出去。

均价合约还有一层容易被忽视的平滑效应。单日价格 99.9% 分位是 96.07，但 22 日均值的分布会由于平均而收窄，孤立的单日尖峰被摊薄。然而跳跃强度本身随状态切换，压力态一旦持续多日，22 日均值就不再是简单的方差缩小：6000 条路径里均价高于 65 的样本来自少数几条连续压力态路径，而不是均匀分散的独立尖峰。这正是正态模型最危险的错位——它把极端事件当成独立同分布的尾部点，真实电价的风险却是成簇出现的。用日度波动率乘以 22 的平方根去外推月度合约波动，会同时高估常规波动、低估尾部跳升。折现率 3% 在三周尺度上只有约 0.18% 的影响，真正决定价格的是路径里有没有落到压力态。

![均价看涨定价对比](/images/electricity-spike-jump-pricing/fig3_peak_call_prices.png)

## 五、负电价让对数正态失去左尾

把新能源概率提到 0.40、压价幅度提到 45，4000 天样本里有 20.00% 的价格为负，均值 28.1903、标准差 25.0888。这时看跌和看涨的定价逻辑都变了。

零执行价看跌的收益就是负价格的绝对值，跳扩散折现后为 1.6793。对数正态在结构上给不出任何负值，因此报 0——这不是参数误差，而是支撑集本身的矛盾。10 执行价看涨在算术模型里是 22.3738；只拿正价格拟合对数正态得到 35.6017，高估 59.1%，因为条件分布被抬高后所有质量都堆在零以上。若改成用全样本均值方差做矩匹配，对数正态报 18.6797，又低估 16.5%。同一个标的，两种看似合理的对数正态校准给出方向相反的误差，正说明左尾不能被对数变换凭空造出来。

![负电价分布与期权](/images/electricity-spike-jump-pricing/fig4_negative_price.png)

## 六、可交易的含义与边界

一个稳妥的流程是：先只用主体数据估季节与均值回复参数，再用剔除新能源阶跃后的残差估跳跃强度，最后分别报告平静态与压力态强度下的期权价格区间。不要用一个混合强度给全部期限报价，因为压力态占比本身会随季节变化。

对冲上也有直接含义。尖峰期的风险敞口变化不是连续的，跨越压力态转移概率时会出现跳变，用历史波动率算出的对冲比率会在尖峰来临时失效。更稳妥的做法是同时持有少量虚值看涨作为状态保险，并按压力态的稳态概率 10.26% 而不是平静期样本来给这份保险定价。

模型仍有明显边界。跳跃幅度被设成与状态无关的固定分布，真实电价的尖峰幅度与燃料成本、机组组合强相关；新能源压价被写成独立伯努利变量，忽略了风电出力在空间上的相关性；本文也没有纳入市场限价与规则干预，而现实中价格上限会截断最右尾。这些都会改变绝对价格，但不会改变核心结论：正态模型错在结构，不在参数。

![强度敏感性与错误定价](/images/electricity-spike-jump-pricing/fig5_intensity_sensitivity.png)

## 七、完整可运行代码

下面代码生成全部数据、五张图与 results.json，可独立运行。

```python
# -*- coding: utf-8 -*-
"""Regime-switching jump diffusion for electricity prices and peak-call pricing."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

SEED = 20261128
N_DAYS = 3 * 365
PHI = 0.72
SIGMA = 4.0
P01 = 0.008
P10 = 0.070
P_JUMP_CALM = 0.012
P_JUMP_STRESS = 0.080
WIND_PROB = 0.18
WIND_SHIFT = 30.0
JUMP_UP_P = 0.72
JUMP_UP_MU = 38.0
JUMP_UP_SD = 12.0
JUMP_DN_MU = 28.0
JUMP_DN_SD = 9.0
RATE = 0.03
N_PATHS = 6000
HORIZON = 22
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "public", "images", "electricity-spike-jump-pricing")


def seasonal(t):
    return 45.0 + 12.0 * np.cos(2.0 * np.pi * (t - 20.0) / 365.0) + 4.0 * np.sin(4.0 * np.pi * t / 365.0)


def simulate(rng, n, start_state=0, wind_prob=WIND_PROB, wind_shift=WIND_SHIFT,
             phi=PHI, sigma=SIGMA, calm_p=P_JUMP_CALM, stress_p=P_JUMP_STRESS,
             t0=0, track=False):
    state = start_state
    x = 0.0
    prices = np.empty(n)
    jumps = np.empty(n)
    winds = np.empty(n, dtype=bool)
    states = np.empty(n, dtype=int)
    eps = rng.standard_normal(n)
    unif = rng.random(n)
    jsz = rng.standard_normal(n)
    for i in range(n):
        t = t0 + i
        if state == 0 and unif[i] < P01:
            state = 1
        elif state == 1 and unif[i] < P10:
            state = 0
        w = rng.random() < wind_prob
        pj = stress_p if state == 1 else calm_p
        j = 0.0
        if unif[i] < pj:
            if rng.random() < JUMP_UP_P:
                j = JUMP_UP_MU + JUMP_UP_SD * jsz[i]
            else:
                j = -JUMP_DN_MU + JUMP_DN_SD * jsz[i]
        x = phi * x + sigma * eps[i] + j
        prices[i] = seasonal(t) - wind_shift * w + x
        jumps[i] = j
        winds[i] = w
        states[i] = state
    if track:
        return prices, jumps, winds, states
    return prices


def main():
    rng = np.random.default_rng(SEED)
    hist, hjump, hwind, hstate = simulate(rng, N_DAYS, track=True)

    hist_mean = float(np.mean(hist))
    hist_sd = float(np.std(hist, ddof=1))
    neg_share = float(np.mean(hist < 0.0))
    q995 = float(np.quantile(hist, 0.995))
    q999 = float(np.quantile(hist, 0.999))
    gauss_q995 = hist_mean + norm.ppf(0.995) * hist_sd
    gauss_q999 = hist_mean + norm.ppf(0.999) * hist_sd
    tail_ratio = q999 / gauss_q999

    diff = np.diff(hist)
    seas_diff = np.array([seasonal(t + 1) - seasonal(t) for t in range(N_DAYS - 1)])
    wind_diff = -WIND_SHIFT * (hwind[1:].astype(float) - hwind[:-1].astype(float))
    resid_diff = diff - seas_diff - wind_diff
    det_counts = {}
    det_rates = {}
    for thr in (12.0, 18.0, 25.0):
        cnt = int(np.sum(np.abs(resid_diff) > thr))
        det_counts[str(thr)] = cnt
        det_rates[str(thr)] = round(cnt / (N_DAYS - 1), 6)

    med = float(np.median(hist))
    mad_sd = 1.4826 * float(np.median(np.abs(hist - med)))
    robust_q999 = med + norm.ppf(0.999) * mad_sd

    # reference density: stationary stress probability from the two-state chain
    pi_stress = P01 / (P01 + P10)
    avg_jump_p = (1.0 - pi_stress) * P_JUMP_CALM + pi_stress * P_JUMP_STRESS

    # monthly peak-call: average of 22 day-ahead prices starting at t=0
    strikes = [45.0, 55.0, 65.0, 75.0, 85.0]
    disc = np.exp(-RATE * HORIZON / 365.0)
    jump_avg = np.empty(N_PATHS)
    gauss_avg = np.empty(N_PATHS)
    low_avg = np.empty(N_PATHS)
    for i in range(N_PATHS):
        p = simulate(rng, HORIZON, start_state=0, t0=0)
        jump_avg[i] = np.mean(p)
        g = np.full(HORIZON, hist_mean)
        eps = rng.standard_normal(HORIZON)
        x = np.zeros(HORIZON)
        sig_g = hist_sd * np.sqrt(1.0 - PHI ** 2)
        for k in range(HORIZON):
            x[k] = PHI * x[k - 1] + sig_g * eps[k] if k > 0 else sig_g * eps[k]
        gauss_avg[i] = np.mean(g + x)
        low_avg[i] = np.mean(simulate(rng, HORIZON, start_state=0, t0=0,
                                      calm_p=P_JUMP_CALM * 0.35, stress_p=P_JUMP_STRESS * 0.35))

    def call_price(samples, k):
        return disc * float(np.mean(np.maximum(samples - k, 0.0)))

    jump_prices = [call_price(jump_avg, k) for k in strikes]
    gauss_prices = [call_price(gauss_avg, k) for k in strikes]
    low_prices = [call_price(low_avg, k) for k in strikes]
    rel_uplift = {str(int(k)): round((jump_prices[i] / gauss_prices[i] - 1.0) * 100.0, 2)
                  for i, k in enumerate(strikes) if gauss_prices[i] > 1e-9}
    underprice_low = {str(int(k)): round((1.0 - low_prices[i] / jump_prices[i]) * 100.0, 2)
                      for i, k in enumerate(strikes) if jump_prices[i] > 1e-9}

    # high-renewable scenario: negative prices and one-day-ahead put at zero strike
    hi_rng = np.random.default_rng(SEED + 7)
    hi = simulate(hi_rng, 4000, wind_prob=0.40, wind_shift=45.0)
    hi_neg = float(np.mean(hi < 0.0))
    hi_mean = float(np.mean(hi))
    hi_sd = float(np.std(hi, ddof=1))
    true_zero_put = float(np.mean(np.maximum(-hi, 0.0))) * np.exp(-RATE / 365.0)
    pos = hi[hi > 0.0]
    mu_ln = float(np.mean(np.log(pos)))
    sd_ln = float(np.std(np.log(pos), ddof=1))
    lognormal_zero_put = 0.0
    lognormal_call10 = np.exp(-RATE / 365.0) * (
        np.exp(mu_ln + 0.5 * sd_ln ** 2) * norm.cdf((mu_ln + sd_ln ** 2 - np.log(10.0)) / sd_ln)
        - 10.0 * norm.cdf((mu_ln - np.log(10.0)) / sd_ln))
    ln_sd_mom = np.sqrt(np.log(1.0 + hi_sd ** 2 / hi_mean ** 2))
    ln_mu_mom = np.log(hi_mean) - 0.5 * ln_sd_mom ** 2
    lognormal_mom_call10 = np.exp(-RATE / 365.0) * (
        np.exp(ln_mu_mom + 0.5 * ln_sd_mom ** 2) * norm.cdf((ln_mu_mom + ln_sd_mom ** 2 - np.log(10.0)) / ln_sd_mom)
        - 10.0 * norm.cdf((ln_mu_mom - np.log(10.0)) / ln_sd_mom))
    true_call10 = float(np.mean(np.maximum(hi - 10.0, 0.0))) * np.exp(-RATE / 365.0)

    # Figure 1: sample path with jumps and wind-driven negative periods
    fig, ax = plt.subplots(figsize=(11, 4))
    show = slice(0, 420)
    idx = np.arange(*show.indices(N_DAYS))
    ax.plot(idx, hist[show], lw=0.8, color="#3b5b8c", label="day-ahead price")
    ax.plot(idx, seasonal(idx), lw=1.0, color="#555555", ls="--", label="seasonal mean")
    ax.axhline(0.0, color="#c0392b", lw=0.8)
    neg_idx = idx[hwind[show]]
    ax.scatter(neg_idx, hist[show][hwind[show]], s=6, color="#4aa3a2", label="wind state")
    jup = idx[hjump[show] > 15]
    ax.scatter(jup, hist[show][hjump[show] > 15], s=18, marker="^", color="#e07b39", label="upward jump")
    ax.set_title("Simulated day-ahead electricity price")
    ax.set_xlabel("day")
    ax.set_ylabel("price (EUR/MWh)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig1_price_path.png"), dpi=150)
    plt.close(fig)

    # Figure 2: tail versus Gaussian calibration
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    xgrid = np.linspace(min(-40.0, hist.min()), max(120.0, hist.max()), 240)
    axes[0].hist(hist, bins=90, density=True, color="#8fb3d9", alpha=0.85)
    axes[0].plot(xgrid, norm.pdf(xgrid, hist_mean, hist_sd), color="#c0392b", lw=1.4, label="fitted normal")
    axes[0].set_xlim(-40, 140)
    axes[0].set_title("Price distribution and normal fit")
    axes[0].set_xlabel("price")
    axes[0].legend()
    probs = np.array([0.90, 0.95, 0.99, 0.995, 0.999])
    emp_q = np.quantile(hist, probs)
    nor_q = norm.ppf(probs, hist_mean, hist_sd)
    axes[1].plot(nor_q, emp_q, "o", color="#3b5b8c")
    axes[1].plot([nor_q.min(), nor_q.max()], [nor_q.min(), nor_q.max()], "--", color="#888888")
    axes[1].set_xlabel("normal quantile")
    axes[1].set_ylabel("empirical quantile")
    axes[1].set_title("Tail QQ: spikes above the normal line")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig2_tail_distribution.png"), dpi=150)
    plt.close(fig)

    # Figure 3: peak-call price curves
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(strikes, jump_prices, "o-", color="#c0392b", label="jump model")
    axes[0].plot(strikes, gauss_prices, "s-", color="#3b5b8c", label="normal model")
    axes[0].plot(strikes, low_prices, "^--", color="#7a8b8b", label="underestimated intensity")
    axes[0].set_title("22-day average-price call")
    axes[0].set_xlabel("strike")
    axes[0].set_ylabel("discounted price")
    axes[0].legend()
    axes[1].bar([str(int(k)) for k in strikes], [jump_prices[i] - gauss_prices[i] for i in range(len(strikes))],
                color="#e07b39")
    axes[1].set_title("Normal-model underpricing")
    axes[1].set_xlabel("strike")
    axes[1].set_ylabel("price gap")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig3_peak_call_prices.png"), dpi=150)
    plt.close(fig)

    # Figure 4: negative prices under high renewables
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].hist(hi, bins=80, density=True, color="#4aa3a2", alpha=0.85)
    axes[0].axvline(0.0, color="#c0392b", ls="--", lw=1.2)
    axes[0].set_title("High-renewable price distribution")
    axes[0].set_xlabel("price")
    axes[0].set_ylabel("density")
    axes[1].bar(["zero-strike put", "10-strike call"],
                [true_zero_put, true_call10], color="#3b5b8c", label="arithmetic model")
    axes[1].bar(["zero-strike put", "10-strike call"],
                [lognormal_zero_put, lognormal_call10], color="#c0392b", alpha=0.7, label="lognormal on positives")
    axes[1].bar(["", ""], [0.0, lognormal_mom_call10], color="#e0a33e", alpha=0.8, label="lognormal on moments")
    axes[1].set_title("One-day options under negative prices")
    axes[1].set_ylabel("price")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig4_negative_price.png"), dpi=150)
    plt.close(fig)

    # Figure 5: jump-detection threshold and intensity error
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    thrs = [12.0, 18.0, 25.0]
    axes[0].plot(thrs, [det_rates[str(t)] for t in thrs], "o-", color="#8e6cbb")
    axes[0].axhline(avg_jump_p, color="#c0392b", ls="--", label="true mean intensity")
    axes[0].set_title("Detected jump rate versus threshold")
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("rate")
    axes[0].legend()
    uplift_ks = [k for k in strikes if str(int(k)) in rel_uplift]
    axes[1].plot(uplift_ks, [rel_uplift[str(int(k))] for k in uplift_ks], "o-", color="#c0392b", label="normal model")
    low_ks = [k for k in strikes if str(int(k)) in underprice_low]
    axes[1].plot(low_ks, [underprice_low[str(int(k))] for k in low_ks], "s-", color="#7a8b8b", label="low intensity")
    axes[1].legend(fontsize=8)
    axes[1].axhline(0.0, color="#888888", lw=0.8)
    axes[1].set_title("Pricing error from misestimated intensity")
    axes[1].set_xlabel("strike")
    axes[1].set_ylabel("percent")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig5_intensity_sensitivity.png"), dpi=150)
    plt.close(fig)

    out = {
        "seed": SEED,
        "n_days": N_DAYS,
        "hist_mean": round(hist_mean, 6),
        "hist_sd": round(hist_sd, 6),
        "negative_price_share": round(neg_share, 6),
        "emp_q995": round(q995, 6),
        "emp_q999": round(q999, 6),
        "normal_q995": round(gauss_q995, 6),
        "normal_q999": round(gauss_q999, 6),
        "robust_body_sd": round(mad_sd, 6),
        "robust_q999": round(robust_q999, 6),
        "tail_ratio_q999": round(tail_ratio, 6),
        "tail_ratio_robust_q999": round(q999 / robust_q999, 6),
        "stationary_stress_prob": round(pi_stress, 6),
        "mean_jump_intensity": round(avg_jump_p, 6),
        "detected_jump_counts": det_counts,
        "detected_jump_rates": det_rates,
        "strikes": strikes,
        "jump_call_prices": [round(v, 6) for v in jump_prices],
        "normal_call_prices": [round(v, 6) for v in gauss_prices],
        "low_intensity_call_prices": [round(v, 6) for v in low_prices],
        "normal_underpricing_pct": rel_uplift,
        "low_intensity_underpricing_pct": underprice_low,
        "high_renewable_negative_share": round(hi_neg, 6),
        "high_renewable_mean": round(hi_mean, 6),
        "high_renewable_sd": round(hi_sd, 6),
        "true_zero_strike_put": round(true_zero_put, 6),
        "lognormal_zero_strike_put": round(lognormal_zero_put, 6),
        "true_10_strike_call": round(true_call10, 6),
        "lognormal_10_strike_call": round(lognormal_call10, 6),
        "lognormal_moment_10_strike_call": round(lognormal_mom_call10, 6),
        "n_paths": N_PATHS,
        "horizon_days": HORIZON,
    }
    with open(os.path.join(IMG, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
```

