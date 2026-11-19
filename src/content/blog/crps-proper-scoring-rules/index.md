---
title: "CRPS 与 Log Score 到底选谁：三种评分规则都诚实，但一个极端值就让 Log Score 多赔 5 倍"
description: "给概率预测打分，CRPS、Log Score、Brier 都是严格适当规则——报告真分布时期望损失最小，30 万次蒙特卡洛把三条曲线的谷底钉在 mu=0、sigma=1，CRPS 闭式与定义积分对拍最大偏差 1.1e-8。但适当不等于稳健：观测从 1 个标准差走到 10 个，Log Score 的惩罚暴涨 49.5（二次），CRPS 只涨 8.83（线性）；100 个干净样本里混入 1 个 y=10，Log Score 的样本均值被抬高 0.491、单点独占 27.5% 的总损失，CRPS 只抬 0.088、占 15.5%。最实用的反转在最优宽度上：真分布是肥尾 t(3) 时，Log Score 要你把高斯预测放宽到 sigma=1.70 才满意，CRPS 认为 1.25 最好，两者差 0.45——同一批预测，换把尺子最优参数就变。样本量扫描显示 n=50 时谁都分不清两个接近模型、n=200 才都能分。附完整可运行代码与五张真实计算图。"
publishDate: '2026-11-19T01:20:00Z'
tags:
  - 量化
  - 交易
  - 概率预测
  - 风险度量
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

做收益分布预测、VaR、概率方向判断，最后都要回答一个问题：**这个概率报得准不准？** 评分规则（scoring rule）就是把"概率预测 + 实际结果"映射成一个损失的函数。CRPS、Log Score、Brier 是三项主流选择，教科书会告诉你它们都是**严格适当**（strictly proper）的——即诚实报告你真正相信的分布，期望损失最小。

这句话没错，但远不够。本文用真值已知的蒙特卡洛把三件事拆开，结论分三层：

1. **适当性确实成立**。30 万次正态样本上，Log Score、CRPS、Brier 三条期望损失曲线的谷底精确落在真参数 $\mu=0$、$\sigma=1$；CRPS 的闭式公式与定义积分逐点对拍，最大偏差 $1.1\times10^{-8}$。
2. **适当不等于稳健**。观测从 1 个标准差走到 10 个标准差，Log Score 的惩罚暴涨 49.5（二次增长），CRPS 只涨 8.83（近似线性）。在 100 个干净样本里混入 1 个 $y=10$，Log Score 的样本均值被抬高 0.491，这一个点独占 27.5% 的总损失；CRPS 只抬 0.088，占 15.5%。**用 Log Score 评一只偶尔跳空的股票，你的排名可能被一次极端行情完全改写。**
3. **换把尺子，最优参数就变**。真分布是肥尾 $t(3)$ 时，我们扫描高斯预测的宽度 $\sigma$：Log Score 要放宽到 $\sigma=1.70$ 才满意，CRPS 认为 $\sigma=1.25$ 最好，差 0.45。这不是数值噪声，是两种规则对尾部赋予的权重本质不同。

## 二、什么叫"适当"，为什么要先验证它

设预测分布为 $F$、实际观测为 $y$、评分规则 $S(F,y)$ 给出损失（越小越好）。规则是**适当的**，当且仅当对任意真分布 $G$，期望损失 $E_{y\sim G}[S(F,y)]$ 在 $F=G$ 时取最小；**严格适当**还要求这是唯一最小点。这条性质保证了"说真话"是预测者的占优策略——你不会通过故意把概率报偏来骗到更低损失。

三项规则的定义：

- **Log Score**：$S_{\log}(F,y)=-\log f(y)$，$f$ 是 $F$ 的密度。只对观测点处的密度打分。
- **CRPS**（连续 ranked 概率积分）：$S_{\mathrm{CRPS}}(F,y)=\int_{-\infty}^{\infty}\big(F(x)-\mathbf 1\{y\le x\}\big)^2\,dx$。比较整条分布函数，对正态有闭式 $S=\sigma\big[z(2\Phi(z)-1)+2\phi(z)-1/\sqrt\pi\big]$，$z=(y-\mu)/\sigma$。
- **Brier**（针对事件）：对事件 $\{y\le 0\}$ 报概率 $p$，损失 $(p-\mathbf 1\{y\le0\})^2$。

先验证严格适当性。真分布取 $N(0,1)$：固定 $\sigma=1$ 扫描报告均值 $\mu\in[-0.5,0.5]$，再把 $\mu=0$ 固定扫描 $\sigma\in[0.6,1.6]$。

![图1：三种损失曲线的谷底都在真参数处](/images/crps-proper-scoring-rules/fig1_properness.png)

结果：三条曲线的argmin 全在 $\mu=0$、两种损失的argmin 全在 $\sigma=1$（离散网格上的最近点），事件的真实发生率 0.5005 也和理论 0.5 吻合。同时把 CRPS 闭式与按定义做的数值积分在 8 个观测点上对拍，最大偏差 $1.1\times10^{-8}$——闭式实现可信，后面所有 CRPS 数字都用它。

## 三、尾部：二次惩罚与线性惩罚的分水岭

适当性讲的是"平均而言说实话最优"，却没说"错的时候要赔多少"。把观测偏离从 $z=0$ 推到 $z=10$（预测固定 $N(0,1)$），两种规则的惩罚曲线彻底分道扬镳。

![图2：CRPS 闭式对拍与尾部惩罚形状](/images/crps-proper-scoring-rules/fig2_crps_tail.png)

Log Score 是 $-\log\phi(z)=\tfrac12 z^2+\tfrac12\log(2\pi)$，二次增长：$z=1$ 处约 0.919，$z=10$ 处高达 50.92，从 1 到 10 涨了 49.5。CRPS 在 $z$ 大时近似 $z-1/\sqrt\pi$，线性增长：$z=10$ 处只有 9.44，同期涨幅 8.83。差了 5.6 倍。

这个差异在实盘里会被单个观测放大。取 100 个 $N(0,1)$ 干净样本，再混入一个 $y=10$：

![图3：一个极端点把 Log Score 平均拉高 0.491，CRPS 只抬 0.088](/images/crps-proper-scoring-rules/fig3_outlier_dominance.png)

Log Score 的样本均值从 1.340 抬到 1.831，涨 0.491；那一个点单独贡献 50.92，占 101 点总损失的 27.5%——**1% 的样本掌握了超过四分之一的裁决权**。CRPS 的均值只从 0.514 抬到 0.602（涨 0.088），单点占比 15.5%。所以在厚尾资产上，Log Score 更像一场"谁最不怕极端值"的竞赛，CRPS 更接近"整体分布谁更像真的"。

## 四、关键反转：最优预测宽度差了 0.45

如果说尾部敏感性只是"打分波动大"，下面这个实验才是真正影响建模决策的。真分布取自由度 3 的 Student-t（方差 3，典型肥尾），我们只用高斯 $N(0,\sigma)$ 去近似——扫描 $\sigma$，看两种规则各自认为多宽最好。

![图4：真分布 t(3) 下 Log Score 最优 sigma=1.70，CRPS 最优 1.25](/images/crps-proper-scoring-rules/fig4_optimal_width.png)

Log Score 的期望损失在 $\sigma=1.70$ 取最小，CRPS 在 $\sigma=1.25$ 取最小，相差 0.45（约 36%）。原因正是上一节的尾部权重：$t(3)$ 偶尔抛出远离中心的观测，Log Score 对这些点施以二次惩罚，逼你把预测分布撑宽来"接住"它们；CRPS 线性看待偏离，觉得没必要为稀有尾部牺牲中心的贴合度。

这不纯是学术趣味。你用 Log Score 做验证选出的波动率参数，会系统性地比用 CRPS 选的更宽、更保守。两套流程都会自称"严格适当"，却在同一个测试集上给出不同的最优模型。**选评分规则的那一刻，你就已经偷偷选了对风险的态度。**

## 五、样本量：多快能分辨两个接近的预测

最后一个实操问题：两个很接近的预测 $N(0,1)$ 和 $N(0,1.15)$，要多少样本才分得出高低？用配对得分差的 t 统计量衡量（同一批样本上算两个模型得分的差，除以差的标准误）。

![图5：区分两个接近预测，两种规则各自需要的样本](/images/crps-proper-scoring-rules/fig5_discriminability.png)

$n=50$ 时两种规则的 t 都不到 2（CRPS 甚至是负的 $-0.16$），谁都分不清；$n=200$ 起都越过 2，进入可区分区间；到 $n=3200$，Log Score 的 t 升到 5.78，CRPS 是 4.50。大样本下 Log Score 的区分力更强——它用二次惩罚把模型间的差异放大了。但请把它和第三节连起来读：**正是同一个二次项，既给了 Log Score 更高的区分度，也让它对单个极端值脆弱。** 这是同一枚硬币的两面，不存在只要优点不要代价的评分规则。

## 六、怎么选：三条边界

落到实盘，我的取舍是：**看排序、做模型选择时，CRPS 更稳**——它不会被一次跳空改写结论，且给出更贴合主体的预测宽度；**看密度拟合精度、样本干净且量大时，Log Score information 更足**，区分度更高。Brier 只在二分类事件（涨/跌、破/不破阈值）上天然适用，别拿它评连续分布。

三条必须写明的边界。第一，本文的正态 CRPS 闭式只对高斯预测成立；换成 $t$ 分布或经验分布，要么用数值积分，要么用积分表示 $S=\tfrac12E|X-y|-E|X-X'|$ 的样本近似。第二，"适当"是相对真分布属于同一可测类而言的；预测族被人为限制（比如只允许高斯）时，两种规则给出不同的"最优投影"，第四节就是活例。第三，$n=50$ 时出现的负 t 是抽样噪声，不应解读成"CRPS 偏好错误模型"，只是在小样本上差异还没从噪声里浮出来。

代码把适当性验证、对拍、尾部扫描、最优宽度与样本量实验全部放在一起，改参数即可复算。

```python
# -*- coding: utf-8 -*-
"""严格适当评分规则：CRPS / Log Score / Brier 的适当性、尾部敏感与最优宽度分歧。

本文全部数字的唯一来源。运行本脚本会：
1. 在真值已知的正态分布上验证 log score、CRPS、Brier 都是严格适当规则
   （报告真分布时期望损失最小），并做 CRPS 闭式与数值积分的逐点对拍；
2. 展示单个极端观测如何主导样本平均：log score 二次增长 vs CRPS 线性增长；
3. 关键反转：真分布为肥尾 t(3) 时，扫描高斯预测的宽度 sigma，
   log score 与 CRPS 给出不同的最优 sigma；
4. 样本量扫描：区分两个接近预测所需的样本，两种规则的区分度对比；
5. 画 5 张图到 public/images/crps-proper-scoring-rules/，
   全部统计量写入同目录 results.json。
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261119
N_MC = 300_000       # 期望得分的蒙特卡洛样本数
SQRT_PI = math.sqrt(math.pi)

_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "crps-proper-scoring-rules",
)
os.makedirs(_OUT, exist_ok=True)


def log_score(y, mu, sigma):
    """负对数密度 -log f(y)，越小越好。"""
    return -stats.norm.logpdf(y, loc=mu, scale=sigma)


def crps_normal(y, mu, sigma):
    """正态 CRPS 闭式：sigma·[z(2Phi(z)-1)+2phi(z)-1/sqrt(pi)]，z=(y-mu)/sigma。"""
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / SQRT_PI)


def crps_numeric(y, mu, sigma):
    """CRPS 定义 integral (F(x)-1[y<=x])^2 dx 的数值积分，用于独立对拍。"""
    def integrand(x):
        F = stats.norm.cdf(x, loc=mu, scale=sigma)
        return (F - (1.0 if y <= x else 0.0)) ** 2
    lo, hi = min(mu, y) - 12 * sigma, max(mu, y) + 12 * sigma
    val, _ = integrate.quad(integrand, lo, hi, limit=200)
    return val


def brier_event(y, p, threshold=0.0):
    """事件 {y<=threshold} 的 Brier 损失 (p-1[event])^2，y 可为标量或数组。"""
    y = np.asarray(y)
    return (p - (y <= threshold).astype(float)) ** 2


def properness_scan():
    """真分布 N(0,1)：扫描报告 mu（固定 sigma=1）与 sigma（固定 mu=0），验证期望损失在真值处最小。"""
    rng = np.random.default_rng(SEED)
    y = rng.standard_normal(N_MC)
    p_true = float(np.mean(y <= 0.0))
    mus = np.linspace(-0.5, 0.5, 21)
    ls_mu, crps_mu, brier_mu = [], [], []
    for mu in mus:
        ls_mu.append(float(np.mean(log_score(y, mu, 1.0))))
        crps_mu.append(float(np.mean(crps_normal(y, mu, 1.0))))
        brier_mu.append(float(np.mean(brier_event(y, stats.norm.cdf(-mu)))))  # P(Y<=0)=Phi(-mu)
    sigmas = np.linspace(0.6, 1.6, 21)
    ls_sig, crps_sig = [], []
    for s in sigmas:
        ls_sig.append(float(np.mean(log_score(y, 0.0, s))))
        crps_sig.append(float(np.mean(crps_normal(y, 0.0, s))))
    arg_mu = lambda arr: float(mus[int(np.argmin(arr))])
    return {
        "p_true_event": p_true,
        "mus": mus.tolist(), "ls_mu": ls_mu, "crps_mu": crps_mu, "brier_mu": brier_mu,
        "sigmas": sigmas.tolist(), "ls_sig": ls_sig, "crps_sig": crps_sig,
        "argmin_mu_ls": arg_mu(ls_mu), "argmin_mu_crps": arg_mu(crps_mu),
        "argmin_mu_brier": arg_mu(brier_mu),
        "argmin_sigma_ls": float(sigmas[int(np.argmin(ls_sig))]),
        "argmin_sigma_crps": float(sigmas[int(np.argmin(crps_sig))]),
    }


def crps_formula_check():
    """CRPS 闭式 vs 数值积分对拍，并给出尾部惩罚 z=0..10 的 log/CRPS 增长。"""
    pts = [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    closed = [crps_normal(y, 0.0, 1.0) for y in pts]
    numeric = [crps_numeric(y, 0.0, 1.0) for y in pts]
    max_dev = max(abs(a - b) for a, b in zip(closed, numeric))
    zs = np.linspace(0, 10, 11)
    ls_tail = [float(log_score(z, 0.0, 1.0)) for z in zs]
    crps_tail = [float(crps_normal(z, 0.0, 1.0)) for z in zs]
    return {
        "points": pts, "crps_closed": closed, "crps_numeric": numeric,
        "max_dev": float(max_dev),
        "z": zs.tolist(), "ls_tail": ls_tail, "crps_tail": crps_tail,
        "ls_at_10": ls_tail[-1], "crps_at_10": crps_tail[-1],
        "ls_growth_1_to_10": ls_tail[-1] - ls_tail[1],
        "crps_growth_1_to_10": crps_tail[-1] - crps_tail[1],
    }


def outlier_dominance():
    """100 个 N(0,1) 干净样本中混入 1 个 y=10，看平均得分被单点主导的程度。"""
    rng = np.random.default_rng(SEED + 1)
    clean = rng.standard_normal(100)
    ls_clean = float(np.mean(log_score(clean, 0.0, 1.0)))
    crps_clean = float(np.mean(crps_normal(clean, 0.0, 1.0)))
    dirty = np.append(clean, 10.0)
    ls_dirty = float(np.mean(log_score(dirty, 0.0, 1.0)))
    crps_dirty = float(np.mean(crps_normal(dirty, 0.0, 1.0)))
    # 单个点本身对总和的贡献，占全部 101 点平均的比重
    ls_single = float(log_score(10.0, 0.0, 1.0))
    crps_single = float(crps_normal(10.0, 0.0, 1.0))
    return {
        "n_clean": 100, "outlier": 10.0,
        "ls_clean": ls_clean, "ls_dirty": ls_dirty, "ls_rise": ls_dirty - ls_clean,
        "crps_clean": crps_clean, "crps_dirty": crps_dirty, "crps_rise": crps_dirty - crps_clean,
        "ls_single": ls_single, "crps_single": crps_single,
        "ls_single_share": ls_single / (ls_dirty * 101),
        "crps_single_share": crps_single / (crps_dirty * 101),
    }


def optimal_width_divergence():
    """关键反转：真分布 t(3)，扫描高斯预测 sigma，两种规则给出不同最优宽度。"""
    rng = np.random.default_rng(SEED + 2)
    y = stats.t.rvs(df=3, size=N_MC, random_state=rng)  # 方差 = 3
    sigmas = np.linspace(0.8, 3.0, 45)
    ls_curve = [float(np.mean(log_score(y, 0.0, s))) for s in sigmas]
    crps_curve = [float(np.mean(crps_normal(y, 0.0, s))) for s in sigmas]
    i_ls = int(np.argmin(ls_curve))
    i_crps = int(np.argmin(crps_curve))
    return {
        "true_df": 3, "true_var": 3.0, "n_mc": N_MC,
        "sigmas": sigmas.tolist(), "ls_curve": ls_curve, "crps_curve": crps_curve,
        "ls_opt_sigma": float(sigmas[i_ls]), "crps_opt_sigma": float(sigmas[i_crps]),
        "ls_opt_loss": ls_curve[i_ls], "crps_opt_loss": crps_curve[i_crps],
        "sigma_gap": float(sigmas[i_ls] - sigmas[i_crps]),
    }


def discriminability():
    """样本量扫描：真分布 N(0,1)，区分模型 N(0,1) 与 N(0,1.15) 的配对得分差 t 统计。"""
    rng = np.random.default_rng(SEED + 3)
    rows = []
    for n in [50, 200, 800, 3200]:
        y = rng.standard_normal(n)
        d_ls = log_score(y, 0.0, 1.15) - log_score(y, 0.0, 1.0)
        d_crps = crps_normal(y, 0.0, 1.15) - crps_normal(y, 0.0, 1.0)
        t_ls = float(np.mean(d_ls) / (np.std(d_ls, ddof=1) / math.sqrt(n)))
        t_crps = float(np.mean(d_crps) / (np.std(d_crps, ddof=1) / math.sqrt(n)))
        rows.append({"n": n, "t_ls": t_ls, "t_crps": t_crps,
                     "mean_d_ls": float(np.mean(d_ls)), "mean_d_crps": float(np.mean(d_crps)),
                     "sd_d_ls": float(np.std(d_ls, ddof=1)), "sd_d_crps": float(np.std(d_crps, ddof=1))})
    return rows


def _save(fig, name):
    path = os.path.join(_OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot1(pr):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(pr["mus"], pr["ls_mu"], "o-", ms=4, color="#c0392b", label="Log Score")
    ax.plot(pr["mus"], pr["crps_mu"], "s-", ms=4, color="#2980b9", label="CRPS")
    ax.plot(pr["mus"], pr["brier_mu"], "^-", ms=4, color="#27ae60", label="Brier (Y≤0)")
    ax.axvline(0, color="k", ls="--", lw=1.2)
    ax.set_xlabel("报告均值 mu（真值 0）")
    ax.set_ylabel("期望损失")
    ax.set_title("适当性：三种损失都在 mu=0 取最小")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(pr["sigmas"], pr["ls_sig"], "o-", ms=4, color="#c0392b", label="Log Score")
    ax.plot(pr["sigmas"], pr["crps_sig"], "s-", ms=4, color="#2980b9", label="CRPS")
    ax.axvline(1, color="k", ls="--", lw=1.2)
    ax.set_xlabel("报告标准差 sigma（真值 1）")
    ax.set_ylabel("期望损失")
    ax.set_title("适当性：两种损失都在 sigma=1 取最小")
    ax.legend(fontsize=8)
    fig.suptitle(f"图1：真分布 N(0,1) 上验证严格适当性（{N_MC} 次蒙特卡洛）", fontsize=11)
    return _save(fig, "fig1_properness.png")


def plot2(chk):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(chk["points"], chk["crps_closed"], "o", ms=6, color="#2980b9", label="闭式公式")
    ax.plot(chk["points"], chk["crps_numeric"], "x", ms=7, color="#c0392b", label="数值积分")
    ax.set_xlabel("观测 y（预测 N(0,1)）")
    ax.set_ylabel("CRPS")
    ax.set_title(f"CRPS 闭式 vs 定义积分，最大偏差 {chk['max_dev']:.2e}")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(chk["z"], chk["ls_tail"], "o-", color="#c0392b", label="Log Score（二次）")
    ax.plot(chk["z"], chk["crps_tail"], "s-", color="#2980b9", label="CRPS（近似线性）")
    ax.set_xlabel("观测偏离 z（预测 N(0,1)）")
    ax.set_ylabel("损失")
    ax.set_title(f"尾部惩罚：z=1→10，Log 增 {chk['ls_growth_1_to_10']:.2f}，CRPS 增 {chk['crps_growth_1_to_10']:.2f}")
    ax.legend(fontsize=8)
    fig.suptitle("图2：CRPS 公式对拍与尾部惩罚形状", fontsize=11)
    return _save(fig, "fig2_crps_tail.png")


def plot3(od):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    names = ["Log Score\n干净", "Log Score\n混入1个y=10", "CRPS\n干净", "CRPS\n混入1个y=10"]
    vals = [od["ls_clean"], od["ls_dirty"], od["crps_clean"], od["crps_dirty"]]
    colors = ["#e57373", "#c0392b", "#85c1e9", "#2980b9"]
    bars = ax.bar(names, vals, 0.55, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("样本平均损失（101 点）")
    ax.set_title(f"图3：一个极端点把 Log Score 平均拉高 {od['ls_rise']:.3f}，CRPS 只抬 {od['crps_rise']:.3f}")
    return _save(fig, "fig3_outlier_dominance.png")


def plot4(wd):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    s = wd["sigmas"]
    ax.plot(s, wd["ls_curve"], "-", color="#c0392b", lw=2, label="Log Score 期望损失")
    ax.plot(s, wd["crps_curve"], "-", color="#2980b9", lw=2, label="CRPS 期望损失")
    ax.axvline(wd["ls_opt_sigma"], color="#c0392b", ls="--", lw=1.4)
    ax.axvline(wd["crps_opt_sigma"], color="#2980b9", ls="--", lw=1.4)
    ax.plot([wd["ls_opt_sigma"]], [wd["ls_opt_loss"]], "o", color="#c0392b", ms=8)
    ax.plot([wd["crps_opt_sigma"]], [wd["crps_opt_loss"]], "s", color="#2980b9", ms=8)
    ax.set_xlabel("高斯预测标准差 sigma")
    ax.set_ylabel("期望损失（各规则纵轴独立）")
    ax.set_title(f"图4：真分布 t(3) 下 Log Score 最优 sigma={wd['ls_opt_sigma']:.2f}，CRPS 最优 {wd['crps_opt_sigma']:.2f}（差 {wd['sigma_gap']:.2f}）")
    ax.legend(fontsize=9)
    return _save(fig, "fig4_optimal_width.png")


def plot5(disc):
    ns = [r["n"] for r in disc]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(ns, [r["t_ls"] for r in disc], "o-", color="#c0392b", lw=2, label="Log Score 的 t 统计")
    ax.plot(ns, [r["t_crps"] for r in disc], "s-", color="#2980b9", lw=2, label="CRPS 的 t 统计")
    ax.axhline(2.0, color="k", ls="--", lw=1.2)
    ax.text(ns[0], 2.1, "t=2 约可区分", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("样本量 n")
    ax.set_ylabel("配对得分差的 t 统计量")
    ax.set_title("图5：区分 N(0,1) 与 N(0,1.15)，两种规则各自需要多少样本")
    ax.legend(fontsize=9)
    return _save(fig, "fig5_discriminability.png")


def main():
    pr = properness_scan()
    chk = crps_formula_check()
    od = outlier_dominance()
    wd = optimal_width_divergence()
    disc = discriminability()
    results = {
        "setup": {"seed": SEED, "n_mc": N_MC},
        "properness": pr, "crps_check": chk, "outlier": od,
        "optimal_width": wd, "discriminability": disc,
    }
    results["figs"] = [
        plot1(pr), plot2(chk), plot3(od), plot4(wd), plot5(disc),
    ]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()
```

