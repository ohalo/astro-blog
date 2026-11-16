---
title: "更新过程与检查悖论：你统计的平均挂单间隔，天生就高估 35%"
description: "在随机时刻做一次快照，你观察到的挂单间隔、持仓时长、故障间隔，全部系统性地偏长——这不是样本不够，是抽样机制本身在偏爱长区间。对数正态间隔（形状参数 0.55，均值归一为 1）的更新过程上，40 万个跨实现观察测得检查长度 1.3536 天，对理论值 E[L²]/E[L]=1.3532 分毫不差；CV² 扫描 8 个点里仿真与理论 1+CV² 最大偏离 0.027；残差寿命经验 CDF 对平衡分布的 K-S 偏离仅 0.0037。更反直觉的是：n 从 25 加到 10000，朴素均值的 35% 偏差纹丝不动，而调和均值校正把偏差压到 -0.04%。持仓快照实验里，全部订单平均持有 1.0001 天，在场截面却报 1.3505 天，超 3 天的长拿党占比从 1.16% 被放大到 4.33%。"
publishDate: '2026-11-16T06:40:00Z'
tags:
  - 量化
  - 交易
  - 随机过程
  - 统计
  - 可靠性
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

先给三个从真实仿真里跑出来的数字（对数正态间隔、均值归一为 1 天，形状参数 0.55，即 CV²=0.353）：

| 你统计的东西 | 快照告诉你 | 真相 | 被放大 |
|---|---|---|---|
| 随机时刻撞上的挂单间隔 | 1.3531 天 | 0.9996 天 | 1.3536 倍 |
| 全部订单的平均持有期 | 在场截面 1.3505 天 | 全体 1.0001 天 | 1.3503 倍 |
| 持有超过 3 天的"长拿党"比例 | 快照里 4.33% | 全体 1.16% | 3.7 倍 |

放大倍数不是误差，是**结构性偏差**：样本从 25 个加到 10000 个，朴素均值的偏差稳定在 35% 上下，一点没有收敛的意思——因为偏差来自"你在什么时刻去看"，而不是"你看了多少个"。把观测样本取**调和均值**，偏差从 n=25 时的 2.57% 一路收敛到 n=10000 时的 -0.04%。这套机制的数学名字叫**检查悖论**（inspection paradox），属更新过程理论，1970 年代就被 Cox 讲透了，但做微观结构统计、因子数据清洗的人至今还在踩。

## 二、为什么随机快照必然偏爱长区间

想象挂单间隔是独立同分布的随机变量序列 $L_1, L_2, \ldots$，你在一个与序列无关的随机时刻 $t$ 探头看了一眼。你看到的不是"下一个间隔"，而是**把 $t$ 夹在中间的那个间隔**。一个长度为 5 天的间隔被看到的概率，是 1 天间隔的 5 倍——区间越长，它覆盖随机观察点的"地盘"越大。数学上说，观测值的分布是原分布的**长度偏差变换**：

$$ f^*(x) = \frac{x\, f(x)}{\mathbb{E}[L]}, \qquad \mathbb{E}[L^*] = \frac{\mathbb{E}[L^2]}{\mathbb{E}[L]} = \mathbb{E}[L]\,(1+\mathrm{CV}^2). $$

这就是经典的"公交车悖论"：公交平均 10 分钟一班，你在随机时刻到站，感受到的间隔期望却远大于 10 分钟。同一枚硬币还解释了：为什么学生总觉得"同学平均朋友圈数"超过自己（朋友数量抽样偏差）、为什么抽查时刻还"健在"的疾病病程总显得更长（存活时间偏倚）、为什么宕机调查总把平均故障间隔说得比运维日志里的长。

对交易而言，触发场景无处不在：随机挑一天截一张订单流快照去估计"平均挂单存活期"；月底拉一张在场持仓表统计"散户平均持有多久"；巡检时刻记录"系统平均恢复时长"。只要观察机制是"随机时刻看正在发生的事"，你采到的每个样本都自动带上了 $x f(x)/\mathbb{E}[L]$ 的权重。

## 三、主实验：观察到的间隔分布长这样

仿真设定：$L=\exp(\sigma Z-\sigma^2/2)$（$Z$ 为标准正态），取 $\sigma=0.55$，均值精确归一为 1。为了拿到干净的遍历极限，每 400 个观察时刻使用一条独立的 2 万天长窗口更新过程（长窗口保证时间平均收敛，跨实现平均压掉单条轨迹的涨落），共 40 万个观察。

![原始间隔与观察区间的直方图对比，黑色虚线为理论密度 x f(x)/E[L]](/images/renewal-process-inspection-paradox/fig1_inspection_paradox.png)

图上红色柱是随机观察时刻撞上的区间长度：整个分布右移、右尾更肥。仿真均值 1.3531 对理论 $\mathbb{E}[L^2]/\mathbb{E}[L]=1.3532$，40 万样本下两者在小数点后三位一致。蓝色点线是原始间隔均值 0.9996。偏差 35.36% 恰好等于理论值 CV²=35.32%。

## 四、偏差多大，只看一件事：分散度

$\mathbb{E}[L^*]/\mathbb{E}[L]=1+\mathrm{CV}^2$ 给出一个非常实用的口算规则：偏差倍数只由变异系数决定。间隔越"规律"（CV 小），快照越诚实；间隔越"看运气"（CV 大），快照越离谱。把 $\sigma$ 从 0 扫到 1.1（对应 CV² 从 0 到 2.353），每个点 15 万个跨实现观察：

![CV² 扫描：仿真点贴合理论曲线 1+CV²](/images/renewal-process-inspection-paradox/fig2_cv_sweep.png)

八个仿真点全部贴合理论曲线，最大偏离 0.027（出现在 CV²=2.35 的重尾端，那里方差本身爆炸）。几个刻度值得记住：间隔基本规律（CV²=0.04）时快照只虚高 4%，可以不管；CV²=0.28 时虚高 29%——挂单、成交流这类数据大多在这个量级；CV²=0.9 时直接翻倍（1.898）；一旦间隔接近柯西式的重尾（CV²>2），"平均间隔"这个量对快照几乎失去意义，图上 CV²=2.35 处观测均值已达 3.38 倍。你的数据属于哪一档，取样本算一下 CV² 就知道。

## 五、不仅间隔被拉长，"还要等多久"也有自己的分布

检查悖论还有个孪生结论：观察时刻的**残差寿命**（从现在到下一次更新还需等的时长）不服从原分布，而服从**平衡分布**

$$ F_e(x) = \frac{1}{\mathbb{E}[L]} \int_0^x \big(1-F(u)\big)\, du, $$

其期望为 $\mathbb{E}[L^2]/(2\mathbb{E}[L])$——恰好是观察区间期望的一半。这解释了排队论里著名的"剩余寿命反直觉"：对指数分布（无记忆）残差均值等于平均间隔，但对一般分布它会更大。30 万个观察的残差样本对理论 CDF 作图：

![残差寿命经验 CDF 与平衡分布理论 CDF 几乎重合](/images/renewal-process-inspection-paradox/fig3_residual_life.png)

300 个格点上经验与理论 CDF 的最大偏离（K-S 统计量）只有 0.0037；仿真残差均值 0.6784、已老化时长均值 0.6769，两者之和≈1.355 正是被拉长的区间均值，且 age 与 resid 的经验分布几乎重合（对称性）。中位残差仿真 0.5236 对理论 0.5255。做补单时机、退役调度这类"还剩多久"预测时，训练标签应该是平衡分布下的样本，不是原始间隔分布。

## 六、最反直觉的一步：加样本修不掉偏差

既然偏差是结构性的，那么常见的"多采集点数据"完全无效。仿真直接演示这一点：目标是用快照估计真均值 $\mathbb{E}[L]=1$。朴素估计量就是观测样本的算术平均；校正估计量取观测样本的**调和均值** $n/\sum(1/L_i^*)$。它无偏的原理一行就能写完：size-biased 抽样下

$$ \mathbb{E}\!\left[\frac{1}{L^*}\right] = \mathbb{E}\!\left[L \cdot \frac{1}{L} \Big/ \mathbb{E}[L]\right] = \frac{1}{\mathbb{E}[L]}, $$

长区间被多抽的倍数 $L$，恰好被 $1/L$ 的原样抵消。每个样本量重复 300 次：

![样本量扫描：朴素均值偏差平在 35%，调和均值校正偏差趋于 0](/images/renewal-process-inspection-paradox/fig4_estimator_bias.png)

朴素均值的偏差从 n=25 的 35.67% 到 n=10000 的 35.26%，全程贴着理论线 CV²=35.32% 走——样本量对它是无效的；它的 RMSE 更是钉死在 35% 上下，因为 RMSE 里偏差分毫未减。调和均值这边：n=25 时偏差 2.57%、n=10000 时只剩 -0.04%，RMSE 从 13.05% 收敛到 0.59%。注意代价：n=25 时调和均值 RMSE 13% 小于朴素 39%，因为 35% 的偏差比什么都贵——无偏高方差的估计量，在偏差主导的区域反而完胜有偏低方差。这是偏差-方差权衡最干净的一个教学案例。

## 七、持仓快照：你看到的"散户平均持有期"是真实值的 1.35 倍

最后一个实验最贴近实盘。仿真 12 万张订单，持有期 $H$ 用同一个对数正态分布，开单时刻在 40 天窗口内均匀分布，然后在一个随机时刻拉"在场持仓表"。能进表的订单必须满足"已开单且未平掉"——这正是长度偏差的准入条件：

![持仓快照的截面均值与全体均值对比](/images/renewal-process-inspection-paradox/fig5_position_snapshot.png)

2955 张在场订单的截面平均持有 1.3505 天，而 12 万张订单的真实平均只有 1.0001 天，放大 1.3503 倍（理论 1.3532）。更刺眼的是尾部结构被改写：全体订单里持有超 3 天的仅 1.16%，快照里却占 4.33%，接近 3.7 倍。也就是说，用截面数据做"长拿党 vs 短线客"画像、用在场持仓表反推持有期分布、用巡检日志统计平均故障修复时长，做的都是同一件事——把长尾按长度重新加权。任何"从在场样本反推全体"的分析，要么用调和均值校正，要么明确声明自己在报告 size-biased 分布。

## 八、完整代码与复现

以下脚本是本文所有数字的唯一来源：五个实验各用固定种子，重跑一次 `python article_core_renewal.py` 即可复现全部数字与 5 张图，统计量落盘在同目录 `results.json`。

```python
# -*- coding: utf-8 -*-
"""更新过程检查悖论：随机时点观察到的"平均间隔"为什么系统性偏大。

本文全部数字的唯一来源。运行本脚本会：
1. 用对数正态间隔生成更新过程，蒙特卡洛测「检查悖论」——随机观察时刻所
   落入区间长度的期望 E[L*]，与理论 E[L*]=E[L^2]/E[L]=E[L](1+CV^2) 对拍；
2. 扫 CV：不同分散度下偏差倍数 1+CV^2 的仿真-理论曲线；
3. 残差寿命分布 vs 平衡分布理论 CDF 对拍（K-S 偏离）；
4. 固定日历采样估计平均挂单间隔：朴素均值偏差随样本量不收敛（≈CV²），
   而长度偏差的无偏校正——观测样本的调和均值——偏差收敛到 0；
5. 持仓快照的截面平均被长度偏差放大的倍数；
6. 画 5 张图到 public/images/renewal-process-inspection-paradox/；
7. 把全部统计量写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 202611154
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "renewal-process-inspection-paradox"
)
os.makedirs(_OUT, exist_ok=True)

SIG0 = 0.55  # 主实验用的对数正态形状参数


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


def lognormal_params(sigma):
    """L=exp(sigma*Z-mu0) 归一到均值 1：mu0=sigma^2/2。返回 (mu1, mu2, cv2)。"""
    mu0 = sigma ** 2 / 2.0
    mu1 = 1.0
    mu2 = np.exp(sigma ** 2)  # E[L^2]，均值已归一
    return mu1, mu2, mu2 - 1.0


def draw_L(rng, sigma, n):
    return np.exp(sigma * rng.standard_normal(n) - sigma ** 2 / 2.0)


def inspected_lengths(rng, sigma, n_obs, horizon=None):
    """跨实现观察：每 400 个观察时刻用一条独立的长窗口（2 万天）更新过程，

    长窗口保证时间平均遍历收敛，跨实现平均消除单条实现的随机涨落。
    返回观察时刻落入区间的 (长度, 起点, 已老化时长 age, 残差 resid)。
    """
    w = 20000.0
    o_L, o_S, o_A, o_R = [], [], [], []
    rem = n_obs
    while rem > 0:
        g = min(rem, 400)
        rem -= g
        L = draw_L(rng, sigma, int(2.5 * w) + 10)
        S = np.concatenate(([0.0], np.cumsum(L)))
        while S[-1] < 2.5 * w + 10.0:  # 覆盖保证：观测窗 [1.5w, 2.5w)
            L = np.concatenate([L, draw_L(rng, sigma, int(2.5 * w))])
            S = np.concatenate(([0.0], np.cumsum(L)))
        t = rng.uniform(1.5 * w, 2.5 * w, g)
        i = np.searchsorted(S, t, side="right") - 1
        o_L.append(L[i]); o_S.append(S[i] - 1.5 * w)
        o_A.append(t - S[i]); o_R.append(S[i + 1] - t)
    return (np.concatenate(o_L), np.concatenate(o_S),
            np.concatenate(o_A), np.concatenate(o_R))


def exp1_inspection(rng):
    """检查悖论主实验：观察区间长度 vs 原始间隔分布。"""
    mu1, mu2, cv2 = lognormal_params(SIG0)
    L_true = draw_L(rng, SIG0, 400_000)
    L_ins, _, _, _ = inspected_lengths(rng, SIG0, 400_000, 300.0)
    stat = {
        "sigma": SIG0,
        "E_L": float(L_true.mean()),
        "E_L_inspected": float(L_ins.mean()),
        "ratio_sim": float(L_ins.mean() / L_true.mean()),
        "ratio_theory": float(mu2 / mu1),  # = 1+CV^2
        "cv2_theory": float(cv2),
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bins = np.linspace(0, 5.0, 80)
    ax.hist(L_true, bins=bins, density=True, alpha=0.55, color="#2f6db3",
            label="原始间隔 L 的密度")
    ax.hist(L_ins, bins=bins, density=True, alpha=0.55, color="#c94f4f",
            label="随机时点观察到的区间 L*")
    xg = np.linspace(1e-3, 5, 400)
    dens = np.exp(-np.log(xg) ** 2 / (2 * SIG0 ** 2)) / (xg * SIG0 * np.sqrt(2 * np.pi))
    ax.plot(xg, xg * dens / mu1, "k--", lw=1.6,
            label=r"理论密度 $x f(x)/E[L]$")
    ax.axvline(L_true.mean(), color="#2f6db3", lw=1, ls=":")
    ax.axvline(L_ins.mean(), color="#c94f4f", lw=1, ls=":")
    ax.annotate(f"E[L]={L_true.mean():.3f}", (L_true.mean(), 1.6),
                xytext=(0.35, 1.6), color="#2f6db3")
    ax.annotate(f"E[L*]={L_ins.mean():.3f}", (L_ins.mean(), 1.2),
                xytext=(2.6, 1.2), color="#c94f4f")
    ax.set_xlabel("挂单间隔（天）")
    ax.set_ylabel("密度")
    ax.set_title(f"检查悖论：观察到的间隔被拉长 {stat['ratio_sim']:.3f} 倍"
                 f"（理论 1+CV²={stat['ratio_theory']:.3f}）")
    ax.legend()
    _save(fig, "fig1_inspection_paradox.png")
    return stat


def exp2_cv_sweep(rng):
    """扫 CV：偏差倍数 1+CV^2 的仿真点 vs 理论曲线。"""
    sigmas = np.array([0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1])
    rows = []
    for s in sigmas:
        _, mu2, cv2 = lognormal_params(s)
        if s == 0.0:
            r = 1.0
        else:
            L_ins, _, _, _ = inspected_lengths(rng, s, 150_000, 400.0)
            r = float(L_ins.mean())  # E[L]=1 归一
        rows.append({"sigma": float(s), "cv2": float(cv2),
                     "ratio_sim": r, "ratio_theory": float(mu2)})
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = [q["cv2"] for q in rows]
    ax.plot(xs, [q["ratio_theory"] for q in rows], "-", color="#2f6db3",
            lw=1.8, label="理论 1+CV²")
    ax.scatter(xs, [q["ratio_sim"] for q in rows], color="#c94f4f", zorder=5,
               label="蒙特卡洛仿真")
    for q in rows[1::2]:
        ax.annotate(f"CV²={q['cv2']:.2f}→{q['ratio_sim']:.2f}",
                    (q["cv2"], q["ratio_sim"]), textcoords="offset points",
                    xytext=(6, 7), fontsize=8)
    ax.set_xlabel("间隔分布的变异系数平方 CV²")
    ax.set_ylabel("观察均值 / 真值")
    ax.set_title("分散度越大，快照估计的偏差越大")
    ax.legend()
    _save(fig, "fig2_cv_sweep.png")
    return {"sweep": rows, "max_abs_dev": max(abs(q["ratio_sim"] - q["ratio_theory"])
                                              for q in rows)}


def eq_cdf(x, sigma):
    """平衡分布 CDF：F_e(x)=E[min(L,x)]/E[L]，L 为归一均值 1 的对数正态。"""
    x = np.asarray(x, dtype=float)
    from scipy.stats import norm

    # E[min(L,x)] = ∫0^x P(L>t)dt = x - E[(L-x)^+]；用数值积分更稳
    tt = np.linspace(0.0, x.max(), 20001)
    tail = np.empty_like(tt)
    pos = tt > 0
    tail[pos] = norm.sf(np.log(tt[pos]) / sigma + sigma / 2.0)
    tail[~pos] = 1.0
    integ = np.concatenate(([0.0], np.cumsum((tail[1:] + tail[:-1]) / 2 * np.diff(tt))))
    Fe = np.interp(x, tt, integ)  # E[L]=1
    return Fe


def exp3_residual(rng):
    """残差寿命 vs 平衡分布理论 CDF。"""
    _, _, L_start, resid = inspected_lengths(rng, SIG0, 300_000, 300.0)
    sr = np.sort(resid)
    xs = sr[np.linspace(0, sr.size - 1, 300).astype(int)]
    emp = (np.arange(1, xs.size + 1)) / xs.size  # xs 取自 sr 的等距格点
    theo = eq_cdf(xs, SIG0)
    ks = float(np.max(np.abs(emp - theo)))
    stat = {"sigma": SIG0, "n": int(resid.size), "ks_stat": ks,
            "mean_residual": float(resid.mean()),
            "mean_age": float(L_start.mean()),
            "half_quantile_theory": float(xs[np.argmax(theo >= 0.5)]),
            "half_quantile_sim": float(np.quantile(resid, 0.5))}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(xs, emp, "o", ms=4, color="#c94f4f", label="仿真残差寿命经验 CDF")
    ax.plot(xs, theo, "-", lw=1.8, color="#2f6db3", label="平衡分布理论 CDF")
    ax.set_xlabel("残差寿命（天）")
    ax.set_ylabel("累计概率")
    ax.set_title(f"残差寿命服从平衡分布（K-S 偏离 {ks:.4f}）")
    ax.legend()
    _save(fig, "fig3_residual_life.png")
    return stat


def exp4_estimator(rng):
    """固定日历采样估计平均挂单间隔：朴素均值 vs 1/(1+CV²) 校正。"""
    mu1, mu2, cv2 = lognormal_params(SIG0)
    ns = [25, 60, 150, 400, 1000, 3000, 10000]
    reps = 300
    rows = []
    for n in ns:
        naive, corr = [], []
        for _ in range(reps):
            L_ins, _, _, _ = inspected_lengths(rng, SIG0, n, 300.0)
            naive.append(L_ins.mean())
            # size-biased 样本的调和均值是 E[L] 的无偏估计：E[1/L*]=1/E[L]
            corr.append(n / np.sum(1.0 / L_ins))
        naive = np.array(naive)
        corr = np.array(corr)
        rows.append({"n": n, "naive_bias_pct": float((naive.mean() / mu1 - 1) * 100),
                     "corr_bias_pct": float((corr.mean() / mu1 - 1) * 100),
                     "naive_rmse_pct": float(np.sqrt(np.mean((naive / mu1 - 1) ** 2)) * 100),
                     "corr_rmse_pct": float(np.sqrt(np.mean((corr / mu1 - 1) ** 2)) * 100)})
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot([q["n"] for q in rows], [q["naive_bias_pct"] for q in rows],
            "o-", color="#c94f4f", label="朴素均值偏差（不收敛）")
    ax.plot([q["n"] for q in rows], [q["corr_bias_pct"] for q in rows],
            "s-", color="#2f6db3", label="调和均值校正偏差（→0）")
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.axhline(cv2 * 100, color="#c94f4f", lw=1, ls="--",
               label=f"理论偏差 CV²={cv2:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("快照样本量 n")
    ax.set_ylabel("对真值 E[L] 的偏差（%）")
    ax.set_title("样本再多也修不掉偏差：必须做长度偏差校正")
    ax.legend()
    _save(fig, "fig4_estimator_bias.png")
    return {"cv2_theory_pct": float(cv2 * 100), "sweep": rows}


def exp5_snapshot(rng):
    """持仓快照：截面上「平均持仓时长」被长度偏差放大的倍数。"""
    n_traders = 120_000
    H = draw_L(rng, SIG0, n_traders)  # 单位=天，均值≈1
    t_obs = rng.uniform(0, 40.0, 1)
    # 开单时刻在 [0,t_obs] 内均匀分布；快照只看到 open_t<=t_obs<open_t+H 的在场单
    open_t = rng.uniform(0, 40.0, n_traders)
    active = (open_t <= t_obs) & (open_t + H > t_obs)
    snap = H[active]
    stat = {"n_active": int(active.sum()), "true_mean_H": float(H.mean()),
            "snapshot_mean_H": float(snap.mean()),
            "amplification": float(snap.mean() / H.mean()),
            "theory_1pCV2": float(lognormal_params(SIG0)[1]),
            "pct_long_holders_true": float((H > 3).mean() * 100),
            "pct_long_holders_snap": float((snap > 3).mean() * 100)}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bins = np.linspace(0, 6, 60)
    ax.hist(H, bins=bins, density=True, alpha=0.55, color="#2f6db3",
            label="全部订单的真实持仓时长")
    ax.hist(snap, bins=bins, density=True, alpha=0.55, color="#c94f4f",
            label="快照中在场订单的持仓时长")
    ax.axvline(H.mean(), color="#2f6db3", ls=":", lw=1.2)
    ax.axvline(snap.mean(), color="#c94f4f", ls=":", lw=1.2)
    ax.set_xlabel("持仓时长（天）")
    ax.set_ylabel("密度")
    ax.set_title(f"持仓快照截面均值 {snap.mean():.3f} 天 vs 真值 {H.mean():.3f} 天"
                 f"（放大 {stat['amplification']:.3f}×）")
    ax.legend()
    _save(fig, "fig5_position_snapshot.png")
    return stat


def main():
    rng = np.random.default_rng(SEED)
    stats = {
        "seed": SEED,
        "exp1_inspection": exp1_inspection(rng),
        "exp2_cv_sweep": exp2_cv_sweep(rng),
        "exp3_residual": exp3_residual(rng),
        "exp4_estimator": exp4_estimator(rng),
        "exp5_snapshot": exp5_snapshot(rng),
    }
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


复现提示：核心函数 `inspected_lengths` 里"每 400 个观察换一条独立的 2 万天过程"是数值上的关键设计——早期版本所有观察共用一条轨迹时，重尾端（CV²=2.35）的仿真值曾偏离理论 30% 以上，因为单条实现的 300 天窗口时间平均等效样本量只有几百个。跨实现平均之后，最大偏离收敛到 0.027。做蒙特卡洛验证时，"样本多"不等于"独立样本多"，这是本文自己踩过的第二个坑。

## 九、带走的东西

- 凡是"随机时刻看正在进行的事件"得到的时长统计，都乘了 $xf(x)/\mathbb{E}[L]$，期望虚高 $1+\mathrm{CV}^2$，先算 CV² 再决定要不要校正；
- 修偏差用调和均值，不是加样本；n 再大也修不掉结构性偏差；
- "还要等多久"的标签分布是平衡分布，期望 $\mathbb{E}[L^2]/(2\mathbb{E}[L])$，比"平均间隔的一半"还准的说法来自它；
- 持仓、挂单、宕机修复、病程，四类统计共享同一个陷阱，报告时注明你给的是 size-biased 口径还是全体口径。

