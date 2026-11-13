---
title: "连 oracle 都过不了 MZ 检验：r² proxy 之下的波动率预测评估"
description: "用真值已知的 GARCH(1,1) 现场对拍波动率预测的两套评估流程。真值 proxy 下一切正常：GARCH QMLE 滤波 b=1.0668、R²=0.9997，EWMA b=0.8783 被 300 次重复以 98% 的拒绝率正确揪出。换成实务默认的 r² proxy，三件事同时发生：MZ 的 R² 全线塌到 7.3%（理论 Var(h)/Var(r²)=0.0927），连 oracle 也不豁免，与预测好坏无关；b̂ 在 300 次重复里均值只有 0.9158——向下偏 8%，根源是 h_{t+1} 内生于当期噪声 z²_t，固定 h 路径重抽噪声即恢复 0.985；NW(10) t 检验实际 size 16.33% 对名义 5%。损失端同样被稀释：常数预测与 GARCH 的 MSE 差距从 118.7 倍压到 1.10 倍，QLIKE 从 46.7 倍压到 1.08 倍；但 QLIKE 对高估更宽容（×1.014 对 MSE ×1.025），proxy 噪声扫描下 MSE 排名翻转率升到 6.3% 而 QLIKE 只有 0.7%。附完整可运行代码、结果 JSON 与五张真实计算图。"
publishDate: '2026-11-13T07:12:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 波动率
  - Python
language: zh
difficulty: advanced
---

## 一、MZ 回归：评估波动率预测的第一站，和它的两个 proxy

Mincer-Zarnowitz 回归是预测评估的入门仪式：把已实现的因变量对预测值回归，y = a + b·f，无偏且有效的预测应当 a=0、b=1。放到波动率上，第一个麻烦立刻出现——真实的条件方差 h_t 不可观测，你只有两个候选 proxy：实务里默认用日收益平方 r²_t（名义上 E[r²_t|h_t]=h_t，无偏）；研究里偶尔用真值已知的设计来校准。

这篇文章用真值已知的 GARCH(1,1)（ω=2e-6、α=0.08、β=0.90、日频，训练 3000 天做高斯 QMLE、评估 2000 天）把两套 proxy 各跑一遍，结论是：**换成 r² proxy 之后，MZ 检验的三处指标同时失灵，而且失灵的方式都有明确的力学来源**。五个预测参与全程对比：oracle（真条件方差）、GARCH QMLE 滤波、EWMA（λ=0.94）、常数（长期方差 2e-5）、以及一个故意偏高 15% 的「改良模型」（GARCH×1.15）——最后这个用来测试损失函数会不会被骗。

![图1 评估窗口上的预测路径：EWMA 是滞后峰值的平滑复制](/images/mincer-zarnowitz-vol-forecast-eval/fig1_forecast_paths.png)

## 二、真值 proxy 下的基准：大体正常，但 t 值已经在虚报精度

![图2 真值 proxy 下的 MZ 回归：oracle 恒等，GARCH 轻微偏离](/images/mincer-zarnowitz-vol-forecast-eval/fig2_mz_true_proxy.png)

图 2 是真值 proxy 的散点与回归。oracle 对真值是恒等回归（b=1、R²=1），这是基准的定义。GARCH QMLE 滤波 b=1.0668、R²=0.9997，点估计轻微过冲；EWMA b=0.8783——EWMA 本质是滞后实现方差的指数平滑，峰值来了它慢半拍，图 1 里看得清清楚楚，被 MZ 抓住是应该的：300 次重复里它被 |t|>1.96 拒绝的比例是 98%，b 均值 0.8800。

但先别急着信那个 t 值。单一切分里 GARCH 的 NW(10) t 值是 42.48——而 300 次重复实测 b̂ 的标准差是 0.0175，单切分的 b=1.0668 离 1 不过 3.8 个 MC 标准差，42 倍的「显著」纯属虚报。原因写在残差里：真值 proxy 下 MZ 残差的一阶自相关高达 0.8989——两条平滑的波动率路径相减，残差是高度持久的光滑序列，Newey-West 取 10 阶滞后远远吃不掉这种依赖，标准误被压到真实抽样波动的一成以下。**平滑 proxy 之下的 MZ t 值一律要打折看**，这一点在 r² proxy 一节还会再撞上一次。

## 三、r² proxy：R² 崩塌，而且与预测好坏无关

![图3 换成 r² proxy：连 oracle 的 R² 都塌到理论水平](/images/mincer-zarnowitz-vol-forecast-eval/fig3_mz_r2_proxy.png)

图 3 是本文第一记重锤。换成 r² proxy，四个预测的 MZ R² 分别是 0.0731、0.0730、0.0701、0——oracle 与 GARCH 几乎并列，EWMA 跟着陪跑。这不是巧合，是力学：R² 的总体值等于 Var(h)/Var(r²)，而 Var(r²) = 3E[h²] − (E[h])²，被 z² 噪声乘出一个 3 倍的因子再叠加波动聚集，本例算出来 0.0927，与单切分实测 0.073、300 次重复均值 0.104 同量级。**r² 的方差里九成是 z² 噪声，任何预测都只能解释剩下那一成**。如果你的汇报里写着「波动率模型只解释了日收益平方方差的 7%，改进空间很大」，这句话在 r² proxy 下没有信息量——oracle 来了也只能解释这么多。

值得替 r² 说一句公道话：它至今仍是实务默认，原因不是无知而是便宜——收盘价人人有，日内高频数据要花钱、要清洗、要做微结构去噪。所以现实的用法不是弃用 r²，而是知道它的刻度：用 r² 比较两个模型的**相对**差异（都乘同一片噪声，稀释是共通的）大体公平，用 r² 下结论「模型解释力不足」「预测系统性偏低」则会把 proxy 的病记到模型头上。判断手里的评估属于哪一类，只需要问一句：换成更好的 proxy，这个结论的方向会不会翻——图 3 里 oracle 从 R²=1 掉到 0.073，已经替你回答了。

MZ 的斜率 b 呢？单切分里 oracle 的 b̂=0.8382、GARCH 0.8941、EWMA 0.7387，全都小于 1，看起来像「全体预测系统性偏低」。真的有偏，而且偏得不小——下一节给出它的力学。

## 四、b̂ 的有限样本向下偏倚：回归元是内生的

300 次重复的蒙特卡洛把这件事钉死：oracle 对 r² proxy 的 b̂ 均值 0.9158、标准差 0.0959——**期望意义下向下偏 8%**，300 次里没有一次均值回到 1 附近。固定 h 路径、独立重抽 z² 的分解实验则恢复到 0.985 ± 0.011，与 1 相容。偏倚不在 OLS 机制里，在 h 与噪声的耦合里。

机制值得写清楚。教科书证明 E[b̂|h]=1 用的是「给定回归元，误差条件均值为零」，但这里不成立：MZ 的设计矩阵里不止有 h_t，还有 h_{t+1}，而 h_{t+1} = ω + αh_t·z²_t + βh_t 内生于当期噪声 z²_t。给定整条 h 路径，z²_t 已经被 h_{t+1} 反解出来了——E[r²_t|整条 h 路径] 不再等于 h_t，而是被未来实现的 h 拉向平滑值。GARCH 持续性越强（β 越大），拉得越狠，偏倚越大。这与预测回归文献里的 Stambaugh 型偏倚同宗：持续性回归元与噪声不独立，小样本斜率有系统位移。

后果叠加成两条。其一，点评估不可信：单切分 b̂=0.838 会被误读成「模型系统性低估波动率」，实际是 proxy 病。其二，检验 size 失真：NW(10) 的 t 检验在 b=1 的原假设下实际拒绝率 16.33%——名义 5%，实际 16.3%，而且此时残差自相关只有 0.0017（r² 噪声把持久残差冲成了白噪声），所以这不是前面那类「滞后数不够」的问题，是有限样本分布本身偏了。顺带一提，真值 proxy 下同一个 NW(10) 检验 size 是 1.0%（过保守），两套 proxy 的失效方向相反，混用必翻车。

## 五、损失函数：区分度被稀释 10-40 倍，QLIKE 的宽容是不对称的

![图4 r² proxy 把损失区分度稀释 10-40 倍](/images/mincer-zarnowitz-vol-forecast-eval/fig4_loss_discrimination.png)

评估的另一半是损失函数。图 4 把三个候选预测相对 GARCH 的损失倍数铺开（对数轴）：真值 proxy 下，常数预测的 MSE 是 GARCH 的 118.72 倍、EWMA 6.89 倍、偏高 15% 的 bias15 是 15.66 倍——区分度健康。换 r² proxy，全部压扁：常数只剩 1.10 倍、EWMA 1.0096 倍、bias15 1.0250 倍。**一天一格的 r² 噪声把 100 倍的模型差距压缩到 10% 以内**，这就是为什么用日频 MSE 挑波动率模型经常挑不出显著差异——不是模型差不多，是尺子的刻度被噪声淹了。QLIKE 同样被稀释（46.74 倍压到 1.08 倍），但有两点不同。

第一点是个诚实的反转：QLIKE 对**高估**更宽容。bias15（偏高 15%）在 r² proxy 下 QIKE 倍数 1.0138，比 MSE 的 1.0250 还小。这不是 bug，是 QLIKE 的损失形状：r²/f 在 f 偏大时温和、在 f 偏小时爆炸，低估的代价远高于高估。做风险计量（VaR 输入）时这个不对称恰好是美德——宁可高估不可低估；做交易仓位时它可能反过来咬你。第二点是稳定性：给 proxy 逐级加对数正态噪声（尺度 s 从 0 到 1），MSE 下 bias15 战胜无偏 GARCH 的频率从 0 爬到 6.3%，QLIKE 全程压在 0.7% 以内——排名稳定性确实是 QLIKE 好，但要按实际量级预期：是「6.3% 对 0.7%」的改善，不是「经常翻转对从不翻转」的神话。

![图5 proxy 噪声越大 MSE 排名越容易被翻转，QLIKE 贴地](/images/mincer-zarnowitz-vol-forecast-eval/fig5_proxy_noise_robustness.png)

## 六、清单与边界

落到操作层面五条：第一，汇报波动率模型的 MZ 结果时必须注明 proxy 是什么，r² proxy 下的 R² 和 b̂ 不具备跨研究可比性；第二，r² proxy 下的 R² 低不是模型的错，想报解释力就用已实现波动率（日内高频聚合）做 proxy，噪声随采样频率平方下降；第三，r² proxy 下的 b̂ 本身带约 8% 的向下偏（本例参数下），比较两个 b̂ 的相对大小仍可用，绝对水平不可信；第四，t 检验的滞后数要配得上残差的持久性，更稳妥的是直接报告蒙特卡洛校准的 size；第五，损失函数选 QLIKE 防排名翻转，但记住它罚低估、宽高估，与 VaR 场景对齐、与仓位场景未必对齐。

最后补一个容易忽略的坑：这套实验里所有预测都严格做到了一步 ahead——滤波递归只用当期之前的信息。实务复现时最常见的自欺是「准同步」：用包含当期收益的数据更新了条件方差再去「预测」当期，MZ 的 b 会漂亮得反常。本文代码里 filter_garch 的索引值得逐行看一遍：h[t] 由 r²[t−1] 递推，任何一处索引写成都用上 r²[t]，评估就从预测变成了内插，R² 轻松上 0.9——那不是模型好，是日历错了。

边界：本文所有结论来自 GARCH(1,1) 正态 innovations 的单一数据生成过程，参数取日频常见值；realized volatility 类 proxy 的噪声结构（微结构噪声、采样频率）没有建模；QLIKE 的稳健性只在本设计的排名频率意义上验证，未触及 Patton (2011) 的函数形式条件；偏倚的 8% 依赖具体参数，持续性更低的市场会更小。参考文献：Mincer & Zarnowitz (1969)、Andersen & Bollerslev (1998) 关于 r² proxy 的经典批评、Patton (2011) 关于稳健损失函数。

```python
# -*- coding: utf-8 -*-
"""波动率预测评估：Mincer-Zarnowitz 回归与 QLIKE 的现场对拍。

本文所有数字的唯一来源。运行本脚本会：
1. 现场生成 GARCH(1,1) 数据（固定随机种子，可复现）；
2. 手写高斯 QMLE 估计 GARCH 参数；
3. 画 5 张图到 public/images/mincer-zarnowitz-vol-forecast-eval/；
4. 把全部统计量写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261113
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "mincer-zarnowitz-vol-forecast-eval"
)
os.makedirs(_OUT, exist_ok=True)

# 数据生成参数（日频 GARCH(1,1)）
OMEGA, ALPHA, BETA = 2e-6, 0.08, 0.90
LRV = OMEGA / (1.0 - ALPHA - BETA)  # 长期方差


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


def simulate_garch(n, rng, burn=200):
    """返回 (r, h)，先丢弃 burn 期。"""
    r = np.empty(n + burn)
    h = np.empty(n + burn)
    h[0] = LRV
    z = rng.standard_normal(n + burn)
    r[0] = np.sqrt(h[0]) * z[0]
    for t in range(1, n + burn):
        h[t] = OMEGA + ALPHA * r[t - 1] ** 2 + BETA * h[t - 1]
        r[t] = np.sqrt(h[t]) * z[t]
    return r[burn:], h[burn:]


def garch_mle(r):
    """高斯 QMLE 估计 GARCH(1,1)。返回 (omega, alpha, beta)。"""
    r2 = r ** 2
    v0 = r2.var()

    def nll(p):
        om, a, b = p
        if om <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
            return 1e10
        h = np.empty_like(r2)
        h[0] = v0
        for t in range(1, len(r2)):
            h[t] = om + a * r2[t - 1] + b * h[t - 1]
        if np.any(h <= 0):
            return 1e10
        return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + r2 / h))

    best = None
    for a0 in (0.05, 0.10):
        for b0 in (0.85, 0.92):
            res = minimize(nll, x0=[1e-6, a0, b0], method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-10,
                                    "fatol": 1e-12})
            if best is None or res.fun < best.fun:
                best = res
    om, a, b = best.x
    return float(om), float(a), float(b)


def filter_garch(r2, om, a, b):
    """用给定参数沿已实现收益滤波出条件方差序列。"""
    h = np.empty_like(r2)
    h[0] = om / (1.0 - a - b) if a + b < 1 else r2.mean()
    for t in range(1, len(r2)):
        h[t] = om + a * r2[t - 1] + b * h[t - 1]
    return h


# ---------------------------------------------------------------- MZ 回归
def mz_regression(proxy, forecast, nw_lags=10):
    """MZ 回归 proxy = a + b*f。返回 (a, b, NW 标准误, t(b=1), R^2)。"""
    x = np.vstack([np.ones(len(forecast)), forecast]).T
    coef, *_ = np.linalg.lstsq(x, proxy, rcond=None)
    resid = proxy - x @ coef
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    s = np.zeros((k, k))
    for lag in range(nw_lags + 1):
        w = 1.0 - lag / (nw_lags + 1.0)
        g = (x[lag:] * resid[lag:, None]).T @ (x[: n - lag] * resid[: n - lag, None])
        s += w * (g + g.T) if lag > 0 else g
    v = bread @ s @ bread
    se_b = float(np.sqrt(max(v[1, 1], 0.0) * n / (n - k)))
    b = float(coef[1])
    t_b1 = float((b - 1.0) / se_b) if se_b > 0 else None
    sst = float(((proxy - proxy.mean()) ** 2).sum())
    r_squared = float(1.0 - (resid ** 2).sum() / sst) if sst > 0 else None
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    return float(coef[0]), b, se_b, t_b1, r_squared, r1


# ---------------------------------------------------------------- 损失
def mse_loss(forecast, proxy):
    return float(np.mean((proxy - forecast) ** 2))


def qlike_loss(forecast, proxy):
    ratio = proxy / forecast
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def main():
    rng = np.random.default_rng(SEED)
    T_TRAIN, T_EVAL = 3000, 2000
    r_full, h_full = simulate_garch(T_TRAIN + T_EVAL, rng, burn=500)
    r_train, h_train = r_full[:T_TRAIN], h_full[:T_TRAIN]
    r_eval, h_eval = r_full[T_TRAIN:], h_full[T_TRAIN:]
    r2_eval = r_eval ** 2

    om_hat, a_hat, b_hat = garch_mle(r_train)
    # 各预测（一步 ahead，只用当期之前的信息）
    h_garch = filter_garch(r_full ** 2, om_hat, a_hat, b_hat)[T_TRAIN:]
    h_ewma = filter_garch(r_full ** 2, 0.0, 1.0 - 0.94, 0.94)[T_TRAIN:]
    h_const = np.full(T_EVAL, om_hat / (1.0 - a_hat - b_hat))
    h_oracle = h_eval
    forecasts = {
        "oracle": h_oracle,
        "garch": h_garch,
        "ewma": h_ewma,
        "const": h_const,
        "bias15": 1.15 * h_garch,
    }

    # ---- 单一切分：MZ 回归（两个 proxy）与损失
    mz = {}
    for pname, proxy in (("h_true", h_eval), ("r2", r2_eval)):
        for fname, fvec in forecasts.items():
            a, b, se, t, rsq, acf1 = mz_regression(proxy, fvec)
            mz[f"{pname}|{fname}"] = {"a": a, "b": b, "se_b": se, "t_b1": t,
                                      "R2": rsq, "resid_acf1": acf1}
    # r² proxy 的 R² 理论值：Var(h) / Var(r²)，与预测好坏无关
    eh = float(h_eval.mean())
    eh2 = float((h_eval ** 2).mean())
    vh = eh2 - eh ** 2
    r2_theory = vh / (3.0 * eh2 - eh ** 2)

    losses = {}
    for pname, proxy in (("h_true", h_eval), ("r2", r2_eval)):
        for fname, fvec in forecasts.items():
            losses[f"{pname}|mse|{fname}"] = mse_loss(fvec, proxy)
            losses[f"{pname}|qlike|{fname}"] = qlike_loss(fvec, proxy)

    # ---- 图1：预测路径对比
    sl = slice(1200, 1800)
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    idx = np.arange(T_EVAL)[sl]
    ax.plot(idx, np.abs(r_eval[sl]), color="#d0d7de", lw=0.7,
            label="|日收益|（噪声基准）")
    ax.plot(idx, np.sqrt(h_oracle[sl]), color="#d1242f", lw=1.6,
            label="真实条件波动（oracle）")
    ax.plot(idx, np.sqrt(h_garch[sl]), color="#1f6feb", lw=1.2,
            label="GARCH(1,1) QMLE 滤波")
    ax.plot(idx, np.sqrt(h_ewma[sl]), color="#8250df", lw=1.2, ls="--",
            label="EWMA (λ=0.94)")
    ax.set_xlabel("评估集交易日")
    ax.set_ylabel("日波动率")
    ax.set_title("图1 评估窗口上的波动率预测路径：EWMA 是滞后峰值的平滑复制")
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, "fig1_forecast_paths.png")

    # ---- 图2：真值 proxy 下的 MZ 散点
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True, sharey=True)
    for ax, fname, color in ((axes[0], "oracle", "#d1242f"),
                             (axes[1], "garch", "#1f6feb")):
        fvec = forecasts[fname]
        sub = slice(0, 2000)
        ax.scatter(fvec[sub] * 1e4, h_eval[sub] * 1e4, s=4, alpha=0.25,
                   color=color)
        lim = max(fvec.max(), h_eval.max()) * 1e4
        ax.plot([0, lim], [0, lim], color="gray", ls=":", lw=1)
        m = mz[f"h_true|{fname}"]
        xs = np.linspace(1e-6, lim * 1e-4, 10)
        ax.plot(xs * 1e4, (m["a"] + m["b"] * xs) * 1e4, color="black", lw=1.4)
        t_txt = "恒等回归" if m["t_b1"] is None else f"t(b=1)={m['t_b1']:.2f}"
        ax.set_title(
            f"{fname}:  b={m['b']:.4f}  R²={m['R2']:.3f}  {t_txt}",
            fontsize=10)
        ax.set_xlabel("预测 h（×1e-4）")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("真实条件方差 h（×1e-4）")
    fig.suptitle("图2 真值 proxy 下的 MZ 回归：oracle 应当 b=1，GARCH 轻微偏离", fontsize=11)
    _save(fig, "fig2_mz_true_proxy.png")

    # ---- 图3：r² proxy 下 R² 全线塌到理论水平，与预测好坏无关
    names = ["oracle", "garch", "ewma", "const"]
    rsqs = [mz[f"r2|{n}"]["R2"] for n in names]
    bs_r2 = [mz[f"r2|{n}"]["b"] for n in names]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(names))
    ax.bar(x, rsqs, 0.6, color="#1f6feb", alpha=0.85)
    ax.axhline(r2_theory, color="#d1242f", lw=1.6, ls="--",
               label=r"理论 $\mathrm{Var}(h)/\mathrm{Var}(r^2)$"
                     f" = {r2_theory:.4f}")
    for xi, b, rsq in zip(x, bs_r2, rsqs):
        ax.text(xi, rsq + 0.004, f"b={b:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"MZ 回归的 $R^2$（proxy = $r^2$）")
    ax.set_title(r"图3 换成 $r^2$ proxy：连 oracle 的 $R^2$ 都塌到 $\approx$ 9%，"
                 "与预测好坏无关")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig3_mz_r2_proxy.png")

    # ---- MC 实验 1：MZ + NW t 检验的实际 size（b=1 的拒绝率）
    N_MC = 300
    rng_mc = np.random.default_rng(SEED + 1)
    pairs = [("oracle", "r2"), ("garch", "r2"), ("garch", "h_true"),
             ("ewma", "h_true")]
    stats = {f"{f}|{p}": {"rej": 0, "b": [], "R2": []} for f, p in pairs}
    for rep in range(N_MC):
        r_m, h_m = simulate_garch(T_EVAL, rng_mc, burn=200)
        fcands = {
            "oracle": h_m,
            "garch": filter_garch(r_m ** 2, OMEGA, ALPHA, BETA),
            "ewma": filter_garch(r_m ** 2, 0.0, 1.0 - 0.94, 0.94),
        }
        for fname, fvec in fcands.items():
            for pname, proxy in (("r2", r_m ** 2), ("h_true", h_m)):
                key = f"{fname}|{pname}"
                if key not in stats:
                    continue
                _, b, _, t, rsq, _ = mz_regression(proxy, fvec)
                stats[key]["rej"] += int(t is not None and abs(t) > 1.96)
                stats[key]["b"].append(b)
                stats[key]["R2"].append(rsq)
    # 偏倚来源分解：固定 h 路径、独立重抽 z^2，E[b|h]=1 应当恢复
    r_fix, h_fix = simulate_garch(T_EVAL, rng_mc, burn=200)
    bs_fix = []
    for k in range(200):
        z2 = np.random.default_rng(SEED + 7 + k).standard_normal(T_EVAL) ** 2
        x = np.vstack([np.ones(T_EVAL), h_fix]).T
        coef, *_ = np.linalg.lstsq(x, h_fix * z2, rcond=None)
        bs_fix.append(float(coef[1]))
    mc_size = {
        "fixed_h_redraw_z": {"b_mean": float(np.mean(bs_fix)),
                             "b_sd": float(np.std(bs_fix, ddof=1)),
                             "N": 200},
    }
    for key, st in stats.items():
        mc_size[key] = {
            "rej_rate": st["rej"] / N_MC,
            "b_mean": float(np.mean(st["b"])),
            "b_sd": float(np.std(st["b"], ddof=1)),
            "R2_mean": float(np.mean(st["R2"])),
        }

    # ---- MC 实验 2 / 图4：损失区分度——各预测相对 GARCH 的倍数
    multiples = {}
    for pname in ("h_true", "r2"):
        for loss_name, fn in (("mse", mse_loss), ("qlike", qlike_loss)):
            base = fn(forecasts["garch"],
                      h_eval if pname == "h_true" else r2_eval)
            for fname in ("ewma", "const", "bias15"):
                multiples[f"{pname}|{loss_name}|{fname}"] = (
                    fn(forecasts[fname],
                       h_eval if pname == "h_true" else r2_eval) / base)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    groups = ["h_true|mse", "h_true|qlike", "r2|mse", "r2|qlike"]
    width = 0.26
    xg = np.arange(len(groups))
    for i, fname in enumerate(("ewma", "const", "bias15")):
        vals = [multiples[f"{g}|{fname}"] for g in groups]
        ax.bar(xg + (i - 1) * width, vals, width,
               label={"ewma": "EWMA(0.94)", "const": "常数=长期方差",
                      "bias15": "GARCH×1.15（偏高 15%）"}[fname], alpha=0.88)
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", ls=":", lw=1.2)
    for i, g in enumerate(groups):
        for j, fname in enumerate(("ewma", "const", "bias15")):
            v = multiples[f"{g}|{fname}"]
            ax.text(i + (j - 1) * width, v * 1.08, f"×{v:.2f}", ha="center",
                    fontsize=7.2, rotation=90)
    ax.set_xticks(xg)
    ax.set_xticklabels([gg.replace("h_true", "proxy=h").replace("r2", "proxy=r²")
                        for gg in groups])
    ax.set_ylabel("损失 / GARCH 损失（对数轴）")
    ax.set_title("图4 r² proxy 把损失区分度稀释 10-40 倍；QLIKE 对高估更宽容")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis="y", which="both")
    _save(fig, "fig4_loss_discrimination.png")

    # ---- proxy 噪声稳健性扫描
    scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    rng_mc2 = np.random.default_rng(SEED + 2)
    sweep = {"scales": scales, "mse_flip": [], "qlike_flip": []}
    paths = [simulate_garch(T_EVAL, rng_mc2, burn=200) for _ in range(N_MC)]
    for s in scales:
        fm = fq = 0
        rng_n = np.random.default_rng(SEED + 3 + int(s * 1000))
        for r_m, h_m in paths:
            h_filt = filter_garch(r_m ** 2, OMEGA, ALPHA, BETA)
            h_b = 1.15 * h_filt
            z = rng_n.standard_normal(T_EVAL)
            proxy = h_m * np.exp(s * z - 0.5 * s * s)
            fm += mse_loss(h_b, proxy) < mse_loss(h_filt, proxy)
            fq += qlike_loss(h_b, proxy) < qlike_loss(h_filt, proxy)
        sweep["mse_flip"].append(fm / N_MC)
        sweep["qlike_flip"].append(fq / N_MC)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(scales, sweep["mse_flip"], "o-", color="#d1242f",
            label="MSE 排名翻转概率")
    ax.plot(scales, sweep["qlike_flip"], "s-", color="#1f6feb",
            label="QLIKE 排名翻转概率")
    ax.set_xlabel("proxy 对数正态噪声尺度 s")
    ax.set_ylabel("P( +15% 有偏预测胜过无偏滤波 )")
    ax.set_title("图5 proxy 噪声越大 MSE 越容易被骗，QLIKE 全程贴地")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "fig5_proxy_noise_robustness.png")

    results = {
        "dgp": {"omega": OMEGA, "alpha": ALPHA, "beta": BETA,
                "long_run_var": LRV},
        "mle": {"omega": om_hat, "alpha": a_hat, "beta": b_hat,
                "T_train": T_TRAIN, "T_eval": T_EVAL},
        "mz": mz,
        "r2_theory": r2_theory,
        "mc_size": mc_size,
        "loss_multiples": multiples,
        "losses_single_split": losses,
        "noise_sweep": sweep,
    }
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

