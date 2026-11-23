---
title: "预测回归的伪可预测性：股息率真能预测收益吗，Stambaugh 偏误如何拧歪一致估计的 OLS 斜率，以及为什么 bootstrap 修不动"
description: "预测变量持续且与收益新息同期相关时，OLS 斜率虽一致却有 O(1/n) 的系统偏误，符号跟着 delta 翻转。真值已知实验：delta 从 -0.8 到 +0.8，OLS 斜率偏误从 +0.0147 翻到 -0.0160，Stambaugh 解析修正把两端各砍到 +0.0072 与 -0.0085；n 从 50 到 800，偏误按 1/n 从 0.0620 缩到 0.0031。但 t 检验的 size 才是真坑——phi=0.98 且 delta=+0.8 时虚警率冲到 10.5%，凭空造出可预测性。残差 bootstrap 修正与 OLS 几乎重合，救不了这种偏误。附完整代码与五张真实计算图。"
publishDate: '2026-11-23T09:50:00Z'
tags:
  - 量化
  - 计量经济学
  - 预测回归
  - 自助法
  - Python
language: zh
difficulty: advanced
image: "/images/predictive-regression-stambaugh-bias/fig1_delta_bias.png"
---

## 一、结论先行

用股息率、估值、期限利差去预测下一期收益，是量化里最古老的套路之一。回归写法都是 $r_t=\alpha+\beta\,x_{t-1}+v_t$，然后盯 $\beta$ 的 t 值。问题在于：这类预测变量天生持续（AR 系数 $\phi$ 常年在 0.9 以上），而且它的创新 $u_t$ 往往和同期收益新息 $v_t$ 相关。两件事凑一起，就掉进 Stambaugh (1999) 挖的坑。

先把话说清楚，避免一个常见误读：**在这个设定下 OLS 是一致的**。因为回归元是 $x_{t-1}$，与当期创新 $u_t$ 独立，所以 $\hat\beta$ 的概率极限就是真值。偏误不是遗漏变量那种不会消失的不一致，而是有限样本的 $O(1/n)$ 效应。它小、但系统；真正危险的其实不是偏误本身，而是 t 检验的虚警率。

真值已知实验（$N=200$、$\beta=0$）给出三个核心数字：

- **偏误跟着相关性符号翻转**：$\delta=\text{corr}(u_t,v_t)$ 从 -0.8 到 +0.8，OLS 斜率偏误从 +0.0147 翻到 -0.0160，中间 $\delta=0$ 时几乎为零。量级只有百分之几，但方向完全可预测。
- **Stambaugh 解析修正能减半**：同样的两端，修正后偏误降到 +0.0072 与 -0.0085；小样本更明显，$n=50$ 时从 0.0620 砍到 0.0355。它不是神药，是稳定地把偏误压掉大约一半。
- **t 检验的 size 才是真坑**：数据里根本没有可预测性（$\beta=0$），但当 $\phi=0.98$、$\delta=+0.8$ 时，$|t|>1.96$ 的频率冲到 10.5%——名义 5% 的检验，虚警翻倍。这就是"凭空造出来的可预测性"。

最后一个诚实的负面结论：**残差 bootstrap 修不动**。它的偏误曲线与 OLS 几乎重合（$\delta=-0.8$ 时 0.0148 对 0.0147），因为对残差重抽样只是在复制同一个产生偏误的机制，无法把它剥离。

## 二、数据生成与三种估计

设定如下，所有参数真值已知：

$$
x_t=\phi\,x_{t-1}+u_t,\qquad u_t\sim N(0,1)
$$

$$
v_t=\delta\,u_t+\sqrt{1-\delta^2}\,e_t,\qquad r_t=\alpha+\beta\,x_{t-1}+v_t
$$

$x_{t-1}$ 持续（$\phi$ 可调），收益新息 $v_t$ 里掺了 $\delta$ 比例的 $u_t$。$\delta\neq0$ 就是"预测变量的当下意外和收益的当下意外一起动"。

三种斜率估计：

1. **OLS**：直接把 $r_t$ 回归到 $x_{t-1}$，报告斜率与 t 值。
2. **Stambaugh 解析修正**：先估 $\hat\delta=\text{corr}(\hat u_t,\hat v_t)$（两条辅助回归的残差同期相关），再用近似偏误公式 $\hat\delta\,(\hat\sigma_v/\hat\sigma_u)(1+\hat\phi)/n$ 反向扣掉。
3. **残差 bootstrap**：对辅助回归的残差重抽样生成伪样本，用 $\hat\beta^{\text{bc}}=2\hat\beta-\overline{\hat\beta^*}$ 做偏差校正。

为了看清结构，我把回归设定成 $r_t$ 对 $x_{t-1}$（$t=1,\dots,n-1$），两条辅助回归也用同样的样本，保证 $\hat u_t$ 和 $\hat v_t$ 的同期对齐。这个对齐是整套估计的命门——错位一期，$\hat\delta$ 就归零，修正彻底失效，这也是新手复现 Stambaugh 时最常踩的坑。

## 三、偏误随 delta 翻转，修正减半

第一组实验固定 $\phi=0.9$、$n=200$、$\beta=0$，把 $\delta$ 从 -0.8 扫到 +0.8。

OLS 斜率偏误依次为 +0.0147、+0.0090、+0.0004、-0.0085、-0.0160，几乎是 $\delta$ 的线性函数，穿过原点。这个符号有直觉解释：当 $\delta<0$，预测变量的正面意外伴随收益的负面意外，回归把这种反向共动记成负斜率；但恰好因为回归元滞后一期，这个共动只在有限样本里漏进来一点，形成正的小偏误。

![delta与OLS斜率偏误](/images/predictive-regression-stambaugh-bias/fig1_delta_bias.png)

Stambaugh 修正把五档依次压到 +0.0072、+0.0052、+0.0004、-0.0047、-0.0085——每档都朝零收缩约一半，且从不越过零反向放大。这正是解析修正该有的样子：吃掉的幅度稳定，不制造新的偏差。bootstrap 则几乎原地踏步，五档全和 OLS 重合。

一句话总结这组：$\hat\delta$ 本身估得很准（$\delta=-0.8$ 时均值 -0.800、+0.8 时 +0.798），所以解析修正靠谱；bootstrap 不依赖 $\hat\delta$，它靠重抽样自己找偏误，而重抽样过程又复刻了偏误，等于用一把带同样刻度的尺子去量误差。

## 四、size 失真：真正危险的地方

第二组实验才是本文的重点。固定 $\beta=0$（没有任何可预测性），看 $|t|>1.96$ 的拒绝频率，名义应该是 5%。

![t检验的虚警率](/images/predictive-regression-stambaugh-bias/fig2_size_distortion.png)

结果分成两片。$\delta=0$ 的三档（$\phi=0.5/0.9/0.98$）虚警率是 0.0375、0.0475、0.0500，守在 5% 附近——没有同期相关，持续性再高也不会凭空造出显著性。一旦 $\delta\neq0$，虚警率抬头：$\phi=0.9$、$\delta=\pm0.8$ 时是 0.0650 和 0.0625，到了 $\phi=0.98$ 就冲到 0.0900 和 0.1050。

这才是 Stambaugh 偏误的真正杀伤力。斜率偏误才 0.016，对收益预测的经济意义几乎可以忽略；但 t 检验的虚警翻倍，意味着你在一堆真正没有预测力的变量里，会有约十分之一被判为"显著"。如果同时扫描几十个估值指标，按这个虚警率，几乎必然能挑出几个"显著"的——赢家诅咒和 Stambaugh 偏误叠加，就是伪因子泛滥的流水线。

所以要分清两件事：斜率偏误是统计上的小瑕疵，修正一下就好；size 失真是决策上的大坑，它直接污染变量筛选。而 bootstrap 偏差校正对第二件事帮不上忙，因为它修的是点估计的偏误，不是检验的抽样分布。真要做稳健推断，得换 HAC 标准误、块自助或者做多重检验校正——那是另一个话题。

## 五、样本量与持续性：偏误怎么缩放

第三组固定 $\phi=0.95$、$\delta=-0.7$，把 $n$ 从 50 扫到 800。OLS 偏误依次是 0.0620、0.0298、0.0142、0.0061、0.0031——每翻倍大约减半，干净的 $1/n$ 衰减。Stambaugh 修正在每一档都再砍一半：0.0355、0.0162、0.0074、0.0027、0.0014。小样本（$n=50$）收益最大，偏误从 0.062 降到 0.036；$n=800$ 时两者都小到无所谓。

![样本量对偏误的影响](/images/predictive-regression-stambaugh-bias/fig3_sample_size.png)

第四组固定 $\delta=-0.7$、$n=200$，扫 $\phi$。OLS 偏误从 $\phi=0$ 的 0.0011 爬到 $\phi=0.98$ 的 0.0167，翻了十几倍；Stambaugh 修正同步从 -0.0024（这里轻微过冲）到 0.0099。持续性是偏误的放大器：$\phi$ 越高，$x$ 越接近随机游走，滞后回归元与当期新息的那点残余相关越难被样本平均掉。

![持续性对偏误的放大](/images/predictive-regression-stambaugh-bias/fig4_phi_bias.png)

注意 $\phi=0$ 那一档 Stambaugh 修正出现 -0.0024 的轻微过冲。这提醒修正公式是一阶近似，在偏误本来就接近零的场景会稍微修过头。实践中偏误小于千分之几时，修不修差别不大，没必要为了"用了高级方法"硬套。

## 六、一个样本里的伪信号

最后看一个 $n=120$、$\phi=0.95$、$\delta=-0.7$、真值 $\beta=0$ 的具体样本。OLS 斜率 -0.0084、t 值 -0.39——这个样本里没有触发虚警，斜率也 small。Stambaugh 修正给出 -0.0198，反而离零更远。

![单个样本里的伪可预测性](/images/predictive-regression-stambaugh-bias/fig5_spurious_scatter.png)

这个"修正后更差"的例子必须写进来，否则就是选择性呈现。解析修正保证的是**期望意义上**把偏误减半，不是保证每一个样本都更接近真值。$\hat\delta$ 在这个样本里的抽样波动，会让修正项偶尔过冲。这就是偏差-方差的老规矩：修正降低偏差，但引入一点额外的抽样噪声。用它做批量推断有价值，拿它包装单个回归的"精确估计"就过头了。

## 七、实务清单

| 现象 | 诊断 | 应对 |
|---|---|---|
| 预测变量 $\phi>0.9$ | 高度持续，检查是否与收益新息同期相关 | 估 $\hat\delta$，别只看 t 值 |
| 斜率偏误百分之几 | Stambaugh $O(1/n)$ 效应 | 解析修正减半，或增大 $n$ |
| $|t|>1.96$ 频繁出现 | size 失真，尤其 $\delta\neq0$ 时 | HAC 标准误 / 块自助 / 多重检验校正 |
| bootstrap 偏差校正无效 | 重抽样复刻了偏误机制 | 别指望它，换思路 |

最实用的一条：看到"股息率显著预测收益"这类结论，先问三件事——预测变量的 AR 系数多高、它和同期收益新息的相关 $\hat\delta$ 多大、t 检验在 $\beta=0$ 下的真实虚警率是多少。这三个数比那个显著性星号本身重要得多。

还有一层容易被混淆的关系值得点明：本文的 Stambaugh 设定与"弱工具变量"是两种不同的病。弱工具是回归元与工具相关太弱导致估计量和区间爆炸；这里的问题是回归元滞后一期、与同期新息独立，一致性没问题，只是有限样本里斜率被轻微拧歪、检验被放大虚警。两者的诊断信号也不一样——弱工具看一阶段 F 和区间宽度，Stambaugh 看 $\hat\delta$ 和 size。把两种病用同一副药（比如多找工具）去治，往往越治越糟。分清病因，才谈得上选对药。

```python
# -*- coding: utf-8 -*-
"""Predictive regression: Stambaugh endogeneity bias and its correction.

Single source of truth for every number and figure in the article.

DGP (true parameters known), the Stambaugh (1999) setup:
    x_t = phi * x_{t-1} + u_t,     u_t ~ N(0,1)
    v_t = delta * u_t + sqrt(1-delta^2) * e_t     (v_t correlated with u_t)
    r_t = alpha + beta * x_{t-1} + v_t

Because x_{t-1} is independent of u_t, OLS is CONSISTENT: the bias is a
finite-sample O(1/n) effect, not an omitted-variable inconsistency. But the
bias is systematic (sign flips with delta, grows with phi) and, more
importantly, the t-test on beta=0 is size distorted when delta != 0 -- the
classic "spurious predictability" trap. We compare three slope estimates:
    OLS, Stambaugh's analytic bias correction (delta-hat plugged in),
    and a residual-bootstrap bias correction,
with sweeps over delta, phi, and n. Fixed seeds keep exec-replay identical to
results.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261123
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "predictive-regression-stambaugh-bias")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.0


def simulate(n, phi, delta, beta, rng, warm=120):
    """Return x (n,), r (n,) under the Stambaugh setup (same-t u/v correlation)."""
    total = n + warm
    u = rng.standard_normal(total)
    e = rng.standard_normal(total)
    v = delta * u + np.sqrt(1.0 - delta ** 2) * e
    x = np.empty(total)
    x[0] = u[0] / np.sqrt(1.0 - phi ** 2)
    for t in range(1, total):
        x[t] = phi * x[t - 1] + u[t]
    r = ALPHA + beta * np.concatenate([[0.0], x[:-1]]) + v
    return x[warm:], r[warm:]


def ols_fit(x_lag, r):
    """OLS slope of r on x_lag; returns (b, se, t, residual)."""
    X = np.column_stack([np.ones(len(x_lag)), x_lag])
    coef = np.linalg.lstsq(X, r, rcond=None)[0]
    resid = r - X @ coef
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return coef[1], se, coef[1] / se if se > 0 else 0.0, resid


def aux_resid(x, r):
    """Return u_hat (from x_t~x_{t-1}) and v_hat (from r_t~x_{t-1}), both len n-1, same-t."""
    n = len(x)
    x_lag, x_now, rr = x[:-1], x[1:], r[1:]
    Xa = np.column_stack([np.ones(n - 1), x_lag])
    ca = np.linalg.lstsq(Xa, x_now, rcond=None)[0]
    u_hat = x_now - Xa @ ca
    Xr = np.column_stack([np.ones(n - 1), x_lag])
    cr = np.linalg.lstsq(Xr, rr, rcond=None)[0]
    v_hat = rr - Xr @ cr
    return u_hat, v_hat, ca[1]


def stambaugh_corrected(x, r):
    """b_sc = b_ols - delta_hat*(sigma_v/sigma_u)*(1+phi_hat)/n."""
    x_lag, rr = x[:-1], r[1:]
    b_ols, _, _, _ = ols_fit(x_lag, rr)
    n = len(rr)
    u_hat, v_hat, phi_hat = aux_resid(x, r)
    denom = np.sqrt((u_hat @ u_hat) * (v_hat @ v_hat))
    delta_hat = float(u_hat @ v_hat / denom) if denom > 0 else 0.0
    sd_u = np.sqrt(u_hat @ u_hat / n)
    sd_v = np.sqrt(v_hat @ v_hat / n)
    phi_c = float(np.clip(phi_hat, -0.999, 0.999))
    # OLS slope bias runs opposite in sign to delta; subtract delta-hat term in that direction
    bias_hat = -delta_hat * (sd_v / sd_u) * (1.0 + phi_c) / n
    return b_ols - bias_hat, delta_hat, phi_hat


def bootstrap_corrected(x, r, n_boot=149, seed=0):
    """Residual bootstrap bias correction: b_bc = 2*b_ols - mean(b_boot*)."""
    rng = np.random.default_rng(seed)
    n = len(x) - 1
    x_lag, rr = x[:-1], r[1:]
    b_ols, _, _, _ = ols_fit(x_lag, rr)
    u_hat, v_hat, phi_hat = aux_resid(x, r)
    u_hat = u_hat - u_hat.mean()
    v_hat = v_hat - v_hat.mean()
    bs = np.empty(n_boot)
    for b in range(n_boot):
        us = rng.choice(u_hat, size=n, replace=True)
        vs = rng.choice(v_hat, size=n, replace=True)
        xs = np.empty(n + 1)
        xs[0] = 0.0
        for t in range(1, n + 1):
            xs[t] = phi_hat * xs[t - 1] + us[t - 1]
        rs = b_ols * xs[:-1] + vs
        bs[b] = ols_fit(xs[:-1], rs)[0]
    return float(2.0 * b_ols - bs.mean())


def one_rep(n, phi, delta, beta, rng, boot_seed=0):
    x, r = simulate(n, phi, delta, beta, rng)
    x_lag, rr = x[:-1], r[1:]
    b_ols, se, t_ols, _ = ols_fit(x_lag, rr)
    b_sc, dh, ph = stambaugh_corrected(x, r)
    b_bc = bootstrap_corrected(x, r, n_boot=149, seed=boot_seed)
    return b_ols, t_ols, b_sc, b_bc, dh, ph


def summarize(n, phi, delta, beta, R, seed=SEED):
    rng = np.random.default_rng(seed)
    B = np.empty((R, 6))
    for i in range(R):
        B[i] = one_rep(n, phi, delta, beta, rng, boot_seed=seed + i)
    out = {
        "n": n, "phi": phi, "delta": delta, "beta": beta, "R": R,
        "delta_hat_mean": float(B[:, 4].mean()),
        "phi_hat_mean": float(B[:, 5].mean()),
    }
    for nm, bi in zip(["ols", "stambaugh", "bootstrap"], [0, 2, 3]):
        out[nm + "_bias"] = float((B[:, bi] - beta).mean())
        out[nm + "_rmse"] = float(np.sqrt(((B[:, bi] - beta) ** 2).mean()))
    out["ols_t_reject"] = float(np.mean(np.abs(B[:, 1]) > 1.96))
    return out


def exp_delta_bias():
    return [summarize(200, 0.9, d, 0.0, 400, seed=SEED + int((d + 1) * 100))
            for d in (-0.8, -0.4, 0.0, 0.4, 0.8)]


def exp_size():
    return [summarize(200, p, dl, 0.0, 400, seed=SEED + int((p + 1) * 1000 + (dl + 1) * 10))
            for p, dl in ((0.5, 0.0), (0.9, 0.0), (0.98, 0.0),
                          (0.9, -0.8), (0.9, 0.8), (0.98, -0.8), (0.98, 0.8))]


def exp_sample_size():
    return [summarize(nn, 0.95, -0.7, 0.0, 400, seed=SEED + nn)
            for nn in (50, 100, 200, 400, 800)]


def exp_phi_bias():
    return [summarize(200, p, -0.7, 0.0, 400, seed=SEED + int(p * 1000))
            for p in (0.0, 0.3, 0.6, 0.85, 0.95, 0.98)]


# ------------------------------------------------------------------- figures
COL = ["#e8684a", "#5b8ff9", "#2ca25f"]
LAB = ["OLS", "Stambaugh", "Bootstrap"]
KEYS = ["ols", "stambaugh", "bootstrap"]


def _lines(ax, rows, xkey, field, title, xlabel, ylabel, hline=None):
    xs = [r[xkey] for r in rows]
    for k, nm in enumerate(KEYS):
        ax.plot(xs, [r[nm + field] for r in rows], "o-", color=COL[k], label=LAB[k])
    if hline is not None:
        ax.axhline(hline, color="black", ls="--", lw=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8)


def fig1_delta(rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _lines(ax, rows, "delta", "_bias",
           "OLS slope bias flips sign with delta (beta=0, phi=0.9)",
           "delta = corr(u_t, v_t)", "mean beta-hat (truth=0)", hline=0.0)
    fig.tight_layout(); p = "fig1_delta_bias.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig2_size(rows):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    labels = ["phi=%.2f\nd=%+.1f" % (r["phi"], r["delta"]) for r in rows]
    vals = [r["ols_t_reject"] for r in rows]
    colors = ["#888888" if r["delta"] == 0.0 else "#e8684a" for r in rows]
    ax.bar(range(len(rows)), vals, color=colors)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.05, color="black", ls="--", lw=1, label="5% nominal size")
    ax.set_ylabel("P(|t|>1.96) under beta=0"); ax.set_title("Spurious predictability: t-test size")
    ax.legend(fontsize=9)
    fig.tight_layout(); p = "fig2_size_distortion.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig3_sample(rows):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    _lines(ax[0], rows, "n", "_bias", "Bias shrinks as n grows (phi=0.95, delta=-0.7)",
           "sample size n", "mean beta-hat", hline=0.0)
    _lines(ax[1], rows, "n", "_rmse", "RMSE also falls with n", "sample size n", "RMSE")
    fig.tight_layout(); p = "fig3_sample_size.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig4_phi(rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _lines(ax, rows, "phi", "_bias", "Persistence phi magnifies the finite-sample bias",
           "phi", "mean beta-hat (delta=-0.7)", hline=0.0)
    fig.tight_layout(); p = "fig4_phi_bias.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p


def fig5_scatter():
    rng = np.random.default_rng(SEED + 54321)
    n = 120
    x, r = simulate(n, 0.95, -0.7, 0.0, rng)
    x_lag, rr = x[:-1], r[1:]
    b_ols, se, t_ols, _ = ols_fit(x_lag, rr)
    b_sc, _, _ = stambaugh_corrected(x, r)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(x_lag, rr, s=16, color="#5b8ff9", alpha=0.5)
    xs = np.linspace(x_lag.min(), x_lag.max(), 50)
    ax.plot(xs, b_ols * xs, color="#e8684a", lw=2,
            label="OLS slope=%.3f (t=%.2f)" % (b_ols, t_ols))
    ax.plot(xs, b_sc * xs, color="#2ca25f", lw=2,
            label="Stambaugh slope=%.3f" % b_sc)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("x_{t-1} (persistent predictor)")
    ax.set_ylabel("r_t")
    ax.set_title("Spurious predictability in one sample")
    ax.legend(fontsize=9)
    fig.tight_layout(); p = "fig5_spurious_scatter.png"
    fig.savefig(os.path.join(OUT, p), dpi=115); plt.close(fig)
    return p, [float(b_ols), float(b_sc), float(t_ols)]


def main():
    de = exp_delta_bias()
    sz = exp_size()
    nn = exp_sample_size()
    ph = exp_phi_bias()
    f5, f5vals = fig5_scatter()
    r = {
        "setup": {"seed": SEED, "model": "x_t=phi*x_{t-1}+u_t; v_t=delta*u_t+sqrt(1-delta^2)*e_t; r_t=alpha+beta*x_{t-1}+v_t",
                  "estimators": LAB},
        "delta_bias": de,
        "size": sz,
        "sample_size": nn,
        "phi_bias": ph,
        "one_sample": {"ols_slope": f5vals[0], "stambaugh_slope": f5vals[1],
                       "ols_t": f5vals[2]},
    }
    r["figs"] = [fig1_delta(de), fig2_size(sz), fig3_sample(nn),
                 fig4_phi(ph), f5]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


## 八、小结

预测回归里的伪可预测性有两层。浅层是 Stambaugh 有限样本偏误：斜率虽一致，却有符号跟着 $\delta$ 翻转、大小按 $1/n$ 缩放、被持续性放大的系统偏误，解析修正能稳定减半。深层是 t 检验的 size 失真：$\phi=0.98$、$\delta=+0.8$ 时虚警率翻倍到 10.5%，这才是筛选伪因子的真凶，而 bootstrap 偏差校正对它无能为力。本文所有数字来自同一份可复现代码，五张图与逐项统计都在上方——记住那句最关键的：偏误小可修，虚警大要命，两者不是一回事。
