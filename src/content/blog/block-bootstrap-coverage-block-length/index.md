---
title: "区块自助法的块长不是拍脑袋定的：名义 95% 的夏普置信区间实际只覆盖 83.5%"
description: "回测出一个年化夏普 0.5，顺手用 bootstrap 重抽 250 次给个 95% 置信区间——在 AR(1) 均值加 GARCH(1,1) 波动的真实依赖下，400 条独立路径实测这个区间只有 83.5% 装得住真值，朴素解析标准误也只有 85.0%。本文把 iid、移动区块、循环区块、平稳自助法与 Newey-West HAC 放在同一批路径上对照：MBB 块长 5 天覆盖 0.9200、CBB 块长 10 天 0.9200、HAC NW(5) 0.9275，代价是区间宽度从 3.8779 涨到 5.2532；块长拉到 40 天反而掉回 0.8600。三盆冷水：换成年化均值结论几乎不变，HAC 自己也有带宽且滞后 40 阶掉到 0.8950，样本量从 125 加到 1000 天 iid 覆盖度只从 0.8075 走到 0.7900。附完整可运行代码、结果 JSON 与五张真实计算图。"
publishDate: '2026-11-10T09:10:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - Bootstrap
  - Python
language: zh
difficulty: advanced
---

## 一、把「95%」当成事实之前

回测报告里最常见的一行字是：年化夏普 0.50，bootstrap 95% 置信区间 [−2.1, 3.1]。这个区间看起来严谨，但它默认了一件几乎从不成立的事——每天的收益可以当作独立样本单独重抽。真实收益序列里，均值有自相关，方差有聚集，两者都会让「样本均值 / 样本标准差」的抽样分布比 iid 公式宽得多。把 iid 自助法直接套上去，得到的不是保守估计，而是系统性偏窄的区间。

我用一个真值已知的生成过程把这件事量化。设 $r_t=\mu+u_t$，$u_t=\phi u_{t-1}+\varepsilon_t$，$\varepsilon_t=\sigma_t z_t$，$\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2$，参数 $\mu=0.0004$、$\phi=0.35$、$\omega=5.76\times10^{-6}$、$\alpha=0.08$、$\beta=0.88$。因为 $\varepsilon_t$ 是鞅差，$\mathrm{Var}(\varepsilon)=\omega/(1-\alpha-\beta)=0.000144$，$\mathrm{Var}(u)=\mathrm{Var}(\varepsilon)/(1-\phi^2)=0.00016410$，于是真年化夏普 $\mu/\sqrt{\mathrm{Var}(u)}\cdot\sqrt{252}=0.495681$，真年化均值 $0.1008$。这两个数不靠模拟估，是解析算出来的，所以覆盖度判定没有「用估计值当真值」的循环论证。

每条路径 250 天，重抽 250 次，重复 400 条独立路径。图 1 上方是一条代表性路径，下方是收益与收益平方的自相关：一阶自相关约 0.33（对应 $\phi=0.35$），而平方收益在滞后 4、11 天仍明显越过 iid 95% 带——这正是 $\alpha+\beta=0.96$ 的波动聚集。依赖不只在均值里，也在方差里，这一条决定了后面所有方法的成败。

![AR(1)+GARCH 代表性路径与收益、平方收益的自相关](/images/block-bootstrap-coverage-block-length/fig1_dependence_path.png)

## 二、五种区间，两条解析基线

对照的是四种重抽样加两条解析公式。iid 自助法逐日有放回抽样；移动区块自助法（MBB）把连续 $L$ 天当成一块，块起点只能在 $0..n-L$ 内取，拼接后截到 $n$；循环区块（CBB）允许块起点落在任意位置并按 $n$ 取模，因此每条重抽样本的天数严格等于 $n$；平稳自助法用几何分布 $\mathrm{Geom}(1/L)$ 生成随机块长，是唯一能让重抽序列平稳的方案。四者都用百分位区间（2.5% 与 97.5% 分位）。

解析侧有两条基线：朴素 iid delta 方法 $\mathrm{se}(\hat S)=\sqrt{(1+\hat S_d^2/2)/n}\cdot\sqrt{252}$；以及 Newey-West HAC，用夏普比的影响函数 $\psi_t=(r_t-\mu)/\sigma-\frac{\mu}{2\sigma^3}\big[(r_t-\mu)^2-\sigma^2\big]$ 配 Bartlett 核算长期方差，再乘 $\sqrt{252/n}$。后者不需要重抽，但**同样有一个带宽**：滞后截断阶数 $L$。

平稳自助法的向量化实现值得单说一句：在每个重抽样本里标记新块起点，把「块编号 × n + 随机起点」编码后做逐行累积最大值，就能一次还原当前块的起点和块内偏移，不需要 Python 循环。我对这个实现做了独立校验：目标平均块长 5/10/20/40，实测 4.991/9.978/19.940/39.718，十几万到四十万个块的样本上吻合，说明后面关于平稳自助法的结论不是实现 bug。

还有两处实现取舍值得交代。所有区间都用百分位法，而不是拿重抽标准差去配正态分位数：百分位区间不要求统计量近似正态，能容纳夏普比在小样本下的偏斜；代价是它把点估计自身的偏误一并继承下来，本文记录的自助分布中位数相对点估计的偏移（iid 下 $+0.0044$、MBB $L=10$ 下 $-0.0143$）就是这个偏误的量级。另外 MBB 的块起点被限制在 $0..n-L$ 内、末尾会被截断，CBB 用取模补齐，因此后者每条重抽样本的天数严格等于 $n$；两者在 $L=10$ 时覆盖度 0.9175 与 0.9200 的差距，在本文的蒙特卡洛精度下并不构成显著差异，不必据此挑一个「赢家」。真正需要报告的是整个带宽曲线，而不是单点冠军。

## 三、实测：iid 只有 0.835，最好的也没到 0.95

把 400 条路径、六种块长摊开，结果如下（覆盖度指名义 95% 区间真正装住 0.495681 的比例，宽度是区间平均长度）：

| 方法 | 覆盖度 | 平均宽度 |
| --- | --- | --- |
| iid 自助法 | 0.8350 | 3.8779 |
| 朴素 iid 标准误 | 0.8500 | 3.9442 |
| MBB $L=5$ | 0.9200 | 5.1071 |
| MBB $L=10$ | 0.9175 | 5.2532 |
| MBB $L=40$ | 0.8600 | 4.8886 |
| CBB $L=10$ | 0.9200 | 5.2625 |
| 平稳自助法 $L=10$ | 0.9100 | 5.1186 |
| HAC NW(5) | 0.9275 | 5.2053 |

![夏普置信区间覆盖度与区间宽度随块长的变化](/images/block-bootstrap-coverage-block-length/fig2_coverage_vs_block_length.png)

三个结论。第一，iid 自助法与朴素标准误一起失手：0.8350 和 0.8500 距离名义 0.95 差了 10 个百分点以上，也就是说每六份「显著为正」的报告里就有一份是区间偏窄造成的错觉。第二，区块方法确实修好了大部分，$L=5$ 的 MBB 到 0.9200、$L=10$ 的 CBB 到 0.9200，但**没有一个到 0.95**，HAC NW(5) 的 0.9275 是全场最高——在这个实验里，不重抽的解析方法反而赢了重抽样。第三，修好的代价要明码标价：宽度从 3.8779 涨到 5.2532，扩张 35.5%，原本「显著」的夏普很容易被拉回跨越零点的区间。

块长不是越大越好。图 2 左是明显的倒 U 型：$L=1$ 时 0.8425，$L=5$ 到 0.9200，$L=40$ 又掉回 0.8600。原因很朴素——250 天切成 40 天一块只有 6 个有效块，重抽分布被极少数块的随机性主导，区间方差估计反而更不稳。所以「按自然月取 21 天」这类经验规则没有普适性，它是带宽，不是常数。

## 四、四个容易被误读的地方

**换统计量不会换结论。** 我同时算了年化均值的区间。iid 下夏普 0.8350、均值 0.8375；MBB $L=10$ 下两者都是 0.9175；CBB $L=10$ 下都是 0.9200。在这个参数下 $\sigma$ 的抽样波动远小于 $\mu$ 的噪声，夏普比的非线性没有独立制造额外的覆盖失败。图 3 左把两条曲线画在一起，它们几乎重合——别指望「我算的不是夏普而是均值」能绕开依赖问题。

![均值与夏普区间同步失效，以及 HAC 滞后阶数的带宽效应](/images/block-bootstrap-coverage-block-length/fig3_bandwidth_sensitivity.png)

**HAC 也有自己的旋钮。** 图 3 右把 Newey-West 滞后阶从 1 扫到 40：0.8975 → 0.9200 → 0.9275 → 0.9250 → 0.9250 → 0.8950。滞后太小漏掉长期方差，太大则 Bartlett 核把大量噪声自相关估计塞进来，同样是倒 U 型。任何声称「用 HAC 就不用调参」的说法都站不住。这里所有 HAC 扫描复用主实验的 400 条路径，滞后 5 阶的值与主表 0.9275 完全一致。

**更多数据救不了错误的标准误。** 图 4 把样本量从 125 加到 1000 天：iid 自助法覆盖度 0.8075 → 0.8350 → 0.7750 → 0.7900，没有向 0.95 收敛；朴素标准误 0.8250 → 0.8500 → 0.7750 → 0.7925；MBB $L=10$ 大致稳定在 0.8825～0.9175。与此同时区间宽度从 7.2116 缩到 2.6508。机制是：区间按 $1/\sqrt n$ 变窄，但被忽略的长期方差占比并不随 $n$ 消失——$\alpha+\beta=0.96$ 让 $\psi_t$ 的自相关以 $0.96^k$ 缓慢衰减，固定 5 阶截断永远漏掉同一块尾巴。换句话说，这是设定偏误，不是小样本偏误，堆历史长度不解决问题。

![样本量从 125 到 1000 天的覆盖度与区间宽度](/images/block-bootstrap-coverage-block-length/fig4_sample_size_scan.png)

**肥尾不是主犯，序列依赖才是。** 我把同样的标准正态冲击经 $t(6)$ 分位映射（保序、并标准化到单位方差）重跑一遍，路径逐一配对：iid 0.8350 → 0.8400，MBB 0.9175 → 0.9250，HAC 0.9275 → 0.9250，几乎不动。反过来扫 $\phi$ 就剧烈得多：$\phi=0$ 时 iid 覆盖 0.9525，正好回到名义水平；$\phi=0.6$ 时塌到 0.7025，同期 HAC 仍有 0.9100。图 5 把这两组对照放在一张图里，结论清楚：在本文设定下，把厚尾当成头号敌人是找错了靶子。

![AR(1) 强度扫描与 t(6) 厚尾对照](/images/block-bootstrap-coverage-block-length/fig5_dependence_and_tail_scan.png)

## 五、该怎么做，以及本文的边界

工程上四条：给夏普比做区间之前先看收益与平方收益的自相关，有依赖就不要用 iid 自助法；块长当带宽对待，扫描后**同时**报告覆盖度与区间宽度，别只挑看起来最显著的那档；把 HAC 当作必配的低成本基线，而不是把「我们用了 block bootstrap」当成结论；最后，把「实测覆盖度 0.92 而非 0.95」写进报告，别把修正后的区间继续当精确 95% 用。

方法上还有几处必须说清。所有对照复用同一批随机路径（共同随机数），$n=250$ 那一档在所有扫描里直接复用主实验结果，因此图 2、图 4 上同一个点不会出现两个不同数字；400 条路径下覆盖度的蒙特卡洛标准误约 1.1 个百分点，0.9175 与 0.9275 这类接近值的差距不宜过度解读。本文只覆盖一个合成 DGP：真实市场还有结构突变、杠杆效应、非平稳波动与日内季节性，覆盖度只会更差不会更好；夏普比点估计本身也有小样本偏误，百分位区间并不自动修正它；自助法对极值统计量（最大回撤、最优策略选择）的表现又是另一套问题。想看固定月度区块在 Rank-IC 上的用法可参考 [Bootstrap IC 置信区间](/blog/bootstrap-ic-confidence-interval/)，偏度峰度校正的解析路线见 [概率夏普率 PSR](/blog/probabilistic-sharpe-ratio/)，平稳自助法用于数据窥探检验见 [White Reality Check](/blog/white-reality-check/)，HAC 标准误的推导见 [Newey-West 稳健标准误](/blog/newey-west-robust-se/)；本篇不重复这些内容，只回答「块长与带宽到底怎么影响覆盖度」。

下面的代码就是本文全部实验的源码，现场生成五张图与 `results.json`，可直接运行；文中每个数字都能在结果文件里逐项对上。

```python
"""Block bootstrap CI coverage under serial dependence.

This file IS the runnable code block embedded in the article. Run it directly:

    python3 article_core_block_bootstrap_20261110.py

It writes five real PNG figures and results.json into ARTICLE_OUT (default: cwd).
All numbers quoted in the article come from results.json. No placeholders,
no fabricated curves, no hardcoded statistics.

Dependencies: numpy, scipy, matplotlib.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t as student_t

OUT = Path(os.environ.get("ARTICLE_OUT", ".")).resolve()
SEED = 20261110
ANN = 252.0

# ---------- data generating process ----------
MU = 0.0004
PHI = 0.35
OMEGA = 5.76e-6
ALPHA = 0.08
BETA = 0.88
VAR_EPS = OMEGA / (1.0 - ALPHA - BETA)          # 0.000144
VAR_U = VAR_EPS / (1.0 - PHI ** 2)              # AR(1) driven by an MDS
TRUE_SR = MU / np.sqrt(VAR_U) * np.sqrt(ANN)
TRUE_MEAN = MU * ANN


def simulate(rng, n, phi=PHI, innov="normal", burn=500):
    """AR(1) conditional mean + GARCH(1,1) variance, both with known truth."""
    tot = n + burn
    z = rng.standard_normal(tot)
    if innov == "t6":
        # Same shock ranks as the normal case, mapped through the t(6) quantile
        # and rescaled to unit variance. Keeps the two robustness runs paired.
        z = student_t.ppf(norm.cdf(z), df=6) / np.sqrt(6.0 / 4.0)
    s2 = VAR_EPS
    u = np.empty(tot)
    state = 0.0
    for t in range(tot):
        e = np.sqrt(s2) * z[t]
        state = phi * state + e
        u[t] = state
        s2 = OMEGA + ALPHA * e * e + BETA * s2
    return MU + u[burn:]


# ---------- resampling schemes (all vectorised over B replicates) ----------
def iid_idx(rng, n, B):
    return rng.integers(0, n, size=(B, n))


def mbb_idx(rng, n, B, L):
    """Moving block bootstrap: starts only in 0..n-L, no wrap, truncated to n."""
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(B, k))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, k * L)
    return idx[:, :n]


def cbb_idx(rng, n, B, L):
    """Circular block bootstrap: starts anywhere, wrap around modulo n."""
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, k))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, k * L)
    return idx[:, :n] % n


def stationary_idx(rng, n, B, L):
    """Stationary bootstrap: geometric(p=1/L) block lengths, circular starts.

    Vectorised carry-forward trick: encode (block_id * n + random_start) at
    every block start; because block_id strictly increases, a running maximum
    recovers the current block's start without any Python loop.
    """
    p = 1.0 / L
    new = rng.random((B, n)) < p
    new[:, 0] = True
    pos = np.arange(n)[None, :]
    tag = np.cumsum(new, axis=1)
    rand_start = rng.integers(0, n, size=(B, n))
    encoded = np.where(new, tag * n + rand_start, 0)
    carry = np.maximum.accumulate(encoded, axis=1)
    start = carry % n
    first_pos = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    offset = pos - first_pos
    return (start + offset) % n


def sr_rows(X):
    m = X.mean(axis=1)
    s = X.std(axis=1, ddof=1)
    return m / s * np.sqrt(ANN)


def mean_rows(X):
    return X.mean(axis=1) * ANN


def pct_ci(stat, level=0.95):
    lo, hi = np.quantile(stat, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


# ---------- analytic standard errors ----------
def lrv_nw(psi, L):
    """Newey-West long-run variance with Bartlett kernel."""
    n = len(psi)
    psi = psi - psi.mean()
    out = float(np.dot(psi, psi) / n)
    for k in range(1, L + 1):
        gk = float(np.dot(psi[k:], psi[:-k]) / n)
        out += 2.0 * (1.0 - k / (L + 1.0)) * gk
    return max(out, 1e-14)


def analytic_ci(x, L=5):
    """Return (sr_ci, mean_ci) from naive iid SE and HAC/delta-method SE."""
    n = len(x)
    mu, sd = x.mean(), x.std(ddof=1)
    sr = mu / sd * np.sqrt(ANN)
    # iid delta method: Var(S_daily) = (1 + S_daily^2 / 2) / n
    se_naive_sr = np.sqrt((1.0 + 0.5 * (mu / sd) ** 2) / n) * np.sqrt(ANN)
    se_naive_mean = sd / np.sqrt(n) * ANN
    # influence function of the Sharpe ratio
    psi_sr = (x - mu) / sd - mu / (2.0 * sd ** 3) * ((x - mu) ** 2 - sd ** 2)
    se_hac_sr = np.sqrt(ANN * lrv_nw(psi_sr, L) / n)
    se_hac_mean = np.sqrt(ANN ** 2 * lrv_nw(x, L) / n)
    z = 1.959963984540054
    return {
        "naive_sr": (sr - z * se_naive_sr, sr + z * se_naive_sr),
        "hac_sr": (sr - z * se_hac_sr, sr + z * se_hac_sr),
        "naive_mean": (mu * ANN - z * se_naive_mean, mu * ANN + z * se_naive_mean),
        "hac_mean": (mu * ANN - z * se_hac_mean, mu * ANN + z * se_hac_mean),
    }


def contains(ci, truth):
    return bool(ci[0] <= truth <= ci[1])


def width(ci):
    return float(ci[1] - ci[0])


# ---------- one Monte-Carlo section ----------
L_GRID = [1, 3, 5, 10, 20, 40]
METHODS = ["iid", "mbb", "cbb", "stationary"]


def run_section(section_id, M, B, n, L_values, phi=PHI, innov="normal"):
    """Empirical coverage of 95% CIs over M independent paths.

    Every method/scan uses the same M simulated paths (common random numbers),
    so paired differences between methods are measured with low noise.
    """
    keys = [("iid", 1)]
    for m in ["mbb", "cbb", "stationary"]:
        for L in L_values:
            keys.append((m, L))
    keys += [("naive", 0), ("hac", 5)]
    acc = {k: {"sr": [], "mean": [], "w_sr": [], "w_mean": [], "bias": []} for k in keys}

    for rep in range(M):
        rng_path = np.random.default_rng([SEED, section_id, 0, rep])
        x = simulate(rng_path, n, phi=phi, innov=innov)
        sr_hat = sr_rows(x[None, :])[0]
        for mi, (method, L) in enumerate(keys):
            if method in ("naive", "hac"):
                cis = analytic_ci(x, L=5)
                ci_sr = cis["naive_sr"] if method == "naive" else cis["hac_sr"]
                ci_mean = cis["naive_mean"] if method == "naive" else cis["hac_mean"]
                boot_sr = None
            else:
                rng_b = np.random.default_rng([SEED, section_id, mi + 1, rep])
                if method == "iid":
                    idx = iid_idx(rng_b, n, B)
                elif method == "mbb":
                    idx = mbb_idx(rng_b, n, B, L)
                elif method == "cbb":
                    idx = cbb_idx(rng_b, n, B, L)
                else:
                    idx = stationary_idx(rng_b, n, B, L)
                X = x[idx]
                boot_sr = sr_rows(X)
                boot_mean = mean_rows(X)
                ci_sr = pct_ci(boot_sr)
                ci_mean = pct_ci(boot_mean)
            a = acc[(method, L)]
            a["sr"].append(contains(ci_sr, TRUE_SR))
            a["mean"].append(contains(ci_mean, TRUE_MEAN))
            a["w_sr"].append(width(ci_sr))
            a["w_mean"].append(width(ci_mean))
            a["bias"].append(float(np.median(boot_sr) - sr_hat) if boot_sr is not None else 0.0)

    out = {}
    for k, a in acc.items():
        out[f"{k[0]}_{k[1]}"] = {
            "coverage_sr": round(float(np.mean(a["sr"])), 4),
            "coverage_mean": round(float(np.mean(a["mean"])), 4),
            "width_sr": round(float(np.mean(a["w_sr"])), 4),
            "width_mean": round(float(np.mean(a["w_mean"])), 4),
            "median_bootstrap_bias_sr": round(float(np.mean(a["bias"])), 4),
        }
    return out


def run_hac_scan(M=400, n=250, lags=(1, 3, 5, 10, 20, 40)):
    """The analytic alternative has a bandwidth too. Coverage vs NW lag.

    Reuses main's [SEED, 1, 0, rep] paths so lag=5 matches the main table.
    """
    acc = {L: [] for L in lags}
    naive = []
    for rep in range(M):
        rng = np.random.default_rng([SEED, 1, 0, rep])
        x = simulate(rng, n)
        cis = analytic_ci(x, L=1)
        naive.append(contains(cis["naive_sr"], TRUE_SR))
        for L in lags:
            cis = analytic_ci(x, L=L)
            acc[L].append(contains(cis["hac_sr"], TRUE_SR))
    return {"naive": round(float(np.mean(naive)), 4),
            **{str(L): round(float(np.mean(v)), 4) for L, v in acc.items()}}


def validate_stationary(B=2000, cols=4000):
    """Sanity check: geometric(1/L) blocks really have mean length L."""
    rng = np.random.default_rng([SEED, 8, 8])
    out = {}
    for L in [5, 10, 20, 40]:
        new = rng.random((B, cols)) < 1.0 / L
        new[:, 0] = True
        lens = []
        for b in range(0, B, 4):
            st = np.flatnonzero(new[b])
            lens.extend(np.diff(st).tolist())
        out[str(L)] = {"empirical_mean": round(float(np.mean(lens)), 3),
                       "theoretical_mean": L,
                       "n_blocks_measured": len(lens)}
    return out


def run_all(verbose=True):
    stats = {
        "meta": {
            "seed": SEED,
            "mu": MU, "phi": PHI, "omega": OMEGA, "alpha": ALPHA, "beta": BETA,
            "var_eps": VAR_EPS, "var_u": VAR_U,
            "true_annual_sharpe": round(float(TRUE_SR), 6),
            "true_annual_mean": round(float(TRUE_MEAN), 6),
            "L_grid": L_GRID,
            "main": {"M": 400, "B": 250, "n": 250},
            "n_scan": {"M": 400, "B": 250, "n_values": [125, 250, 500, 1000],
                       "note": "n=250 reuses the main experiment's paths"},
            "phi_scan": {"M": 400, "B": 250, "phi_values": [0.0, 0.2, 0.4, 0.6]},
            "tail_scan": {"M": 400, "B": 250, "innovations": ["normal", "t6"],
                          "note": "t6 is a rank-preserving transform of main's shocks"},
            "hac_scan": {"M": 400, "n": 250, "lags": [1, 3, 5, 10, 20, 40],
                         "note": "reuses main's paths; lag=5 matches main table"},
        }
    }

    main = run_section(1, M=400, B=250, n=250, L_values=L_GRID)
    stats["main"] = main
    if verbose:
        print("[1/5] main L-grid done")
        for k in ["iid_1", "mbb_10", "cbb_10", "stationary_10", "naive_0", "hac_5"]:
            print("   ", k, main[k])

    n_scan = {}
    for n in [125, 250, 500, 1000]:
        if n == 250:
            r = main  # identical paths, identical bootstrap draws, identical numbers
        else:
            r = run_section(2, M=400, B=250, n=n, L_values=[10])
        n_scan[str(n)] = {k: r[k]["coverage_sr"] for k in ["iid_1", "mbb_10", "cbb_10", "stationary_10", "naive_0", "hac_5"]}
        n_scan[str(n)]["width_mbb_10"] = r["mbb_10"]["width_sr"]
    stats["n_scan"] = n_scan
    if verbose:
        print("[2/5] sample-size scan done:", n_scan)

    phi_scan = {}
    for phi in [0.0, 0.2, 0.4, 0.6]:
        r = run_section(3, M=400, B=250, n=250, L_values=[10], phi=phi)
        phi_scan[str(phi)] = {k: r[k]["coverage_sr"] for k in ["iid_1", "mbb_10", "cbb_10", "stationary_10", "naive_0", "hac_5"]}
    stats["phi_scan"] = phi_scan
    if verbose:
        print("[3/5] phi scan done:", phi_scan)

    # t6 innovations are a rank-preserving transform of main's shocks, and use
    # the same [SEED, 1, 0, rep] path seeds: the comparison is exactly paired.
    r_t6 = run_section(1, M=400, B=250, n=250, L_values=[10], innov="t6")
    tail_scan = {
        "normal": {k: main[k]["coverage_sr"] for k in ["iid_1", "mbb_10", "hac_5"]},
        "t6": {k: r_t6[k]["coverage_sr"] for k in ["iid_1", "mbb_10", "hac_5"]},
    }
    stats["tail_scan"] = tail_scan
    if verbose:
        print("[4/5] innovation scan done:", tail_scan)

    hac_scan = run_hac_scan(M=400, n=250)
    stats["hac_scan"] = hac_scan
    stats["stationary_block_length_check"] = validate_stationary()
    if verbose:
        print("[5/5] HAC lag scan done:", hac_scan)
        print("      stationary block-length check:", stats["stationary_block_length_check"])

    # representative path + ACF for figure 1
    rng_rep = np.random.default_rng([SEED, 9, 9])
    rep_path = simulate(rng_rep, 750)
    def acf(v, lags):
        v = v - v.mean()
        den = float(np.dot(v, v))
        return [float(np.dot(v[k:], v[:-k]) / den) if k > 0 else 1.0 for k in lags]
    lags = list(range(0, 21))
    figs = {
        "path": rep_path.tolist(),
        "acf_ret": acf(rep_path, lags),
        "acf_sq": acf(rep_path ** 2, lags),
        "lags": lags,
        "main": main, "n_scan": n_scan, "phi_scan": phi_scan,
        "tail_scan": tail_scan, "hac_scan": hac_scan,
    }
    return stats, figs


# ---------- figures ----------
COL = {"iid": "#888888", "mbb": "#d62728", "cbb": "#1f77b4", "stationary": "#2ca02c"}


def make_figures(figs):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

    # Fig 1: representative path + ACF of returns / squared returns
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.2))
    ax[0].plot(figs["path"], lw=0.7, color="#374151")
    ax[0].axhline(MU, color="#d62728", lw=1, ls="--", label="true mean")
    ax[0].set_title("Representative path: AR(1) mean + GARCH(1,1) variance (750 days)")
    ax[0].set_ylabel("daily return")
    ax[0].legend(frameon=False)
    lags = figs["lags"]
    ax[1].plot(lags, figs["acf_ret"], "o-", ms=4, label="ACF of returns", color="#1f77b4")
    ax[1].plot(lags, figs["acf_sq"], "s-", ms=4, label="ACF of squared returns", color="#d62728")
    ax[1].axhline(0, color="#374151", lw=0.8)
    ax[1].axhline(1.96 / np.sqrt(750), color="#9ca3af", ls=":", lw=1, label="iid 95% band")
    ax[1].axhline(-1.96 / np.sqrt(750), color="#9ca3af", ls=":", lw=1)
    ax[1].set_title("Dependence lives in both the mean and the variance")
    ax[1].set_xlabel("lag (days)")
    ax[1].set_ylabel("autocorrelation")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_dependence_path.png")
    plt.close(fig)

    main = figs["main"]
    Ls = [1, 3, 5, 10, 20, 40]
    # Fig 2: coverage and width vs block length, Sharpe statistic
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for m in ["mbb", "cbb", "stationary"]:
        ax[0].plot(Ls, [main[f"{m}_{L}"]["coverage_sr"] for L in Ls], "o-", ms=5,
                   color=COL[m], label=m.upper())
        ax[1].plot(Ls, [main[f"{m}_{L}"]["width_sr"] for L in Ls], "o-", ms=5,
                   color=COL[m], label=m.upper())
    ax[0].axhline(main["iid_1"]["coverage_sr"], color=COL["iid"], ls="-.", lw=1.4,
                  label="iid bootstrap (no blocks)")
    ax[1].axhline(main["iid_1"]["width_sr"], color=COL["iid"], ls="-.", lw=1.4)
    ax[0].axhline(0.95, color="#111827", ls="--", lw=1.2, label="nominal 0.95")
    ax[0].axhline(main["hac_5"]["coverage_sr"], color="#8b5cf6", ls=":", lw=1.4,
                  label="HAC NW(5)")
    ax[0].axhline(main["naive_0"]["coverage_sr"], color="#f59e0b", ls=":", lw=1.4,
                  label="naive iid SE")
    ax[1].axhline(main["hac_5"]["width_sr"], color="#8b5cf6", ls=":", lw=1.4)
    ax[0].set_xscale("log", base=2)
    ax[1].set_xscale("log", base=2)
    ax[0].set_xticks(Ls)
    ax[0].set_xticklabels([str(L) for L in Ls])
    ax[1].set_xticks(Ls)
    ax[1].set_xticklabels([str(L) for L in Ls])
    ax[0].set_xlabel("block length L (days)")
    ax[1].set_xlabel("block length L (days)")
    ax[0].set_ylabel("empirical coverage of 95% CI")
    ax[1].set_ylabel("mean CI width (annualised Sharpe)")
    ax[0].set_ylim(0.55, 1.0)
    ax[0].set_title("Sharpe CI: coverage vs block length")
    ax[1].set_title("Sharpe CI: the price is interval width")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_coverage_vs_block_length.png")
    plt.close(fig)

    # Fig 3: two bandwidth knobs, neither free; and mean vs Sharpe agreement
    hac = figs["hac_scan"]
    hlags = [1, 3, 5, 10, 20, 40]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(Ls, [main[f"mbb_{L}"]["coverage_sr"] for L in Ls], "o-", ms=5,
               color=COL["mbb"], label="MBB: Sharpe CI")
    ax[0].plot(Ls, [main[f"mbb_{L}"]["coverage_mean"] for L in Ls], "s--", ms=5,
               color=COL["cbb"], label="MBB: annual mean CI")
    ax[0].axhline(0.95, color="#111827", ls="--", lw=1.2, label="nominal 0.95")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(Ls)
    ax[0].set_xticklabels([str(L) for L in Ls])
    ax[0].set_ylim(0.55, 1.0)
    ax[0].set_xlabel("block length L (days)")
    ax[0].set_ylabel("empirical coverage of 95% CI")
    ax[0].set_title("Sharpe and mean CIs fail together")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].plot(hlags, [hac[str(L)] for L in hlags], "o-", ms=5, color="#8b5cf6",
               label="HAC Newey-West")
    ax[1].axhline(0.95, color="#111827", ls="--", lw=1.2, label="nominal 0.95")
    ax[1].axhline(hac["naive"], color="#f59e0b", ls=":", lw=1.4, label="naive iid SE")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks(hlags)
    ax[1].set_xticklabels([str(L) for L in hlags])
    ax[1].set_ylim(0.55, 1.0)
    ax[1].set_xlabel("Newey-West lag truncation L")
    ax[1].set_ylabel("coverage of 95% Sharpe CI")
    ax[1].set_title("The analytic fix has a bandwidth too")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_bandwidth_sensitivity.png")
    plt.close(fig)

    # Fig 4: sample size does not fully cure it
    ns = [125, 250, 500, 1000]
    nsc = figs["n_scan"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for k, c, lab in [("iid_1", "#888888", "iid bootstrap"), ("mbb_10", "#d62728", "MBB L=10"),
                      ("cbb_10", "#1f77b4", "CBB L=10"), ("hac_5", "#8b5cf6", "HAC NW(5)"),
                      ("naive_0", "#f59e0b", "naive iid SE")]:
        ax[0].plot(ns, [nsc[str(n)][k] for n in ns], "o-", ms=5, color=c, label=lab)
    ax[0].axhline(0.95, color="#111827", ls="--", lw=1.2)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(ns)
    ax[0].set_xticklabels([str(n) for n in ns])
    ax[0].set_ylim(0.55, 1.0)
    ax[0].set_xlabel("sample size n (days)")
    ax[0].set_ylabel("coverage of 95% Sharpe CI")
    ax[0].set_title("Longer samples do not rescue a wrong standard error")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].plot(ns, [nsc[str(n)]["width_mbb_10"] for n in ns], "s-", color="#d62728", ms=5, label="MBB L=10 width")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks(ns)
    ax[1].set_xticklabels([str(n) for n in ns])
    ax[1].set_xlabel("sample size n (days)")
    ax[1].set_ylabel("mean CI width")
    ax[1].set_title("CI width shrinks, coverage does not converge")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_sample_size_scan.png")
    plt.close(fig)

    # Fig 5: dependence strength and fat-tailed innovations
    phis = [0.0, 0.2, 0.4, 0.6]
    ps = figs["phi_scan"]
    ts = figs["tail_scan"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for k, c, lab in [("iid_1", "#888888", "iid bootstrap"), ("mbb_10", "#d62728", "MBB L=10"),
                      ("cbb_10", "#1f77b4", "CBB L=10"), ("stationary_10", "#2ca02c", "stationary L=10"),
                      ("hac_5", "#8b5cf6", "HAC NW(5)")]:
        ax[0].plot(phis, [ps[str(p)][k] for p in phis], "o-", ms=5, color=c, label=lab)
    ax[0].axhline(0.95, color="#111827", ls="--", lw=1.2)
    ax[0].set_xlabel("AR(1) coefficient phi")
    ax[0].set_ylabel("coverage of 95% Sharpe CI")
    ax[0].set_ylim(0.55, 1.0)
    ax[0].set_title("Stronger serial dependence, worse coverage")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    labels = ["iid", "MBB L=10", "HAC NW(5)"]
    keys = ["iid_1", "mbb_10", "hac_5"]
    xpos = np.arange(len(labels))
    for off, innov, c, hatch in [(-0.2, "normal", "#94a3b8", ""), (0.2, "t6", "#d62728", "//")]:
        ax[1].bar(xpos + off, [ts[innov][k] for k in keys], width=0.38, color=c,
                  hatch=hatch, edgecolor="white", label=f"innovations: {innov}")
    for xpos_i, k in zip(xpos, keys):
        ax[1].text(xpos_i - 0.2, ts["normal"][k] + 0.008, f"{ts['normal'][k]:.3f}", ha="center", fontsize=8)
        ax[1].text(xpos_i + 0.2, ts["t6"][k] + 0.008, f"{ts['t6'][k]:.3f}", ha="center", fontsize=8)
    ax[1].axhline(0.95, color="#111827", ls="--", lw=1.2)
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylim(0.55, 1.0)
    ax[1].set_ylabel("coverage of 95% Sharpe CI")
    ax[1].set_title("t(6) innovations on top of the same GARCH")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_dependence_and_tail_scan.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats, figs = run_all()
    make_figures(figs)
    (OUT / "results.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nTRUE annualised Sharpe =", round(float(TRUE_SR), 6))
    print("Main results (coverage of nominal 95% CI for Sharpe):")
    for k in ["iid_1", "mbb_3", "mbb_5", "mbb_10", "mbb_20", "mbb_40",
              "cbb_10", "stationary_10", "naive_0", "hac_5"]:
        print(f"  {k:16s} coverage={stats['main'][k]['coverage_sr']:.4f}  width={stats['main'][k]['width_sr']:.4f}")
    print("Wrote 5 PNGs + results.json to", OUT)


if __name__ == "__main__":
    main()
```
