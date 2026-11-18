---
title: "夏普比率的置信区间：i.i.d. 公式在自相关下崩到 79.95%，HAC 与移动块自助谁能救场"
description: "夏普比率的点估计人人会算，置信区间却常被误用 i.i.d. 公式。本文用 AR(1) 收益过程做 2000 次蒙特卡洛，真值 SR0=0.03：无自相关时 i.i.d. 公式覆盖率 0.9495 达标，φ=0.4 时崩到 0.7995，φ=-0.2 时又虚高到 0.9825。Newey-West HAC 稳定在 0.9115-0.9560，移动块自助在 0.9215-0.9455，代价是区间变宽。块长扫描揭示覆盖率与宽度的权衡，BBL=16 达峰 0.921 后回落。40 万日长序列核对 Lo(2002) 时间聚合因子 η(q)：φ=0.4 时 q=12 实测放大 2.363 倍，朴素 √q=3.464 高估 47%。最后复现平滑化虚增：MA(2) 把朴素年化夏普从 0.476 抬到 0.824（虚增 73%），Lo 校正 η(252)=9.18 把 0.824 拉回 0.476。附完整代码、JSON 与五张真实计算图。"
publishDate: '2026-11-18T09:30:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 夏普比率
  - 自助法
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

夏普比率（Sharpe Ratio）是基金业绩评价的第一指标，但绝大多数人只报点估计 $\hat{S}=\hat{\mu}/\hat{\sigma}$，从不给区间。问题在于：一个 252 交易日的样本夏普，抽样噪声大得惊人，标准差可达 0.06 以上——点估计 0.03 的 95% 区间可以宽到 ±0.24。更隐蔽的坑是，**教科书那套 i.i.d. 公式在收益有自相关时会系统性失真**。

本文全部数字来自一段 252 日、真值每期夏普 $SR_0=0.03$ 的 AR(1) 收益蒙特卡洛（2000 次重复），逐一核对三种 95% 置信区间的真实覆盖率：

| 场景 | i.i.d. 公式 | Newey-West HAC | 移动块自助 MBB |
|---|---|---|---|
| φ=0（无自相关） | 0.9495 | 0.9410 | 0.9305 |
| φ=+0.4（强正自相关） | **0.7995** | 0.9115 | 0.9180 |
| φ=-0.2（负自相关） | **0.9825** | 0.9480 | 0.9325 |

三个结论：

1. i.i.d. 公式只在 φ=0 时恰好达标（0.9495≈0.95）；正自相关把它压到 0.7995——**名义 95% 的区间真实只盖住 80%**；负自相关又虚高到 0.9825，区间过度保守。
2. Newey-West HAC 与移动块自助（MBB）在四种 φ、两种新息分布下都把覆盖率稳在 0.91–0.96，代价是 φ=+0.4 时平均区间宽度从 0.247 涨到 0.335–0.351。
3. 时间聚合更危险：收益正自相关时，用 $\sqrt{q}$ 年化会系统性高估。φ=0.4 时 q=12 个月的真实放大因子是 2.363，$\sqrt{12}=3.464$ 高估约 47%；一条经 MA(2) 平滑的 alpha 序列，朴素年化夏普被虚增 73%（0.476→0.824），Lo 的 $\eta(252)=9.18$ 校正把它拉回 0.476。

## 二、i.i.d. 公式为什么失效

Lo（2002）在《The Statistics of Sharpe Ratios》给出 i.i.d. 正态收益下夏普的标准误：

$$SE(\hat{S})\approx\sqrt{\frac{1+\hat{S}^2/2}{n}}$$

本文 $SR_0=0.03$、n=252，代入得 $SE\approx\sqrt{(1+0.0009)/252}\approx0.0630$，于是 95% 区间半宽约 0.247。图 1 把 φ=0 与 φ=0.4 的样本夏普蒙特卡洛分布叠在 i.i.d. 正态密度上：φ=0 时两者吻合，φ=0.4 时实测散布明显更宽、尾部更厚——正态钟形罩不住真实抽样分布，区间自然盖不住真值。

根因是 AR(1) 收益 $r_t=\mu+\phi(r_{t-1}-\mu)+\varepsilon_t$ 存在自相关，样本均值的方差被放大。i.i.d. 公式假设观测独立，把长程自相关带来的额外方差完全忽略。负自相关（φ=-0.2）方向相反：相邻收益互相抵消，样本均值比独立情形更稳，公式反而高估了方差，覆盖率虚高。

![样本夏普的抽样分布与 i.i.d. 正态密度对比](/images/sharpe-ratio-confidence-interval/fig1_sampling_distribution.png)

## 三、三种区间的覆盖率对决

要修复就得让标准误"看见"自相关。两条主流路线：

- **Newey-West HAC**：对 $\hat{S}=\hat{\mu}/\hat{\sigma}$ 做 delta 方法，把矩向量 $g_t=(r_t-\hat{\mu},(r_t-\hat{\mu})^2-\hat{\sigma}^2)$ 的长程协方差用 Bartlett 核加总，带宽 $g=\lfloor4(n/100)^{2/9}\rfloor=4$。
- **移动块自助（Künsch 1989）**：不闭式推方差，而是把序列切成随机起点的重叠块（BBL=10）重新拼接，直接用自助分布的 2.5%/97.5% 分位数做区间。

图 2 是正态新息与 t(5) 新息各四种 φ 下的覆盖率对决。HAC 与 MBB 全程贴近 0.95，i.i.d. 公式随 φ 大幅摆动；厚尾新息（t(5)）下三者略有下降但相对格局不变——说明自相关比非正态对区间覆盖率的威胁更大。

![三种 95% 区间的蒙特卡洛覆盖率对决](/images/sharpe-ratio-confidence-interval/fig2_coverage_battle.png)

移动块自助有个关键调参：块长。图 3 扫描 BBL=2,4,8,16,32,64，覆盖率从 BBL=2 的 0.8615 单调爬到 BBL=16 的 0.9210，之后回落（BBL=64 只剩 0.8425）；同时平均区间宽度从 0.293 涨到 0.353。**块太短留不住自相关结构，块太长则有效重抽样数太少**——这是自助法里没有免费午餐的权衡。

![移动块自助的块长与覆盖率、宽度权衡](/images/sharpe-ratio-confidence-interval/fig3_block_length_tradeoff.png)

## 四、时间聚合：√q 年化的系统性偏差

业内习惯把每期夏普乘 $\sqrt{252}$ 年化，把 q 期收益的夏普乘 $\sqrt{q}$。这只对 i.i.d. 收益成立。收益自相关时，q 期聚合夏普的真实放大因子是 Lo 的

$$\eta(q)=\frac{q}{\sqrt{q+2\sum_{k=1}^{q-1}(q-k)\rho_k}}$$

本文用 40 万日长序列、φ=0.4（实测一阶自相关 $\hat{\rho}_1=0.3997$）核对，图 4 显示实测曲线与 $\eta(q)$ 理论值几乎重合，却与 $\sqrt{q}$ 渐行渐远：q=12 时实测 2.363、理论 2.364，而 $\sqrt{12}=3.464$；q=24 时实测 3.279、理论 3.273，$\sqrt{24}=4.899$。

![时间聚合下 √q 法则与 Lo η(q) 的偏差](/images/sharpe-ratio-confidence-interval/fig4_time_aggregation.png)

## 五、平滑化虚增夏普的现场复现

对冲基金常用低价期权、估值平滑或流动性不足的资产，把净值曲线"熨平"，人为抬高夏普。本文复现这一手法：同一条 i.i.d. alpha，取 MA(2) 滑动平均后，每期夏普从 0.0300 涨到 0.0519；朴素 $\sqrt{252}$ 年化把 0.4759 抬到 0.8241，**虚增 73%**。但真实年化夏普两者都是 0.472（平滑不改变真实收益）。用 Lo 的 $\eta(252)=9.18$（由平滑后一、二阶自相关 $\rho_1=0.667,\rho_2=0.335$ 算出）校正，0.824 被拉回 0.476，与真值吻合。图 5 一目了然。

![平滑化虚增年化夏普与 Lo 校正](/images/sharpe-ratio-confidence-interval/fig5_smoothed_alpha.png)

结论很实际：**看到漂亮的年化夏普，先查收益的自相关结构**。报告区间优于只报点估计，自相关下必须上 HAC 或块自助，涉及不同频率的夏普比较必须用 $\eta(q)$ 而非 $\sqrt{q}$。

## 六、三种方法的取舍与适用边界

把本文三组实验连起来看，可以给出一套可操作的选取规则。i.i.d. 公式并非一无是处：在 φ=0 的正态与 t(5) 新息下，它的覆盖率分别是 0.9495 与 0.9455，几乎正中名义水平，且区间最窄（0.247）、计算只要一行公式。如果你的策略是高频、资金流独立的做市或统计套利，收益近似无自相关，它仍然够用。但只要净值是月度平滑的、持仓有惯性的、或底层资产流动性差，自相关就几乎必然存在——这时 i.i.d. 公式的覆盖率会从 0.95 一路滑到 0.80，你报出的"95% 区间"实际每五次就有一次漏掉真值。

HAC 与 MBB 怎么选？HAC 是闭式解、速度快、可复现性强，适合大批量策略的批量风控扫描；本文负自相关下它甚至略优于 MBB（φ=-0.2 时 0.9480 对 0.9325）。MBB 不依赖 delta 方法的可微条件，对夏普这种比值统计量的非线性、对厚尾都更稳健，但需要对块长做敏感性检查——本文 BBL=16 覆盖率先升后降，说明块长不是越大越好。实务中的稳妥做法是：主口径用 HAC，另取 BBL 在 8–20 之间重算一遍做稳健性对照，两者结论一致才敢下判断。

还有一点常被忽视：置信区间回答的是"真值大概在哪"，不是"策略好不好"。一个区间 [−0.21, 0.27] 的策略，点估计 0.03 看起来为正，但区间跨零，统计上无法拒绝"真实夏普为零"。这时该做的不是挑一个更窄的区间公式自欺，而是承认样本量不够、继续积累数据。本文反复强调覆盖率而非点估计精度，正是因为这个道理——区间方法的全部价值，在于它对真实不确定性的诚实度量。

## 七、完整可运行代码

以下脚本是本文所有数字与五张图的唯一来源，自带数据生成，`python3` 直接运行即可复现 results.json：

```python
# -*- coding: utf-8 -*-
"""夏普比率置信区间：i.i.d. 公式、Newey-West HAC 与移动块自助法的覆盖率对决。

本文全部数字的唯一来源。运行本脚本会：
1. 用 AR(1) 收益过程（φ∈{0,±0.2,0.4}，正态与 t(5) 新息）做蒙特卡洛，
   逐一对拍三种 95% 置信区间的经验覆盖率与平均宽度；
2. 扫描移动块自助法的块长，展示『覆盖率修复 ↔ 区间变宽』的权衡；
3. 在长序列上核对 Lo(2002) 时间聚合因子 η(q) 与朴素 √q 法则的偏差；
4. 现场复现『收益平滑化虚增年化夏普』：同一条 alpha 经 MA(2) 平滑后
   朴素年化虚增近一倍，Lo 的 η(252) 校正把它拉回真值；
5. 画 5 张图到 public/images/sharpe-ratio-confidence-interval/，
   全部统计量写入同目录 results.json。
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261118
_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "sharpe-ratio-confidence-interval",
)
os.makedirs(_OUT, exist_ok=True)

Z975 = stats.norm.ppf(0.975)  # 1.959963985
MU, SIGMA = 0.0003, 0.01      # 日频：真值每期夏普 SR0 = 0.03
N_DAYS = 252
R_ML = 2000                    # 覆盖率蒙特卡洛重复数
B_BOOT = 1500                  # 自助重抽样次数


def _save(fig, name):
    path = os.path.join(_OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def simulate_ar1(rng, phi, n, reps, dist="norm"):
    """生成 reps 条长度 n 的 AR(1) 收益，边际均值 MU、边际标准差 SIGMA。

    r_t = MU + phi*(r_{t-1}-MU) + eps_t, eps 方差取 SIGMA^2*(1-phi^2)，
    平稳解的边际方差恰为 SIGMA^2，故真实每期夏普 = MU/SIGMA 与 phi 无关。
    形状 (reps, n)。
    """
    eps_sd = SIGMA * math.sqrt(1.0 - phi * phi)
    if dist == "norm":
        eps = rng.normal(0.0, eps_sd, size=(reps, n))
    else:  # t(5)：标准化到方差 eps_sd^2
        eps = rng.standard_t(5, size=(reps, n)) / math.sqrt(5.0 / 3.0) * eps_sd
    r = np.empty_like(eps)
    prev = np.full(reps, MU)  # 从平稳均值起步（burn-in 另做）
    burn = int(max(50, 5.0 / max(1e-6, 1.0 - abs(phi))))
    warm = rng.normal(0.0, eps_sd, size=(reps, burn))
    for t in range(burn):
        prev = MU + phi * (prev - MU) + warm[:, t]
    for t in range(n):
        prev = MU + phi * (prev - MU) + eps[:, t]
        r[:, t] = prev
    return r


def sharpe_est(r):
    """每期样本夏普（逐行），sigma 用 ddof=1。"""
    mu = r.mean(axis=1)
    sd = r.std(axis=1, ddof=1)
    return mu / sd


def ci_iid(sr):
    """i.i.d. 正态公式（Lo 2002 Prop.1 特例 / Mertens-Linton）：
    SE = sqrt((1 + SR^2/2)/n)。"""
    se = np.sqrt((1.0 + 0.5 * sr ** 2) / N_DAYS)
    return sr - Z975 * se, sr + Z975 * se


def ci_hac(r, g=None):
    """Newey-West HAC 的 delta 方法区间（Lo 2002 的 GMM 思路实现）。

    对 SR=mu/sigma，梯度 grad=(1/sigma, -SR/(2*sigma^2))（对 (mu, sigma2)），
    矩向量 g_t=((r-mu), (r-mu)^2-sigma2)，长程协方差 Omega 用 Bartlett 核修正。
    """
    n = r.shape[1]
    if g is None:
        g = int(4.0 * (n / 100.0) ** (2.0 / 9.0))  # Newey-West 经验法则
    mu = r.mean(axis=1, keepdims=True)
    m2 = ((r - mu) ** 2).mean(axis=1, keepdims=True)  # sigma2, ddof=0
    sd = np.sqrt(m2)
    sr = (mu / sd).ravel()
    g1 = r - mu
    g2 = (r - mu) ** 2 - m2
    G = np.stack([g1, g2], axis=2)  # (reps, n, 2)
    Omega = np.einsum("rni,rnj->rij", G, G) / n
    for k in range(1, g + 1):
        w = 1.0 - k / (g + 1.0)
        Gk = np.einsum("rni,rnj->rij", G[:, k:], G[:, :-k]) / n
        Omega += w * (Gk + np.transpose(Gk, (0, 2, 1)))
    grad = np.stack([1.0 / sd.ravel(), -sr / (2.0 * m2.ravel())], axis=1)
    var_sr = np.einsum("ri,rij,rj->r", grad, Omega, grad) / n
    se = np.sqrt(np.maximum(var_sr, 1e-18))
    return sr - Z975 * se, sr + Z975 * se, g


def mbb_resample(r, block, rng, reps_boot):
    """移动块自助（Künsch 1989）：随机起点 + 环形拼接，块长 block。

    对每条原始序列做 reps_boot 次自助，逐行收集自助夏普，避免一次性
    物化 (reps, reps_boot, n) 的大数组。返回形状 (reps, reps_boot)。
    """
    n = r.shape[1]
    n_blocks = int(math.ceil(n / block))
    out = np.empty((r.shape[0], reps_boot))
    for i in range(r.shape[0]):
        starts = rng.integers(0, n, size=(reps_boot, n_blocks))
        cols = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
        idx = cols.reshape(reps_boot, n_blocks * block)[:, :n]
        boot = r[i][idx]
        out[i] = boot.mean(axis=1) / boot.std(axis=1, ddof=1)
    return out


def coverage_run(dist):
    """对 4 个 φ 值统计三种区间的覆盖率与平均宽度。"""
    rng = np.random.default_rng(SEED + (0 if dist == "norm" else 7))
    SR0 = MU / SIGMA
    out = {}
    for phi in [0.0, 0.2, 0.4, -0.2]:
        r = simulate_ar1(rng, phi, N_DAYS, R_ML, dist)
        sr = sharpe_est(r)
        lo1, hi1 = ci_iid(sr)
        lo2, hi2, g_used = ci_hac(r)
        rb = mbb_resample(r, 10, np.random.default_rng(SEED + int((phi + 0.5) * 10)), B_BOOT)
        lo3 = np.quantile(rb, 0.025, axis=1)
        hi3 = np.quantile(rb, 0.975, axis=1)
        cov = lambda lo, hi: float(((lo <= SR0) & (SR0 <= hi)).mean())
        wid = lambda lo, hi: float((hi - lo).mean())
        out[f"phi_{phi:+.1f}_{dist}"] = {
            "cov_iid": cov(lo1, hi1), "cov_hac": cov(lo2, hi2), "cov_mbb": cov(lo3, hi3),
            "width_iid": wid(lo1, hi1), "width_hac": wid(lo2, hi2), "width_mbb": wid(lo3, hi3),
            "hac_bandwidth": int(g_used),
            "sr_mean": float(sr.mean()), "sr_sd": float(sr.std(ddof=1)),
        }
    return out


def block_length_scan():
    """φ=0.4 正态新息下扫描 MBB 块长：覆盖率与平均宽度。"""
    rng = np.random.default_rng(SEED + 101)
    r = simulate_ar1(rng, 0.4, N_DAYS, R_ML, "norm")
    SR0 = MU / SIGMA
    res = {}
    for block in [2, 4, 8, 16, 32, 64]:
        rb = mbb_resample(r, block, np.random.default_rng(SEED + 200 + block), B_BOOT)
        lo = np.quantile(rb, 0.025, axis=1)
        hi = np.quantile(rb, 0.975, axis=1)
        res[f"block_{block}"] = {
            "coverage": float(((lo <= SR0) & (SR0 <= hi)).mean()),
            "width": float((hi - lo).mean()),
        }
    return res


def time_aggregation():
    """长序列核对 Lo(2002) 的 η(q)：φ=0.4，q=1..24。"""
    T = 400_000
    phi = 0.4
    rng = np.random.default_rng(SEED + 303)
    r = simulate_ar1(rng, phi, T, 1, "norm")[0]
    sr1 = r.mean() / r.std(ddof=1)
    rho_hat = [float(np.corrcoef(r[:-k], r[k:])[0, 1]) for k in range(1, 30)]
    rows = {}
    for q in range(2, 25):
        nb = T // q
        agg = r[: nb * q].reshape(nb, q).sum(axis=1)
        srq = agg.mean() / agg.std(ddof=1)
        eta_theory = q / math.sqrt(q + 2.0 * sum((q - k) * phi ** k for k in range(1, q)))
        rows[f"q_{q}"] = {
            "sr_q_over_sr_1": float(srq / sr1),
            "eta_theory": float(eta_theory),
            "sqrt_q": float(math.sqrt(q)),
        }
    return {"sr1": float(sr1), "phi": phi, "rho_hat_lag1": rho_hat[0], "rows": rows}


def smoothed_alpha_demo():
    """同一条 i.i.d. alpha 与它的 MA(2) 平滑版：朴素年化夏普 vs Lo η(252) 校正。"""
    T = 400_000
    rng = np.random.default_rng(SEED + 404)
    raw = rng.normal(MU, SIGMA, size=T)
    y = np.convolve(raw, np.ones(3) / 3.0, mode="same")  # MA(2) 平滑
    y = y[5:].copy()  # 去边界
    nb = T // 252
    ann_raw = raw[: nb * 252].reshape(nb, 252).sum(axis=1)
    ann_y = y[: nb * 252].reshape(nb, 252).sum(axis=1)
    sr_raw_p = raw.mean() / raw.std(ddof=1)
    sr_y_p = y.mean() / y.std(ddof=1)
    rho1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    rho2 = float(np.corrcoef(y[:-2], y[2:])[0, 1])
    q = 252
    eta = q / math.sqrt(q + 2.0 * ((q - 1) * rho1 + (q - 2) * rho2))
    return {
        "true_annual_sharpe": float(ann_raw.mean() / ann_raw.std(ddof=1)),
        "true_annual_sharpe_y": float(ann_y.mean() / ann_y.std(ddof=1)),
        "sr_per_raw": float(sr_raw_p), "sr_per_y": float(sr_y_p),
        "rho1_y": rho1, "rho2_y": rho2,
        "naive_annual_raw": float(sr_raw_p * math.sqrt(252)),
        "naive_annual_y": float(sr_y_p * math.sqrt(252)),
        "eta_252": float(eta),
        "lo_corrected_annual_y": float(sr_y_p * eta),
        "inflation_ratio": float((sr_y_p * math.sqrt(252)) / (sr_raw_p * math.sqrt(252))),
    }


def plot1(cov_norm):
    rng = np.random.default_rng(SEED + 505)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, phi in zip(axes, [0.0, 0.4]):
        r = simulate_ar1(rng, phi, N_DAYS, 1200, "norm")
        sr = sharpe_est(r)
        ax.hist(sr, bins=48, density=True, color="#c0392b", alpha=0.65,
                label="样本夏普蒙特卡洛分布")
        se_iid = math.sqrt((1.0 + 0.5 * (MU / SIGMA) ** 2) / N_DAYS)
        xs = np.linspace(sr.min(), sr.max(), 300)
        ax.plot(xs, stats.norm.pdf(xs, MU / SIGMA, se_iid), "k-", lw=2,
                label="i.i.d. 公式正态密度")
        ax.axvline(MU / SIGMA, color="#2c3e50", ls="--", lw=1.5, label=f"真值 SR0={MU/SIGMA}")
        ax.set_title(f"φ = {phi:+.1f}（252 个交易日，1200 次重复）")
        ax.set_xlabel("样本每期夏普")
        ax.legend(fontsize=8)
    fig.suptitle("图1：i.i.d. 公式的钟形对不上实际抽样分布（φ=0.4 时实测散布明显更宽）", fontsize=11)
    return _save(fig, "fig1_sampling_distribution.png")


def plot2(cov_norm, cov_t):
    phis = ["phi_+0.0", "phi_+0.2", "phi_+0.4", "phi_-0.2"]
    labels = ["φ=0", "φ=+0.2", "φ=+0.4", "φ=-0.2"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    width = 0.26
    x = np.arange(4)
    for ax, (cov, dist, dkey) in zip(
        axes, [(cov_norm, "正态新息", "norm"), (cov_t, "t(5) 新息", "t5")]
    ):
        for off, (key, name, color) in enumerate([
            ("cov_iid", "i.i.d. 公式", "#e74c3c"),
            ("cov_hac", "Newey-West HAC", "#2980b9"),
            ("cov_mbb", "移动块自助 BBL=10", "#27ae60"),
        ]):
            vals = [cov[f"{p}_{dkey}"][key] for p in phis]
            ax.bar(x + (off - 1) * width, vals, width, label=name, color=color)
            for xi, v in zip(x + (off - 1) * width, vals):
                ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
        ax.axhline(0.95, color="k", ls="--", lw=1.2)
        ax.text(3.45, 0.955, "名义 95%", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{dist}（真值 SR0=0.03，n=252，2000 次重复）")
        ax.set_ylim(0.6, 1.0)
    axes[0].set_ylabel("95% 区间覆盖率")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("图2：三种 95% 置信区间的蒙特卡洛覆盖率对决", fontsize=11)
    return _save(fig, "fig2_coverage_battle.png")


def plot3(scan):
    blocks = [2, 4, 8, 16, 32, 64]
    covs = [scan[f"block_{b}"]["coverage"] for b in blocks]
    wids = [scan[f"block_{b}"]["width"] for b in blocks]
    fig, ax1 = plt.subplots(figsize=(8.5, 4.4))
    ax1.plot(blocks, covs, "o-", color="#27ae60", lw=2, label="MBB 覆盖率")
    for b, c in zip(blocks, covs):
        ax1.text(b, c + 0.005, f"{c:.3f}", ha="center", fontsize=8, color="#1e8449")
    ax1.axhline(0.95, color="k", ls="--", lw=1.1)
    ax1.set_xlabel("块长 BBL（交易日）")
    ax1.set_ylabel("覆盖率", color="#1e8449")
    ax1.set_ylim(0.7, 1.0)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(blocks)
    ax1.set_xticklabels(blocks)
    ax2 = ax1.twinx()
    ax2.plot(blocks, wids, "s--", color="#8e44ad", lw=2, label="平均区间宽度")
    for b, w in zip(blocks, wids):
        ax2.text(b, w - 0.012, f"{w:.3f}", ha="center", fontsize=8, color="#6c3483")
    ax2.set_ylabel("平均 95% 区间宽度", color="#6c3483")
    ax1.set_title("图3：φ=0.4 时 MBB 的权衡——块长换回覆盖率，但区间同步变宽")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    return _save(fig, "fig3_block_length_tradeoff.png")


def plot4(agg):
    qs = sorted(int(k.split("_")[1]) for k in agg["rows"])
    emp = [agg["rows"][f"q_{q}"]["sr_q_over_sr_1"] for q in qs]
    eta = [agg["rows"][f"q_{q}"]["eta_theory"] for q in qs]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(qs, [math.sqrt(q) for q in qs], "k--", lw=1.8, label="朴素法则 √q")
    ax.plot(qs, eta, "-", color="#2980b9", lw=2.2, label="Lo(2002) 理论 η(q)，φ=0.4")
    ax.plot(qs, emp, "o", ms=5, color="#e74c3c", label="长序列实测 SR(q)/SR(1)")
    for q in [3, 6, 12, 24]:
        ax.annotate(f"{emp[qs.index(q)]:.3f}", (q, emp[qs.index(q)]),
                    textcoords="offset points", xytext=(0, -14), fontsize=8, ha="center")
    ax.set_xlabel("聚合期数 q")
    ax.set_ylabel("q 期夏普 ÷ 每期夏普")
    ax.set_title("图4：收益正自相关时，√q 年化系统性高估（φ=0.4，40 万日样本）")
    ax.legend(fontsize=9)
    return _save(fig, "fig4_time_aggregation.png")


def plot5(demo):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    names = ["原始 i.i.d. 收益\n朴素 √252 年化", "MA(2) 平滑收益\n朴素 √252 年化",
             "MA(2) 平滑收益\nLo η(252) 校正"]
    vals = [demo["naive_annual_raw"], demo["naive_annual_y"], demo["lo_corrected_annual_y"]]
    colors = ["#27ae60", "#e74c3c", "#2980b9"]
    bars = ax.bar(names, vals, 0.55, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    ax.axhline(demo["true_annual_sharpe"], color="k", ls="--", lw=1.4)
    ax.text(2.4, demo["true_annual_sharpe"] + 0.012, f"实测年化真值 {demo['true_annual_sharpe']:.4f}",
            fontsize=8, ha="right")
    ax.set_ylabel("年化夏普")
    ax.set_title("图5：同一条 alpha，平滑定价把朴素年化夏普虚增 73%，Lo 校正拉回真值")
    return _save(fig, "fig5_smoothed_alpha.png")


def main():
    results = {
        "setup": {"mu_daily": MU, "sigma_daily": SIGMA, "sr0": MU / SIGMA,
                  "n_days": N_DAYS, "reps": R_ML, "boot_B": B_BOOT, "seed": SEED},
        "coverage_normal": coverage_run("norm"),
        "coverage_t5": coverage_run("t5"),
        "block_scan": block_length_scan(),
        "time_aggregation": time_aggregation(),
        "smoothed_alpha": smoothed_alpha_demo(),
    }
    results["figs"] = [
        plot1(results["coverage_normal"]),
        plot2(results["coverage_normal"], results["coverage_t5"]),
        plot3(results["block_scan"]),
        plot4(results["time_aggregation"]),
        plot5(results["smoothed_alpha"]),
    ]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()
```

