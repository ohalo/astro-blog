---
title: "MMD 两样本检验不是免费午餐：带宽选错，形状漂移上比 KS 还差，median 自动选也踩坑"
description: "回测收益和实盘收益到底是不是同一个分布？很多人第一反应是 MMD——把分布嵌进再生核希尔伯特空间，理论上能抓任意差异。本文在真值已知的五个场景上把 Welch-t、两样本 KS 与 MMD 正面对拍：H0 下七种检验的 size 全部落在 0.036–0.060；纯均值漂移 +0.6 时 Welch-t 是 1.000、MMD(σ=0.5) 只有 0.960；方差漂移 ×2.2 时 Welch-t 彻底失效只有 0.048，而 MMD 全带宽 1.000。真正的反转在形状与尾部：同为均值 0 方差 1 的正态对 Laplace，Welch-t 0.052、KS 0.160，MMD(σ=0.5) 拿到 0.456 看着很美，但把带宽换到 σ=4.0 就塌回 0.040——和随机猜没区别；尾部加厚场景同样从 0.712 掉到 0.028。更隐蔽的坑是 median heuristic：它在方差漂移上给出 1.490 的好带宽，却在形状/尾部场景给出 0.889/0.830 的偏大带宽，功效只剩 0.292/0.656，输给固定小带宽。带宽扫描图把这条倒 U 曲线画得很清楚：形状差异的 peak 在 σ≈0.4，尾部差异在 σ≈0.6，均值漂移却一路爬到 0.996 不回头。结论：MMD 卖的是核把形状差异变成可检测距离的能力，但这份能力完全押在带宽上，没有免费午餐。"
publishDate: '2026-11-20T03:10:00Z'
tags:
  - 量化
  - 交易
  - 统计检验
  - 分布漂移
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

「回测期赚的钱，实盘还能不能赚」这个问题，本质是在问两段收益样本是不是来自同一个分布。均值差不多、波动差不多，就能放心上线吗？不能——均值方差相同、形状完全不同的两个分布，在量化里太常见了：牛市的收益更接近正态，危机期变成尖峰厚尾；同一个因子的 IC 分布在样本内外可能均值不变、尾部频率翻倍。

于是很多人把 MMD（Maximum Mean Discrepancy，最大均值差异）当成万能药：它把每个分布映射到再生核希尔伯特空间的一个平均嵌入点，两个分布的差异就是这两个点的距离，理论上任何分布差异都能被检测。本文用真值已知的蒙特卡洛把这件事钉死，结论有三条，后两条是坏消息：

1. **在均值和方差漂移上，MMD 不比传统方法强，有时还略输。** 纯均值漂移 +0.6 时 Welch-t 的检验功效是 1.000，MMD(σ=0.5) 只有 0.960；方差漂移 ×2.2 时 Welch-t 彻底失效掉到 0.048，但所有 MMD 变体和 KS 都是 1.000。在这两类场景里，MMD 只是「跟上了」，没有超越。
2. **MMD 真正的价值在形状与尾部，但带宽选错会把这份价值全部抹掉。** 同为均值 0、方差 1 的正态对 Laplace，Welch-t 功效 0.052（等于瞎猜），KS 只有 0.160，MMD 用 σ=0.5 能拿到 0.456；可一旦把带宽换成 σ=4.0，功效塌回 0.040，比 KS 还差。尾部加厚场景同样从 0.712 掉到 0.028。**这不是方法失效，是带宽失效——而这两件事在实际使用里几乎无法区分。**
3. **流行的 median heuristic 自动带宽，在它最该起作用的地方系统性偏大。** 它把核带宽设成所有点两两距离的中位数：方差漂移场景下这个值升到 1.490，MMD 功效满格 1.000；但形状场景它给出 0.889、尾部场景 0.830，都比最优带宽大了一倍，功效只剩 0.292 和 0.656，输给固定 σ=0.5 的 0.456 和 0.712。

![图1：五种真值场景——KS/t 只看均值方差，形状与尾部差异藏在分布里](/images/mmd-two-sample-distribution-drift/fig1_scenarios.png)

## 二、MMD 到底在比较什么

给定两个样本 $X=\{x_i\}_{i=1}^m$ 与 $Y=\{y_j\}_{j=1}^n$，MMD 的核心是把每个样本点通过特征映射 $\varphi$ 送进一个高维空间，然后比较两堆映射点的均值：

$$\mathrm{MMD}^2(P,Q)=\|\mu_P-\mu_Q\|_{\mathcal H}^2,\qquad \mu_P=\mathbb E_{x\sim P}[\varphi(x)]$$

这个空间选得足够好（特征核），$\|\mu_P-\mu_Q\|$ 为零当且仅当 $P=Q$。我们不用显式写出 $\varphi$，只用核函数 $k(x,y)=\langle\varphi(x),\varphi(y)\rangle$ 就能算出平方距离。无偏估计是：

$$\widehat{\mathrm{MMD}}_u^2=\frac{1}{m(m-1)}\sum_{i\ne j}k(x_i,x_j)+\frac{1}{n(n-1)}\sum_{i\ne j}k(y_i,y_j)-\frac{2}{mn}\sum_{i,j}k(x_i,y_j)$$

第一项和第二项分别是两个样本内部的「自我相似度」，第三项是跨样本相似度。如果两个分布相同，内部相似度与跨样本相似度应当一致，三者相减接近零；如果分布不同，跨样本相似度会系统性偏低，MMD 变大。本文用 RBF 核 $k(x,y)=\exp(-\|x-y\|^2/(2\sigma^2))$ 与 IMQ 核 $k=1/(1+\|x-y\|^2)$，显著性用 99 次标签置换给出——把两批样本混在一起重新随机分组，看真实 MMD 在置换零分布里排第几。

核带宽 $\sigma$ 决定了「多近才算相似」。σ 太小，只有几乎重合的点才相似，每个点都是自己的小岛，任何两组样本看起来都一样不相似，MMD 抓不到整体差异；σ 太大，所有点都差不多相似，核矩阵趋近常数，只剩均值信息，形状差异被抹平。**这两端夹出一个中间的最优带宽，且最优位置取决于你要抓的是哪种差异**——这正是第三节的倒 U 曲线。

## 三、五场景对决：谁能抓到什么

本文设了五个真值场景，每样本量 120，蒙特卡洛 250 次：H0 两批都是标准正态；均值漂移第二批均值 +0.6；方差漂移第二批标准差 ×2.2；形状漂移第二批换成 Laplace(0, 1/√2)（均值方差都与正态相同）；尾部漂移第二批换成缩放后的 t(3)（方差归一为 1，但尾部更厚）。

先看 H0 下的 size（误报率，应接近 0.05）：Welch-t 0.060、KS 0.036、MMD(σ=0.5) 0.040、MMD(σ=1.5) 0.060、MMD(σ=4.0) 0.056、MMD(median) 0.036、MMD(IMQ) 0.052。七种检验全部落在 0.036–0.060，置换检验没有系统性偏倚，后续的功效对比是在同一校准水平上做的，公平。

![图2：H0 列越靠近虚线越校准；shape/tail 两列才是 MMD 的主场](/images/mmd-two-sample-distribution-drift/fig2_power_table.png)

均值漂移场景，Welch-t 是 1.000 的满分，KS 0.976，MMD 各带宽 0.960–0.996——这是最「无聊」的场景， everyone wins，MMD 没有优势。方差漂移场景出现第一次分化：Welch-t 掉到 0.048，因为它只检验均值、对纯方差变化天生无感；KS 和所有 MMD 变体都是 0.968–1.000。到这里为止，MMD 和 KS 打平。

真正的分化在形状与尾部。形状漂移（正态 vs Laplace）下 Welch-t 0.052、KS 0.160、MMD(σ=0.5) 0.456、MMD(σ=1.5) 0.100、MMD(σ=4.0) 0.040、MMD(median) 0.292、MMD(IMQ) 0.352。尾部漂移（正态 vs t(3)）下 Welch-t 0.028、KS 0.216、MMD(σ=0.5) 0.712、MMD(σ=1.5) 0.280、MMD(σ=4.0) 0.028、MMD(median) 0.656、MMD(IMQ) 0.648。**同一个 MMD 方法，只因带宽不同，功效从 0.712 一路掉到 0.028——跨越了近 26 倍。** 这意味着「我用 MMD 做漂移监控」这句话本身没有任何信息量，必须说清楚用什么核、什么带宽，否则结果不可复现也不可信。

## 四、带宽扫描：倒 U 与单调上升的分野

把 RBF 带宽从 0.15 扫到 20，形状、尾部、均值三个场景的功效曲线走向完全不同。形状场景的 peak 在 σ≈0.4 处达到 0.464，σ=0.6 时还有 0.420，过了 σ=0.9 就掉到 0.236，σ≥4 基本贴着 0.05 的显著性线。尾部场景的 peak 在 σ≈0.6 处达到 0.768，σ=0.4 时 0.680，σ=1.3 掉到 0.456，σ≥2.8 同样归零。两条曲线都是清晰的先升后降（倒 U）。均值漂移场景却是另一副模样：从 σ=0.15 的 0.696 单调爬到 σ=6 的 1.000，之后稳在顶端不下来——因为均值差异是低频信息，带宽越大越只看均值，反而越敏感。

![图3：带宽扫描——形状/尾部差异的 power 先升后降（倒 U），均值漂移却一路走高](/images/mmd-two-sample-distribution-drift/fig3_bandwidth.png)

这解释了一个反直觉现象：如果你用 MMD 同时监控均值、方差、形状三类漂移，没有一个带宽能通吃。大带宽对均值最敏感、对形状最迟钝；小带宽反过来。median heuristic 想用一个自动规则同时照顾所有情况，但它算出的中位数距离本质上是数据的整体尺度——方差一大它就跟着变大（方差场景 1.490），形状变了它反而变小（形状场景 0.889、尾部 0.830），恰好把最需要小带宽的场景推向大带宽。这就是它在形状/尾部上输给固定 σ=0.5 的原因，不是随机波动，是规则本身的偏置方向错了。

样本量扫描给出一致性证据：形状漂移下 MMD(median) 的功效从 n=40 的 0.120 爬到 n=640 的 0.952，KS 从 0.068 爬到 0.836，Welch-t 全程在 0.032–0.072 之间横着走——它根本看不见这个差异，加再多样本也没用。MMD 的 double-digit 优势在 n=320 时最明显（0.744 vs 0.488），到 n=640 时 KS 也追上来（0.836 vs 0.952），因为样本够多时经验 CDF 的分辨率足以分辨这种形状差异。

![图4：同均值同方差的形状漂移——只有 MMD 随样本量爬起来](/images/mmd-two-sample-distribution-drift/fig4_sample_size.png)

## 五、实践建议与边界

![图5：RBF 核矩阵——MMD 比较的是两块平均嵌入的距离](/images/mmd-two-sample-distribution-drift/fig5_kernel_matrix.png)

要把 MMD 用在漂移监控上，本文的建议是按「你怕什么」选带宽，别指望一个配置打天下。怕尾部变厚和形状改变，用接近个体点尺度的 bandwidth（本实验 σ≈0.4–0.6，约为 median distance 的一半）；怕均值漂移，直接用 Welch-t，又快又准，没必要上核方法；方差漂移用 KS 或 MMD 都行。如果你确实需要同时覆盖多种漂移，最稳妥的做法是**多带宽并行检验 + Bonferroni 或 e-process 校正**，而不是寄希望于 median heuristic 一步到位——本文数据显示后者在形状场景上把功效从 0.456 压到 0.292，代价是实打实的。

必须说清的边界：本文所有结论都建立在每样本 120 个观测、99 次置换、5% 显著性、一维高斯族及其变体上。换到高维时核距离会集中（distance concentration），RBF 核需要按维度缩放带宽，否则所有点两两距离趋同，MMD 直接失效；换成强相关的时间序列样本时，置换检验的「可交换性」假设被破坏，误报率会升高，需要改用块置换或 HAC 版本的渐进方差。这些都没有在本文实验里覆盖，不能把「一维独立样本上 MMD 带宽很关键」外推成「任何场景都必须这么做」。

## 六、可运行代码

下面这段代码就是生成本文全部数字与五张图的完整实现：五种真值场景、七种检验、99 次标签置换、带宽扫描、样本量扫描都在里面，直接运行即可复现，数字写入 `results.json` 与图同目录。核心是用核矩阵的索引重算做置换——合并样本算一次核矩阵，置换时只重排行列索引，避免每次重新算距离。

```python
# -*- coding utf-8 -*-
"""最大均值差异 MMD 两样本检验：带宽决定功效，同均值同方差的形状漂移才是它的主场。

本脚本是文章全部数字的唯一来源。运行它会：
1. 在真值已知的五个场景上对决 Welch-t / 两样本 KS / MMD(RBF 三带宽 + median heuristic + IMQ)；
2. 场景包含 H0 校准、纯均值漂移、纯方差漂移、同均值同方差换族(Laplace)、尾部加厚(t3)；
3. 用置换检验给 MMD 算 p 值，报告 size 与各备择 power；
4. 扫描 RBF 带宽看 power 的 U 型曲线，扫描样本量看一致性；
5. 画 5 张图到 public/images/mmd-two-sample-distribution-drift/，统计写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261120
N_PER = 120          # 每样本量
REPS = 250           # 蒙特卡洛重复
B_PERM = 99          # 置换次数
ALPHA = 0.05
SIGMAS = [0.5, 1.5, 4.0]   # 三档 RBF 带宽
TAIL_SCALE = 1.0 / np.sqrt(3.0)   # t(3) 缩放至单位方差

_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "mmd-two-sample-distribution-drift",
)
os.makedirs(_OUT, exist_ok=True)


def rbf_k(a, b, sigma):
    d2 = cdist(a, b, "sqeuclidean")
    return np.exp(-d2 / (2.0 * sigma ** 2))


def imq_k(a, b, c=1.0):
    d2 = cdist(a, b, "sqeuclidean")
    return 1.0 / (1.0 + d2 / c)


def mmd_u_from_K(K, m, n):
    """无偏 MMD^2：K 为合并 (m+n) 方阵，前 m 行是 X。"""
    A = K[:m, :m].sum() - np.trace(K[:m, :m])
    Bv = K[m:, m:].sum() - np.trace(K[m:, m:])
    C = K[:m, m:].sum()
    return A / (m * (m - 1)) + Bv / (n * (n - 1)) - 2.0 * C / (m * n)


def perm_test(x, y, kern, n_perm, rng):
    """置换检验：合并后随机重排标签，用同一 K 的索引重算 MMD^2。返回 (MMD², p)。"""
    m, n = len(x), len(y)
    N = m + n
    z = np.vstack([x, y])
    K = kern(z, z)
    obs = mmd_u_from_K(K, m, n)
    perms = np.argsort(rng.random((n_perm, N)), axis=1)
    idx = perms[:, :m]
    idx2 = perms[:, m:]
    A = K[idx[:, :, None], idx[:, None, :]].sum(axis=(1, 2)) - K[idx, idx].sum(axis=1)
    Bv = K[idx2[:, :, None], idx2[:, None, :]].sum(axis=(1, 2)) - K[idx2, idx2].sum(axis=1)
    # total_offdiag - A - Bv = 2*cross_sum，故 -2*cross/(mn) = -(total_offdiag - A - Bv)/(mn)
    null = A / (m * (m - 1)) + Bv / (n * (n - 1)) - (K.sum() - np.trace(K) - A - Bv) / (m * n)
    p = (1.0 + np.sum(null >= obs - 1e-12)) / (n_perm + 1.0)
    return float(obs), float(p)


def draw(rng, scenario, n):
    if scenario == "h0":
        return rng.standard_normal(n).reshape(-1, 1), rng.standard_normal(n).reshape(-1, 1)
    if scenario == "mean":
        return rng.standard_normal(n).reshape(-1, 1), (rng.standard_normal(n) + 0.6).reshape(-1, 1)
    if scenario == "var":
        return rng.standard_normal(n).reshape(-1, 1), (rng.standard_normal(n) * 2.2).reshape(-1, 1)
    if scenario == "shape":
        # 同为均值 0、方差 1：Laplace 尺度 1/sqrt(2)
        return rng.standard_normal(n).reshape(-1, 1), (rng.laplace(0, 1 / np.sqrt(2), n)).reshape(-1, 1)
    if scenario == "tail":
        return rng.standard_normal(n).reshape(-1, 1), (rng.standard_t(3, n) * TAIL_SCALE).reshape(-1, 1)
    raise ValueError(scenario)


SCENARIOS = ["h0", "mean", "var", "shape", "tail"]


def run_power_table():
    """返回每个场景下各检验的拒绝率；MMD 用固定 σ 与 median heuristic。"""
    rng = np.random.default_rng(SEED)
    methods = ["welch_t", "ks", "mmd_s0.5", "mmd_s1.5", "mmd_s4.0", "mmd_med", "mmd_imq"]
    out = {s: {k: 0 for k in methods} for s in SCENARIOS}
    meds = {s: [] for s in SCENARIOS}
    for s in SCENARIOS:
        for _ in range(REPS):
            x, y = draw(rng, s, N_PER)
            out[s]["welch_t"] += stats.ttest_ind(x.ravel(), y.ravel(), equal_var=False).pvalue <= ALPHA
            out[s]["ks"] += stats.ks_2samp(x.ravel(), y.ravel()).pvalue <= ALPHA
            z = np.vstack([x, y])
            Dmed = np.median(cdist(z, z, "euclidean")[np.triu_indices(len(z), 1)])
            meds[s].append(Dmed)
            for sig in SIGMAS:
                _, p = perm_test(x, y, lambda a, b, sg=sig: rbf_k(a, b, sg), B_PERM, rng)
                out[s][f"mmd_s{sig}"] += p <= ALPHA
            _, p = perm_test(x, y, lambda a, b, sg=Dmed: rbf_k(a, b, sg), B_PERM, rng)
            out[s]["mmd_med"] += p <= ALPHA
            _, p = perm_test(x, y, imq_k, B_PERM, rng)
            out[s]["mmd_imq"] += p <= ALPHA
        for k in methods:
            out[s][k] = out[s][k] / REPS
    med_stat = {s: {"median": float(np.median(meds[s])), "q25": float(np.quantile(meds[s], .25)),
                    "q75": float(np.quantile(meds[s], .75))} for s in SCENARIOS}
    return out, med_stat


def bandwidth_scan():
    """形状漂移场景下，RBF 带宽从 0.1 到 20 扫描 power，展示 U 型。"""
    rng = np.random.default_rng(SEED + 1)
    sigmas = [0.15, 0.25, 0.4, 0.6, 0.9, 1.3, 1.9, 2.8, 4.0, 6.0, 9.0, 14.0, 20.0]
    powers = {s: [] for s in ["shape", "mean", "tail"]}
    for sig in sigmas:
        for s in powers:
            cnt = 0
            for _ in range(REPS):
                x, y = draw(rng, s, N_PER)
                _, p = perm_test(x, y, lambda a, b, sg=sig: rbf_k(a, b, sg), B_PERM, rng)
                cnt += p <= ALPHA
            powers[s].append(cnt / REPS)
    return {"sigmas": sigmas, "power": powers}


def sample_scan():
    """样本量 n 扫描：形状漂移下 MMD(median) / KS / Welch-t 的 power。"""
    rng = np.random.default_rng(SEED + 2)
    ns = [40, 80, 160, 320, 640]
    res = {"n": ns, "mmd": [], "ks": [], "welch": []}
    for n in ns:
        c_mmd = c_ks = c_t = 0
        for _ in range(REPS):
            x, y = draw(rng, "shape", n)
            z = np.vstack([x, y])
            Dmed = np.median(cdist(z, z, "euclidean")[np.triu_indices(len(z), 1)])
            _, p = perm_test(x, y, lambda a, b, sg=Dmed: rbf_k(a, b, sg), min(B_PERM, 99), rng)
            c_mmd += p <= ALPHA
            c_ks += stats.ks_2samp(x.ravel(), y.ravel()).pvalue <= ALPHA
            c_t += stats.ttest_ind(x.ravel(), y.ravel(), equal_var=False).pvalue <= ALPHA
        res["mmd"].append(c_mmd / REPS)
        res["ks"].append(c_ks / REPS)
        res["welch"].append(c_t / REPS)
    return res


def plot1():
    rng = np.random.default_rng(SEED + 3)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.1))
    labels = {"h0": "H0：同分布", "mean": "均值漂移 +0.6", "var": "方差漂移 ×2.2",
              "shape": "同均值方差换族", "tail": "尾部加厚 t(3)"}
    grid = np.linspace(-5, 5, 300)
    for ax, s in zip(axes, SCENARIOS):
        x, y = draw(rng, s, 4000)
        ax.hist(x.ravel(), bins=60, density=True, alpha=.55, color="#2980b9", label="X")
        ax.hist(y.ravel(), bins=60, density=True, alpha=.55, color="#c0392b", label="Y")
        ax.set_title(labels[s], fontsize=10)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=8)
    fig.suptitle("图1：五种真值场景——KS/t 只看均值方差，形状与尾部差异藏在分布里", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .92])
    p = os.path.join(_OUT, "fig1_scenarios.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig1_scenarios.png"


def plot2(power, med):
    methods = ["welch_t", "ks", "mmd_s0.5", "mmd_s1.5", "mmd_s4.0", "mmd_med", "mmd_imq"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = .12
    base = np.arange(len(SCENARIOS))
    colors = ["#7f8c8d", "#2980b9", "#e67e22", "#f39c12", "#d35400", "#27ae60", "#8e44ad"]
    for i, (k, c) in enumerate(zip(methods, colors)):
        vals = [power[s][k] for s in SCENARIOS]
        ax.bar(base + (i - 3) * w, vals, w, label=k, color=c)
    ax.axhline(ALPHA, color="k", ls="--", lw=1)
    ax.text(-.45, ALPHA + .012, "5% 显著性", fontsize=8)
    ax.set_xticks(base); ax.set_xticklabels(SCENARIOS)
    ax.set_ylabel("拒绝率（size / power）"); ax.set_ylim(0, 1.05)
    ax.set_title("图2：H0 列越靠近虚线越校准；shape/tail 两列才是 MMD 的主场")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_power_table.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig2_power_table.png"


def plot3(scan):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for s, c in [("shape", "#27ae60"), ("mean", "#c0392b"), ("tail", "#8e44ad")]:
        ax.semilogx(scan["sigmas"], scan["power"][s], "o-", color=c, lw=2, label=s)
    ax.axhline(ALPHA, color="k", ls="--", lw=1)
    ax.set_xlabel("RBF 带宽 σ"); ax.set_ylabel("power（拒绝率）")
    ax.set_title("图3：带宽扫描——形状/尾部差异的 power 先升后降（倒 U），均值漂移却一路走高")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_bandwidth.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig3_bandwidth.png"


def plot4(scan):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(scan["n"], scan["mmd"], "o-", color="#27ae60", lw=2, label="MMD(median)")
    ax.plot(scan["n"], scan["ks"], "s-", color="#2980b9", lw=2, label="KS")
    ax.plot(scan["n"], scan["welch"], "^--", color="#c0392b", lw=2, label="Welch-t")
    ax.axhline(ALPHA, color="k", ls=":", lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(scan["n"]); ax.set_xticklabels(scan["n"])
    ax.set_xlabel("每样本量 n"); ax.set_ylabel("power")
    ax.set_title("图4：同均值同方差的形状漂移——只有 MMD 随样本量爬起来")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_sample_size.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig4_sample_size.png"


def plot5():
    rng = np.random.default_rng(SEED + 4)
    z = np.vstack([rng.standard_normal(N_PER), rng.laplace(0, 1/np.sqrt(2), N_PER)])
    K = rbf_k(z, z, 1.5)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(K, cmap="viridis")
    ax.axvline(N_PER - .5, color="red", lw=1.5); ax.axhline(N_PER - .5, color="red", lw=1.5)
    ax.text(N_PER/2, N_PER + 8, "Y (Laplace)", color="white", ha="center", fontsize=9)
    ax.text(3*N_PER/2, N_PER + 8, "Y", color="white", ha="center", fontsize=9)
    ax.text(N_PER/2, N_PER - 8, "X (Normal)", color="white", ha="center", fontsize=9)
    ax.set_title("图5：RBF 核矩阵——MMD 比较的是两块平均嵌入的距离")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_kernel_matrix.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig5_kernel_matrix.png"


def main():
    power, med = run_power_table()
    scan = bandwidth_scan()
    sscan = sample_scan()
    results = {
        "setup": {"seed": SEED, "n_per": N_PER, "reps": REPS, "n_perm": B_PERM,
                  "alpha": ALPHA, "sigmas": SIGMAS, "tail_scale": TAIL_SCALE},
        "power": power,
        "median_distance": med,
        "bandwidth_scan": scan,
        "sample_scan": sscan,
    }
    results["figs"] = [plot1(), plot2(power, med), plot3(scan), plot4(sscan), plot5()]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

