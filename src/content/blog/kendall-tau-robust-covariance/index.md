---
title: "Kendall tau 稳健相关：1% 的数据污染就能毁掉 Pearson 相关矩阵"
description: "相关矩阵是所有组合优化的输入，却几乎没人检查它有多脆。本文在真值已知的合成面板上现场对照四种估计量：干净高斯下 Pearson 相对 Frobenius 误差 0.1741 最优，Kendall 变换 0.1858 略逊；只要掺进 1% 的离群样本，Pearson 误差跳到 0.2675 而 Kendall 仍是 0.1897，10% 污染时变成 0.6224 对 0.1998。厚尾、无偏性、GMV 样本外波动与高维正定性压力测试一并给出：矩阵误差差三倍，组合波动只差 1.7%；而 p=40、n=60 时逐元素 sin 变换让 Kendall 矩阵 100% 非正定。附五张真实计算图与完整可运行代码。"
publishDate: '2026-11-08'
tags:
  - 量化
  - 投资
  - 风险管理
  - Python
language: zh
difficulty: advanced
---

## 一、组合优化最脆的一块输入

均值-方差优化、风险平价、Barra 归因、因子协方差收缩，入口都是同一个东西：相关矩阵或协方差矩阵。它通常被当成一个「算出来就行」的中间件——`np.corrcoef` 一行搞定，然后直接喂进优化器。但真实数据里从来不缺脏东西：财报单位错位、复权因子跳变、停牌补零、涨跌停期间的异常成交、跨市场时间戳没对齐。哪怕只有百分之一的样本被污染，Pearson 相关也会因为平方项被单点拉走，而且这种偏差是**系统性向下**的，不是噪声。

更麻烦的是它不会报错。矩阵照样对称、对角线照样是 1，优化器照样给出一组看起来很合理的权重，直到某天某个极端行情把隐藏的错配暴露出来。

本文把四种相关矩阵估计量放在真值已知的同一张实验台上对拍，把「稳健」这件事量化到具体数字，并且专门查两个容易被跳过的问题：秩相关的正弦变换在厚尾下还无偏吗？逐元素变换后的矩阵还正定吗？

结论先行：**1% 污染就足以让 Pearson 的相关矩阵误差上涨 54%，而 Kendall 变换几乎不动；但干净数据下秩方法要付约 6.7% 的效率代价，且高维小样本时逐元素变换会破坏正定性。没有免费的稳健。**

## 二、四种估计量与两个变换

设 `X` 是 `n × p` 的收益率面板。四种做法：

- **Pearson**：`np.corrcoef(X)`，直接用原始值的线性相关。
- **Winsorized**：每列先裁剪到自身 1% / 99% 分位数，再算 Pearson。最朴素的稳健化。
- **Spearman**：先取秩，算秩的 Pearson 相关 `ρ_s`，再用 `2 sin(π ρ_s / 6)` 还原成 Pearson 尺度。
- **Kendall**：用一致对与不一致对算 Kendall tau `τ`，再用 `sin(π τ / 2)` 还原。

两个变换不是经验拟合，而是椭圆分布族（多元正态、多元 t、混合正态等）下的解析关系，见 Lindskog、McNeil 与 Schmock 2003 年关于椭圆分布 Kendall tau 的工作；秩相关本身的源头是 Spearman 1904 与 Kendall 1938。用它们的原因是：如果直接把 `τ` 当相关系数塞进协方差矩阵，量纲就不对了（同样 `ρ=0.5` 时 `τ` 只有约 0.333），后续所有比较都会失真。

Kendall tau 的现场计算用了一个向量化写法：先取列秩矩阵 `R`，构造三维符号张量 `D[a,b,i] = sign(R[b,i] − R[a,i])`，再用 `einsum('abi,abj->ij')` 一次算出所有资产对的 `2(C−D)`，除以 `n(n−1)` 即得完整的 tau 矩阵。本文把它与 `scipy.stats.kendalltau` 逐元素对拍过，最大绝对差 1.4e-17。这比双重循环快一个数量级，也让 120 次重复实验能在十秒内跑完。

## 三、实验台

真实结构是等相关矩阵：10 个资产，两两真实相关系数 `ρ = 0.35`，日波动 `σ = 0.015`。等相关不是偷懒——它让最小方差组合的理论最优权重恰好是等权，于是有一个可解析的 oracle 基准，任何估计误差带来的损失都能直接量出来。

三类生成机制：标准高斯；自由度 3 的多元 t（用同一个卡方变量缩放整行，保持相关结构但制造厚尾）；以及高斯加污染——按概率 `ε` 抽中若干行，随机选两列各加一个 `±8σ` 的冲击。主实验 `n=200`、`p=10`、每个配置重复 120 次，`ε` 扫 0、1%、2%、5%、10%。

评价指标有三个层次：相对 Frobenius 误差（只看非对角块，衡量矩阵本身准不准）、非对角平均偏差（看是不是系统性低估）、以及 GMV 组合在**真实协方差**下的样本外年化波动（衡量误差最终有没有变成真金白银的损失）。估计矩阵非正定时先做特征值截断投影再求权重。

## 四、五张图逐一读

![](/images/kendall-tau-robust-covariance/fig1_scatter_dgps.png)

图 1 是三种机制下前两个资产的散点，真实相关系数三者完全相同。左图高斯点云紧凑；中图 t(3) 出现明显的长尾；右图 5% 污染把几个点甩到角落，主体的点云反而被坐标轴压缩得看起来更「圆」。这正是 Pearson 的死穴——那几个角落里的点带平方权重，足以把 0.35 拉到接近 0 甚至负数。

![](/images/kendall-tau-robust-covariance/fig2_frobenius_vs_contamination.png)

图 2 是主结果。干净数据下 Pearson 误差 0.1741 最低，Winsorized 0.1743，Spearman 0.1860，Kendall 0.1858——秩方法确实要付约 6.7% 的效率代价，因为丢弃了数值大小信息，这是诚实的第一笔账。

但 `ε` 一到 0.01，Pearson 就跳到 0.2675（涨 53.6%），Kendall 只从 0.1858 走到 0.1897。之后 Pearson 一路 0.3399 → 0.4669 → 0.6224，Kendall 稳在 0.1875 → 0.1891 → 0.1998。10% 污染时两者差 3.1 倍。非对角平均偏差更能说明方向：Pearson 在 `ε=0.10` 时是 −0.1897，Kendall 是 −0.0226——**污染不是让估计变 noisy，是让它系统性地把相关性往零压**，而组合优化会因此误判分散化收益。

Winsorized 的表现值得单独说：1% 和 2% 污染时它很好（0.1815 / 0.1867），5% 时退化到 0.2491，10% 时崩到 0.4554，甚至比 Pearson 在 5% 时还差。因为裁剪比例写死在 1%/99%，污染率一旦超过裁剪能力，离群点就原样留在数据里；同时裁剪又削掉了真实尾部，偏差 −0.1263。它不是一个自适应的稳健估计量，只是一个便宜的补丁。

![](/images/kendall-tau-robust-covariance/fig3_sin_transform_unbiasedness.png)

图 3 检验两个正弦变换在真 `ρ` 从 0 到 0.8 时能否还原。高斯下三者几乎重合：真 0.8 时 Pearson 0.8014、Spearman 0.7987、Kendall 0.8006，`sin(πτ/2)` 的无偏性得到验证。厚尾 t(3) 下出现分化：真 0.8 时 Kendall 仍给 0.7977，Spearman 掉到 0.7779，Pearson 0.7900。理论上 t(3) 同属椭圆分布族、Kendall 变换应无偏，可见的差距来自有限样本下方差与变换的非线性；但 Spearman 的偏差明显更大，说明厚尾场景下 Kendall 是更稳的那个。这一条也提醒：图 2 里 Kendall 与 Spearman 贴近，不代表两者可以互换。

![](/images/kendall-tau-robust-covariance/fig4_gmv_oos_volatility.png)

图 4 是本文最想让人冷静的一张。把估计矩阵拿去解 GMV 权重，再放到**真实**协方差下算年化波动：`ε=0.05` 时 Pearson 0.157916、Winsorized 0.156092、Spearman 0.155233、Kendall 0.155247，而 oracle 等权是 0.153397。差距最大也只有 0.0045，约 2.9%；相对最优解，Kendall 只比 Pearson 好 1.7%。

矩阵 Frobenius 误差差了 2.5 倍，组合波动却几乎没差。原因藏在实验设计里：等相关结构下最优权重就是等权，任何相关矩阵的扰动主要改变的是**权重的方向**，而这个方向在等相关时恰好无所谓。还有个反直觉的细节：Pearson 在 `ε=0.10` 的波动 0.156642 反而低于 `ε=0.05` 的 0.157916——污染把相关性整体压低，误打误撞让权重更接近等权。**所以「相关矩阵更准」不能被直接翻译成「组合业绩更好」，这是本文明确不主张的一件事。** 真正的损失会出现在异质相关、有约束、或者用到逆矩阵其他部分的场景（条件相关、风险归因、协方差稀疏化），那里没有等权这层保护。

![](/images/kendall-tau-robust-covariance/fig5_pn_scan_and_psd.png)

图 5 左半是 `p ∈ {6,10,20}`、`n ∈ {60,120,250,500}` 在 5% 污染下的误差扫描。Kendall 在所有格点上都优于 Pearson：`p=20, n=60` 是 0.2391 对 0.3614；`p=20, n=500` 是 0.1196 对 0.2846。注意 Pearson 的误差随 `n` 增大下降得很慢——污染比例固定时，更多样本也带来更多离群点，估计量不相合，加大样本救不了它。

右半是正定性压力测试，也是本文第二处「不能省的检查」。主实验用等相关矩阵，逐元素做完 `sin(πτ/2)` 之后仍是等相关，结构上必然正定，所以 120 次重复里非正定率是 0.000——这个零没有信息量。换成 `p=40` 的异质相关矩阵（低秩因子加个体差异）后，边界立刻现形：`n=30` 时 `n<p`，四种方法全部非正定，这是样本秩不足，与变换无关；关键在于 `n=60`（已经 `n>p`），Pearson 非正定率降到 0.000，Spearman 仍有 0.975、Kendall 仍有 1.000，Kendall 矩阵平均最小特征值 −0.0493。到 `n=120` 才全部恢复（平均最小特征值 0.0615），`n=250` 时 0.1401。

原因是逐元素非线性变换不保持矩阵的锥结构：它保证每个元素落在 `[-1,1]`，但不保证三元组的传递性。所以**高维小样本下用秩相关，必须显式做 nearest-PSD 投影或改用 shrinkage**，不能直接求逆。本文 GMV 部分已经内置了特征值截断，主实验之所以没触发，只是因为 `p=10`、等相关。

## 五、数字汇总

| 场景 | Pearson | Winsorized | Spearman | Kendall |
|------|---------|-----------|----------|---------|
| 干净高斯，Frobenius | **0.1741** | 0.1743 | 0.1860 | 0.1858 |
| 1% 污染 | 0.2675 | 0.1815 | 0.1907 | **0.1897** |
| 5% 污染 | 0.4669 | 0.2491 | 0.1900 | **0.1891** |
| 10% 污染 | 0.6224 | 0.4554 | 0.2017 | **0.1998** |
| t(3) 厚尾 | 0.4277 | 0.2674 | 0.2080 | **0.2081** |
| GMV 波动 @5% | 0.157916 | 0.156092 | **0.155233** | 0.155247 |

最后一行 Spearman 与 Kendall 差 0.000014，在 120 次重复的误差范围内，不应解读成谁更优。oracle 波动 0.153397。

## 六、边界与工程建议

第一，本实验的污染是**独立地打在随机行和随机列上**的。真实数据里的错误往往是结构性的（整段时间戳错位、某个数据源全错、同一板块同时异常），秩相关对这类成片污染的抵抗力会明显下降，别把 0.1998 当成普遍保证。

第二，等相关真值结构给了等权一层保护，压低了 GMV 层面的差异。异质相关下的业绩差距需要另做实验，本文不下结论。

第三，`sin(πτ/2)` 的解析关系只对椭圆分布族成立。若收益存在强烈偏度或非线性依赖（比如期权类资产、尾部非对称的信用组合），变换后量纲可能仍有偏，此时更适合直接用 Copula 依赖建模。

第四，计算成本。Kendall 的 einsum 实现是 `O(n²p²)`，`p=40` 时一次约几十毫秒，日频批量还能接受；分钟级高维面板需要换成随机投影或分块计算。

工程上的落地顺序建议是：先用 Kendall 相关矩阵算一遍、同时和 Pearson 对照，如果两者差异显著，优先去查数据质量而不是急着换模型；确认要用秩相关后再检查最小特征值，必要时做 PSD 投影；最后才把矩阵交给优化器。稳健不是换个估计量就完事，是三层检查。

## 七、完整可运行代码

下面这段代码就是生成全部五张图与 `results.json` 的脚本，在项目根目录直接运行即可。它自包含：包含三种数据生成机制、四种估计量、GMV 评价、无偏性验证、`p/n` 扫描与异质相关的正定性压力测试，不依赖任何外部数据文件，只用到 numpy 与 matplotlib。

```python
"""秩相关在厚尾与污染数据下的稳健性：完整可运行实验。

运行方式（在项目根目录）：
    python3 scripts/article_core_kendall_20261108.py

产出：
    public/images/kendall-tau-robust-covariance/fig1..fig5.png
    public/images/kendall-tau-robust-covariance/results.json

对照四种相关矩阵估计量：Pearson、Winsorize+Pearson、Spearman 变换、
Kendall 变换。所有数字均由本脚本现场计算，无外部数据。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261108
OUT = "public/images/kendall-tau-robust-covariance"
SIGMA = 0.015          # 日波动率
RHO = 0.35             # 真实等相关系数
CONTAM_LEVELS = (0.0, 0.01, 0.02, 0.05, 0.10)
METHODS = ("Pearson", "Winsorized", "Spearman", "Kendall")


def true_cov(p, rho=RHO, sigma=SIGMA):
    c = np.full((p, p), rho)
    np.fill_diagonal(c, 1.0)
    return (sigma ** 2) * c


def hetero_corr(rng, p, rank=3):
    """异质相关矩阵：低秩因子 + 个体差异，逐元素非线性变换后才可能破坏正定。"""
    B = rng.normal(size=(p, rank)) * 0.6
    C = B @ B.T + np.diag(rng.uniform(0.5, 1.5, size=p) ** 2) + np.eye(p) * 0.3
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def gen_data(rng, n, p, dgp, eps=0.0, rho=RHO, sigma=SIGMA, corr=None):
    """dgp: 'gaussian' | 't3' | 'contaminated'；返回 (n,p) 观测矩阵。

    给定 corr 时使用任意相关结构，否则使用等相关 rho。
    """
    if corr is None:
        c = np.full((p, p), rho)
        np.fill_diagonal(c, 1.0)
    else:
        c = corr
    L = np.linalg.cholesky(c)
    G = rng.standard_normal((n, p)) @ L.T
    if dgp == "t3":
        w = rng.chisquare(3.0, size=n) / 3.0
        X = G / np.sqrt(w)[:, None]
    else:
        X = G.copy()
    if dgp == "contaminated" and eps > 0:
        n_bad = max(1, int(round(n * eps)))
        rows = rng.choice(n, size=n_bad, replace=False)
        for r in rows:
            cols = rng.choice(p, size=2, replace=False)
            X[r, cols] += rng.choice([-1.0, 1.0], size=2) * 8.0
    return X * sigma


def kendall_from_ranks(R):
    """一次性算出所有配对的 Kendall tau（O(n^2 p^2) 的 einsum 实现）。

    D[a,b,i] = sign(rank[b,i] - rank[a,i])，
    sum_{a,b} D[a,b,i]*D[a,b,j] = 2*(concordant - discordant)，
    除以 n(n-1) 即得 tau。
    """
    n, p = R.shape
    D = np.sign(R[None, :, :] - R[:, None, :])
    return np.einsum("abi,abj->ij", D, D, optimize=True) / (n * (n - 1))


def estimates(X):
    """返回四种估计量的相关矩阵字典。"""
    n, p = X.shape
    R = np.argsort(np.argsort(X, axis=0), axis=0).astype(float)
    pear = np.corrcoef(X, rowvar=False)
    Z = np.column_stack([np.clip(X[:, j],
                                 np.quantile(X[:, j], 0.01),
                                 np.quantile(X[:, j], 0.99))
                         for j in range(p)])
    wins = np.corrcoef(Z, rowvar=False)
    spear = np.corrcoef(R, rowvar=False)
    kend = kendall_from_ranks(R)
    return {
        "Pearson": pear,
        "Winsorized": wins,
        "Spearman": 2.0 * np.sin(np.pi * spear / 6.0),
        "Kendall": np.sin(np.pi * kend / 2.0),
    }


def rel_frobenius(chat, ctrue):
    mask = ~np.eye(chat.shape[0], dtype=bool)
    return float(np.linalg.norm((chat - ctrue)[mask]) /
                 np.linalg.norm(ctrue[mask]))


def bias(chat, rho=RHO):
    p = chat.shape[0]
    off = chat[~np.eye(p, dtype=bool)]
    return float(off.mean() - rho)


def make_psd(A, floor=1e-8):
    ev, V = np.linalg.eigh(A)
    return (V * np.clip(ev, floor, None)) @ V.T, int((ev < 0).any())


def gmv_oos_vol(chat, ctrue, annual=252):
    """最小方差组合在真实协方差下的年化波动；非正定时先做 PSD 投影。"""
    A, bad = make_psd(chat)
    ones = np.ones(A.shape[0])
    try:
        w = np.linalg.solve(A, ones)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, ones, rcond=None)[0]
    w = w / w.sum()
    var = float(w @ ctrue @ w)
    return float(np.sqrt(annual * max(var, 0.0))), bad


def oracle_vol(ctrue, annual=252):
    p = ctrue.shape[0]
    w = np.ones(p) / p  # 等相关矩阵的最优 GMV 权重就是等权
    return float(np.sqrt(annual * float(w @ ctrue @ w)))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    n, p = 200, 10
    C = true_cov(p) / (SIGMA ** 2)  # 相关系数矩阵，作为估计目标
    COV = true_cov(p)
    R = 120

    # 图 1：三种 DGP 的散点示例
    samples = {
        "Gaussian": gen_data(rng, 400, 2, "gaussian", rho=RHO),
        "Student-t(3)": gen_data(rng, 400, 2, "t3", rho=RHO),
        "Contaminated (5%)": gen_data(rng, 400, 2, "contaminated", 0.05, RHO),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    for ax, (name, X) in zip(axes, samples.items()):
        ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.55, color="#2f5597")
        ax.set_title(name)
        ax.set_xlabel("asset 1")
        lim = 0.12 if "Contaminated" in name else (0.09 if "t(3)" in name else 0.06)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    axes[0].set_ylabel("asset 2")
    fig.suptitle("Same true correlation rho=0.35, three noise models")
    fig.tight_layout()
    save(fig, "fig1_scatter_dgps.png")

    # 主实验：污染比例扫描
    scan = []
    for eps in CONTAM_LEVELS:
        acc = {m: {"fro": [], "bias": [], "vol": [], "psd_bad": 0}
               for m in METHODS}
        for _ in range(R):
            X = gen_data(rng, n, p, "contaminated" if eps > 0 else "gaussian",
                         eps=eps)
            for m, chat in estimates(X).items():
                acc[m]["fro"].append(rel_frobenius(chat, C))
                acc[m]["bias"].append(bias(chat))
                v, bad = gmv_oos_vol(chat, COV)
                acc[m]["vol"].append(v)
                acc[m]["psd_bad"] += bad
        row = {"eps": eps}
        for m in METHODS:
            row[m] = {
                "rel_frobenius": float(np.mean(acc[m]["fro"])),
                "bias": float(np.mean(acc[m]["bias"])),
                "gmv_oos_vol": float(np.mean(acc[m]["vol"])),
                "psd_fail_rate": acc[m]["psd_bad"] / R,
            }
        scan.append(row)

    oracle = oracle_vol(COV)

    # 图 2：相对 Frobenius 误差 vs 污染比例
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = {"Pearson": "#c0392b", "Winsorized": "#e67e22",
              "Spearman": "#2f5597", "Kendall": "#1e8449"}
    for m in METHODS:
        ax.plot(CONTAM_LEVELS, [r[m]["rel_frobenius"] for r in scan],
                "o-", color=colors[m], label=m)
    ax.set_xlabel("contamination fraction epsilon")
    ax.set_ylabel("relative Frobenius error of correlation matrix")
    ax.set_title(f"n={n}, p={p}, {R} replications")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig2_frobenius_vs_contamination.png")

    # 图 4：GMV 样本外年化波动 vs 污染比例
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for m in METHODS:
        ax.plot(CONTAM_LEVELS, [r[m]["gmv_oos_vol"] for r in scan],
                "o-", color=colors[m], label=m)
    ax.axhline(oracle, ls="--", color="#555555", label="oracle (equal weight)")
    ax.set_xlabel("contamination fraction epsilon")
    ax.set_ylabel("out-of-sample annualized volatility")
    ax.set_title("Global minimum variance portfolio under corrupted inputs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig4_gmv_oos_volatility.png")

    # 无偏性验证：sin(pi*tau/2) 在椭圆分布下是否还原 Pearson rho
    rhos = (0.0, 0.2, 0.4, 0.6, 0.8)
    unbiased = []
    for dgp in ("gaussian", "t3"):
        for rho in rhos:
            vals = {m: [] for m in METHODS}
            for _ in range(60):
                X = gen_data(rng, 250, p, dgp, rho=rho)
                for m, chat in estimates(X).items():
                    vals[m].append(bias(chat, rho) + rho)
            unbiased.append({"dgp": dgp, "rho": rho,
                             **{m: float(np.mean(vals[m])) for m in METHODS}})

    # 图 3：估计均值 vs 真实 rho
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, dgp in zip(axes, ("gaussian", "t3")):
        sub = [u for u in unbiased if u["dgp"] == dgp]
        for m in ("Pearson", "Spearman", "Kendall"):
            ax.plot(rhos, [u[m] for u in sub], "o-", color=colors[m], label=m)
        ax.plot(rhos, rhos, ls="--", color="#666666", label="identity")
        ax.set_title("Gaussian" if dgp == "gaussian" else "Student-t(3)")
        ax.set_xlabel("true rho")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("estimated rho")
    axes[0].legend(fontsize=8)
    fig.suptitle("Does sin(pi*tau/2) recover the Pearson correlation?")
    fig.tight_layout()
    save(fig, "fig3_sin_transform_unbiasedness.png")

    # p/n 扫描：误差与正定性失败率
    pn = []
    for pp in (6, 10, 20):
        Cp = true_cov(pp) / (SIGMA ** 2)
        COVp = true_cov(pp)
        for nn in (60, 120, 250, 500):
            fp, fk, bad_k = [], [], 0
            for _ in range(40):
                X = gen_data(rng, nn, pp, "contaminated", 0.05)
                e = estimates(X)
                fp.append(rel_frobenius(e["Pearson"], Cp))
                fk.append(rel_frobenius(e["Kendall"], Cp))
                _, b = gmv_oos_vol(e["Kendall"], COVp)
                bad_k += b
            pn.append({"p": pp, "n": nn,
                       "pearson_fro": float(np.mean(fp)),
                       "kendall_fro": float(np.mean(fk)),
                       "kendall_psd_fail_rate": bad_k / 40})

    # 正定性压力测试：等相关矩阵被逐元素变换后仍是等相关，结构上必然 PSD；
    # 只有异质相关结构才能检验「逐元素 sin 变换破坏正定性」这条边界。
    psd_stress = []
    p_stress = 40
    for nn in (30, 60, 120, 250):
        bad = {m: 0 for m in METHODS}
        mineigs = []
        for _ in range(40):
            Ch = hetero_corr(rng, p_stress)
            X = gen_data(rng, nn, p_stress, "contaminated", 0.05, corr=Ch)
            e = estimates(X)
            for m, chat in e.items():
                if np.linalg.eigvalsh(chat)[0] < 0:
                    bad[m] += 1
            mineigs.append(np.linalg.eigvalsh(e["Kendall"])[0])
        psd_stress.append({
            "p": p_stress, "n": nn,
            **{m: bad[m] / 40 for m in METHODS},
            "kendall_mean_min_eig": float(np.mean(mineigs)),
        })

    # 图 5：p/n 扫描 + 正定性压力测试
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for pp in (6, 10, 20):
        sub = [r for r in pn if r["p"] == pp]
        xs = [r["n"] for r in sub]
        axes[0].plot(xs, [r["pearson_fro"] for r in sub], "s--",
                     color=colors["Pearson"], alpha=0.8, label=f"Pearson p={pp}")
        axes[0].plot(xs, [r["kendall_fro"] for r in sub], "o-",
                     color=colors["Kendall"], alpha=0.8, label=f"Kendall p={pp}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("sample size n")
    axes[0].set_ylabel("relative Frobenius error")
    axes[0].set_title("Error vs sample size (5% contamination)")
    axes[0].legend(fontsize=7, ncol=2)
    for m in ("Pearson", "Spearman", "Kendall"):
        axes[1].plot([r["n"] for r in psd_stress],
                     [r[m] for r in psd_stress], "o-",
                     color=colors[m], label=m)
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("sample size n")
    axes[1].set_ylabel("non-PSD rate")
    axes[1].set_title(f"Heterogeneous corr (p={p_stress}): PSD failure")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig5_pn_scan_and_psd.png")

    # DGP 对照（无污染 vs 厚尾 vs 污染）
    dgp_cmp = []
    for dgp, eps in (("gaussian", 0.0), ("t3", 0.0), ("contaminated", 0.05)):
        vals = {m: [] for m in METHODS}
        for _ in range(R):
            X = gen_data(rng, n, p, dgp, eps)
            for m, chat in estimates(X).items():
                vals[m].append(rel_frobenius(chat, C))
        dgp_cmp.append({"dgp": dgp, "eps": eps,
                        **{m: float(np.mean(vals[m])) for m in METHODS}})

    results = {
        "seed": SEED,
        "n": n, "p": p, "rho": RHO, "sigma": SIGMA,
        "replications_main": R,
        "contamination_scan": scan,
        "dgp_comparison": dgp_cmp,
        "unbiasedness": unbiased,
        "pn_scan": pn,
        "psd_stress": psd_stress,
        "oracle_gmv_annual_vol": oracle,
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"oracle GMV annual vol = {oracle:.6f}")
    print("=== contamination scan (relative Frobenius error) ===")
    for r in scan:
        print(f"eps={r['eps']:.2f}  " + "  ".join(
            f"{m}={r[m]['rel_frobenius']:.4f}" for m in METHODS))
    print("=== GMV out-of-sample annual vol @ eps=0.05 ===")
    row = [r for r in scan if r["eps"] == 0.05][0]
    for m in METHODS:
        print(f"  {m:11s} vol={row[m]['gmv_oos_vol']:.6f}  "
              f"psd_fail={row[m]['psd_fail_rate']:.3f}")
    print("=== DGP comparison (relative Frobenius error) ===")
    for r in dgp_cmp:
        print(f"{r['dgp']:13s} " + "  ".join(
            f"{m}={r[m]:.4f}" for m in METHODS))
    print("=== unbiasedness (Kendall sin transform, Gaussian) ===")
    for u in [u for u in unbiased if u["dgp"] == "gaussian"]:
        print(f"  rho={u['rho']:.1f}  Pearson={u['Pearson']:.4f}  "
              f"Spearman={u['Spearman']:.4f}  Kendall={u['Kendall']:.4f}")
    print("=== PSD stress (heterogeneous corr, p=40) ===")
    for r in psd_stress:
        print(f"  n={r['n']:3d}  " + "  ".join(
            f"{m}={r[m]:.3f}" for m in METHODS) +
            f"  min_eig={r['kendall_mean_min_eig']:.4f}")
    return results


if __name__ == "__main__":
    run()
```

## 八、小结

Kendall tau 的价值不在于「更高级」，而在于它把每个样本的影响力从数值大小压成了次序，于是一个 `±8σ` 的脏数据最多贡献一票，不会平方放大。代价是干净数据下约 6.7% 的效率损失、高维小样本下额外的正定性风险，以及一个必须说清楚的事实：本文里相关矩阵的巨大误差差距，并没有等比例变成组合波动的差距。把「矩阵准」和「业绩好」分开记账，比挑一个赢家更重要。
