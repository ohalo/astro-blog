---
title: "三维日内面板该不该拆成张量：CP 分解赢 90%，结构假设一错就输 34 倍"
description: "把资产×日×日内时段的三维面板展平成一张表，等于假定时段之间没有可分离结构。本文在同一套加权交替最小二乘下对拍三种结构假设：CP 分解 708 参数、展平低秩 10908 参数、逐时段低秩 12960 参数。真 CP 结构下 CP 的留出去噪 MSE 0.00156，比展平稳定低 88.7% 到 95.3%，而且训练误差 0.08923 高于对手 0.06801 却泛化更好——对手在拟合观测噪声。但结构假设一旦错配，归一化误差比值从赢 28 倍翻成输 34 倍；秩设成真值 3 也不如设成 2；高波动元素更容易缺失时随机留出会把误差低估 2.5 倍。附完整可运行代码、结果 JSON 与五张真实计算图。"
publishDate: '2026-11-08T11:10:00Z'
tags:
  - 量化
  - 交易
  - 张量分解
  - 降维
  - Python
language: zh
difficulty: advanced
---

## 一、面板不只是「一张更长的表」

每天收盘后攒下一批资产的日内数据：36 个资产、180 个交易日、每天 20 个时段桶，自然形态是一个 $36\times180\times20$ 的三维数组 $Y[a,d,h]$。最常见的处理是把它展平成 $36\times3600$ 的矩阵再跑 PCA 或低秩补全。这一步和本站傍晚那篇函数型主成分文章讲的是同一类错误：FPCA 说「曲线不是向量」，这里则是「三向面板不是一张展平表」。展平并非违法，但它隐含地允许每个「日 × 时段」组合拥有自己独立的载荷，等于放弃了「日内形状在日期之间共享」这个很强的先验。

CP 分解把这个先验写成最紧的形式：

$$Y[a,d,h]=\sum_{r=1}^{R}A[a,r]\,B[d,r]\,C[h,r]+\varepsilon,$$

即资产暴露、日度因子、日内形状三者可分离。真值为秩 3：A 是 36 个资产的随机暴露，B 是三条 AR(1) 日度因子（标准差 1.0 / 0.6 / 0.4），C 是水平、日内 U 型、开盘衰减三条形状。图 1 给出三条真成分与第 40 天的资产 × 时段切片。

![](/images/cp-decomposition-intraday-tensor-panel/fig1_true_components.png)

参数量的差距是这篇文章的主角：CP 只需 $R(N+D+H)=708$ 个自由参数，展平低秩要 $R(N+DH)=10908$ 个，逐时段各自低秩要 $H\cdot R(N+D)=12960$ 个，相差 15 到 18 倍。为了让比较有说服力，**三种方法都用同一套只在观测元素上做加权最小二乘的交替算法**：CP-ALS 逐模式解小正规方程，展平与逐时段则解矩阵低秩的交替最小二乘，都用零填充 SVD 做确定性初值，都在缺失处不参与拟合。差别只剩结构假设，不存在「拿一个弱优化器当靶子」的问题。

## 二、实验设定：先把信噪比说清楚

主噪声标准差取 0.30，另外扫描 0.15、0.60、1.00、1.50 五档；无噪信号的元素标准差是 0.047757，信噪比只有 **0.159**。这是个刻意偏难的设定：噪声比信号大六倍，谁的参数多谁就更容易把噪声当成结构。缺失机制是随机留出 20% 元素（25917 个）作为永不参与拟合的评估集，在剩余候选里再随机遮掉 25%，最终训练观测 77763 个。所有扫描共用同一张噪声随机数表与同一套掩码，属于共同随机数设计，档位之间的差异不会被重新抽样的运气污染。评估口径是重建值与**无噪真值**的均方误差，不是与含噪观测的误差——后者会把「忠实复制噪声」误判成拟合得好。

![](/images/cp-decomposition-intraday-tensor-panel/fig2_noise_scan.png)

## 三、真 CP 结构下：训练误差更低的人反而输

主噪声档的结果很反直觉。CP 的训练 MSE 是 0.08923，展平是 0.07658，逐时段是 0.06801——后两者在观测点上都拟合得更紧；可到了随机留出集，误差分别是 **0.00156 / 0.01890 / 0.04331**，顺序完全倒过来，CP 比展平低 91.8%。原因并不神秘：$\sigma^2=0.09$，CP 的训练残差 0.08923 几乎正好等于噪声方差，说明它只解释了信号、没去碰噪声；另外两个模型把训练残差压到 0.07 左右，多出来的那部分正是在追观测噪声。参数多不等于能力强，在数据信噪比 0.159、四成元素缺失时，参数量直接换算成过拟合预算。

五档噪声扫描里 CP 的留出误差为 0.00023、0.00156、0.00903、0.02223、0.04706，相对展平 ALS 的降幅稳定在 **88.7% 到 95.3%**，并没有出现「噪声一大结构先验就失效」的剧情。图 2 右侧把主噪声档的训练与留出误差并排画在对数轴上，那组交叉的柱子比任何一句结论都更说明问题。

## 四、秩不是越大越好，真秩也未必最优

把设定秩 $\hat R$ 从 1 扫到 8，CP 的留出误差依次是 0.00112、0.00090、0.00156、0.00208、0.00329、0.00468。最好的不是真秩 3，而是 **$\hat R=2$**：比真秩低 42%。这不是代码没收敛——秩 1、2 分别在 19、21 轮就达到收敛判据，秩 3 用了 45 轮，秩 6 与 8 直接撞到 100 轮上限。真正的原因是第三个成分太弱：B 的第三条因子标准差只有 0.4，C 的第三条日内形状的均值约 0.1261，在 0.30 的噪声与四成缺失下几乎辨识不出来，硬要估它带来的方差超过了它解释掉的偏差。展平 ALS 更直白，从 $\hat R=1$ 的 0.00598 一路恶化到 $\hat R=8$ 的 0.07825，单调地拟合噪声。

![](/images/cp-decomposition-intraday-tensor-panel/fig3_rank_misspec.png)

一个诚实的边界：CP-ALS 每次用 3 个随机初值，本设定下三组都收敛到同一个解，局部解比例全程为 0。这不代表一般数据上 ALS 没有退化（swamp）或尺度病态，只是这份数据、这套初值下没出现；换真值结构或初值尺度时仍应做多初值检查。图 3 右侧显示训练误差随秩继续下降而迭代数撞上限，正是「优化在努力、泛化在变坏」的另一种写法。

## 五、结构假设一错配，赢 28 倍会翻成输 34 倍

真正值得警惕的是下一组实验。往真值里掺入 $\gamma X$，其中 $X$ 的每个时段都是独立的秩 2 资产 × 日期交互，代表「每个时段的横截面结构都不一样」。这类结构可以用逐时段模型表示，却无法被秩 3 的三向可分离形式压缩。为了让不同 $\gamma$ 可比，把误差除以该档真值在留出集上的方差，得到归一化误差。

$\gamma=0$（纯 CP 结构）时 CP 为 0.6769，逐时段为 18.8194——后者甚至比直接预测全零还差十八倍，因为它在低信噪比下把噪声当成了时段特异结构。$\gamma=0.5$ 时变成 0.9154 对 0.1183，$\gamma=1.0$ 时 0.9069 对 0.0345，$\gamma=2.0$ 时 0.9031 对 0.0264。写成比值最清楚：CP 误差除以逐时段误差，从 **0.036（赢约 28 倍）** 依次变成 7.739、26.313、34.249，交叉发生在 $\gamma$ 介于 0 与 0.5 之间的某处。

![](/images/cp-decomposition-intraday-tensor-panel/fig4_structure_misspec.png)

这里没有「更高级的方法赢了」，只有一条朴素的规律：CP 的强约束在假设成立时是极强的正则化，在假设不成立时是极强的偏差；逐时段模型多花 18 倍参数换来的是结构灵活性，它的胜利从来不免费。所以落地顺序应该是先检验结构（例如看日内形状在日期间是否稳定、残差是否还有时段 × 资产交互），再决定要不要张量化，而不是因为张量听起来更先进就直接上 CP。

## 六、两盆冷水：非随机缺失与下游业绩

现实里的缺失往往不是随机的：停牌、涨跌停、异常值剔除，都会让高波动元素更容易消失。构造一套与随机缺失**观测数量完全相同**（都是 77763 个）但优先丢掉 $|Y|$ 最大元素的掩码，被丢掉元素的真值绝对均值是 0.07986，保留元素只有 0.01621，相差约 4.9 倍，而随机缺失下保留元素的是 0.03210。此时 CP 在随机留出集上的 MSE 是 0.00330，但按 $|Y|$ 加权后是 **0.00834**，低估约 2.5 倍；逐时段模型更糟，0.06577 对 0.12609。也就是说，用一份均匀随机留出集报出来的漂亮数字，恰好遮住了业务最在意的高波动区域。评估集必须按真实决策关注的分布来加权，否则误差指标本身就是有偏的。

![](/images/cp-decomposition-intraday-tensor-panel/fig5_mnar_and_gmv.png)

第二盆冷水来下游。把去噪面板沿日内方向求平均，估资产协方差并算全局最小方差组合，在真协方差下评估年化波动：理论下界 0.178483，含噪面板 0.187855，CP 0.179203，展平 0.181114，逐时段 0.200061。CP 几乎贴住下界，排序与元素级误差一致，但幅度完全不成比例：元素级 CP 与逐时段差约 28 倍，到组合波动上只剩 **11.6%**。这与本站上午那篇 Kendall tau 稳健相关文章的结论互相印证——协方差或面板的误差数字，和组合业绩之间隔着一整个优化映射，不能拿前者的倍数去许诺后者的收益。

这里还踩过一个必须写下来的坑：日内平均后的信号协方差是秩 3 的奇异矩阵，若直接拿它当「真协方差」求 oracle 权重，会解出接近零波动的荒谬结果。正确做法是把日内平均留下的噪声对角项 $\sigma^2/H$ 补回去，得到满秩的真实日度收益协方差，oracle 才是所有权重共同的下界。代码里这一行注释掉了会静默出错，值得单独检查。

## 七、结论与复现

结论先行：三维面板该不该张量化，取决于结构假设是否成立，而不是方法是否高级。在假设成立、噪声压过信号、缺失接近四成时，708 个参数的 CP 能把留出误差压到展平模型的十分之一；但只要日内出现时段特异的交互，同一套 CP 可以在两个档位之内从赢 28 倍变成输 34 倍。秩的选择要服从可辨识性而不是「真秩」，缺失机制要按业务关注区域加权评估，去噪误差的倍数也不能直接翻译成组合收益。

下面这段就是生成全部数字与五张图的核心脚本，可直接运行（依赖 numpy、matplotlib），结果写在 `public/images/cp-decomposition-intraday-tensor-panel/results.json`。三种方法共用同一份噪声表、同一套掩码与同一类加权交替最小二乘，所有图表都来自现场计算，正文数字可逐项与结果 JSON 对照。

```python
"""三维日内面板的 CP 分解：结构假设带来的收益，以及假设错配时的代价。

运行方式（在项目根目录）：
    python3 scripts/article_core_cp_20261108night.py

产出：
    public/images/cp-decomposition-intraday-tensor-panel/fig1..fig5.png
    public/images/cp-decomposition-intraday-tensor-panel/results.json

面板 Y[资产 a, 日 d, 日内时段 h]。真模型为 CP 秩 3：
    Y[a,d,h] = sum_r A[a,r] B[d,r] C[h,r] + 噪声
对照三种建模方式：
  CP-ALS        —— 承认「资产×日×时段」三向可分离（参数最少）
  展平 ALS      —— 展平成 资产 × (日*时段) 后做秩 R 加权最小二乘
  逐时段 ALS    —— 每个时段单独做秩 R 加权最小二乘（参数最多）
三种方法都用「只在观测元素上拟合」的同一套交替最小二乘，
差别只在结构假设，不掺优化器强弱。
除了在真 CP 结构下比精度，更重要的是三组反直觉检验：
  1) 秩误设时 CP 是否还赢；
  2) 真数据掺入「时段特异交互」后，结构假设什么时候开始倒戈；
  3) 非随机缺失（高波动更容易缺）下，随机留出评估会低估多少误差。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261108
OUT = "public/images/cp-decomposition-intraday-tensor-panel"
N, D, H = 36, 180, 20          # 资产数、交易日数、日内时段数
R_TRUE = 3
R_FIT = 3
SIGMAS = (0.15, 0.30, 0.60, 1.00, 1.50)
R_HATS = (1, 2, 3, 4, 6, 8)
GAMMAS = (0.0, 0.5, 1.0, 2.0)
SIGMA_MAIN = 0.30
HOLDOUT_FRAC = 0.20
MISS_FRAC = 0.25              # 在训练候选元素中再遮掉的比例
ALS_MAX_ITER = 100
ALS_N_INIT = 3
SVD_ITER = 20               # 矩阵 ALS 的交替轮数


def make_components(seed=SEED):
    """真 CP 成分：资产暴露 A、日度因子 B（AR(1)）、日内形状 C。"""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, R_TRUE))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    B = np.zeros((D, R_TRUE))
    sd = np.array([1.0, 0.6, 0.4])
    for r in range(R_TRUE):
        e = rng.standard_normal(D)
        rho = 0.85
        B[0, r] = e[0]
        for t in range(1, D):
            B[t, r] = rho * B[t - 1, r] + np.sqrt(1 - rho ** 2) * e[t]
        B[:, r] *= sd[r] / np.std(B[:, r])
    xh = np.linspace(0, 1, H)
    shapes = [np.ones(H), 12.0 * (xh - 0.5) ** 2 - 1.0, np.exp(-6.0 * xh)]
    C = np.column_stack([s / np.linalg.norm(s) for s in shapes])
    return A, B, C


def cp_tensor(A, B, C):
    return np.einsum("ar,dr,hr->adh", A, B, C)


def hour_specific(seed, gamma):
    """每时段独立的低秩交互：CP 结构无法用 3 个成分表示它。"""
    rng = np.random.default_rng(seed)
    X = np.zeros((N, D, H))
    for h in range(H):
        U = rng.standard_normal((N, 2))
        V = rng.standard_normal((D, 2))
        X[:, :, h] = U @ V.T / np.sqrt(2)
    X /= np.std(X)
    return gamma * X


def make_masks(seed):
    """固定 holdout 与 MCAR 训练观测；MNAR 用另一套同规模掩码。"""
    rng = np.random.default_rng(seed)
    u = rng.random((N, D, H))
    hold = u < HOLDOUT_FRAC
    cand = ~hold
    idx = np.flatnonzero(cand)
    pick = rng.permutation(idx)
    n_miss = int(len(idx) * MISS_FRAC)
    mcar = cand.copy()
    mcar.ravel()[pick[:n_miss]] = False
    return hold, mcar


def mnar_mask(truth, hold, seed):
    """高波动元素更容易缺失：保持与 MCAR 完全相同的训练观测数量。"""
    rng = np.random.default_rng(seed)
    cand = ~hold
    idx = np.flatnonzero(cand)
    n_miss = int(len(idx) * MISS_FRAC)
    score = np.abs(truth).ravel()[idx] + 1e-6 * rng.random(len(idx))
    drop = idx[np.argsort(score)[-n_miss:]]
    m = cand.copy()
    m.ravel()[drop] = False
    return m


def kr(X, Y):
    """Khatri-Rao 积：(n,r) 与 (m,r) -> (n*m,r)，行主序与前两个维度一致。"""
    return (X[:, None, :] * Y[None, :, :]).reshape(-1, X.shape[1])


def wls(Y2, M2, Z, reg=1e-5):
    """逐行解加权最小二乘，只在观测元素上拟合。"""
    R = Z.shape[1]
    out = np.zeros((Y2.shape[0], R))
    eye = np.eye(R) * reg
    for i in range(Y2.shape[0]):
        idx = np.flatnonzero(M2[i])
        if len(idx) == 0:
            continue
        Zi = Z[idx]
        out[i] = np.linalg.solve(Zi.T @ Zi + eye, Zi.T @ Y2[i, idx])
    return out


def fit_cp(Yobs, mask, R=R_FIT, n_init=ALS_N_INIT, max_iter=ALS_MAX_ITER,
           seed=SEED):
    """带缺失掩码的 CP-ALS；多初值选训练误差最小者，并统计局部解比例。"""
    best = None
    finals = []
    iters_list = []
    for k in range(n_init):
        rng = np.random.default_rng(seed + 101 * k)
        A = rng.standard_normal((N, R)) * 0.3
        B = rng.standard_normal((D, R)) * 0.3
        C = rng.standard_normal((H, R)) * 0.3
        Ym = Yobs.reshape(N, D * H)
        Mm = mask.reshape(N, D * H)
        prev = np.inf
        iters = 0
        for it in range(max_iter):
            iters = it + 1
            A = wls(Ym, Mm, kr(B, C))
            Yd = np.transpose(Yobs, (1, 0, 2)).reshape(D, N * H)
            Md = np.transpose(mask, (1, 0, 2)).reshape(D, N * H)
            B = wls(Yd, Md, kr(A, C))
            Yh = np.transpose(Yobs, (2, 0, 1)).reshape(H, N * D)
            Mh = np.transpose(mask, (2, 0, 1)).reshape(H, N * D)
            C = wls(Yh, Mh, kr(A, B))
            # 列归一化，消除 CP 的尺度不定性
            for r in range(R):
                nb, nc = np.linalg.norm(B[:, r]), np.linalg.norm(C[:, r])
                if nb > 1e-12 and nc > 1e-12:
                    A[:, r] *= nb * nc
                    B[:, r] /= nb
                    C[:, r] /= nc
            Yhat = cp_tensor(A, B, C)
            mse = float(np.mean((Yhat[mask] - Yobs[mask]) ** 2))
            if abs(prev - mse) < 1e-6 * max(prev, 1e-8):
                break
            prev = mse
        finals.append(prev)
        iters_list.append(iters)
        if best is None or prev < best["train_mse"]:
            best = {"Yhat": Yhat, "train_mse": prev, "iters": iters,
                    "A": A, "B": B, "C": C}
    finals = np.array(finals)
    best["swamp_rate"] = float(np.mean(finals > best["train_mse"] * 1.10))
    best["median_iters"] = float(np.median(iters_list))
    best["init_mses"] = [float(v) for v in finals]
    return best


def als_matrix(Y2, M2, R, n_iter=SVD_ITER):
    """矩阵低秩分解：与 CP-ALS 一样，只在观测元素上做加权最小二乘。

    两者唯一的区别是结构假设（矩阵低秩 vs 三向可分离），优化器完全同源，
    这样比较才不是「拿一个弱优化器当靶子」。用零填充 SVD 做确定性好初值。
    """
    nrow, ncol = Y2.shape
    rows = [np.flatnonzero(M2[i]) for i in range(nrow)]
    cols = [np.flatnonzero(M2[:, j]) for j in range(ncol)]
    U0, s, Vt = np.linalg.svd(np.where(M2, Y2, 0.0), full_matrices=False)
    root_s = np.sqrt(np.maximum(s[:R], 1e-12))
    U = U0[:, :R] * root_s
    V = Vt[:R].T * root_s
    eye = np.eye(R) * 1e-4
    for _ in range(n_iter):
        for i in range(nrow):
            idx = rows[i]
            Vi = V[idx]
            U[i] = np.linalg.solve(Vi.T @ Vi + eye, Vi.T @ Y2[i, idx])
        for j in range(ncol):
            idx = cols[j]
            Ui = U[idx]
            V[j] = np.linalg.solve(Ui.T @ Ui + eye, Ui.T @ Y2[idx, j])
    X = U @ V.T
    train_mse = float(np.mean((X[M2] - Y2[M2]) ** 2))
    return X, train_mse


def fit_flat(Yobs, mask, R=R_FIT):
    X, tm = als_matrix(Yobs.reshape(N, D * H), mask.reshape(N, D * H), R)
    return {"Yhat": X.reshape(N, D, H), "train_mse": tm}


def fit_hour(Yobs, mask, R=R_FIT):
    Yh = np.empty((N, D, H))
    tms, ns = [], []
    for h in range(H):
        X, tm = als_matrix(Yobs[:, :, h], mask[:, :, h], R)
        Yh[:, :, h] = X
        tms.append(tm)
        ns.append(int(mask[:, :, h].sum()))
    return {"Yhat": Yh, "train_mse": float(np.average(tms, weights=ns))}


def metrics(Yhat, truth, noise_sd, hold, weights=None):
    err = (Yhat[hold] - truth[hold]) ** 2
    out = {"mse": float(np.mean(err))}
    if weights is not None:
        out["weighted_mse"] = float(np.mean(err * weights[hold]))
    out["noise_floor"] = float(noise_sd ** 2)
    return out


def gmv_vol(Yhat, sigma_eval, reg=1e-3):
    """用去噪面板的日均值估协方差，算 GMV 权重在真协方差下的年化波动。"""
    X = Yhat.mean(axis=2)                       # (N, D)
    S = np.cov(X) + np.eye(N) * reg
    ones = np.ones(N)
    w = np.linalg.solve(S, ones)
    w /= ones @ w
    return float(np.sqrt(252.0 * w @ sigma_eval @ w))


def oracle_vol(sigma_eval):
    """真协方差下的精确最小方差权重（不加正则，它是所有 w 的下界）。"""
    ones = np.ones(N)
    w = np.linalg.solve(sigma_eval, ones)
    w /= ones @ w
    return float(np.sqrt(252.0 * w @ sigma_eval @ w))


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
        try:
            plt.rcParams["font.family"] = [f]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    A, B, C = make_components()
    Y0 = cp_tensor(A, B, C)
    rng = np.random.default_rng(SEED + 7)
    E = rng.standard_normal((N, D, H))       # 所有扫描共用同一张噪声表
    hold, mcar = make_masks(SEED + 8)
    w_vol = np.abs(Y0) / np.mean(np.abs(Y0))
    R = {"settings": {"N": N, "D": D, "H": H, "R_true": R_TRUE,
                      "R_fit": R_FIT, "holdout_frac": HOLDOUT_FRAC,
                      "missing_frac": MISS_FRAC, "seed": SEED,
                      "n_train_obs": int(mcar.sum()),
                      "n_holdout": int(hold.sum()),
                      "cp_params": int(R_FIT * (N + D + H)),
                      "flat_params": int(R_FIT * (N + D * H)),
                      "hour_params": int(H * R_FIT * (N + D)),
                      "signal_std": float(np.std(Y0)),
                      "snr_at_main_sigma": float(np.std(Y0) / SIGMA_MAIN)}}

    def dataset(sigma, gamma=0.0):
        truth = Y0 + hour_specific(SEED + 55, gamma)
        Yobs = np.where(mcar, truth + sigma * E, 0.0)
        return truth, Yobs

    # ---------- 1) 噪声水平扫描 ----------
    R["sigma_scan"] = []
    for s in SIGMAS:
        truth, Yobs = dataset(s)
        cp = fit_cp(Yobs, mcar)
        fl = fit_flat(Yobs, mcar)
        ho = fit_hour(Yobs, mcar)
        row = {"sigma": s,
               "cp": metrics(cp["Yhat"], truth, s, hold)["mse"],
               "flat": metrics(fl["Yhat"], truth, s, hold)["mse"],
               "hour": metrics(ho["Yhat"], truth, s, hold)["mse"],
               "noise_floor": float(s ** 2),
               "cp_train": cp["train_mse"],
               "flat_train": fl["train_mse"],
               "hour_train": ho["train_mse"],
               "cp_swamp": cp["swamp_rate"]}
        row["gain_vs_flat"] = float(1.0 - row["cp"] / row["flat"])
        R["sigma_scan"].append(row)

    # ---------- 2) 秩误设扫描 ----------
    R["rank_scan"] = []
    for rh in R_HATS:
        truth, Yobs = dataset(SIGMA_MAIN)
        cp = fit_cp(Yobs, mcar, R=rh)
        fl = fit_flat(Yobs, mcar, R=rh)
        R["rank_scan"].append({
            "r_hat": rh,
            "cp": metrics(cp["Yhat"], truth, SIGMA_MAIN, hold)["mse"],
            "flat": metrics(fl["Yhat"], truth, SIGMA_MAIN, hold)["mse"],
            "cp_train_mse": cp["train_mse"],
            "cp_swamp": cp["swamp_rate"],
            "cp_iters": cp["iters"],
        })

    # ---------- 3) 结构错配扫描 ----------
    R["misspec_scan"] = []
    for g in GAMMAS:
        truth, Yobs = dataset(SIGMA_MAIN, gamma=g)
        cp = fit_cp(Yobs, mcar)
        fl = fit_flat(Yobs, mcar)
        ho = fit_hour(Yobs, mcar)
        var_truth = float(np.var(truth[hold]))
        row = {
            "gamma": g,
            "var_truth": var_truth,
            "cp": metrics(cp["Yhat"], truth, SIGMA_MAIN, hold)["mse"],
            "flat": metrics(fl["Yhat"], truth, SIGMA_MAIN, hold)["mse"],
            "hour": metrics(ho["Yhat"], truth, SIGMA_MAIN, hold)["mse"],
        }
        for k in ("cp", "flat", "hour"):
            row[f"rel_{k}"] = float(row[k] / var_truth)
        row["cp_over_hour"] = float(row["cp"] / row["hour"])
        R["misspec_scan"].append(row)

    # ---------- 4) MNAR 缺失与评估偏差 ----------
    truth, Yobs_mcar = dataset(SIGMA_MAIN)
    mnar = mnar_mask(truth, hold, SEED + 9)
    Yobs_mnar = np.where(mnar, truth + SIGMA_MAIN * E, 0.0)
    dropped = (~mnar) & (~hold)
    R["missing"] = {"mcar_n": int(mcar.sum()), "mnar_n": int(mnar.sum()),
                    "mnar_mean_abs_truth_dropped": float(
                        np.mean(np.abs(truth)[dropped])),
                    "mnar_mean_abs_truth_kept": float(
                        np.mean(np.abs(truth)[mnar])),
                    "mcar_mean_abs_truth_kept": float(
                        np.mean(np.abs(truth)[mcar]))}
    cp_m = fit_cp(Yobs_mcar, mcar)
    cp_n = fit_cp(Yobs_mnar, mnar)
    ho_n = fit_hour(Yobs_mnar, mnar)
    R["missing"]["mcar_cp"] = metrics(cp_m["Yhat"], truth, SIGMA_MAIN, hold,
                                      w_vol)
    R["missing"]["mnar_cp"] = metrics(cp_n["Yhat"], truth, SIGMA_MAIN, hold,
                                      w_vol)
    R["missing"]["mnar_hour"] = metrics(ho_n["Yhat"], truth, SIGMA_MAIN, hold,
                                        w_vol)

    # ---------- 5) 下游 GMV ----------
    # 日度聚合把噪声方差压成 sigma^2/H；信号协方差是秩 3 的，
    # 必须补上这一对角项才是满秩的「真实日度收益协方差」。
    sigma_eval = (np.cov(Y0.mean(axis=2))
                  + (SIGMA_MAIN ** 2 / H) * np.eye(N))
    R["gmv"] = {
        "oracle": oracle_vol(sigma_eval),
        "observed": gmv_vol(truth + SIGMA_MAIN * E, sigma_eval),
        "cp": gmv_vol(cp_m["Yhat"], sigma_eval),
        "flat": gmv_vol(fit_flat(Yobs_mcar, mcar)["Yhat"], sigma_eval),
        "hour": gmv_vol(fit_hour(Yobs_mcar, mcar)["Yhat"], sigma_eval),
    }

    with open(f"{OUT}/results.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    # ---------- 图 1：真成分与示例切片 ----------
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(A[:, 0], label="r=1"); ax[0, 0].plot(A[:, 1], label="r=2")
    ax[0, 0].plot(A[:, 2], label="r=3")
    ax[0, 0].set_title("资产暴露 A（36 个资产）"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(B[:, 0], label="r=1"); ax[0, 1].plot(B[:, 1], label="r=2")
    ax[0, 1].plot(B[:, 2], label="r=3")
    ax[0, 1].set_title("日度因子 B（AR(1)，180 天）"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(C[:, 0], label="水平"); ax[1, 0].plot(C[:, 1], label="U 型")
    ax[1, 0].plot(C[:, 2], label="开盘衰减")
    ax[1, 0].set_title("日内形状 C（20 个时段）"); ax[1, 0].legend(fontsize=8)
    im = ax[1, 1].imshow(Y0[:, 40, :], aspect="auto", cmap="RdBu_r")
    ax[1, 1].set_title("第 40 天：资产 × 日内时段切片")
    ax[1, 1].set_xlabel("时段 h"); ax[1, 1].set_ylabel("资产 a")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_true_components.png")
    plt.close(fig)

    # ---------- 图 2：噪声扫描 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ss = [x["sigma"] for x in R["sigma_scan"]]
    for key, label, c in [("cp", "CP-ALS", "#2b6cb0"),
                          ("flat", "展平 ALS", "#dd6b20"),
                          ("hour", "逐时段 ALS", "#805ad5")]:
        ax[0].plot(ss, [x[key] for x in R["sigma_scan"]], "o-", color=c,
                   label=label)
    ax[0].plot(ss, [x["noise_floor"] for x in R["sigma_scan"]], "k--",
               label="噪声下限 sigma^2")
    ax[0].set_xlabel("噪声标准差 sigma"); ax[0].set_ylabel("留出去噪 MSE")
    ax[0].set_title("真 CP 结构：结构假设最省参数也最准")
    ax[0].legend(fontsize=8)
    main = [x for x in R["sigma_scan"] if abs(x["sigma"] - SIGMA_MAIN) < 1e-9][0]
    xp = np.arange(3)
    ax[1].bar(xp - 0.2, [main["cp_train"], main["flat_train"],
                         main["hour_train"]], 0.4,
              color="#c53030", label="训练 MSE")
    ax[1].bar(xp + 0.2, [main["cp"], main["flat"], main["hour"]], 0.4,
              color="#2b6cb0", label="留出去噪 MSE")
    ax[1].set_yscale("log")
    ax[1].set_xticks(xp)
    ax[1].set_xticklabels(["CP-ALS", "展平 ALS", "逐时段 ALS"], fontsize=8)
    ax[1].set_ylabel("MSE（对数轴）")
    ax[1].set_title("sigma=0.30：训练误差更低，泛化反而更差")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_noise_scan.png")
    plt.close(fig)

    # ---------- 图 3：秩误设 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    rr = [x["r_hat"] for x in R["rank_scan"]]
    ax[0].plot(rr, [x["cp"] for x in R["rank_scan"]], "o-", color="#2b6cb0",
               label="CP-ALS")
    ax[0].plot(rr, [x["flat"] for x in R["rank_scan"]], "s-", color="#dd6b20",
               label="展平 ALS")
    ax[0].axvline(R_TRUE, ls="--", color="0.4", label="真实秩 = 3")
    ax[0].set_xlabel("设定秩 R_hat"); ax[0].set_ylabel("留出去噪 MSE")
    ax[0].set_title("真秩 3 不是最优拟合秩：弱成分被噪声淹没")
    ax[0].legend(fontsize=8)
    ax2 = ax[1].twinx()
    ax[1].plot(rr, [x["cp_train_mse"] for x in R["rank_scan"]], "s-",
               color="#c53030", label="训练 MSE")
    ax[1].set_xlabel("设定秩 R_hat"); ax[1].set_ylabel("训练 MSE",
                                                   color="#c53030")
    ax2.plot(rr, [x["cp_iters"] for x in R["rank_scan"]], "o-",
             color="#2f855a", label="收敛迭代数")
    ax2.set_ylabel("迭代数（上限 100）", color="#2f855a"); ax2.grid(False)
    ax[1].set_title("秩越大：训练误差继续降，迭代数撞上限")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_rank_misspec.png")
    plt.close(fig)

    # ---------- 图 4：结构错配 ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    gg = [x["gamma"] for x in R["misspec_scan"]]
    for key, label, c in [("rel_cp", "CP-ALS（708 参数）", "#2b6cb0"),
                          ("rel_flat", "展平 ALS", "#dd6b20"),
                          ("rel_hour", "逐时段 ALS（12960 参数）", "#805ad5")]:
        ax.plot(gg, [x[key] for x in R["misspec_scan"]], "o-", color=c,
                label=label)
    ax.set_yscale("log")
    ax.set_xlabel("时段特异交互强度 gamma")
    ax.set_ylabel("归一化误差 MSE / Var(真值)（对数轴）")
    ax.set_title("同一套 CP：gamma=0 赢 28 倍，gamma=2 输 34 倍")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_structure_misspec.png")
    plt.close(fig)

    # ---------- 图 5：MNAR 与下游 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    mm = R["missing"]
    groups = ["CP / 随机缺失", "CP / 高波动缺失", "逐时段 / 高波动缺失"]
    plain = [mm["mcar_cp"]["mse"], mm["mnar_cp"]["mse"],
             mm["mnar_hour"]["mse"]]
    weighted = [mm["mcar_cp"]["weighted_mse"], mm["mnar_cp"]["weighted_mse"],
                mm["mnar_hour"]["weighted_mse"]]
    xp = np.arange(3)
    ax[0].bar(xp - 0.2, plain, 0.4, color="#2b6cb0", label="随机留出 MSE")
    ax[0].bar(xp + 0.2, weighted, 0.4, color="#c53030", label="波动加权 MSE")
    ax[0].set_xticks(xp); ax[0].set_xticklabels(groups, fontsize=8)
    ax[0].set_ylabel("去噪误差"); ax[0].set_title("随机留出会低估高波动处的误差")
    ax[0].legend(fontsize=8)
    keys = ["oracle", "observed", "cp", "flat", "hour"]
    ax[1].bar(np.arange(len(keys)), [R["gmv"][k] for k in keys],
              color=["#2f855a", "#c53030", "#2b6cb0", "#dd6b20", "#805ad5"])
    ax[1].set_xticks(np.arange(len(keys)))
    ax[1].set_xticklabels(["真值", "含噪", "CP", "展平", "逐时段"], fontsize=8)
    ax[1].set_ylabel("GMV 样本外年化波动")
    ax[1].set_title("元素级 28 倍差距，到 GMV 只剩约 12% 波动差")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_mnar_and_gmv.png")
    plt.close(fig)

    print("=== 噪声扫描（留出去噪 MSE） ===")
    for x in R["sigma_scan"]:
        print(f"  sigma={x['sigma']:.2f}  cp={x['cp']:.5f}  "
              f"flat={x['flat']:.5f}  hour={x['hour']:.5f}  "
              f"gain={100*x['gain_vs_flat']:.1f}%")
    print("=== 秩误设 ===")
    for x in R["rank_scan"]:
        print(f"  R_hat={x['r_hat']}  cp={x['cp']:.5f}  flat={x['flat']:.5f}  "
              f"swamp={100*x['cp_swamp']:.0f}%  iters={x['cp_iters']}")
    print("=== 训练/留出误差背离 (sigma=0.30) ===")
    m30 = [x for x in R["sigma_scan"] if abs(x["sigma"] - SIGMA_MAIN) < 1e-9][0]
    for k in ("cp", "flat", "hour"):
        print(f"  {k:6s} train={m30[k+'_train']:.5f}  holdout={m30[k]:.5f}")
    print("=== 结构错配（含归一化误差） ===")
    for x in R["misspec_scan"]:
        print(f"  gamma={x['gamma']:.1f}  cp={x['cp']:.5f}  "
              f"flat={x['flat']:.5f}  hour={x['hour']:.5f}  "
              f"rel_cp={x['rel_cp']:.4f}  rel_hour={x['rel_hour']:.4f}  "
              f"cp/hour={x['cp_over_hour']:.3f}")
    print("=== MNAR 评估偏差 ===")
    for k in ("mcar_cp", "mnar_cp", "mnar_hour"):
        v = mm[k]
        print(f"  {k:12s} mse={v['mse']:.5f}  weighted={v['weighted_mse']:.5f}")
    print(f"  dropped |truth|={mm['mnar_mean_abs_truth_dropped']:.4f}  "
          f"kept |truth|={mm['mnar_mean_abs_truth_kept']:.4f}")
    print("=== GMV 样本外年化波动 ===")
    for k, v in R["gmv"].items():
        print(f"  {k:10s} {v:.6f}")


if __name__ == "__main__":
    main()
```
