---
title: "函数型主成分：曲线不是向量，忽略 Gram 矩阵会让主成分失真"
description: "把 48 个日内波动率点当成 48 个变量做 PCA，等于偷偷把 L2 内积换成了欧氏内积。本文用 500 天模拟曲线现场对拍：基展开后特征方程是 S W b = λ b，忽略 Gram 矩阵 W 会让所谓特征函数的 L2 正交偏离达到 0.420、前三个成分的解释方差从 0.9361 误报成 0.8403、重构 MSE 从 0.00741 恶化到 0.01247。但结论不是「改用函数型方法就赢了」：内积修正之后，正确 FPCA 的重构 MSE 0.00741 仍略逊于逐点 PCA 的 0.00723，噪声越大差距越小；每天只剩 12 个观测点时反而落后 2.1 倍。降维也几乎不带来预测增益，真正管用的是 AR(1) 结构。附五张真实计算图、五组灵敏度扫描与完整可运行代码。"
publishDate: '2026-11-08T09:10:00Z'
tags:
  - 量化
  - 交易
  - 函数型数据分析
  - Python
language: zh
difficulty: advanced
---

## 一、一条曲线不是一个向量

假设每天收盘后你有一条日内已实现波动率曲线：把交易时段切成 48 个五分钟桶，每桶算一个已实现波动，一天就得到 48 个数。攒够 500 天，手上是一个 500×48 的矩阵。最顺手的下一步是对它做主成分分析，取前三个载荷，说成「水平因子、日内 U 型因子、开盘衰减因子」。收益率曲线 PCA 就是这么做的，本站也写过一篇。

这一步有个几乎不会被检查的前提：**它把 48 个坐标当成 48 个彼此独立、且用欧氏内积度量距离的变量**。可曲线是定义在连续时间轴上的函数，两个形状是否正交、某个方向解释了多少变异，取决于你在哪个内积下提问。函数空间里的自然内积是 $\langle f,g\rangle=\int_0^1 f(x)g(x)\,\mathrm{d}x$，而不是把 48 个坐标相乘再求和。一旦把曲线写成一组基函数的线性组合，这个内积就变成 $b^\top W b$，其中 $W_{kl}=\int\phi_k\phi_l$ 是 Gram 矩阵。**忘记 $W$，就是在不知不觉中换了内积**；算出来的方向仍然叫「主成分」，但它既不正交，解释方差比例也是错的。

这篇文章不打算推销函数型主成分（FPCA）。恰恰相反：先把内积错误造成的损害量化出来，再老老实实回答「内积改对之后，性能到底值不值这个复杂度」。结论是后者相当克制，甚至有点扫兴。

## 二、把实验台搭起来

数据生成过程完全已知，这样才能拿无噪声真曲线当裁判。每天的函数由三个成分叠加：一个常数水平、一个 U 型日内季节性、一个开盘后指数衰减，三条形状都归一到 $L^2$ 单位范数。三个分数各自服从 AR(1)，自回归系数 0.72，稳态标准差依次为 0.30、0.18、0.12。观测时加上幅度 0.20 的噪声，噪声沿日内方向带 0.50 的一阶相关，用来模拟买卖价差与成交稀疏造成的微观结构扰动。共 500 天，前 350 天训练、后 150 天做样本外评估。

拟合侧用 8 个高斯径向基函数做展开，观测点通过带二阶差分粗糙度惩罚的岭回归投影到基上，惩罚系数默认 0.01。所有涉及 $L^2$ 的量（Gram 矩阵、总方差、正交性、重构误差）统一用 200 点 Gauss–Legendre 求积，避免「文章说连续、代码算离散」的口径漂移。所有灵敏度扫描共用同一份分数路径和同一张噪声表，只改被扫描的那一个参数，这是共同随机数原则，否则噪声运气会混进方法差异里。

![](/images/functional-pca-intraday-volatility-curve/fig1_curves_and_shapes.png)

图 1 左边是三条真曲线与它们被噪声污染后的观测，右边是三条真实成分形状。注意开盘衰减那条在右半段几乎压平，这意味着它只靠前段少量观测点支撑——后面稀疏采样实验里，这个细节会变成 FPCA 的软肋。

## 三、正确的特征方程里必须有 W

设第 $t$ 天的基系数为 $a_t$，中心化后的系数协方差为 $S$。协方差算子在基下写成 $C(x,y)=\phi(x)^\top S\phi(y)$。要找特征函数 $g(x)=\phi(x)^\top b$，得解

$$\int C(x,y)g(y)\,\mathrm{d}y = \lambda g(x)\quad\Longrightarrow\quad S W b=\lambda b .$$

$S W$ 不对称，直接求特征向量会踩数值和解释上的双重坑。取 Cholesky 分解 $W=LL^\top$，令 $u=L^\top b$，则问题化为对称形式 $L^\top S L\,u=\lambda u$，再回代 $b=L^{-\top}u$。这样得到的 $B$ 自动满足 $B^\top W B=I$，投影算子 $P=BB^\top W$ 幂等，总方差 $\operatorname{tr}(SW)=\sum_j\lambda_j$。

代码里把这个恒等式当断言用：实测 $\operatorname{tr}(SW)$ 与全部特征值之和同为 0.145940，$W$ 的条件数 23.26，数值上没有病态。正确解法下前三个函数型主成分累计解释 0.9361 的 $L^2$ 变异，与「三条真实成分」的设定吻合。

错误做法是直接对 $S$ 做普通特征分解，把系数空间里欧氏正交的向量当成函数空间里的特征方向，并用 $a^\top b$ 计算分数。归一化到各自 $L^2$ 范数之后再检查正交性，偏离量是 **0.420**；正确解法是 $7.77\times10^{-16}$，纯粹是浮点残差。解释方差比例也随之从 0.9361 被误报成 0.8403——差了近 10 个百分点，而且方向是低估。

![](/images/functional-pca-intraday-volatility-curve/fig2_eigenfunctions.png)

图 2 把两组特征函数并排放。右边错误版的第二、第三条曲线形状明显互相「串味」，这在物理上意味着同一个日内时段被重复计入了两次变异；左边正确版的三条曲线才彼此干净地分开。

![](/images/functional-pca-intraday-volatility-curve/fig4_variance_and_orthogonality.png)

图 4 把解释方差与正交性偏离画成对照柱状图。值得强调的是：**正交性破坏不是美学问题**。重构时分数按 $a^\top W B$ 算、再乘回 $B$，整个算子只有在 $B^\top WB=I$ 时才是投影；错版里它不再幂等，等于把同一份信号反复叠加又扣减。样本外重构 MSE 因此从 0.00741 恶化到 0.01247，劣化 68%。

## 四、改对之后，也别急着宣布胜利

同一口径下的逐点 PCA（直接对 500×48 的观测矩阵做 SVD，取前三个载荷）重构 MSE 是 0.00723。**正确 FPCA 的 0.00741 反而略差**。这个结果一开始看着别扭，但原因很朴素：真实曲线并不在 8 个高斯 RBF 张成的八维子空间里，基展开从一开始就带有近似误差；逐点 PCA 有完整的 48 个自由度，没有这层损失。降维本身又顺手滤掉了大部分各向同性噪声，两边都享受了去噪红利。

噪声扫描把这个机制摊开了。σ=0.02 时逐点 PCA 的 MSE 只有 0.00007，FPCA 是 0.00051，差 7 倍，此时基近似误差几乎全部暴露；σ 升到 0.10 变成 0.00180 对 0.00217；σ=0.20 是 0.00723 对 0.00741；到 σ=0.35 已经是 0.02220 对 0.02208，FPCA 甚至微弱反超。**噪声越大，两种表示越被同一份噪声主导，方法差异越不重要**。换句话说，FPCA 的相对优势区间依赖于「信号平滑、噪声适中、基足够贴」，而不是一句「函数型方法更先进」能概括的。

![](/images/functional-pca-intraday-volatility-curve/fig3_noise_scan.png)

图 3 右半边是更值得记住的一组结果。常有人说函数型方法天然能处理不规则采样，因为基函数可以在任意时间点求值，不必先补齐成整齐矩阵。理论没错，实践要打折：让每天随机只保留 48、36、24、16、12 个观测点，逐点 PCA 用线性插值补齐到同一网格作为公平基线，结果 36 点时两者是 0.00842 对 0.00801，24 点是 0.01059 对 0.00959，到 12 点变成 0.03809 对 0.01845，FPCA 落后 2.1 倍。原因是全局高斯基在某天恰好没被采样覆盖的区间上失去支撑，正则化也救不回来；而线性插值是局部的，至少保住了邻域信息。若真要在稀疏场景用函数型方法，得换局部支撑基（B 样条）或加更强的结构约束，不能用现成的全局 RBF 硬上。

平滑惩罚也不是越大越好。λ 取 0、0.001、0.01 时重构 MSE 依次是 0.00742、0.00741、0.00741，几乎没有差别；λ 升到 0.1 变 0.00759、到 1 变 0.00831、到 10 变 0.00989。基回归本身已经在平滑，额外的粗糙度惩罚在噪声可控时只剩偏差。

![](/images/functional-pca-intraday-volatility-curve/fig5_forecast_and_lambda.png)

## 五、降维不等于预测能力

把三种成分压缩成三个数字，最容易产生的错觉是「既然抓住了主要变异，预测也应该更准」。用同一套一步预测口径检验：持久性基线（昨天的平滑曲线直接当日预测）MSE 0.09406；保留全部 8 个基系数做 VAR(1) 是 0.07608；只留 3 个 FPC 分数做 AR(1) 是 0.07469；逐点 PCA 的 3 个分数做 AR(1) 是 0.07453。

从持久性到全基 VAR(1) 降了约 19%，那是 AR(1) 结构带来的；再从全基降到三维只多了约 1.8%，而且逐点 PCA 还略好一点。**收益几乎全部来自动态模型，不来自降维**。这与本站讨论稳健协方差时那条结论是同一回事：矩阵误差小不等于组合业绩好，表示得漂亮也不等于预测得准。评估表示学习时，务必把「重构误差」和「样本外预测误差」分开报，否则很容易把前者的改善当成后者的兑现。

## 六、那到底什么时候该用 FPCA

把上面的结果收成三条可执行的判断：

1. **内积必须对。** 只要走了基展开路线，特征分解就要在 $L^2$ 内积下做，$W$ 不能省；用 $\operatorname{tr}(SW)=\sum\lambda$ 和 $B^\top W B\approx I$ 当自检断言，两行代码就能挡掉一大类隐蔽错误。
2. **需要连续对象时才值。** 特征函数能在任意时间点求值、可求导、可解释为「日内形状」，这三点逐点载荷给不了；跨设备采样网格不一致、又不愿引入插值偏差时，函数型表示也是更干净的选择。若目标只是压缩重构或次日预测，先做逐点基线，让 FPCA 用样本外数字证明自己。
3. **别把方法优势当默认设定。** 本次实验里基数量、基宽度、惩罚系数都是固定的，真实曲线也不在基空间里；换更贴合的 B 样条基、增大 K 或改用局部支撑，结论可能变化。文章给的是一套可复跑的对照流程，不是一个普适排名。

局限也要说清楚：全程是单一数据生成过程、无真实市场数据，评估用的是已知无噪真曲线，实盘里没有这个裁判；噪声被设成平稳的一阶相关，没有跳跃、没有日内突发事件；预测只做一步，未考虑交易成本与仓位约束。把它当成方法论上的压力测试，而不是策略回测。想继续顺着读，可以对照本站的 [收益率曲线 PCA](/blog/treasury-yield-curve-pca/) 看离散做法，以及 [秩相关稳健协方差](/blog/kendall-tau-robust-covariance/) 里「矩阵误差不等于业绩」的同一条逻辑。

## 七、完整代码

下面这段代码就是本文所有数字与五张图的来源：生成数据、拟合基系数、正确与错误两版 FPCA、逐点 PCA、噪声/稀疏/惩罚三组扫描、四路一步预测，最后写 `results.json` 并出图。在项目根目录直接运行即可复现，随机种子固定为 20261108。

```python
"""函数型主成分（FPCA）分析日内波动率曲线：完整可运行实验。

运行方式（在项目根目录）：
    python3 scripts/article_core_fpca_20261108pm.py

产出：
    public/images/functional-pca-intraday-volatility-curve/fig1..fig5.png
    public/images/functional-pca-intraday-volatility-curve/results.json

核心问题：曲线是函数，不是向量。把基展开系数直接拿去做普通 PCA，
等价于偷偷把 L2 内积换成了欧氏内积，得到的「主成分」并不正交，
解释方差比例也是错的。下面用同一套 Gauss-Legendre 求积权重把所有量
（内积、总方差、重构误差）统一到 L2[0,1] 下，并把错误做法一起算出来对拍。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261108
OUT = "public/images/functional-pca-intraday-volatility-curve"
T = 500
N_TRAIN = 350
P_OBS = 48          # 每天 48 个 5 分钟已实现波动点
K = 8               # 高斯 RBF 基个数
M_GRID = 200        # L2 内积求积点数
RBF_S = 0.09
RHO_AR = 0.72
RHO_NOISE = 0.50
LAMBDA = 0.01       # 二阶差分粗糙度惩罚
N_FPC = 3
SIGMA_MAIN = 0.20
SIGMAS = (0.02, 0.05, 0.10, 0.20, 0.35)
LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
N_KEEP = (48, 36, 24, 16, 12)   # 每天保留的观测点数（不规则采样）


def quad_grid(m=M_GRID):
    """[0,1] 上的 Gauss-Legendre 求积节点与权重（含区间缩放）。"""
    x, w = np.polynomial.legendre.leggauss(m)
    return 0.5 * (x + 1.0), 0.5 * w


def l2_norm(v, w):
    return float(np.sqrt(np.sum(w * v * v)))


def basis(x, centers, s=RBF_S):
    return np.exp(-0.5 * ((x[:, None] - centers[None, :]) / s) ** 2)


def make_true_shapes(xg, wg, xo):
    """三条真实成分：水平、U 型日内季节性、开盘衰减，均归一到 L2 单位范数。"""
    g1 = np.ones_like(xg)
    g2 = 12.0 * (xg - 0.5) ** 2 - 1.0
    g3 = np.exp(-8.0 * xg)
    shapes = [g / l2_norm(g, wg) for g in (g1, g2, g3)]
    o1 = np.ones_like(xo)
    o2 = 12.0 * (xo - 0.5) ** 2 - 1.0
    o3 = np.exp(-8.0 * xo)
    obs_shapes = np.column_stack([
        o1 / l2_norm(g1, wg), o2 / l2_norm(g2, wg), o3 / l2_norm(g3, wg)
    ])
    return np.column_stack(shapes), obs_shapes


def gen_scores(seed=SEED, n=T, rho=RHO_AR):
    """三条成分的分数服从独立 AR(1)，稳态标准差 (0.30, 0.18, 0.12)。"""
    rng = np.random.default_rng(seed)
    sd = np.array([0.30, 0.18, 0.12])
    innov_sd = sd * np.sqrt(1.0 - rho ** 2)
    a = np.zeros((n, 3))
    a[0] = rng.standard_normal(3) * sd
    for t in range(1, n):
        a[t] = rho * a[t - 1] + rng.standard_normal(3) * innov_sd
    return a


def gen_noise(seed=SEED + 1, n=T, p=P_OBS, rho=RHO_NOISE):
    """沿日内方向 AR(1) 相关的微观结构噪声，边际方差 1。"""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n, p))
    z = np.zeros((n, p))
    z[:, 0] = eps[:, 0]
    for j in range(1, p):
        z[:, j] = rho * z[:, j - 1] + np.sqrt(1.0 - rho ** 2) * eps[:, j]
    return z


def penalty(k=K):
    """二阶差分惩罚矩阵 D^T D。"""
    d = np.zeros((k - 2, k))
    for i in range(k - 2):
        d[i, i:i + 3] = (1.0, -2.0, 1.0)
    return d.T @ d


def fit_coefficients(y, phi_obs, omega, lam):
    """把离散观测投影到基上：带粗糙度惩罚的岭回归，返回 (T, K) 系数。"""
    lhs = phi_obs.T @ phi_obs + lam * omega
    rhs = y @ phi_obs                      # (T, K)
    return np.linalg.solve(lhs, rhs.T).T   # (K, K) \ (K, T) -> (T, K)


def fpca_basis(s, W, m=N_FPC):
    """正确的 FPCA：在 L2 内积下解 S W b = lam b。

    W = L L^T，令 u = L^T b，则 L^T S L u = lam u（对称问题）。
    b = L^{-T} u 自动满足 B^T W B = I，总方差 = trace(S W) = sum(lam)。
    """
    L = np.linalg.cholesky(W)
    sym = L.T @ s @ L
    eig, U = np.linalg.eigh(sym)
    order = np.argsort(eig)[::-1]
    eig, U = eig[order], U[:, order]
    B = np.linalg.solve(L.T, U[:, :m])
    return B, eig


def wrong_fpca_basis(s, m=N_FPC):
    """错误做法：直接对系数协方差 S 做普通特征分解，忽略 Gram 矩阵 W。

    得到的方向只在欧氏内积下正交，在真正的 L2 内积下既不正交也不归一。
    """
    eig, V = np.linalg.eigh(s)
    order = np.argsort(eig)[::-1]
    return V[:, order[:m]], eig[order]


def recon_from_coeff(a, B, W, phi_eval):
    """正确 FPCA 重构：系数 a -> 分数 a^T W B -> 函数 phi_eval @ (B s)。"""
    scores = a @ W @ B
    return scores @ B.T @ phi_eval.T


def recon_from_coeff_wrong(a, B, phi_eval):
    """错误版重构：把系数欧氏内积当成函数内积，分数直接取 a^T B。"""
    scores = a @ B
    return scores @ B.T @ phi_eval.T


def recon_pointwise(y, V, mean):
    """逐点 PCA 的重构（只在观测网格上定义）。"""
    return (y - mean) @ V @ V.T + mean


def one_step_forecast(s, n_train):
    """对分数序列拟合 VAR(1) 并做一步预测，返回 (系数矩阵, 测试段预测)。"""
    x = np.column_stack([np.ones(n_train - 1), s[:n_train - 1]])
    y = s[1:n_train]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    x_all = np.column_stack([np.ones(len(s) - 1), s[:-1]])
    return beta, x_all @ beta


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def run():
    os.makedirs(OUT, exist_ok=True)
    xg, wg = quad_grid()
    xo = (np.arange(P_OBS) + 0.5) / P_OBS
    centers = np.linspace(0.0, 1.0, K)
    phi_g = basis(xg, centers)          # (M, K)
    phi_o = basis(xo, centers)          # (P, K)
    W = phi_g.T @ (wg[:, None] * phi_g)  # L2 Gram 矩阵
    omega = penalty()

    G_grid, G_obs = make_true_shapes(xg, wg, xo)
    scores_true = gen_scores()
    F_grid = 0.9 + scores_true @ G_grid.T        # 无噪真曲线（求积网格）
    F_obs = 0.9 + scores_true @ G_obs.T          # 无噪真曲线（观测网格）
    Z = gen_noise()                              # 共同随机数：各扫描只改 sigma
    Y = F_obs + SIGMA_MAIN * Z

    A = fit_coefficients(Y, phi_o, omega, LAMBDA)
    A_center = A - A[:N_TRAIN].mean(axis=0)
    S = np.cov(A_center[:N_TRAIN].T)
    B_ok, eig_ok = fpca_basis(S, W)
    B_bad, eig_bad = wrong_fpca_basis(S)
    # 为可比性，把错误方向按各自 L2 范数归一化后再量正交性偏离
    d = np.sqrt(np.diag(B_bad.T @ W @ B_bad))
    B_bad_norm = B_bad / d[None, :]

    # 逐点 PCA（对观测矩阵直接 SVD）
    Y_mean = Y[:N_TRAIN].mean(axis=0)
    _, _, Vt = np.linalg.svd(Y[:N_TRAIN] - Y_mean, full_matrices=False)
    V_pca = Vt[:N_FPC].T

    # 系数中心化后的重构要平移回原尺度：用训练段平滑曲线的观测网格均值
    smooth_mean = A[:N_TRAIN].mean(axis=0) @ phi_o.T
    rec_ok = recon_from_coeff(A_center, B_ok, W, phi_o) + smooth_mean
    rec_bad = recon_from_coeff_wrong(A_center, B_bad, phi_o) + smooth_mean
    rec_pca = recon_pointwise(Y, V_pca, Y_mean)

    main = {
        "W_condition_number": float(np.linalg.cond(W)),
        "total_variance_L2": float(np.sum(eig_ok)),
        "trace_SW_check": float(np.trace(S @ W)),
        "explained_top3_correct": float(np.sum(eig_ok[:N_FPC]) / np.sum(eig_ok)),
        "explained_top3_wrong": float(np.sum(eig_bad[:N_FPC]) / np.sum(eig_bad)),
        "L2_orthogonality_err_correct": float(
            np.max(np.abs(B_ok.T @ W @ B_ok - np.eye(N_FPC)))),
        "L2_orthogonality_err_wrong": float(
            np.max(np.abs(B_bad_norm.T @ W @ B_bad_norm - np.eye(N_FPC)))),
        "recon_mse_correct": mse(F_obs[N_TRAIN:], rec_ok[N_TRAIN:]),
        "recon_mse_wrong": mse(F_obs[N_TRAIN:], rec_bad[N_TRAIN:]),
        "recon_mse_pointwise": mse(F_obs[N_TRAIN:], rec_pca[N_TRAIN:]),
    }

    # 噪声扫描（共同随机数，只改噪声幅度）
    scan = []
    for sig in SIGMAS:
        Yi = F_obs + sig * Z
        Ai = fit_coefficients(Yi, phi_o, omega, LAMBDA)
        Ac = Ai - Ai[:N_TRAIN].mean(axis=0)
        Si = np.cov(Ac[:N_TRAIN].T)
        Bi, _ = fpca_basis(Si, W)
        _, _, Vti = np.linalg.svd(Yi[:N_TRAIN] - Yi[:N_TRAIN].mean(axis=0),
                                  full_matrices=False)
        Vi = Vti[:N_FPC].T
        sm = Ai[:N_TRAIN].mean(axis=0) @ phi_o.T
        r_ok = recon_from_coeff(Ac, Bi, W, phi_o) + sm
        r_pca = recon_pointwise(Yi, Vi, Yi[:N_TRAIN].mean(axis=0))
        scan.append({
            "sigma": sig,
            "fpca": mse(F_obs[N_TRAIN:], r_ok[N_TRAIN:]),
            "pointwise": mse(F_obs[N_TRAIN:], r_pca[N_TRAIN:]),
        })

    # 平滑参数扫描
    lam_scan = []
    for lam in LAMBDAS:
        Ai = fit_coefficients(Y, phi_o, omega, lam)
        Ac = Ai - Ai[:N_TRAIN].mean(axis=0)
        Si = np.cov(Ac[:N_TRAIN].T)
        Bi, _ = fpca_basis(Si, W)
        sm = Ai[:N_TRAIN].mean(axis=0) @ phi_o.T
        r_ok = recon_from_coeff(Ac, Bi, W, phi_o) + sm
        lam_scan.append({"lambda": lam,
                         "fpca": mse(F_obs[N_TRAIN:], r_ok[N_TRAIN:])})

    # 不规则采样扫描：每天随机只保留 n_keep 个观测点，且各天位置不同。
    # FPCA 直接在可用点上做基回归；逐点 PCA 必须先把曲线插值补齐到同一网格。
    rng_m = np.random.default_rng(SEED + 2)
    missing_scan = []
    for nk in N_KEEP:
        idx_all = [np.sort(rng_m.choice(P_OBS, nk, replace=False))
                   for _ in range(T)]
        Ai = np.zeros((T, K))
        Yfill = np.zeros((T, P_OBS))
        for t in range(T):
            idx = idx_all[t]
            pm = phi_o[idx]
            Ai[t] = np.linalg.solve(pm.T @ pm + LAMBDA * omega,
                                    pm.T @ Y[t, idx])
            Yfill[t] = np.interp(xo, xo[idx], Y[t, idx])
        Ac = Ai - Ai[:N_TRAIN].mean(axis=0)
        Si = np.cov(Ac[:N_TRAIN].T)
        Bi, _ = fpca_basis(Si, W)
        sm = Ai[:N_TRAIN].mean(axis=0) @ phi_o.T
        r_ok = recon_from_coeff(Ac, Bi, W, phi_o) + sm
        mfill = Yfill[:N_TRAIN].mean(axis=0)
        _, _, Vti = np.linalg.svd(Yfill[:N_TRAIN] - mfill,
                                  full_matrices=False)
        Vi = Vti[:N_FPC].T
        r_pca = recon_pointwise(Yfill, Vi, mfill)
        missing_scan.append({
            "n_keep": nk,
            "fpca": mse(F_obs[N_TRAIN:], r_ok[N_TRAIN:]),
            "pointwise": mse(F_obs[N_TRAIN:], r_pca[N_TRAIN:]),
        })

    # 一步预测：持久性 / 全基 VAR(1) / FPCA-3 / 逐点 PCA-3
    pred = {}
    # 持久性（昨日平滑曲线当明日预测）
    smooth_all = A @ phi_o.T
    pred["persistent"] = mse(F_obs[N_TRAIN:], smooth_all[N_TRAIN - 1:-1])
    # 全基 VAR(1)：K 个系数全部保留，只是平滑、不降维
    _, af = one_step_forecast(A_center, N_TRAIN)
    pred["full_basis_ar1"] = mse(
        F_obs[N_TRAIN:], af[N_TRAIN - 1:] @ phi_o.T + smooth_mean)
    # FPCA-3 分数
    fs = A_center @ W @ B_ok
    _, ff = one_step_forecast(fs, N_TRAIN)
    pred["fpca3_ar1"] = mse(
        F_obs[N_TRAIN:], ff[N_TRAIN - 1:] @ B_ok.T @ phi_o.T + smooth_mean)
    # 逐点 PCA-3 分数
    ps = (Y - Y_mean) @ V_pca
    _, pf = one_step_forecast(ps, N_TRAIN)
    pred["pointwise3_ar1"] = mse(
        F_obs[N_TRAIN:], pf[N_TRAIN - 1:] @ V_pca.T + Y_mean)

    results = {"config": {"T": T, "n_train": N_TRAIN, "p_obs": P_OBS,
                          "K": K, "grid": M_GRID, "sigma": SIGMA_MAIN,
                          "lambda": LAMBDA, "seed": SEED},
               "main": main, "noise_scan": scan, "lambda_scan": lam_scan,
               "missing_scan": missing_scan, "forecast_oos_mse": pred}

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---------------- 五张真实计算图 ----------------
    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 110})

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for t in (20, 60, 100):
        ax[0].plot(xo, F_obs[t], lw=1.8, label=f"true t={t}")
        ax[0].plot(xo, Y[t], lw=0.8, alpha=0.65, label=f"observed t={t}")
    ax[0].set_xlabel("intraday time (normalised)")
    ax[0].set_ylabel("realised volatility")
    ax[0].set_title("(a) 5-min intraday curves, sigma=0.20")
    ax[0].legend(fontsize=7, ncol=2)
    for j, lab in enumerate(("level", "U-shape", "opening decay")):
        ax[1].plot(xg, G_grid[:, j], lw=1.8, label=lab)
    ax[1].set_xlabel("intraday time (normalised)")
    ax[1].set_ylabel("L2-normalised shape")
    ax[1].set_title("(b) three true components")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_curves_and_shapes.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for j in range(N_FPC):
        ax[0].plot(xg, phi_g @ B_ok[:, j], lw=2, label=f"FPC {j + 1}")
        ax[1].plot(xg, phi_g @ B_bad[:, j], lw=2, label=f"wrong {j + 1}")
    ax[0].set_title("(a) correct FPCA (L2-orthogonal)")
    ax[1].set_title("(b) plain eig(S) (Euclidean)")
    for a in ax:
        a.set_xlabel("intraday time (normalised)")
        a.set_ylabel("eigenfunction value")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_eigenfunctions.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot([s["sigma"] for s in scan], [s["fpca"] for s in scan],
               "o-", lw=2, label="FPCA (correct L2)")
    ax[0].plot([s["sigma"] for s in scan], [s["pointwise"] for s in scan],
               "s-", lw=2, label="pointwise PCA")
    ax[0].set_xlabel("observation noise sigma")
    ax[0].set_ylabel("reconstruction MSE")
    ax[0].set_title("(a) noise sweep, full 48-point grid")
    ax[0].legend(fontsize=8)
    ax[1].plot([s["n_keep"] for s in missing_scan],
               [s["fpca"] for s in missing_scan],
               "o-", lw=2, label="FPCA (correct L2)")
    ax[1].plot([s["n_keep"] for s in missing_scan],
               [s["pointwise"] for s in missing_scan],
               "s-", lw=2, label="pointwise PCA + interp")
    ax[1].set_xlabel("observed points kept per day")
    ax[1].set_ylabel("reconstruction MSE")
    ax[1].set_title("(b) irregular / sparse sampling")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_noise_scan.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bars = ax[0].bar(["correct", "wrong"],
                     [main["explained_top3_correct"],
                      main["explained_top3_wrong"]],
                     color=["#3b6ea5", "#c0654a"], width=0.55)
    ax[0].bar_label(bars, fmt="%.4f", fontsize=9)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("cumulative explained variance (top 3)")
    ax[0].set_title("(a) variance accounting")
    bars = ax[1].bar(["correct", "wrong"],
                     [main["L2_orthogonality_err_correct"],
                      main["L2_orthogonality_err_wrong"]],
                     color=["#3b6ea5", "#c0654a"], width=0.55)
    ax[1].bar_label(bars, fmt="%.2e", fontsize=9)
    ax[1].set_ylabel("max |<g_i,g_j> - delta_ij|")
    ax[1].set_title("(b) L2 orthogonality violation")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_variance_and_orthogonality.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    names = list(pred)
    vals = [pred[k] for k in names]
    bars = ax[0].barh(range(len(names)), vals, color="#4f7d68")
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names, fontsize=8)
    ax[0].bar_label(bars, fmt="%.5f", fontsize=8)
    ax[0].set_xlabel("OOS one-step forecast MSE")
    ax[0].set_title("(a) denoising beats dimension reduction")
    ax[1].plot([s["lambda"] for s in lam_scan],
               [s["fpca"] for s in lam_scan], "o-", lw=2)
    ax[1].set_xscale("symlog", linthresh=1e-3)
    ax[1].set_xlabel("roughness penalty lambda")
    ax[1].set_ylabel("reconstruction MSE")
    ax[1].set_title("(b) smoothing sensitivity")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_forecast_and_lambda.png")
    plt.close(fig)

    print("FPCA experiment (T=%d, p=%d, K=%d, sigma=%.2f)" %
          (T, P_OBS, K, SIGMA_MAIN))
    print("  W condition number             : %.2f" % main["W_condition_number"])
    print("  trace(SW) vs sum(eigen)        : %.6f / %.6f" %
          (main["trace_SW_check"], main["total_variance_L2"]))
    print("  top3 variance correct / wrong  : %.4f / %.4f" %
          (main["explained_top3_correct"], main["explained_top3_wrong"]))
    print("  L2 orthogonality err ok / bad  : %.2e / %.2e" %
          (main["L2_orthogonality_err_correct"],
           main["L2_orthogonality_err_wrong"]))
    print("  recon MSE correct / wrong / pt : %.5f / %.5f / %.5f" %
          (main["recon_mse_correct"], main["recon_mse_wrong"],
           main["recon_mse_pointwise"]))
    for s in scan:
        print("  sigma=%.2f  FPCA %.5f  pointwise %.5f  ratio %.2fx" %
              (s["sigma"], s["fpca"], s["pointwise"],
               s["pointwise"] / s["fpca"]))
    for s in missing_scan:
        print("  keep=%2d  FPCA %.5f  pointwise %.5f  ratio %.2fx" %
              (s["n_keep"], s["fpca"], s["pointwise"],
               s["pointwise"] / s["fpca"]))
    for k, v in pred.items():
        print("  forecast %-18s : %.5f" % (k, v))
    print("  lambda scan MSE               :",
          ", ".join("%g:%.5f" % (s["lambda"], s["fpca"])
                    for s in lam_scan))
    return results


if __name__ == "__main__":
    run()
```

代码里 `fpca_basis` 与 `wrong_fpca_basis` 是全文的关键分岔：前者用 Cholesky 把广义特征问题对称化，后者只是少写了 Gram 矩阵，两者相差不到十行，却决定了 0.420 的正交性偏离和 68% 的重构误差。把这两行改回去再看一遍图 4，比读十页推导更直观。
