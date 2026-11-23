---
title: "切片逆回归 SIR：用 y 反推 x 的方向，为什么它能把 PCA 打趴下的指数模型在对称 link 上又输给随机猜"
description: "单指数模型 y=f(beta'x)+eps 只要恢复方向 beta，PCA 只用 x 的方差、完全不看 y，方向命中只有 0.2667；SIR 把 y 切片后逐片平均 x，线性单调模型上命中 0.9962、立方模型 0.9969。但 SIR 依赖 E[x|beta'x] 对 beta'x 线性这个设计条件，对称 link t^2 下切片均值相互抵消，命中塌到 0.2368——比随机方向 0.3162 还差，OLS 反而有 0.4596。本文用真值已知实验拆解切片数、样本量、维度与双指数模型，附完整代码与五张真实计算图。"
publishDate: '2026-11-23T09:20:00Z'
tags:
  - 量化
  - 降维
  - 充分降维
  - 因子模型
  - Python
language: zh
difficulty: advanced
image: "/images/sliced-inverse-regression-sir/fig1_links.png"
---

## 一、结论先行

因子模型里一个常见任务是：y 由少数几个 x 的线性组合驱动，但驱动函数 f 未知。模型写成 $y=f(\beta^\top x)+\varepsilon$，$x\in\mathbb{R}^{10}$，目标只是恢复方向 $\beta$，不需要知道 f 长什么样。

主成分分析（PCA）会告诉你 x 方差最大的方向，可它从头到尾没看 y 一眼。真实数据里驱动方向往往不是方差最大的那个——本文实验里 PCA 与真 $\beta$ 的 $|\cos|$ 只有 0.2667，跟随机猜一个方向差不多。

切片逆回归（Sliced Inverse Regression，Li 1991）反过来做：不看 $x\to y$，看 $y\to x$。把 y 排序切成 H 片，每片算 x 的均值，把这些片均值拎出来做广义特征分解。单调 link 上它几乎完美——线性模型 0.9962、立方模型 0.9969。

但 SIR 有一道硬前提：$E[x\mid\beta^\top x]$ 必须对 $\beta^\top x$ 线性（x 椭圆分布时成立）。换成对称 link $t^2$，正 eta 和负 eta 映射到同一个 y，切片均值相互抵消，SIR 命中塌到 0.2368——比随机方向 0.3162 还差，连 OLS 的 0.4596 都不如。这不是数值瑕疵，是方法的结构性盲区。

## 二、SIR 在做什么

把 x 标准化去均值、白化协方差：

$$
z=\Sigma_x^{-1/2}(x-\bar x)
$$

按 y 的大小把样本切成 H 片，每片算白化后 z 的均值 $m_h$，权重是片占比 $p_h=n_h/n$，构造片均值协方差：

$$
M=\sum_{h=1}^{H}p_h\,m_h m_h^\top
$$

取 M 的最大特征向量 $v_1$（在白化空间里），再变回原空间 $\hat\beta=\Sigma_x^{-1/2}v_1$。

直觉是：如果 y 只通过 $\beta^\top z$ 变化，那么 y 的高片和低片里，z 的均值应该沿着同一个方向分别偏向两端。把这些片均值的方向平均起来，就逼近了 $\beta$。这叫「逆回归」，因为它估计的是 $E[z\mid y]$ 而不是 $E[y\mid z]$。

白化是关键一步。它保证 SIR 在 x 各方向尺度不一时仍然正确，也让广义特征问题 $M v=\lambda \Sigma_x v$ 的解落在合适度量下。

## 三、四种 link 的正面对决

![四种 link 函数下 SIR、OLS、PCA 的方向命中对比](/images/sliced-inverse-regression-sir/fig1_links.png)

实验设定：x 从标准正态抽，n=400、p=10，真 $\beta$ 前两个分量 1.0 和 0.5 归一化，噪声 $\sigma=0.5$，切片 H=10，每个场景 200 次重复，用 $|\cos(\hat\beta,\beta)|$ 衡量方向命中。随机猜一个方向的期望是 $1/\sqrt{10}=0.3162$。

- **线性 $t$**：SIR 0.9962、OLS 0.9970、PCA 0.2667。SIR 和 OLS 都几乎完美，PCA 接近随机。f 线性时 OLS 本来就是最优估计，SIR 追平它。
- **单调立方 $t^3$**：SIR 0.9969、OLS 0.9925、PCA 0.2533。f 非线性但单调时，OLS 因为强行线性拟合略有损失，SIR 反超——这正是 SIR 的主场：它不需要知道 f。
- **对称平方 $t^2$**：SIR 0.2368、OLS 0.4596、PCA 0.2252。SIR 彻底失效，甚至低于随机方向。原因如前：$t^2$ 把正负 eta 折到一起，低 y 片对应 eta≈0、高 y 片对应 $|\eta|$ 大但正负对称，片均值的 $\beta$ 分量相互抵消。OLS 虽也受损，但仍能抓到一点线性相关。
- **振荡 $\sin(2t)$**：SIR 0.8751、OLS 0.9000、PCA 0.2774。非单调但周期性的 link 下，SIR 部分失效，OLS 反而更好。这说明「单调」不是充分条件——真正要紧的是 $E[x\mid\beta^\top x]$ 的线性设计条件成立，且 f 不能把不同 eta 值混成一团。即便 x 满足线性设计条件，只要 link 把太多不同的 eta 值压到同一个 y 上，逆回归的信号就会被抹平，这是 SIR 与 OLS 在振荡场景下互换胜负的根本原因。

一句话：SIR 赢的是「单调、单指数、不知道 f」这块地，不是所有非线性。

## 四、切片数的取舍

![切片数与样本量对 SIR 方向命中的影响](/images/sliced-inverse-regression-sir/fig2_slices_n.png)

左图固定 n=400、p=10、立方 link，扫 H=2、4、8、12、20、40：

- H=2：0.9838
- H=4：0.9936
- H=8：0.9963
- H=12：0.9967
- H=20：0.9972
- H=40：0.9971

有意思的是，H 大到 40（每片只有 10 个样本）命中也没掉。传统直觉说「片太多噪声大」，但这项实验里 H=2 已经很强，往上加只是缓慢饱和。原因是 n=400 足够大，片均值的噪声还压不住信号方向。工程上 H 取 10 到 20 是稳妥区间，不必纠结。

## 五、样本量越大越准

右图固定 H=10、p=10，扫 n=50 到 800：

- n=50：0.9539
- n=100：0.9830
- n=200：0.9930
- n=400：0.9967
- n=800：0.9983

SIR 的样本效率很好——n=100、p=10 就已经 0.9830。这和它只估方向、不估 f 有关：无需拟合复杂函数，收敛主要受协方差矩阵估计精度支配，而后者在 n 远大于 p 时很快稳下来。

## 六、维度升高会拖后腿

![维度 p 增长时 SIR 方向命中的衰减](/images/sliced-inverse-regression-sir/fig3_dimension.png)

固定 n=400、H=10，扫 p=5 到 80：

- p=5：0.9986
- p=10：0.9967
- p=20：0.9929
- p=40：0.9847
- p=80：0.9662

维度涨 16 倍，命中从 0.9986 缓慢滑到 0.9662。衰减来自两个源头：白化矩阵 $\Sigma_x^{-1/2}$ 在高维下估计误差累积；片均值协方差 M 的秩最多 H−1=9，p 越大，未被 M 覆盖的零空间维度越高，噪音越容易钻进来。

当 n 逼近甚至小于 p 时，$\Sigma_x$ 本身奇异，必须加 ridge 正则。本文的 `sir_direction` 函数留了 `ridge` 参数，但实验全部在 n≫p 区间，正则设 0——一旦进入 n≤p，那是另一套需要单独验证的问题，不要直接套这里的数字。

## 七、双指数模型也能抓

![双指数模型下 SIR 找到两个方向的主角夹角](/images/sliced-inverse-regression-sir/fig4_two_index.png)

SIR 不限于单指数。设 $y=(\beta_1^\top x)^3+0.8\,\beta_2^\top x+\varepsilon$，n=600、p=10、H=12，取 d=2 个方向，算与主空间的主夹角余弦：

- 方向 1：0.9986
- 方向 2：0.9583
- 随机基准：0.3162

两个方向都接近 1，第二个稍弱但差距明显。M 的秩最多 H−1=11，足够容纳 2 维信号。这验证了 SIR 在多维指数模型上的扩展能力。

## 八、为什么 PCA 在这里帮不上忙

一个自然的疑问是：既然要找 x 里「有用的方向」，PCA 找方差最大的方向不也行吗？答案是，PCA 的目标函数里根本没有 y。

本文实验里真 $\beta$ 只有两个非零分量（1.0 和 0.5），归一化后它落在第 1、第 2 坐标轴张成的平面里；而 x 是各向同性标准正态，所有方向方差都是 1，PCA 的最大特征方向完全由抽样噪声决定。于是 PCA 与真 $\beta$ 的 $|\cos|$ 只有 0.2667，跟随机方向 0.3162 一个量级——这不是 PCA 实现有误，是它问的问题（「x 往哪儿最散」）和我们要的问题（「x 往哪儿变会带动 y」）根本不是一回事。

把 x 换成相关结构明显的情形，PCA 会更稳，但那个「大方差方向」依然可能和驱动 y 的方向无关。量化里这种错位很常见：市值、成交量这类方差最大的因子，未必是收益的主驱动。SIR 把 y 的信息引进来，等于给降维加了一个监督信号，这才是它相对 PCA 的真正优势。

## 九、线性设计条件到底在要求什么

SIR 的命门是 $E[x\mid\beta^\top x]$ 对 $\beta^\top x$ 线性。这句话抽象，展开看很直观。

x 从标准正态抽时，$E[x\mid\beta^\top x=t]=\beta t/\|\beta\|^2$——正好是 t 的线性函数。这个性质叫线性设计条件，椭球对称分布（正态、t 分布、球面均匀分布）都满足。满足它时，$E[z\mid y]$ 的方向就会沿着 $\beta$ 走，切片均值能指向真方向。

对称 link $t^2$ 为什么崩？因为 $y=t^2$ 把 $t$ 和 $-t$ 映射到同一个 y。给定「y 很大」这片，x 的 eta 可能是很大的正数，也可能是很大的负数，两者在 x 空间里沿 $\beta$ 正负抵消。切片均值在 $\beta$ 方向的分量期望为 0，M 矩阵里那个本该最强的信号就没了。实测 SIR 命中 0.2368，连随机方向都不如——因为仅剩的信号全是噪音，比随机猜还糟。

这就划出了 SIR 的适用边界：它适合单调、非对称的 link（线性、立方、指数、logit 这类），不适合偶函数、U 型、周期这类「多对一」映射。如果你的 y 是收益、f 是某种单调递增的风险偏好变换，SIR 大概率能用；如果 y 是绝对收益、平方误差这类对称量，先做个单调性变换或换方法。

振荡 $\sin(2t)$ 的结果（SIR 0.8751、OLS 0.9000）也印证这点：它非单调、多对一，SIR 部分失效；但因为它不像 $t^2$ 那样完全对称抵消，所以还留了大半信号。

## 十、一个样本的全景

![单一样本：y 与真指数的关系，以及切片均值如何描出方向](/images/sliced-inverse-regression-sir/fig5_demo.png)

最后看一个具体样本：n=400、p=10、立方 link，SIR 估出的方向与真 $\beta$ 的 $|\cos|=0.9974$。左图是 y 对真指数的散点，立方曲线被噪声包裹；右图把样本按 y 切片后，画每片在 SIR 方向上的平均得分——从低 y 片到高 y 片单调爬升，正是 SIR 赖以定位 $\beta$ 的信号。

代码在下方，从采样、白化、切片到五张图全部现场计算，固定种子 20261123，改参数即可完整复现本文每一个数字。想试自己的 link 函数，改 `exp_single_index` 里的 `links` 字典即可，但请先想清楚它在不在 SIR 的适用边界内——对称的、周期的函数很可能让方法失效，这时换 OLS 或非线性回归反而更实在。

```python
# -*- coding: utf-8 -*-
"""Sliced Inverse Regression (SIR): when inverse regression finds the index.

Single source of truth for every number and figure in the article.

Single-index model  y = f(beta'x) + eps, x ~ N(0, I_p). The goal is the
direction beta (f is a nuisance). SIR (Li 1991) standardises x, slices y into
H bins, and takes the eigenvectors of the sliced means' covariance under the
whitening metric. It is y-aware, unlike PCA which just finds the largest
variance direction of x.

    z = Sigma_x^{-1/2}(x - xbar)
    M = sum_h (n_h/n) m_h m_h' ,  m_h = mean(z | y in slice h)
    beta_hat = Sigma_x^{-1/2} v_1 ,  v_1 = top eigenvector of M

The catch: SIR relies on the linear-design condition E[x | beta'x] being
linear in beta'x (true for elliptical x). For a symmetric link like f(t)=t^2,
E[z | y] cancels to zero and SIR collapses -- we show this honestly. We also
sweep slices H, sample size n, dimension p, and a two-index model, always
measuring |cos(angle)| with the truth (sign unidentifiable). Fixed seeds keep
exec-replay identical to results.json.
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
OUT = os.path.join(ROOT, "public", "images", "sliced-inverse-regression-sir")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------- SIR core
def sir_direction(X, y, H, d=1, ridge=0.0):
    """Return d estimated directions in the ORIGINAL x space (columns)."""
    n, p = X.shape
    xc = X - X.mean(axis=0)
    cov = xc.T @ xc / n
    # whiten with optional ridge for n<=p
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-10, None)
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(w + ridge)) @ V.T
    Z = xc @ inv_sqrt

    order = np.argsort(y)
    zs = Z[order]
    ys = y[order]
    cuts = np.linspace(0, n, H + 1).astype(int)
    M = np.zeros((p, p))
    for h in range(H):
        lo, hi = cuts[h], cuts[h + 1]
        if hi - lo < 2:
            continue
        mh = zs[lo:hi].mean(axis=0)
        ph = (hi - lo) / n
        M += ph * np.outer(mh, mh)

    ev, evec = np.linalg.eigh(M)
    top = evec[:, ::-1][:, :d]
    return inv_sqrt @ top          # back to original scale


def abs_cos(u, v):
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    den = np.linalg.norm(u) * np.linalg.norm(v)
    return float(abs(u @ v) / den) if den > 0 else 0.0


def principal_angles(U, V):
    """cos of principal angles between two subspaces (columns orthonormal)."""
    Q, _ = np.linalg.qr(U)
    P, _ = np.linalg.qr(V)
    s = np.linalg.svd(Q.T @ P, compute_uv=False)
    return np.clip(s, 0, 1)


def sample_single(n, p, beta, link, sigma, rng):
    X = rng.standard_normal((n, p))
    eta = X @ beta
    y = link(eta) + sigma * rng.standard_normal(n)
    return X, y


def ols_direction(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return b[1:]


def pca_direction(X):
    xc = X - X.mean(axis=0)
    ev, evec = np.linalg.eigh(xc.T @ xc / len(X))
    return evec[:, -1]


def dir_quality(X, y, beta, H, ridge=0.0):
    b_sir = sir_direction(X, y, H, d=1, ridge=ridge)
    b_ols = ols_direction(X, y)
    b_pca = pca_direction(X)
    return {
        "SIR": abs_cos(b_sir, beta),
        "OLS": abs_cos(b_ols, beta),
        "PCA": abs_cos(b_pca, beta),
    }


# ---------------------------------------------------------------- experiments
def exp_single_index():
    """Linear / monotone / symmetric / oscillatory link functions."""
    rng = np.random.default_rng(6001)
    n, p, R, H, sigma = 400, 10, 200, 10, 0.5
    beta = np.zeros(p); beta[0] = 1.0; beta[1] = 0.5
    beta /= np.linalg.norm(beta)
    links = {
        "linear t": lambda t: t,
        "monotone t^3": lambda t: t ** 3,
        "symmetric t^2": lambda t: t ** 2,
        "oscillatory sin(2t)": lambda t: np.sin(2 * t),
    }
    out = {}
    for name, f in links.items():
        acc = {"SIR": [], "OLS": [], "PCA": []}
        for _ in range(R):
            X, y = sample_single(n, p, beta, f, sigma, rng)
            q = dir_quality(X, y, beta, H)
            for k in acc:
                acc[k].append(q[k])
        out[name] = {
            "R": R, "n": n, "p": p, "H": H,
            "mean": {k: float(np.mean(v)) for k, v in acc.items()},
            "sd": {k: float(np.std(v)) for k, v in acc.items()},
            "q05": {k: float(np.quantile(v, 0.05)) for k, v in acc.items()},
        }
    return out


def exp_slices():
    rng = np.random.default_rng(7001)
    n, p, R, sigma = 400, 10, 160, 0.5
    beta = np.zeros(p); beta[0] = 1.0; beta[1] = 0.5
    beta /= np.linalg.norm(beta)
    f = lambda t: t ** 3
    out = {}
    for H in [2, 4, 8, 12, 20, 40]:
        vals = []
        for _ in range(R):
            X, y = sample_single(n, p, beta, f, sigma, rng)
            vals.append(dir_quality(X, y, beta, H)["SIR"])
        out[str(H)] = {"H": H, "R": R, "mean": float(np.mean(vals)),
                       "sd": float(np.std(vals))}
    return out


def exp_sample_size():
    rng = np.random.default_rng(8001)
    p, R, H, sigma = 10, 160, 10, 0.5
    beta = np.zeros(p); beta[0] = 1.0; beta[1] = 0.5
    beta /= np.linalg.norm(beta)
    f = lambda t: t ** 3
    out = {}
    for n in [50, 100, 200, 400, 800]:
        vals = []
        for _ in range(R):
            X, y = sample_single(n, p, beta, f, sigma, rng)
            vals.append(dir_quality(X, y, beta, H)["SIR"])
        out[str(n)] = {"n": n, "R": R, "mean": float(np.mean(vals)),
                       "sd": float(np.std(vals))}
    return out


def exp_dimension():
    rng = np.random.default_rng(9001)
    n, R, H, sigma = 400, 120, 10, 0.5
    f = lambda t: t ** 3
    out = {}
    for p in [5, 10, 20, 40, 80]:
        beta = np.zeros(p); beta[0] = 1.0; beta[1] = 0.5
        beta /= np.linalg.norm(beta)
        vals = []
        for _ in range(R):
            X, y = sample_single(n, p, beta, f, sigma, rng)
            vals.append(dir_quality(X, y, beta, H)["SIR"])
        out[str(p)] = {"p": p, "n": n, "R": R, "mean": float(np.mean(vals)),
                       "sd": float(np.std(vals))}
    return out


def exp_two_index():
    rng = np.random.default_rng(10001)
    n, p, R, H, sigma = 600, 10, 120, 12, 0.5
    b1 = np.zeros(p); b1[0] = 1.0
    b2 = np.zeros(p); b2[1] = 1.0
    B_true = np.column_stack([b1, b2])
    out = {"R": R, "n": n, "p": p, "H": H, "sir": [], "pca": []}
    for _ in range(R):
        X = rng.standard_normal((n, p))
        y = (X @ b1) ** 3 + 0.8 * (X @ b2) + sigma * rng.standard_normal(n)
        U = sir_direction(X, y, H, d=2)
        out["sir"].append(principal_angles(U, B_true).tolist())
        out["pca"].append(principal_angles(pca_direction(X).reshape(-1, 1),
                                           B_true[:, :1]).tolist())
    return out


# ------------------------------------------------------------------- figures
def fig1_links(res):
    names = list(res)
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 5))
    for j, (m, color) in enumerate([("SIR", "#b23a2b"), ("OLS", "#5b8ff9"),
                                    ("PCA", "#9aa6b2")]):
        vals = [res[k]["mean"][m] for k in names]
        err = [res[k]["sd"][m] for k in names]
        ax.bar(x + (j - 1) * w, vals, w, yerr=err, capsize=3,
               color=color, label=m)
    ax.axhline(1 / np.sqrt(10), color="#333", ls=":", lw=1.2,
               label="random direction 1/sqrt(p)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("|cos| with true beta")
    ax.set_title("SIR recovers monotone index directions; symmetric t^2 breaks it")
    ax.legend(fontsize=9)
    fig.tight_layout()
    pth = os.path.join(OUT, "fig1_links.png")
    fig.savefig(pth, dpi=115); plt.close(fig)
    return "fig1_links.png"


def fig2_slices(sl, ss):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    Hs = sorted(int(k) for k in sl)
    a0 = ax[0]
    a0.errorbar(Hs, [sl[str(h)]["mean"] for h in Hs],
                yerr=[sl[str(h)]["sd"] for h in Hs],
                fmt="o-", color="#b23a2b", capsize=3)
    a0.axhline(1 / np.sqrt(10), color="#333", ls=":", lw=1.2)
    a0.set_xlabel("slices H (n=400, p=10)")
    a0.set_ylabel("|cos| with beta")
    a0.set_title("More slices help up to a point, then plateau")
    ns = sorted(int(k) for k in ss)
    ax[1].errorbar(ns, [ss[str(n)]["mean"] for n in ns],
                   yerr=[ss[str(n)]["sd"] for n in ns],
                   fmt="s-", color="#2e7d55", capsize=3)
    ax[1].axhline(1 / np.sqrt(10), color="#333", ls=":", lw=1.2)
    ax[1].set_xlabel("sample size n (p=10)")
    ax[1].set_ylabel("|cos| with beta")
    ax[1].set_title("Direction quality improves with n")
    fig.tight_layout()
    pth = os.path.join(OUT, "fig2_slices_n.png")
    fig.savefig(pth, dpi=115); plt.close(fig)
    return "fig2_slices_n.png"


def fig3_dimension(dim):
    ps = sorted(int(k) for k in dim)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.errorbar(ps, [dim[str(p)]["mean"] for p in ps],
                yerr=[dim[str(p)]["sd"] for p in ps],
                fmt="o-", color="#8a5a2b", capsize=3)
    ax.plot(ps, [1 / np.sqrt(p) for p in ps], "^--", color="#333",
            label="random direction 1/sqrt(p)")
    ax.set_xlabel("ambient dimension p (n=400)")
    ax.set_ylabel("|cos| with beta")
    ax.set_title("SIR degrades as p grows; linear-design noise accumulates")
    ax.legend(fontsize=9)
    fig.tight_layout()
    pth = os.path.join(OUT, "fig3_dimension.png")
    fig.savefig(pth, dpi=115); plt.close(fig)
    return "fig3_dimension.png"


def fig4_two_index(two):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    sir = np.array(two["sir"])
    for j, lab in enumerate(["dir 1", "dir 2"]):
        ax.hist(sir[:, j], bins=30, alpha=0.7, density=True,
                label="SIR " + lab)
    ax.axvline(1 / np.sqrt(two["p"]), color="#333", ls="--", lw=1.4,
               label="random 1/sqrt(p)")
    ax.set_xlabel("principal-angle cosine with true 2-D space")
    ax.set_ylabel("density")
    ax.set_title("Two-index model: SIR finds both directions (d=2)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    pth = os.path.join(OUT, "fig4_two_index.png")
    fig.savefig(pth, dpi=115); plt.close(fig)
    return "fig4_two_index.png"


def fig5_demo():
    """A single monotone-nonlinear sample: y vs beta'x and SIR weight curve."""
    rng = np.random.default_rng(11001)
    n, p, H = 400, 10, 10
    beta = np.zeros(p); beta[0] = 1.0; beta[1] = 0.5
    beta /= np.linalg.norm(beta)
    X = rng.standard_normal((n, p))
    y = (X @ beta) ** 3 + 0.5 * rng.standard_normal(n)
    eta = X @ beta
    bidx = np.argsort(np.argsort(y))
    slice_id = np.minimum((bidx * H) // n, H - 1)
    b_sir = sir_direction(X, y, H, d=1)
    scores = X @ (b_sir / np.linalg.norm(b_sir))
    slice_mean = [float(scores[slice_id == h].mean()) for h in range(H)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].scatter(eta, y, s=10, color="#5b8ff9", alpha=0.5)
    ax[0].set_xlabel("true index beta'x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Single-index data: y = (beta'x)^3 + noise")
    ax[1].plot(range(H), slice_mean, "o-", color="#b23a2b")
    ax[1].set_xlabel("y slice (low to high)")
    ax[1].set_ylabel("mean of SIR score")
    ax[1].set_title("Sliced means trace the index: |cos|=%.3f"
                    % abs_cos(b_sir, beta))
    fig.tight_layout()
    pth = os.path.join(OUT, "fig5_demo.png")
    fig.savefig(pth, dpi=115); plt.close(fig)
    return "fig5_demo.png", abs_cos(b_sir, beta)


def main():
    links = exp_single_index()
    sl = exp_slices()
    ss = exp_sample_size()
    dim = exp_dimension()
    two = exp_two_index()
    f5, f5cos = fig5_demo()
    r = {
        "setup": {"seed": SEED, "model": "y=f(beta'x)+eps", "sigma": 0.5,
                  "random_dir": 1 / np.sqrt(10)},
        "single_index": links,
        "slices": sl,
        "sample_size": ss,
        "dimension": dim,
        "two_index": {
            "R": two["R"], "n": two["n"], "p": two["p"], "H": two["H"],
            "sir_dir1_mean": float(np.mean(np.array(two["sir"])[:, 0])),
            "sir_dir2_mean": float(np.mean(np.array(two["sir"])[:, 1])),
            "random_dir": 1 / np.sqrt(two["p"]),
        },
        "demo_cos": f5cos,
    }
    r["figs"] = [fig1_links(links), fig2_slices(sl, ss), fig3_dimension(dim),
                 fig4_two_index(two), f5]
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

```
