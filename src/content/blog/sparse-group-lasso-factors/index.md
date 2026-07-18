---
title: "稀疏组 Lasso 变量选择：把几百个因子压成少数可解释分组"
description: "你有 600 个因子、30 个经济含义分组，但只有 6 个组真有信号——普通 Lasso 会把整组零碎系数逐个挑出来，破坏分组结构、也难解释。稀疏组 Lasso(SGL) 用「先组级、再组内」的复合 L1 惩罚，把变量选择变成「整组进/整组出」+「组内精炼」两步。本文用高维(N<P)合成面板从零实现 SGL 的 FISTA 近端梯度求解，5 折交叉验证选 λ，对照 OLS/Lasso/GroupLasso：OLS 样本外 R² 崩到 −3.58（过度记忆），SGL 拉回 +0.18，并找回 86% 真信号组的系数，附完整 Python 与四类真实陷阱（高阶）。"
publishDate: '2026-07-19'
tags:
  - 量化交易
  - 变量选择
  - 稀疏组 Lasso
  - 正则化
  - 高维回归
  - 因子模型
  - 机器学习
  - Python
language: Chinese
difficulty: intermediate
---

你手上压着一张**几百列因子表**：估值、动量、质量、波动率、另类数据……它们天然成组——「估值组」里埋着 BP、EP、SP、CFP，「动量组」里是 1M/3M/12M 收益率。麻烦来了：**只有少数几个组真有预测力**，而且组内还只有部分成员有效。

普通 **Lasso** 怎么办？它用 L1 惩罚把系数**逐个**压到 0。结果是：同一估值组的 BP 被保留、EP 被清零——分组结构被撕碎，事后你没法说「我选了估值组」，只能说「我选了 BP 这一个孤零零的因子」。更糟的是，组内的相关性会让 Lasso 在「留谁舍谁」上变得不稳定、难解释。

**稀疏组 Lasso（Sparse Group Lasso, SGL）** 把问题想得更干净：

- **先组级**：整组一起进、整组一起出——一个组里只要有一个成员有用，整组就保留；
- **再组内**：组保留后，组内再用 L1 把没用的成员清零。

两步合起来，变量选择的结果天然是「**少数可解释分组 + 组内精炼**」，而不是一堆散落的孤点。

> 本文用一份**高维合成面板（N=480 样本、P=600 因子、60 组）**从零实现 SGL 的 FISTA 近端梯度求解，用 5 折交叉验证 + 1-SE 规则选 λ，并和 OLS / Lasso / Group Lasso 正面对比。结论先放：在高维下普通 OLS 样本外 R² 直接崩到 **−3.58**（它在记忆训练集），而 SGL 把它拉回 **+0.18**。

## 一、为什么这里是「高维」，而不只是「多变量」

关键区别：**样本数 N 小于变量数 P**（这里 N=480 < P=600）。

一旦 N<P，普通最小二乘就没有唯一解，OLS 会去**精确记住训练样本**——训练误差为 0，但样本外一塌糊涂。我们造的数据：

- 600 个因子分 60 组（每组 10 个），**组内高度相关**（模拟真实因子表的同质性）；
- 只有 **6 个组**含真信号，且组内仅 3~5 个成员非零；
- 信噪比让「真信号」只占一小部分，其余全是噪声因子。

```python
import numpy as np

rng = np.random.default_rng(20260719)
N, P, G, GS = 480, 600, 60, 10
group = np.repeat(np.arange(G), GS)              # 每个因子归到哪一组
base = rng.normal(0, 1, (N, G))
X = np.empty((N, P))
for j in range(P):
    X[:, j] = 0.78 * base[:, group[j]] + 0.62 * rng.normal(0, 1, N)   # 组内相关
X = (X - X.mean(0)) / X.std(0)

beta_true = np.zeros(P)
signal_groups = rng.choice(G, 6, replace=False)  # 仅 6 组有信号
for g in signal_groups:
    members = np.where(group == g)[0]
    pick = rng.choice(members, rng.integers(3, 6), replace=False)
    beta_true[pick] = rng.uniform(-1.2, 1.2, len(pick))

y = (X @ beta_true); y /= y.std()
y = y + rng.normal(0, 1.0/0.55, N); y /= y.std()
```

## 二、SGL 的目标函数：组级 + 组内 复合惩罚

SGL 在普通最小二乘损失上，加**两层**惩罚（Simon et al., 2013）：

$$
\min_{\beta}\;\underbrace{\frac{1}{2n}\lVert y - X\beta\rVert^2}_{\text{拟合}}
+ \lambda\Bigg[\underbrace{\alpha\sum_{g}\sqrt{p_g}\,\lVert\beta_g\rVert_2}_{\text{组级 L2 (整组)}}
+ \underbrace{\frac{1-\alpha}{2}\lVert\beta\rVert_2^2}_{\text{组内 L2 (保凸)}}\Bigg]
$$

- 第一项 $\sum_g \sqrt{p_g}\lVert\beta_g\rVert_2$ 是**组级 L2 惩罚**：它不直接把系数压 0，而是**让整组一起缩小/放大**；配合后文的近端算子，它会把「不值得保留的组」整体清零（整组出）。
- 第二项 $\lVert\beta\rVert_2^2$ 是组内 L2，作用是**保证目标函数严格凸**（否则组近端算子没有唯一闭式解），$\alpha$ 越大越接近纯稀疏。
- $\alpha=0.95$ 是经典取值：几乎纯稀疏，但保留一点点 L2 让优化稳。

**为什么不直接用 L2（岭）？** 岭只缩小不归零，P=600 时它保留全部 600 个因子，完全不稀疏、不可解释。**为什么不直接用 Group Lasso（纯组惩罚）？** 它整组保留或整组清零——一旦保留某组，组内所有 10 个成员全留下，噪声也跟着进来。**SGL 的妙处**就在「组级决定进/出，组内 L1 再精炼」，两步缺一不可。

## 三、从零实现：FISTA + 闭式组近端算子

SGL 没有 OLS 式解析解，但组级 L2 项有**闭式近端算子**（proximal operator），所以我们用**近端梯度法**（这里用带动量加速的 FISTA）：

对每个组 $g$，设 $z$ 是上一步的梯度下降结果，$\tilde z_g = z_g/(t\cdot s)$，则：

$$
\beta_g = \max\!\left(0,\; 1 - \frac{\lambda\alpha\sqrt{p_g}}{\lVert \tilde z_g\rVert}\right)\tilde z_g
$$

一句话：**组范数不够大，整组清零；够大，则整体软阈值收缩**。组内再没有额外 L1 项——「组内清零」其实来自这个软阈值在 $\lVert\tilde z_g\rVert$ 接近阈值时把个别分量压到 0。完整求解如下（带 warm start，沿 λ 扫描时把上一 λ 的解当作初值，提速明显）：

```python
import numpy as np

def fit_sgl(Xd, yd, lam, alpha, groups, beta0=None, max_iter=900, tol=1e-6):
    nn = Xd.shape[0]; p = Xd.shape[1]
    ug = np.unique(groups)
    sizes = np.array([np.sum(groups == g) for g in ug])
    XtX = Xd.T @ Xd; Xty = Xd.T @ yd
    L = np.linalg.eigvalsh(XtX / nn).max()       # Lipschitz 常数
    t = 1.0 / L
    s = lam * (1 - alpha) + 1.0 / t
    w_g = lam * alpha * np.sqrt(sizes)            # 每组阈值系数
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    beta_old = beta.copy()
    for k in range(1, max_iter + 1):
        momentum = (k - 1) / (k + 2)              # FISTA 动量
        y_k = beta + momentum * (beta - beta_old)
        grad = (XtX @ y_k - Xty) / nn
        z = y_k - t * grad
        zt = z / (t * s)
        u = np.zeros(p)
        for gi, g in enumerate(ug):
            idx = np.where(groups == g)[0]
            zg = zt[idx]; norm = np.linalg.norm(zg)
            if norm > 1e-12:
                u[idx] = max(0.0, 1.0 - w_g[gi] / norm) * zg
        beta_old = beta.copy(); beta = u
        if np.linalg.norm(beta - beta_old) / (np.linalg.norm(beta_old) + 1e-9) < tol:
            break
    return beta
```

把 λ 从大到小扫一遍，画每条**组系数路径**：

![稀疏组 Lasso 系数路径：组被逐个整组清零](/images/sparse-group-lasso-factors/sgl_group_path.png)

红线是 6 个真信号组（起点高、衰减慢），灰线是噪声组（早早被压到 0）。能看到惩罚一加大，灰线成片塌到横轴——这就是「**整组清零**」的直观样子。

## 四、λ 怎么选：5 折交叉验证 + 1-SE 规则

高维下 λ 不能拍脑袋。标准做法是 **5 折交叉验证**：把数据分成 5 份，轮流留一份当验证集，其余训练，记样本外 R²，取平均；再用 **1-SE 规则**——在「最优 λ 的 CV 误差 − 1 倍折间标准误」以上区间内，挑**最稀疏**（λ 最大）的那个，避免过配。

![各方法 5 折 CV R² 对比](/images/sparse-group-lasso-factors/sgl_oos_r2.png)

三条曲线清清楚楚：

| 方法 | 5 折 CV R² 峰值 | 说明 |
|---|---|---|
| **SGL (α=0.95)** | **0.10** | 组结构 + 稀疏，防过拟合最优 |
| Group Lasso（纯组） | 0.10 | 同样好，但组内不精炼 |
| Lasso（无组） | 0.08 | 撕裂分组，略逊 |
| OLS 全量 | **−3.58** | 高维下记忆训练集，样本外崩盘 |
| Ridge | −2.22 | 缩小但不归零，仍过拟合 |

注意 **OLS 的 −3.58**：它把训练集「背」下来了，碰到新数据误差反而放大——这正是高维必须正则化的铁证。SGL 把 R² 拉回正区间。

## 五、它到底选出了什么：稀疏阶梯与真值还原

沿 λ 增大，活跃组数和活跃系数数同步塌缩：

![稀疏阶梯：组级与组内双重稀疏同步发生](/images/sparse-group-lasso-factors/sgl_sparsity.png)

在 CV 选出的 λ\*≈0.0032 处，模型把 **60 组砍到 20 组**、600 个系数压到 200 个，并**召回了 86% 真信号组的系数**（recall=0.864），噪声组的误留率也从「全留」降到 31%。再看估计系数 vs 真系数：

![SGL 估计系数 vs 真系数（红=真信号组，灰=噪声组）](/images/sparse-group-lasso-factors/sgl_est_vs_true.png)

红点（真信号组）基本落在对角线上，灰点（噪声组）紧贴原点——SGL 成功把信号找回来、把噪声压住。对比纯 Lasso 那条 0.08 的 CV 曲线，SGL 凭「组结构先验」在高维下更稳定地定位真信号。

## 六、四类真实陷阱（落地前必读）

1. **组的定义决定一切**：SGL 的「组」是你的先验结构。组划错了（把本该分开的因子塞一组、或该成组的拆散），惩罚就惩罚错地方。先用领域知识定组，别拿聚类硬分。
2. **组内相关会稀释组内 L1 的清零力**：组内高度相关时，SGL 倾向「整组留、组内难清零」——这时 α 调高（更偏 L1）或先做组内正交化（如 PCA 预处理）更有效。
3. **λ 用 CV 选，但别迷信 CV 峰值**：1-SE 规则比裸峰值更稳。裸峰值常挑到「刚好过拟合那一根」，样本外反而抖。
4. **N<P 时评估必须在独立测试集**：我们用一份同分布独立测试集算 OOS R²。若在训练集上算 R²，OLS 会显示「完美 1.0」——那是插值，不是预测。

## 七、小结与落地路径

稀疏组 Lasso 把「从几百因子里挑少数可交易、可解释的分组」这件事，从拍脑袋变成了**有结构约束的凸优化**：

- **组级 L2** 负责「整组进/出」——保住经济含义分组；
- **组内精炼** 负责「组内挑有用的」——最终落在一小撮可解释系数；
- **交叉验证 + 1-SE** 负责「别过拟合」——高维下的安全阀。

真实落地时，把 synthetic 数据换成你的因子面板：先用 IC/单因子测试粗略定组，再跑 SGL 拿稀疏权重，最后把「保留的组 + 组内系数」喂进组合优化或信号合成。它不保证盈利，但能让你**在几百个因子里，清晰地知道自己信的是哪几组、为什么**——这本身就是量化研究最稀缺的东西。

> 文中所有图表均由上述 Python 代码在合成数据上真实计算生成；数字仅用于演示方法，实盘请以你自己的因子与样本外验证为准。
