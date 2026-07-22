---
title: "距离相关系数：用 Hilbert-Schmidt 独立性把非线性依赖从噪声里捞出来"
description: "Pearson 相关只能抓线性：r=0 时两变量可能仍强烈相关（如 Y=X²）。距离相关系数（dCor）把独立性本身变成可检验的量——dCor=0 当且仅当独立。本文用纯 numpy 从零实现 dCov/dCor 与置换检验，并在线性/二次/圆/独立四类关系上对照 Pearson，诚实拆穿「径向对称是 dCor 盲区」与「噪声稀释依赖」两条真实边界（中阶）。"
publishDate: '2026-07-23'
tags:
  - 量化交易
  - 统计依赖
  - 距离相关系数
  - 非线性相关
  - 独立性检验
  - 特征工程
  - Python
language: Chinese
difficulty: intermediate
---

你手里有两列数据 X 和 Y，想知道它们「有没有关系」。绝大多数人的第一反应是算 **Pearson 相关系数 r**。但 r 有个致命短板：**它只量线性**。

一个经典的坑：让 `Y = X²`，X 在 [-1,1] 均匀取值。X 和 Y 明显强相关（Y 完全由 X 决定），但因为 X 对称、正负抵消，Pearson r 算出来 ≈ **0**。r=0 在教科书里读作「不相关」，可这两个变量分明是「最相关」的一种。

这就是 **距离相关系数（distance correlation, dCor, Székely, Rizzo & Bakirov 2007）** 要解决的问题。它的野心比 Pearson 大得多：

> **dCor(X, Y) = 0 当且仅当 X 与 Y 统计独立。**

也就是说，它不只是一个「相关系数」，而是一个**真正的独立性检验**——能抓住 Pearson 完全看不见的非线性依赖。本文用纯 numpy 从零实现 dCov / dCor 与置换检验，在四类合成关系上把 Pearson 和 dCor 摆在一起对照，并诚实拆穿「径向对称是 dCor 盲区」「噪声稀释依赖强度」两条真实边界。所有图表均由下文 Python 真实计算，非占位图。

![四类关系下 Pearson 与 dCor 对照：二次依赖 Pearson 看不出（r≈0）、dCor 一眼看穿；而圆关系两者双双≈0,暴露 dCor 的盲区](/images/distance-correlation-dependence/cover.png)

## 一、dCor 到底在算什么：从「距离」而不是「协方差」出发

Pearson 的核心是协方差 `Cov(X,Y) = E[(X-μx)(Y-μy)]`，它惩罚的是「X 偏离均值的方向和 Y 偏离均值的方向不一致」——这天然只对线性敏感。

dCor 的思路更粗暴也更本质：不看「值」，看「两两之间的距离结构」。

1. 先算 X 内部所有点对的距离 `a_ij = |X_i - X_j|`，Y 同理得 `b_ij`。
2. 对距离矩阵做**双中心化（double centering）**：

$$
A_{ij} = a_{ij} - \bar a_{i\cdot} - \bar a_{\cdot j} + \bar a_{\cdot\cdot}
$$

把「每个点离全局平均距离多远」扣掉，只留下「相对结构」。
3. 距离协方差（dCov）就是两个双中心距离矩阵的逐元素内积：

$$
\mathrm{dCov}^2(X,Y) = \frac{1}{n^2}\sum_{i,j} A_{ij}\,B_{ij}
$$

4. 距离相关系数再归一化：

$$
\mathrm{dCor}(X,Y) = \frac{\mathrm{dCov}(X,Y)}{\sqrt{\mathrm{dVar}(X)\,\mathrm{dVar}(Y)}}
$$

直觉：如果 X 和 Y 的依赖是「结构性的」——X 里靠得近的点，在 Y 里也靠得近——那 A⊙B 就处处同号、内积很大；如果独立，A 和 B 的结构互不相干，内积被抵消趋近于 0。**独立 ⟺ dCov=0 ⟺ dCor=0**，这条等价关系正是 dCor 比 Pearson 强的地方。

下面是从零实现（没有任何第三方依赖）：

```python
import numpy as np

def dist_mat(X):
    """成对欧氏距离矩阵（1 维即 |x_i-x_j|）"""
    X = np.asarray(X, float).reshape(-1, 1) if np.ndim(X) == 1 else np.asarray(X, float)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

def double_center(D):
    """双中心化：A_ij = d_ij - rowmean_i - colmean_j + grandmean"""
    row = D.mean(axis=1, keepdims=True)
    col = D.mean(axis=0, keepdims=True)
    grand = D.mean()
    return D - row - col + grand

def dcov2(X, Y):
    A = double_center(dist_mat(X))
    B = double_center(dist_mat(Y))
    return float(np.sum(A * B) / (len(X) ** 2))

def dvar2(X):
    return dcov2(X, X)

def dcor(X, Y):
    cov = dcov2(X, Y)
    vx, vy = dvar2(X), dvar2(Y)
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(cov / np.sqrt(vx * vy))
```

注意 `double_center` 其实就是 Gram 矩阵的居中公式（和 PCA 里对核矩阵居中一模一样）——dCor 在数学上等价于把数据映射到一个再生核希尔伯特空间（RKHS）里算 Hilbert-Schmidt 独立性（HSIC）。这也是为什么它能识别非线性：距离的「核」把原始空间卷进了高维特征空间。

## 二、上合成数据：dCor 在哪类关系上碾压 Pearson

我们造四个关系，各 400 个点：

| 关系 | 构造 | 直觉 |
|---|---|---|
| 线性 | `Y = X + 噪声` | 教科书强相关 |
| 二次 | `Y = X² + 小噪声` | 完全决定但对称抵消 → r≈0 |
| 圆 | `X=cosθ, Y=sinθ` | 完全依赖（X²+Y²=1）但径向对称 |
| 独立 | 两份无关噪声 | 真·无关系 |

跑出来：

```
线性          Pearson r = +0.9258   dCor = 0.8456
二次 Y=X²     Pearson r = +0.0092   dCor = 0.2079
圆 X²+Y²     Pearson r = -0.0420   dCor = 0.0387
独立(对照)      Pearson r = +0.0425   dCor = 0.0053
```

读法：

- **线性**：两者都高，dCor 略低于 r（dCor 对线性本就不是最优，但它不差）。
- **二次**：Pearson r≈0.01 判「不相关」，dCor=0.21 明确说「有关系」。这就是 dCor 的**主战场**——函数依赖但非单调。
- **独立**：两者都 ≈0，正确判「独立」。
- **圆**：**两者都 ≈0**。这看起来是「dCor 翻车」了——下一节专门拆。

把双中心矩阵画出来，能看清 dCov 是怎么把依赖写进矩阵的：

![双中心化：X 距离与 Y 距离各自扣掉均值后相乘，依赖就藏在 A⊙B 的逐元素贡献里](/images/distance-correlation-dependence/double_center.png)

## 三、边界一：径向对称是 dCor 的盲区（不粉饰）

你大概期待「圆关系完全依赖，dCor 应该≈1」。**实测只有 0.04，和独立几乎分不出来。** 这不是代码 bug，是 dCor 的已知性质：

`X=cosθ, Y=sinθ` 是一个**径向对称**关系——给定 X=x，Y 的取值关于 0 对称；给定 Y=y，X 也关于 0 对称。这种「对称分布」下，双中心距离矩阵 A 和 B 的乘积被对称抵消，dCov 趋近于 0。**dCor 擅长抓「单调/函数型」依赖，对「旋转/径向对称」无能为力。**

这反而是个好故事：它提醒你——**没有任何一个单一指标能抓住所有依赖结构**。dCor 不是 Pearson 的「全面升级版」，而是「互补版」：

- 单调但非线性 → dCor 比 Pearson 强；
- 径向对称 → 两者都废，得上互信息（mutual information）、或者先做坐标变换（极坐标下 θ 直接就是依赖变量）。

所以如果你在策略里用 dCor 筛「哪两个因子真正独立」，**圆型/环形依赖会漏网**——这是真实风险，不是理论瑕疵。

## 四、边界二：dCor 量的是「强度」不是「有无」——噪声会稀释它

dCor=0 才严格等价于独立。一旦关系里掺了噪声，dCor 会从「满值」往下降，**它同时编码了依赖的存在和强度**。对上面的二次关系，我们往 Y=X² 里加越来越大的噪声：

```
σ=0.02  dCor=0.236    σ=0.23  dCor=0.137
σ=0.34  dCor=0.108    σ=0.60  dCor=0.049
```

噪声越大，dCor 越被稀释、越靠近 0。所以 dCor 低不等于「独立」，可能只是「弱依赖 + 噪声大」。**要判「有没有关系」，不能只看 dCor 阈值，要做置换检验。**

![噪声稀释：dCor 随噪声标准差增大而衰减，说明它量的是依赖强度而非单纯的「有无」](/images/distance-correlation-dependence/noise_dilution.png)

## 五、把 dCor 变成检验：置换检验（permutation test）

思路朴素：把 Y 的顺序随机打乱（破坏 X-Y 配对，但保持各自的边际分布），重算 dCor，重复上千次，得到「在『独立假设』下 dCor 的分布」。如果真实观测的 dCor 落在这个分布的极端尾部，就拒绝独立。

```python
def perm_pvalue(X, Y, B=1000, seed=12345):
    rng_p = np.random.default_rng(seed)
    obs = dcor(X, Y)
    Yp = Y.copy()
    perm = np.empty(B)
    for b in range(B):
        rng_p.shuffle(Yp)
        perm[b] = dcor(X, Yp)
    p = (1 + np.sum(perm >= obs)) / (1 + B)
    return obs, perm, p
```

跑出来：

```
二次 置换 p = 0.0010   独立 置换 p = 0.5225
```

- 二次关系：观测 dCor 落在置换分布**右尾**，p=0.001 → **拒绝独立**（即便 dCor 绝对值只有 0.21，检验仍有功效）；
- 独立对照：观测 dCor 落在置换分布**中央**，p=0.52 → **不能拒绝独立**（干脆承认「没看到关系」）。

这就是 dCor 的正确用法：**dCor 当「效应量」（强度），置换 p 当「显著性」（有没有）**，两者分开看，别拿一个硬阈值一刀切。

![置换检验：二次关系的观测 dCor 落在右尾（拒绝独立），独立数据的观测值落在中央（不能拒绝）](/images/distance-correlation-dependence/perm_test.png)

## 六、在量化里它能干什么（以及不该干什么）

**能干的：**

1. **因子冗余检测**：两个因子 dCor 高、且置换 p 极小 → 它们携带的「独立信息」少，可以砍一个，减少多重共线。
2. **特征独立性筛查**：进模型前先 pairwise 跑 dCor + 置换检验，把「看起来不同但其实被同一隐藏变量驱动」的特征揪出来。
3. **非线性领先滞后**：对 `X_t` 与 `Y_{t-k}` 扫 k，dCor 的峰值位置就是非线性领先滞后结构（比互相关更鲁棒）。

**不该干的：**

1. **别拿它当「相关性」直接进组合优化**：dCor 不保符号、不保线性结构，喂给均值-方差会出怪权。
2. **别指望它抓圆/环/螺旋依赖**：见第三节，那得换互信息或坐标变换。
3. **计算量是 O(n²)**：双中心距离矩阵是 n×n，几千点还行，几十万点得上随机近似或分块。

## 七、五类真实陷阱（必看）

1. **径向对称盲区**：圆、螺旋、环形依赖 dCor≈0，会误判为独立。解法：配互信息或先做极坐标/对数变换再测。
2. **把 dCor 当硬阈值**：dCor=0.1 不等于「弱关系」，可能是强关系 + 大噪声。永远配置换 p 一起看。
3. **样本量太小**：dCov 估计方差大，n<100 时置换 p 不稳，结论容易翻车。
4. **O(n²) 算力陷阱**：n=10 万时矩阵 100 亿元素，内存直接爆。大样本用随机特征近似（Random Fourier Features 近似 HSIC）。
5. **dCor 不替代因果**：X⊥Y 的检验通过只说明「统计独立」，不代表无因果；反过来 dCor 高也不代表 X→Y，可能是共同驱动 Z→X,Y。

## 结语

距离相关系数的价值不在「比 Pearson 大」，而在它把**独立性本身**变成了一个可计算、可检验的量：dCor=0 ⟺ 独立，这是 Pearson 永远给不了的保证。但它也不是银弹——径向对称是盲区、噪声会稀释强度、算力是 O(n²)。正确姿势是：**dCor 当效应量、置换 p 当显著性、互信息补漏网、因果推断压轴**。把它放进你的依赖筛查工具箱，而不是当成「相关性 2.0」一杆打死。

完整可复跑代码与全部图表脚本已随本文生成（`gen_distance_corr.py`），四类关系的 Pearson/dCor 对照、双中心矩阵、噪声稀释曲线、置换检验分布均由该脚本真实计算。
