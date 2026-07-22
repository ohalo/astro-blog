---
title: "DCov 距离协方差：用特征映射的核把依赖关系写成距离"
description: "Pearson 只能抓线性、dCor 把「独立」变成可检验量，但两者的数学本质到底是什么？距离协方差（DCov）给出最干净的回答：它等于把数据映射到再生核希尔伯特空间（RKHS）后的协方差，也等于「能量距离」在依赖维度上的投影。本文从零实现 dCov/dCor 与 HSIC（核视角），并用「能量距离」这个真正的分布间距离做两样本检验——把一段行情自动识别为平静态还是危机态（正确率 83%）。附完整 Python 与五类真实陷阱（中阶）。"
publishDate: '2026-07-23'
tags:
  - 量化交易
  - 距离协方差
  - DCov
  - 核方法
  - HSIC
  - 能量距离
  - 非线性依赖
  - 两样本检验
  - Python
language: Chinese
difficulty: intermediate
---

上一篇文章我们聊了**距离相关系数 dCor**——dCor=0 ⟺ 独立，比 Pearson 强在它把「独立性」本身变成可检验的量。但 dCor 背后那个「距离协方差 DCov」到底在算什么，当时一笔带过。本文把这一刀补上，而且补到「能直接落进策略」的程度。

结论先放这：**DCov 的数学本质有两条等价读法**——

1. **核读出**：DCov 等价于把 X、Y 分别映射到某再生核希尔伯特空间（RKHS）后的「协方差」。dCor 就是那个空间里的相关系数。这解释了为什么它能抓非线性：距离的「核」把原始空间卷进了高维特征空间。
2. **能量读出**：DCov 在数学上等价于「能量距离」在依赖维度上的投影。能量距离是一个**真正的分布间距离**（D(P,Q)=0 当且仅当 P=Q），所以它能拿来做**两样本检验**——给你两段行情收益，判定「它们是不是来自同一种分布」。

沿着第二条读法，本文给你一个能直接用的工具：**用能距离把行情切成「平静/危机」两种 regime**（一段含 regime 切换的合成序列上正确率 **83%**）。所有图表均由下文 Python 真实计算，非占位图。

![Pearson 抓不住的，dCor 与 HSIC 在二次依赖上双双看穿：同一回事两种写法，距离视角(dCor)与核视角(HSIC)在二次关系上都能识别 Pearson 的盲区](/images/dcov-nonlinear-dependence/cover.png)

## 一、DCov 到底在算什么：从「距离」而不是「协方差」出发

Pearson 的核心是 `Cov(X,Y)=E[(X-μx)(Y-μy)]`，它惩罚「X 偏离均值的方向和 Y 偏离均值的方向不一致」——天生只对线性敏感。

DCov 换个角度：不看「值」，看「两两之间的距离结构」。

1. 算 X 内部所有点对的距离 `a_ij=|X_i-X_j|`，Y 同理得 `b_ij`。
2. 对距离矩阵做**双中心化**（和 PCA 对核矩阵居中一模一样）：

$$
A_{ij}=a_{ij}-\bar a_{i\cdot}-\bar a_{\cdot j}+\bar a_{\cdot\cdot}
$$

3. 距离协方差就是两个双中心距离矩阵的逐元素内积：

$$
\mathrm{dCov}^2(X,Y)=\frac{1}{n^2}\sum_{i,j}A_{ij}\,B_{ij}
$$

4. 距离相关系数再归一化：

$$
\mathrm{dCor}(X,Y)=\frac{\mathrm{dCov}(X,Y)}{\sqrt{\mathrm{dVar}(X)\,\mathrm{dVar}(Y)}}
$$

下面是从零实现（无第三方依赖）：

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

**关键洞察**：`double_center` 就是 Gram 矩阵的居中公式。`dist_mat` 生成的距离矩阵，本质是一个**核矩阵**（欧氏距离 ↔ 平方指数核）。所以 DCov 已经在「核空间」里做事了——只是这个核是距离诱导出来的特定核。这就自然引出下节。

## 二、核视角：HSIC 把 DCov 的「核」显式写出来

DCov 用的核是「距离核」。如果我们显式换成一个更灵活的核（比如 RBF 高斯核），就得到了**希尔伯特-施密特独立性准则 HSIC（Gretton 2005）**。它和 DCov 是同一棵树的两个分支：

$$
\mathrm{HSIC}(X,Y)=\frac{1}{n^2}\,\mathrm{tr}\bigl(K\,H\,L\,H\bigr)
$$

- `K`、`L`：X、Y 各自的 RBF 核矩阵（用中位数启发式选带宽 γ）；
- `H=I-(1/n)11ᵀ`：中心化矩阵；
- `tr(KHLH)`：被 H 居中后的核矩阵迹——和 DCov 的「双中心距离矩阵内积」是同一操作，只是基函数从距离换成了高斯核。

HSIC=0 ⟺ 在对应 RKHS 里独立。实现：

```python
def sq_dists(X):
    X = np.asarray(X, float).reshape(-1, 1) if np.ndim(X) == 1 else np.asarray(X, float)
    return np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

def rbf_kernel(X, gamma):
    return np.exp(-gamma * sq_dists(X))

def hsic(X, Y, gamma=None):
    X = np.asarray(X, float).reshape(-1, 1) if np.ndim(X) == 1 else np.asarray(X, float)
    Y = np.asarray(Y, float).reshape(-1, 1) if np.ndim(Y) == 1 else np.asarray(Y, float)
    n = len(X)
    dX = np.sqrt(sq_dists(X)); dY = np.sqrt(sq_dists(Y))
    dX[dX == 0] = np.nan; dY[dY == 0] = np.nan
    gx = 1.0 / (2 * np.nanmedian(dX) ** 2)
    gy = 1.0 / (2 * np.nanmedian(dY) ** 2)
    gamma = (gx + gy) / 2 if gamma is None else gamma   # 中位数启发式带宽
    K = rbf_kernel(X, gamma); L = rbf_kernel(Y, gamma)
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H) / (n * n))
```

把 Pearson / dCor / HSIC 在四类关系上摆一起（各 400 点）：

```
线性          Pearson r = +0.9258   dCor = 0.8456   HSIC = 0.0743
二次 Y=X²     Pearson r = +0.0092   dCor = 0.2079   HSIC = 0.0385
圆 X²+Y²     Pearson r = -0.0420   dCor = 0.0387   HSIC = 0.0078
独立(对照)      Pearson r = +0.0425   dCor = 0.0053   HSIC = 0.0004
```

读法：

- **线性**：三者都高（dCor、r 高，HSIC 量纲不同但也明显非零）——正常。
- **二次 Y=X²**：Pearson r≈0.01 判「不相关」，dCor=0.21、HSIC=0.04 都明确说「有关系」。这是 DCov/HSIC 的**主战场**——函数依赖但非单调。
- **圆 X²+Y²**：三者都 ≈0。和 dCor 一样，**径向对称是核方法的共同盲区**（不是 bug，是「对称分布下核交互被抵消」）。
- **独立对照**：三者都 ≈0，正确判「独立」。

注意 HSIC 的绝对值比 dCor 小一个量级——那是核带宽与量纲差异，不是功效差异。**判「有没有关系」看的是「相对独立对照是否显著偏离 0」，不是绝对值大小**（和 dCor 一样要配置换检验）。把 HSIC 的交互核 `K·H·L·H` 画出来，依赖就藏在非对角结构里：

![HSIC 的交互核 K·H·L·H：依赖被写进核矩阵的非对角结构（二次关系），二次依赖下蓝色负贡献与红色正贡献明显不对称](/images/dcov-nonlinear-dependence/hsic_kernel.png)

## 三、能量距离：一个真正的「分布间距离」→ 两样本检验

DCov 还有个更狠的读法。Székely(2002) 证明：**DCov² 与「能量距离」的平方只差一个常数符号**。能量距离本身是一个**度量（metric）**：

$$
\mathcal{E}(P,Q)=\sqrt{2\,\mathbb{E}|X-Y|-\mathbb{E}|X-X'|-\mathbb{E}|Y-Y'|}\ge 0
$$

> $\mathcal{E}(P,Q)=0$ **当且仅当 P=Q**。这是「距离」而不是「系数」——它满足非负性、对称性、三角不等式，能直接回答「两段样本是不是同一种分布」。

这正是两样本检验（two-sample test）的硬通货。在量化里，我们拿它做 **regime 指纹**：

- 事先存好两类参考样本：平静态（低波动零漂移收益）、危机态（高波动负漂移收益）；
- 对滚动窗口的收益序列，分别算它到两个参考样本的能量距离；
- 「离谁更近」就判为哪种 regime。

一段含「平静-危机-平静-危机」切换的合成收益序列（每段 150 点）上跑出来：

```
能量距离两样本分类正确率 = 83.0%
平静窗口 → 平静参考 = 0.00045   危机窗口 → 危机参考 = 0.00072  （同类，距离小）
平静窗口 → 危机参考 = 0.00491   危机窗口 → 平静参考 = 0.00471  （异类，距离大 ~10 倍）
```

同类窗口离自己参考的距离约 **0.0005**，异类窗口离错参考的距离约 **0.0047**——差出一个数量级，分类正确率 **83%**。能量距离在这里就是「分布指纹」：

![能量距离当「分布指纹」：危机窗口离危机参考更近、离平静参考更远，背景阴影标出真实 regime，两曲线明显分离](/images/dcov-nonlinear-dependence/energy_two_sample.png)

实现极简：

```python
def energy_distance(a, b):
    """Székely(2002) 能量距离（样本版，平方）：
    E² = 2E|X-Y| - E|X-X'| - E|Y-Y'|，≥ 0，且 =0 当且仅当同分布"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    Daa = np.abs(a[:, None] - a[None, :])
    Dbb = np.abs(b[:, None] - b[None, :])
    Dab = np.abs(a[:, None] - b[None, :])
    e_xx = 2 * np.sum(np.triu(Daa, 1)) / (na * (na - 1))
    e_yy = 2 * np.sum(np.triu(Dbb, 1)) / (nb * (nb - 1))
    e_xy = np.mean(Dab)
    return float(2 * e_xy - e_xx - e_yy)
```

## 四、诚实边界：两种写法都量「强度」不是「有无」——噪声稀释

和 dCor 一样，DCov/HSIC 量的是**依赖强度**，不是单纯的「有无」。对二次关系 Y=X² 加越来越大的噪声：

```
dCor 随噪声: 0.02→0.236  0.23→0.137  0.44→0.065  0.60→0.049
HSIC 随噪声: 0.02→0.047  0.23→0.022  0.44→0.007  0.60→0.005
```

噪声一上来，两者都被稀释、往 0 靠。**所以「DCov 低」不等于「独立」**，可能只是「强依赖 + 大噪声」。要判有没有关系，老规矩——做置换检验（打乱 Y 顺序、重算、看观测值是否落分布尾部），把 DCov 当效应量、p 值当显著性，别拿硬阈值一刀切。

![噪声稀释：dCor 与 HSIC 随噪声标准差增大同步衰减，说明两者量的是依赖强度而非单纯的「有无」](/images/dcov-nonlinear-dependence/noise_dilution.png)

## 五、在量化里它能干什么（以及不该干什么）

**DCov/dCor/HSIC 能干的：**

1. **因子冗余检测**：两因子 DCov 高、置换 p 极小 → 独立信息少，砍一个降共线。
2. **特征独立性筛查**：进模型前 pairwise 跑 DCov + 置换检验，揪出「被同一隐藏变量驱动」的特征。
3. **非线性领先滞后**：扫 `X_t` 与 `Y_{t-k}`，DCov 峰值位置即非线性领先滞后结构。
4. **regime 指纹（能量距离专用）**：滚动窗口对参考分布算能量距离，做平静/危机/高相关态分类——这是 DCov 比 dCor 多出来的「分布级」能力。

**不该干的：**

1. **别拿 DCov 当「相关性」直接喂组合优化**：它不保符号、不保线性结构，进均值-方差会出怪权。
2. **别指望它抓圆/螺旋/环依赖**：径向对称下核交互被抵消，DCov=HSIC≈0，得上互信息或先做坐标变换。
3. **计算量 O(n²)**：n×n 核/距离矩阵，几千点还行，几十万点得上随机特征近似（Random Fourier Features 近似 HSIC）。
4. **能量距离对位置尺度敏感**：两分布若只是「同形状、差一个漂移」，能量距离也会报「不同」。做 regime 分类时参考样本要覆盖真实漂移结构，否则会把「有漂移的平静市」误判为危机。

## 六、五类真实陷阱（必看）

1. **径向对称盲区**：圆、螺旋、环形依赖 DCov/HSIC≈0，会误判独立。解法：配互信息或先做极坐标/对数变换。
2. **把依赖量当硬阈值**：DCov=0.04 不等于「弱关系」，可能是强关系 + 大噪声。永远配置换 p 一起看。
3. **量纲/带宽陷阱**：HSIC 绝对值随核带宽剧变，不同 γ 下数值不可比；中位数启发式只是默认，极端分布要手调。
4. **样本量太小**：DCov/HSIC 估计方差大，n<100 时置换 p 不稳，结论易翻车。
5. **能量距离的漂移误判**：两分布「同形异位」能量距离也大，做 regime 分类必须让参考样本含真实漂移，否则危机/平静会混淆。

## 结语

距离协方差 DCov 的价值不在「比 Pearson 大」，而在它把**依赖关系写成了距离、写成了核内积、写成了分布间的距离**——三条等价读法（核读出 / 能量读出 / 双中心内积）是同一件事。这让它能做 Pearson 做不了的两件事：**抓非线性依赖**（二次关系一眼看穿）和**做分布级的两样本检验**（能量距离当 regime 指纹，83% 正确率）。但它也不是银弹：径向对称是共同盲区、噪声稀释强度、算力 O(n²)、能量距离对漂移敏感。正确姿势是：**DCov/HSIC 当非线性依赖度量、能量距离当 regime 指纹、置换 p 压显著性、互信息补漏网**。把它和上一篇的 dCor 一起收进你的依赖筛查工具箱——一个管「变量间有没有非线性关系」，一个管「两段行情是不是同一种状态」。

完整可复跑代码与全部图表脚本已随本文生成（`gen_dcov.py`），四类关系的 Pearson/dCor/HSIC 对照、HSIC 交互核、能量距离两样本检验、噪声稀释曲线均由该脚本真实计算。
