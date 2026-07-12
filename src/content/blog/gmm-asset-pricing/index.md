---
title: GMM 广义矩估计在资产定价中的应用：用矩条件替代强假设
publishDate: '2026-07-12'
description: GMM 广义矩估计在资产定价中的应用：用矩条件替代强假设 - halo的技术博客
tags:
  - 量化交易
  - 量化专栏
  - 资产定价
  - 计量经济学
  - 因子投资
language: Chinese
difficulty: advanced
---

## 为什么资产定价离不开 GMM

传统计量方法（OLS、MLE）都建立在一堆强假设上：误差正态、同方差、序列无关、分布已知。而金融数据几乎每一条都违反——收益率厚尾、波动率聚集、异方差、序列相关。你用 OLS 跑 CAPM，标准误全是错的；你用 MLE，先得赌对分布形式。

**广义矩估计（Generalized Method of Moments, GMM）**是 Hansen (1982) 给出的解药，它也让 Hansen 拿了 2013 年诺贝尔经济学奖。GMM 的哲学非常简单粗暴：**我不假设分布，我只用经济理论给出的矩条件（moment conditions）。** 资产定价理论天然就是一堆矩条件——欧拉方程 E[m·R]=1 就是最干净的一个。GMM 把"理论上应该成立的期望等式"和"样本里实际算出来的均值"对齐，剩下的交给渐近理论。

本文用一个可控的单因子随机折现因子（SDF）世界，从零实现 GMM，讲清楚：矩条件怎么来、一步 vs 两步估计、过度识别检验 J、以及正确模型和错误模型在定价误差上的天壤之别。

## 一、资产定价的核心矩条件

所有理性资产定价模型都能写成一句话：**存在一个随机折现因子 m，使得任意资产的折现后价格等于其期望。** 对毛收益 R（相对无风险利率的总回报）而言，这就是欧拉方程：

$$E[m_{t+1} R_{t+1}] = 1$$

这个等式对**每一个**资产都成立。如果有 N 个测试资产，就有 N 个矩条件。SDF m 长什么样由模型决定。最常用的**线性 SDF**：

$$m_{t+1} = 1 - b'(f_{t+1} - E[f])$$

其中 f 是定价因子（如市场超额收益、消费增长、Fama-French 因子），b 是待估的因子载荷。b 越大，说明该因子承载越多的定价信息。

于是矩条件写成：

$$g(b) = E\big[(1 - b'(f - \bar f)) R - 1\big] = 0$$

样本版本就是把期望换成样本均值：

$$g_T(b) = \frac{1}{T}\sum_{t=1}^{T} \big[(1 - b'(f_t - \bar f)) R_t - 1\big]$$

```python
import numpy as np

def sdf(b, f):
    """线性随机折现因子 m = 1 - b*(f - E[f])"""
    return 1.0 - b * (f - f.mean())

def moment_conditions(b, f, R_gross):
    """每个资产、每个时刻的矩条件 m*R - 1，返回 (T, N)"""
    m = sdf(b, f)[:, None]
    return m * R_gross - 1.0

def gbar(b, f, R_gross):
    """样本矩向量 g_T(b)，长度 N（每个资产一个）"""
    return moment_conditions(b, f, R_gross).mean(axis=0)
```

## 二、GMM 的目标函数：最小化加权矩

有 N 个矩条件、却只有 k 个参数（这里 k=1，只有 b）。当 N > k 时称为**过度识别（overidentified）**——矩条件比参数多，不可能让所有 gᵢ 同时精确等于 0。GMM 的做法是最小化一个加权二次型：

$$\hat b = \arg\min_b \; g_T(b)' \, W \, g_T(b)$$

W 是一个正定权重矩阵，决定"哪个矩条件更重要"。权重的选择直接影响估计效率：

- **一步 GMM**：取 W = I（单位矩阵），所有矩条件等权。简单，但不是最有效的。
- **两步 GMM**：取 W = S⁻¹，其中 S 是矩条件的协方差矩阵。Hansen 证明这是**渐近最有效**的选择——它自动给噪声大的矩条件更低权重。

下图展示了两条目标函数曲线。真值 b=3（绿虚线），两步 GMM（红线）的最小值点 b̂=2.90 精确锁定真值；一步 GMM（蓝线）也在附近，但曲率不同，说明有效性有差异。

![GMM 目标函数：最小化加权矩条件二次型](/images/gmm-asset-pricing/gmm_objective.png)

```python
def gmm_two_step(f, R_gross, b_grid):
    """两步 GMM：先用单位权重，再用最优权重 S^{-1} 重估。"""
    # 第一步：W = I
    obj_I = np.array([gbar(b, f, R_gross) @ gbar(b, f, R_gross) for b in b_grid])
    b_step1 = b_grid[np.argmin(obj_I)]

    # 用第一步估计构造最优权重 S = (1/T) g'g
    g0 = moment_conditions(b_step1, f, R_gross)
    S = (g0.T @ g0) / len(f)
    Sinv = np.linalg.pinv(S)

    # 第二步：W = S^{-1}
    obj_W = np.array([gbar(b, f, R_gross) @ Sinv @ gbar(b, f, R_gross) for b in b_grid])
    b_step2 = b_grid[np.argmin(obj_W)]
    return b_step1, b_step2, Sinv

b_grid = np.linspace(0.0, 8.0, 500)
# b1, b2, Sinv = gmm_two_step(f, R_gross, b_grid)
# print(f"一步 b̂={b1:.2f}  两步 b̂={b2:.2f}")   # 接近真值 3.0
```

实务中会用 GMM 的解析一阶条件（配合数值优化）而非网格搜索，但网格能最直观地看到目标函数形状，也避免局部最优的坑。

## 三、过度识别检验 J：模型对不对

GMM 最优雅的地方是它自带一个**模型设定检验**——Hansen 的 J 统计量。既然有 N 个矩条件、只用了 k 个自由度，剩下 N−k 个"约束"应该在数据里近似成立。如果模型设定正确，这些残余矩应该只是噪声：

$$J = T \cdot g_T(\hat b)' \, S^{-1} \, g_T(\hat b) \; \sim \; \chi^2(N - k)$$

J 太大 → 拒绝原假设 → **模型被数据否定**（矩条件系统性不成立，说明 SDF 设定错了）。这是 GMM 相对于 OLS 的巨大优势：OLS 只告诉你系数，GMM 还告诉你**整个模型该不该信**。

下图是 2000 次蒙特卡洛模拟得到的 J 统计量经验分布（紫色直方图），叠加理论 χ²(df=9) 曲线（橙线）。两者几乎完美重合，5% 临界值处（红虚线）的经验拒绝率是 **3.6%**，接近名义的 5%——说明在正确设定下，J 检验的规模是准的。

![过度识别检验 J≈χ²(N−k)：模型设定正确则 J 落在分布内](/images/gmm-asset-pricing/j_test_chi2.png)

```python
from scipy import stats

def j_test(b_hat, f, R_gross, Sinv, k=1):
    """Hansen 过度识别检验：J ~ chi2(N - k)。"""
    T, N = len(f), R_gross.shape[1]
    gb = gbar(b_hat, f, R_gross)
    J = T * (gb @ Sinv @ gb)
    dof = N - k
    p_value = 1 - stats.chi2.cdf(J, dof)
    return J, dof, p_value

# J, dof, p = j_test(b2, f, R_gross, Sinv)
# print(f"J={J:.1f}  df={dof}  p={p:.3f}")
# p > 0.05 → 不能拒绝模型；p < 0.05 → 模型被否定
```

**读法**：p 值大（如 0.3）是好消息，意味着模型和数据不矛盾；p 值小（如 0.001）意味着你的 SDF 漏掉了重要因子，横截面定价误差大到无法用噪声解释。

## 四、定价误差：正确模型 vs 错误模型

理论说得再好，最终要落到"能不能给资产定对价"。GMM 拟合出的 SDF 应该让所有资产的矩条件残差 E[m·R]−1 都接近 0。如果一个模型系统性地在某些资产上偏离，那它就是错的。

下图对比了两个模型在 10 个测试资产（按 β 递增排列）上的平均定价误差：

- **绿色（GMM 拟合 SDF）**：误差全部压在 0 附近，随机分布无系统性偏离——模型定价成功。
- **红色（错误模型，b=0 的常数 SDF）**：定价误差随 β 单调递增，呈系统性偏离——因为常数 SDF 无法解释横截面收益差异，高 β 资产被系统性错误定价。

![定价误差对比：正确 SDF 把横截面误差压到≈0，错误模型系统性偏离](/images/gmm-asset-pricing/pricing_errors.png)

```python
def pricing_errors(b, f, R_gross):
    """每个资产的平均定价误差 = E[m*R] - 1。正确模型应≈0。"""
    return moment_conditions(b, f, R_gross).mean(axis=0)

# 正确模型：用 GMM 估计的 b
err_ok = pricing_errors(b2, f, R_gross)
# 错误模型：b=0（常数 SDF，等价于风险中性无因子）
err_bad = pricing_errors(0.0, f, R_gross)
# print(f"正确模型 |误差|均值 {np.abs(err_ok).mean():.4f}")
# print(f"错误模型 |误差|均值 {np.abs(err_bad).mean():.4f}")   # 显著更大
```

这张图是理解 SDF 方法的钥匙：**定价误差的系统性结构，就是模型缺失因子的指纹。** 如果误差随某个特征（规模、账面市值比、动量）单调变化，那这个特征很可能就是你漏掉的定价因子。这也是 Fama-French 从 CAPM 一步步扩展到三因子、五因子的方法论内核。

## 五、GMM 与其他方法的关系

GMM 是一个统一框架，很多熟悉的估计量都是它的特例：

- **OLS** = 矩条件为 E[x·ε]=0 的恰好识别 GMM（矩数=参数数）。
- **IV / 2SLS** = 用工具变量作矩条件的 GMM。
- **MLE** = 矩条件取 score function（对数似然的一阶导）的 GMM。
- **Fama-MacBeth** 两步横截面回归，可以重述为 GMM，而且 GMM 版本的标准误自动修正了 Shanken (1992) 指出的"generated regressor"问题。

换句话说，学会 GMM，你就有了一把能统一解释大半计量工具箱的钥匙。在资产定价里，它让你能在**不假设收益分布**的前提下，估计 SDF、检验因子模型、比较竞争模型——这正是它历久弥新的原因。

## 六、六个真实陷阱

1. **弱识别**：如果因子和收益的协方差很小（弱因子），目标函数在真值附近极平坦，GMM 估计不稳定、标准误巨大。检验前先看目标函数曲率。
2. **权重矩阵估计误差**：两步 GMM 的 S⁻¹ 在小样本里估得很差，反而可能不如一步 GMM。样本短时优先用单位权重或迭代 GMM。
3. **HAC 修正**：收益有序列相关和异方差时，S 必须用 Newey-West（HAC）估计，否则 J 检验和标准误全错。本文合成数据无自相关，实盘务必加。
4. **J 检验的功效有限**：J 检验规模准（不容易假拒绝），但**功效弱**——设定错误时未必总能发现。别把 "J 不显著" 当成 "模型正确" 的铁证。
5. **测试资产的选择**：用什么组合当测试资产（25 个规模/BM 组合 vs 行业组合）会显著改变结论。这就是著名的 "测试资产依赖" 问题。
6. **迭代 vs 两步**：两步 GMM 只迭代一次权重，结果依赖第一步估计；迭代 GMM 反复更新直到收敛，通常更稳健但计算更重。

## 结语

GMM 的魅力在于它的**诚实**：它不假装知道收益分布，只用经济理论给出的矩条件说话；它不只给你系数，还用 J 统计量告诉你"这个模型到底站不站得住"。在资产定价里，这意味着你可以严肃地问一个问题——**我的因子模型，真的把风险定对价了吗？**——并得到一个有统计意义的答案。

从 CAPM 到 Fama-French 再到消费 CAPM，几乎所有严肃的实证资产定价研究都跑在 GMM 之上。理解了 E[m·R]=1 这一个矩条件，你就握住了现代资产定价的主轴。

> 本文所有数据均由单因子线性 SDF 世界自洽合成，用于演示方法论。真实落地需接入测试资产收益（如 Kenneth French 数据库的规模/价值组合）、因子序列，并使用 Newey-West HAC 权重矩阵与迭代 GMM 以处理序列相关和异方差。
