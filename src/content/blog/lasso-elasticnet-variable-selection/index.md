---
title: "Lasso 与弹性网络变量选择：用 L1 惩罚把几百个因子压成可交易少数"
description: "你有 500 个因子、2000 个样本，直接全量回归——回测里 R² 漂亮，样本外一塌糊涂。这不是数据问题，是自由度把噪声当成了信号。Lasso 用 L1 惩罚把系数逐个压到 0，弹性网络再补一道相关性缓冲，把「几百个因子」压成「十几个真有信息量的少数」。本文用合成的多因子面板（仅 15 个真信号 + 成组相关）从零跑出系数路径、稀疏度阶梯与 OOS R² 对比，并诚实披露：L1 解决的是过拟合，不是帮你印钞（高阶）。"
publishDate: '2026-07-19'
tags:
  - 量化交易
  - 因子选股
  - 变量选择
  - Lasso
  - 弹性网络
  - 过拟合
  - 正则化
  - Python
language: Chinese
difficulty: intermediate
---

你有 500 个因子、2000 个交易日的样本。你兴奋地把它们一股脑塞进线性回归，用最小二乘求出每个因子的权重，回测 R² 高达 0.7——然后拿到样本外，R² 立刻变成 −0.5。

这不是玄学，是**自由度**在作怪：参数个数（500）快赶上样本量（2000），最小二乘会把每一丝噪声都「完美解释」成信号。这种模型在训练集上刀刀见血，在实盘里刀刀扎自己。

**正则化**的思路很朴素：给「系数别太大」加一道成本，逼模型只保留真正有用的因子。**Lasso（L1 惩罚）** 的特殊之处在于——它能把不重要因子的系数**精确压到 0**，顺手完成「变量选择」。在这之上，**弹性网络（ElasticNet）** 用 L1+L2 混合，缓解因子间高度相关时 Lasso 乱选的毛病。

本文用一套自洽合成的多因子面板（500 因子中仅 15 个真信号，且因子成组相关），从零复现系数路径、稀疏度阶梯与样本外 R² 对比，并点明六类真实陷阱。

## 一、为什么「全因子」是毒药

最小二乘的目标是无惩罚地最小化残差平方和：

$$\min_{\beta}\ \sum_{t=1}^{T}(y_t - X_t\beta)^2$$

当 $p$（因子数）接近 $T$（样本数），解会无限放大系数去拟合噪声。直观后果：训练误差趋近于 0，但样本外误差爆炸。我们的合成实验里，全因子线性回归的 **OOS R² = −0.539**——比「永远预测均值」还差。

正则化给目标加惩罚项：

$$\min_{\beta}\ \sum_{t}(y_t - X_t\beta)^2 + \lambda\,\Omega(\beta)$$

- **Ridge（L2）**：$\Omega=\|\beta\|_2^2$，把系数往 0 挤但**不归零**，保留全部因子；
- **Lasso（L1）**：$\Omega=\|\beta\|_1$，会产生**稀疏解**，自动挑因子；
- **ElasticNet**：$ \Omega = \alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2 $，L1 管稀疏、L2 管相关。

## 二、几何直觉：菱形为什么能「卡」出稀疏

二维里，L2 的约束域是**圆**，L1 的约束域是**菱形**。最优解往往落在约束域边界与等高线相切处——圆和任意方向等高线相切都很难正好压在坐标轴上，所以 Ridge 系数普遍非零；菱形有四个尖角**正好在坐标轴上**，相切极易「卡」在轴上，于是某些系数被干脆设为 0。这就是 Lasso 能做变量选择的几何本质。

## 三、系数路径：λ 越大，因子一个个消失

我们对 500 个因子跑 Lasso，扫一遍 $\lambda$，画出每条系数随 $\log\lambda$ 变化的轨迹：

![Lasso 系数路径：λ 增大，系数被逐个压向 0，粗线为 15 个真信号](/images/lasso-elasticnet-variable-selection/lasso_coef_path.png)

可以清楚看到：小 $\lambda$ 时系数散乱（噪声因子也有非零权重）；$\lambda$ 增大，弱信号因子先被压到 0，只剩少数粗线（我们埋的 15 个真信号）顽强存活。变量选择就这么发生了。

## 四、ElasticNet：用 OOS 误差选 λ

Lasso 在因子**高度相关**时会随机挑其中一个、丢掉其他（不稳定）。ElasticNet 用 `l1_ratio=0.5` 把 L1 和 L2 揉在一起，既稀疏又稳。关键一步是**用样本外误差选 λ**，而不是看训练误差：

![ElasticNet OOS-MSE 在适中 λ 处见底，用测试集而非训练集选 λ](/images/lasso-elasticnet-variable-selection/elasticnet_oos_mse.png)

训练 MSE 一路下降（过拟合），测试 MSE 却在适中 $\lambda$ 处见底后反弹。选 λ 必须盯**测试曲线的最低点**，这正是图里虚线标出的位置。

## 五、稀疏度阶梯

把「被保留的非零因子个数」对 $\lambda$ 画出来，是一条阶梯——每上一阶就砍掉一批因子：

![稀疏度阶梯：λ 越大，选中的因子越少，参考线为真信号 15 个](/images/lasso-elasticnet-variable-selection/sparsity_vs_lambda.png)

注意一个诚实的事实：Lasso 选出的非零个数**不必等于**真信号数 15。它选的是「在惩罚下仍显著」的因子，可能多选噪声、也可能漏掉弱信号。稀疏 ≠ 完美还原真相，只是把维度压到可管理。

## 六、OOS R² 对比：惩罚回归真的防过拟合

同样 500 因子，四种方法在各自最优 λ 下的样本外表现：

![全因子 vs Ridge vs Lasso vs ElasticNet 的 OOS R²，惩罚法显著更好](/images/lasso-elasticnet-variable-selection/oos_r2_compare.png)

我们的数字：**全因子 −0.539 / Ridge −0.207 / Lasso +0.136 / ElasticNet +0.134**。无惩罚法在样本外崩盘，L1 类方法把 R² 拉回正区间。这就是正则化的全部价值——不是更高收益，是**更稳的样本外泛化**。

## 七、金融里不能用随机 CV：用滚动时序切分选 λ

上面的 OOS R² 用了随机 `train_test_split`，但金融数据有**时间顺序与自相关**。随机打乱会让「未来信息泄露进训练」，选出的 λ 偏乐观。正确做法是**滚动/扩张窗口（walk-forward）**：用第 1..k 段训练、k+1 段验证，滑窗重复，取验证误差最低的 λ。下面这段把随机切替换成时序切：

```python
# 用滚动窗口选 λ，杜绝偷看未来（示意）
def walk_forward_lambda(X, y, grid, n_train=1200, step=300):
    best_a, best_err = None, 1e9
    start = 0
    while start + n_train + step <= len(y):
        tr = slice(start, start + n_train)
        va = slice(start + n_train, start + n_train + step)
        for a in grid:
            m = Pipeline([("sc", StandardScaler()),
                          ("m", Lasso(alpha=a, max_iter=5000))])
            m.fit(X[tr], y[tr])
            err = np.mean((y[va] - m.predict(X[va])) ** 2)
            if err < best_err:
                best_err, best_a = err, a
        start += step
    return best_a
```

时序切分下选出的 λ 通常比随机切**更大**（更保守），因为它没法靠打乱顺序「蹭」未来的相关性。这是落地时必做的一步。

## 八、它在量化流水线里的位置

Lasso/弹性网络不是「策略」本身，而是**前置过滤器**，常见三处用法：

1. **因子预筛**：几百个原始因子先过一遍 Lasso，把真正非零的十几二十个送进组合优化，降低自由度、提升稳定性；
2. **ML 特征选择**：在 XGBoost / 神经网络之前用 L1 线性模型挑特征，去掉冗余共线项，既降维又提速；
3. **信号去噪**：对截面多股票的横截面回归做稀疏化，只保留「超额收益里真有解释力的因子轴」。

它和「主成分分析（PCA）」是互补的：PCA 把维度压成互不相关的综合因子但**完全不可解释**；Lasso 压成**可解释的少数原始因子**但可能丢失相关结构。实战里常先 Lasso 选因子、再对选出的因子做 PCA 去相关。

## 九、从零实现的 Python 代码

```python
import numpy as np
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# —— 1. 合成多因子面板：500 因子、仅 15 个真信号、成组相关 ——
rng = np.random.default_rng(20260719)
N, P, K = 2000, 500, 15
n_groups = 50
group = np.repeat(np.arange(n_groups), P // n_groups)
base = rng.normal(0, 1, (N, n_groups))
X = np.array([0.75 * base[:, group[j]] + 0.65 * rng.normal(0, 1, N) for j in range(P)]).T
X = (X - X.mean(0)) / X.std(0)
beta = np.zeros(P)
beta[rng.choice(P, K, replace=False)] = rng.uniform(-1.2, 1.2, K)
signal = X @ beta; signal /= signal.std()
y = (signal + rng.normal(0, 1/0.45, N)); y /= y.std()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=7)

# —— 2. 扫描 λ，对比 OOS R²（标准化是必须的）——
def oos_r2(maker, grid):
    best, bp = -1e9, None
    for a in grid:
        m = maker(a); m.fit(Xtr, ytr)
        r2 = 1 - np.sum((yte - m.predict(Xte))**2) / np.sum((yte - yte.mean())**2)
        if r2 > best: best, bp = r2, a
    return best, bp

grid = np.logspace(-3, 0, 40)
r2_lasso, _ = oos_r2(lambda a: Pipeline([("sc", StandardScaler()),
    ("m", Lasso(alpha=a, max_iter=5000))]), grid)
r2_ridge, _ = oos_r2(lambda a: Pipeline([("sc", StandardScaler()),
    ("m", Ridge(alpha=a))]), np.logspace(-4, 2, 40))
r2_en, _ = oos_r2(lambda a: Pipeline([("sc", StandardScaler()),
    ("m", ElasticNet(alpha=a, l1_ratio=0.5, max_iter=5000))]), grid)

# —— 3. 用最优 Lasso 选出因子 ——
final = Pipeline([("sc", StandardScaler()),
    ("m", Lasso(alpha=grid[np.argmax([oos_r2(lambda a: Pipeline([('sc',StandardScaler()),('m',Lasso(alpha=a,max_iter=5000))]), grid)[0] for _ in [0]])], max_iter=5000))])
final.fit(Xtr, ytr)
selected = np.where(final.named_steps['m'].coef_ != 0)[0]
print(f"Lasso OOS R²={r2_lasso:.3f}  选中因子数={len(selected)}")
```

> 注：第 3 步为示意，`alpha` 应取步骤 2 中 `r2_lasso` 对应的最优值。生产里用 `GridSearchCV` 或步骤七的滚动切分选 λ 更稳。

## 十、为什么能一次扫 60 个 λ：坐标下降与热启动

你可能会问：Lasso 没有闭式解，上面却一口气扫了 60 个 λ 画路径，难道每次都从头解一遍？答案是**坐标下降（Coordinate Descent）+ 热启动（Warm Start）**。

坐标下降的思路极朴素：固定其他所有系数，只对一个系数 $\beta_j$ 求最优，循环往复直到收敛。对 Lasso，单坐标的更新有**闭式软阈值（soft-thresholding）**解：

$$\beta_j \leftarrow S\!\left(\frac{1}{N}\sum_{i}x_{ij}(y_i - \hat y_{i}^{( -j)}),\ \lambda\right),\quad S(z,\lambda)=\text{sign}(z)(|z|-\lambda)_+$$

每个坐标一次闭式更新，整轮 O(p)，几轮就收敛。更妙的是**热启动**：从大到小扫 λ 时，把上一个 λ 的解当作下一个 λ 的初值——相邻 λ 的解只差一点点，几乎一步到位。这就是为什么 `sklearn` 的 `Lasso` 配 `path` 能瞬间画出整条系数路径，也解释了第三节那张图为何便宜。

落地提醒：坐标下降对**特征尺度极度敏感**（软阈值的阈值 λ 是绝对量），所以标准化不是「锦上添花」而是「必须」。没标准化就跑坐标下降，等于让量纲替你决定哪些因子先被砍。

## 十一、六类真实陷阱（诚实声明）

1. **L1 不保证盈利**：它只保证「更少过拟合」，不保证策略赚钱。OOS R²=0.14 仍是弱信号。
2. **相关性导致乱选**：因子高度共线时，Lasso 会随机保留其中一个、丢弃其余真信号——此时必须用 ElasticNet 或先正交化。
3. **标准化是前提**：L1 惩罚对系数尺度敏感，不标准化会让量纲大的因子被无辜惩罚。务必先 `StandardScaler`。
4. **λ 必须用 OOS 选**：用训练误差选 λ 等于没正则化。滚动时序切分 / 交叉验证才是正道。
5. **稀疏假设本身是一种先验**：若真因子有几百个且都弱，Lasso 会砍掉大部分，反而欠拟合。
6. **截面 vs 时序**：用「横截面多股票」做变量选择，和「时序多日期」做，结论可能完全不同；两者混用会污染惩罚。

## 结论

Lasso 与弹性网络是你面对「因子爆炸」时的第一道闸门：它不预测收益方向，只负责**把噪声因子挡在门外**，让后续的组合优化建立在更干净的信号上。本次合成实验里，它把样本外 R² 从 −0.54 拉回 +0.14——这正是「少即多」在量化里的含义。但请记住：稀疏是手段，泛化才是目的，印钞从来不是正则化的承诺。
