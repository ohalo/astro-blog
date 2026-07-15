---
title: "PCA 特征组合：把相关性矩阵的主成分变成可交易因子"
publishDate: '2026-07-15'
description: "PCA 特征组合：把相关性矩阵的主成分变成可交易因子 - halo的技术博客"
tags:
  - 量化交易
  - 投资组合
language: Chinese
---

![PCA 特征组合](/images/pca-eigenportfolio/pca_scree.png)

做量化的人，几乎都从「选几只心仪的股票、配个权重」开始。但当你手上不是 5 只、而是 300 只股票，问题就变了：你不是在选股，而是在和一只看不见的「市场之手」博弈——绝大多数股票的涨跌，都被同一个系统性因子牵着走。如果你无视它，做出来的组合可能在回测里很漂亮，实盘却因为高度暴露在单一市场因子上而剧烈波动。

本文讲一套既朴素又有深度的解法：**用主成分分析（PCA）把相关性矩阵拆成主成分，再把主成分本身变成可交易的特征组合（eigenportfolio）。** 它既是理解「市场到底由什么驱动」的显微镜，也是构造分散化因子、做统计套利、做风险归因的现成工具。全程附完整 Python，数据用可复现的合成示例，并点出 5 类真实陷阱。

## 一、为什么是 PCA，而不是「随便配个权重」

假设我们有一篮子 N 只股票，日收益率矩阵 R（T×N）。传统 Markowitz 框架第一件事是估计协方差 Σ，但 Σ 有 N(N+1)/2 个参数，N=300 时就是 4.5 万个——而样本只有 250 个交易日，参数远多于数据，估计出来的 Σ 大半是噪声（这正是 Ledoit-Wolf 协方差收缩要解决的问题，本专栏已有专文）。

我们换一个视角看 Σ。先标准化（用相关系数矩阵 C 而不是协方差，等价于把每只股票先 z-score），对 C 做特征分解：

$$C = V \Lambda V^T,\quad \Lambda=\text{diag}(\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_N\ge 0)$$

- $\lambda_k$ 是第 k 个主成分（PC）解释的方差大小；
- $v_k$ 是对应的特征向量，它给出了「第 k 个主成分」在 N 只股票上的载荷；
- 把 $v_k$ 当成权重向量，就得到第 k 个**本征组合（eigenportfolio）**：$w_k=v_k$（或按需要归一化、约束为纯多头）。

关键点：**PC1 几乎总是一个近似等权、全为正的多头组合**——它就是「市场因子」的数学化身。后面的 PC 解释剩余的、越来越正交的系统性结构（行业板块、风格因子……）。

![相关性矩阵](/images/pca-eigenportfolio/pca_corr_heatmap.png)

## 二、从相关性矩阵到主成分：直觉与代码

先构造一个能复现真实结构的合成例子：30 只股票分成 3 个板块（各 10 只），板块内相关高、板块间相关低，外加一个对所有股票都起作用的共同「市场」因子。这样 PCA 应该抽出一个解释力极强的 PC1（市场），以及若干捕捉板块结构的 PC。

```python
import numpy as np

def build_corr(n=30, block=10, within=0.55, market=0.45, seed=7):
    rng = np.random.default_rng(seed)
    blocks = [0]*block + [1]*block + [2]*block
    # 板块内正相关的「期望相关」骨架
    base = np.array([[within if blocks[i] == blocks[j] else 0.0
                      for j in range(n)] for i in range(n)])
    corr = base + market                 # 共同市场因子叠加
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2           # 对称化
    # 投影回合法相关矩阵（保证半正定）
    w, v = np.linalg.eigh(corr)
    w = np.clip(w, 1e-6, None)
    corr = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    return corr

corr = build_corr()
evals, evecs = np.linalg.eigh(corr)
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]

explained = evals / evals.sum()
cum = np.cumsum(explained)
print(f"PC1 解释方差: {explained[0]:.1%}")
print(f"前3成分累计: {cum[2]:.1%}")
print(f"PC1 权重符号一致(全正): {np.all(evecs[:,0] > 0)}")
```

运行后会看到：PC1 单独吞掉接近一半的相关性，前 3 个 PC 就能解释绝大部分截面波动；而 PC1 的权重几乎全为正——这正是「市场」本征组合的标志。**看到 PC1 全正时，你其实已经在数学上「重新发现」了市值加权指数。**

![PCA 解释方差](/images/pca-eigenportfolio/pca_scree.png)

## 三、把主成分变成可交易因子

光看懂不够，要能交易。这里有三种常见用法。

### 用法 1：用 PC1 做市场中性化

很多 alpha 因子（动量、价值、质量）都暗藏对市场的暴露 $\beta$。在做横截面选股前，先用 PC1 把市场暴露「减掉」，得到真正的残差 alpha：

```python
# R: 已标准化的收益率矩阵 (T x N)；mkt = R @ evecs[:,0]
mkt = R @ evecs[:, 0]
# 对每个股票收益回归掉市场因子，取残差
beta = (R.T @ mkt) / (mkt @ mkt)
resid = R - np.outer(mkt, beta.T)      # (T x N) 残差收益
# resid 已与 PC1 正交 -> 横截面选股不再被市场 beta 污染
```

这一步等价于「先做市场中性再选股」，能显著压低组合在大盘暴跌时的同步回撤。

### 用法 2：把板块 PC 当行业轮动信号

PC2、PC3 往往对应板块结构（在我们的合成里是三个板块的「相对强弱」）。你可以：

1. 计算每日的 PC 得分 $s_{t,k}=R_t\cdot v_k$；
2. 对 $s_{t,k}$ 做动量/均值回复，决定超配哪个板块、低配哪个板块；
3. 用板块内等权或市值加权落地成多空组合。

```python
# 每日主成分得分 (T x K)
scores = R @ evecs[:, :5]
# 例如：做多得分上升最快的 PC，做空得分下降最快的 PC
signal = np.sign(scores.diff(20).iloc[-1])   # 20日变化方向
```

### 用法 3：用 PC 载荷直接构造分散化组合

如果你想要「尽量不暴露在任意单一共同因子」的组合，可以强制权重与 PC1…PCk 正交：

```python
from scipy.optimize import minimize

K = 3
V = evecs[:, :K]                     # 想压制的共同因子载荷
def obj(w):
    port_var = w @ corr @ w
    return port_var
cons = [{"type": "eq", "fun": lambda w: V.T @ w}]   # 与共同因子正交
cons.append({"type": "eq", "fun": lambda w: w.sum() - 1})
bnds = [(-1, 1)] * n
w0 = np.repeat(1/n, n)
res = minimize(obj, w0, constraints=cons, bounds=bnds)
w_neutral = res.x                    # 分散化、低共同因子暴露的组合
```

注意：正交约束会让权重里有正有负，实盘要处理做空成本与可卖空标的限制（见第七节陷阱）。

## 四、一个最小可回测的 PCA 因子策略

把上面的思路串成一个能回测的例子：每月用过去 250 日的收益率估计相关性矩阵、做 PCA，取 PC2/PC3（避开 PC1 这个市场因子），按「得分变化方向」对板块排序，做多得分上升组合、做空得分下降组合。

```python
def pca_sector_signal(R_window, K=3, lookback=20):
    c = np.corrcoef(R_window, rowvar=False)
    ev, vec = np.linalg.eigh(c)
    o = np.argsort(ev)[::-1]
    vec = vec[:, o]
    scores = R_window @ vec[:, 1:K]          # 跳过 PC1（市场）
    # 板块代理：把 N 只股票按载荷分成 K-1 组
    load = vec[:, 1]                          # 用 PC2 载荷做分组
    long_mask = load > np.median(load)
    short_mask = load <= np.median(load)
    return long_mask, short_mask

# 回测骨架（示意）
for t in monthly_dates:
    R_win = R.loc[:t].tail(250)
    long_m, short_m = pca_sector_signal(R_win)
    w_long = long_m / long_m.sum()
    w_short = short_m / short_m.sum()
    port_ret = (w_long - w_short) @ R.loc[t]   # 多空组合该期收益
```

这套框架的价值不在「立刻赚钱」，而在于它把「市场/板块/风格」这些模糊概念变成了**可度量、可正交、可交易的向量**。

![PC1 本征组合权重](/images/pca-eigenportfolio/pca_eigenportfolio_weights.png)

## 五、和既有方法的联系

- **与 Ledoit-Wolf 协方差收缩**：PCA 也是对协方差降噪，但它是「因子化降噪」——只保留大特征值对应的信号方向；LW 是「平滑降噪」——把整个矩阵朝目标收缩。两者可以叠加：先用 LW 估计 Σ，再做 PCA。
- **与风险平价 / HRP**：HRP 用聚类+层次切分达到分散化；PCA 直接告诉你「分散化要避开的是哪些共同方向」。两者目标一致，PCA 更透明。
- **与统计套利**：协整配对交易是「两两」去共同趋势；PCA 是「全局」去共同趋势，更系统，也更适合多标的篮子。

## 六、收益归因：PCA 因子到底赚的是什么

把组合收益对 PC 得分回归：

```python
# port_ret: 组合逐期收益; scores_t: 同期 PC 得分
coef = np.linalg.lstsq(scores_t, port_ret, rcond=None)[0]
print("各 PC 的收益暴露:", coef)
```

如果你的组合在 PC1 上暴露接近 0、却在 PC2/PC3 上有显著正暴露，说明你赚的是**板块/风格相对强弱的钱**，而不是大盘 beta 的钱——这恰恰是统计套利想要的「与市场低相关」的纯净 alpha。反过来，如果 coef[0] 很大，那你的「PCA 策略」其实只是买了一只伪装成因子选股的市场多头。

## 七、五类真实陷阱（必看）

1. **PC1 全正 ≠ 可做多**：PC1 是市场因子，直接做多它等价于买指数，没什么信息增益；真要做交易，要么用它做中性化、要么避开它做 PC2+。
2. **样本协方差在 N 接近 T 时全是噪声**：N=300、T=250 时特征值的估计误差极大，小主成分（PC10 以后）基本是数值噪声，千万别拿去做信号。务必先降维（只取前 K 个），或用 Ledoit-Wolf 收缩。
3. **主成分旋转不定**：PCA 给出的特征向量在「同一方向簇」内可任意旋转，PC2 和 PC3 的「含义」不一定稳定对应固定板块，跨期可能换含义。用载荷做分组时要做稳定性检验。
4. **正交 ≠ 实盘可交易**：理论组合权重里常有负权重（做空），A 股个股做空受限、美股做空有借券成本，落地必须加可卖空约束或改成板块内多空。
5. **前视偏差**：估计相关性必须用「截至 t 日」的窗口数据，绝不能用未来数据；每月重估时严格用 `R.loc[:t]` 而非全样本。

## 八、结语

PCA 特征组合的美，在于它把「市场由什么驱动」这件事从哲学问题变成了线性代数问题：相关性矩阵一分解，市场因子、板块结构、残差噪声就各归各位。它是因子研究的「地基」——理解了它，你再看 Black-Litterman、风险平价、统计套利，都会多一层「它们其实都在和这些主成分打交道」的透视感。

下一篇我们顺着这条线走下去：均值-方差只用了均值和方差（二阶矩），可真实收益分布有偏度、有峰度，**把三阶、四阶矩也写进目标函数**，组合会变成什么样？
