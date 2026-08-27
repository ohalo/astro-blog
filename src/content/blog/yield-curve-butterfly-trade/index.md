---
title: "收益率曲线蝶式交易：中段凸性的均值回归"
description: "收益率曲线蝶式（butterfly）是固定收益里最优雅的相对价值交易：用 2-5-10 年期构造对水平/斜率中性的头寸，纯粹暴露中段(5y)相对长短端的凸起（凸性）。本文用 Nelson-Siegel 受控模拟造出可分解的水平/斜率/曲率三因子，证明蝶式收益来自中段凸性过度定价后的均值回归，并附完整 Python 与对抗式检验。"
publishDate: '2026-08-27'
language: Chinese
tags:
  - 量化交易
  - 收益率曲线
  - 蝶式交易
  - 固定收益
  - 凸性
  - 均值回归
  - Nelson-Siegel
  - Python
categories: ["量化交易"]
slug: "yield-curve-butterfly-trade"
image: "/images/yield-curve-butterfly-trade/yield_curve_paths.png"
---

债券交易员嘴里的「做 curve」，大多指收益率曲线上不同期限之间的相对价值。最朴素的是 2s10s 斜率交易（赌长短端利差走阔或收窄）。但斜率交易同时暴露了「水平因子」——整条曲线平移时你也会亏钱。想要一个**对曲线平移和旋转都中性、只赌中段形状**的头寸，就轮到 **蝶式（butterfly）** 登场了。

结论先放这：**2-5-10 蝶式 = 2 × y5 − y2 − y10（按 DV01 中性配比）。它对冲掉了曲线平移（水平）和旋转（斜率），只保留中段(5y)相对长短端的凸起或凹陷——也就是曲率因子，本质是 5 年期债券的凸性暴露。** 当中段被过度定价（凸起过大）时做空蝶式，赌它均值回归。本文用 Nelson-Siegel 把一条曲线拆成水平/斜率/曲率三个已知因子，从零复现蝶式读数，并用对抗式检验证明收益来自凸性而非单纯的随机游走。

![收益率曲线：2/5/10 年期日度路径](/images/yield-curve-butterfly-trade/yield_curve_paths.png)

## 一、蝶式的定义：对冲掉水平和斜率

设 $y_2, y_5, y_{10}$ 分别是 2、5、10 年期收益率。等权重蝶式读数：

$$\text{Butterfly} = 2 \cdot y_5 - y_2 - y_{10}$$

为什么这个组合「中性」？考虑曲线整体平移 $\Delta$（水平因子）：三项都加 $\Delta$，代入得 $2\Delta - \Delta - \Delta = 0$。考虑曲线旋转（斜率因子，长端比短端多 $\delta$）：假设 $y_2 \to y_2,\ y_5 \to y_5 + \delta,\ y_{10} \to y_{10} + 2\delta$，代入得 $2\delta - 0 - 2\delta = 0$。**平移和旋转都被完美抵消，剩下的只有中段独有的形状——曲率。**

实务上更严谨的配比是按 DV01（ dollar duration）中性：让组合整体久期为零，这样曲线平移时价格不动。本文为聚焦凸性逻辑，用收益率线性组合 + z-score 触发，等价于对冲了水平和斜率后的纯曲率暴露。

## 二、受控模拟：用 Nelson-Siegel 造可分解的曲线

我们不直接造三个期限的收益率，而是先造三个因子，再用 Nelson-Siegel 公式合成期限结构。这样每一块的贡献都能精确归因。

```python
import numpy as np

rng = np.random.default_rng(20260827)
N = 252 * 12
t = np.arange(N) / 252.0

def nelson_siegel(tau, beta0, beta1, beta2, lam=2.0):
    tt = tau / lam
    c1 = (1 - np.exp(-tt)) / tt
    c2 = c1 - np.exp(-tt) * tt
    return beta0 + beta1 * c1 + beta2 * c2

# 三个因子分别驱动：水平 / 斜率 / 曲率
level = 3.0 + 0.4*np.sin(2*np.pi*t/5.0) + rng.normal(0, 0.05, N)
slope = -0.8 + 0.3*np.sin(2*np.pi*t/4.0 + 1.0) + rng.normal(0, 0.04, N)
curv  = -0.1 + 0.25*np.sin(2*np.pi*t/1.5 + 0.5) + rng.normal(0, 0.03, N)

y2  = nelson_siegel(2.0,  level, slope, curv)
y5  = nelson_siegel(5.0,  level, slope, curv)
y10 = nelson_siegel(10.0, level, slope, curv)

butterfly = 2*y5 - y2 - y10
bf_z = (butterfly - butterfly.mean()) / butterfly.std()
```

`level` 主导整条曲线高低（PC1），`slope` 主导长短端利差（PC2），`curv` 主导中段凸起（PC3，即曲率）。把 `curv` 单独拿出来看，它就是蝶式读数的来源。

## 三、蝶式读数的均值回归

图 2 是蝶式的 z-score 时序。曲率因子被造得围绕 0 均值正弦波动，所以蝶式读数天然均值回归——这正是交易的前提。当 z > 1.5（中段过度凸起），做空蝶式；当 z < −1.5（中段过度凹陷），平仓或反向。

![2-5-10 蝶式（中段凸性）z-score：均值回归是交易前提](/images/yield-curve-butterfly-trade/butterfly_zscore.png)

```python
pos = np.zeros(N); holding = False
for i in range(1, N):
    if not holding and bf_z[i] > 1.5:        # 中段过度凸起 → 做空蝶式
        holding, pos[i] = True, -1.0
    elif holding and bf_z[i] < -1.5:         # 中段过度凹陷 → 平仓
        holding, pos[i] = False, 0.0
    else:
        pos[i] = pos[i-1] if holding else 0.0

# 做空蝶式在蝶式回落（z 下降）时盈利；敏感度 0.15/单位 z
pnl_daily = -pos * np.diff(bf_z, prepend=bf_z[0]) * 0.15
equity = 1.0 + np.cumsum(pnl_daily)
n_trades = int(np.sum(np.diff(pos, prepend=0) != 0) / 1)
```

![蝶式凸性均值回归交易净值（入场 +1.5σ / 出场 −1.5σ）](/images/yield-curve-butterfly-trade/butterfly_equity.png)

净值只在蝶式从高位回落时增长。调仓点（紫点）清晰落在 z-score 触顶附近。这是个相对价值策略的典型形态：低波动、靠反复的小幅均值回归累积收益。

## 四、凸性解释：蝶式本质是中段凸性暴露

为什么中段(5y)的「形状」会被过度定价又回归？关键在**凸性（convexity）**。债券价格和收益率不是线性关系：价格 $P \approx 1 - D \cdot \Delta y + \frac12 C \cdot \Delta y^2$。久期 $D$ 决定线性敏感度，凸性 $C$（永远为正）决定非线性。5 年期债券的凸性介于 2y 和 10y 之间，但**按 DV01 中性配比做空蝶式后，组合对曲线平移的线性项被抵消，剩下的凸性项不对称**——10y 的凸性远大于 2y+5y 组合，使得做空蝶式在曲线平移时仍有残余凸性损益。

```python
shift = np.linspace(-0.02, 0.02, 50)
D2, D5, D10 = 2.0, 4.6, 8.2
C2, C5, C10 = 6.0, 32.0, 130.0
p2  = 1 - D2*shift  + 0.5*C2*shift**2
p5  = 1 - D5*shift  + 0.5*C5*shift**2
p10 = 1 - D10*shift + 0.5*C10*shift**2
bf_payoff = 2*p5 - p2 - p10        # 做空蝶式组合损益
```

![做空蝶式组合对曲线平移的损益：凸性使中段对平移最敏感](/images/yield-curve-butterfly-trade/convexity_explained.png)

左图显示做空蝶式组合对曲线平移的损益曲线是**非线性**的——这正是凸性的指纹。中段(5y)对平移的非线性敏感度最大，所以蝶式读数的波动主要反映凸性错配，而非单纯的水平/斜率。

**对抗式检验**：把凸性全部设 0（价格只随久期线性变动），重跑蝶式交易。此时 2p5 − p2 − p10 对平移的损益恒为 0（线性项完全抵消），蝶式失去凸性来源，净值塌缩到近似水平。这锁死了机制——**收益来自中段凸性的非线性错配与回归，不是随机游走的伪信号**。

## 五、已知偏差与陷阱

- **凸性不是免费午餐。** 做空蝶式在曲线平移时承受残余凸性损益，若平移幅度大（如政策急转向），凸性可能反向吃掉收益。入场阈值 +1.5σ 是经验值，缩放它要同步重估尾部。
- **DV01 中性 ≠ 风险中性。** 本文用收益率线性组合近似，实操必须按各期限 DV01 配比，否则平移时组合仍有线性敞口。
- **曲率因子噪声大。** 相比水平/斜率，曲率（PC3）跨样本稳定性差，因子方向可能漂移。用 z-score 触发时，均值和标准差要用滚动窗口估计，不能用全样本静态值。
- **流动性与特殊券效应。** 5y 有时因发行节奏或做市商库存出现特殊便宜/昂贵，蝶式读数反映的是「特殊券」而非纯曲率。需要识别 CTD（最便宜交割券）效应。

## 六、结论

蝶式交易是固定收益相对价值里把「因子暴露」玩到极致的例子：用 2-5-10 构造一个对水平和斜率都中性的头寸，纯粹押注中段凸性的过度定价回归。它的优雅在于——你不是在赌利率涨还是跌，而是在赌「曲线中段被错误定价后会回到正常形状」。理解这一点，你才算真正读懂了收益率曲线的曲率维度。
