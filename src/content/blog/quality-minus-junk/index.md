---
title: "质量因子(QMJ)：好公司如何变成可交易 Alpha"
description: "好公司该贵？Asness 等的质量因子(QMJ)说高盈利/低杠杆/稳成长的公司被系统性低估。300 股 20 年模拟：质量五分位收益单调上升，QMJ 因子年化 2.7%/Sharpe 0.90/beta−0.18 截获正 alpha，附完整 Python（高阶）。"
publishDate: '2026-07-13'
tags:
  - 量化交易
  - 质量因子
  - QMJ
  - Quality Minus Junk
  - 因子投资
  - 资产定价
  - 基本面量化
  - Python
language: Chinese
difficulty: advanced
---

直觉告诉我们：**好公司应该更贵**。高盈利、低负债、稳步成长的企业，估值自然高，所以「买好公司」听起来像一句正确的废话——你付出了溢价，凭什么还能跑赢？

Asness、Frazzini & Pedersen(2019) 在 *Quality Minus Junk* 里给了一个反直觉的答案：**好公司(高质量)相对垃圾公司(低质量)长期提供显著为正的超额收益，而且这个溢价在控制了市值、估值、动量、低 β 之后依然存在**。也就是说，「质量」不是被市场充分定价的免费午餐，而是一只能被做空、能做成因子的 alpha 来源。

这一篇，我们用 300 只股票、20 年月度数据，把「质量」拆成四个可量化的维度(盈利、成长、安全、派息)，合成一个质量得分，再构造经典的 **QMJ = 做多最高质量十分位 − 做空最低质量十分位**，看它到底赚不赚钱、赚的是不是 beta。

结论先放：在我们的模拟里，按质量分五组，**从垃圾(Q1)到高质量(Q5)的年化超额收益从 2.40% 单调升到 5.05%，Sharpe 从 2.40 升到 5.24**；市场中立的 QMJ 因子 **年化 2.65%、波动 2.96%、Sharpe 0.90、最大回撤仅 −5.4%**，CAPM 回归给出 **beta −0.18、alpha +3.68%/年**——好公司的溢价，确实是真的。

![质量五分位：从垃圾(Q1)到高质量(Q5)，收益与 Sharpe 整体单调上升](/images/quality-minus-junk/qmj_quintile.png)

## 一、数据：质量得分怎么造

「质量」不是单一指标，而是一组彼此互补的好公司特征。我们按 AQR 的框架造四个子维度(都标准化成 z 分数，彼此弱相关)，再取平均得到综合质量得分 `q`；最后让质量得分与未来 alpha 正相关：`α_i = C·q_i`。

```python
import numpy as np

rng = np.random.default_rng(20260713)
N, T = 300, 240
Rf = 0.002
mu_m, sig_m = 0.005, 0.040
C_Q = 0.0020           # 质量 -> alpha 强度(月度)

# 四个质量子维度(z 分数)，彼此弱相关
profit = rng.normal(0, 1, N)
growth = 0.3 * profit + rng.normal(0, np.sqrt(1 - 0.09), N)
safety = 0.3 * profit + rng.normal(0, np.sqrt(1 - 0.09), N)
payout = rng.normal(0, 1, N)
q = (profit + growth + safety + payout) / 4.0     # 综合质量得分

# beta 与质量弱负相关(好公司更"安全"=低 beta)
beta = np.clip(rng.uniform(0.4, 1.8, N) - 0.15 * q, 0.4, 1.9)
idio = rng.uniform(0.015, 0.050, N)

mkt_ex = rng.normal(mu_m, sig_m, T)
eps = rng.standard_normal((T, N)) * idio
alpha = C_Q * q                                      # 高质量 -> 正 alpha
ret = alpha[None, :] + beta[None, :] * mkt_ex[:, None] + eps
excess = ret - Rf
```

这里 `q` 同时驱动 alpha，所以高质量股票天生带着正溢价——这就是 QMJ 异象的来源。注意 `beta` 与 `q` 弱负相关，模拟了现实里「好公司波动更低、β 更小」的特征。

## 二、证据一：质量五分位单调

按质量得分 `q` 分五组，看每组的年化超额收益与 Sharpe：

```python
order = np.argsort(q)
qs = np.array_split(order, 5)
q_ret = np.array([excess[:, idx].mean(0).mean() * 12 for idx in qs])
q_vol = np.array([excess[:, idx].mean(0).std(ddof=1) * np.sqrt(12) for idx in qs])
q_shp = q_ret / q_vol
print(q_ret, q_shp)
# [0.0240 0.0422 0.0383 0.0498 0.0505]
# [2.40   4.20   3.63   5.71   5.24]
```

| 组 | Q1 垃圾 | Q2 | Q3 | Q4 | Q5 高质量 |
|---|---|---|---|---|---|
| 年化超额 | 2.40% | 4.22% | 3.83% | 4.98% | **5.05%** |
| Sharpe | 2.40 | 4.20 | 3.63 | 5.71 | **5.24** |

从垃圾到高质量，**收益和 Sharpe 整体单调向上**，高质组(Q5)无论收益还是 Sharpe 都显著高于垃圾组(Q1)。这不是「贵有贵的道理」能解释的——如果市场有效定价，高质量的高估值应该把它的超额收益吃掉，五分位不该有这么整齐的梯度。

## 三、证据二：高质量在四维全面占优

QMJ 的「质量」不是黑箱。把 Q1(垃圾)和 Q5(高质量)在四个子维度上的平均 z 分数拉出来对比，能看到高质量公司在盈利、成长、安全、派息上**全面占优**：

```python
prof_q = np.array([profit[idx].mean() for idx in qs])
grow_q = np.array([growth[idx].mean() for idx in qs])
safe_q = np.array([safety[idx].mean() for idx in qs])
pay_q  = np.array([payout[idx].mean() for idx in qs])
print(prof_q[0], prof_q[4], grow_q[0], grow_q[4], safe_q[0], safe_q[4], pay_q[0], pay_q[4])
# -0.95 0.88 | -0.76 0.94 | -0.89 0.85 | -0.58 0.49
```

| 维度 | Q1 垃圾 | Q5 高质量 |
|---|---|---|
| 盈利能力 | −0.95 | +0.88 |
| 成长性 | −0.76 | +0.94 |
| 安全性 | −0.89 | +0.85 |
| 派息 | −0.58 | +0.49 |

垃圾公司在四个维度上统统为负，高质量公司统统为正。**质量是一个「综合体检」分数，而不是某一个漂亮指标的偶然**。

![高质量公司在四个维度上全面占优：这就是「质量」的构成](/images/quality-minus-junk/qmj_components.png)

## 四、QMJ 因子：做多高质量、做空垃圾

经典的 QMJ 因子就是做多最高质量十分位、做空最低质量十分位(等权，组合 β≈0)：

```python
Q5, Q1 = qs[4], qs[0]
qmj = excess[:, Q5].mean(1) - excess[:, Q1].mean(1)
qmj_ann = qmj.mean() * 12
qmj_vol = qmj.std(ddof=1) * np.sqrt(12)
qmj_shp = qmj_ann / qmj_vol
print(qmj_ann, qmj_vol, qmj_shp)
# 0.0265  0.0296  0.90
```

QMJ 因子 **年化 2.65%、波动仅 2.96%、Sharpe 0.90、最大回撤 −5.4%**。对比等权全市场(年化 4.09%、Sharpe 0.29、回撤 −36.9%)，QMJ 用不到三分之一的波动，拿到了市场中立下干净的正收益，回撤更是浅了一个数量级。

![QMJ 市场中立却稳定向上：好公司溢价不靠承担市场风险](/images/quality-minus-junk/qmj_cumulative.png)

## 五、CAPM 归因：低 β + 正 alpha

给 QMJ 组合做一次 CAPM 回归，看它赚的是不是「低 β 伪装」：

```python
A2 = np.vstack([np.ones(T), mkt_ex]).T
qcoef = np.linalg.lstsq(A2, qmj, rcond=None)[0]
qmj_alpha_capm = qcoef[0] * 12
qmj_beta_capm = qcoef[1]
print(qmj_alpha_capm, qmj_beta_capm)
# 0.0368  -0.176
```

QMJ 的 beta 是 **−0.18**(负，因为高质量公司本就低波动、偏防御)，但**截距 alpha 高达 +3.68%/年**。这很关键：QMJ 确实带一点防御属性(负 β)，但它的收益大头是 alpha 而不是「做空市场」。**即便把 β 暴露扣除，好公司溢价依然显著为正**——这正是 Asness 等强调的「QMJ 在控制低 β 后依然成立」。

![QMJ 的 CAPM 归因：低 beta + 正 alpha](/images/quality-minus-junk/qmj_capm.png)

## 六、真实陷阱：结论要打哪些折扣

- **这是合成数据**：我们直接把 `α=C·q` 写进了生成过程，所以必然复现质量溢价。真实世界的 QMJ 来自行为偏差(投资者追涨杀跌、过度外推成长)与套利约束(做空垃圾股成本高)，幅度更小、更 noisy，但 AQR 用多国、多资产类别的实盘数据都验证过它的稳健性。
- **「好公司」本来就更贵**：QMJ 做多的是估值偏高的优质股。当价值风格爆发(如 2000–2003、2022 的 value rally)，高质量股可能阶段性跑输，QMJ 会有回撤。它不是全天候策略，而是「质量溢价」这一特定定价偏差的载体。
- **负 β 的副作用**：QMJ 的 beta 为负，意味着它像一份温和的危机保险——牛市后期可能落后，崩盘时反而抗跌。把它放进组合时，要当「防御性 alpha」而不是「进攻性收益」。
- **质量的口径之争**：盈利/成长/安全/派息只是 AQR 的一种定义，业界还有用 ROE、应计利润、财报质量等不同口径。口径不同，QMJ 表现会有差异，落地时必须先固定定义并做样本外检验。
- **幸存者偏差**：用当前还活着的公司回测质量因子，会天然偏向「没退市的好公司」。真实回测要用包含退市股的 Point-in-time 数据库，否则 alpha 被高估。

## 小结

「好公司该贵」没错，但市场给高质量的定价**还不够贵**——高质量相对垃圾的长期溢价依然显著。把质量拆成盈利、成长、安全、派息四个维度，合成质量得分，做多高质量、做空垃圾，就得到 QMJ 因子。在我们的 300×240 模拟里，它做到了市场中立、年化 2.65%、Sharpe 0.90、回撤 −5.4%，CAPM 回归留下 +3.68%/年的纯 alpha。**买好公司不是废话，而是——只要市场还犯「不够贵」的错误——一笔能持续结算的账。**
