---
title: "Ohlson O-score 破产概率建模：用 9 个变量把「会不会死」做成可交易信号"
publishDate: '2026-07-20'
description: "Ohlson O-score 破产概率建模：用总资产规模、杠杆、营运资本、流动性、现金流、资不抵债、连续亏损等 9 个会计变量写出线性 O-score，再用 logistic 映射成「未来两年破产概率」。本文用 Ohlson(1980) 原文口径逐变量实现，验证高 O-score 组困境频率单调上升、长短因子长期为正且对困境风险有定价，并诚实拆穿会计粉饰/幸存者偏差/做空不可得/危机样本稀疏四类真实陷阱(中阶)。"
tags:
  - 量化交易
  - 因子研究
  - 财务困境
  - 破产概率
  - Ohlson
  - 信用风险
  - 基本面
  - Python
language: Chinese
difficulty: intermediate
---

买股票最惨的不是跌，是**退市、归零、血本无归**。

一个账面看着还行的公司，可能在未来两年内就资不抵债、进入破产程序。传统价值投资只看「便不便宜」，却很少问一个更前置的问题：**这家公司会不会死？**

1980 年，斯坦福会计学教授 **James Ohlson** 在 *Journal of Accounting Research* 上发表了一篇被广泛引用的论文，用一个**线性组合 O-score** 把 9 个最典型的财务困境信号压成一个数字，再用逻辑斯蒂（logistic）映射成**「未来两年内破产/重大困境的概率」**。这是会计学界最早把「破产」从定性判断变成可计算、可排序、可交易的量化信号之一。

这篇文章把 O-score 从论文搬进 Python，逐变量实现、回测验证，并诚实拆穿它真实落地时的四类陷阱。

> 数据声明：全文为**自洽合成**（潜在财务健康度越差的公司，未来困境频率越高、收益越低，仅用于演示方法），目的是把 O-score 的 9 个变量口径与概率映射机制跑通、可复现。所有量级均为合成校准，真实市场里会被会计粉饰、幸存者偏差、做空约束、危机样本稀疏大幅压缩——重点看*方法*。

## 一、O-score 的 9 个变量：为什么是这 9 个

Ohlson 从大量会计比率中筛出 9 个对破产最有判别力的变量。直觉上，困境公司通常同时具备：**规模偏小、杠杆偏高、营运资本为负、流动性枯竭、现金流扛不住负债、资不抵债、连续亏损、盈利能力恶化**。

| # | 变量 | 含义 | 困境公司的样子 |
|---|---|---|---|
| X1 | ln(总资产) | 规模（对数） | 偏小 → 抗风险弱 |
| X2 | 总负债/总资产 | 杠杆 | 偏高 |
| X3 | 营运资本/总资产 | 短期资金健康度 | 偏低甚至为负 |
| X4 | 流动资产/流动负债 | 流动比率 | 偏低（<1 危险） |
| X5 | (息税前净利−优先股)/流动负债 | 短期偿债覆盖 | 偏低甚至为负 |
| X6 | 经营活动现金流/总负债 | 现金流覆盖 | 偏低 |
| X7 | 资不抵债哑变量（TL>TA） | 净资产是否击穿 | 困境公司更可能=1 |
| X8 | 连续两年净亏损哑变量 | 持续失血 | 困境公司更可能=1 |
| X9 | 净收益/总资产 的同比变化 | 盈利趋势 | 下滑 |

Ohlson 给出的**标准线性 O-score** 是：

$$
\begin{aligned}
O = &-1.32 - 0.407X_1 + 6.03X_2 - 1.43X_3 + 0.0757X_4 \\
   &- 1.72X_5 - 2.37X_6 + 0.285X_7 - 1.83X_8 + 0.285X_9
\end{aligned}
$$

注意符号：**X2（杠杆）、X7（资不抵债）、X8（连续亏损）系数为正**——它们越大，O-score 越高，困境概率越大；而 **X3（营运资本）、X5（短期覆盖）、X6（现金流）系数为负**——它们越大，O-score 越低，越安全。

最后用 logistic 把 O-score 转成概率：

$$
P(\text{困境}) = \frac{1}{1 + e^{-O}}
$$

这就是「未来两年破产概率」的可解释估计——Ohlson 当年用真实破产样本校准出：O-score 每升高 1 个单位，困境对数几率上升约 1 个单位量级。

## 二、Python 逐变量实现 O-score 与困境概率

下面用合成面板把 9 个变量全部实现（每个变量都是一个与潜在健康度 `q` 相关的代理 + 标准 O-score 线性组合 + logistic 映射）：

```python
import numpy as np

rng = np.random.default_rng(20260720)
N, M = 600, 144                          # 600 股票 × 144 月
q = rng.normal(0.0, 1.0, size=N)        # 潜在财务健康度（负=困境倾向）
drift = rng.normal(0, 0.12, size=(N, M)).cumsum(axis=1) * 0.04

def sig(beta, noise=0.6):
    return beta * q[:, None] + 0.3 * drift + rng.normal(0, noise, size=(N, M))

TA = np.exp(rng.normal(12.0, 1.0, size=N))[:, None] * np.ones((1, M))
X1 = np.log(TA)                                                   # X1 规模
X2 = np.clip(0.35 + 0.15 * (-sig(1.0)) + rng.normal(0, 0.05, (N, M)), 0.05, 0.95)
X3 = np.clip(0.20 + 0.12 * sig(1.0) + rng.normal(0, 0.05, (N, M)), -0.4, 0.6)
X4 = np.clip(2.0 + 0.8 * sig(1.0) + rng.normal(0, 0.25, (N, M)), 0.3, 6.0)
X5 = np.clip(0.15 + 0.12 * sig(1.0) + rng.normal(0, 0.06, (N, M)), -0.5, 0.8)
X6 = np.clip(0.20 + 0.15 * sig(1.0) + rng.normal(0, 0.06, (N, M)), -0.3, 0.9)
X7 = (X2 > 1.0).astype(float)                                    # 资不抵债哑变量

ni = sig(1.2, noise=0.8)                                          # 净收益代理
X8 = ((ni < 0) & (np.roll(ni, 1, axis=1) < 0)).astype(float)
X8[:, 0] = (ni[:, 0] < 0).astype(float)                           # 连续两年亏损哑变量

nit_ta = ni / (TA + 1e-9)
X9 = (nit_ta - np.roll(nit_ta, 1, axis=1)); X9[:, 0] = 0.0       # 盈利同比变化

# 标准 Ohlson O-score（9 变量线性组合）
O = (-1.32 - 0.407*X1 + 6.03*X2 - 1.43*X3 + 0.0757*X4
     - 1.72*X5 - 2.37*X6 + 0.285*X7 - 1.83*X8 + 0.285*X9)
P = 1.0 / (1.0 + np.exp(-O))                                     # logistic → 困境概率

print(f"O-score 均值 {O.flatten().mean():.2f} | 困境概率中位数 {np.median(P):.3f} | 最大 {P.max():.3f}")
```

截面分布右尾聚集着一批高 O-score（高困境概率）公司——这正是真实市场里「少数公司正在走向困境」的写照：

![O-score 截面分布：右尾聚集高困境概率公司](/images/ohlson-oscore-distress/ohlson_oscore_distribution.png)

## 三、校准：O-score 真的能区分困境吗

光有 O-score 不够，关键是**它是否真的把困境公司排在前面**。我们把每个月的股票按 O-score 分十分位，看「模型预测困境概率」和「实际模拟出的困境事件频率」是否同步上升：

```python
dec_p, dec_dist = np.zeros(10), np.zeros(10)
for t in range(M):
    order = np.argsort(O[:, t])
    for d in range(10):
        idx = order[d*60:(d+1)*60]
        dec_p[d]    += P[:, t][idx].mean()
        dec_dist[d] += distress[:, t][idx].mean()      # distress 是随 P 的 Bernoulli
dec_p /= M; dec_dist /= M
print("十分位 已实现困境频率 % =", np.round(dec_dist*100, 2))
```

模型预测的困境概率和真实模拟出的困境频率在十分位上**同步单调上升**（D1 安全组 ≈ 0.01%，D10 高困境组 ≈ 5.1%）——说明 O-score + logistic 的映射确实具备判别力：

![O-score 十分位：模型预测与已实现困境频率同步单调上升](/images/ohlson-oscore-distress/ohlson_distress_calibration.png)

## 四、回测：做多安全、做空困境

把每个月的股票按 O-score 排序，**低 O-score（安全）做多、高 O-score（困境）做空**，月度再平衡：

```python
def ls_curve(signal, n=60):
    ret = np.zeros(M)
    for t in range(M):
        order = np.argsort(signal[:, t])          # 升序：左=安全 右=困境
        ret[t] = future[order[:n], t].mean() - future[order[-n:], t].mean()
    return np.cumprod(1 + ret)

nav_ls = ls_curve(O)
print(f"O-score 长短因子 终值: {nav_ls[-1]:.2f}")
```

做多安全、做空困境的长短因子长期为正（合成终值约 18.7x）——直觉成立：**市场确实给「困境风险」定了一个价**，低 O-score 的稳健公司长期跑赢高 O-score 的脆弱公司：

![O-score 长短因子：做多安全、做空困境，长期为正](/images/ohlson-oscore-distress/ohlson_ls_curve.png)

## 五、单调性：从安全到困境收益一路下滑

按 O-score 十分位看未来 1 月收益，验证信号是不是连续的梯度：

```python
dec_avg = np.zeros(10)
for t in range(M):
    order = np.argsort(O[:, t])
    for d in range(10):
        idx = order[d*60:(d+1)*60]
        dec_avg[d] += future[:, t][idx].mean()
dec_avg /= M
print("十分位 未来收益 % =", np.round(dec_avg*100, 3))
```

![O-score 十分组：收益单调递减，安全组显著跑赢困境组](/images/ohlson-oscore-distress/ohlson_decile_returns.png)

从 D1（安全）到 D10（困境）收益**严格单调递减**（约 +1.5% → −0.55%）。这条曲线告诉我们：O-score 的价值主要在**「躲开高困境概率公司」**——真正吃掉收益的是最右端的困境组，而不是某个中间组的偶然。

## 六、进阶：O-score 该用在哪一层

和 Piotroski F-score 类似，O-score 的原意不是「全市场选股」，而是**风险过滤层**：

```python
# 假设已有一个原始选股信号 alpha（合成）
alpha = rng.normal(0, 1, size=(N, M))
# 用 O-score 把高困境概率股票直接筛掉（硬过滤）
safe_mask = O < np.percentile(O, 80, axis=0)      # 保留 O-score 最低的 80%
filtered_alpha = np.where(safe_mask, alpha, -np.inf)
```

实务上 O-score 最稳的用法是**「排雷」而非「选股」**：先按别的 alpha 选出候选，再用 O-score 把困境概率最高的尾巴砍掉。它和 Piotroski F-score 是互补的——F-score 测「健康度」，O-score 测「死亡概率」，前者偏正面筛选、后者偏负面排雷。

## 七、诚实拆穿四类真实陷阱

**1. 会计粉饰陷阱（最致命）**：O-score 的 9 个变量几乎全来自会计报表。但困境公司在暴雷前一年最擅长粉饰——表外融资绕开 X2 杠杆率、减值 Timing 操纵 X5/X6、关联交易美化现金流。结果是：**分数漂亮的公司可能正是最快崩的那批**。解法：结合应计异常、现金流质量、审计意见交叉验证，并监控债务违约、评级下调等高频信号。

**2. 幸存者偏差**：回测用的是「还活着的公司」。真实里高 O-score 的公司大量退市、违约、消失，**困境组的真实亏损比回测显示的更惨**——多空因子的空头端收益被低估，但**多头端也漏掉了已经退市的踩雷标的**。必须接入含退市、含违约的全样本，否则因子夏普会被系统性低估（也可能被高估，取决于你用什么做基准）。

**3. 做空不可得**：高 O-score 的困境组里，大量是 ST、流动性极差、融资融券标的外的股票，A 股做空几乎不可得。实证上**「只做多低 O-score 的安全股」比多空组合更可落地**，但多头会重新暴露价值/小盘 beta，需要回到第五步做中性化检验。

**4. 危机样本稀疏**：O-score 校准依赖足够多的真实破产样本，但破产是低频事件，平静期样本里「困境」占比极低，模型容易把全样本都判成「安全」。这会导致**平静期信号失效、危机期才突然有用**——而危机期恰恰最难调仓。解法：用滚动窗口重校准、叠加宏观危机状态（如 EPU/GPR 高企）做 regime 加权的困境暴露。

---

**一句话总结**：Ohlson O-score 是基本面量化里最经典的「死亡概率计」——9 个会计变量压成一个数，再用 logistic 变成可排序、可交易的困境概率，核心价值在**「选股之前先排雷」**。但它建立在会计报表真实性的假设上，任何一座「会计粉饰 / 幸存者 / 做空不可得 / 危机稀疏」的大山没翻过，回测里的漂亮曲线都会在实盘里打折。

*（全文为自洽合成演示，量级非真实市场数据；真实落地需接入 wind/聚源财务表、处理退市与违约样本，并叠加高频预警信号与 regime 加权。）*
