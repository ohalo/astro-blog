---
title: "下跌 Beta 与危机 Alpha 的多空构建：让组合在崩盘时反而赚钱"
publishDate: '2026-07-20'
description: "标准 Beta 把上涨和下跌一视同仁，但真正区分好资产与坏资产的，是「下跌时跟不跟、涨时跟不跟」。本文用 Python 从零构造下行 Beta（仅用市场负收益片段回归），据此把股票分成「危机 Alpha」象限，构建「多低下行β / 空高下行β」多空组合，验证它在危机窗口的抗跌与超额，并诚实拆穿合成自证 / 下行阈值主观 / 做空成本 / 样本内过拟合 / 危机样本稀疏 / 流动性门槛六类真实陷阱(中阶)。"
tags:
  - 量化交易
  - 因子研究
  - 风险管理
  - 多空策略
  - 危机Alpha
  - Beta
  - Python
language: Chinese
difficulty: intermediate
---

传统 CAPM 告诉你：Beta 越高，系统性风险越大，长期该拿更高收益补偿。但 Beta 有个致命盲区——**它把上涨和下跌一视同仁地加权**。一只「涨时跟着涨、跌时躲得比谁都快」的股票，和一只「涨时趴着、跌时比市场还惨」的股票，标准 Beta 可能完全一样。可这两只股票，根本不是同一种资产。

本文要拆的，就是 Beta 的「上下不对称」：用**下行 Beta（Downside Beta）**只刻画资产在市场下跌时的敏感度，并据此构建「多低下行β / 空高下行β」的多空组合——理论上，多头在崩盘时少跌、空头在崩盘时多跌，组合应该**在危机里反而赚钱**，这叫「危机 Alpha」。

> 数据声明：全文收益序列为**自洽合成**（温和漂移 + 时变波动 + 一段约 80 日的合成危机窗口），目的是把下行 Beta 的「机制」讲清楚、跑通可复现代码。所有量级（如年化 / 终值）均为合成校准下的量级，真实市场里会被摩擦、做空成本、幸存者偏差大幅压缩——重点看*结构性保护*，不抄量级。

## 一、标准 Beta 与下行 Beta 的代数缺口

标准 Beta 是股票收益对市场收益的普通最小二乘斜率：

$$\beta = \frac{\text{Cov}(r_i, r_m)}{\text{Var}(r_m)}$$

下行 Beta（Estrada, 2002）只在市场收益低于某个阈值 $MAR$（通常用市场均值）时，拿这一段数据回归：

$$\beta^- = \frac{\text{Cov}(r_i,\, r_m \mid r_m < MAR)}{\text{Var}(r_m \mid r_m < MAR)}$$

对称资产（涨跌幅分布对称）两者相等；**不对称资产两者分裂**——下跌时跟得紧、上涨时跟得松，下行 Beta < 标准 Beta，这才是我们想要的「防御型」特征。

下面用 Python 构造 60 只合成股票，每只随机赋予上行 Beta、下行 Beta 与特质噪声，再分别估计：

```python
import numpy as np

rng = np.random.default_rng(20260720)
T, N = 2010, 60                       # 约 8 年交易日
market = rng.normal(0.0004, 0.010, T)
crisis = (np.arange(T) >= 1300) & (np.arange(T) <= 1380)
market[crisis] = rng.normal(-0.014, 0.028, crisis.sum())   # 注入一段危机

up_beta  = rng.uniform(0.6, 1.5, N)   # 上行敏感度
dn_beta  = rng.uniform(0.3, 1.6, N)   # 下行敏感度（真实值，仅用于造数据）
stock_ret = np.zeros((T, N))
for i in range(N):
    m_up = np.maximum(market, 0)
    m_dn = np.minimum(market, 0)
    stock_ret[:, i] = up_beta[i]*m_up + dn_beta[i]*m_dn + rng.normal(0, 0.011, T)

# 标准 Beta（全样本）
std_beta = np.array([np.cov(stock_ret[:, i], market)[0, 1] / np.var(market)
                     for i in range(N)])
# 下行 Beta：仅市场收益低于其均值时
mask = market < market.mean()
mkt_dn = market[mask] - market[mask].mean()
dn_beta_est = np.array([np.cov(stock_ret[mask, i], market[mask])[0, 1]
                        / np.var(market[mask]) for i in range(N)])
```

![标准 Beta 与下行 Beta 散点：绿色区即危机 Alpha 股票](/images/downside-beta-crisis-alpha/beta_scatter.png)

散点图里灰虚线是「标准 Beta = 下行 Beta」对称线。落在**右下方（下行 Beta 低、标准 Beta 高）的绿点**，就是「涨时跟涨、跌时脱钩」的危机 Alpha 候选——它们恰恰是被单一 Beta 数字掩盖的好资产。

## 二、把「危机 Alpha」做成一个可交易多空

直觉很直接：**多头买下行 Beta 最低的 20 只（跌得最少的防御股），空头卖空下行 Beta 最高的 20 只（跌得最惨的高风险股）**。危机来临时，空头腿暴跌贡献正收益，多头腿少跌，组合理论上*逆势上涨*。

```python
long_idx  = np.argsort(dn_beta_est)[:20]   # 低下行β → 多头
short_idx = np.argsort(dn_beta_est)[-20:]  # 高下行β → 空头
ls_ret = stock_ret[:, long_idx].mean(axis=1) - stock_ret[:, short_idx].mean(axis=1)
ls_cum = np.cumprod(1 + ls_ret)
mkt_cum = np.cumprod(1 + market)
```

![多空组合 vs 市场：危机期下行保护 + 长期超额](/images/downside-beta-crisis-alpha/ls_vs_market.png)

红线阴影标出危机窗口。合成校准下，多头书（最低下行β 20 只）在危机窗口仅回撤 **−21.8%**，而同期市场暴跌 **−74.3%**；多空组合（多空双腿叠加）危机窗口反而 **+288.3%**。这就是「危机 Alpha」的字面含义——别人崩盘，你赚钱。

## 三、单调性检验：因子到底干不干活

光看一条净值曲线不够，得确认下行 Beta 和收益是不是**单调**关系。把 60 只按下行 Beta 五分位分层：

```python
order = np.argsort(dn_beta_est)
quintile_ann = []
for q in range(5):
    idx = order[q*12:(q+1)*12]
    r = stock_ret[:, idx].mean(axis=1)
    quintile_ann.append((np.prod(1+r)**(252/T) - 1) * 100)
# 合成校准结果（年化 %）：Q1 最低 68.2 | Q2 | Q3 | Q4 | Q5 最高 -31.1
```

![下行 Beta 五分位：越低越抗跌，越高越拖累](/images/downside-beta-crisis-alpha/quintile_return.png)

最低分位（防御股）年化显著为正、最高分位（高风险股）为负，中间递减——单调性成立，说明下行 Beta 不是噪声，而是带了*横截面定价信息*的因子。

## 四、回撤：多空比市场浅得多

```python
def max_drawdown(cum):
    peak = np.maximum.accumulate(cum)
    return cum/peak - 1
dd_ls, dd_mkt = max_drawdown(ls_cum), max_drawdown(mkt_cum)
# 合成校准：多空最大回撤 -4.5%  vs 市场 -79.5%
```

![回撤对比：多空组合回撤更浅且更快修复](/images/downside-beta-crisis-alpha/drawdown.png)

多空组合最大回撤仅 **−4.5%**，市场同段 **−79.5%**。这不是因为策略多神，而是因为**空头腿在崩盘时替你承担了「市场该跌的部分」**——回撤被结构性对冲掉了。

## 五、六类真实陷阱（必须诚实拆穿）

1. **合成自证**：本文收益是造出来的，下行 Beta 与收益的关系是我*写进数据生成器*的。真实数据里二者相关但远没这么干净，绝不能直接把 418x 终值当真。
2. **下行阈值主观**：用市场均值当 MAR 是惯例，但换成 0、用分位数、用滚动阈值，分出来的「下行 Beta」会变，因子收益也随之漂移——这是典型的阈值敏感。
3. **做空成本致命**：高下行 Beta 的股票，往往也是小盘、高换手、融券费率最贵的。空头腿的借券成本 + 卖空约束（涨时逼空），会把「危机 Alpha」在平静市里慢慢磨掉。
4. **样本内过拟合**：我只在*一段*合成危机里看到逆势赚钱；真实市场危机稀少（8 年可能就 1–2 次），factor 在危机外的表现才是常态，不能用危机样本代表全样本。
5. **危机样本稀疏**：用 80 天危机窗口去估计「危机 Alpha」，统计功效极低，一个异常日就能改写结论。Walk-forward / Bootstrap 才勉强可信。
6. **流动性门槛**：真正崩盘时，你最想做空的那些高 Beta 小票，往往先跌停、先停牌、先没流动性——嘴上能做空，真到危机时根本建不了仓。

## 六、落地建议

- 下行 Beta 更适合当**风控筛子**而非纯 alpha：用它把组合里「看着 Beta 低、其实跌起来很凶」的伪防御股揪出来。
- 若真做多空，空头腿务必考虑**融券可得性 + 费率**，否则平静市躺亏、危机时还建不上仓。
- 把下行 Beta 与波动率目标结合：危机来时自动降总敞口，比单纯赌「空头腿赚钱」更稳。

下行 Beta 的价值，不在于给你一条 418x 的曲线，而在于它逼你承认一件事：**资产的风险是分方向的**。把「涨时弹性」和「跌时粘性」拆开看，你才真正开始理解自己持仓里，哪些是在帮你，哪些只是在陪你跌。
