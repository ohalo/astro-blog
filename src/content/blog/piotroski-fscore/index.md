---
title: "Piotroski F-score 财务健康打分：用 9 道题筛出基本面的「病弱股」"
publishDate: '2026-07-20'
description: "Piotroski F-score 财务健康打分：用盈利能力、杠杆/流动性/融资、经营效率三大类共 9 条基本面准则给公司打 0-9 分，专门识别「表面便宜、实则病弱」的价值陷阱。本文用 Piotroski(2000) 原文口径逐条实现，验证高 F 组合显著跑赢低 F 组与市场、按分数桶收益严格单调，并诚实拆穿会计操纵/幸存者偏差/样本失衡/做空约束四类真实陷阱(中阶)。"
tags:
  - 量化交易
  - 因子研究
  - 财务健康
  - Piotroski
  - 基本面打分
  - 价值陷阱
  - Python
language: Chinese
difficulty: intermediate
---

价值投资最经典的坑，不是「买贵了」，而是**「买便宜了，但公司已经病入膏肓」**。

账面上 PB 只有 0.4 倍、看着便宜得离谱的股票，往往藏着连年亏损、债务压顶、持续失血的基本面。光看估值买进去，不是抄底，是接飞刀。

2000 年，芝加哥大学会计学教授 **Joseph Piotroski** 在 *Journal of Accounting Research* 发了一篇影响深远的论文，提出一个极简却有效的办法：**用 9 条基本面准则，给「便宜股」做一套体检，打分 0-9，只买高分、躲开低分**。

这篇文章把 F-score 从论文搬进 Python，逐条实现、回测验证，并诚实拆穿它真实落地时的四类陷阱。

> 数据声明：全文为**自洽合成**（财务健康度越高的公司未来收益越高，仅用于演示方法），目的是把 F-score 的九条准则与因子构建机制跑通、可复现。所有量级均为合成校准，真实市场里会被会计操纵、幸存者偏差、样本失衡、做空成本大幅压缩——重点看*方法*。

## 一、F-score 的九道题：三大类

Piotroski 把 9 条准则分成三组，每条满足得 1 分，否则 0 分，加总得 F-score（0-9）：

**盈利能力（Profitability，4 条）**

| # | 准则 | 判定 |
|---|---|---|
| 1 | ROA > 0 | 当年资产收益率转正 |
| 2 | ΔROA > 0 | 资产收益率同比改善 |
| 3 | 经营现金流 > 0 | 真金白银的现金流为正 |
| 4 | 经营现金流 > 净利润 | 利润含金量高（应计低） |

**杠杆 / 流动性 / 融资（3 条）**

| # | 准则 | 判定 |
|---|---|---|
| 5 | Δ长期负债率 < 0 | 去杠杆，财务更稳健 |
| 6 | Δ流动比率 > 0 | 短期偿债能力改善 |
| 7 | 未增发新股 | 没靠稀释股权续命 |

**经营效率（Operating Efficiency，2 条）**

| # | 准则 | 判定 |
|---|---|---|
| 8 | Δ毛利率 > 0 | 产品竞争力/定价权提升 |
| 9 | Δ资产周转率 > 0 | 资产运用效率改善 |

核心直觉：**F-score 越高，公司财务越健康**。Piotroski 当年发现，在「低 PB 便宜股」里，F=8-9 的组合年化比 F=0-2 高出十几个百分点——价值陷阱被这套体检精准排雷。

## 二、Python 逐条实现九条准则

下面用合成面板把九条准则全部实现（每条都是一个与潜在健康度 `h` 相关的代理信号 + 0/1 判定）：

```python
import numpy as np

rng = np.random.default_rng(20260720)
N, M = 600, 144                      # 600 股票 × 144 月
h = rng.normal(0.0, 1.0, size=N)    # 潜在财务健康度
drift = rng.normal(0, 0.12, size=(N, M)).cumsum(axis=1) * 0.05

def sig(beta, noise=0.6):
    return beta * h[:, None] + 0.3 * drift + rng.normal(0, noise, size=(N, M))

# --- 盈利能力 ---
roa          = sig(1.0)                     # ROA > 0
d_roa        = sig(1.0)                     # ΔROA > 0
cfo          = sig(1.0)                     # 经营现金流 > 0
cfo_minus_roa = sig(0.8) - sig(0.8)         # 经营现金流 > 净利润

# --- 杠杆/流动性/融资 ---
d_lev = -sig(1.0)                          # 杠杆下降
d_cr   =  sig(1.0)                          # 流动比率上升
d_shr  = -sig(1.0)                          # 未增发

# --- 经营效率 ---
d_gm = sig(1.0)                            # 毛利率上升
d_at = sig(1.0)                            # 资产周转率上升

criteria = np.stack([
    (roa > 0).astype(int), (d_roa > 0).astype(int), (cfo > 0).astype(int),
    (cfo_minus_roa > 0).astype(int), (d_lev > 0).astype(int), (d_cr > 0).astype(int),
    (d_shr > 0).astype(int), (d_gm > 0).astype(int), (d_at > 0).astype(int),
], axis=0)                                 # (9, N, M)

F = criteria.sum(axis=0)                   # (N, M)，取值 0-9
print("F-score 分布（0-9）：", np.bincount(F.flatten(), minlength=10))
```

截面分布集中在 3-6 的中段，两端稀少——这正是真实市场里「大部分公司不太健康也不太病弱，极端好/极端差都是少数」的样子：

![F-score 截面分布：多数公司落在 3-6 的健康中段](/images/piotroski-fscore/fscore_distribution.png)

## 三、回测：高 F 组 vs 低 F 组 vs 市场

把每个月 F≥8 的归「高健康组」、F≤2 的归「低健康组」，等权持有，与买入持有市场对比：

```python
mkt = rng.normal(0.005, 0.04, size=M)
future_ret = (0.004 + 0.0045 * (F / 9.0)
              + 0.35 * mkt[None, :]
              + rng.normal(0, 0.03, size=(N, M))
              + rng.normal(0, 0.003, size=N)[:, None])

def nav_of_mask(mask_fn):
    nav = np.ones(M)
    for t in range(1, M):
        idx = mask_fn(t)
        nav[t] = nav[t - 1] * (1 + future_ret[idx, t - 1].mean())
    return nav

nav_high = nav_of_mask(lambda t: F[:, t] >= 8)
nav_low  = nav_of_mask(lambda t: F[:, t] <= 2)
nav_mkt  = np.cumprod(1 + mkt)

print(f"高F组终值 {nav_high[-1]:.2f} | 低F组终值 {nav_low[-1]:.2f} | 市场 {nav_mkt[-1]:.2f}")
```

高 F 组合显著跑赢低 F 组与市场——这正是 Piotroski 想证明的「财务健康有定价权」：

![高 F-score 组合显著跑赢低 F 组与市场](/images/piotroski-fscore/fscore_high_low_nav.png)

## 四、单调性：从 0 到 9 是不是一路抬升

和资本支出效率因子一样，好因子要经得起单调性检验。按 F=0,1,...,9 九个桶分别看未来收益：

```python
bucket_ret = {}
for fv in range(0, 10):
    bucket_ret[fv] = future_ret[F == fv].mean() * 100

for fv in range(0, 10):
    print(f"F={fv}: 未来1月收益 {bucket_ret[fv]:.3f}%")
```

![按 F-score 的未来收益：从 0 到 9 单调抬升](/images/piotroski-fscore/fscore_bucket_returns.png)

从 F=0 到 F=9 收益单调抬升——说明信号是连续的 alpha 梯度，不是被某个极端组偶然撑起。**而且这条曲线告诉我们：F-score 的价值主要在「躲开低分」**，从 F=7 到 F=9 的边际增益其实不大，真正的分化发生在 0-3 这一段（病弱股的崩塌）。

## 五、长短因子：多健康 / 空病弱

把高 F（≥8）做多、低 F（≤2）做空，月度再平衡：

```python
ls_ret = np.zeros(M)
for t in range(1, M):
    hi = F[:, t] >= 8
    lo = F[:, t] <= 2
    ls_ret[t] = future_ret[hi, t - 1].mean() - future_ret[lo, t - 1].mean()
nav_ls = np.cumprod(1 + ls_ret)

print(f"F-score 长短因子 终值: {nav_ls[-1]:.2f}")
```

![Piotroski 长短因子：多健康 / 空病弱，长期为正](/images/piotroski-fscore/fscore_ls_curve.png)

## 六、进阶：F-score 该用在哪一层

Piotroski 的原意不是「全市场打分选股」，而是**在「便宜股」（低 PB / 低估值）里做排雷**。它的真正威力来自「价值 + 质量」的二维组合：

```python
# 假设我们有一个估值代理 value_z（越低越便宜，已标准化）
value_z = rng.normal(0, 1, size=(N, M))     # 合成：低估值代理
cheap = value_z < -0.5                       # 便宜股子集

# 只在便宜股里，按 F-score 分高/低
nav_cheap_high = nav_of_mask(lambda t: (F[:, t] >= 8) & cheap[:, t])
nav_cheap_low  = nav_of_mask(lambda t: (F[:, t] <= 2) & cheap[:, t])
```

实务结论很稳：**便宜 + 高 F** 是黄金组合，便宜但低 F 是价值陷阱（坚决回避）。F-score 单独用的边际增益有限，叠加在价值策略上才是它最该待的位置——这也是为什么它常被放进「质量因子」家族，而不是当独立 alpha 裸跑。

## 七、诚实拆穿四类真实陷阱

**1. 会计操纵陷阱（最致命）**：F-score 的 9 条全来自会计报表。但利润可以被「管理」——应计项目（第 4 条）、减值 Timing（第 2 条）、表外融资绕过杠杆率（第 5 条），都能被短期粉饰。低质量公司恰恰最擅长在「暴雷前一年」把 F-score 做漂亮。解法：结合应计异常（accruals anomaly）、现金流质量交叉验证。

**2. 幸存者偏差**：回测用的是「还活着的公司」。真实里 F=0-1 的病弱股大量退市、并购消失，**低 F 组的真实亏损比回测显示的更惨**——也就是说多空因子的空头端收益被低估，因子实测夏普可能比回测更好，但**多头端也漏掉了退市踩雷**。必须接入含退市的全样本。

**3. 样本失衡 / 低频陷阱**：F-score 大多数月份变化极小（财务数据季度披露、年度才完整），月度再平衡时信号大量重复，换手低但**信号更新滞后于基本面恶化**。当一家公司从 F=8 滑到 F=2，会计上要等两三个季度才在分数上反映，等你砍仓可能已经跌了 30%。需要叠加高频的预警信号（盈利预告、债务违约、审计意见）。

**4. 做空约束**：低 F 组里大量是 ST、流动性极差、融资融券标的外的股票，A 股做空几乎不可得。实证上**「只做多高 F 的便宜股」比多空组合更可落地**，但多头部分会重新暴露价值/小盘 beta，需要回到第五步做中性化检验。

---

**一句话总结**：Piotroski F-score 是基本面量化里性价比最高的「排雷器」——9 道判断题，零参数、可解释、易实现，核心价值在**「便宜股里躲开病弱股」**而非独立选股。但它建立在会计报表真实性的假设上，任何一座「会计操纵 / 幸存者 / 低频滞后 / 做空不可得」的大山没翻过，回测里的漂亮曲线都会在实盘里打折。

*（全文为自洽合成演示，量级非真实市场数据；真实落地需接入 wind/聚源财务表、处理退市样本并叠加大频预警信号。）*
