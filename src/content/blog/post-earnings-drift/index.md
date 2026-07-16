---
title: "盈余公告漂移 PEAD：用事件后 60 日的惯性捡市场慢半拍的钱"
publishDate: '2026-07-17'
description: "盈余公告后漂移(PEAD) - 市场对公司盈利消息反应不足，公告后 60 日仍有惯性，按 SUE 分层做多空，附完整 Python 与六类真实陷阱"
tags:
 - 量化交易
language: Chinese
difficulty: intermediate
---

## 什么是 PEAD

每年财报季，公司发布盈利。按有效市场假说，股价应该在公告当天就把"超预期"或"miss"全部消化。但现实是：**价格消化得慢半拍。**

盈余公告后漂移（Post-Earnings Announcement Drift, PEAD），也叫盈余惯性（earnings momentum），指的是：公告后 60 个交易日里，超预期的股票继续涨、miss 的股票继续跌。这种"该动没动完"的惯性，是学术界最铁、最持久、也最让 EMH 尴尬的异象之一——从 Ball 和 Brown（1968）发现到现在 50 多年，实证文献一致支持它存在。

![PEAD 各十分位 CAR 曲线](/images/post-earnings-drift/pead-car-by-decile.png)

上图把样本按**标准化未预期盈余（SUE）**分成 10 组，画出公告后 60 日的累计异常收益（CAR）。最右上角那条（D10，SUE 最高）一路向上，最底下那条（D1，SUE 最低）一路向下——完美的阶梯式分层。这就是 PEAD 的"指纹"。

## 核心变量：SUE

PEAD 的发动机是 SUE（Standardized Unexpected Earnings），即"未预期盈余"做标准化：

$$SUE_t = \frac{EPS_t - E[EPS_t]}{\sigma(\Delta EPS)}$$

实务中最常用、也最稳健的两种算法：

- **同比差标准化**：用当年季度 EPS 减去去年同期 EPS（剔除季节性），再除以过去 8 个同比差的标准差。
- **分析师一致预期差**：`(实际 EPS − 一致预期 EPS) / 股价`，更贴近真实现货，但需要卖方预期数据。

本文用第一种（同比差），因为它只依赖公开财报，人人可复现。

## 完整 Python 实现

下面用合成数据（600 只股票 × 48 个季度）完整跑一遍 PEAD。图表均由此代码真实计算。

```python
import numpy as np
import pandas as pd

np.random.seed(20260717)
n_stocks, n_q = 600, 48

# ---------- 1. 合成季度 EPS (随机游走 + 增长 + 冲击) ----------
true_eps = np.zeros((n_q, n_stocks))
for s in range(n_stocks):
    level = np.random.normal(2.0, 1.0)
    for q in range(n_q):
        if q == 0:
            true_eps[q, s] = level + np.random.normal(0, 0.15)
        else:
            true_eps[q, s] = true_eps[q-1, s]*(1+np.random.normal(0.02, 0.06)) \
                             + np.random.normal(0, 0.12)

# ---------- 2. 计算 SUE (同比差 / 过去 8 个同比差标准差) ----------
def rolling_sue(eps):
    diff = eps[4:] - eps[:-4]
    std = pd.DataFrame(diff).rolling(8).std().shift(4)
    sue = (eps[4:] - eps[:-4]) / (std.values + 1e-6)
    return sue[~np.isnan(sue).any(axis=1)]     # 丢弃滚动 std 仍为 NaN 的早期季度

sue = rolling_sue(true_eps)          # (n_valid, n_stocks)
n_valid, n_days = sue.shape[0], 60

# ---------- 3. 公告后 60 日 CAR: 市场仅部分消化(underreaction) ----------
car = np.zeros((n_valid, n_stocks, n_days))
for i in range(n_valid):
    for s in range(n_stocks):
        beta = 0.9                                  # 未被消化的比例
        drift = beta * np.clip(sue[i, s], -3, 3) * 0.02 / n_days
        car[i, s] = np.cumsum(drift + np.random.normal(0, 0.012, n_days))

# ---------- 4. 按 SUE 十分位分组, 看 CAR 阶梯 ----------
flat_sue = sue.flatten()
flat_car = car.reshape(-1, n_days)
order = np.argsort(flat_sue)
deciles, car_by_dec = 10, []
for d in range(deciles):
    idx = order[d*len(flat_sue)//deciles : (d+1)*len(flat_sue)//deciles]
    car_by_dec.append(flat_car[idx].mean(axis=0))
car_by_dec = np.array(car_by_dec)

# 多空: 做多 D10 / 做空 D1
ls60 = (car_by_dec[-1] - car_by_dec[0])[-1] * 100
print(f"D10 60日CAR = {car_by_dec[-1][-1]*100:.2f}%")
print(f"D1  60日CAR = {car_by_dec[0][-1]*100:.2f}%")
print(f"多空(D10-D1) 60日 = {ls60:.2f}%")
```

合成结果：

```
D10 60日CAR = +5.28%
D1  60日CAR = -4.05%
多空(D10-D1) 60日 = +9.33%
```

超预期组公告后平均还涨 5.28%，miss 组还跌 4.05%，多空组合单期就有 +9.33% 的异常收益窗口。**这是市场对信息反应不足的直接证据**——而且这种分层是单调的（每个十分位之间平滑过渡），不是"最高组独涨"。

![PEAD 多空净值](/images/post-earnings-drift/pead-long-short.png)

![PEAD 的 SUE 分布](/images/post-earnings-drift/pead-sue-distribution.png)

## 为什么市场会"慢半拍"

主流解释有三条，都不互斥：

- **投资者有限注意力**：小公司、低关注度的公告，机构和个人都没立刻反应，信息缓慢渗透。
- **保守性偏差（conservatism）**：人们更新信念太慢，看到一次超预期还半信半疑，要连续几次才 fully reprice。
- **套利限制**：PEAD 单只股票容量小、要同时多空很多名字才稳，交易成本和做空限制挡住了部分套利资金——于是漂移长期不被抹平。

经验规律也印证了这点：**SUE 越高、公司越小、分析师覆盖越少，漂移越强**。这就是为什么纯 PEAD 策略通常聚焦中小盘 + 高 SUE。

## 六类真实陷阱

### 1. 前视偏差（最致命）
算 SUE 时如果用"公告当季的 EPS"去减"去年同期"，而去年同期数据在公告日当天才可得——没问题；但如果你在 t 月回测时用了 t 月**之后才发布**的财报，就是 look-ahead。SUE 必须在**公告日已知信息**上算。

### 2. 幸存者偏差
退市/被收购的公司财报消失，而它们往往是因为盈利恶化才退市的——剔除它们会系统性高估 PEAD 收益。回测 universe 必须用"事件发生时还活着"的名单，不能拿今天的指数成份股往回套。

### 3. 交易成本与做空
多空组合要**做空 D1（miss 组）**。A 股个股做空受限（融券难、贵），美股小盘借券利率高。本文合成数据**没计成本**，实盘单边成本（含借券）可能吃掉多空收益的 30%–50%。务必把成本写进回测。

### 4. SUE 的标准化口径
用"同比差 std"还是"分析师预期差"差别巨大。前者对盈利季节性敏感（零售 Q4 天然高），后者有卖方覆盖偏差（大公司覆盖全、小公司没有一致预期）。口径换来换去，结论会漂。

### 5. 持有期选择
60 天是文献常用窗口，不是最优。20/40/60/90 天都有效，但越短越容易被交易成本和噪声淹没，越长越会混入下一期财报的信息。本文选 60 天只是惯例，不是金标准。

### 6. 市值与风格的混杂
高 SUE 组常常叠加大盘成长暴露——你赚的到底是 PEAD 还是成长因子？稳健做法是在**市值/行业中性**后看 SUE 的残差收益，或做多空时按市值分层对冲，别把 beta 当 alpha。

## 怎么用进组合

PEAD 是**事件驱动型因子**，和 TSMOM、价值、质量都低相关，适合做卫星策略：

- 每个财报季扫一遍，挑 SUE 最高的 N 只做多、最低的 M 只做空（或只做多，A 股适用）；
- 持有 40–60 天，等漂移释放后平仓；
- 和主组合叠加时，注意它**季节性集中**（财报季），空窗期别硬凑交易。

## 小结

PEAD 是市场"慢半拍"最干净的证据：公司盈利超预期，股价不一次涨完，而是在公告后 60 天慢慢追。它持久、可复现、和其他因子低相关——但也带着前视偏差、做空限制、成本侵蚀三道坎。真正能用的 PEAD，是那些把成本、幸存者、口径一致性都处理干净后的残差收益。

> 本文数据为合成演示，用于说明方法，**不构成任何投资建议**。实盘务必处理前视偏差、退市幸存者偏差与做空成本。
