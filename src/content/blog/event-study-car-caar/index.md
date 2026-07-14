---
title: "事件研究 CAR/CAAR：用市场模型量化公告的异常收益"
description: "事件研究法是量化「一则公告到底带来了多少超额收益」的标准工具。从市场模型估计正常收益、算出异常收益 AR，再累积成 CAR/CAAR，并用 t 检验判断显著性。从零实现估计窗-事件窗切分、横截面聚合与显著性检验，附完整 Python 与六类真实陷阱（高阶）。"
publishDate: '2026-07-14'
tags:
  - 量化交易
  - 事件研究
  - 异常收益
  - 市场模型
  - CAR
  - CAAR
  - 显著性检验
  - Python
language: Chinese
difficulty: advanced
cover: "/images/event-study-car-caar/cover.png"
---

财报超预期、突发并购、监管处罚、指数调仓……市场每天都在被各种「公告」冲击。做量化的人迟早会问一个问题：**这则公告，到底给股价带来了多少「本来不会有」的收益？**

这不是拍脑袋能回答的。当天股价涨了 3%，可能是大盘也涨了 2.5%，真正归因于公告的其实只有 0.5%。把「市场本身的贡献」剥离掉，剩下的那部分才叫**异常收益（Abnormal Return, AR）**。而把 AR 沿时间累积、再跨事件平均，就得到事件研究的两大核心量：**CAR（累计异常收益）** 与 **CAAR（平均累计异常收益）**。

结论先放这：**事件研究不是「看图说话」，而是一套有统计检验兜底的因果量化方法。** 在本文的自洽合成里（120 个事件、市场模型剔除 beta 暴露），公告日 AAR 达 +1.2% 且 t 统计量高达 11.4，CAAR 在事件窗末累积到 +2.75%——但如果你不做市场模型调整、不做横截面显著性检验，很容易把「大盘的贝塔」误当成「公告的阿尔法」。

---

## 1. 为什么不能直接看原始收益

假设某公司周二发布财报，当天股价 +3%。你能说「财报带来 3% 收益」吗？不能。因为：

- 如果当天沪深 300 涨了 2.6%，这只股票 beta=1.1，那么「正常」应该涨 2.86%，真正的异常收益只有 **+0.14%**；
- 如果当天大盘跌了 1%，同样 beta=1.1，正常应该跌 1.1%，那么 +3% 里其实有 **+4.1%** 是异常的。

**原始收益混杂了系统性市场波动，必须先扣掉。** 这就是「市场模型（Market Model）」要做的事。

事件研究的整个逻辑链是这样的：

```
估计窗（事件前 T1~T2）→ 拟合正常收益模型（如 market model）
       ↓
事件窗（事件前后 -k~+m）→ 用模型预测「正常收益」
       ↓
异常收益 AR = 实际收益 - 正常收益
       ↓
CAR = 沿时间累积 AR      CAAR = 跨事件平均 CAR
       ↓
t 检验 / 横截面检验 → 判断异常收益是否显著不为 0
```

---

## 2. 三种正常收益模型

「正常收益」怎么定义，直接决定 AR 的质量。常见三种：

| 模型 | 正常收益定义 | 优点 | 缺点 |
|---|---|---|---|
| **均值调整（Mean-adjusted）** | 估计窗内的平均收益 | 最简单 | 完全忽略市场波动，噪声大 |
| **市场调整（Market-adjusted）** | 直接用市场收益 R_m | 无需估计窗 | 隐含假设 α=0、β=1，偏差大 |
| **市场模型（Market Model）** | R_i = α_i + β_i·R_m + ε | 剔除个股 beta 暴露，标准做法 | 需要足够长的估计窗 |

本文采用**市场模型**，这是学术界和业界的默认选择。它在估计窗上用 OLS 回归拟合每只股票的 α 和 β：

$$R_{i,t} = \alpha_i + \beta_i R_{m,t} + \varepsilon_{i,t}$$

事件窗内的正常收益就是 $\hat{\alpha}_i + \hat{\beta}_i R_{m,t}$，异常收益为：

$$AR_{i,t} = R_{i,t} - (\hat{\alpha}_i + \hat{\beta}_i R_{m,t})$$

![单个事件 AR 到 CAR](/images/event-study-car-caar/single_event_car.png)

上图展示了单个事件的 AR（蓝柱，每日异常收益）如何累积成 CAR（红线）。注意公告日 τ=0 的 AR 明显跳升，之后 CAR 台阶式抬高——这就是信息释放被逐步吸收的典型形态。

---

## 3. 从零实现：估计窗、事件窗与 AR

先搭数据结构。关键是**估计窗和事件窗不能重叠**——如果用事件窗内的数据估计 β，事件本身的冲击会污染 β 估计，这是初学者最常见的 look-ahead 陷阱。

```python
import numpy as np
import pandas as pd

def market_model_ar(stock_ret, mkt_ret, event_idx,
                    est_window=(-130, -11), event_window=(-10, 20)):
    """
    stock_ret, mkt_ret: 对齐的日收益序列 (pd.Series, 索引为交易日)
    event_idx: 事件日在序列中的整数位置
    est_window: 估计窗相对事件日的偏移 (含负号)
    event_window: 事件窗相对事件日的偏移
    返回: 事件窗内每日 AR (np.array), 以及估计出的 alpha/beta
    """
    e0, e1 = est_window
    v0, v1 = event_window

    # 1) 切估计窗（严格在事件窗之前，不重叠）
    est_slice = slice(event_idx + e0, event_idx + e1 + 1)
    r_i = stock_ret.iloc[est_slice].values
    r_m = mkt_ret.iloc[est_slice].values

    # 2) OLS 拟合 market model: R_i = a + b*R_m
    X = np.column_stack([np.ones_like(r_m), r_m])
    coef, *_ = np.linalg.lstsq(X, r_i, rcond=None)
    alpha, beta = coef[0], coef[1]

    # 残差标准差（用于单事件 t 检验）
    resid = r_i - (alpha + beta * r_m)
    sigma_ar = resid.std(ddof=2)

    # 3) 事件窗内算 AR
    ev_slice = slice(event_idx + v0, event_idx + v1 + 1)
    ri_ev = stock_ret.iloc[ev_slice].values
    rm_ev = mkt_ret.iloc[ev_slice].values
    ar = ri_ev - (alpha + beta * rm_ev)

    return ar, alpha, beta, sigma_ar
```

**几个关键细节：**

1. `est_window=(-130, -11)`：估计窗结束在事件前第 11 天，留了 10 天缓冲，避免「预期泄漏」（市场可能提前反应传闻）；
2. `ddof=2`：残差自由度扣 2（估了 α、β 两个参数）；
3. 估计窗至少要 120 个交易日，否则 β 估计不稳。

---

## 4. 累积成 CAR，再跨事件聚合成 CAAR

单个事件的 AR 噪声很大，事件研究的威力在于**跨大量事件平均**——噪声相互抵消，信号浮现。

```python
def aggregate_events(ar_matrix):
    """
    ar_matrix: shape (N_events, L) 每行是一个事件的事件窗 AR 序列
    返回 AAR, CAAR 及其标准误
    """
    N, L = ar_matrix.shape

    # 每日平均异常收益 AAR_t = mean_i(AR_i,t)
    aar = ar_matrix.mean(axis=0)

    # CAAR = 沿时间累积 AAR
    caar = np.cumsum(aar)

    # AAR 的横截面标准误
    se_aar = ar_matrix.std(axis=0, ddof=1) / np.sqrt(N)

    # CAAR 标准误（独立近似：方差可加）
    se_caar = np.sqrt(np.cumsum(se_aar ** 2))

    # 每个事件的总 CAR（用于横截面检验）
    car_per_event = ar_matrix.cumsum(axis=1)[:, -1]

    return {
        "AAR": aar, "CAAR": caar,
        "SE_AAR": se_aar, "SE_CAAR": se_caar,
        "CAR_per_event": car_per_event,
    }
```

CAAR 曲线是事件研究最标志性的图。它回答一个问题：**平均而言，这类事件在公告前后如何影响股价？**

![CAAR 曲线与置信带](/images/event-study-car-caar/caar_curve.png)

上图中，CAAR 在 τ=0 前基本贴着 0（说明没有信息泄漏），在公告日附近陡升，随后进入平台期。95% 置信带（阴影）显示：公告日之后 CAAR 显著大于 0——市场对这类公告有真实、可量化的正向反应。

---

## 5. 显著性检验：区分「真信号」和「运气」

看到 CAAR 涨了不等于就有 alpha。必须做**统计检验**。事件研究里有两类主流检验：

### 5.1 时间序列 t 检验（针对 AAR）

对每一天的 AAR，检验它是否显著不为 0：

$$t_{AAR_\tau} = \frac{AAR_\tau}{SE(AAR_\tau)}$$

```python
def significance_tests(res):
    aar, se_aar = res["AAR"], res["SE_AAR"]
    caar, se_caar = res["CAAR"], res["SE_CAAR"]

    t_aar = aar / se_aar           # 每日 AAR 的 t 值
    t_caar = caar / se_caar        # 累积 CAAR 的 t 值

    # 横截面检验：所有事件的总 CAR 是否显著
    car = res["CAR_per_event"]
    t_cross = car.mean() / (car.std(ddof=1) / np.sqrt(len(car)))

    return t_aar, t_caar, t_cross
```

![AAR 与 t 统计量](/images/event-study-car-caar/aar_tstat.png)

上图上半部分是每日 AAR，下半部分是对应的 t 统计量。红色柱表示 t>1.96（5% 显著），灰色表示不显著。**只有公告日及其后 1-2 天的 AAR 是统计显著的**——这正符合「信息在公告日集中释放、随后快速吸收」的市场有效性直觉。在本文合成中，公告日 t 统计量高达 11.4，说明信号极强。

### 5.2 横截面 t 检验（针对 CAR 分布）

时间序列检验假设各事件独立，但如果事件在日历上聚集（比如财报季所有公司同一周发布），横截面相关会低估标准误。更稳健的做法是直接看**每个事件总 CAR 的横截面分布**：

![横截面 CAR 分布](/images/event-study-car-caar/car_cross_section.png)

上图是 120 个事件各自总 CAR 的直方图。均值明显为正（+2.75%），但分布很宽、有噪声尾部——**这提醒我们：平均有 alpha，不代表每个事件都赚钱。** 有相当一部分事件的 CAR 是负的，只是被正向的多数拉高了均值。这对实盘极其重要：事件驱动策略的收益是「偏斜」的，仓位管理必须考虑尾部。

---

## 6. 完整调用流程

把上面的模块串起来：

```python
def run_event_study(panel_returns, market_returns, event_list):
    """
    panel_returns: dict{ticker: pd.Series 日收益}
    market_returns: pd.Series 市场日收益（如沪深300）
    event_list: list of (ticker, event_date)
    """
    ar_rows = []
    for ticker, ev_date in event_list:
        r_i = panel_returns[ticker]
        # 对齐市场收益索引
        r_m = market_returns.reindex(r_i.index)
        ev_idx = r_i.index.get_loc(ev_date)

        # 边界检查：估计窗和事件窗都要在数据范围内
        if ev_idx < 130 or ev_idx + 20 >= len(r_i):
            continue

        ar, a, b, s = market_model_ar(r_i, r_m, ev_idx)
        ar_rows.append(ar)

    ar_matrix = np.vstack(ar_rows)
    res = aggregate_events(ar_matrix)
    t_aar, t_caar, t_cross = significance_tests(res)

    print(f"事件数: {len(ar_rows)}")
    print(f"公告日 AAR: {res['AAR'][10]*100:.3f}%  t={t_aar[10]:.2f}")
    print(f"事件窗末 CAAR: {res['CAAR'][-1]*100:.3f}%  t={t_caar[-1]:.2f}")
    print(f"横截面 CAR 检验 t={t_cross:.2f}")
    return res
```

> 注意 `res['AAR'][10]` 里的索引 10——事件窗是 `(-10, 20)`，所以位置 10 对应 τ=0（公告日）。这种索引偏移是事件研究代码最容易写错的地方，务必和事件窗定义对齐。

---

## 7. 六类真实陷阱

事件研究看起来简单，但坑非常深。以下是六个最常见、也最致命的陷阱：

### 陷阱 1：估计窗与事件窗重叠（look-ahead）
如果用包含事件日的数据估 β，事件冲击会污染 β，导致 AR 被系统性低估。**估计窗必须严格在事件窗之前，且留缓冲带**。

### 陷阱 2：忽略事件聚集导致的横截面相关
财报季、政策发布日，大量事件同一天发生。此时各事件的 AR 高度相关，时间序列标准误会**严重低估**，t 值虚高。解决：用横截面检验，或用 Boehmer 等的标准化残差检验（BMP test）。

### 陷阱 3：事件日定义错误
公告是盘后发的还是盘中发的？盘后公告的市场反应应算在**次日**。搞错事件日会把真正的反应日排除在事件窗外，AR 稀释成噪声。

### 陷阱 4：幸存者偏差与前视选样
「选出所有做过并购的公司」——但那些并购后退市的公司呢？如果只用现在还活着的公司，样本天然偏向成功案例，CAAR 被高估。

### 陷阱 5：混淆事件（confounding events）
公告当天如果同时有分红、高管变动等其他事件，AR 就无法干净归因。**需要人工筛掉事件窗内有其他重大公告的样本**。

### 陷阱 6：正态假设失效
t 检验假设 AR 近似正态，但个股收益有肥尾。小样本（事件数 < 30）时 t 检验不可靠，应改用**非参数检验**（如符号检验、秩检验）或 bootstrap。

---

## 8. 从合成到实盘：数据接入

本文用合成数据是为了让 CAR/CAAR 的机制清晰可复现。接入真实数据时，流程是：

1. **取事件列表**：财报超预期、降准、指数调仓等，事件日期需精确到「市场可交易的第一天」；
2. **取个股日收益**：用后复权价算收益，避免除权除息造成的假跳变；
3. **取市场基准**：A 股用沪深 300，美股用标普 500 或 CRSP 市值加权指数；
4. **对齐交易日历**：停牌、节假日必须对齐，否则 AR 错位。

其中「事件日期的精确确认」尤其重要——一个日期错一天，整个事件窗就偏移，结论可能完全反转。

---

## 9. 总结

事件研究的核心，是把「公告有没有用」这个模糊问题，变成一个**有统计检验兜底的可证伪命题**：

- **市场模型**剔除 beta 暴露，把原始收益净化成异常收益 AR；
- **CAR/CAAR** 沿时间和跨事件双重累积，让信号从噪声里浮现；
- **t 检验 + 横截面检验**判断异常收益是否真的显著，而不是运气。

在本文合成中，公告日 AAR +1.2%（t=11.4）、事件窗末 CAAR +2.75%——信号清晰而显著。但真正的价值不在这个数字，而在于**方法论**：任何时候你想问「某个事件对股价有没有影响、影响多大」，事件研究就是那把标尺。

记住三段式判断：**先剔除市场（别把 beta 当 alpha）→ 再累积聚合（别信单个事件）→ 最后做显著性检验（别把运气当信号）**。三步都过了，你才真正量化了一则公告的价值。
