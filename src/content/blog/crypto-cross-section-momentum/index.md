---
title: "加密横截面动量因子：在币种之间做多强空弱的轮动策略"
publishDate: '2026-07-20'
description: "加密横截面动量因子：在币种之间做多强空弱的轮动策略 - halo的技术博客"
tags:
 - 量化投资
 - 加密货币
language: Chinese
---

传统的动量策略问「这个资产过去涨得好不好，未来会不会继续涨」——这是**时间序列动量（Time-Series Momentum）**。而横截面动量（Cross-Sectional Momentum）问的是另一个问题：「在一篮子资产里，哪些相对更强、哪些相对更弱」，然后**做多相对强者、做空相对弱者**。

加密市场是横截面动量的理想试验场：几百个流动性尚可的币种、7×24 小时交易、极高的波动率、明显的板块轮动。本文用纯 Python 构造一套完整的加密横截面动量因子，从排序、分组到多空组合构建，并诚实拆穿这个在股票市场被验证过无数次的因子，搬到加密市场时会遇到的独特陷阱。

![加密货币动量](/images/crypto-cross-section-momentum/momentum-crypto.jpg)

## 横截面动量的核心逻辑

横截面动量的经济学根基是**羊群效应与信息扩散的滞后**：

- 一个利好消息出现后，价格不会瞬间充分反映，而是逐步扩散，形成持续的相对强势；
- 资金有惯性，强势资产吸引更多买盘，形成正反馈；
- 投资者的处置效应（过早卖出盈利、死扛亏损）也会延缓价格调整。

这套逻辑在股票市场由 Jegadeesh & Titman (1993) 系统验证：过去 3-12 个月的赢家组合，未来会继续跑赢输家组合。加密市场因为参与者更情绪化、板块轮动更剧烈，横截面动量的**信号强度往往更高，但衰减也更快**。

## 关键区别：横截面 vs 时间序列

很多人把这两个概念混为一谈，但它们是完全不同的策略：

| 维度 | 时间序列动量 | 横截面动量 |
|------|-------------|-----------|
| 提问 | 这个资产自己涨得好吗 | 这个资产比别人强吗 |
| 多空 | 单资产做多或做空 | 同时多强者、空弱者 |
| 市场暴露 | 有净方向暴露 | 接近市场中性 |
| 收益来源 | 趋势延续 | 相对强弱分化 |

横截面动量的一个巨大优势是**市场中性**：多头和空头相互对冲，理论上能剥离掉「整个加密市场一起涨跌」的 beta，只保留「币种之间分化」的 alpha。这在加密这种系统性风险极高的市场里尤其宝贵。

## 用 Python 构造多币种价格面板

真实场景应从交易所或数据商拉取多个币种的历史 K 线。这里用自洽的合成过程构造一个包含 20 个币种的价格面板，每个币种有不同的动量强度，用于演示因子构建的完整流程：

```python
import numpy as np
import pandas as pd

np.random.seed(7)

def simulate_universe(n_assets=20, n_days=500):
    """
    构造多币种价格面板
    每个币种有各自的动量因子暴露 + 共同的市场 beta + 特质噪声
    """
    dates = pd.date_range('2025-01-01', periods=n_days, freq='D')

    # 共同市场因子（所有币一起波动的部分）
    market = np.random.randn(n_days) * 0.03

    # 每个币种的动量暴露（有的币趋势强、有的弱）
    momentum_beta = np.random.uniform(-0.5, 0.5, n_assets)

    prices = {}
    for i in range(n_assets):
        rets = np.zeros(n_days)
        for t in range(1, n_days):
            # 动量效应：过去 20 日累计收益影响未来
            if t > 20:
                past_ret = (prices_tmp[t-1] / prices_tmp[t-21] - 1) if t > 21 else 0
                mom_effect = momentum_beta[i] * 0.15 * past_ret
            else:
                mom_effect = 0
            beta = np.random.uniform(0.8, 1.5)
            idio = np.random.randn() * 0.04
            rets[t] = beta * market[t] + mom_effect + idio
            prices_tmp[t] = prices_tmp[t-1] * (1 + rets[t])
        prices[f'COIN{i:02d}'] = prices_tmp.copy()

    return pd.DataFrame(prices, index=dates)

# 需要先初始化 prices_tmp
n_days = 500
prices_tmp = np.zeros(n_days)
prices_tmp[0] = 100

# 重写为更清晰的向量化版本
def simulate_universe_v2(n_assets=20, n_days=500):
    dates = pd.date_range('2025-01-01', periods=n_days, freq='D')
    market = np.random.randn(n_days) * 0.03
    momentum_beta = np.random.uniform(-0.3, 0.6, n_assets)
    panel = {}
    for i in range(n_assets):
        p = np.zeros(n_days); p[0] = 100
        beta = np.random.uniform(0.8, 1.5)
        for t in range(1, n_days):
            mom = momentum_beta[i] * 0.12 * (p[t-1]/p[t-21]-1) if t > 21 else 0
            r = beta*market[t] + mom + np.random.randn()*0.04
            p[t] = p[t-1]*(1+r)
        panel[f'COIN{i:02d}'] = p
    return pd.DataFrame(panel, index=dates)

prices = simulate_universe_v2()
print(prices.iloc[-3:, :5])
```

**注意**：这是合成数据，动量效应是人为注入的，用于演示因子构建流程。真实 alpha 必须在真实多币种数据上验证。

## 计算动量因子并做横截面排序

横截面动量的核心是：在每个调仓日，对所有币种按过去 N 日收益排序，取头部做多、尾部做空。

```python
def compute_momentum_factor(prices, lookback=30, skip=1):
    """
    动量因子 = 过去 lookback 日收益，跳过最近 skip 日
    skip 用于避免短期反转（反转效应会污染动量信号）
    """
    # 跳过最近 skip 日，用 [t-lookback-skip, t-skip] 区间的收益
    momentum = prices.shift(skip) / prices.shift(lookback + skip) - 1
    return momentum

momentum = compute_momentum_factor(prices, lookback=30, skip=1)
print(momentum.iloc[-1].sort_values(ascending=False).head())
```

**为什么要 skip 最近 1 天？** 加密市场存在明显的**短期反转效应**——昨天暴涨的币，今天往往回调。如果动量因子包含最近 1-2 天的收益，会把短期反转的噪声混进来，稀释中期动量信号。跳过最近 1 天是学术界和业界的标准处理。

## 构建多空组合

按因子值分组，头部（赢家）做多、尾部（输家）做空，等权配置：

```python
def build_long_short_portfolio(prices, momentum, n_quantile=5, rebalance=7):
    """
    多空组合：做多动量最强的 1/5，做空最弱的 1/5
    每 rebalance 天调仓一次
    """
    dates = prices.index
    returns = prices.pct_change()

    portfolio_ret = pd.Series(0.0, index=dates)
    long_ret = pd.Series(0.0, index=dates)
    short_ret = pd.Series(0.0, index=dates)

    current_long = []
    current_short = []

    for t in range(len(dates)):
        # 调仓日：重新选股（信号在 t 确认，t+1 起生效）
        if t % rebalance == 0:
            mom_t = momentum.iloc[t].dropna()
            if len(mom_t) >= n_quantile * 2:
                ranked = mom_t.sort_values(ascending=False)
                n_pick = len(ranked) // n_quantile
                current_long = ranked.head(n_pick).index.tolist()
                current_short = ranked.tail(n_pick).index.tolist()

        # 用 t 日确定的持仓，赚 t 日的收益（注意：动量已 shift，无前视）
        if current_long and t > 0:
            l = returns.iloc[t][current_long].mean()
            s = returns.iloc[t][current_short].mean()
            long_ret.iloc[t] = l
            short_ret.iloc[t] = s
            # 多空组合：做多赢家、做空输家
            portfolio_ret.iloc[t] = l - s

    return pd.DataFrame({
        'long': long_ret,
        'short': short_ret,
        'long_short': portfolio_ret
    })

pnl = build_long_short_portfolio(prices, momentum)
```

**时序安全性说明**：`momentum` 因子在计算时已经 `shift(skip)`，所以第 t 日用的动量信号完全基于 t 日之前的数据，用它决定的持仓赚取 t 日收益不构成前视偏差。这是横截面回测里最容易出错的地方——一定要确认因子和收益的时间对齐。

![组合再平衡](/images/crypto-cross-section-momentum/portfolio-rebalance.jpg)

## 回测评估：多头、空头、多空分别看

多空组合的关键是分别看三条曲线——只有把多头贡献、空头贡献、组合表现拆开，才能知道 alpha 到底来自哪一边。

```python
def evaluate_portfolio(pnl, fee=0.001, rebalance=7):
    warmup = 35   # lookback+skip 之后才有有效信号
    ev = pnl.iloc[warmup:].copy()

    # 交易成本：每次调仓两边都换仓
    n_rebalances = len(ev) // rebalance
    total_cost = n_rebalances * fee * 2
    ev['ls_net'] = ev['long_short'] - (total_cost / len(ev))

    equity = (1 + ev['ls_net']).cumprod()

    total_return = equity.iloc[-1] - 1
    ann_return = (1 + total_return) ** (365 / len(ev)) - 1
    sharpe = ev['ls_net'].mean() / ev['ls_net'].std() * np.sqrt(365)

    cummax = equity.cummax()
    max_dd = ((equity - cummax) / cummax).min()

    print(f"多空组合总收益: {total_return:.2%}")
    print(f"年化收益:      {ann_return:.2%}")
    print(f"夏普比率:      {sharpe:.2f}")
    print(f"最大回撤:      {max_dd:.2%}")
    print(f"多头累计:      {(1+ev['long']).prod()-1:.2%}")
    print(f"空头累计:      {(1+ev['short']).prod()-1:.2%}")
    return equity

equity = evaluate_portfolio(pnl)
```

**评估口径**：所有指标在 warmup（35 日）切片之后重算，避免动量因子未成型的初期污染夏普和回撤。多头、空头单独统计，用来判断策略是「靠做多强者赚钱」还是「靠做空弱者赚钱」——这在不同市场环境下含义完全不同。

## 分层检验：因子是否单调有效

一个真正有效的因子，应该表现出**单调性**：动量越强的组，未来收益越高。做一个五分位分层检验：

```python
def quantile_analysis(prices, momentum, n_q=5, rebalance=7):
    returns = prices.pct_change()
    dates = prices.index
    q_returns = {f'Q{i+1}': [] for i in range(n_q)}

    for t in range(35, len(dates)):
        if t % rebalance != 0:
            continue
        mom_t = momentum.iloc[t].dropna()
        if len(mom_t) < n_q:
            continue
        ranked = mom_t.sort_values(ascending=False)
        groups = np.array_split(ranked.index, n_q)
        # 未来 rebalance 天的收益
        fwd = returns.iloc[t+1:t+1+rebalance]
        for i, g in enumerate(groups):
            q_returns[f'Q{i+1}'].append(fwd[list(g)].mean().mean())

    summary = {q: np.mean(v) for q, v in q_returns.items()}
    for q, r in summary.items():
        print(f"{q} (Q1=最强动量): {r:.4%}")
    return summary

quantile_analysis(prices, momentum)
```

理想结果是 Q1（最强动量）收益 > Q2 > ... > Q5（最弱）。如果分层不单调，比如 Q1 和 Q3 收益差不多、Q5 反而最高，那说明这个因子在当前市场里不是稳定的动量，可能混进了反转或其他效应。**单调性是因子有效性的黄金标准，比单看多空收益更能揭示真相。**

![加密币种排名](/images/crypto-cross-section-momentum/crypto-ranking.jpg)

## 加密横截面动量的六个真实陷阱

**坑一：币种数量与流动性门槛。** 横截面策略需要足够宽的 universe 才能分出有意义的头尾。但加密市场真正有流动性的币种可能只有几十个，尾部小市值币做空成本极高、甚至借不到币。名义上「做空最弱的 1/5」在实操中往往无法执行。

**坑二：做空的现实约束。** 加密永续合约虽然能做空，但小市值币种的永续合约深度差、资金费率极端、清算风险高。学术回测里干净的「多空对冲」，在真实交易中空头这条腿的摩擦成本可能吃掉大半 alpha。

**坑三：动量崩溃（Momentum Crash）。** 动量因子有一个恶名昭著的特性——在市场剧烈反转时会遭遇「动量崩溃」。当暴跌后市场 V 型反转，之前跌最狠的输家（你的空头）暴力反弹，动量策略会在极短时间内巨额亏损。加密市场的反转比股市更暴烈，这个风险被放大数倍。

**坑四：调仓频率与成本的权衡。** 加密动量衰减快，理论上要频繁调仓才能抓住。但调仓越频繁，交易成本越高。加密的手续费、滑点、资金费率叠加，高频调仓很容易把纸面收益磨光。找到「信号衰减速度」与「交易成本」的平衡点是核心难题。

**坑五：稳定币与锚定资产的污染。** 如果 universe 里混进了 USDT、USDC 这类稳定币或包装资产（wBTC 等），它们的「动量」几乎为零，会严重扭曲排序和分组。构建 universe 时必须严格清洗掉锚定资产。

**坑六：合成数据的自证陷阱。** 本文的价格路径人为注入了动量效应，所以因子必然有效。这是演示流程与代码逻辑，绝不是证明加密横截面动量真实存在。真实验证必须用真实多币种历史数据、严格的样本外测试、并把上述所有摩擦成本建模进去。

## 结语

加密横截面动量是一个理论优雅、逻辑扎实的因子策略——它剥离市场 beta、只赚币种间的相对强弱分化，在系统性风险极高的加密市场里，这种市场中性的思路格外有价值。

但从「回测曲线漂亮」到「真实能赚钱」，中间隔着做空约束、动量崩溃、流动性门槛、调仓成本这几座大山。**横截面动量在股票市场是被反复验证的经典因子，但直接搬到加密市场，任何一个摩擦项都可能让它从盈利变亏损。**

做量化最忌讳的就是看到一条漂亮的净值曲线就上头。真正的功夫，在于把每一个真实世界的摩擦——借不到的币、暴烈的反转、吃人的滑点——一项项诚实地建模进去，然后看策略还剩下多少。剩下的那部分，才是你真正能拿走的 alpha。
