---
title: "可转债量化策略:债底保护、转股期权与双低实战"
description: "可转债是「债券 + 看涨期权」的混合体,天然具备债底保护和上涨弹性。本文拆解纯债价值、转股价值与转股溢价率三大指标,用 Python 复现价值拆解、双低选股与月度换仓回测框架,并指出流动性、信用风险与强赎踩踏等真实陷阱。"
publishDate: '2026-07-10'
tags:
  - 量化交易
  - 可转债
  - 双低策略
  - 期权
  - Python
language: Chinese
difficulty: advanced
---

散户常说可转债"下有保底、上不封顶",这句话一半对、一半是营销话术。对量化而言,可转债的本质非常清晰:**它是一张债券,外加一份以正股为标的的看涨期权**。债底决定了下行保护,期权决定了上行弹性。把这两部分拆开、量化、再组合起来交易,就是可转债策略的全部基本功。

本文拆解三个核心指标,并用 Python 把双低策略从选股到回测完整跑一遍。

![可转债价值拆解:债底保护 + 转股期权](/images/convertible-bond-quant-strategy/cb_value_decomposition.png)

## 一、可转债的价值从哪里来

一张可转债的面值通常是 100 元,票息很低(常见年化 0.5%-2%)。它的价值由两部分叠加:

- **纯债价值(债底)**:假设它永远不转股,只当债券持有,按市场收益率(YTM)把未来票息和本金折现到今天。债底是可转债价格的"地板"。
- **转股价值**:转股价 `K` 把面值 100 元换成 `100/K` 股正股,所以转股价值 = `100/K × 正股价`。正股涨,转股价值线性上涨。
- **期权价值**:市价与"债底和转股价值二者孰高"之间的差额,就是那份看涨期权的隐性价值。

下面用代码把这三块算出来。注意转股溢价率与纯债溢价率都用**小数**表示,后面双低公式会乘以 100 对齐价格量纲。

```python
import numpy as np

def bond_floor(coupon, face, ytm, T):
    """纯债价值(债底):票息 + 本金按 YTM 折现。"""
    t = np.arange(1, T + 1)
    return np.sum(coupon / (1 + ytm) ** t) + face / (1 + ytm) ** T

def cb_metrics(price, stock_px, conv_price, face=100.0, coupon=2.0, ytm=0.03, T=3.0):
    bond = bond_floor(coupon, face, ytm, T)     # 债底
    conv_value = face / conv_price * stock_px   # 转股价值
    conv_premium = (price - conv_value) / conv_value   # 转股溢价率(小数)
    bond_premium = (price - bond) / bond               # 纯债溢价率(小数)
    return dict(bond=round(bond, 2), conv_value=round(conv_value, 2),
                conv_premium=round(conv_premium, 4), bond_premium=round(bond_premium, 4))

print(cb_metrics(price=125, stock_px=18, conv_price=15))
```

输出 `{'bond': 97.17, 'conv_value': 120.0, 'conv_premium': 0.0417, 'bond_premium': 0.2864}`:债底约 97 元(低于市价 125,说明价格主要不是靠债底撑着),转股价值 120 元,转股溢价率 4.17%(接近平价,股性较强),纯债溢价率 28.6%(相对债底偏贵)。再把正股价从 18 拉到 20:转股价值升到 133.3,转股溢价率降到 (125−133.3)/133.3≈−6.3%,转债进入折价——此时它几乎完全跟随正股,债性退场、股性主导。这正是可转债“涨时像股、跌时像债”的结构性来源:正股低迷时债底托底,正股上涨时期权价值兑现。

## 二、双低策略:把"便宜"和"股性"合成一个分数

最经典的可转债轮动策略是**双低**:用 `双低 = 价格 + 转股溢价率×100` 给每只转债打分,分数越低越"便宜且股性强",选最低的 N 只等权持有、定期轮动。

直觉:价格低 → 债底保护厚、下行空间小;转股溢价率低 → 跟涨正股紧密、股性强。两者都低,就是"便宜的好品种"。

![双低策略:价格-转股溢价率散点](/images/convertible-bond-quant-strategy/cb_double_low_scatter.png)

```python
import pandas as pd

def double_low_select(universe: pd.DataFrame, top=10):
    """双低 = 价格 + 转股溢价率×100;升序选取最低 top 只。"""
    u = universe.copy()
    u["double_low"] = u["price"] + u["conv_premium"] * 100
    u = u.sort_values("double_low").head(top)
    return u[["code", "price", "conv_premium", "double_low"]]

univ = pd.DataFrame({
    "code": [f"CB{i:03d}" for i in range(6)],
    "price": [108, 132, 99, 145, 115, 121],
    "conv_premium": [0.12, 0.35, -0.02, 0.48, 0.22, 0.28],  # 小数
})
print(double_low_select(univ, top=3))
```

结果按双低升序选出了 `CB002(97.0)`、`CB000(120.0)`、`CB004(137.0)`。注意 `CB002` 转股溢价率为负(-2%),是折价转债(转股价值高于市价),双低分数最低,天然进入组合。

![可转债转股溢价率分布](/images/convertible-bond-quant-strategy/cb_premium_distribution.png)

## 三、Python 实战:双低月度换仓回测框架

下面搭一个最小可用的双低回测骨架:每月按双低排序选前 N 只、等权持有到下次换仓,用区间平均收益近似组合收益。**这是示意框架,真实回测必须处理换仓日流动性、交易成本和退市**。

```python
import pandas as pd, numpy as np

def backtest_double_low(history, top=10, rebal=1):
    """按月换仓的双低回测(示意框架)。history 含 date/code/price/conv_premium/ret。"""
    records = []
    dates = sorted(history["date"].unique())
    for i in range(0, len(dates), rebal):
        d = dates[i]
        snap = history[history["date"] == d].copy()
        snap["double_low"] = snap["price"] + snap["conv_premium"] * 100
        hold = snap.sort_values("double_low").head(top)["code"].tolist()
        nxt = dates[min(i + rebal, len(dates) - 1)]
        sub = history[(history["date"] > d) & (history["date"] <= nxt)]
        ret = sub.groupby("code")["ret"].mean()
        port = ret[ret.index.isin(hold)].mean()
        records.append((d, len(hold), round(float(port), 5)))
    return pd.DataFrame(records, columns=["date", "n_hold", "port_ret"])

rng = np.random.default_rng(0)
codes = [f"CB{i:03d}" for i in range(40)]
dates = pd.date_range("2024-01-31", periods=12, freq="ME")
rows = []
for d in dates:
    for c in codes:
        rows.append({"date": d, "code": c, "price": rng.uniform(95, 150),
                     "conv_premium": rng.uniform(-0.05, 0.50),
                     "ret": rng.normal(0.005, 0.03)})
hist = pd.DataFrame(rows)
print(backtest_double_low(hist, top=10, rebal=1))
```

框架会逐月输出持有数量与组合收益。注意最后一个月 `port_ret` 为 `NaN`(无下一区间可计算),真实系统里应单独处理。这个骨架的价值不在于"收益多高",而在于**把选股-换仓-绩效归因的完整链路跑通**,方便你后续加上成本、退市、规模约束。

## 四、条款博弈:下修、强赎与回售

可转债真正的“阿尔法增量”常藏在条款里,理解它们是做转债量化的必修课:

- **下修转股价**:当正股持续低迷,董事会可提议下调转股价 `K`。转股价一降,转股价值 `100/K × 正股价` 立刻抬升,转债价格随之重估——这是低价转债最重要的向上期权,但下修与否、幅度完全由公司决策,只能做概率博弈,不能当确定性收益。
- **强制赎回**:多数转债约定,正股连续 N 日(通常 15/30)高于转股价 130%,发行人有权以面值+利息赎回。一旦触发,持有人必须在赎回前转股或卖出,否则按约 100 元出头被赎回。举例:某转债市价 130 元、已进入强赎倒计时且溢价率为正,若你持有不动被赎回,瞬间亏掉约 30%——**双低策略必须剔除强赎倒计时内的正溢价标的**。
- **回售条款**:正股持续低于转股价 70% 时,持有人可按面值+利息把转债卖回给公司,这是债底之外的一道保护,但也意味着发行人有更强动机在下修前避免回售。

这三类条款共同决定了转债的“时间价值结构”,任何忽略条款状态的双低打分,都是在盲人摸象。

## 五、双低策略的进阶变体

基础双低只是起点,实战中常见的增强:

1. **双低 + 低余额**:叠加转债余额过滤,优先选规模小(如 < 5 亿)的标的,弹性与资金关注度更高,但要小心流动性。
2. **双低 + 溢价率硬约束**:剔除溢价率过高(如 > 40%)的标的,避免选到“股性死、债性贵”的僵尸债。
3. **双低 + 正股质量**:叠加正股动量或基本面评分,让组合在“便宜”之外再押注正股质量。
4. **加权而非等权**:按双低分数倒数加权,分数越低权重越高,放大便宜标的的贡献。

这些变体本质都是在双低这个分数上叠加新的维度,核心不变:**用可量化的打分把“便宜且股性强”变成可排序、可换仓的规则。**

## 六、常见陷阱

1. **流动性陷阱**:很多低价转债日均成交仅几百万元,回测里能买、实盘根本吃不到量。双低组合常偏向小盘低价债,必须加日成交/余额过滤。
2. **信用违约风险**:债底不是“无风险地板”,发债主体违约时债底会崩。低价转债往往正是因为正股差、信用风险高才便宜——“下有保底”在违约场景下不成立。
3. **强赎踩踏**:正股连续多日高于转股价 130% 会触发强制赎回,持有人必须在赎回前转股或卖出,否则按面值+利息被赎回,瞬间巨额亏损。双低策略必须剔除已进入强赎倒计时、且溢价率为正的标的。
4. **下修博弈的不确定性**:董事会下修转股价能凭空抬升转股价值,是重要 alpha 来源,但下修与否、下修幅度完全由公司决策,无法严格前视,只能做概率博弈。
5. **估值用收盘价**:用收盘价算转股价值默认“随时能按市价转股”,但转股有 T+1 到账、正股又涨又跌,真实套利收益远小于账面。回测若直接用收盘价做无风险套利假设,会严重高估。
6. **忽略条款状态**:不区分“已进入强赎倒计时”“满足回售条件”“董事会提议下修”的标的,等于把时间价值结构完全不同的转债放在同一个打分池里,结果自然失真。

## 小结

可转债是少数同时给量化提供**下行保护(债底)和上行弹性(期权)**的品种,双低策略则是把这种结构变成可执行打分的最简范式。但它的每一个"便宜"信号背后,都对应一个真实风险(流动性、信用、强赎)。把本文的 `cb_metrics` / `double_low_select` / `backtest_double_low` 三段代码接上真实行情与退市数据,再逐条补上上面的陷阱过滤器,你才真正拥有一套能活过实盘的可转债策略。
