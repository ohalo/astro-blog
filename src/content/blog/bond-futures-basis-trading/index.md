---
title: "国债期货基差交易与 CTD 期权：把「最便宜交割权」变成可交易的 Alpha"
description: "国债期货基差 = 现券净价 - 期货价格 × 转换因子，看似简单的相对价值价差，却内含空方交割期权。本文拆解基差、净基差、最便宜可交割券(CTD)与隐含回购利率(IRR)的数学关系，用 Python 复现 CTD 筛选、IRR 计算与基差交易损益情景，并指出 CTD 切换、资金利率上行、逼空三大真实陷阱。"
publishDate: '2026-07-10'
tags:
  - 量化交易
  - 国债期货
  - 基差交易
  - 相对价值
  - Python
language: Chinese
difficulty: advanced
---

在利率量化里，有一种策略既不赌方向、也不赌波动率，而是赚"定价偏差收敛 + 持有收益(carry)"的钱——**国债期货基差交易(basis trade)**。它被认为是固定收益相对价值(relative value)的基石。但很多新手一上手就亏，原因往往是忽略了一个隐藏变量：**空方交割期权**。

本文把基差这件"小事"讲透：先定义基差与净基差，再讲清楚 CTD（最便宜可交割券）和内含期权，最后用 Python 把三件事跑出来——CTD 筛选、IRR 计算、基差交易损益情景，并指出它真正会亏在哪儿。

![国债期货基差随交割临近收敛：净基差=内含交割期权价值](/images/bond-futures-basis-trading/basis_over_time.png)

## 一、基差与转换因子：两个价格之间的桥

中金所（或 CBOT）的国债期货，不是绑定某一只特定债券，而是绑定一篮子**可交割券**(deliverable basket)。卖方（空方）在交割时，可以从篮子里任选一只券交割给买方。为了把不同票面、不同期限的券折算成"标准券"，交易所定义了**转换因子(Conversion Factor, CF)**：

> 基差(Basis) = 现券净价(Bond Clean Price) − 期货价格(Futures Price) × CF

直观理解：你持有 100 元面值的某只可交割券，做空 1 张对应期货（按 CF 折算后等价于 CF×100 元面值），到期交割时，期货端按 `期货价 × CF` 结算，现券端按净价结算，两者之差就是基差。基差交易的核心赌注是：**这个差值会收敛到它的"公允价值"——也就是内含期权价值。**

注意我们用的是**净价**(clean price)，不含应计利息。期货报价本身也是净价口径，所以等式两边口径一致。

## 二、净基差：基差里藏着的期权价值

基差还可以进一步拆成"持有收益"和"期权价值"两部分：

> 净基差(Net Basis) = 基差 − 持有收益(Carry)
>
> 持有收益 ≈ (票息收益 − 融资成本) × 距交割天数 / 365 × 现券全价

持有收益(carry)是你持有现券到交割日能赚的"时间价值"：票息减去你借钱买券的资金成本。如果你做多基差（多现券+空期货），这段时间你吃票息、付融资，净carry就是你的天然收益。

**净基差不为零的部分，就是期权价值。** 为什么会有期权？因为空方有选择权：

1. **券种选择权(quality option)**：在交割月，空方可以选篮子里最便宜的券交割——这就是 CTD。
2. **月末期权(timing option / end-of-month option)**：空方可以在交割月最后几个交易日里，挑一天完成交割，利用这段时间的波动占便宜。
3. **Wild card option**：在 CBT 规则下，空方甚至在收盘后还能根据结算价微调交割决策。

这些期权是空头"免费"拥有的，所以多头（做多基差的人）必须为之付费——体现在净基差长期为正。当交割临近、期权价值归零时，净基差也收敛到 0，基差收敛到 carry。

## 三、Python 实战 1：计算基差、净基差并找出 CTD

下面用一组演示可交割券，把基差、净基差、持有收益算出来，并选出净基差最低（最便宜）的券——理论上它就是市场会选的 CTD。

```python
import numpy as np

# 演示用：某 10 年期国债期货的可交割券篮子
# 字段: 券名, 净价, 转换因子CF, 票息%, 距交割天数, 年化融资利率%
basket = [
    ("券A-7Y",  101.20, 0.918, 2.80, 120, 1.90),
    ("券B-10Y", 103.45, 0.952, 3.10, 120, 1.90),
    ("券C-10Y", 102.10, 0.934, 2.95, 120, 1.90),
    ("券D-7Y",  100.55, 0.905, 2.65, 120, 1.90),
    ("券E-5Y",   99.80, 0.880, 2.40, 120, 1.90),
    ("券F-10Y", 104.10, 0.961, 3.25, 120, 1.90),
]
futures_price = 107.30  # 期货报价(净价口径)

def carry(clean_price, coupon, days, funding):
    """近似持有收益(元/百元面值): 票息收益 - 融资成本。"""
    coupon_income = coupon / 100 * clean_price * days / 365
    financing     = funding / 100 * clean_price * days / 365
    return coupon_income - financing

print(f"{'券名':<10}{'基差':>8}{'持有收益':>10}{'净基差':>10}")
best_ctd, best_net = None, np.inf
for name, clean, cf, cp, days, fund in basket:
    basis = clean - futures_price * cf
    cy = carry(clean, cp, days, fund)
    net = basis - cy
    if net < best_net:
        best_net, best_ctd = net, name
    print(f"{name:<10}{basis:>8.3f}{cy:>10.3f}{net:>10.3f}")

print(f"\n→ 净基差最低(最便宜)的券 = {best_ctd}，净基差 {best_net:.3f}")
```

跑出来的结论是：**净基差最低的券，就是市场公认的 CTD**——因为它交割时"亏得最少"（或者说现券相对期货最便宜）。做空期货的人会优先交这只券，所以基差交易首先要判断 CTD 是哪个。

## 四、隐含回购利率(IRR)：CTD 的另一种判据

除了净基差，业界更常用**隐含回购利率(Implied Repo Rate, IRR)**来判断 CTD：

> IRR = (交割日空方收到的钱 − 今天买入现券花的钱) / 今天买入现券花的钱，再年化

交割日空方收到的钱 = 期货价 × CF + 持有期间票息再投资收益。IRR 可以理解为"你借钱买这只券、同时做空期货锁定交割、到期交割还款"这笔"回购"的年化收益率。**IRR 最高的券，就是 CTD**——因为它让空方赚钱最多。

```python
def implied_repo_rate(clean, cf, coupon, futures, days, fund_ignore=0.0):
    """计算单只可交割券的隐含回购利率(年化 %)。"""
    # 交割收入 = 期货结算价×CF + 票息(简化不计票息再投资)
    delivery_proceeds = futures * cf + coupon / 100 * clean * days / 365
    cost = clean                      # 今天买入现券的净价(近似)
    irr = (delivery_proceeds - cost) / cost * 365 / days * 100
    return irr

print(f"{'券名':<10}{'IRR(%)':>10}")
irrs = {}
for name, clean, cf, cp, days, fund in basket:
    r = implied_repo_rate(clean, cf, cp, futures_price, days)
    irrs[name] = r
    print(f"{name:<10}{r:>10.3f}")

ctd = max(irrs, key=irrs.get)
print(f"\n→ IRR 最高的券 = {ctd} ({irrs[ctd]:.3f}%) = CTD")
```

![可交割券 IRR 对比：IRR 最高者即为最便宜可交割券(CTD)](/images/bond-futures-basis-trading/ctd_irr.png)

IRR 和净基差其实是同一枚硬币的两面：**净基差最低 ≈ IRR 最高**。实务上两个都算，互相验证。当两者指向不同券时，往往意味着转换因子近似或票息再投资假设有偏差，需要人工复核。

## 五、基差交易实务：做多还是做空？

基差交易的方向，本质是"赌净基差（期权价值）怎么变"：

- **做多基差(Buy Basis) = 多现券 + 空期货**。赚持有收益 + 期权价值收敛到 0。适合净基差被高估、且你认为期权价值会如期归零的情形。交割时如果现券就是 CTD，你能稳定吃到 carry。
- **做空基差(Sell Basis) = 空现券 + 多期货**。赌期权价值**现在**就归零（比如临近交割、波动率极低），净基差很快收敛。这是"收期权费"的策略，但一旦期权价值不归零甚至扩大，你就亏。

关键约束：**做空基差的人，交割时没有券选择权**——如果到期你被要求用非 CTD 的贵券交割，成本会很高。所以做空基差通常只在交割前极短窗口、且确信 CTD 稳定的时候做。

## 六、Python 实战 2：基差交易损益情景

下面给一个做多基差的组合 PnL 模型，覆盖四种真实情景。注意多空两端的风险来源是**不对称**的。

```python
def basis_trade_pnl(scenario, notional=100_000_000):
    """
    做多基差(多现券+空期货)的情景损益，单位: 元。
    scenario: 'converge' 期权归零 / 'ctd_switch' CTD切换 /
              'rates_up' 资金利率上行 / 'squeeze' 逼空
    """
    face = notional / 100.0 * 1.0  # 粗略按百元面值缩放
    if scenario == "converge":
        return +face * 0.30      # 净基差如期收敛, 赚 carry+期权
    if scenario == "ctd_switch":
        return -face * 0.55      # 原 CTD 变贵, 交割券切换, 基差不收敛
    if scenario == "rates_up":
        return -face * 0.40      # 融资利率上行, carry 转负
    if scenario == "squeeze":
        return -face * 0.90      # 现券稀缺逼空, 空方高价回补
    return 0.0

for s in ["converge", "ctd_switch", "rates_up", "squeeze"]:
    pnl = basis_trade_pnl(s)
    print(f"{s:<12} 做多基差 PnL = {pnl:>+12,.0f} 元")
```

![基差交易情景分析：多空两端的风险来源截然不同](/images/bond-futures-basis-trading/basis_trade_pnl.png)

可以看到：**做多基差的最大亏损来自 CTD 切换和逼空**——这两件事都和"空方选择权"有关；而做空基差的最大亏损则来自期权价值不归零、甚至因为波动率上升而扩大。所以基差交易不是"无风险套利"，而是**承担特定风险换取 carry**。

## 七、Python 实战 3：用净基差监控 CTD 稳定性

CTD 不是一成不变的。当收益率曲线移动时，哪只券最便宜会切换。下面用一个简单框架，扫描不同扁平/陡峭收益率冲击下的 CTD：

```python
def ctd_under_shock(basket, futures_price, yield_shift_bp):
    """对现券净价施加收益率冲击, 重新算净基差并选 CTD。"""
    best, best_net = None, np.inf
    for name, clean, cf, cp, days, fund in basket:
        # 久期近似: 10Y 券久期~8, 7Y~6, 5Y~4
        dur = 8.0 if "10Y" in name else (6.0 if "7Y" in name else 4.0)
        new_clean = clean - dur * (yield_shift_bp / 1e4) * clean
        basis = new_clean - futures_price * cf
        cy = carry(new_clean, cp, days, fund)
        net = basis - cy
        if net < best_net:
            best_net, best = net, name
    return best, best_net

for shift in [-20, 0, +20, +50]:
    ctd, net = ctd_under_shock(basket, futures_price, shift)
    print(f"收益率冲击 {shift:+4d}bp -> CTD = {ctd:<10} 净基差 {net:.3f}")
```

这段代码的实战价值在于：**如果收益率上行 20bp 就让你的 CTD 换了券，说明你做的基差交易极脆弱**——一旦市场真的这么走，基差不收敛反而扩大，你的多头头寸就会吃亏。这是基差交易风控的第一道防线。

## 八、真实陷阱

- **CTD 切换陷阱**：你做多某只券的基差，结果收益率一动，CTD 变成另一只更便宜的券。你的券不再是交割首选，基差不收敛甚至走阔。解法：入场前做收益率冲击测试（如上），只在 CTD 稳定区做。
- **资金利率上行陷阱**：carry = 票息 − 融资。当资金面收紧、Repo 利率飙升，carry 可能转负，做多基差每天都在"流血"。解法：监控滚动回购利率，设置 carry 转负即减仓。
- **逼空(short squeeze)陷阱**：当 CTD 现券在二级市场上被大量锁定（机构囤券交割），空方找不到券，只能高价回补，基差瞬间拉爆。做空基差的人尤其危险。解法：交割月前降低做空基差敞口，避开"老券稀缺"时段。
- **期权价值误判陷阱**：低利率、高波动环境下，月末期权和 wild card 期权价值很高，净基差长期为正且收敛很慢。把"做多基差"当成稳赚 carry，结果期权价值迟迟不归零。解法：用期权定价思想（而非简单持有收益）给净基差定价的下界。
- **转换因子近似陷阱**：CF 是交易所用近似票息(3%)算的，对高/低票息券有偏差，直接用 CF 算基差会系统性偏误。解法：用更精细的全价基差 + 期权调整。

## 九、落地清单

1. 每个交易日收盘后，重算篮子内所有可交割券的基差、净基差、IRR，确认 CTD 是谁。
2. 对 CTD 做 ±20bp / ±50bp 收益率冲击测试，确认其稳定性。
3. 监控滚动 Repo 利率，carry 转负即触发减仓。
4. 交割月前 2 周，降低做空基差敞口，规避逼空与 wild card 风险。
5. 用净基差的"期权价值部分"而非单纯 carry 来评估安全边际。

## 结语

国债期货基差交易，表面是"现券和期货的差价"，骨子里是**对空方交割期权的定价与博弈**。它不赌方向，赚的是 carry 与定价偏差收敛的钱，但代价是你要持续盯住 CTD 会不会切换、资金利率会不会上行、现券会不会被逼空。把这三件事盯住了，基差交易就是你组合里最稳的相对价值来源之一；盯不住，它就会在某个交割月给你上一堂昂贵的课。

