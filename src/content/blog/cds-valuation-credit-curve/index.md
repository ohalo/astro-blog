---
title: "信用违约互换 CDS 估值与信用曲线 bootstrap"
description: "CDS 是信用市场最标准的违约保险：买方定期付保费、卖方在违约时赔(1−R)·名义本金。本文用约化型危险率模型从零估值，给出保护 leg / RPV01 / 平价利差公式，并用逐段 bootstrap 把市场 CDS 溢价曲线反推成违约强度 λ(t) 与累计违约概率，自洽校验还原偏差 0.0000bps，并诚实拆穿「用全期限算单期限利差/忽略回收率/把平价利差当违约概率/忽略 convexity」四类真实陷阱（中阶）。"
publishDate: '2026-07-20'
tags:
  - 量化交易
  - 信用违约互换
  - CDS
  - 信用曲线
  - Bootstrap
  - 危险率模型
  - 信用衍生品
  - Python
language: Chinese
difficulty: intermediate
---

信用违约互换（Credit Default Swap, CDS）是信用市场里最像"保险"的东西：你（买方）每年付一笔保费（spread），万一参考实体违约，卖方赔你 `(1 − R)·名义本金`（R 是回收率，违约后还能收回的比例）。它不持有底层债券，纯粹交易"违约风险"本身。

结论先放这：**CDS 估值的两边是"保护 leg"和"保费 leg（RPV01）"，平价时两者相除得到市场平价利差 s。** 用约化型（危险率）模型，把违约强度 λ(t) 当分段常数，就能用**逐段 bootstrap** 反推出市场 CDS 溢价曲线背后的 λ(t) 与累计违约概率。在 R=40%、r=3%、市场曲线 1y=40bps 到 10y=180bps 的基准下，bootstrap 还原市场曲线的**最大偏差 = 0.0000bps**，10 年累计违约概率 ≈ **28.3%**，5 年 RPV01 ≈ **4.43**。附完整 Python 与四类真实陷阱（中阶）。

![市场 CDS 溢价曲线 (bps)：信用越差，长端越陡](/images/cds-valuation-credit-curve/cds_spread_curve.png)

## 一、CDS 的两只腿

设名义本金为 1，违约时间 τ 服从危险率 λ(t)，生存概率 `S(t) = exp(−∫₀ᵗ λ(u)du)`。CDS 两条现金流：

- **保护 leg（Protection Leg）**：违约发生时付 `1 − R`。在离散期限 t_j 上近似：
  `PV_prot = (1 − R) · Σⱼ DFⱼ · (S_{j−1} − S_j)`
  其中 `S_{j−1} − S_j` 是第 j 段内的违约概率，`DFⱼ = e^{−r·tⱼ}` 是无风险贴现因子。
- **保费 leg（Premium Leg，RPV01）**：每年付 s·名义本金，在存活期间才付：
  `PV_prem = s · Σⱼ DFⱼ · (S_{j−1} + S_j)/2 · Δt_j`
  括号里是这段的平均生存概率（近似存活期），乘 Δtⱼ 是存活年数。

**平价条件**：买方不赚不亏时 `PV_prot = PV_prem`，于是市场平价利差

```
s = PV_prot / RPV01 ,   RPV01 = Σⱼ DFⱼ · (S_{j−1}+S_j)/2 · Δtⱼ
```

RPV01 就是"每 1bp 利差的现值"，是 CDS 世界的 DV01 等价物。

```python
import numpy as np

def cds_legs_upto(h, maturities, df, R, upto):
    """算 1..upto 段的 protection / premium legs (逐期限 CDS 平价用)。"""
    surv = 1.0; prev = 0.0; rpv = 0.0; prot = 0.0
    for i in range(upto):
        m = maturities[i]; dt = m - prev
        s_end = surv * np.exp(-h[i] * dt)
        prot += (1 - R) * df[i] * (surv - s_end)
        rpv  += df[i] * (surv + s_end) / 2.0 * dt
        surv = s_end; prev = m
    return rpv, prot

def par_spread_upto(h, maturities, df, R, upto):
    rpv, prot = cds_legs_upto(h, maturities, df, R, upto)
    return prot / rpv if rpv > 0 else np.inf
```

## 二、Bootstrap 信用曲线：从溢价反推违约强度

实务里我们**先有市场 CDS 利差曲线**（各期限平价 spread），想反推背后的违约强度 λ(t)。关键规则：**第 i 段只由"到期日 ≤ mat[i] 的 CDS"决定**——更长期限的合约在 mat[i] 时还没到期，不提供该段的信息。

逐段做法：从第 1 段开始，已知前 i−1 段 λ，用二分法解 λ_i，使得"期限为 mat[i] 的平价 CDS 利差 = 市场报价"。

```python
r, R = 0.03, 0.40
maturities = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
spreads    = np.array([40,55,70,85,100,115,130,150,165,180], dtype=float)
df = np.exp(-r * maturities)
h = np.zeros_like(maturities)

for i in range(len(maturities)):
    target = spreads[i] / 1e4
    lo, hi = 1e-7, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        h_try = h.copy(); h_try[i] = mid
        s = par_spread_upto(h_try, maturities, df, R, i + 1)
        if s > target:
            hi = mid
        else:
            lo = mid
    h[i] = (lo + hi) / 2.0
```

自洽校验：把 bootstrap 出的 h 逐期限重算平价利差，应和市场报价完全重合。本例**最大偏差 = 0.0000bps**——曲线被完美还原。反推出的违约强度 λ(t) 从 1 年 ≈ 67bps 一路升到 10 年 ≈ 626bps，正是"信用越差、长端越陡"的微观成因。

![Bootstrap 违约强度 λ(t) (年化 bps)：由溢价曲线反推](/images/cds-valuation-credit-curve/cds_hazard_curve.png)

## 三、从 λ(t) 到累计违约概率

有了 λ(t)，累计违约概率 `P(τ ≤ t) = 1 − S(t)` 直接可积。逐段累加各段违约概率：

```python
surv = 1.0; prev = 0.0; cum_def = np.zeros_like(maturities)
for i, m in enumerate(maturities):
    dt = m - prev
    seg = surv * (1 - np.exp(-h[i] * dt))
    cum_def[i] = cum_def[i-1] + seg if i > 0 else seg
    surv *= np.exp(-h[i] * dt); prev = m
```

结果：10 年累计违约概率 ≈ **28.3%**。注意它**不是**简单把利差加起来——利差里混着回收率、贴现和期限结构，必须通过危险率模型正确拆解。

![累计违约概率 P(τ ≤ t)：10y 约违约 28.3%](/images/cds-valuation-credit-curve/cds_cum_default.png)

## 四、公允净现值：你做多还是做空保护

CDS 净现值（ NPV）就是保护 leg 与保费 leg 之差。固定市场 5y=100bps，看你以什么合约利差成交：

```
NPV = (s_contract − s_market) · RPV01 · 名义本金
```

卖保护（收保费）且合约利差 > 市场 → 你赚；买保护且合约利差 < 市场 → 你亏。RPV01 把"利差差"线性放大成钱。

![5y CDS 公允净现值：以 120bps 卖保护赚, 以 80bps 买保护亏](/images/cds-valuation-credit-curve/cds_npv.png)

```python
idx5 = list(maturities).index(5.0)
rpv5, _ = cds_legs_upto(h[:idx5+1], maturities[:idx5+1], df[:idx5+1], R, idx5+1)
notional = 1e7; s_mkt = 100.0 / 1e4
contract = np.linspace(60, 160, 100) / 1e4
npv = (contract - s_mkt) * rpv5 * notional / 1e4   # 万元
print(f"5y RPV01 = {rpv5:.4f}")   # 4.43
```

## 五、四类真实陷阱（必看）

**1. 不要用"全期限"去算"单期限"平价利差。** 这是最常见的实现 bug：bootstrap 第 i 段时，若把 `par_spread` 算在全部 1..10y 上（后段 λ 置 0），反推出的 λ 会严重错乱（比如 1 年 λ 被解成 570bps，而市场只报 40bps）。正确做法是 `par_spread_upto(h, ..., i+1)`——只滚到当前期限。漏掉这点，信用曲线全是噪声。

**2. 回收率 R 不是装饰。** 公式里保护 leg 赔的是 `1 − R` 而非 1。R 从 40% 降到 20%，同样 λ 下保护 leg 翻倍，平价利差直接翻倍。实务里 R 要按资产类别选（优先级债 ~40–60%，次级债 ~20%），乱填 R 会让整条曲线量级错误。

**3. 平价利差 ≠ 违约概率。** 市场报的 100bps 不是"明年违约 1%"。它混了回收率、无风险贴现和期限结构，是经风险中性定价后的**年化保费**。真要违约概率，必须走本文第二节的 bootstrap，由 λ(t) 积分得到（10y 累计违约 28.3%，而 10y 利差才 180bps）。直接把 bps 当概率，信用风险评估会系统性低估一个数量级。

**4. 忽略了 convexity（曲率风险）。** 上面的线性 NPV 公式只在"利差平移"假设下成立。真实 CDS 对利差的敏感度本身随期限结构非线性（risky duration 不是常数），大利差变动下 NPV 会偏离线性外推。做信用风险套保或跨期限 CDS 曲线交易时，要算 second-order 项，否则在危机利差暴涨时对冲会明显失真。

## 结语

CDS 估值的核心是把"违约风险"拆成两笔可贴现的现金流——保护 leg 和保费 leg（RPV01），再用危险率 λ(t) 把生存概率串起来。bootstrap 则是反过来：从市场平价利差曲线，逐段反推 λ(t)、积分出违约概率。这条链路既给了你一条可交易、可对冲的信用曲线，也暴露了"利差不是概率、回收率不是装饰、单期限只能用单期限信息"三道红线。下回看到 CDS 报价，你知道它背后站着一棵 λ(t) 的树，而不是一个孤立的百分比。
