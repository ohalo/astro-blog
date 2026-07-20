---
title: "资本支出效率因子：把「钱有没有变成利润」做成可交易 alpha"
publishDate: '2026-07-20'
description: "资本支出效率因子：不是看一家公司花了多少钱投资，而是看它花出去的钱有没有变成经营利润。本文用「增量资本回报率」思路构建月度再平衡横截面因子，验证十分组收益严格单调、长短因子长期为正且对市场 beta 近零——独立的低 beta alpha，并诚实拆穿会计口径/投资周期错配/成长陷阱/做空约束四类真实陷阱(中阶)。"
tags:
  - 量化交易
  - 因子研究
  - 资本支出
  - 基本面因子
  - 横截面因子
  - 会计质量
  - Python
language: Chinese
difficulty: intermediate
---

如果有两个 CEO，一个每年砸 100 亿建厂但利润纹丝不动，另一个每年只投 20 亿却把 ROIC 从 8% 拉到 15%——**谁更值得长期持有，答案显而易见**。

但绝大多数「成长」叙事只盯着**投资规模**（资本开支增速、CAPEX/营收），忘了问一句最关键的话：**这些钱，有没有变成利润？**

这篇文章把「资本支出效率」做成一个可回测的横截面因子：

> 用 **增量资本回报率（ROIIC 思路）** 作为信号，月度再平衡，多高效率 / 空低效率的公司，验证它长期为正、十分组严格单调，且**因子对市场 beta 接近零**——说明它不是「押成长」或「押市场」，而是独立的低 beta alpha。

> 数据声明：全文为**自洽合成**（高效率公司未来收益更高，仅用于演示方法），目的是把「资本支出效率因子」的构建与检验机制跑通、可复现。所有量级均为合成校准，真实市场里会被会计口径、投资周期错配、成长陷阱、做空成本大幅压缩——重点看*方法*。

## 一、为什么「花钱多」不等于「好公司」

资本支出（CAPEX）本身是中性的。建产能、买设备、搞研发，都是为未来下注。问题在于：**下注的回报率天差地别**。

- 一家公司 CAPEX 翻倍，但新增产能利用率只有 40%、毛利率还在下滑——这是**价值毁灭**，钱烧进了泥潭。
- 另一家公司 CAPEX 克制，但每一块钱投入都带来更高的经营利润——这是**资本纪律**，钱变成了 ROIC。

学术界很早就注意到了这一点。Titman, Wei & Xie (2004) 发现**过度投资**（over-investment）的公司长期跑输；而「投入资本回报率」类的质量因子（Novy-Marx 的 gross profitability、Hou, Xue & Zhang 的投资因子）本质上都是在回答同一个问题：**这家公司把资本用在了什么地方、用出了什么回报。**

所以我们不看「花了多少」，而看**「花出去的钱 → 赚回来的钱」这个转化效率**。

## 二、信号怎么度量：增量资本回报率 ROIIC

最干净的思想实验是 **ROIIC（Return On Incremental Invested Capital）**：

```
ROIIC_t = ΔEBIT_t / ΔInvestedCapital_{t-1→t}
```

分子是经营利润的变化，分母是期间投入资本（净经营资产）的变化。它衡量的是「**新投进去的这一块钱，产生了多少经营利润**」——比静态 ROIC 更聚焦边际、更不容易被历史低基数美化。

但在横截面上直接算 ROIIC 噪声极大（分母 ΔInvestedCapital 可能很小甚至为负，导致比值爆炸）。实务上有两个处理：

1. **用多年滚动窗口平滑**分母，避免单期异常；
2. **先做行业市值中性化**，再按截面 z 标准化，得到可比的效率排序。

下面用 Python 把整套合成 + 信号构建 + 因子检验跑通。

```python
import numpy as np
import pandas as pd

# ---------- 1) 合成面板：N=600 股票 × M=144 月 ----------
rng = np.random.default_rng(20260720)
N, M = 600, 144
eta = rng.normal(0.0, 1.0, size=N)          # 结构性资本支出效率 η_i
drift = rng.normal(0, 0.12, size=(N, M)).cumsum(axis=1) * 0.05
capex_growth = rng.normal(0, 1.0, size=N)    # 投资强度（与效率独立）

# 实现的效率信号：横截面由 η 主导，含噪声
signal = eta[:, None] + 0.25 * drift + rng.normal(0, 0.45, size=(N, M))

mkt = rng.normal(0.005, 0.04, size=M)
# 未来1月收益：随效率正相关（高效率→跑赢），过度投资轻微拖累
future_ret = (0.004 + 0.005 * eta[:, None]
              - 0.002 * capex_growth[:, None]
              + 0.35 * mkt
              + rng.normal(0, 0.03, size=(N, M))
              + rng.normal(0, 0.003, size=N)[:, None])

# ---------- 2) 截面 z 标准化（与文中配图同口径）----------
signal_z = (signal - signal.mean(axis=0, keepdims=True)) / \
           (signal.std(axis=0, keepdims=True) + 1e-6)
```

截面分布右偏，左尾是「烧钱」公司——它们投入了资本却没换来利润，信号显著为负：

![资本支出效率截面：右偏，左尾是「烧钱」公司](/images/capex-efficiency-factor/capex_distribution.png)

## 三、做成横截面因子：月度再平衡、多高/空低

把标准化信号在每个月对全市场排序，取最高 10% 做多、最低 10% 做空，月度再平衡：

```python
def long_short(sig, ret, n=60):
    """多前 n 名 / 空后 n 名，月度再平衡，返回累计净值序列"""
    M = sig.shape[1]
    nav = np.ones(M)
    monthly = np.zeros(M)
    for t in range(1, M):
        order = np.argsort(sig[:, t - 1])         # 用 t-1 期信号
        long_avg = ret[order[-n:], t].mean()
        short_avg = ret[order[:n], t].mean()
        monthly[t] = long_avg - short_avg
        nav[t] = nav[t - 1] * (1 + monthly[t])
    return nav, monthly

nav_capex, _ = long_short(signal_z, future_ret)
print(f"资本支出效率长短因子 终值: {nav_capex[-1]:.3f}  年化: "
      f"{(nav_capex[-1]**(12/M) - 1)*100:.1f}%")
```

注意一个关键细节：**信号用的是 t-1 期，收益用的是 t 期**。这是避免 look-ahead 的硬性约束——你不能在月底用「已经发生的下月收益」去排序。月度再平衡 + 信号滞后一期，是横截面因子最朴素的防作弊姿势。

长短因子净值长期向上：

![资本支出效率长短因子：长期为正](/images/capex-efficiency-factor/capex_ls_curve.png)

## 四、单调性检验：因子有没有「梯度」

一个好的横截面因子，不应该只有「多/空两头」赚钱，而应该是**十分组收益严格单调**——从 D1 到 D10 一路抬升。这才能证明信号是连续的 alpha 来源，而不是被某两个极端组偶然撑起来的。

```python
def decile_returns(sig, ret, M):
    dec_avg = np.zeros(10)
    for t in range(1, M):
        order = np.argsort(sig[:, t - 1])
        for d in range(10):
            idx = order[d * 60:(d + 1) * 60]
            dec_avg[d] += ret[idx, t].mean()
    return dec_avg / (M - 1)

dec = decile_returns(signal_z, future_ret, M)
mono = np.all(np.diff(dec) > 0)
print(f"十分组收益（%）：{np.round(dec*100, 2)}")
print(f"严格单调递增: {mono}")
```

结果严格单调，D10 - D1 显著为正。这意味着**你不需要做空**，光是「低配低效率、超配高效率」的多头倾斜就能吃到大部分溢价：

![资本支出效率十分组：单调递增，D10-D1 显著为正](/images/capex-efficiency-factor/capex_decile.png)

## 五、它是 alpha 还是 beta？低 beta、正截距

横截面因子最怕的一件事是：**所谓因子收益，其实只是「偷偷押了某个风格」**。比如小盘因子、价值因子，很多时候是在暴露市值/账面市值比。检验方法：把长短因子收益对**市场收益**做 OLS 回归，看 beta 和截距。

```python
beta, alpha = np.polyfit(mkt[1:], nav_capex[1:] / nav_capex[:-1] - 1, 1)
print(f"对市场的 beta = {beta:.2f}  每月截距 = {alpha*100:.3f}%")
```

![资本支出效率因子 vs 市场：低 beta、正截距 = 独立 alpha](/images/capex-efficiency-factor/capex_beta_scatter.png)

beta 接近 0、截距显著为正，说明这个因子的收益**不是市场涨了它才涨**，而是独立于市场方向的超额回报——这才是真正有价值的低 beta alpha。如果 beta 很高、截距不显著，那它只是「加了杠杆的 market beta」，毫无信息含量。

## 六、进阶：和「投资过度」拆开看

前面合成里，效率和投资强度是**独立**的。这正好对应一个重要的二维洞察：**高投资 + 低效率 = 价值毁灭；低投资 + 高效率 = 资本纪律**。

实务上可以做一个 2×2 分组：

```python
def quad_view(eff, inv, ret, M, thr=0.0):
    out = {}
    for e in ["hi_eff", "lo_eff"]:
        for v in ["hi_inv", "lo_inv"]:
            mask = np.zeros((600, M), dtype=bool)
            for t in range(1, M):
                e_m = (eff[:, t-1] > thr) if e == "hi_eff" else (eff[:, t-1] <= thr)
                v_m = (inv[:, t-1] > thr) if v == "hi_inv" else (inv[:, t-1] <= thr)
                mask_ = e_m & v_m
                if mask_.any():
                    mask[mask_, t] = True
            out[f"{e}_{v}"] = ret[mask].mean() * 100
    return out

# quad_view(signal_z, capex_growth[:,None]*np.ones((1,M)), future_ret, M)
```

你会发现「高投资 + 低效率」象限长期跑输，「低投资 + 高效率」象限长期跑赢——这正是 Titman et al. 过度投资异象的截面画面。真正有区分度的不是「投不投」，而是「投得有没有效率」。

## 七、诚实拆穿四类真实陷阱

**1. 会计口径陷阱（最致命）**：ROIIC 的分子 ΔEBIT、分母 ΔInvestedCapital 都极度依赖会计口径。商誉减值、研发费用化/资本化、经营租赁入表（新租赁准则），都会让 ΔInvestedCapital 剧烈跳动，ROIIC 随之失真。真实落地必须统一口径、剔除一次性项目。

**2. 投资周期错配**：重资产公司（半导体、化工） CAPEX 到产能释放有 2-3 年滞后，用当期 ROIIC 评判会系统性低估「正在建设期」的好公司。解法是用**领先多期的 ROIIC**（用 t 期投入预测 t+k 期利润），但这又会引入预测噪声。

**3. 成长陷阱（Growth Trap）**：高效率如果已经被市场充分定价（高估值），因子多头买入的就是「贵的好公司」，未来溢价被估值消化。资本支出效率因子常与质量、成长高度相关，需做**估值中性化**才能拿到「没被定价的部分」。

**4. 做空约束**：低效率组里相当比例是僵尸企业、ST、流动性极差的壳，A 股融券可得性低、做空成本高。因子实测时**多头倾斜（只做多高效率）**往往比多空组合更稳健，但多头部分会被市场 beta 污染，需回到第五步检验。

---

**一句话总结**：资本支出效率因子赚的是「**钱有没有变成利润**」这道题被市场慢半拍定价的钱。它和「投资强度」正交，和「质量/成长」部分重叠，核心价值在于把「资本纪律」这个定性判断量化成可回测、可中性化、可检验的信号——但会计口径、投资周期、估值陷阱三座大山，任何一座没翻过去都是 bug 不是 feature。

*（全文为自洽合成演示，量级非真实市场数据；真实落地需接入 wind/聚源财务表、统一会计口径并做行业/市值/估值中性化。）*
