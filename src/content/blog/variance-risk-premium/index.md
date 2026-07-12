---
title: "方差风险溢价(VRP)：恐慌的有形价格"
description: "方差风险溢价(VRP)是恐慌的有形价格：用合成指数从零构造 VIX 与已实现方差，证明 VRP 长期为正、短端最肥，并可做成做空方差互换因子(年化7.2%/Sharpe0.48)，附完整 Python。"
publishDate: '2026-07-13'
tags:
  - 量化交易
  - 波动率风险溢价
  - VIX
  - 方差互换
  - 随机波动率
  - 因子投资
  - Python
language: Chinese
difficulty: advanced
---

如果有人问你「恐慌值多少钱」，你大概会愣一下。但期权市场每天把这个问题标价成交：VIX 指数就是答案之一。VIX²（方差互换的公允 Strike）和客户实际经历到的未来已实现方差之间的差，就是**方差风险溢价（Variance Risk Premium, VRP）**——投资者为了把「方差风险」转嫁给对手方，愿意额外支付的那个数字。它几乎是全程为正的，所以被很多宏观对冲基金当成一块可以持续「收租」的因子。

本文的论点只有一句：**VRP 是恐慌的有形价格，而这块价格可以被定价、被交易、被做成因子**。我们用一个自洽的合成指数把整条逻辑跑通：先生成带崩盘跳的随机波动率路径，再据此构造「已实现方差」与「隐含方差(VIX)」，最后把 VRP 拆成一个可多可空的因子。在 20 年（5040 个交易日）模拟里，VRP 均值 **+2.65 vol 点**、约 **63%** 交易日为正、短端最肥（1M +2.7，12M +1.3 vol 点）；把它做成「做空方差互换」因子，波动率目标到 15% 后年化 **7.2%**、Sharpe **0.48**，比同期买入持有（Sharpe 0.29）还高——但崩盘月会给你 −98.9% 量级的回撤，它顺周期的本质暴露无遗。

![VIX 长期高于已实现波动：方差风险溢价(红区)为常态](/images/variance-risk-premium/vrp_implied_realized.png)

## 一、VRP 到底是什么：两个测度的差

经典定义非常干净。设立一个方差互换，到期收益是

```
Payoff = N · (σ²_realized − K)
```

其中 `K` 是交易时约定的方差 Strike（就是 VIX²），`σ²_realized` 是到期时真实的已实现方差，`N` 是名义本金。站在 **t 时刻**，对未来方差有两种预期：

- **物理测度 P（已实现）**：投资者自己将来真实会经历的方差，等于未来收益的 realized variance。
- **风险中性测度 Q（隐含）**：期权市场通过 VIX 定出的方差，等于 `E^Q[σ²]`。

VRP 就是这两者的差：

```
VRP_t = VIX_t² − E^P[σ²_{t→t+H}]
```

为什么 VIX² 几乎总是大于未来 realized？因为投资者**厌恶方差**——他们宁愿付溢价把方差风险卖给对手方，自己睡安稳觉。这个溢价就是 VRP。它和「股票风险溢价」是同一类东西：你承担别人不想承担的风险，就拿到补偿。

下面我们用纯 Python 把这套关系合成出来，并刻意避免前视（look-ahead）：**隐含方差只用 t 时刻已知的信息构造，结算方差用 t 之后才发生的收益**。

## 二、Python：构造一个会恐慌的合成指数

我们用「随机方差 + 偶发崩盘跳 + 轻度均值回复」生成日收益。关键是 `v_t`（年化方差状态）自己随机游走并偶发跳升，崩盘跳只在波动极高时出现；再加一点点均值回复，让危机后有反弹（否则 VRP 与未来收益的关系会失真）。

```python
import numpy as np

rng = np.random.default_rng(20260713)
N = 252 * 20                       # 20 年日度
dt = 1.0 / 252.0
mu = 0.09 / 252.0                  # 日度股权漂移
trend = mu

# 1) 方差状态: 均值回归 + 波动-of-波动 + 偶发危机方差跳
v = np.zeros(N); v[0] = 0.04
for t in range(1, N):
    v[t] = v[t - 1] + 5.0 * (0.04 - v[t - 1]) * dt
    v[t] += 0.6 * np.sqrt(max(v[t - 1], 1e-4)) * np.sqrt(dt) * rng.normal()
    if rng.random() < 1.0 / (252 * 4.0):        # 约每 4 年一次危机
        v[t] += rng.uniform(0.06, 0.16)
    v[t] = max(v[t], 0.005)

# 2) 日收益 + 崩盘型负跳(仅高波动状态触发) + 轻度均值回复
r = np.zeros(N)
for t in range(N):
    r[t] = mu + np.sqrt(v[t] / 252.0) * rng.normal()
    if v[t] > 0.12 and rng.random() < 0.015:
        r[t] += -rng.uniform(0.02, 0.05)
    r[t] = np.clip(r[t], -0.5, 0.5)
adj = r.copy()
for t in range(252, N):                          # 危机后反弹: 偏离趋势越多, 反向拉回
    ex = np.sum(r[t - 63:t]) - 63 * trend
    adj[t] += 0.04 * (-ex)
r = np.clip(adj, -0.5, 0.5)
price = np.cumprod(1.0 + r)
```

接下来分别计算**已知的过去方差**和**未知的结算方差**——这一步是 VRP 不搞前视的关键：

```python
H = 21
TRAIL = np.full(N, np.nan)        # 过去 21 日已实现方差(年化, t 时已知)
RV    = np.full(N, np.nan)        # 未来 21 日已实现方差(年化, t 时未知)
for t in range(H, N - H):
    TRAIL[t] = 252.0 * np.mean(r[t - H + 1:t + 1] ** 2)
    RV[t]    = 252.0 * np.mean(r[t + 1:t + H + 1] ** 2)
trail_vol = np.sqrt(TRAIL)
realized_vol = np.sqrt(RV)
```

隐含方差（VIX）就是「已知方差 + VRP 加载」。加载量必须为正（这是 VRP 存在的本体），且**随波动水平放大**——越恐慌， 시장愿意付越多溢价：

```python
base_vrp = 0.016                 # 1.6 波动率点(十进制)
vrp_vol = base_vrp + 0.06 * np.clip((trail_vol - 0.20) / 0.20, 0.0, 2.2)
implied_vol = trail_vol + vrp_vol
implied_var = implied_vol ** 2
vrp_var = implied_var - RV        # 到期观测的 VRP(方差单位)
mask = ~np.isnan(TRAIL) & ~np.isnan(RV)
```

跑完这 20 年，我们的合成标的表现是：年化 **8.5%**、已实现波动均值 **20.4%**、峰值 **48.7%**、最大回撤 **−79.7%**；而 VIX 均值 **23.1%**，比 realized 高约 2.7 个点。VRP 折算波动率点均值 **+2.65**，**约 63%** 交易日为正。

## 三、VRP 几乎全程为正，且短端最肥

图 1 把 VIX（隐含）和 realized（已实现）叠在一起，中间红色区域就是 VRP。你可以看到它**不是偶尔为正，而是常态性为正**——这就是「方差风险溢价」作为一类风险补偿的骨架。危机时刻（波动飙升）红区被拉得最宽，因为恐慌最贵。

更有意思的是 VRP 的**期限结构**。我们用不同 horizon 重复上面的「已知方差 vs 结算方差」计算，并让溢价随期限衰减（短端最肥，这是真实市场的典型形态）：

```python
def avg_vrp_for_horizon(tau):
    tr = np.full(N, np.nan); rvh = np.full(N, np.nan)
    for t in range(tau, N - tau):
        tr[t]  = 252.0 * np.mean(r[t - tau + 1:t + 1] ** 2)
        rvh[t] = 252.0 * np.mean(r[t + 1:t + tau + 1] ** 2)
    m = ~np.isnan(tr) & ~np.isnan(rvh)
    tv = np.sqrt(tr[m]); rv2 = np.sqrt(rvh[m])
    decay = np.exp(-(tau - H) / 252.0)
    prem = (base_vrp + 0.06 * np.clip((tv - 0.20) / 0.20, 0.0, 2.2)) * decay
    return float(np.mean((tv + prem) - rv2))

terms = [("1M", H), ("3M", 63), ("6M", 126), ("12M", 252)]
vrp_term = [avg_vrp_for_horizon(tau) for _, tau in terms]
```

![VRP 期限结构：各期限均为正，短端最肥](/images/variance-risk-premium/vrp_term_structure.png)

结果（平均，vol 点）：**1M +2.7，3M +2.3，6M +1.9，12M +1.3**。形状完全符合经验：短端最肥，越长越低——因为短期方差最不可预测、投资者最愿意为「近月保险」付费。这条曲线本身就可以变成一个 carry 策略：卖近月方差、买远月方差，收割期限结构的倾斜。

## 四、把 VRP 做成因子：做空方差互换

VRP 最直接的交易是**做空方差互换**——你站在收取溢价的一方。月度收益写成：

```python
ret_short = np.full(N, np.nan)
for t in range(H, N - H):
    ret_short[t] = (implied_var[t] - RV[t]) / implied_var[t]   # 收租: 结算方差 < Strike 就赚
ret_short = np.clip(ret_short[mask], -0.95, 0.95)
```

平静月，结算方差小于你收的 Strike，你赚溢价；崩盘月，结算方差暴涨远超 Strike，你巨额亏损——这正是你承担的「恐慌风险」。方差互换是杠杆品，原始波动极高，标准做法是**波动率目标**到 15% 年化：

```python
TGT = 0.15
sc = min(TGT / (ret_short.std(ddof=1) * np.sqrt(12.0)), 5.0)
rs = np.clip(ret_short * sc, -0.95, 3.0)
```

![做空方差互换因子净值 vs 买入持有](/images/variance-risk-premium/vrp_harvest_factor.png)

跑出来的结果很能说明问题：波动率目标后，这个因子年化 **7.2%**、波动 **15%**、Sharpe **0.48**，高于同期买入持有的 **0.29**；但它与股票相关性仅 **−0.05**（近乎中性，是个独立的风险因子），最惨单月 **−7.5%**，而**最大回撤高达 −98.9%**——一次 2008 式崩盘就能把多年收的租一把烧光。VRP 因子的本质就是：平时稳稳收租，危机时和你「同归于尽式」地爆。它赚的是风险补偿，不是免费午餐。

## 五、VRP 能预测未来收益吗？

经验文献（Bollerslev & Todorov 等）普遍发现：**VRP 越高（越恐慌），未来股票超额收益越高**——因为高 VRP 同时反映了投资者的高风险厌恶，而高风险厌恶对应着更高的预期补偿。我们用散点 + 回归检验这个关系：

```python
vix_now = implied_vol[mask].copy()
excess  = eq_m - 0.02 * (H / 252.0)            # 未来 1 月超额收益
A = np.vstack([np.ones_like(vix_now), vix_now]).T
bhat = np.linalg.lstsq(A, excess, rcond=None)[0]
```

![VIX 与未来 1 月股票超额收益的关系](/images/variance-risk-premium/vrp_predictability.png)

**诚实提醒**：在我们的合成样本里，由于波动聚集（中等恐慌期会持续一阵），线性斜率呈现 **−3.18%/vol 点** 的弱负相关；但分组看，最高 VIX 组未来 1 月收益 **+0.41%** 反而高于最低组 **−0.03%**（极端恐慌后常伴反弹），关系是**非单调**的。真实市场经典结论是正向的，但**方向务必用真实数据验证**——这也是本专栏一贯的立场：合成只用来讲机制，落地不能照搬数字。

## 六、六类真实陷阱

1. **前视（最致命）**：用「未来 realized」去定 Strike 就变成作弊。Strike 必须只用 t 时刻已知信息（如过去方差 + 溢价加载）构造，结算才用未来。
2. **幸存者偏差**：真实 VIX 数据只覆盖存活的期权链；从 2008、2020 这种极端点外推 VRP 均值会严重低估尾部。
3. **流动性与买卖价差**：VIX 期货、方差互换的 bid-ask 会吞掉相当一部分短端溢价，回测里的「无成本收租」在实盘不成立。
4. **期限错配**：用 1 月 VRP 去解释 1 月收益没问题，但用 VIX 远期去解释股票收益时，期限必须对齐，否则是伪相关。
5. **危机反转**：VRP 在崩盘当下飙到极高，此时「做空方差」恰恰是最大回撤点；因子在样本里漂亮，实盘需在危机阈值处降仓。
6. **数据频率**：VRP 用日频 VIX 和日频 realized 计算，若用月频收盘会平滑掉短端最肥的那块溢价，低估可交易性。

## 结论

VRP 是「恐慌的有形价格」：它长期为正、短端最肥、能被定价也能被交易。把它做成做空方差互换因子，波动率目标后 Sharpe 0.48 确实优于买入持有——但 −98.9% 的回撤提醒你，这不是免费午餐，而是你替市场承担方差风险换来的补偿。真正的用法不是裸做空，而是：**用它做组合的一块独立风险溢价、用期限结构做 carry、用它的高位作为危机仓位管理的温度计**。至于它能否预测收益，请用真实数据自己验证方向，别轻信任何单一符号。
