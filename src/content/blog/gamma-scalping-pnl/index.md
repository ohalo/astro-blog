---
title: "Gamma Scalping 损益拆解：对冲频率、已实现波动与期权时间价值的博弈"
description: "买入期权做多 gamma、动态 delta 对冲赚已实现波动率——这是期权做市与波动率交易的基本功。但这门生意的账本远比『低买高卖』复杂：期望收益由（已实现波动 − 隐含波动）决定，对冲频率完全不改变期望、只压缩方差，每天的 gamma 收益必须跑赢 theta 租金才不亏钱。本文用 Black-Scholes 逐步推导 gamma scalping 的 PnL 分解公式，再用 2000 条蒙特卡洛路径逐一验证三条核心结论，最后给出四类实盘陷阱：交易成本重新引入频率权衡、gamma 敞口随标的漂移而漂移、跳跃日的对冲失效、以及用错波动率算 delta 时 PnL 路径依赖的经典结果（中高阶）。"
publishDate: '2026-07-29'
tags:
  - 量化交易
  - Gamma Scalping
  - 期权交易
  - 波动率交易
  - Delta对冲
  - 期权希腊字母
  - Python
language: Chinese
difficulty: advanced
---

买一份平值跨式（straddle），然后每天把 delta 对冲回零——股价上涨你就卖出一点股票，下跌就买回一点。股价来回震荡,你就在不断地高卖低买。这就是 **gamma scalping**，期权做市商和波动率交易员的基本功。

听起来像永动机：只要股价动，就有钱赚。但天下没有免费的午餐，这门生意的完整账本是：

**Gamma scalping 的期望收益 = 已实现波动率与隐含波动率的差，对冲频率不改变期望收益、只改变收益的方差，而每一天你都在用 theta（时间价值损耗）支付『持有 gamma 的租金』。**

这三句话每一句都可以严格推导、也可以用蒙特卡洛精确验证。本文把它们全部跑一遍。

## 一、PnL 分解：从 Black-Scholes PDE 直接读出来

持有一份期权（价格 $V$），用股票把 delta 对冲到零。在一个小时间段 $\delta t$ 内，对冲组合的损益按泰勒展开：

$$
\delta \Pi = \underbrace{\frac{1}{2}\Gamma (\delta S)^2}_{\text{gamma 收益}} + \underbrace{\Theta\, \delta t}_{\text{theta 损耗}} + \text{高阶项}
$$

delta 项被对冲掉了，剩下的主角只有两个：**gamma 项**（股价平方变动带来的凸性收益，恒为正）和 **theta 项**（时间流逝的损耗，做多期权时恒为负）。

而 Black-Scholes PDE 本身告诉我们这两项之间的精确关系。在无风险利率为零时：

$$
\Theta = -\frac{1}{2}\Gamma S^2 \sigma_{\text{impl}}^2
$$

代回去：

$$
\delta \Pi = \frac{1}{2}\Gamma S^2\left[\left(\frac{\delta S}{S}\right)^2 - \sigma_{\text{impl}}^2\, \delta t\right]
$$

这个公式是整篇文章的核心。它说的是：**每一个对冲区间里，你赚的是『实际发生的平方收益率』超出『隐含波动率预算』的部分。** 把所有区间加总，总 PnL 约等于：

$$
\text{PnL} \approx \int_0^T \frac{1}{2}\Gamma S^2 \left(\sigma_{\text{real}}^2 - \sigma_{\text{impl}}^2\right) dt
$$

已实现波动率高于建仓时付的隐含波动率，你赚钱；低于，你亏钱。Gamma scalping 本质上是**做多已实现方差、以隐含方差为成本价**的交易。

## 二、蒙特卡洛验证一：PnL 由 σ_real − σ_impl 决定

搭一个干净的实验：以 20% 隐含波动率买入 30 天平值跨式，每日收盘 delta 对冲，让真实世界的波动率从 12% 一路扫到 32%，每档跑 800 条路径。

```python
import numpy as np
from scipy.stats import norm

def bs_delta(S, K, T, r, sigma, cp=1):
    if T <= 0:
        return float(cp * (S > K))
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1) if cp == 1 else norm.cdf(d1) - 1

def simulate_pnl(sigma_real, hedge_every, n_paths=800, seed=7):
    """买入跨式 + 定期 delta 对冲，返回每条路径的总 PnL"""
    g = np.random.default_rng(seed)
    S0, K, r = 100.0, 100.0, 0.0
    sigma_impl = 0.20
    T = 30/252
    n = 30 * 78                     # 5 分钟一个 bar
    dt = T / n
    pnls = np.zeros(n_paths)
    for p in range(n_paths):
        S, tau = S0, T
        prem = bs_price(S,K,T,r,sigma_impl,1) + bs_price(S,K,T,r,sigma_impl,-1)
        delta = bs_delta(S,K,tau,r,sigma_impl,1) + bs_delta(S,K,tau,r,sigma_impl,-1)
        stock_pos = -delta          # 对冲仓位
        cash = -prem - stock_pos*S
        z = g.standard_normal(n)
        for i in range(n):
            S = S*np.exp(-0.5*sigma_real**2*dt + sigma_real*np.sqrt(dt)*z[i])
            tau -= dt
            if (i+1) % hedge_every == 0 and tau > 0:
                new_d = bs_delta(S,K,tau,r,sigma_impl,1) + bs_delta(S,K,tau,r,sigma_impl,-1)
                cash -= (-new_d - stock_pos)*S
                stock_pos = -new_d
        payoff = max(S-K,0) + max(K-S,0)
        pnls[p] = cash + stock_pos*S + payoff
    return pnls
```

注意一个容易被忽略的细节：**对冲 delta 始终用建仓隐含波动率 20% 计算**，而不是用真实波动率——因为实盘中你不知道真实波动率是多少，你只有市场报价。这个选择对结果有微妙影响，第五节会回来讨论。

![Gamma Scalping PnL vs 已实现波动率](/images/gamma-scalping-pnl/pnl-vs-realized-vol.png)

结果与理论严丝合缝：

- **PnL 曲线在 σ_real = 20%（建仓隐波）处精确穿越零点**。已实现波动等于付出的隐含波动，期望不赚不亏
- σ_real 每高一档，平均 PnL 单调上移；低一档则单调下移——**这不是方向性交易，是纯粹的波动率多头**
- 阴影带（±1σ 路径离散）提醒你：即使期望为正，单条路径照样可能亏钱。这就引出了第二个问题——对冲频率

## 三、蒙特卡洛验证二：对冲频率不改变期望，只压缩方差

一个流传很广的误解是"对冲越勤快，scalp 到的钱越多"。理论上这是错的：gamma 收益来自 $(\delta S)^2$ 的累积，而布朗运动的平方变差与切分粒度无关——切得再细，一段时间内的总平方变差期望不变。

**对冲频率改变的不是你赚多少钱的期望，而是这笔钱到账的确定性。**

固定 σ_real = 25%（高于隐波 20%，期望为正），只改变对冲频率，从每 5 分钟到每 3 天：

![对冲频率的作用](/images/gamma-scalping-pnl/hedge-frequency-effect.png)

左图：五种频率的平均 PnL 几乎是同一个数——**期望与频率无关**得到干净验证。右图：PnL 标准差随频率降低单调上升，每 3 天对冲一次的标准差是每 5 分钟对冲的数倍。

分布视角看得更直观：

![PnL分布对比](/images/gamma-scalping-pnl/pnl-distribution-hedge-freq.png)

两个分布中心几乎重合，但高频对冲（蓝）又高又瘦，低频对冲（红）又矮又胖、左尾深入亏损区。**同样的期望收益，完全不同的持有体验**——低频对冲者本质上在 gamma 交易之上又叠加了一层"未对冲 delta 的方向性赌博"，这层赌博期望为零，但方差是真的。

直觉解释：对冲不及时，delta 会随股价移动而积累，组合暂时变成方向性头寸。运气好方向对了多赚，运气差多亏，平均下来两不相欠——但你的资金曲线记得每一次心跳。

## 四、每一天的博弈：gamma 收益 vs theta 租金

把镜头从 30 天拉近到单日。持有跨式的每一天，你都面对同一道算术题：

$$
\text{当日净PnL} \approx \frac{1}{2}\Gamma(\Delta S)^2 + \Theta_{\text{day}}
$$

Theta 是确定支出（时间一定会流逝），gamma 收益取决于当天股价实际走了多远。令两者相等，解出**盈亏平衡移动幅度**：

$$
|\Delta S|_{\text{breakeven}} = \sqrt{\frac{-2\Theta_{\text{day}}}{\Gamma}} = S\,\sigma_{\text{impl}}\sqrt{\delta t} \approx S \times \frac{\sigma_{\text{impl}}}{\sqrt{252}}
$$

最后一步化简揭示了一个非常优雅的事实：**盈亏平衡幅度恰好等于隐含波动率的日化值**。买了 20% 隐波的期权，股价每天要平均移动约 20%/√252 ≈ 1.26% 你才回本——一天不动，theta 白付。

![Gamma vs Theta 盈亏平衡](/images/gamma-scalping-pnl/gamma-vs-theta-breakeven.png)

图中绿色抛物线是 gamma 收益，红色虚线是当日 theta 支出，蓝色曲线是净 PnL，与零轴的两个交点即盈亏平衡点（±1.26 元附近，与理论值一致）。

这也解释了为什么波动率交易员每天开盘想的第一件事是"今天隐波定价的 breakeven 是多少个点"——这就是当天做多 gamma 的门票价格。

## 五、被隐藏的深水区：用哪个波动率算 delta

第二节代码里我们用**隐含波动率**算对冲 delta。如果改用（假设已知的）**真实波动率**呢？这是 Carr、Ahmad-Wilmott 等人反复讨论过的经典问题，结论出人意料：

- **用真实波动率对冲**：总 PnL 从第一天起就锁定为两个理论价值之差 $V(\sigma_{\text{real}}) - V(\sigma_{\text{impl}})$，确定无疑——但中途的资金曲线剧烈波动（mark-to-market 噪声大）
- **用隐含波动率对冲**：每天的 PnL 平滑入账（正是第一节的逐段公式），但**总 PnL 变成路径依赖**——同样的 σ_real，股价路径不同，最终赚到的钱不同。gamma 高的区域（贴近行权价）恰好赶上大波动，就多赚；波动发生在 gamma 低的远处，就少赚

我们模拟中的路径离散（第一张图的阴影带）有一部分正来自这种路径依赖，而不纯粹是采样噪声。实盘选择通常是用隐波对冲——不是因为它更优，而是因为真实波动率不可观测，且平滑的日度 PnL 对风控和心态都更友好。

## 六、把交易成本放回来：频率的最优解

第三节说"对冲越频繁方差越小"，那是零成本世界。每笔对冲要付出买卖价差和冲击成本时，账本变成：

- 对冲太频繁 → 方差小，但成本吃掉期望收益（每次调仓都在过路费上放血）
- 对冲太稀疏 → 省成本，但方差大、且极端路径可能爆掉风控

经典的 Whalley-Wilmott 渐进解给出了定性答案：**最优策略不是定时对冲，而是设一个 delta 容忍带**——delta 漂出带外才动手，带宽正比于 $(\text{成本}/\Gamma)^{1/3}$。gamma 越大（临近到期的平值期权），容忍带越窄、动手越勤；成本越高，带越宽。这就是实务中"delta band rebalancing"的理论根基。

## A. 实现细节

- 标的路径用几何布朗运动生成，5 分钟一个 bar（每日 78 个），信号与对冲同 bar 执行（对冲是对自身持仓的调整，无 look-ahead 问题）
- 期权头寸为 1 份平值跨式（call + put），到期 30 个交易日，无风险利率设 0 以隔离 gamma/theta 效应
- 对冲 delta 全程用建仓隐含波动率 20% 计算，不使用路径的真实波动率——与实盘信息集一致
- 到期结算用内在价值，总 PnL = 现金账户 + 股票市值 + 期权 payoff，现金账户逐笔记录对冲交易
- 每个参数组合 800 条独立路径，图中误差棒为均值 ±2 标准误

## B. 已知偏差

- **无交易成本**：所有对冲零成本成交，这使"高频对冲严格更优"成立；真实世界中第六节的容忍带逻辑才是对的
- **GBM 假设无跳跃**：连续路径下 delta 对冲可以任意精细；真实市场的隔夜跳空使 gamma 收益以离散大块到账、对冲永远慢一步，跳跃日的实际 PnL 分布比模拟的更肥尾
- **恒定波动率路径**：真实的已实现波动率本身随时间波动且与股价负相关（leverage effect），会通过 vanna/volga 项影响 PnL，本文的框架把这些高阶希腊字母全部冻结了

## C. 结果解读

- **Gamma scalping 是方差互换的"手工版"**：期望收益由 σ²_real − σ²_impl 决定，与方差互换的 payoff 同构。区别在于方差互换的 gamma 敞口（按 1/S² 加权后）与股价无关，而单一期权的 gamma 集中在行权价附近——股价漂离行权价，你的"波动率工厂"产能就衰减。这就是为什么专业波动率簿要不断 roll 行权价、或直接用期权条带
- **对冲频率是风控参数而非收益参数**：模拟明确显示五种频率的期望 PnL 相同、标准差差数倍。选频率时问的正确问题不是"怎样赚更多"而是"我能忍受多大的日内 delta 敞口、愿意付多少过路费来消灭它"
- **每天 1.26% 的 breakeven 是这门生意的日租金**：20% 隐波下标的日均移动必须超过 1.26% 才覆盖 theta。低波动的横盘市里做多 gamma 是慢性失血——这正是"卖 gamma 者在平静期天天收租、崩盘日一夜还清"的镜像
- **用隐波对冲的 PnL 路径依赖不是 bug 而是常态**：即使方向看对了（σ_real > σ_impl），大波动若发生在股价远离行权价、gamma 稀薄的区域，实际落袋会显著低于理论值。评估一段 gamma scalping 业绩时，"赚了多少"必须对照"波动发生时我有多少 gamma 在场"
- **适用边界**：本框架适合流动性好、可连续对冲的标的（指数期权、大盘股期权）。对 A 股这类无个股期权、涨跌停限制路径连续性的市场，只有 50ETF/300ETF/500ETF 等期权品种可以落地，且 T+0 的期权对 T+1 的现货对冲还会引入额外错配
