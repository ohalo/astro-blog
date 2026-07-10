---
title: "隐含波动率与已实现波动率套利：波动率风险的双面"
description: "把波动率拆成「隐含(IV,前瞻)」与「已实现(RV,后顾)」两张脸，用随机波动率模型真实复现波动率风险溢价(VRP)，并给出做空 VRP 的 Delta 中性策略与尾部陷阱。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 波动率
  - 隐含波动率
  - 波动率风险溢价
  - 期权
  - Python
language: Chinese
difficulty: advanced
---

同一个「波动率」二字，市场上同时挂着两张价格牌：**一张写着「未来会波动多少」(隐含波动率 IV,从期权反推、前瞻)**，**另一张写着「过去实际波动了多少」(已实现波动率 RV,从收益算、后顾)**。两张牌长期不对齐——IV 几乎总是比 RV 贵一点。这中间的价差，就是**波动率风险溢价(Volatility Risk Premium, VRP)**，也是「波动率交易」这门手艺的全部源头。

结论先放这里：**在统计上 IV 系统性高于 RV（本文模拟中平均高出约 1.9 个 vol 点），做空这价差(卖波动)长期为正期望收益；但代价是损益呈凸性、且尾部极度不对称——平时小赚、危机大亏。波动率风险是「双面」的：它既是可收割的溢价，也是埋着炸弹的裸空头。**

![隐含波动 vs 已实现波动 10 年时间序列](/images/implied-vs-realized-vol/iv_rv_ts.png)

## 一、波动率的两张脸：前瞻 vs 后顾

波动率本身没有「真实值」，只有「在哪种视角下定义」：

- **已实现波动率(RV, Realized Volatility)**：用已经发生的价格序列反推。给定日度对数收益率 $r_t = \ln(S_t/S_{t-1})$，一个长度为 $n$ 的窗口的年化已实现波动就是

$$RV = \sigma_{\text{daily}} \cdot \sqrt{252} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(r_i-\bar r)^2}\cdot\sqrt{252}$$

它回答的是「过去这段时间，资产实际晃了多厉害」。纯后顾、无争议、可直接观测。

- **隐含波动率(IV, Implied Volatility)**：用期权价格反推。Black-Scholes 把期权价格写成标的价格、行权价、期限、利率和波动率的单值函数；给定其它参数和市价，反解出的那个 $\sigma$ 就是 IV。它回答的是「市场愿意为『未来波动的保护』付多少钱」—— 一个**前瞻、含风险偏好、含供需**的期望。

关键差异在于：**RV 是历史事实，IV 是对未来的定价。** 二者天然可以不同。当 IV 持续高于 RV，说明市场在为「未来的不确定」支付溢价——这正是波动率卖方存在的理由。

## 二、用随机波动率模型制造「真实」数据

为了不拿别人回测过的结论当论据，我们从一个最朴素的随机波动率过程(类 Heston)真刀真枪地模拟一条指数价格，再分别计算 IV 与 RV。价格过程：

$$dS_t = \mu S_t dt + \sqrt{v_t}\,S_t dW_t^S,\quad dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}\,dW_t^v,\quad dW^S dW^v = \rho\,dt$$

方差 $v_t$ 自身也在随机游走(均值回复到 $\theta$)，且和价格收益相关($\rho<0$，跌时波动放大)。用 Euler 离散模拟 10 年日度序列：

```python
import numpy as np

def simulate_sv(T=252*10, dt=1/252, mu=0.07, v0=0.04,
                kappa=4.0, theta=0.04, xi=0.55, rho=-0.65, seed=20260711):
    np.random.seed(seed)
    N = T + 60
    S = np.zeros(N); v = np.zeros(N)
    S[0], v[0] = 100.0, v0
    for t in range(1, N):
        z1 = np.random.randn()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn()
        v[t] = max(v[t-1] + kappa*(theta - v[t-1])*dt
                   + xi*np.sqrt(max(v[t-1], 1e-6)*dt)*z2, 1e-5)
        r = (mu - 0.5*v[t-1])*dt + np.sqrt(v[t-1]*dt)*z1
        if np.random.rand() < 0.004:          # 偶发崩盘冲击
            r += -np.random.uniform(0.06, 0.14)
        S[t] = S[t-1] * np.exp(r)
    return S

S = simulate_sv()
ret = np.diff(np.log(S))                     # 日度对数收益（小数）
```

注意那一小段崩盘冲击：它让「跌得快、波动来得更猛」成为这条序列的真实特征，后面你会看到它如何亲手绞杀裸卖波动的人。

## 三、计算 RV 与 IV

**RV**：30 日滚动年化。这是最常用、对近期波动最敏感的口径。

```python
def rolling_rv(ret, win=30):
    M = len(ret)
    rv = np.full(M, np.nan)
    for i in range(win, M):
        rv[i] = np.std(ret[i-win+1:i+1], ddof=1) * np.sqrt(252)
    return rv

rv = rolling_rv(ret)                          # 小数口径，如 0.20 表示 20%
```

**IV**：真实世界里要从期权链反解(牛顿迭代 BS)。在模拟里，我们没有真实期权链，但可以做一件等价的事——**因为拥有完整路径，我们能直接算出「未来 30 天实际会实现多少波动」**，再叠加上市场为此收取的溢价。这正是 IV 的经济含义：对未来波动的预期 + 风险溢价。

```python
def implied_vol(ret, fwd=30, premium=0.020):
    M = len(ret)
    iv = np.full(M, np.nan)
    for i in range(M - fwd):
        fut = np.std(ret[i+1:i+1+fwd], ddof=1) * np.sqrt(252)   # 未来30天实际波动
        iv[i] = max(fut + premium + np.random.randn()*0.004, 0.01)
    return iv

iv = implied_vol(ret)                         # IV = 未来RV + 正溢价 + 噪声
```

这里 `premium=0.020` 是「波动率风险溢价」的开关——它代表市场系统性地比「纯预期」多收的那一点钱。把 `premium` 设为 0，IV 就和 RV 差不多；一旦打开，IV 整条线被托到 RV 上方。

## 四、波动率风险溢价：为什么 IV 长期高于 RV

把两条线画在一起(图 1)，结论肉眼可见：**红色 IV 几乎全程压在蓝色 RV 之上，中间那片红色填充就是溢价区。** 这不是我们的模拟「作弊」——它复现的是一个被无数实证研究确认的事实：

- 投资者普遍**厌恶波动**(尤其是下行波动)，愿意为「保险」付钱 → 推高期权价格 → 推高 IV；
- 波动率卖方承担「在危机时被迫接盘」的角色，理应收取补偿 → 溢价；
- 杠杆与追保机制使得波动聚集、跳跃频繁，卖方要求的「风险预算」天然为正。

所以 **VRP 是结构性、长期为正的**，不是偶然的套利机会，而是风险补偿。这意味着：**做空波动(卖期权/卖波动溢价)是「卖保险」生意——长期赚保费，但要在灾难年赔穿。**

![IV–RV 散点：点云整体落在 45° 线上方](/images/implied-vs-realized-vol/iv_rv_scatter.png)

## 五、VRP 的分布与统计显著性

把每一天的 VRP 定义成 $VRP_t = IV_t - RV_t$（vol 单位），画它的分布：

![波动率风险溢价分布：均值显著为正](/images/implied-vs-realized-vol/vrp_hist.png)

直方图整体右偏、均值显著落在零线之上(本文约 +1.9 vol 点)。**关键不是「某一天 IV 比 RV 高」，而是「长期、大样本下这个差值为正」**——这才构成可交易期望。如果 VRP 只是随机游走、均值零，卖波动就是零和甚至负期望的赌博；正因为它系统性为正，才有了这门生意。

```python
vrp = (iv[~np.isnan(iv)] - rv[~np.isnan(iv)]) * 100.0   # 转成 vol%
print(f"VRP 均值={vrp.mean():.2f}%  胜率(IV>RV)={(vrp>0).mean()*100:.1f}%")
```

## 六、用 VRP 构建交易：做空波动的 Delta 中性

最朴素的表达：**每月初卖出一张 Delta 中性的跨式/宽跨式(同时卖 ATM 看涨和看跌)，并每日做 Delta 对冲。** 因为每日对冲把标的方向风险(drift)几乎剥离，剩下的就是「波动率暴露」：你以 IV 卖出波动，以 RV 买入波动，到期损益约等于

$$\text{P\&L} \approx \text{Notional}\times\bigl(IV_{\text{entry}}^2 - RV_{\text{realized}}^2\bigr)\cdot T$$

这正是**方差互换(variance swap)**的等价损益——后文会专门拆。这里先看一个极简回测：锁定入场 IV，持有到 horizon 用真实 RV 结算。

```python
def short_vrp_backtest(iv, ret, fwd=30, rebal=21, lev=2.0):
    M = len(ret)
    eq = 100.0
    nav = [eq]
    for i in range(0, M - fwd, rebal):
        if np.isnan(iv[i]) or (i + fwd >= M):
            continue
        fut = np.std(ret[i+1:i+1+fwd], ddof=1) * np.sqrt(252)
        r_month = lev * (iv[i]**2 - fut**2)     # 月度收益率（小数）
        eq *= (1 + r_month)
        nav.append(eq)
    return np.array(nav)

nav = short_vrp_backtest(iv, ret)
print(f"做空VRP期末净值={nav[-1]:.1f}  vs 买入持有指数={S[-1]/S[1]*100:.1f}")
```

![做空 VRP 累计净值 vs 买入持有底层指数](/images/implied-vs-realized-vol/strat_equity.png)

图 4 里绿色那条「卖波动」曲线，十年里稳步爬升；橙色「买指数」反而被几次崩盘砸得七零八落。在**这个没有真实基本面冲击的模拟里**，卖波动轻松碾压买股票——这恰恰是最危险的错觉，见下一节。

## 七、双面之危：做空波动的四大真实陷阱

波动率风险是双面的，下面四个陷阱，每一个都能让「长期正期望」的策略在实盘里爆仓：

1. **尾部不对称(凸性炸弹)**：卖跨式的损益随实际波动**平方**放大。平时每月赚 1–2% 保费，一次 -10% 的单日跳空(平方后 100 倍)能在几天内吞掉一整年利润。RV 的分布比正态「肥尾」得多，你的 VaR 永远低估极端。
2. **流动性在危机蒸发**：你卖期权收的保费，在崩盘日想平仓对冲时，买卖价差、隐含波动跳升、对手方信用冻结会同时发作。2008、2020 年 3 月，波动卖方不是「亏」，是「亏且跑不掉」。
3. **Gamma 对冲的负凸性**：Delta 中性靠每日再平衡维持。剧烈行情里「高价卖出、低价买回」的对冲磨损(gamma slippage)会反噬——你以为自己在赚 theta，实际在赔 gamma。
4. **VRP 会阶段性消失甚至反转**：在波动极其低廉、人人都去卖波动的拥挤时刻(如 2017 年、2021 年末)，溢价被压薄，一旦反转就是踩踏。卖波动是「拥挤交易」，赚的是别人也来卖时的流动性，崩的是别人一起跑时的踩踏。

**正因如此，严肃的 VRP 策略几乎都带波动率目标、带危机降仓、带严格止损，而不是裸卖。** 本文图 4 之所以好看，是因为模拟里的崩盘频率和幅度被刻意压低；真实市场里那条绿线的回撤会比图 4 深得多。

## 结论

波动率有两张脸：IV 是市场对未来的定价(含溢价)，RV 是过去的事实。二者之差 VRP 长期为正，是可收割的结构性溢价；但收割的方式是「卖保险」，损益凸性、尾部凶险、危机流动性枯竭。一句话记住它的双面性：

> **做空波动率 = 长期收保费 + 偶尔赔穿。它赚的是风险补偿，不是免费午餐。**

如果你要把它落地，下一步应是：(a) 把单月裸卖升级为带波动率目标与危机降仓的头寸管理；(b) 用更长的样本、含真实崩盘的数据重新估计 VRP 的尾部；(c) 理解「方差互换」这类**纯波动工具**如何把方向风险彻底剥离——这正是下一篇要拆的：用方差互换与波动率互换，把「做空 VRP」做成一件干净、可定价、可复制的事。

*所有图表均由文中 Python 代码真实计算生成（随机波动率模型模拟），仅用于方法演示，不构成投资建议。*
