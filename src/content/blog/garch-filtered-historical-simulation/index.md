---
title: "GARCH 滤波历史模拟：把波动率标准化后再抽历史"
description: "普通历史模拟的 VaR 是一条僵硬的地板：市场进入高波动它浑然不觉，突破全挤在危机段。FHS（Barone-Adesi et al. 1999）分两步修复：GARCH 滤波把收益标准化成近似 iid 的残差（|r|² 自相关 0.21→-0.03），经验分位数保住肥尾（峰度 1.8 仍在），再乘明日波动率预报还原条件尺度。GARCH-t(6) 模拟实测：HS 的 95% VaR 突破 π11=0.179 独立性检验 p<0.0001 处决，GARCH-正态 99% VaR 突破 25 次被 Kupiec 拒绝（尾部太瘦），FHS 六项检验全过——滤波管动态、经验分位数管形状，各司其职。附完整 Python 与四类真实陷阱（中阶）。"
publishDate: '2026-07-27'
tags:
  - 量化交易
  - GARCH
  - 历史模拟
  - FHS
  - VaR
  - 波动率
  - 风险管理
  - Python
language: Chinese
difficulty: intermediate
---

历史模拟法（HS）是银行业算 VaR 的默认姿势：取过去 500 天收益，读 1% 分位数，完事。它的卖点是**零分布假设**——肥尾、偏度、跳跃全在历史数据里，不用建模。但它有个致命的结构缺陷：**窗口内所有日子权重相等，它算出的是无条件分位数**。市场昨天刚进入高波动状态？HS 不知道，它的 VaR 还是那条由过去 500 天平均决定的地板，要等突破一次次砸进窗口才慢慢下沉。结果就是突破在时间上扎堆——低波动期 VaR 太深浪费资本，高波动期太浅连续被击穿。

结论先放这：**滤波历史模拟（FHS，Barone-Adesi, Giannopoulos & Vosper 1999）用"GARCH 管动态、经验分位数管形状"的分工修复这个缺陷**。三步：用 GARCH 把收益标准化成近似 iid 的残差 → 取残差的经验分位数（不套正态，肥尾原样保留）→ 乘明日波动率预报还原条件尺度。GARCH-t(6) 模拟世界 1500 天滚动回测的判决很干净：**HS 的 95% VaR 突破后次日再突破的概率 17.9%（无条件应为 5%），[Christoffersen 独立性检验](/blog/christoffersen-independence-test/) p<0.0001 处决；GARCH-正态的 99% VaR 突破 25 次（期望 15），[Kupiec POF](/blog/kupiec-pof-test/) p=0.018 拒绝——动态对了但尾巴太瘦；FHS 两个层级、频率与独立性共六项检验全部通过**。

## 分工的逻辑：为什么两个"半吊子"拼起来是全的

先看两个纯策略各自的死法：

- **纯历史模拟**：形状对（肥尾在数据里），动态错（无条件）。死于突破扎堆。
- **纯 GARCH-正态参数法**：动态对（条件波动率），形状错（正态尾巴太瘦）。死于 99% 这种深尾层级系统性低估。

FHS 的观察是：这两个错误**正交**。金融收益的标准化残差 $z_t = r_t/\sigma_t$ 近似 iid——波动聚集几乎全部由 $\sigma_t$ 解释，剩下的 $z_t$ 虽然仍是肥尾的（GARCH 滤波消不掉条件分布的肥尾），但至少不再有时间依赖。于是：

$$\text{VaR}_{t+1}^\alpha = \hat\sigma_{t+1} \cdot \hat{q}_\alpha(z)$$

$\hat\sigma_{t+1}$ 来自 GARCH 预报（负责"今天风险有多大"），$\hat q_\alpha(z)$ 来自标准化残差的经验分位数（负责"尾巴长什么形状"）。**模型只被要求做它擅长的事**：GARCH 不需要猜对残差分布（QML 下波动率估计照样一致），经验分位数不需要处理时间依赖（残差已经近似 iid，独立抽样的前提成立了）。

## 实现：五十行核心代码

```python
import numpy as np
from scipy import stats
from scipy.optimize import minimize

def garch_negll(params, r):
    """GARCH(1,1) 高斯 QML 负对数似然"""
    omega, a, b = params
    if omega <= 0 or a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    sig2 = np.empty(len(r)); sig2[0] = np.var(r)
    for t in range(1, len(r)):
        sig2[t] = omega + a * r[t-1]**2 + b * sig2[t-1]
    return 0.5 * np.sum(np.log(sig2) + r**2 / sig2)

def fhs_var(window, alpha, params):
    """FHS 单日 VaR：滤波 -> 经验分位数 -> 还原尺度"""
    omega, a, b = params
    T = len(window)
    sig2 = np.empty(T + 1); sig2[0] = np.var(window)
    for t in range(1, T + 1):                 # 最后一格是 t+1 预报
        sig2[t] = omega + a * window[t-1]**2 + b * sig2[t-1]
    z = window / np.sqrt(sig2[:-1])           # 标准化残差
    return np.quantile(z, alpha) * np.sqrt(sig2[-1])

# 滚动回测：500 天窗口，每 25 天重估 GARCH 参数
for t in range(win, T):
    w = r[t-win:t]
    if (t - win) % 25 == 0:
        params = min(
            (minimize(garch_negll, x0, args=(w,), method="Nelder-Mead")
             for x0 in [(0.05,0.1,0.85), (0.1,0.15,0.75)]),
            key=lambda res: res.fun).x
    var_fhs[t] = fhs_var(w, 0.01, params)
```

两个工程细节值得强调：**QML 用高斯似然是故意的**——即使真实残差是 t 分布，高斯 QML 的波动率参数估计依然一致（Bollerslev-Wooldridge），而残差的肥尾交给经验分位数处理，不需要在似然里建模；**每 25 天重估参数**是速度与新鲜度的折中，GARCH 参数漂移很慢，逐日重估纯属浪费。

## 滤波前后：残差长什么样

在 GARCH(1,1)-t(6)（真实参数 ω=0.05, α=0.13, β=0.84，持续性 0.97）的模拟序列上，取 500 天窗口做滤波：

- **时间依赖**：原始收益的平方自相关 0.21（波动聚集显著），标准化残差降到 **-0.03**——滤波把可预测的部分榨干了
- **肥尾**：原始收益超额峰度 7.8，残差仍有 **1.8**——这不是滤波失败，恰恰是 FHS 的资产：t(6) 的理论超额峰度正是 3/(6-4)=1.5，残差把条件分布的真实肥尾保留下来了，如果此处套正态就回到 GARCH-正态的死法

![滤波前后对比](/images/garch-filtered-historical-simulation/fhs-filtering-demo.jpg)

右下角对数坐标直方图最能说明问题：残差的尾部依然明显厚于正态参考线。**GARCH 滤波消灭的是波动聚集，不是肥尾**——两者经常被混为一谈，但前者是时间维度的依赖，后者是截面分布的形状。

## 回测判决：六项检验

1500 个评估日，三个模型 × 两个层级，用 Kupiec POF（频率）+ Christoffersen（独立性）双卡：

| 模型 | 95% 突破(期望75) | POF | IND | 99% 突破(期望15) | POF | IND |
|---|---|---|---|---|---|---|
| HS-500 | 78 | p=0.72 ✓ | **p<0.0001 ✗** | 22 | p=0.089 ✓ | **p=0.003 ✗** |
| GARCH-正态 | 69 | p=0.47 ✓ | p=0.32 ✓ | **25** | **p=0.018 ✗** | p=0.43 ✓ |
| **FHS** | 76 | p=0.91 ✓ | p=0.55 ✓ | 18 | p=0.45 ✓ | p=0.21 ✓ |

两个死法清晰分离：**HS 频率完美但突破扎堆**——π01=0.045 vs π11=0.179，突破后次日再突破的概率是平时的 4 倍，正是"僵硬地板遇上波动聚集"的签名；**GARCH-正态独立性完美但 99% 层级频率超标**——动态跟上了，每一天都在系统性低估深尾。FHS 六项全绿。

![VaR 路径对比](/images/garch-filtered-historical-simulation/fhs-var-paths.jpg)

路径图更直观：HS 的 99% VaR 是一条阶梯状的地板（只有极端观测进出窗口才跳一格），FHS 跟着波动率呼吸。按真实波动率三分位分层看 95% VaR 突破率：HS 低/中/高波动段 2.0%/3.4%/**10.2%**，FHS 是 4.4%/5.2%/5.6%——**HS 把全部超额突破堆在高波动段，FHS 三段基本持平**。风控视角下这是天壤之别：突破扎堆意味着损失以连续大额的形式到来，正是资本最脆弱的时刻。

![回测记分牌](/images/garch-filtered-historical-simulation/fhs-scoreboard.jpg)

## FHS 的软肋

![分层突破率与尾部样本](/images/garch-filtered-historical-simulation/fhs-vol-buckets.jpg)

1. **深尾分位数由极少数残差决定**。500 天窗口的 1% 经验分位数本质上是第 5、6 个最小残差的插值——**约 5 个点决定了整个 99% VaR 的形状部分**。窗口里恰好没有极端日，深尾就被低估；想做 99.9% 层级，经验分位数直接没有数据支撑，得在残差上再套 EVT（[GPD 尾部拟合](/blog/evt-tail-risk-gpd/)是标准续接）。
2. **滤波质量上限是 GARCH 模型的正确性**。真实世界有杠杆效应（负收益推升波动更多）、长记忆、跳跃——GARCH(1,1) 滤不干净时残差仍带依赖，"iid 后再抽样"的前提被侵蚀。诊断方法：查残差平方的 Ljung-Box；升级路线：[GJR/门限 GARCH](/blog/threshold-garch-asymmetry/) 或 [FIGARCH](/blog/figarch-long-memory/)。
3. **多日 VaR 不能开平方根**。√t 规则在条件波动率框架下系统性错误（波动率会均值回归）。正确做法是从残差池里**带放回抽样多步路径**，让每步的 σ 按 GARCH 递推更新——这也是 FHS 原论文的主要贡献（为期权组合做多日情景），单日 VaR 只是它最简单的特例。
4. **危机切换的第一周依然裸奔**。FHS 对波动的反应速度受 GARCH 的 α+β 限制，突发跳升（如熔断日）当天预报仍来自昨天的信息集。这是所有条件模型的共同边界，不是 FHS 独有，但用户经常对"条件化"抱有它能预知跳跃的幻觉。

## 与前作的连接

这篇是 VaR 建模-回测系列的合拢点：[Kupiec POF](/blog/kupiec-pof-test/)、[Christoffersen](/blog/christoffersen-independence-test/)、[DQ 检验](/blog/dynamic-quantile-test-dq/)回答"怎么判死刑"，FHS 回答"怎么造一个判不死的模型"。它的分工哲学——**参数模型管条件动态、非参数方法管分布形状**——是量化里罕见的几乎免费的午餐：不比 HS 多任何分布假设，不比 GARCH-正态多任何计算负担，两边的死穴恰好互补。半参数方法在风险管理里长盛不衰，FHS 是最好的教科书案例。

## 参考文献

- Barone-Adesi, G., Giannopoulos, K., & Vosper, L. (1999). VaR without Correlations for Portfolios of Derivative Securities. *Journal of Futures Markets*, 19(5), 583-602.
- Hull, J., & White, A. (1998). Incorporating Volatility Updating into the Historical Simulation Method for Value-at-Risk. *Journal of Risk*, 1(1), 5-19.
- Bollerslev, T., & Wooldridge, J. M. (1992). Quasi-Maximum Likelihood Estimation and Inference in Dynamic Models with Time-Varying Covariances. *Econometric Reviews*, 11(2), 143-172.
- Kuester, K., Mittnik, S., & Paolella, M. S. (2006). Value-at-Risk Prediction: A Comparison of Alternative Strategies. *Journal of Financial Econometrics*, 4(1), 53-89.
- Pritsker, M. (2006). The Hidden Dangers of Historical Simulation. *Journal of Banking & Finance*, 30(2), 561-582.
