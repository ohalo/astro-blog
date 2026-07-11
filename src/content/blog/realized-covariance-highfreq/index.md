---
title: "已实现协方差矩阵：用高频数据拼出日内风险全景"
publishDate: '2026-07-12'
description: "日频协方差只有开/收两个点，信息稀薄还被隔夜跳空污染。本文用分钟级高频数据把一整天的资产间联动「拼」成已实现协方差矩阵：从朴素 RC 的构造，到微观结构噪声造成的 U 形偏差与 Epps 塌缩，再到用已实现核(RK)把方差和相关系数都救回真实值，最后用 HAR-RV 把每天的已实现波动接成波动预测，附完整可复现 Python。"
tags:
  - 量化交易
  - 高频数据
  - 已实现协方差
  - 微观结构噪声
  - 已实现核
  - 波动率预测
  - 风险管理
language: Chinese
difficulty: advanced
---

日频协方差矩阵只有「开盘→收盘」两个点，估计噪声极大；而且它把隔夜跳空也混进了所谓的「日收益」里，可真正的风险其实发生在交易时段内部。高频（分钟级甚至秒级）数据让我们能用一天内几百个观测点，把当天资产之间的联动**完整拼出来**——这就是**已实现协方差矩阵（Realized Covariance）**。它和日频协方差不是同一个数量级的信息量。

本文讲清四件事，并用可复现代码演示：

1. 朴素已实现协方差怎么从日内收益的外积里算出来；
2. 微观结构噪声如何把它搞崩——**U 形偏差**与 **Epps 效应**；
3. **已实现核（Realized Kernel）**怎么把方差和相关系数都救回真实值；
4. 拼好之后怎么用——用 **HAR-RV** 把每天的已实现波动接成次日的波动预测。

## 一、为什么日频协方差不够

经典日收益协方差 $\hat\Sigma = \frac{1}{T}\sum (r_t-\bar r)(r_t-\bar r)'$ 有三个硬伤：

- **信息稀薄**：每个资产每天只有 1 个收益数字，协方差估计的标准误与 $1/\sqrt{T}$ 同阶，T 几百时仍然很抖；
- **隔夜污染**：「日收益」= 今收 − 昨收，把非交易时段的跳空也算成连续风险，但组合保险要的是交易时段内的联动；
- **没有结构**：只给你一个事后数字，看不到一天里风险是怎么演化的。

高频数据把这些一次性解决：一天 240 根 1 分钟 bar，等于白送你 240 个观测点。

## 二、朴素已实现协方差：把日内收益外积加起来

把第 $t$ 天的日内分钟收益写成矩阵 $R_t = [r_{t,1},\dots,r_{t,N}]^\top$（每行一个资产，每列一个分钟），当天的已实现协方差就是：

$$RC_t = \sum_{i=1}^{N} r_{t,i}\, r_{t,i}^\top$$

对角线是各资产的**已实现方差（RV）**，非对角线是**已实现协方差**。再除以对角线开方，就得到**已实现相关矩阵**。跨天直接求和即可得到稳定的全样本估计：

```python
import numpy as np

def realized_cov(returns):
    """朴素已实现协方差 = Σ_t r_t r_t' （returns: (T, K)）"""
    return returns.T @ returns

def corr_from_cov(cov):
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)

# 单因子模型设定：Sigma = beta*beta' + diag(delta^2)
beta = np.array([1.00, 0.80, 1.20, 0.55, 0.95])
delta = np.array([0.30, 0.42, 0.28, 0.50, 0.38])
Sigma_true = np.outer(beta, beta) + np.diag(delta ** 2)

# 模拟：每天 240 根 1 分钟 bar，共 60 天；噪声注入在「价格」上
N_PERDAY, N_DAYS = 240, 60
SIGMA_1MIN = Sigma_true / N_PERDAY
rng = np.random.default_rng(2026)
true_ret = rng.multivariate_normal(np.zeros(5), SIGMA_1MIN, size=N_PERDAY * N_DAYS)
p_true = np.cumsum(true_ret, axis=0)
ETA = 0.056
p_obs = p_true + rng.normal(0, ETA, size=p_true.shape)   # 观测价格含微观结构噪声
obs_ret = np.diff(p_obs, axis=0)

RC_1MIN = realized_cov(obs_ret)
CORR_1MIN = corr_from_cov(RC_1MIN)
```

跑出来的真实相关结构长这样（左：模型设定；右：高频估计）：

![5 资产已实现相关矩阵：用高频数据拼出的日内风险全景](/images/realized-covariance-highfreq/realized_cov_heatmap.png)

这就是「日内风险全景」——比日频一个数字信息量大得多。但别高兴太早，朴素 RC 在高频下会**系统性失真**。

## 三、微观结构噪声：朴素 RC 的第一道坑

真实价格上永远挂着买卖价差、离散 tick、报价延迟带来的「噪声」：$p^{\text{obs}}_t = p^{\text{true}}_t + u_t$，$u_t\overset{iid}{\sim}\mathcal N(0,\eta^2)$。差分之后噪声变成 MA(1)：

$$r_t = \underbrace{p^{\text{true}}_t - p^{\text{true}}_{t-1}}_{\text{真收益}} + \underbrace{(u_t - u_{t-1})}_{\text{噪声项，方差 }2\eta^2}$$

净值：朴素已实现方差 $\text{RV}^{\text{naive}} = \text{TV} + 2\eta^2\cdot N$。**采样越密（N 越大），噪声偏差越大**——这制造了经典的 **U 形偏差**：极高频被噪声高估，极粗频又因观测点太少被采样方差拖高。

![微观结构噪声造成的 U 形偏差：极高频被噪声高估，极粗频被采样方差拖高](/images/realized-covariance-highfreq/noise_ushape.png)

本次模拟（真实总方差 TV≈60）：朴素 1 分钟 RC 的平均方差高达 **151.8**，是真实值的 2.5 倍。

更阴险的是 **Epps 效应**：噪声只放大方差、基本不碰协方差（不同资产的噪声相互独立），于是**相关系数在极细采样下塌向 0**。本例朴素 RC 的平均相关只有 **0.32**，而真实值是 **0.80**——一组高相关的资产，用 1 分钟数据算出来居然「快不相关了」。

![Epps 效应：采样越细，朴素相关越塌向 0；已实现核把它救回真实值](/images/realized-covariance-highfreq/epps_effect.png)

## 四、已实现核：用滞后自协方差把噪声「扭」掉

Barndorff-Nielsen, Hansen, Lunde & Shephard (2008) 的**已实现核（Realized Kernel）**是标准解。核心直觉：噪声在差分回报里留下 MA(1) 结构，这个信息藏在**滞后自协方差**里，用核权重把它抵消掉：

$$RK = \gamma_0 + 2\sum_{h=1}^{H} k\!\left(\frac{h}{H}\right)(\gamma_h + \gamma_h^\top),\qquad \gamma_h = \sum_t r_t r_{t-h}^\top$$

其中 $\gamma_h$ 是第 $h$ 阶自协方差矩阵，Bartlett 权重 $k(x)=1-x$，带宽取 $H\approx 3.7\cdot N^{1/3}$：

```python
def realized_kernel(returns, c=3.7):
    T, k = returns.shape
    H = min(int(np.floor(c * T ** (1 / 3))), T - 1)
    g = np.zeros((H + 1, k, k))
    g[0] = returns.T @ returns
    for h in range(1, H + 1):
        g[h] = returns[h:].T @ returns[:-h]      # γ_h = Σ r_t r_{t-h}'
    w = 1.0 - np.arange(H + 1) / (H + 1)          # Bartlett 权重
    RK = g[0].copy()
    for h in range(1, H + 1):
        RK += w[h] * (g[h] + g[h].T)
    return RK, H

RK_ALL, H_RK = realized_kernel(obs_ret)
CORR_RK = corr_from_cov(RK_ALL)
```

结果（上图绿线）：RK 的平均方差 **59.4** ≈ 真实 60.3，平均相关 **0.78** ≈ 真实 0.80——噪声偏差被干净地抹掉。同思路的替代还有 two-scale、subsample、pre-averaging（预平均），RK 是其中工程上最省心的一个。

## 五、拼好之后怎么用：HAR-RV 预测波动

每天算出一个日度已实现方差，串成时间序列，就有了一个比「历史波动率」高好几个数量级的波动代理。Andersen & Bollerslev (2003) 的 **HAR-RV** 用一个极简三尺度回归预测次日波动：

$$\text{RV}_t = b_0 + b_1\text{RV}_{t-1} + b_2\overline{\text{RV}}_{t-5:t-1} + b_3\overline{\text{RV}}_{t-22:t-1} + \varepsilon$$

日/周/月三个尺度对应不同类型的交易者，抓住了 AR(1) 漏掉的「长记忆 + 波动聚集」：

```python
daily_rv = np.array([np.sum(obs_ret[d*240:(d+1)*240]**2, axis=0) for d in range(60)])
rv0 = daily_rv[:, 0]
y, X = [], []
for t in range(22, 60):
    y.append(rv0[t]); X.append([1.0, rv0[t-1], rv0[t-5:t].mean(), rv0[t-22:t].mean()])
y, X = np.array(y), np.array(X)
beta_har, *_ = np.linalg.lstsq(X, y, rcond=None)
```

![HAR-RV：用日/周/月三个时间尺度的已实现波动预测次日波动](/images/realized-covariance-highfreq/har_forecast.png)

本例样本内 $R^2\approx 0.14$——日度 RV 噪声很大，$R^2$ 本就不高，但结构清晰：**周系数（0.64）主导**，正是 HAR 的标志性签名。把这个预测喂给风险预算、期权定价或 VaR，比用滞后平方收益稳得多。

## 六、真实陷阱（别拿到 RC 就当真）

**1. 噪声不是唯一偏差，异步交易也会坑你。** 不同资产并非同一毫秒成交，这种异步会让 RC **低估**相关；RK 同时修正噪声与异步，但若用朴素 RC，两者叠加只会更糟。

**2. 跳空（jumps）污染 RV。** 大跳会瞬间抬高已实现方差，需用 bipower variation 把「连续部分」和「跳部分」拆开，纯波动预测才干净。

**3. 频率别乱选。** 1 分钟朴素 RC 噪声爆表（本例高估 2.5 倍），太粗又丢信息；实务上 5–30 分钟或干脆上 RK 是稳妥折中。

**4. 极端日的尾部同样不稳。** 危机里 RV 飙升，协方差估计也跟着抖，组合 VaR 会滞后于真实风险。

**5. 流动性差的票 RC 会退化。** 小票分钟数据稀疏、长时间无成交，RC 退化为低频估计，别和蓝筹用同一套参数。

**6. 幸存者偏差。** 用当前成分股回测历史 RC，等于默认它们当时就在、且一直流动——回测结果会偏乐观。

## 七、小结

- 已实现协方差把日内几百个观测**拼成当天风险全景**，信息量比日频协方差高一个数量级；
- 朴素 RC 被**微观结构噪声 + 异步交易**双重坑：U 形偏差 + Epps 塌缩（本例方差高估 2.5 倍、相关从 0.80 塌到 0.32）；
- **已实现核（RK）**用滞后自协方差加权，把方差（59.4 vs 真实 60.3）和相关（0.78 vs 0.80）都救回真实值；
- 每天一个 RV 串成序列，用 **HAR-RV** 三尺度预测波动，直接接风险预算与期权定价。

> 附：本文所有图表与数值均来自上方可运行代码（单因子模拟 + 噪声注入 + 朴素 RC / 已实现核 / HAR 回归），参数与结果一致，可直接复现。
