---
title: "GARCH 波动率模型族：从 ARCH 到 EGARCH 的实战预测"
description: "波动率是会被预测的：ARCH 用昨日平方收益、GARCH 加上昨日波动、EGARCH/GJR 再补上「坏消息更可怕」的杠杆效应。从零实现 MLE 拟合与多步预测，并给出模型选择与真实陷阱。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 波动率
  - GARCH
  - ARCH
  - EGARCH
  - Python
language: Chinese
difficulty: advanced
---

波动率最反直觉的一点：**它不是常数，而且会「成群结队」**。安静的行情能连着几周风平浪静，暴跌往往接二连三。如果你用「过去一年波动率是 20%」这种常数假设去给期权定价、设止损、控仓位，就会在暴风雨来临时措手不及。

结论先放这：**GARCH 族模型把波动率建模成一个随时间演化的隐变量——今天的高波动会「传染」给明天。ARCH 只用昨日平方收益驱动，GARCH 再加上昨日波动本身的持续项，EGARCH/GJR 再补上「下跌比上涨更推高波动」的杠杆效应。用极大似然(ML)就能从价格里把它挖出来，并做多步预测。**

![波动率聚集：平静期与暴风雨期成串出现](/images/garch-volatility-family/garch_returns_clustering.png)

## 一、波动率聚集：为什么需要 GARCH

看上图：模拟收益的剧烈波动明显成串出现——这正是真实市场的「波动率聚集」(volatility clustering)。常数方差模型（比如 Black-Scholes 假设）完全描述不了这种结构。

Engle(1982) 的 ARCH 思想很简单：条件方差 $\sigma_t^2$ 是**过去平方收益**的加权平均。Bollerslev(1986) 推广成 GARCH，让方差还能「记住」自己昨天的值：

$$\sigma_t^2 = \omega + \alpha\, r_{t-1}^2 + \beta\, \sigma_{t-1}^2$$

- $\omega$：长期基准方差
- $\alpha$：对「昨日冲击」$r_{t-1}^2$ 的敏感
- $\beta$：波动的持续性（记忆）
- **持续性指标 $\alpha+\beta$**：越接近 1，波动越「黏」——一次冲击要很久才衰减

约束 $\alpha+\beta<1$ 保证方差平稳（长期方差有限）。

## 二、从 ARCH 到 GARCH(1,1)：用 MLE 拟合

在「收益服从条件正态」下，第 $t$ 期的对数似然是：

$$\ell_t = -\frac12\Big(\ln(2\pi) + \ln\sigma_t^2 + \frac{r_t^2}{\sigma_t^2}\Big)$$

把全部 $-\sum\ell_t$ 用 scipy 最小化，参数就出来了。下面是从零实现的完整拟合：

```python
import numpy as np
from scipy.optimize import minimize

def garch11_nll(params, r):
    omega, alpha, beta = params
    n = len(r); sig2 = np.zeros(n); sig2[0] = np.var(r); ll = 0.0
    for t in range(1, n):
        sig2[t] = omega + alpha*r[t-1]**2 + beta*sig2[t-1]
        if sig2[t] <= 1e-12:
            return 1e10                      # 约束方差为正
        ll += 0.5*(np.log(2*np.pi) + np.log(sig2[t]) + r[t]**2/sig2[t])
    return ll

def fit_garch11(r):
    x0 = [1e-5, 0.1, 0.85]
    bnds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.999)]
    res = minimize(garch11_nll, x0, args=(r,), bounds=bnds, method="L-BFGS-B")
    return res.x, res.fun

omega_hat, alpha_hat, beta_hat = fit_garch11(r_fit)[0]
```

下面是一次模拟数据的拟合结果（数据本身用带杠杆效应的 GARCH 生成，细节见第五节）。拟合出的条件波动率紧跟 $|收益|$ 的起伏：

![GARCH(1,1) 拟合：条件波动率紧跟 |收益|](/images/garch-volatility-family/garch_fit_condvol.png)

注意两个实务细节：(1) 必须约束 $\sigma_t^2>0$，否则优化会跑到负数让似然爆炸；(2) **学生 $t$ 似然通常比正态更好**——收益有肥尾，正态似然会系统性高估波动、低估极端风险。把 $\ell_t$ 里的正态密度换成 $t$ 密度即可，框架不变。

作为对照，**ARCH(1) 是 GARCH 砍掉 $\beta$ 项的退化版**：$\sigma_t^2=\omega+\alpha r_{t-1}^2$，波动完全没有「记忆」，全靠昨日冲击驱动。它适合波动切换很快、持续性弱的市场（比如某些高频微观结构噪声），但在日度股票数据上，纯 ARCH 往往要很高阶才能追上 GARCH(1,1) 的表现——这正是 GARCH 用「一个 $\beta$ 项记住昨日波动」省下大量参数的精妙之处。拟合时若发现 $\beta$ 显著不为零（绝大多数股票指数都是），就别用 ARCH 了。

## 三、多步预测与持续性

GARCH 的真正价值在**预测**。已知到 $T$ 的信息，多步条件方差可以递归：

$$\hat\sigma_{T+h|T}^2 = \omega + (\alpha+\beta)\,\hat\sigma_{T+h-1|T}^2 + \alpha\,\mathbb{E}[r_{T+h-1}^2]$$

对未知的将来收益，用 $\mathbb{E}[r^2|\sigma]=\sigma^2$ 替换，递推就只剩不变分量。关键结论：**多步方差预测会指数收敛到长期方差 $\omega/(1-\alpha-\beta)$**，收敛速度由持续性 $\alpha+\beta$ 决定。$\alpha+\beta$ 越接近 1，当前高波动「赖着不走」越久。

```python
h = len(r_oos)
sig2_last = sig_fit[-1]**2
fore_var = np.zeros(h)
fore_var[0] = omega_hat + alpha_hat*r_fit[-1]**2 + beta_hat*sig2_last
for i in range(1, h):
    fore_var[i] = omega_hat + (alpha_hat + beta_hat)*fore_var[i-1]
fore_sig = np.sqrt(fore_var)
```

![样本外预测：GARCH 把波动拉回长期水平](/images/garch-volatility-family/garch_forecast_fan.png)

样本外实测里，GARCH 预测会贴着实现波动走一阵，然后随持续性衰减慢慢回归长期水平——这正是「波动黏性」的可视化。

**一个常被忽略的实用量：冲击的半衰期(half-life)。** 一次收益冲击对条件方差的影响按 $(\alpha+\beta)^h$ 衰减，令其等于 0.5 解得半衰期：

$$h_{1/2} = \frac{\ln 0.5}{\ln(\alpha+\beta)} = \frac{-\ln 2}{\ln(\alpha+\beta)}$$

上面那次拟合得到 $\alpha+\beta=0.994$，代入可得 $h_{1/2}\approx \ln2/0.006\approx 115$ 个交易日，约 5~6 个月——意味着一次大波动的余波要小半年才衰减到一半。半衰期是给风控和做市商定「波动记忆窗口」的硬指标：窗口短于它，你就在低估持续性；长于此，又在对早已消退的旧冲击过度反应。

## 四、杠杆效应：EGARCH 与 GJR

ARCH/GARCH 对「正收益」和「负收益」的冲击一视同仁。但真实市场有个铁律：**同样幅度的下跌，比上涨更能推高次日波动**（恐慌比狂热更持久）。这就是杠杆效应(leverage effect)。

**GJR-GARCH**(Glosten-Jagannathan-Runkle) 用一个指示函数补上非对称：

$$\sigma_t^2 = \omega + \big(\alpha + \gamma\,\mathbf{1}_{r_{t-1}<0}\big) r_{t-1}^2 + \beta\,\sigma_{t-1}^2$$

下跌时（$\mathbf{1}=1$）系数变成 $\alpha+\gamma$，波动冲击更大。

**EGARCH** 则在方差的对数上建模，天然保证正定性，且用标准化残差 $z_t=r_t/\sigma_t$：

$$\ln\sigma_t^2 = \omega + \alpha\big(|z_{t-1}|-\mathbb{E}|z|\big) + \gamma\, z_{t-1} + \beta\ln\sigma_{t-1}^2$$

$\gamma<0$ 即杠杆效应。新闻冲击曲线(news impact curve)把这种非对称性画得最清楚：横轴是前一日收益，纵轴是次日方差增量。

![新闻冲击曲线：坏消息比好消息更能推高波动](/images/garch-volatility-family/garch_news_impact.png)

左半边（下跌）的曲线明显比右半边（上涨）高——对称 GARCH 是抛物线（左右对称），GJR 在负收益区被 $\gamma$ 抬起来。所以**如果你做波动预测或期权定价却忽略杠杆效应，会在大跌后系统性低估接下来的波动**。

量化地看，$\gamma$ 的大小直接告诉你「坏消息的额外冲击」有多大。那次拟合得到 $\gamma\approx0.056$，意味着同样 3% 的跌幅比 3% 的涨幅，次日方差增量多约 $0.056\times0.03^2\approx5\times10^{-5}$（年化约 1.3 个 vol 点）——单看不大，但在连续下跌、波动自我强化时，这个非对称会被层层放大，最终造成「跌得越狠、后面越吓人」的恐慌螺旋。这也是恐慌指数 VIX 在熊市里尖冲的核心机制。

## 五、模型选择：用 AIC / 对数似然说话

多个模型怎么选？别看谁「图好看」，看**信息准则**。对每个候选模型算 MLE 对数似然 $\hat L$，比较：

$$\text{AIC} = 2k - 2\hat L,\qquad \text{BIC} = k\ln T - 2\hat L$$

$k$ 是参数个数，越小（越扣复杂度惩罚）越好。同一组数据上，EGARCH/GJR 若显著优于对称 GARCH，说明杠杆效应真实存在，值得多那一个参数。也可以做**滚动样本外 RMSFE**（均方预测误差）对比——样本内拟合好不等于样本外预测好，RMSFE 才是真刀真枪。

```python
def gjr_nll(params, r):
    omega, alpha, gamma, beta = params
    n = len(r); sig2 = np.zeros(n); sig2[0] = np.var(r); ll = 0.0
    for t in range(1, n):
        sig2[t] = omega + (alpha + gamma*(r[t-1] < 0))*r[t-1]**2 + beta*sig2[t-1]
        if sig2[t] <= 1e-12:
            return 1e10
        ll += 0.5*(np.log(2*np.pi) + np.log(sig2[t]) + r[t]**2/sig2[t])
    return ll

res_gjr = minimize(gjr_nll, [1e-5, 0.05, 0.05, 0.85], args=(r_fit,),
                   bounds=[(1e-8, None)]*3 + [(1e-6, 0.999)], method="L-BFGS-B")
```

**桥梁：RiskMetrics 是 GARCH 的特例。** JP Morgan 的 RiskMetrics 用的就是 IGARCH：$\alpha=1-\lambda,\;\beta=\lambda$，日度 $\lambda=0.94$（即 $\alpha=0.06,\beta=0.94$），且 $\omega\to0$。它是 GARCH(1,1) 把 $\alpha+\beta$ 钉在 1 上的工程简化版——好处是无需估计、实时递推，坏处是方差非平稳、长期预测发散。理解了这个，你就知道为什么很多风控系统的「波动率」其实只是 GARCH 的一个硬编码特例。

**下游：用 GARCH 算 VaR。** 参数法 VaR 直接吃 GARCH 的条件波动：

```python
z_alpha = 1.645                       # 95% 单尾
var_t = z_alpha * fore_sig * portfolio_value   # 当日 VaR 随 σ_t 滚动
```

相比用常数历史波动，GARCH-VaR 会在暴跌后自动抬高，少报「虚假的安全」。

## 六、真实陷阱（别假装没看见）

- **持续性爆表**：若 $\alpha+\beta$ 拟合出 0.99+，说明模型在「用长期记忆掩盖结构性变化」——这种高持续性往往不稳定，遇到 regime switch 会失灵。可加一个慢变变量或上 Markov-Switching。
- **方差平稳假设**：GARCH 要求 $\alpha+\beta<1$。$\ge1$ 是 IGARCH/单位根，方差无界，长期预测发散，别硬用。
- **日内/跳空**：日度 GARCH 把隔夜跳空当成普通收益，会高估波动持续性。用 5 分钟高频算 realized volatility 再做 HAR 模型会更好。
- **前视偏差**：预测时只能用 $t$ 及之前的信息，多步预测里的 $\mathbb{E}[r^2]$ 必须用模型递推，不能拿真实未来收益算。
- **肥尾**：正态似然下，极端日会主导似然函数，参数被 outlier 带偏。换成 $t$ 或 GED 分布更稳。
- **结构突变 vs 持续**：GARCH 用一个固定持续性参数「平均」了整个历史。若市场经历过机制切换（牛熊转换、政策冲击），固定 $\alpha+\beta$ 会同时低估旧 regime 的衰减、高估新 regime 的持续。可上 Markov-Switching GARCH 或滚动窗口。

## 结语

GARCH 族是波动率建模的「瑞士军刀」：ARCH 给思想，GARCH 给记忆，EGARCH/GJR 给非对称。它不神秘——本质就是用极大似然把一个「波动的波动」结构从价格里挖出来。实战里，它是期权波动率预测、风险价值(VaR)计算、波动率目标仓位、甚至做空波动率风险溢价的前置引擎。

把这篇和上篇连起来看：贝叶斯配置解决「收益和协方差的不确定性」，GARCH 解决「波动本身的不确定性」——两者都是「把不确定性显式建模」这一量化纪律的不同侧面。
