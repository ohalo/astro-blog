---
title: "CAViaR 条件自回归 VaR：让分位数自己带记忆演化"
publishDate: '2026-07-27'
description: "CAViaR 条件自回归 VaR：让分位数自己带记忆演化 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

先说结论：**GARCH 算 VaR 的路线是"先建波动率模型 + 再假设一个分布 + 最后取分位数"，三步里任何一步错都会传染到 VaR。Engle 和 Manganelli（2004）的 CAViaR 直接砍掉前两步——让分位数本身服从一个自回归过程 $f_t = \beta_0 + \beta_1 f_{t-1} + \beta_2 |r_{t-1}|$，用 pinball 损失直接估计，全程不对收益分布做任何假设。我在 GARCH-t(6) 模拟上实测：静态正态 VaR 和滚动历史模拟在 [DQ 检验](/blog/dynamic-quantile-test-dq/)下 p=0.0002 和 p=0.0000 双双被拒，CAViaR-SAV 突破 49 次（期望 50）、DQ p=0.90，pinball 损失 0.0973——距离知道真实波动率和真实分布的"神谕"模型（0.0959）只差 1.5%。它不知道数据是 GARCH 生成的，也不知道噪声是 t 分布，却几乎追平了全知模型**。

## 三步路线的传染问题

标准的条件 VaR 流程是：

1. 建一个波动率模型（GARCH、EWMA、SV）得到 $\hat\sigma_t$；
2. 假设标准化残差的分布 $F$（正态、t、EVT 尾部）；
3. VaR 取 $\hat\mu_t + \hat\sigma_t F^{-1}(p)$。

这条链上每个环节都是一个可以出错的假设。[Kupiec POF 检验](/blog/kupiec-pof-test/)那篇文章里我们看到过一个惨案：GARCH-正态模型的波动率是对的、分布是错的，结果比静态模型死得更快——因为动态波动率把"错误的尾部厚度"每天都精确地涂在正确的位置上。

CAViaR 的观察很直接：**我们最终要的只是一个分位数，为什么要绕道去建整个分布？**分位数是收益分布的一个函数泛函，波动率聚集意味着分位数也聚集——那就直接给分位数写一个带记忆的演化方程。

## 模型形式：给 VaR 装一个自回归引擎

Engle-Manganelli 提出了几个具体形式，最常用的两个：

**SAV（Symmetric Absolute Value，对称绝对值）：**

$$f_t(\beta) = \beta_0 + \beta_1 f_{t-1}(\beta) + \beta_2 |r_{t-1}|$$

**AS（Asymmetric Slope，非对称斜率）：**

$$f_t(\beta) = \beta_0 + \beta_1 f_{t-1}(\beta) + \beta_2 \max(r_{t-1}, 0) + \beta_3 \max(-r_{t-1}, 0)$$

这里 $f_t$ 就是第 $t$ 天的条件分位数（对左尾 VaR 来说是个负数）。结构上和 GARCH(1,1) 神似：

- $\beta_1 f_{t-1}$ 是**记忆项**——今天的 VaR 锚定在昨天的 VaR 上，自然产生平滑和聚集；
- $\beta_2 |r_{t-1}|$ 是**新闻项**——昨天市场动静大，今天 VaR 就往外推；
- AS 版本把涨和跌分开，允许杠杆效应（下跌比上涨更能推高风险）。

关键区别在估计方式。GARCH 用极大似然，需要写出完整的条件密度；CAViaR 用**分位数回归的 pinball 损失**：

$$\min_\beta \frac{1}{T}\sum_t \rho_p(r_t - f_t(\beta)), \quad \rho_p(u) = u\,(p - \mathbf{1}\{u < 0\})$$

pinball 损失只关心"预测的分位数把数据切成了 p 比 1-p 的两块没有"，对分布的其他一切——偏度、峰度、尾部形状——完全免疫。这就是"无分布假设"的确切含义。

代价是这个目标函数**不光滑、非凸**：$f_t$ 递归依赖 $\beta$，pinball 又带绝对值折角。Engle-Manganelli 的原始做法是撒几千个随机初值取最好的几个再做局部优化，我照抄：

```python
import numpy as np
from scipy.optimize import minimize

def caviar_sav_path(params, r, q0):
    """递归生成整条 VaR 路径"""
    b0, b1, b2 = params
    q = np.empty(len(r))
    q[0] = q0                      # 用前 300 天经验分位数初始化
    for t in range(1, len(r)):
        q[t] = b0 + b1 * q[t-1] + b2 * abs(r[t-1])
    return q

def pinball_loss(y, q, tau):
    u = y - q
    return np.mean(u * (tau - (u < 0)))

def multistart(obj, n_par, n_draws=2000, n_polish=10):
    """Engle-Manganelli 式多起点优化"""
    rs = np.random.default_rng(1)
    draws = rs.uniform(-1, 1, size=(n_draws, n_par))
    draws[:, 1] = rs.uniform(0.5, 1.0, n_draws)   # 记忆系数先验上应在 (0.5, 1)
    losses = np.array([obj(d) for d in draws])
    best = None
    for i in np.argsort(losses)[:n_polish]:
        res = minimize(obj, draws[i], method="Nelder-Mead",
                       options={"maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    return best
```

## 实验：2500 天 GARCH-t(6)，前 1500 天估计，后 1000 天验尸

数据生成过程和本系列前几篇一致：GARCH(1,1) 波动率（持续性 0.98）+ t(6) 标准化噪声，目标是 5% 左尾 VaR。样本内估计参数后冻结，样本外 1000 天出成绩单。

估出来的 SAV 参数：$\beta_0 = -0.209,\ \beta_1 = 0.742,\ \beta_2 = -0.237$。记忆系数 0.74，说明模型自己从数据里学到了"风险是有惯性的"；$\beta_2$ 为负是因为左尾 VaR 是负数——昨天波动越大，今天的分位数被推得越深。

参赛选手五名：静态正态（样本内均值方差一次定终身）、滚动 250 天历史模拟、CAViaR-SAV、CAViaR-AS、以及知道真实 $\sigma_t$ 和真实 t(6) 分布的**神谕模型**（性能上界）。

![样本外 5% VaR 路径对比](/images/caviar-conditional-autoregressive-var/caviar-var-paths.jpg)

成绩单（样本外 1000 天，期望突破 50 次）：

| 模型 | 突破次数 | Kupiec p | DQ p | pinball 损失 |
|---|---|---|---|---|
| 静态正态 | 60 | 0.159 | **0.0002** | 0.1044 |
| 滚动历史 250d | 60 | 0.159 | **0.0000** | 0.1046 |
| CAViaR-SAV | 49 | 0.884 | 0.900 | 0.0973 |
| CAViaR-AS | 53 | 0.666 | 0.967 | 0.0975 |
| GARCH-t 神谕 | 50 | 1.000 | 0.907 | 0.0959 |

![五个模型的成绩单](/images/caviar-conditional-autoregressive-var/caviar-model-scoreboard.jpg)

三个值得停下来看的点：

**第一，频率检验又一次集体失明。**静态正态和滚动历史都是 60 次突破，Kupiec p=0.159，单看次数无罪。但 DQ 检验直接给出 0.0002 和 0.0000 的死刑判决——突破全部扎堆在高波动段。这张图和 [Christoffersen 检验](/blog/christoffersen-independence-test/)那篇里的完全同构：

![静态模型突破扎堆，CAViaR 摊平](/images/caviar-conditional-autoregressive-var/caviar-hit-clustering.jpg)

**第二，CAViaR 几乎追平神谕。**pinball 损失 0.0973 vs 0.0959，差距 1.5%。要知道神谕模型拿的是作弊剧本——真实波动率路径和真实噪声分布。CAViaR 对这两样一无所知，只靠"分位数自回归 + pinball 损失"就把差距压到了个位数百分比。这就是直接建模目标量的好处：**你不需要对整个分布正确，只需要对你关心的那一个分位数正确**。

**第三，滚动历史模拟不是动态模型。**很多人以为 250 天滚动窗口"会自适应"，实测它和静态正态一样惨。原因是窗口平均把波动率的日频变化抹掉了——高波动日进窗口后要影响 250 天，反应速度慢一个数量级。它的"动态"是月度尺度的，而波动聚集是日度尺度的。

## AS 的非对称是真的吗

AS 版本估出来的涨跌系数分别是 -0.261（上涨）和 -0.193（下跌），样本内 pinball 略优于 SAV（0.09397 vs 0.09411）。如果不加思考，可以编一个"上涨也推高风险"的故事。

但我的数据生成过程是**对称的**——t(6) 噪声没有杠杆效应，涨跌对后续波动的影响天生一样。所以这个"非对称"纯粹是 400 个尾部观测里的估计噪声，样本外 AS 也没有跑赢 SAV（0.0975 vs 0.0973）。

![新闻冲击曲线：SAV 对称，AS 学出了假非对称](/images/caviar-conditional-autoregressive-var/caviar-news-impact.jpg)

这是个便宜但重要的教训：**参数更多的模型永远能在样本内讲出更动听的故事**。真实股票数据上杠杆效应是存在的（Engle-Manganelli 在 GM 股票上发现 AS 显著占优），但你得先在模拟数据上确认自己的流程不会无中生有，才有资格相信真实数据上的非对称是信号。

## 工程清单

实际部署 CAViaR 时的几个要点：

1. **优化必须多起点**。目标函数非凸非光滑，单起点 Nelder-Mead 大概率停在烂局部解。原文配方：撒 10^4 级别均匀随机初值 → 取 pinball 最小的 10 个 → 逐个局部优化 → 取全局最好。我用 2000 个初值也够稳；
2. **记忆系数 $\beta_1$ 检查平稳性**。$|\beta_1| \ge 1$ 意味着 VaR 路径发散，直接拒绝该解、换下一个局部最优；
3. **初始值 $f_0$ 用经验分位数**，比如前 300 天的样本 p 分位数。$\beta_1 < 1$ 保证初始化的影响几何衰减，但前几十天的 VaR 还是别拿去做回测评分；
4. **每个分位数水平单独估一套参数**。5% 和 1% 的 CAViaR 是两个独立模型，参数不通用——这既是灵活性（不同尾部深度可以有不同动态）也是隐患（多条分位数可能交叉，1% 的 VaR 可能算出来比 5% 还浅，需要事后重排）；
5. **回测用 DQ 而不只是 Kupiec**。CAViaR 和 DQ 检验出自同一篇论文不是巧合——模型是"分位数对信息集自适应"，检验是"突破对信息集不可预测"，一体两面。

## 诚实边界

- **1% 层级估计更难**。5% 层级 1500 天有约 75 个尾部观测参与定形，1% 只有 15 个。CAViaR 在极深尾部的参数方差会大到需要用更长历史或把外推交给 EVT；
- **递归结构对离群点敏感**。一个异常的 $|r_{t-1}|$（比如错误数据、股票乌龙指）会通过记忆项污染之后几十天的 VaR。生产系统需要在输入端做 winsorize；
- **无分布假设 ≠ 无假设**。CAViaR 假设了分位数动态的函数形式（线性自回归 + 绝对值新闻项）。如果真实的分位数动态是别的形状（比如政权切换式跳变），SAV/AS 一样会错，只是错的方式和 GARCH 不同；
- **只给分位数不给分布**。需要 ES（预期短缺）时 CAViaR 本身不够用，要么配 [Acerbi-Szekely 式](/blog/acerbi-szekely-es-test/)的扩展（如 Taylor 2019 的联合 elicitable 打分），要么退回半参数路线。

下一篇讲一个 CAViaR 多分位数版本会撞上的通用问题：**分位数交叉**——当 95% 分位数预测得比 90% 还低，模型在逻辑上自相矛盾，怎么修。

## 参考文献

- Engle, R. F., & Manganelli, S. (2004). CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles. *Journal of Business & Economic Statistics*, 22(4), 367-381.
- Koenker, R., & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Taylor, J. W. (2019). Forecasting Value at Risk and Expected Shortfall Using a Semiparametric Approach Based on the Asymmetric Laplace Distribution. *Journal of Business & Economic Statistics*, 37(1), 121-133.
