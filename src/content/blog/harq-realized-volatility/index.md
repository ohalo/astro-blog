---
title: "已实现半方差(HARQ)：把隔夜跳变与日内连续波动拆开预测"
description: "波动率不是一团模糊的『风险』——上行与下行、连续扩散与跳变，对未来波动的预测力完全不同。本文用已实现半方差(RSV)拆开涨跌，再用 HARQ 修正测量误差，样本外把 RV 预测的 RMSE 从 1.34 压到 1.27、QLIKE 同步下降，并逐一拆解跳变、杠杆效应与测量噪声这三道真实落地关。"
publishDate: '2026-07-15'
tags:
  - 量化交易
  - 波动率预测
  - 已实现波动
  - HARQ
  - 半方差
  - 高频数据
  - 风险管理
  - 时间序列
language: Chinese
difficulty: advanced
---

「昨天跌了 3%，今天该恐慌吗？」和「昨天涨了 3%，今天该乐观吗？」——在波动率的世界里，这两句话的答案完全不对称。

大多数波动率模型犯的第一个错，就是把「波动」当成一个没有方向的标量：只要动得大，就是风险大。可真实市场里，**下跌带来的波动放大远比上涨剧烈且持久**——这就是杠杆效应。如果你的模型把涨和跌的平方收益一视同仁地塞进同一个 RV，你就主动扔掉了一半信息。

本文做两件事：第一，用**已实现半方差(Realized Semivariance, RSV)** 把已实现方差按收益符号劈成上行、下行两半；第二，在此基础上引入 **HARQ**——用已实现四次方矩(Realized Quarticity)修正 RV 本身的测量误差。目标只有一个：把明天的波动预测得更准。

---

## 一、先理解:RV 是估计量,不是真值

高频波动率研究的地基是一个朴素的事实：我们观测不到真实的瞬时波动 σ²，只能用日内高频收益的平方和去**估计**它。这个估计量叫已实现方差(Realized Variance)：

$$RV_t = \sum_{i=1}^{M} r_{t,i}^2$$

其中 $r_{t,i}$ 是第 $t$ 天第 $i$ 个日内区间（比如 5 分钟）的对数收益，$M$ 是当天的区间数。当采样越来越密（$M \to \infty$），$RV_t$ 会收敛到当天的积分方差(Integrated Variance)。

关键在于「收敛」两个字——它只是渐近成立。现实里 $M$ 有限，$RV_t$ 带着**测量误差**，而且这个误差的大小并不恒定：**在波动本身很大的日子里，RV 的估计也更不可靠**。这个「误差随波动放大」的现象，正是后面 HARQ 要修正的东西。

先把日内数据模拟出来，累积成 RV，同时把它按收益符号拆成上行半方差 $RSV^+$ 与下行半方差 $RSV^-$：

```python
import numpy as np
import pandas as pd

def realized_measures(intraday_returns):
    """
    从日内收益序列计算：RV、上行半方差、下行半方差、跳变代理、RQ。
    intraday_returns: 一维 array，某一天的所有 5 分钟对数收益。
    """
    r = np.asarray(intraday_returns, dtype=float)
    M = len(r)
    rv      = np.sum(r ** 2)                        # 已实现方差
    rsv_pos = np.sum(r[r > 0] ** 2)                 # 上行半方差
    rsv_neg = np.sum(r[r < 0] ** 2)                 # 下行半方差
    # realized quarticity：RV 测量误差方差的一致估计
    rq      = (M / 3.0) * np.sum(r ** 4)
    return rv, rsv_pos, rsv_neg, rq

# 注意恒等式：RV = RSV_pos + RSV_neg
```

这里有一个必须记住的恒等式：$RV_t = RSV^+_t + RSV^-_t$。半方差不是新造的指标，它只是把已经算好的平方收益按符号归了个类——**零额外数据成本，却多出了一整维方向信息**。

---

## 二、上行 vs 下行:波动的方向记忆

把 180 天的上行与下行半方差画出来（下行做镜像处理放到零轴以下），最直观的一件事就浮现了：**下行半方差的尖峰更高、聚集段更长**。

![已实现半方差分解：下行冲击抬升次日波动更多（杠杆效应）](/images/harq-realized-volatility/harq_semivariance_decomp.png)

这背后的经济含义是 Barndorff-Nielsen、Kinnebrock 与 Shephard（2010）系统化的发现：**下行半方差对未来波动的预测力显著强于上行半方差**。直觉上也说得通——下跌往往伴随保证金追缴、去杠杆、恐慌性抛售，这些机制会自我强化，把波动往后拖；而上涨很少触发同等强度的连锁反应。

所以经典 HAR 模型（Corsi, 2009）里那个「日分量」——用昨天的 RV 预测今天——如果拆成上下行两个分量，让模型对下行赋予更大的权重，预测就应该更准。这就是 **HAR-RS(HAR with Realized Semivariance)**：

$$RV_t = \beta_0 + \beta^+ RSV^+_{t-1} + \beta^- RSV^-_{t-1} + \beta_w RV^{(w)}_{t-1} + \beta_m RV^{(m)}_{t-1} + \varepsilon_t$$

其中 $RV^{(w)}$、$RV^{(m)}$ 是过去 5 天、22 天的 RV 均值（周、月分量），刻画波动的长记忆。

---

## 三、跳变:尖峰厚尾从哪来

把「连续扩散部分」（RV 减去跳变方差）和「跳变方差」画成散点，会看到跳变虽然出现频率低，但一旦出现就贡献巨大的方差——**它是收益分布尖峰厚尾的主要来源**。

![隔夜/日内跳变 vs 连续扩散：跳变贡献尖峰厚尾](/images/harq-realized-volatility/harq_jump_continuous.png)

跳变(jump)和连续扩散(continuous)的区别不只是学术洁癖：

- **连续扩散**部分有很强的持续性——今天的连续波动高，明天大概率还高。
- **跳变**部分几乎没有持续性——今天一个突发利空砸出的跳变，对明天的波动预测贡献很小。

这意味着，如果你能把 RV 里的跳变成分识别出来并单独处理（比如用 Bipower Variation 估计连续部分、用 RV 减去它得到跳变部分），预测又能再进一步。半方差其实已经隐含地部分捕捉了这一点：**大的负跳变会集中落进 $RSV^-$**，这也是下行半方差预测力更强的另一层原因。

---

## 四、HARQ:当 RV 自己都测不准

现在回到第一节埋的伏笔：RV 是带测量误差的估计量，且误差在高波动日更大。

Bollerslev、Patton 与 Quaedvlieg（2016）提出的 **HARQ** 直击这个问题。核心思路是：**当昨天的 RV 测量误差大时，就应该少信任它一点**。而测量误差的方差可以用已实现四次方矩 $RQ$ 来估计。于是 HARQ 把日分量的系数变成「随 $RQ$ 动态调整」的：

$$RV_t = \beta_0 + \left(\beta_d + \beta_{dQ}\sqrt{RQ_{t-1}}\right) RV_{t-1} + \beta_w RV^{(w)}_{t-1} + \beta_m RV^{(m)}_{t-1} + \varepsilon_t$$

括号里那一项就是精髓：当 $RQ_{t-1}$ 大（说明昨天 RV 测得不准），$\beta_{dQ}\sqrt{RQ_{t-1}}$ 通常为负，把日分量的有效权重往下压——**模型自动学会「昨天那个数噪声太大，别太当真」**。

下面是三个模型的完整实现，用样本内估计、样本外预测的规范做法：

```python
def har_design(rv, rsv_pos, rsv_neg, rq):
    """构造 HAR / HAR-RS / HARQ 的设计矩阵分量（全部滞后一期）。"""
    s = pd.Series(rv)
    d     = s.shift(1)                       # 日分量
    w     = s.rolling(5).mean().shift(1)     # 周分量
    m     = s.rolling(22).mean().shift(1)    # 月分量
    d_pos = pd.Series(rsv_pos).shift(1)      # 上行日分量
    d_neg = pd.Series(rsv_neg).shift(1)      # 下行日分量
    rq_l  = pd.Series(rq).shift(1)
    interact = np.sqrt(rq_l.clip(lower=0)) * d   # HARQ 测量误差修正项
    return d, w, m, d_pos, d_neg, interact

def fit_predict(target, Xcols, split):
    """样本内(前 split)拟合 OLS，样本外预测。返回样本外预测与真实值。"""
    X = np.column_stack([c.values if hasattr(c, "values") else c for c in Xcols])
    y = np.asarray(target, dtype=float)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    idx = np.where(valid)[0]
    tr, te = idx[idx < split], idx[idx >= split]
    Xtr = np.column_stack([np.ones(len(tr)), X[tr]])
    beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
    Xte = np.column_stack([np.ones(len(te)), X[te]])
    return Xte @ beta, y[te], beta

# 三个模型
d, w, m, d_pos, d_neg, interact = har_design(rv, rsv_pos, rsv_neg, rq)
pred_har,  yte, _ = fit_predict(rv, [d, w, m], split)
pred_hars, _,   _ = fit_predict(rv, [d_pos, d_neg, w, m], split)
pred_harq, _,   _ = fit_predict(rv, [d, w, m, interact], split)
```

评估波动预测不能只看 RMSE，还要看 **QLIKE**——它对「预测值偏低（低估风险）」的惩罚更重，更贴合风控诉求：

```python
def qlike(y_true, y_pred):
    """QLIKE 损失，越小越好；对低估波动惩罚更重。"""
    y_pred = np.clip(y_pred, 1e-10, None)
    y_true = np.clip(y_true, 1e-10, None)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
```

---

## 五、样本外结果:逐级降损

把 HAR、HAR-RS、HARQ 三个模型（外加一个把两者合并的 HARQ-RS）放到同一段样本外数据上比拼：

![样本外损失对比：拆半方差 + RQ 测量误差修正逐级降低损失](/images/harq-realized-volatility/harq_qlike_bars.png)

模拟数据（潜在波动带杠杆效应 + RV 测量误差随波动放大）上的样本外结果：

| 模型 | 样本外 QLIKE | 样本外 RMSE(×10⁴) | 相对 HAR 提升 |
|---|---|---|---|
| HAR（基准） | 0.0578 | 1.341 | — |
| HAR-RS（拆半方差） | 0.0606 | 1.326 | RMSE −1.1% |
| **HARQ（RQ 修正）** | **0.0565** | **1.267** | **RMSE −5.5%** |
| HARQ-RS（合并） | 0.0606 | 1.325 | RMSE −1.2% |

结论很清楚：**在这个 DGP 里，测量误差修正(HARQ)带来的样本外增益最大**——RMSE 从 1.341 降到 1.267，QLIKE 也同步下降。半方差拆分单独用(HAR-RS)在 RMSE 上有小幅改善，但合并进 HARQ 后反而没有叠加收益，说明两条改进路径捕捉的信息在这份数据里高度重叠。

再把样本外的预测轨迹画出来，能看到 HARQ 相对 HAR 的行为差异：在波动突然跳升的段落，HARQ 跟随更快；在波动回落时，HARQ 的过冲更小——因为它对「噪声大的高波动日」自动打了折扣。

![样本外 HAR vs HARQ：HARQ 在波动突变段跟随更快、过冲更小](/images/harq-realized-volatility/harq_forecast_compare.png)

---

## 六、实现细节与落地关卡

**A. 实现细节**

- **数据口径**：本文用 5 分钟采样（每日 78 个区间）合成日内收益。真实落地务必先做**流动性过滤**——盘口稀薄的品种，5 分钟收益里塞满了微观结构噪声（买卖价差反弹），会把 RV 系统性抬高。常见做法是用「波动率签名图(signature plot)」选采样频率，或改用预平均(pre-averaging)、Two-Scale RV 等抗噪估计量。
- **信号-执行时点**：所有预测变量严格滞后一期（`shift(1)`），用 $t-1$ 及更早的信息预测第 $t$ 天的 RV，杜绝前视。
- **样本切分**：前 60% 样本内估计 OLS 系数，后 40% 完全样本外预测，系数不再更新。真实研究应进一步用滚动窗口/扩张窗口重估。
- **RQ 估计**：用 $RQ_t = \frac{M}{3}\sum r_{t,i}^4$，这是积分四次方矩的一致估计，作为 RV 测量误差方差的代理。

**B. 已知偏差**

- **合成数据的自洽性**：图表数据是自洽模拟——潜在波动被显式设定为「带杠杆效应 + 测量误差随波动放大」，因此 HARQ 占优是被数据生成过程「安排」好的。真实数据上 HARQ 的优势通常存在但幅度更小，且高度依赖品种、采样频率与样本期。
- **测量误差被简化**：真实的微观结构噪声远比「误差方差 ∝ RQ」复杂，包含价格离散化、买卖价差反弹、非同步交易等，单一 RQ 修正只能覆盖一部分。
- **跳变未显式分离**：本文用半方差隐式吸收了部分跳变信息，但没有做正式的 Bipower Variation 跳变检验；对跳变密集的品种，独立建模跳变分量往往还能再提升。
- **单资产、单频率**：没有测试多资产的横截面稳定性，也没测不同采样频率下结论是否翻转。

**C. 结果解读**

- **增益从哪来**：HARQ 的核心不是「加了个变量」，而是让模型在高波动日主动降低对昨日 RV 的信任——这本质是一种数据驱动的稳健化，因此它在**波动突变段**收益最明显，平稳段几乎无差别。
- **QLIKE 与 RMSE 的分歧**：HAR-RS 在 RMSE 上略优于 HAR，但 QLIKE 反而更差，说明拆半方差在「平均误差」上有改善，却在「避免低估风险」上没占到便宜——**选哪个模型，取决于你的损失函数是对称的还是偏向风控的**。
- **叠加不总是更好**：HARQ-RS 没能同时拿到两者的好处，提醒我们特征叠加要看信息是否正交；在真实数据上应当用 Diebold-Mariano 检验判断预测差异是否统计显著，而不是只看小数点。
- **适用边界**：HARQ 家族最适合**波动聚集强、偶有剧烈跳变**的品种（个股、加密货币）；对波动平稳的宽基指数，相对基准 HAR 的增量会明显收窄。

---

波动率预测的进步，往往不来自更复杂的模型，而来自**更诚实地对待数据**——承认 RV 有方向、有跳变、还有测量误差。把这三件事分别处理干净，一个线性回归就能打赢很多黑箱。这也是 HAR 家族至今仍是波动率预测事实基准的原因：它足够简单，简单到你能看清每一个系数在替你做什么决策。
