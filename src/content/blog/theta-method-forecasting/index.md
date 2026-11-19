---
title: "Theta 方法预测：把时间序列拆成趋势与 theta 线，用两个简单预测的平均打赢复杂模型"
description: "M3 竞赛里一个反直觉的结论：把序列做经典分解，对季节调整后的数据分别拟合一条线性趋势线和一条指数平滑线，再把两个预测平均回去——这个叫 Theta 的方法击败了当时绝大多数复杂模型。本文用 168 个月合成序列复现：theta=0 线斜率估出 0.345（真值 0.35），theta=2 线 alpha=0.447；200 次重复里 Theta 拿到 160 次最优、平均 MASE 0.6429，优于 Drift 0.7678 和 SES 1.2512。但诚实披露：单次样本上 Holt 0.6378 略胜 Theta 0.7062，且纯随机游走序列上 Drift 0.3966 才是赢家、Theta 只有 0.4385。附完整 numpy 实现、四张真实计算图与适用性边界。"
publishDate: '2026-11-20'
tags:
  - 量化交易
  - 时间序列预测
  - Theta方法
  - 指数平滑
  - MASE
  - M3竞赛
  - Python
language: Chinese
difficulty: intermediate
---

做量化的人都熟悉一个困境：复杂模型在论文里漂亮，实盘上却常常输给朴素基线。时间序列预测领域有个著名的例子——2000 年的 M3 预测竞赛，最终赢家不是神经网络、不是 ARIMA 的某种变体，而是一个叫 **Theta 方法** 的简单办法。它的核心操作听起来近乎儿戏：把序列做一次经典分解，画两条所谓的「theta 线」，各预测一次，然后取平均。

本文把这个办法从原理到代码走一遍，并在一个真值已知的 168 个月合成序列上验证。结论会支持 Theta 的口碑，但也会给出它失手的地方——任何只讲胜率不讲边界的文章都不可信。

## 一、theta 线到底是什么

设 $x_t$ 是去掉季节项后的序列，对它求二阶差分 $\ddot{x}_t=x_t-2x_{t-1}+x_{t-2}$。**theta 线**定义为

$$\ddot{x}_t(\theta)=x_t-\theta\,\ddot{x}_t$$

直觉是：二阶差分度量了曲线的「弯曲程度」，$\theta$ 控制要抹掉多少弯曲。$\theta=0$ 时原封不动，但边界条件会让这条线退化成一条**线性回归线**——局部曲率被完全平滑掉，剩下纯粹的长期趋势；$\theta=2$ 时二阶差分的系数变成负的，曲率被放大，数学上等价于对 $x_t$ 做**简单指数平滑（SES）**，预测是一条水平的渐近线。

于是 Theta 方法就是：对季节调整后的序列，用 $\theta=0$ 的线性趋势外推得到 $F_0$，用 $\theta=2$ 的指数平滑得到 $F_2$，取 $F=\tfrac12(F_0+F_2)$，最后把季节项加回去。一个偏激进、一个偏保守，平均一下互相抵消误差。这个「组合预测」的思想才是它真正的威力——后来有人证明 Theta 等价于某个带约束的 ETS 状态空间模型，但那是一致性结论，不是它发明的原因。

原始论文里还有个细节常被简化掉：$\theta$ 本来是可以自由取值的参数族，$\theta=0$ 和 $\theta=2$ 只是两个特例。研究者发现这两个特例的组合在大量序列上意外地稳，干脆固定下来，反而省掉了调参的麻烦。换句话说，Theta 方法不是「找到了最优 theta」，而是「发现两个极端 theta 的平均效果最好」。这跟集成学习里「平均多个弱学习器」的逻辑如出一辙——单个预测总会有系统性偏差，但偏差方向不同的两个预测平均，方差能被显著压下来。

从偏差-方差分解看更清楚：$\theta=0$ 的线性外推低偏差、高方差，趋势估计一有扰动就 whole line 跟着转；$\theta=2$ 的指数平滑高偏差、低方差，它只认最近的水平，对长期趋势视而不见但几乎不会被噪声带跑。两者各占一半，正好在中频段取长补短。这也解释了为什么它在带趋势的季节序列上表现好——序列里恰好同时存在「该外推的慢变化」和「该平滑的快波动」两种成分。

## 二、实验设置

合成 168 个月（14 年）数据：水平 $100+0.35t$，叠加周期 12 个月的季节项 $4\sin(2\pi t/12)+1.8\cos(2\pi t/12)$，再加标准差 1.6 的高斯噪声。前 144 个月训练，后 24 个月做样本外预测。图 1 是序列本身和经典分解出的趋势、季节项。

![合成序列与经典分解结果](/images/theta-method-forecasting/01_decomposition.png)

基线选了五个：Naive（最后一值）、Seasonal naive（去年同月）、Drift（首尾连线外推）、SES、Holt 双参数指数平滑。评价指标用 **MASE**——用训练集上季节朴素预测的平均绝对误差做分母，MASE<1 才说明比季节朴素好。这个尺度的好处是无量纲，跨序列可比。

选 MASE 而不是 MSE 有三层考虑。第一，MSE 对量纲敏感，序列水平从 100 换成 1000 数值就变，MASE 除以自身尺度后没有这个问题。第二，预测界早有结论：绝对误差类指标（MAE/MASE）比平方误差更抗离群点，而金融时间序列偏偏离群点一堆。第三，也是实践中最关键的一点——MASE 的分母是「不用模型也能达到的误差」，所以 1.0 是一条天然基线：任何 MASE 大于 1 的模型，都还不如直接抄去年同月，那它就没有任何上线价值。这个可解释性在跟业务方沟通时比一个孤零零的 RMSE 数值有力得多。

Holt 基线的实现里我踩过一个坑，值得单独记一笔。最初把趋势初值设成 $y_1-y_0$，结果只用两个带噪声的点定方向，优化器又把平滑参数顶到边界，24 个月外推直接飘到 191（真值约 158），MASE 高达 4.18。改成用前 24 个月做线性回归给趋势定初值、并对平滑参数做多点起步搜索后，同一条序列的 Holt MASE 回落到 0.6378。这个修正很重要：如果拿一个调坏的 Holt 去对比，Theta 会赢得毫无意义；公平的比较必须先让每个基线都发挥正常水平。

## 三、单条序列的结果

Theta 的两条线拟合得很准：$\theta=0$ 的斜率估出 0.3451，真值 0.35；$\theta=2$ 的平滑参数 $\alpha=0.447$。SES 单独拟合的 $\alpha$ 却高达 0.9046——因为原序列带强趋势，SES 只能靠一个接近 1 的 $\alpha$ 去追着最新值跑，反过来说明分解这一步有多重要。两条线的形态见图 2：一条 steadily 上扬，一条几乎水平。

![季节调整序列上的两条 theta 线](/images/theta-method-forecasting/02_theta_lines.png)

24 个月预测集上的 MASE：Holt 0.6378 最好，Drift 0.6574 次之，Theta 0.7062 第三，Naive 1.0062，SES 1.1090，Seasonal naive 1.6980。**这里必须诚实：单看这一条序列，Theta 并没有赢。** 它的线性加平滑组合稳稳跑赢朴素方法，但被拟合到好参数的 Holt 小胜。各方法的预测曲线见图 3。

![六种方法 24 个月预测与真实值对比](/images/theta-method-forecasting/03_forecasts.png)

为什么 Holt 这次这么强？因为真值的趋势是严格线性的，Holt 的双参数结构正好对口。Theta 的 $\theta=2$ 分支是水平渐近线，会低估一个持续上升的趋势，于是被 Drift 和 Holt 反超。这不是 bug，是 Theta 的设计取舍——它用一个水平分量换稳健性。

## 四、200 次重复的胜率

单条序列的排名有偶然性，于是我固定数据生成机制、只换噪声种子，重复 200 次，看谁最常拿下最低 MASE。结果：Theta 赢 160 次，Drift 赢 34 次，Holt 只有 6 次，其余模型挂零。平均 MASE：Theta 0.6429、Drift 0.7678、Seasonal naive 1.4974、SES 1.2512、Naive 1.2374。Theta 的优势从单次样本上的「第三名」变成了统计意义上的稳定领先。图 4 左panel 是单次 MASE，右 panel 是 200 次胜场。

![MASE 对比与 200 次重复胜场统计](/images/theta-method-forecasting/04_mase_wins.png)

Holt 的表现尤其值得玩味：平均 MASE 2.0436 看着很糟，中位数却有 0.8684。这说明它在大多数重复里表现尚可，但少数几次参数初值没调好就给出离谱外推，把均值拖垮——指数平滑类的趋势外推在数据短、噪声大时尾巴极厚。Theta 因为把趋势估计交给线性回归、把平滑参数交给一维搜索，反而避开了这个坑。这大概就是 M3 结论的本质：不是预测得更准，而是**更少灾难性失手**。

## 五、边界：Theta 不是万能的

我把数据换成纯随机游走（100 起步、每步标准差 1.2 的累积和）再测一次：Drift 以 0.3966 的 MASE 完胜，Theta 0.4385 屈居第二，Naive 0.5401、SES 0.5402、Holt 0.5299，Seasonal naive 0.9400 垫底。没有趋势、没有季节的序列上，Theta 那套分解纯属浪费，连简单的 Drift 都打不过。

所以适用性很清楚：Theta 擅长**带趋势和季节、但趋势不会突然断裂**的中短期序列，这正是大量经济与业务指标的样子，也是它在 M3 上闪光的原因。对于结构性断点频繁的市场价格、或者纯随机游走的资产收益，别指望它。另外本文的季节项是用经典移动平均分解提出来的，对奇数周期、结尾几个月的边界处理会失真；生产环境建议改用 STL 分解再套同样的 theta 组合。

## 六、完整代码

```python
import numpy as np
from scipy.optimize import minimize_scalar

def ses_forecast(series, alpha, h):
    level = series[0]
    for v in series[1:]:
        level = alpha * v + (1 - alpha) * level
    return np.full(h, level)

def fit_ses_alpha(series):
    def loss(a):
        level, e = series[0], 0.0
        for v in series[1:]:
            e += (v - level) ** 2
            level = a * v + (1 - a) * level
        return e
    return float(minimize_scalar(loss, bounds=(0.001, 0.999), method="bounded").x)

def decompose(series, period):
    n = len(series); half = period // 2
    left, right = half, period - 1 - half
    pad = np.pad(series, (left, right), mode="edge")
    trend = np.convolve(pad, np.ones(period)/period, mode="valid")
    detrended = series - trend
    seas = np.array([np.mean(detrended[k::period]) for k in range(period)])
    seas -= seas.mean()
    seas_full = np.tile(seas, n//period + 1)[:n]
    return series - seas_full, seas_full, trend

def theta_forecast(series, period, h):
    sa, seas_full, _ = decompose(series, period)
    n = len(sa); tt = np.arange(n)
    slope, intercept = np.polyfit(tt, sa, 1)           # theta = 0
    f0 = np.array([intercept + slope*(n+i) for i in range(h)])
    a2 = fit_ses_alpha(sa)                              # theta = 2
    f2 = ses_forecast(sa, a2, h)
    f = 0.5*(f0 + f2)
    last_seas = seas_full[-period:]
    return f + np.tile(last_seas, h//period + 1)[:h], f0, f2, a2, slope

# MASE：以训练集季节朴素误差为尺度
def mase(f, actual, train, period):
    scale = np.mean(np.abs(train[period:] - train[:-period]))
    return float(np.mean(np.abs(f - actual)) / scale)

t = np.arange(168)
y = 100 + 0.35*t + 4*np.sin(2*np.pi*t/12) + 1.8*np.cos(2*np.pi*t/12) \
    + np.random.randn(168)*1.6
tr, te = y[:144], y[144:]
pred, f0, f2, a2, slope = theta_forecast(tr, 12, 24)
print("slope", slope, "alpha2", a2, "MASE", mase(pred, te, tr, 12))
```

四张图依次是：原始序列与分解结果、两条 theta 线、六种方法 24 个月预测对比、MASE 柱状图与 200 次胜场统计，见 `public/images/theta-method-forecasting/`。

## 七、小结

Theta 方法给了一个很好的提醒：预测 accuracy 的提升，很多时候来自**误差的稳健化**而非模型的复杂化。把序列拆成趋势和季节、让两条特性互补的预测各出一半力，这种结构在 M3 和本文的实验里都证明了自己。但它不是免费午餐——纯随机游走上输给 Drift、趋势断裂时线性外推会失效、经典分解有边界失真。把它当成基线工具箱里性价比最高的一件，而不是万能武器，才是正确的用法。
