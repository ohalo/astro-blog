---
title: "共形预测 Conformal Prediction：给机器学习收益预测套上误差区间"
publishDate: '2026-07-12'
description: "机器学习给你一个点的收益预测，可你真正想知道的是『我有多不确定』。共形预测用校准集残差的经验分位，在不需要任何分布假设的前提下，给出有限样本可靠的预测区间。本文对比朴素正态区间、普通 Split-Conformal、波动率感知 Conformal 三种方法，配 4 张真实图表与完整 Python。"
tags:
  - 量化交易
  - 机器学习
  - 共形预测
  - Conformal Prediction
  - 不确定性量化
  - 预测区间
  - 风险管理
language: Chinese
difficulty: advanced
---

你的 LSTM / XGBoost / 神经网络给了你一个精准的数字：「明天这只股票预期收益 +0.8%」。然后呢？

然后你拿这个点的预测去下单，却对它**到底有多不准**一无所知。金融收益有两个让「误差区间」特别难做的特点：

1. **异方差**：波动率集群——平静时误差小，恐慌时误差能翻好几倍。固定宽度的区间天然不匹配。
2. **厚尾**：动不动来个几十个 sigma 的跳空。任何「假设正态」的区间都会系统性低估极端风险。

**共形预测（Conformal Prediction）** 就是为这种「我不想假设分布，但我要一个可靠区间」的场景而生的。它不要求残差服从任何分布，只依赖一个温和的「可交换性」假设，就能给出**边缘覆盖保证**：

$$P\big(Y \in \hat{C}(X)\big) \ge 1-\alpha$$

也就是说，长期来看，真实值落在你区间里的概率**至少**是 $1-\alpha$。本文用合成收益数据，对比三种区间构造法，所有图表与数字由文末代码真实生成。

## 一、三种区间，三种哲学

设模型在点 $x$ 上的预测为 $\hat{\mu}(x)$，校准集上算出的残差为 $R_i = |Y_i-\hat{\mu}(X_i)|$。

**1) 朴素正态区间**（baseline，常被滥用）

$$\hat{C}(x) = \big[\hat{\mu}(x) \pm z_{1-\alpha/2}\cdot\hat{\sigma}_{\text{train}}\big]$$

直接用训练残差标准差 $\hat{\sigma}$ 配正态分布分位。**隐含假设：残差同方差且正态**。金融里这两条都不成立。

**2) 普通 Split-Conformal（分割共形）**

把数据分成训练集和校准集，在校准集上取残差绝对值的经验的 $(1-\alpha)$ 分位 $\hat{q}$：

$$\hat{q} = \text{Quantile}\!\left(\{|R_i|\},\ \frac{\lceil(n_{\text{cal}}+1)(1-\alpha)\rceil}{n_{\text{cal}}}\right),\qquad \hat{C}(x)=\big[\hat{\mu}(x)\pm\hat{q}\big]$$

不假设分布，区间宽度由**校准集真实的误差分布**决定。

**3) 波动率感知 Conformal**

把残差先除以一个波动率估计 $\hat{\sigma}_t$（比如 EWMA 波动），对「标准化残差」做共形，再乘回当前波动：

$$\hat{C}(x)=\big[\hat{\mu}(x)\pm \hat{q}_{\text{std}}\cdot\hat{\sigma}_t\big]$$

这样区间宽度会**随市场波动伸缩**——平静时收紧给出精准区间，动荡时放宽兜住风险。

## 二、实验设定：平静训练，遇波动骤升

合成一段 4000 点的收益序列：

$$r_t = \mu(X_t) + \sigma_t\varepsilon_t,\qquad \varepsilon_t\sim t(\text{df}=3)\ (\text{厚尾})$$

- $\sigma_t$ 为 GARCH 式波动聚集；
- **关键设定**：前 50%（训练段）波动平静，后 50%（校准+测试段）波动结构性抬升——这正是真实世界里「模型训于平静市、遇波动骤升」的场景；
- 用 Ridge 回归做点预测（捕捉微小的可预测均值），残差主要由异方差+厚尾主导。

切分：训练 1998 / 校准 999 / 测试 1000。训练残差标准差 $\hat{\sigma}_{\text{train}}=1.614$（平静期尺度）。

## 三、边缘覆盖率：朴素区间系统性漏覆盖

在 80% / 90% / 95% 三种名义水平下，看测试集真实覆盖率：

| 名义水平 | 朴素正态 | 普通共形 | 波动率感知共形 |
|---|---|---|---|
| 80% | **0.693** | 0.763 | 0.797 |
| 90% | **0.774** | 0.888 | **0.897** |
| 95% | **0.828** | 0.946 | **0.944** |

![三种方法在 80/90/95% 名义水平下的边缘覆盖率](/images/conformal-prediction-finance/cp_coverage_levels.png)

结论一目了然：

- **朴素正态区间彻底失守**。名义 90% 它只覆盖了 **77.4%**——因为它用「平静训练期」的 $\hat{\sigma}=1.614$ 去套「高波动测试期」的误差，区间太窄，跳空一穿就破。名义 80% 更惨，只覆盖 69.3%。
- **普通共形命中目标**。90% 名义 → 实际 88.8%，95% → 94.6%，基本贴着名义线走。它不假设正态，阈值直接从校准集（同样是动荡期）的误差分布里取，所以厚尾和 regime shift 都扛得住。
- **波动率感知共形同样命中**（90% → 89.7%），还多了一项能力（见下节）。

> 这正是共形预测最被低估的价值：**它的覆盖率不依赖你对 residual 分布的任何信念**。朴素方法一旦「残差其实不是正态」或「训练和测试分布漂移」，就会 silently 漏覆盖；共形方法用数据自己说话。

## 四、区间宽度随波动伸缩：波动率感知共形的真价值

普通共形给出**恒定宽度**的区间。可市场是异方差的——恒定宽度意味着「平静时区间太宽（浪费精度）、动荡时区间太窄（兜不住）」。波动率感知共形把宽度绑到波动率上：

![区间半宽随当前波动上升：波动率感知共形斜率≈1.59，普通共形为常数](/images/conformal-prediction-finance/cp_width_vs_vol.png)

散点清楚地显示：波动率感知共形的半宽与当前波动显著正相关（拟合斜率 ≈ **1.59**），而普通共形是一条水平线。在 90% 名义下，普通共形平均半宽 3.754，波动率感知共形平均半宽 4.281——后者在动荡样本上自动加宽，在平静样本上自动收窄，**宽度本身就携带了风险信息**。

把测试段前 400 个点的真实收益、点预测、共形区间带画出来：

![共形预测区间覆盖真实收益：绝大多数落在带内](/images/conformal-prediction-finance/cp_intervals.png)

真实收益（蓝线）绝大多数时间都待在区间带（橙影）里，少数突破恰恰发生在波动放大的时刻——这正是区间该有的样子。

## 五、按波动分层的条件覆盖

更严格地，把测试集按当前波动分成五分位，看每一层内的覆盖率（名义 90%）：

![条件覆盖：普通共形在高波动层略漏覆盖，波动率感知共形各层更贴近 90%](/images/conformal-prediction-finance/cp_coverage_by_vol.png)

- 普通共形在**最高波动的 Q5 层覆盖率掉到 0.855**——恒定宽度在区间该宽的时候没宽够。
- 波动率感知共形让宽度跟着波动走，整体更贴合名义水平（其 Q5 偏宽到 0.96，说明对高波动样本宁可略宽也不漏；若配更精准的波动率模型还能进一步收紧）。

> 一句话：普通共形保住了**边际**覆盖（平均来看对），波动率感知共形进一步让覆盖在**不同波动状态**下都更稳——这对实盘更重要，因为你永远不知道下一周是平静还是暴风雨。

## 六、完整 Python 实现

下面代码自包含（用 `scikit-learn` 的 `Ridge`，不依赖专用共形库），复现三种区间与全部指标。

```python
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

# ---------- 0) 合成数据(平静训练 / 高波动测试) ----------
N = 4000; rng = np.random.default_rng(20240713)
h = np.zeros(N)
for t in range(1, N):
    h[t] = 0.90*h[t-1] + 0.25*rng.standard_normal()
L = 3
i1_abs = L + int((N-L)*0.50)
level = np.where(np.arange(N) >= i1_abs, 0.9, 0.0)   # 结构性波动抬升
vol = np.exp((h+level)/2.0)
eps = rng.standard_normal(N)/np.sqrt(rng.chisquare(3, N)/3.0)  # t(df=3) 厚尾
r = np.zeros(N)
for t in range(3, N):
    mu = 0.06*r[t-1] - 0.04*r[t-2] + 0.12*(vol[t-1]-vol.mean())
    r[t] = mu + vol[t]*eps[t]
# 特征 + 滚动波动代理(EWMA, 仅用历史)
rvol = np.zeros(N)
for t in range(1, N):
    rvol[t] = np.sqrt(0.94*rvol[t-1]**2 + 0.06*r[t-1]**2)
X = np.column_stack([r[L-1:N-1], r[L-2:N-2], r[L-3:N-3],
                     np.abs(r[L-1:N-1]), rvol[L-1:N-1]])
y = r[L:N]; rv = rvol[L:N]

# ---------- 1) 时序切分(无前视) ----------
n = len(y); i1, i2 = int(n*0.50), int(n*0.75)
Xtr, ytr = X[:i1], y[:i1]
Xcal, ycal, rvcal = X[i1:i2], y[i1:i2], rv[i1:i2]
Xte, yte, rvte   = X[i2:],  y[i2:],  rv[i2:]

model = Ridge(alpha=1.0).fit(Xtr, ytr)
p_cal = model.predict(Xcal); p_te = model.predict(Xte)
res_cal = np.abs(ycal - p_cal)
sigma_train = (ytr - model.predict(Xtr)).std(ddof=1)

# ---------- 2) 共形阈值 ----------
def conf_quantile(resids, alpha):
    k = min(max(int(np.ceil((len(resids)+1)*(1-alpha))), 1), len(resids))
    return np.sort(resids)[k-1]

def coverage(yv, lo, hi):
    return np.mean((yv >= lo) & (yv <= hi))

alpha = 0.10
z = stats.norm.ppf(1 - alpha/2)

# 朴素
lo_naive, hi_naive = p_te - z*sigma_train, p_te + z*sigma_train
# 普通共形
q_plain = conf_quantile(res_cal, alpha)
lo_plain, hi_plain = p_te - q_plain, p_te + q_plain
# 波动率感知共形
q_vol = conf_quantile(res_cal/rvcal, alpha)
lo_vol, hi_vol = p_te - q_vol*rvte, p_te + q_vol*rvte

print(f"朴素覆盖={coverage(yte,lo_naive,hi_naive):.3f}  "
      f"普通共形={coverage(yte,lo_plain,hi_plain):.3f}  "
      f"波动感知={coverage(yte,lo_vol,hi_vol):.3f}")
```

把 `alpha` 换成 0.05 / 0.20，你就能量出正文那张覆盖表；把 `rvte` 画到横轴、半宽画到纵轴，就是宽度-波动图。

## 七、五个必须知道的坑

1. **共形保证依赖「可交换性」**。校准集和测试集必须来自同一（联合）分布。如果你的测试是「训练之后」的真实未来、且发生了 regime shift，纯共形也会漏覆盖——本文正是用这个设定暴露了朴素方法的脆弱，而波动率感知共形靠「按波动缩放」部分缓解。真实场景里，记得**定期用近期数据刷新校准集**（滚动/在线共形）。
2. **共形给的是「边缘覆盖」，不是「条件覆盖」**。平均来看 90% 在区间内，但不保证「每一类样本都 90%」。普通共形在高波动子样本会掉队，这正是波动率感知共形要补的洞。
3. **波动率代理要「合法」**。共形缩放用的 $\hat{\sigma}_t$ 必须只用 $t$ 时刻之前的信息（EWMA / 滚动窗口 / 历史 GARCH），**不能偷看未来**，否则区间有前视偏差。
4. **分位取法要「对一点点」**。阈值用 $\lceil(n+1)(1-\alpha)\rceil/n$ 分位（而非简单 `np.quantile` 的默认插值），才能保证有限样本下 $P\ge 1-\alpha$ 的理论界。
5. **共形不改变点预测质量**。它只给你的预测「套上区间」，底层模型该烂还是烂。先把 $\hat{\mu}(x)$ 训好，共形负责把「不确定性」诚实地表达出来。

## 八、小结

机器学习的点预测告诉你「预期是多少」，共形预测告诉你「你有多不确定」——而且这种「不确定」**不靠任何分布假设**，靠的是校准集上实实在在的误差经验分布。在金融这种异方差+厚尾+regime shift 三重叠加的市场里，朴素正态区间会静悄悄地漏覆盖（本文 90% 名义只覆盖 77.4%）；普通 Split-Conformal 用数据说话，命中目标；波动率感知共形再进一步，让区间宽度随波动伸缩，把风险信息直接写进区间本身。

下次你准备把模型输出接进仓位管理之前，先给它套一层共形区间——你会惊讶地发现，原来「模型说 +0.8%」这句话，在不同波动环境下，该配的置信带能差出一倍多。

所有图表均由本文代码在合成数据上真实生成，数字可复现；落地时把第三步的合成数据换成你的真实特征与收益即可。
