---
title: "ADF 与 KPSS 联合检验：把是否平稳从猜变成判据"
description: "单位根检验不是猜出来的：ADF 原假设=有单位根(非平稳)，KPSS 原假设=平稳，两个方向互补。用 statsmodels 在随机游走/AR(1)/分数差分三类已知序列上验证联合判定矩阵，附完整 Python 与四类真实陷阱（中阶）。"
publishDate: '2026-07-19'
tags:
  - 量化交易
  - 时间序列
  - 单位根检验
  - ADF
  - KPSS
  - 平稳性
  - Python
language: Chinese
difficulty: intermediate
---

做量化最隐蔽的一个坑：**你拿去回归、拿去做特征、拿去做协整的序列，到底平不平稳？** 不平稳的序列（单位根过程）会有「伪回归」——两个本来毫无关系的随机游走，回归出来 R² 能高得吓人，t 统计量却完全不可信。传统做法靠肉眼看「轨迹像不像水平波动」，这等于在猜。ADF 和 KPSS 把这件事变成**有 p 值、有临界值的统计判据**，而且两个检验方向相反、恰好互补。

核心直觉一句话：**ADF 的原假设是「有单位根（非平稳）」，KPSS 的原假设是「平稳」**。一个要「拒绝非平稳」，一个要「拒绝平稳」。两个一起跑，四种结果组合成一张判定矩阵，把「平稳 / 趋势平稳 / 差分平稳 I(1) / 长记忆」清清楚楚分开。

## 一、为什么两个检验要一起用

只用 ADF 有盲区：ADF 不拒绝，可能是「真的平稳」，也可能是「样本太短检验力不足」——它**从不证明平稳，只证伪非平稳**。只用 KPSS 同样有盲区：KPSS 不拒绝，可能是「真平稳」，也可能是数据恰好在某个尺度上波动。两个方向相反，**一个证伪非平稳、一个证伪平稳**，合起来才能互相兜底。

```python
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

def test(seq):
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(seq, autolag="AIC")
    kpss_stat, kpss_p, kpss_lag, kpss_crit = kpss(seq, regression="c", nlags="auto")
    return adf_stat, adf_p, adf_crit, kpss_stat, kpss_p, kpss_crit
```

`adfuller` 返回 ADF 统计量、p 值、各显著水平的临界值；`kpss` 返回 KPSS 统计量、p 值、滞后、临界值。注意 KPSS 的 p 值是从临界值**反查**的表，**统计量 > 临界值才拒绝**（和 ADF 方向相反，下面图里会画清楚）。

## 二、三类已知状态的序列：肉眼难分，判据能分

用 Davies-Harte 生成三个「答案已知」的序列：

```python
rng = np.random.default_rng(20260719)
N = 1500
# A. 随机游走（单位根，非平稳）
rw = np.cumsum(rng.standard_normal(N))
# B. AR(1) 平稳 φ=0.85
ar = np.zeros(N)
for t in range(1, N):
    ar[t] = 0.85 * ar[t-1] + rng.standard_normal()
# C. 分数差分长记忆 FI(d=0.45)：0<d<1，均值回复但非平稳
d = 0.45
w = np.zeros(N + 50)
for k in range(len(w)):
    w[k] = (1.0 if k == 0 else np.prod([(d + (k - j) - 1) / (k - j) for j in range(k)]))
white = rng.standard_normal(N + 50) * 0.25
frac = np.convolve(white, w, mode="full")[:N]
frac = frac - frac.mean()
```

![三序列轨迹：A(随机游走) 与 C(分数差分) 肉眼几乎一样，但判据能区分](/images/adf-kpss-stationarity/three_series.png)

看上图——**A（随机游走）和 C（长记忆）的轨迹肉眼几乎分不出来**，都在「乱走」。但一个是非平稳的单位根，一个是均值回复但非平稳的长记忆过程。这就是不能靠肉眼的原因。

## 三、联合检验结果与判定矩阵

跑完两类检验，对每个序列做 4 区判定：

```python
def classify(adf_p, kpss_p):
    adf_rej = adf_p < 0.05       # 拒绝单位根 -> 平稳证据
    kpss_rej = kpss_p < 0.05    # 拒绝平稳 -> 非平稳证据
    if (not adf_rej) and (not kpss_rej):
        return "平稳"
    if adf_rej and (not kpss_rej):
        return "平稳"
    if (not adf_rej) and kpss_rej:
        return "差分平稳 I(1) / 长记忆"
    return "趋势平稳（需去趋势）"
```

实跑结果（N=1500，种子 20260719）：

| 序列 | ADF stat / p | KPSS stat / p | 联合判定 |
|---|---|---|---|
| 随机游走 I(1) | −1.96 / 0.31 | 1.97 / 0.01 | 差分平稳 I(1) |
| AR(1) φ=0.85 | −11.93 / 0.00 | 0.16 / 0.10 | 平稳 |
| FI(d=0.45) | −8.43 / 0.00 | 0.70 / 0.01 | 长记忆（KPSS 拒平稳）|

![ADF 检验：统计量 < 临界值(虚线) 则拒绝单位根](/images/adf-kpss-stationarity/adf_test.png)

![KPSS 检验：统计量 > 临界值(虚线) 则拒绝平稳](/images/adf-kpss-stationarity/kpss_test.png)

图 2、图 3 把两个检验的统计量对齐临界值（虚线为 5% 水平）。注意方向：**ADF 统计量越负越拒绝非平稳；KPSS 统计量越正越拒绝平稳**——两者一个往左、一个往右，正是互补的结构。

![联合判定矩阵：两个检验互补，避免单向误判](/images/adf-kpss-stationarity/joint_matrix.png)

图 4 是四象限判定矩阵。最右下角（ADF 拒 + KPSS 拒）是**趋势平稳**——序列有单位根式的漂移，但去趋势后平稳，必须**先去趋势**再做回归，否则仍是伪回归。最左下角（都不拒）是**真正平稳**。

## 四、四类真实陷阱（必须说清）

**① ADF 的滞后阶数 autolag 选择会影响结论。** `autolag="AIC"` 自动定阶，但在强序列相关或小样本下，AIC 选的阶可能不够，导致 ADF 检验力下降、把非平稳误判成平稳。实盘里建议同时报告 AIC / BIC / 固定滞后三种设定，结论一致才放心。

**② KPSS 的 regression 参数决定它检验「水平平稳」还是「趋势平稳」。** `regression="c"` 检验水平平稳（围绕常数波动），`regression="ct"` 才允许趋势。如果你用 `"c"` 去测一个带趋势的序列，它会**错误地拒绝平稳**。本文用 `"c"` 演示水平平稳；带漂移的序列要换 `"ct"`。

**③ 长记忆 FI(d) 是 ADF/KPSS 的灰色地带。** d∈(0,0.5) 时序列均值回复但非平稳，ADF 会拒绝单位根（像平稳），KPSS 会拒绝平稳（像非平稳）——两者矛盾，落进矩阵的「矛盾信号」角。这类序列**差分并不能简单变平稳**，要用分数差分 `(1-L)^d`，不是整数阶 `(1-L)`。

**④ 不平稳就去回归 = 伪回归。** 两个随机游走回归，R² 可以是 0.9、系数「显著」，但全是幻觉。做截面/时序回归前，**先对每个变量跑联合检验**，非平稳的要么差分到平稳、要么用协整框架，绝不能直接拿水平值回归。

## 五、真实落地路径

本文用自洽合成数据演示方法（三类序列答案已知，便于核对判据）。真实落地：① 用 `yfinance` / `westock-data` 拉价格或因子序列，先做对数化；② 对每个序列跑 `adfuller` + `kpss(regression="ct")`；③ 按联合矩阵分类，非平稳的走差分或分数差分，趋势平稳的走去趋势；④ 进回归/协整/特征工程前，**把「已验证平稳」作为硬门槛**写进 pipeline。平稳性检验不是可有可无的装饰，它是后面所有统计推断的**地基**——地基松了，上面的 alpha 全是沙上楼。

> 诚实声明：本文合成序列刻意设成「答案已知」，是为了让判据的判定逻辑可核对；真实金融序列常有结构断点、变波动、短样本，ADF/KPSS 的检验力会下降，结论要谨慎。本文数字演示的是**检验方法本身**，不代表任何真实资产的历史平稳性结论。

## 六、小结：平稳性是统计推断的地基

把整件事收一下：ADF 和 KPSS 不是两套重复的检验，而是**两个方向相反的证伪器**——一个负责「证伪非平稳」，一个负责「证伪平稳」。单独用任何一个，都可能把「检验力不足」误读成「证明平稳/非平稳」；两个一起跑，落进联合判定矩阵的四个角，结论才站得住。

更重要的是：平稳性检验不是文章末尾的装饰，它是你后面所有回归、协整、特征工程、假设检验的**地基**。地基松了，上面算出来的 alpha、IC、t 统计量，统统是沙上楼。任何量化 pipeline 在把一串数喂进模型之前，都该先问一句：**这串数，到底平不平稳？**
