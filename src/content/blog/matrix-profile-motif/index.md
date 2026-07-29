---
title: "Matrix Profile 模体发现：线性时间找出价格序列里的重复形态"
description: "Matrix Profile 是 Keogh 团队 2016 年提出的时间序列挖掘框架：对每个长度为 m 的子序列，记录它到全序列最近邻（排除自身邻域）的 z-normalized 欧氏距离。这一条曲线同时给出两样东西——低谷是模体（有孪生兄弟的重复形态），高峰是 Discord（独一无二的异常段）。3000 日合成价格实验完整复现：5 处植入的三重震荡模体被 MP 精确定位（top-1 对距离 1.06，位置 599/1899 对上植入点 600/1900），闪崩+chirp 扫频异常被 MP 最大值一击命中；MASS 算法用 FFT 把单条查询距离压到 O(n log n)，全对距离矩阵热图直观展示『MP = 每行最小值』。事件研究部分把模体匹配当交易信号跑了 60 日前向路径对比随机基线，并诚实交代三个坑：z-norm 会抹掉幅度信息、平凡匹配需要排除区、以及『历史形态重复出现』与『形态有预测力』是两个完全不同的命题。"
publishDate: '2026-07-30'
tags:
  - 量化交易
  - 时间序列挖掘
  - 模式识别
  - 异常检测
  - Python
language: Chinese
difficulty: advanced
---

## 一句话版本

**Matrix Profile（MP）**：对时间序列每个长度 $m$ 的滑动窗口，算出它与全序列所有其他窗口（排除自身附近）的最小 z-normalized 欧氏距离，得到一条与原序列几乎等长的「最近邻距离曲线」。**曲线的最低点 = 模体（motif，有孪生兄弟的重复形态）；最高点 = Discord（全序列独一无二的异常段）**。一次计算，同时回答「哪段历史重演过」和「哪段历史从未发生过」两个问题。

---

## 一、为什么技术形态挖掘需要一个「诚实的距离」

「历史会重演」是技术分析的根基假设，但传统形态识别（头肩顶、杯柄、双底）有两个致命软肋：形态定义靠人眼和硬编码规则，以及**没有任何统计意义上的可检索性**——你说这是杯柄，我说是噪声，谁也说服不了谁。

Matrix Profile 把问题反过来：不预设任何形态库，直接问数据——**这条序列里有没有两段子序列彼此异常相似？** 如果有，那就是数据自己承认的「重复形态」；如果某段子序列跟全序列任何别处都不像，那就是数据自己承认的「异常」。

形式化：给定序列 $x_1, \dots, x_T$ 和窗口长度 $m$，共有 $n = T - m + 1$ 个子序列。Matrix Profile 定义为

$$
MP[i] = \min_{|j - i| > m/2} \; d\big(z(x_{i:i+m}),\; z(x_{j:j+m})\big)
$$

其中 $z(\cdot)$ 是 z-normalization（减均值除标准差），$d$ 是欧氏距离，$|j-i| > m/2$ 是**排除区（exclusion zone）**——不允许一个窗口跟只挪了一两步的自己匹配（否则所有 MP 值都趋近于 0，即所谓平凡匹配 trivial match）。同时记录 $MPI[i] = \arg\min_j$，即最近邻的位置。

## 二、MASS：单条查询的 O(n log n) 距离计算

MP 的暴力计算是 $O(n^2 m)$，对日线级别尚可忍受，对 tick 级数据是灾难。第一块加速积木是 **MASS 算法**（Mueen's Algorithm for Similarity Search）：query 与全序列所有窗口的 z-norm 距离，可以整体用 FFT 卷积一次算完。

关键恒等式：query $q$ 已 z-normalized 时，它与窗口 $x_{j:j+m}$ 的 z-norm 距离平方为

$$
d^2(q, x_j) = 2m\left(1 - \frac{\langle q, x_{j:j+m}\rangle}{m \, \sigma_j}\right)
$$

滑动点积 $\langle q, x_{j:j+m}\rangle$ 对所有 $j$ 就是一次互相关——FFT 干这个是 $O(T \log T)$；滑动均值方差用累积和 $O(T)$ 搞定：

```python
import numpy as np
from numpy.fft import rfft, irfft

def znorm(a):
    s = a.std()
    return (a - a.mean()) / s if s > 1e-12 else np.zeros_like(a)

def mass(q, ts):
    """query q 对序列 ts 所有窗口的 z-norm 欧氏距离，O(T log T)"""
    m, T = len(q), len(ts)
    q = znorm(q)
    # 滑动均值 / 标准差（累积和技巧）
    cs, cs2 = np.cumsum(ts), np.cumsum(ts**2)
    s  = np.empty(T-m+1); s[0]  = cs[m-1];  s[1:]  = cs[m:]  - cs[:-m]
    s2 = np.empty(T-m+1); s2[0] = cs2[m-1]; s2[1:] = cs2[m:] - cs2[:-m]
    mu  = s / m
    sig = np.sqrt(np.maximum(s2/m - mu**2, 1e-18))
    # FFT 滑动点积
    L = 1
    while L < T + m: L *= 2
    dot = irfft(rfft(ts, L) * rfft(q[::-1], L), L)[m-1:T]
    d2 = 2*m*(1 - dot/(m*sig))
    return np.sqrt(np.maximum(d2, 0))
```

把 MASS 对每个窗口各调一次就得到完整 MP，总复杂度 $O(n \cdot T\log T)$——这正是 **STAMP** 算法。工业界主流的 **STOMP** 进一步利用相邻窗口点积的递推关系压到 $O(n^2)$（无 log 因子、常数极小），GPU 版 SCAMP 已经处理过十亿级点的序列。研究用途上 `stumpy` 库一行 `stumpy.stump(x, m)` 全部搞定；本文为了展示机制，用上面 30 行裸 numpy 实现。

## 三、实验：3000 日合成价格上的完整复现

实验设计刻意做成「有标准答案的考卷」：在纯随机游走（日波动 1%）里植入——

- **5 处相同模体**：长度 120 日的三重震荡形态（$0.12\sin(6\pi t)\sin(\pi t)$ 的 log 价格路径），叠加逐处不同的噪声（σ 从 0.004 到 0.012），位置 600 / 1100 / 1550 / 1900 / 2250；
- **1 处 Discord**：闪崩叠加 chirp 扫频震荡（频率随时间连续变化，保证全序列找不到近似邻居），位置 2500。

对 log 价格跑 $m = 120$ 的完整 MP：

```python
x = np.log(price)
w = 120
n = len(x) - w + 1
mp, mpi = np.full(n, np.inf), np.zeros(n, dtype=int)
excl = w // 2
for i in range(n):
    d = mass(x[i:i+w], x)
    d[max(0, i-excl) : i+excl+1] = np.inf   # 排除区
    j = d.argmin()
    mp[i], mpi[i] = d[j], j

motif_i, motif_j = mp.argmin(), mpi[mp.argmin()]   # 599, 1899
discord_i = mp.argmax()                             # 2555
```

### 结果：全部命中

![价格序列与 Matrix Profile 双面板](/images/matrix-profile-motif/mp-overview.png)

- **Top-1 模体对定位在 599 和 1899**——对上植入位置 600/1900，z-norm 距离 **1.06**，而全序列 MP 的典型值在 5~8 之间。注意价格图上肉眼几乎看不出这两段有什么特别（它们叠加在完全不同的价格水平和局部趋势上），但 MP 曲线在五处植入点都砸出了深谷；
- **Discord 定位在 2555**——落在植入异常区间 [2500, 2620] 内，MP 值 **12.07** 为全序列最大：这段闪崩+扫频形态在 3000 日里找不到任何近似邻居。

![模体对叠放与 Discord 形态](/images/matrix-profile-motif/motif-pair-overlay.png)

把两段模体 z-norm 后叠放，曲线几乎逐点重合——这就是距离 1.06 的直观含义。右图的 Discord 与模体形态毫无相似之处，其「独一无二」不是幅度大，而是**形状在全序列中没有第二次出现**。

![全对距离矩阵与 MP 的关系](/images/matrix-profile-motif/distance-matrix.png)

概念上 MP 就是 $n \times n$ 全对距离矩阵**每行的最小值**。热图（下采样）里模体对表现为对称的深色亮点簇，Discord 表现为整行整列都亮（跟谁都远）。MP 的空间优势正在于此：不存储 $O(n^2)$ 矩阵，只留每行最小值和位置，$O(n)$ 空间。

## 四、从挖掘到信号：模体匹配的事件研究

发现模体只是第一步，量化研究者真正关心的是：**形态出现后，价格接下来怎么走？** 把 top-1 模体作为 query，用 MASS 在全序列检索所有距离低于阈值的匹配（非重叠去重），对每次匹配结束时点做 60 日前向收益的事件研究，对照 400 个随机时点：

```python
d_all = mass(x[motif_i:motif_i+w], x)
hits = []                                    # 贪心非重叠去重
for i in np.argsort(d_all):
    if d_all[i] > 5.0: break
    if abs(i - motif_i) < excl: continue
    if all(abs(i - h) > w for h in hits):
        hits.append(i)

H = 60
fwd = np.array([x[h+w : h+w+H] - x[h+w] for h in sorted(hits) if h+w+H < len(x)])
```

![模体匹配后的前向收益路径](/images/matrix-profile-motif/motif-event-study.png)

检索命中 4 处植入位置（600 附近作为 query 自身被排除区滤掉），匹配后平均路径与随机基线的差异——在这个合成实验里——**在噪声带宽之内**。这是刻意为之的诚实设计：植入的模体形态与其后的收益方向没有任何构造上的因果关联，所以事件研究「应该」测不出显著差异。**「形态重复出现」与「形态有预测力」是两个完全独立的命题**——MP 只负责回答第一个，第二个必须靠事件研究 + 显著性检验单独证明。真实数据上大量「模体交易」回测的伪显著，来自把这两步偷偷合并成一步。

## 五、三个坑：把 MP 用在金融数据上的注意事项

**坑一：z-normalization 抹掉幅度。** z-norm 让「涨 1% 的震荡」和「涨 20% 的震荡」只要形状相同就完全等价。这对形态检索是特性，对风险管理是 bug——一段温和盘整可能匹配上一段剧烈崩盘的形状。金融应用常见处理：对 log 收益序列（而非价格）跑 MP、或改用非归一化欧氏距离 / MPdist，取决于你想让「相似」的语义是什么。**先想清楚语义，再选距离**。

**坑二：排除区与平凡匹配。** 不设排除区，每个窗口的最近邻就是平移一步的自己，MP 恒为近零。惯例排除区是 $m/2$，但金融序列的强自相关（尤其低波动盘整段）会让相隔不远的窗口天然相似，实务上排除区放宽到 $m$ 甚至更大更稳妥。本文实验里 query 自身位置附近 60 个窗口全部被硬性排除。

**坑三：窗口长度 $m$ 是唯一但关键的超参数。** $m$ 太短，所有窗口都像（z-norm 后短窗口的形状空间很小）；$m$ 太长，任何形态都独一无二。没有免费答案——`stumpy` 提供的 pan-matrix-profile（PMP）一次算完一段 $m$ 区间给你选；金融场景更常见的做法是让 $m$ 对齐经济含义：一个财报周期约 60 交易日、一个月约 21 日，让窗口语义先于统计。

## 六、MP 在量化管线里的合理位置

| 用途 | 用 MP 的哪一端 | 对接的下游 |
|---|---|---|
| 形态字典构建 | 低谷（top-k motifs） | 事件研究 / 形态特征工程 |
| 数据清洗 | 高峰（discords） | 剔除坏 tick / 拼接错误 / 乌龙指 |
| 状态切换检测 | FLUSS/FLOSS 弧曲线 | regime 划分、样本分层 |
| 跨序列检索 | AB-join（两条序列互查） | 「今天的走势最像历史哪一天」 |

其中第二行常被低估：**Discord 检测是 MP 在金融数据上最无争议的应用**——不需要任何「形态有预测力」的信仰，独一无二的子序列大概率是数据错误或真实的极端事件，两者都值得人工看一眼。而第四行的 AB-join（用今天的近期走势去全部历史里找最近邻）就是「历史相似日」类研究的严格化版本，比拍脑袋选「相似K线」可复现得多。

## 七、结论

- **机制层面**：MP 用一条最近邻距离曲线同时编码模体与异常，MASS 的 FFT 恒等式把单查询压到 $O(T\log T)$，STOMP/SCAMP 把全量压到 $O(n^2)$ 小常数，工程上已是解决了的问题；
- **实验层面**：3000 日合成序列上，5 处植入模体（含不同噪声水平）被 top-1 对精确命中（距离 1.06 vs 背景 5~8），chirp 型 Discord 被 MP 最大值一击定位，全对距离矩阵热图直观验证「MP = 每行最小值」；
- **信号层面**：模体检索的事件研究显示匹配后路径与随机基线无显著差异——这正是实验设计的本意：**MP 证明形态重复，不证明形态有用**，后者必须独立检验；
- **实战层面**：z-norm 的幅度盲区、排除区宽度、窗口长度三个坑各有明确处理方案；最稳妥的切入点是用 Discord 做数据质量监控，最有想象力的方向是 AB-join 做严格化的「历史相似期」检索。

## 参考文献

- Yeh, C.-C. M., Zhu, Y., Ulanova, L., et al. (2016). Matrix Profile I: All Pairs Similarity Joins for Time Series. *ICDM 2016*.
- Zhu, Y., Zimmerman, Z., Senobari, N. S., et al. (2016). Matrix Profile II: Exploiting a Novel Algorithm and GPUs to Break the One Hundred Million Barrier. *ICDM 2016*.
- Gharghabi, S., Ding, Y., Yeh, C.-C. M., et al. (2017). Matrix Profile VIII: Domain Agnostic Online Semantic Segmentation (FLUSS/FLOSS). *ICDM 2017*.
- Zimmerman, Z., Kamgar, K., Senobari, N. S., et al. (2019). Matrix Profile XIV: Scaling Time Series Motif Discovery with GPUs (SCAMP). *SoCC 2019*.
- Law, S. M. (2019). STUMPY: A Powerful and Scalable Python Library for Time Series Data Mining. *Journal of Open Source Software*, 4(39), 1504.
- Mueen, A., Keogh, E., Zhu, Q., Cash, S., & Westover, B. (2009). Exact Discovery of Time Series Motifs. *SDM 2009*.
