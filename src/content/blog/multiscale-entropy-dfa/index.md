---
title: "多尺度熵与 DFA：用粗粒化把不同时间尺度的不规则性分开"
description: "单尺度样本熵(SampEn)会把白噪声误判成最复杂序列——其实它的复杂度只存在于最细尺度。多尺度熵(MSE)用粗粒化把序列逐尺度平均掉、分尺度重算熵；DFA 用去趋势波动分析给出长程相关的标度指数 α。本文用纯 numpy 从零实现 SampEn/MSE/DFA，在白噪声、1/f 粉红噪声、周期+噪声三类序列上把尺度结构拆开，并诚实指出「MSE 必须用固定容忍半径」这一最易踩的坑（中阶）。"
publishDate: '2026-07-23'
tags:
  - 量化交易
  - 多尺度熵
  - 样本熵
  - 去趋势波动分析
  - 分形
  - 粗粒化
  - 复杂性度量
  - Python
language: Chinese
difficulty: intermediate
---

你有一段价格或收益序列，想量化它「有多不规则、多复杂」。直觉反应是算**样本熵（Sample Entropy, SampEn）**或近似熵——值越大越「乱」。但这个方法藏着一个反直觉的陷阱：**它对白噪声给的分数最高**。

白噪声是最「无结构」的序列，可 SampEn 却把它判成最复杂。为什么？因为 SampEn 只在**单一时间尺度**上比较相邻点的模式：白噪声在最小尺度上确实每个点都不重复、最「 unpredictably」，于是它拿到了最高熵。问题在于——白噪声的复杂度**只存在于最细的尺度**，你只要把序列「粗看」一点（比如每 5 个点取平均），它的不规则性就塌没了。

真正区分「白噪声 / 分形长记忆 / 周期驱动」的，是**不同时间尺度上的复杂度分布**。这就是 **多尺度熵（Multiscale Entropy, MSE, Costa 2002）** 和 **去趋势波动分析（DFA, Peng 1994）** 要解决的问题。本文用纯 numpy 从零实现 SampEn / MSE / DFA，在三类合成序列上把尺度结构拆开，并诚实指出「MSE 必须用固定容忍半径」这一最易踩的坑。所有图表均由下文 Python 真实计算，非占位图。

![三类序列形态对照：白噪声无结构、1/f 粉红噪声呈分形、周期+噪声由单一主导频率驱动](/images/multiscale-entropy-dfa/cover.png)

## 一、样本熵 SampEn：单尺度的「不规则性」

SampEn 的想法很朴素：把序列切成长度为 $m$ 的模板向量，看「再延长一步、仍然匹配」的比例。匹配越多说明越规律，匹配越少说明越复杂。写成公式：

$$
\text{SampEn}(m, r) = -\ln\frac{A}{B}
$$

- $B$：长度为 $m$ 的模板向量中，能找到另一个**不同**位置、且逐点距离 $< r$ 的配对总数；
- $A$：长度为 $m+1$ 的模板向量中，同样匹配的配对总数；
- $r$ 是容忍半径（抗噪声），默认取 $0.15\cdot\sigma$。

值越大 $\Rightarrow$ 模板越难重复 $\Rightarrow$ 越不规则。纯 numpy 实现：

```python
import numpy as np

def sampen(x, m=2, r=None):
    """Sample Entropy：值越大越不规则。"""
    x = np.asarray(x, float)
    if r is None:
        r = 0.15 * np.std(x)
    if np.std(x) == 0:
        return 0.0
    N = len(x)

    def count_m(mm):
        X = np.array([x[i:i + mm] for i in range(N - mm + 1)])
        cm = 0
        for i in range(len(X)):
            diff = np.max(np.abs(X - X[i]), axis=1)   # 与每个模板的逐点最大偏差
            cm += np.sum(diff < r) - 1                # 排除与自身比较
        return cm

    B = count_m(m)
    A = count_m(m + 1)
    if B <= 0 or A <= 0:
        return 0.0
    return float(-np.log(A / B))
```

## 二、多尺度熵 MSE：用粗粒化把尺度分开

MSE 的关键一步是**粗粒化（coarse-graining）**：把序列按窗口 $\tau$ 不重叠平均，得到一条更「粗」、更短的序列。$\tau$ 越大，越抹掉细尺度波动、越暴露粗尺度结构：

$$
y_j^{(\tau)} = \frac{1}{\tau}\sum_{i=(j-1)\tau+1}^{j\tau} x_i
$$

然后**在每个尺度 $\tau$ 上各算一次 SampEn**，得到一条 $MSE(\tau)$ 曲线。不同序列在这条曲线上的形状天差地别——这才是真正的信息：

```python
def coarse_grain(x, tau):
    """粗粒化：窗口 τ 不重叠平均，长度变为 N//τ。"""
    x = np.asarray(x, float)
    n = len(x) // tau
    if n == 0:
        return np.array([])
    return x[: n * tau].reshape(n, tau).mean(axis=1)

def mse(x, scales, m=2, r=None):
    """多尺度熵：**固定** r = 0.15·std(原序列)，跨尺度不变（关键！见第四节）。"""
    x = np.asarray(x, float)
    if r is None:
        r = 0.15 * np.std(x)
    out = []
    for tau in scales:
        cg = coarse_grain(x, tau)
        out.append(sampen(cg, m=m, r=r) if len(cg) > m + 1 else np.nan)
    return np.array(out)

scales = np.arange(1, 21)
mse_white  = mse(white,  scales)   # 白噪声
mse_pink   = mse(pink,   scales)   # 1/f 粉红噪声
mse_period = mse(periodic, scales) # 周期+噪声
```

在白噪声、1/f 粉红噪声、周期+噪声三类序列（各 2400 点）上算出的 MSE 曲线：

![多尺度熵曲线：白噪声随尺度单调塌缩，1/f 粉红噪声在粗尺度反而回升，周期序列单调下降最快](/images/multiscale-entropy-dfa/mse_curves.png)

实测数字（SampEn 在 $\tau=1,5,10,20$ 处）：

| 序列 | τ=1 | τ=5 | τ=10 | τ=20 |
|---|---|---|---|---|
| 白噪声 | 2.476 | 1.717 | 1.391 | **1.124** |
| 1/f 粉红噪声 | 1.987 | 1.891 | 1.731 | **2.565** |
| 周期+噪声 | 1.953 | 1.292 | 1.042 | **0.687** |

三条曲线的形状完全不同：

- **白噪声**：随尺度单调塌缩（2.48→1.12）。因为它没有跨尺度结构，粗化后迅速「变得可预测」——这就是单尺度熵误判它「最复杂」的真相：它的复杂度只活在 τ=1。
- **1/f 粉红噪声**：在粗尺度不降反升（τ=20 冲到 2.565）。它每个尺度都有结构（分形/长程相关），越粗越显出持续的相关性，所以分形过程才是真正「多尺度都复杂」的。
- **周期+噪声**：单调下降最快（1.95→0.69）。粗粒化会把周期平滑掉、只留趋势，于是迅速变规则。

## 三、DFA：给长程相关一个标度指数 α

MSE 看的是「复杂度的尺度分布」，而 **DFA（Detrended Fluctuation Analysis）** 直接问「序列有没有长程记忆、记忆多强」。它的三步：

1. 对去均值序列做累计和 $y_k = \sum_{i=1}^k (x_i - \bar x)$；
2. 把序列切成长度为 $s$ 的窗口，每个窗口用线性拟合去趋势，算残差的 RMS $F(s)$；
3. 对 $\log F(s) \sim \alpha \log s$ 回归，斜率就是标度指数 $\alpha$。

```python
def dfa(x, scales=None):
    y = np.cumsum(np.asarray(x, float) - np.mean(x))
    N = len(y)
    if scales is None:
        scales = np.unique(np.round(np.logspace(np.log10(10),
                                                 np.log10(N // 4), 18))).astype(int)
    Fs, used = [], []
    for s in scales:
        n_parts = N // s
        if n_parts < 2:
            continue
        fv = []
        for p in range(n_parts):
            seg = y[p * s:(p + 1) * s]
            coef = np.polyfit(np.arange(s), seg, 1)   # 线性去趋势
            fv.append(np.sqrt(np.mean((seg - np.polyval(coef, np.arange(s))) ** 2)))
        Fs.append(np.mean(fv)); used.append(s)
    alpha = float(np.polyfit(np.log(used), np.log(Fs), 1)[0])
    return np.array(used), np.array(Fs), alpha
```

三种序列的 $\alpha$：

| 序列 | α | 解读 |
|---|---|---|
| 白噪声 | 0.529 | ≈0.5，近似无记忆（随机游走） |
| 1/f 粉红噪声 | 1.022 | >0.5，强持续相关（分形长记忆） |
| 周期+噪声 | 0.566 | ≈0.5，周期被噪声主导，记忆弱 |

![DFA 标度：log F(s) 对 log s 的斜率 α 把白噪声(~0.5)与粉红噪声(~1.0)一刀切开](/images/multiscale-entropy-dfa/dfa_scaling.png)

$\alpha \approx 0.5$ 读作「无记忆/白噪声」， $\alpha > 0.5$ 读作「持续相关」（涨完还涨、跌完还跌，分形市场的指纹），$\alpha < 0.5$ 读作「反持续」（均值回复）。这正是 MSE 之外一个**互补**的尺度探针：MSE 量复杂度分布，DFA 量相关性结构。

## 四、最易踩的坑：MSE 必须用「固定」容忍半径

很多人实现 MSE 时，在每个粗化尺度上**重新**用该尺度的标准差算 $r=0.15\cdot\sigma(\text{粗化序列})$。这是错的。粗化后序列方差变小，$r$ 跟着变小，会把样本熵人为推高——于是白噪声的 MSE 曲线不降反升，和粉红噪声「撞型」，完全失去区分力。

正确做法是 **Costa 原论文的设定：$r$ 固定在原始序列标准差的 $0.15$ 倍，跨所有尺度不变**。这样白噪声在粗尺度才能真正「塌缩」，粉红噪声才能保持高位。本文所有数字都用固定 $r$ 计算。如果你看到某篇博客里白噪声 MSE 随尺度上升，那八成是踩了这个坑，而不是反直觉的真发现。

## 五、单尺度熵在金融里的无力：regime 切换

把 MSE 用到收益序列还有一个现实问题：单尺度 SampEn 对 regime 极不敏感。构造一段「平静—危机—平静—危机」切换的收益率（各 300 点，危机段波动是平静段的 3.5 倍），用滑动窗 SampEn(τ=1) 监测：

```
平静段平均 SampEn = 2.248
危机段平均 SampEn = 2.545
```

区分只有 0.3，而且单尺度熵在窗内很吵、边界模糊。MSE 的价值在于不盯一个点，而是看**跨尺度结构**是否整体迁移——平静市和危机市的「复杂度谱形状」差异，远大于单个 τ=1 的数字差异。这也是为什么做市场状态监测时，应报告整条 MSE 曲线（或它的统计量，如曲线下面积、斜率），而不是一个标量。

![单尺度熵在 regime 切换里很嘈杂，边界不清晰；MSE 的跨尺度结构才稳定](/images/multiscale-entropy-dfa/regime_single_scale.png)

## 六、五类真实陷阱（必看）

1. **容忍半径 r 随尺度重算**：上文详述，会让白噪声 MSE 不降反升、粉红噪声撞型。永远用固定 $r=0.15\sigma(\text{原序列})$。
2. **短序列 MSE 不可信**：粗化到 τ=20 时序列只剩 N/20 个点，$m+1$ 模板配对严重不足，尾部熵估计方差爆炸。本文 N=2400 尚可，实盘日频 250 天时 τ>5 就要打问号。
3. **DFA 最低尺度受噪声污染**：极小 $s$ 的 $F(s)$ 被微观结构噪声/采样误差主导，回归应截取 $s$ 的中间段（本文用 10~N/4），别把最小点也塞进线性拟合。
4. **周期序列的 α 会骗人**：纯周期的 $F(s)$ 在 $s$ 接近周期整数倍处出现周期性下凹，把周期误读成「长程记忆」。1/f 与周期的 α 接近时要用 MSE 形状区分，不能只看 α。
5. **MSE 量「不规则性」不量「方向」**：高 MSE 可能是趋势市（持续相关）也可能是噪声市（随机），必须配 DFA 的 α 一起看，否则分不清「有结构的复杂」和「无结构的乱」。

## 结语

多尺度熵 MSE 和 DFA 是同一件事的两面：**把序列的不规则性按时间尺度拆开**。SampEn 只在单尺度看「乱不乱」，会错把白噪声捧成最复杂；MSE 用粗粒化让白噪声在粗尺度塌缩、让粉红噪声在每个尺度都保持复杂、让周期序列迅速变规则；DFA 再补一刀，用标度指数 α 把「有没有长程记忆」量化成一个数（白噪声≈0.5、粉红噪声≈1.0）。实战要点就两条：**MSE 必须固定容忍半径**，以及**复杂度谱 + α 配合看，别用单标量**。把它和上一篇的排列熵放在一起——排列熵看「涨跌形状的结构」，MSE/DFA 看「跨尺度的结构」，三个工具合起来才是完整的复杂性度量箱。

完整可复跑代码与全部图表脚本已随本文生成（`gen_multiscale_entropy_dfa.py`），三类序列的 MSE 曲线、DFA 标度、regime 对照均由该脚本真实计算。
