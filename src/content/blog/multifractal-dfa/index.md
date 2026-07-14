---
title: "多重分形去趋势波动分析(MF-DFA)：用分形谱识别市场的多重尺度异象"
description: "MF-DFA 用多重分形谱 f(α) 量化「单分形 vs 多分形」：白噪声谱宽 Δα≈0.05（退化成点），p 模型级联 Δα≈0.81（宽谱）；滚动 Δα 还能当危机/波动聚集的无分布探针，附完整 Python 与六类真实陷阱（高阶）。"
publishDate: '2026-07-14'
tags:
  - 量化交易
  - 分形
  - MF-DFA
  - 多重分形谱
  - 波动率聚集
  - 危机探测
  - 非线性
  - Python
language: Chinese
difficulty: advanced
---

「市场是随机的吗？」如果答案是简单的「是」，那收益率就该是白噪声——每个尺度上长得一样，统计性质不随观察窗口变化。但任何做过日频数据的人都知道：市场有波动聚集(波动率高的时段扎堆)、有跳跃、有微观结构噪声。这些现象背后藏着一个更深的词——**多重分形(multifractal)**。

本文用**多重分形去趋势波动分析(MF-DFA, Kantelhardt et al. 2002)**，把「序列在多少个尺度上表现不同」量化成一条**多重分形谱 f(α)**。谱越宽，说明序列的奇异结构越丰富(危机、跳跃、波动聚集越多)；谱退化成一点，说明它是普通的单分形(如白噪声)。我们用长度 65536 的合成面板，把多分形序列与白噪声并排跑一遍，结论干净利落：

- 白噪声(单分形)：广义 Hurst 指数 h(q) 几乎不随 q 变，谱宽 **Δα≈0.05**；
- p 模型二项级联(多分形)：h(q) 明显弯折，谱宽 **Δα≈0.81**，α 落在 [0.645, 1.457]。

更进一步，把滚动窗口的 Δα 当探针，它能无分布地跟住波动率聚集——与「窗口内高波动态占比」相关 **0.38**，平静窗口 Δα 低到 0.10、最动荡窗口冲到 0.58。这意味着:**多重分形宽度本身就是一个危机/异常强度的度量**。

## 一、从 DFA 到 MF-DFA：为什么要「去趋势」

普通的相关性分析在金融序列上会翻车，因为金融收益有趋势、有周期、有长期漂移。直接算波动，趋势会把一切淹没。**去趋势波动分析(DFA)** 的核心思想很简单：

1. 把序列累积成轮廓线 `y(k) = Σ(x(i) − 均值)`；
2. 把轮廓线切成长度为 `s` 的若干段；
3. 每一段用低阶多项式(通常一阶直线)拟合、减去拟合值得到残差；
4. 残差的 RMS 就是该尺度下的波动函数 `F(s)`。

如果 `F(s) ~ s^H`，那么 `H` 就是 **Hurst 指数**：`H>0.5` 有长记忆(趋势持续)，`H<0.5` 有反持续，`H≈0.5` 接近随机游走。

**MF-DFA** 是在 DFA 上加一个「q 阶矩」：不是只算均方波动，而是对第 q 阶矩加权——

```python
import numpy as np

def mdfa(x, scales, q_list, order=1):
    """MF-DFA：返回每个尺度 s 下、每个 q 的波动 F_q(s)，以及标度指数 h(q)。"""
    x = np.asarray(x, float)
    N = len(x)
    y = np.cumsum(x - x.mean())          # 累积轮廓
    rev = y[::-1]
    F = {}
    for s in scales:
        if s >= N:
            continue
        n = N // s
        # 正向 + 反向两段，避免端点效应
        segs = np.concatenate(
            [y[v*s:(v+1)*s] for v in range(n)] +
            [rev[v*s:(v+1)*s] for v in range(n)]
        ).reshape(2*n, s)
        A = np.vstack([np.ones(s), np.arange(s)]).T     # 一阶去趋势: y = a + b·t
        beta = np.linalg.lstsq(A, segs.T, rcond=None)[0].T
        fit = beta @ A.T
        var_arr = ((segs - fit) ** 2).mean(1) + 1e-12    # 每段去趋势后方差
        fq = {}
        for q in q_list:
            if abs(q) < 1e-9:                            # q=0 用对数平均
                fq[q] = np.exp(0.5 * np.mean(np.log(var_arr)))
            else:
                fq[q] = (np.mean(var_arr ** (q/2.0))) ** (1.0/q)
        F[s] = fq
    # 对每个 q: log F_q(s) ~ h(q) · log s
    scales_arr = np.array(sorted(F.keys()), float)
    ls = np.log10(scales_arr)
    h = {}
    for q in q_list:
        fv = np.array([F[s][q] for s in scales_arr])
        lf = np.log10(fv)
        coef = np.linalg.lstsq(np.vstack([ls, np.ones_like(ls)]).T, lf, rcond=None)[0]
        h[q] = coef[0]
    return F, h
```

`q` 是矩阶：
- **q > 0** 放大大幅波动段(看「肥尾/危机」那一侧)；
- **q < 0** 放大小幅波动段(看「平静」那一侧)；
- **q = 0** 是几何平均，对应经典的 DFA 指数 H。

如果序列是**单分形**(如白噪声)，所有 q 给出的 `h(q)` 都差不多——因为只有一个「尺度法则」；如果是**多分形**，不同 q 看到不同尺度行为，`h(q)` 会明显弯折。

![MF-DFA 标度曲线：多分形序列的 F_q(s) 斜率随 q 明显变化，单分形序列近乎平行](/images/multifractal-dfa/mdfa_scaling.png)

## 二、广义 Hurst 指数 h(q)：多分形的第一张脸

把 `h(q)` 画出来，单分形和多分形的差别一目了然。我们用两条对照序列：

- **单分形基准**：iid 高斯白噪声，`H≈0.5`，理论上 `h(q)` 应为常数；
- **多分形序列**：p 模型二项乘性级联(binomial cascade)，天然带多重分形谱。

```python
def pmodel_cascade(L, p):
    """二项乘性级联：从 2 个盒子递归分裂 L 次，返回长度 2**(L+1) 的正测度。"""
    mu = np.ones(2)
    for _ in range(L):
        choose = rng.random(len(mu)) < 0.5
        left_w = np.where(choose, p, 1.0 - p)
        new = np.empty(len(mu) * 2)
        new[0::2] = mu * left_w
        new[1::2] = mu * (1.0 - left_w)
        mu = new
    return mu

# 单分形: 白噪声; 多分形: p 模型级联(p=0.65)
x_mono  = rng.standard_normal(65536); x_mono  -= x_mono.mean()
cascade = pmodel_cascade(15, 0.65)
x_multi = cascade.copy();           x_multi  -= x_multi.mean()

SCALES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
Q_LIST = list(range(-8, 9))
F_multi, h_multi = mdfa(x_multi,  SCALES, Q_LIST)
F_mono,  h_mono  = mdfa(x_mono,   SCALES, Q_LIST)
```

跑出来：

| 序列 | H(q=0) | h(+8)−h(−8) | 谱宽 Δα |
|---|---|---|---|
| 白噪声(单分形) | 0.504 | −0.008 | **0.047** |
| p 模型级联(多分形) | 1.068 | −0.601 | **0.812** |

白噪声的 `h(q)` 几乎是一条平线(`h(+8)−h(−8)≈0`)；级联序列的 `h(q)` 随 q 明显弯折，跨度约 −0.6。

![广义 Hurst 指数 h(q)：多分形越强，h(q) 对 q 的依赖越明显（跨度 −0.60）](/images/multifractal-dfa/mdfa_hq.png)

> 注：合成的 p 模型级联未严格归一为守恒测度，故 `H(q=0)≈1.07` 略高于理论值 1。本演示关注的是**多重分形宽度 Δα** 而非绝对 `H(0)`，真实市场数据需用真实收益序列、并注意归一化。

## 三、多重分形谱 f(α)：把 q 翻译成「奇异指数」

`h(q)` 好用但不够直观。通过 **Legendre 变换**，可以把 `q` 空间翻成 `α` 空间，得到更经典的**多重分形谱 f(α)**：

- `α` 是**奇异指数**(singularity strength)：局部波动有多剧烈；
- `f(α)` 是对应 `α` 的**分形维数**：有多少个尺度表现出这个剧烈程度。

```python
def spectrum(h, q_list):
    qs = np.array(sorted(q_list), float)
    hs = np.array([h[q] for q in qs])
    tau = qs * hs - 1.0            # 质量指数
    alpha = np.gradient(tau, qs)   # dτ/dq
    f = qs * alpha - tau           # Legendre 变换
    return qs, alpha, f, tau

qs_m, alpha_m, f_m, _ = spectrum(h_multi, Q_LIST)
qs_o, alpha_o, f_o, _ = spectrum(h_mono,  Q_LIST)
width_multi = alpha_m.max() - alpha_m.min()   # Δα ≈ 0.81
width_mono  = alpha_o.max() - alpha_o.min()   # Δα ≈ 0.05
```

谱的**宽度 Δα = α_max − α_min** 就是多重分形的「量尺」：
- **Δα≈0**：所有尺度行为一致 → 单分形(白噪声退化成谱上一个点)；
- **Δα 大**：小尺度(跳跃/危机)和大尺度(趋势)行为天差地别 → 多分形。

本例中级联序列 `α∈[0.645, 1.457]`，谱宽 **0.81**；白噪声谱宽仅 **0.05**，几乎是一个点。

![多重分形谱 f(α)：谱越宽，序列的奇异结构越丰富（危机/跳跃越多）](/images/multifractal-dfa/mdfa_spectrum.png)

## 四、滚动 Δα：一个无分布的「危机探针」

多重分形宽度最实用的地方，是它能当**波动率聚集/异常的无分布探针**。构造一段「平静—动荡」区制切换波动序列(低波动 σ=0.8%、高波动 σ=2.8%，各自强黏性，模拟波动率聚集)，在滚动窗口里算 Δα：

```python
def rolling_width(x, win, step, scales, q_list):
    out, centers = [], []
    s = max(scales)
    for start in range(0, len(x) - win + 1, step):
        seg = x[start:start+win]
        if len(seg) <= s * 4:
            continue
        _, h = mdfa(seg, scales, q_list)
        qs = np.array(sorted(q_list), float)
        hs = np.array([h[q] for q in qs])
        tau = qs * hs - 1.0
        alpha = np.gradient(tau, qs)
        out.append(alpha.max() - alpha.min())
        centers.append(start + win // 2)
    return np.array(centers), np.array(out)

centers, widths = rolling_width(x_regime, win=2000, step=500,
                                scales=[25, 50, 100, 200, 400], q_list=list(range(-4, 5)))
```

结果：滚动 Δα 均值 **0.339**，最平静窗口 **0.098**、最动荡窗口 **0.581**，与「窗口内高波动态占比」的相关系数为 **0.38**。换句话说——**市场越乱，多重分形谱越宽**。这不需要假设任何分布(不依赖正态、不依赖 GARCH)，是一个纯粹从尺度结构里长出来的异常指标。

![滚动 Δα 随波动率聚集增强：与高波动占比相关 0.38](/images/multifractal-dfa/mdfa_rolling.png)

## 五、六大真实陷阱

能算出漂亮的谱，不等于能拿来交易。MF-DFA 至少有六个坑：

1. **尺度区间选错**：`scales` 必须跨足够宽的幂次范围，且远小于序列长度。尺度太少(如只取 2~3 个点)，`h(q)` 的 log-log 回归自由度不足，噪声被当信号。
2. **端点效应**：只做正向分段会在序列末尾引入人为弯折，必须用「正向 + 反向」两段拼接(本文做法)或显式补零。
3. **非平稳污染**：MF-DFA 假设序列是「去均值后」的。若原始价格(非平稳)直接进去做累积，会污染 `h(q)`，必须先做对数收益或分数阶差分。
4. **长记忆伪装成多分形**：单纯的长记忆过程(单分形、H≠0.5)也会让 `h(q)` 轻微变化。判别要看 Δα 是否**显著大于**单分形基准的 Δα(本例 0.81 vs 0.05)，而不是只看 h(q) 是否完全水平。
5. **小样本偏差**：样本太短时，小 q(负矩)会放大尾部单点，Δα 被高估。实务上 q 范围常限制在 [−5, 5] 并做 Monte Carlo 显著性检验。
6. **过度解读 α 的经济含义**：Δα 宽说明「尺度异象多」，但不自动等于「可套利」。要把它接成交易信号，还需做样本外、walk-forward 与成本建模。

## 六、结语

MF-DFA 给我们一把尺子：**多重分形谱宽 Δα**，把「市场到底有几副面孔」从形容词变成数字。它最干净的结论是——单分形(白噪声)的谱退化成一点(Δα≈0.05)，多分形(级联/真实市场)的谱张开(Δα≈0.81)；而滚动 Δα 还能无分布地跟住波动率聚集，是一个天然的危机/异常探针。

对量化研究者而言，MF-DFA 的价值不在「预测涨跌」，而在**描述市场的尺度结构**：用它做 regime 检测、做波动异常的早期预警、或在因子/波动率模型里检查「我是不是把多分形结构当成了噪声」。它和前面聊过的 Hurst 指数(单分形长记忆)、 realized volatility、波动率聚集是同一家族的不同刻度。

**实盘落地清单(给研究者的提醒)**：
- 永远对对数收益(或已实现波动)做 MF-DFA，绝不对原始价格；
- 尺度区间取 `s` 跨 1~2 个数量级、且 `s < N/10`，正向+反向两段；
- 用白噪声/单分形蒙特卡洛建立 Δα 的显著性阈值，再谈「多分形显著」；
- 滚动 Δα 作预警指标时，先验证它在你目标品种上的危机样本里确实张开。

> 注：本文序列为自洽合成数据(白噪声 + p 模型级联 + 区制切换波动)，用于演示方法论；真实市场的多重分形来自波动率聚集 + 跳跃 + 微观结构噪声，落地需接入真实高频/日频数据并做样本外验证。
