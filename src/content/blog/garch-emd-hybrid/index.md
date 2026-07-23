---
title: "GARCH-EMD 混合波动：用经验模态分解剥离趋势再做 GARCH"
description: "直接对收益拟 GARCH(1,1)，慢变的结构性波动漂移会污染估计——把长记忆假象塞进 β，持续性虚高到 0.99。混合思路借 Engle-Rangel Spline-GARCH 的乘性分解 σ²_t=g_t·h_t：先用 EMD（经验模态分解）从对数平方收益里剥出慢变方差水平 g_t，对标准化残差再拟合短期 GARCH 因子 h_t。纯 numpy 从零实现 EMD sifting（cubic-spline 包络+端点 clamp）与 GARCH MLE（网格+坐标下降），在「乘性成分」合成收益（慢正弦+陡峭 regime 跳变 × 单位均值 GARCH）上实测：EMD-GARCH 把持续性从纯 GARCH 的 0.99 压回 0.95（剔除伪长记忆），但样本外一步方差预测 QLIKE=−6.57 略输纯 GARCH 的 −6.68——分解修正了参数偏误，却没换来预测精度碾压。附「剥太多 IMF 把 GARCH 聚集也当趋势吸走」与「EMD 端点样条外推不稳」两处真实翻车，拆穿分解万能/持续性越低越好/混合必胜/EMD 因果/端点无害五类陷阱（中阶）。"
publishDate: '2026-07-24'
tags:
  - 量化交易
  - 波动率建模
  - GARCH
  - 经验模态分解
  - EMD
  - 时间序列
  - Python
language: Chinese
difficulty: intermediate
cover: "/images/garch-emd-hybrid/cover.png"
---

先说结论：**GARCH(1,1) 直接吃原始收益时，最容易被「慢变的结构性波动」骗到——它会把宏观/regime 造成的低频波动漂移，误当成 GARCH 自身的长记忆，把持续性 α+β 拉到 0.99 以上的「近单位根」区间。** 本文用 EMD（经验模态分解）先把这段慢波动剥掉，再对干净残差拟合 GARCH，实测持续性从 0.99 回落到 0.95。但——**参数偏误被修正了，样本外预测精度却没有跟着碾压**，这才是诚实的结果。

![原始收益、EMD 剥出的慢波动包络、以及两种模型的波动率估计](/images/garch-emd-hybrid/cover.png)

## 一、问题：GARCH 的持续性为什么总是虚高

GARCH(1,1) 的条件方差递推是：

$$h_t = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 h_{t-1}$$

其中 $\alpha_1+\beta_1$ 叫**持续性**——它衡量一次波动冲击要多久才衰减。实盘里 GARCH 拟合出来的持续性动辄 0.98~0.995，意味着冲击几乎不衰减、波动有「长记忆」。

但这里有个陷阱。Diebold（1986）和 Lamoureux-Lastrapes（1990）早就指出：**如果真实的波动水平本身在缓慢漂移（比如 2015 股灾、2020 疫情、2022 加息，每段的「常态波动」根本不是同一个数），而你硬用一个常数无条件方差 $\alpha_0/(1-\alpha_1-\beta_1)$ 去套整段样本，GARCH 只能靠把 $\beta_1$ 顶到接近 1 来「假装」自己能跟上水平漂移。** 这个高持续性是**结构性遗漏变量导致的假象**，不是真的波动长记忆。

## 二、乘性分解：把波动拆成「慢水平 × 快聚集」

修正思路来自 Engle-Rangel（2008）的 **Spline-GARCH**：把条件方差写成两个部件相乘

$$\sigma_t^2 = g_t \cdot h_t$$

- $g_t$：**低频、慢变的方差水平**——对应 regime、宏观周期、结构变化。它决定「这段时间的常态波动是高还是低」。
- $h_t$：**单位无条件均值的短期 GARCH 因子**——只负责波动聚集（今天大跌，明天大概率也不平静）。

Spline-GARCH 用样条函数拟 $g_t$。本文换个更「数据驱动」的零件：用 **EMD** 来提这个 $g_t$。

### 为什么用 EMD

EMD（Huang et al. 1998）不预设任何基函数，直接靠信号自身的极值点，把序列自适应地拆成一组 **IMF（本征模态函数）**：从高频到低频排列，最后剩一个单调残余。天然适合「我不知道慢波动长什么样，但我想把最低频的那几层拿出来当趋势」。

具体做法：

1. 对**对数平方收益** $\log(r_t^2+\epsilon)$ 做 EMD（对数化把方差的乘性结构变成加性，EMD 才好拆）；
2. 取**最低频的两层（最低频 IMF + 残余）**求和 = 慢变的 log 方差；
3. 指数还原、对齐到无条件方差尺度，得到 $\hat g_t$；
4. 用 $r_t/\sqrt{\hat g_t}$ 得到**标准化残差**，对它拟合短期 GARCH $h_t$。

上面 cover 图第 ① 栏能看到：EMD 剥出的慢波动包络（蓝）几乎贴合真实慢波动水平（黑虚线），包括第 1300 天那个陡峭的 regime 跳变。第 ② 栏是剥掉慢方差后的标准化残差——波动聚集还在，但整体水平被拉平了。

## 三、从零实现（纯 numpy）

### 3.1 EMD 的 sifting

EMD 的核心是「筛」（sifting）：反复用三次样条连上极大值/极小值包络，减去包络均值，直到剩下的分量是一个合格 IMF。

```python
from scipy.interpolate import CubicSpline
import numpy as np

def _extrema(x):
    n = len(x)
    mx = [i for i in range(1, n-1) if x[i] > x[i-1] and x[i] >= x[i+1]]
    mn = [i for i in range(1, n-1) if x[i] < x[i-1] and x[i] <= x[i+1]]
    return np.array(mx), np.array(mn)

def _envelope(idx, val, n):
    # 端点 clamp: 复制首尾极值, 抑制样条外推爆炸(关键防坑)
    xi = np.concatenate(([0], idx, [n-1]))
    yi = np.concatenate(([val[0]], val, [val[-1]]))
    xi, uniq = np.unique(xi, return_index=True); yi = yi[uniq]
    return CubicSpline(xi, yi)(np.arange(n))

def emd(x, max_imf=8, max_sift=60):
    x = np.asarray(x, float); n = len(x)
    imfs = []; res = x.copy()
    for _ in range(max_imf):
        h = res.copy()
        for _s in range(max_sift):
            mx, mn = _extrema(h)
            if len(mx) < 2 or len(mn) < 2:
                break
            mean_env = 0.5*(_envelope(mx, h[mx], n) + _envelope(mn, h[mn], n))
            h_new = h - mean_env
            if np.mean(mean_env**2) < 1e-10*(np.mean(h**2)+1e-12):
                h = h_new; break
            h = h_new
        imfs.append(h); res = res - h
        mx, mn = _extrema(res)
        if len(mx) + len(mn) < 3:
            break
    imfs.append(res)  # 残余(单调趋势)
    return imfs
```

**端点 clamp 是防坑关键**：三次样条在数据两端会外推，如果不夹住首尾，包络会在端点飞出去，导致最低频 IMF 的首尾严重失真。这个坑后面第五节会用实测数据证明。

分解出来的完整 IMF 谱长这样——高频在上、低频在下、最后是单调残余，我们取最低两层（蓝色）当慢结构：

![对数平方收益的 EMD 完整 IMF 分解谱](/images/garch-emd-hybrid/imfs.png)

### 3.2 GARCH(1,1) 的 MLE

用高斯条件似然，网格初始化 + 坐标下降细化（避开框架，纯 numpy）：

```python
def garch_negll(params, r):
    a0, a1, b1 = params
    if a0 <= 0 or a1 < 0 or b1 < 0 or a1+b1 >= 0.999:
        return 1e10
    n = len(r); h = np.empty(n); h[0] = np.var(r); ll = 0.0
    for i in range(1, n):
        h[i] = a0 + a1*r[i-1]**2 + b1*h[i-1]
        if h[i] <= 0: return 1e10
        ll += 0.5*(np.log(2*np.pi) + np.log(h[i]) + r[i]**2/h[i])
    return ll

def fit_garch(r):
    var_r = np.var(r); best=None; bestll=1e18
    for a1 in [0.03,0.05,0.08,0.12,0.18]:
        for b1 in [0.80,0.86,0.90,0.94]:
            if a1+b1 >= 0.999: continue
            a0 = var_r*(1-a1-b1)
            ll = garch_negll((a0,a1,b1), r)
            if ll < bestll: bestll=ll; best=[a0,a1,b1]
    for _ in range(200):  # 坐标下降细化
        improved=False
        for k,step in [(0,best[0]*0.1),(1,0.005),(2,0.005)]:
            for d in (step,-step):
                cand=best.copy(); cand[k]+=d
                if cand[0]<=0 or cand[1]<0 or cand[2]<0 or cand[1]+cand[2]>=0.999: continue
                ll=garch_negll(cand,r)
                if ll<bestll-1e-9: bestll=ll; best=cand; improved=True
        if not improved: break
    return np.array(best), bestll
```

### 3.3 混合拼装

```python
# 1) EMD 从对数平方收益提慢变方差
proxy = np.log(ret**2 + 1e-8)
imfs = emd(proxy, max_imf=8)
low_freq = imfs[-1] + imfs[-2]                 # 最低两层 = 慢变 log 方差
g_hat = np.exp(low_freq)
g_hat = g_hat / np.mean(g_hat) * np.var(ret)   # 对齐无条件方差尺度

# 2) 标准化残差 -> 短期 GARCH
ret_std = ret / np.sqrt(g_hat)
p_pure, _ = fit_garch(ret[:split])             # 纯 GARCH: 吃原始收益
p_hyb,  _ = fit_garch(ret_std[:split])         # EMD-GARCH: 吃标准化残差
h_hyb = garch_filter(p_hyb, ret_std)

# 3) 乘性重构 sigma_t = sqrt(g_t) * h_t
sig_hyb = np.sqrt(g_hat) * (h_hyb / np.sqrt(np.mean(h_hyb**2)))
```

## 四、实测结果：参数偏误被修好了，预测精度没有

合成数据是**乘性成分**结构：慢变方差水平 $g_t$（缓慢正弦 + 第 1300 天一次陡峭 regime 跳变）× 单位均值的 GARCH(1,1) 因子 $h_t$。2000 个交易日，前 70% 训练、后 30% 样本外。

### 参数对比

| 模型 | $\alpha_1$ | $\beta_1$ | **持续性 $\alpha_1+\beta_1$** |
|---|---|---|---|
| 纯 GARCH（吃原始收益） | 0.05 | 0.94 | **0.990** |
| EMD-GARCH（吃标准化残差） | 0.05 | 0.90 | **0.950** |

**这是本文最干净的一个结果**：纯 GARCH 被慢变水平骗到，把持续性顶到 0.99（伪长记忆）；EMD 先把慢水平剥走，短期 GARCH 只剩下真正的聚集，持续性回落到 0.95。参数偏误确实被修正了。

### 样本外一步方差预测

以真实瞬时方差 $\sigma_t^2$ 为标的，用 QLIKE（波动预测的标准损失）和 MSE 评估后 30%：

| 模型 | 样本外 QLIKE ↓ | 样本外 MSE(σ²) ↓ |
|---|---|---|
| EWMA(λ=0.94) | **−6.682** | **8.70e-08** |
| 纯 GARCH | −6.680 | 9.22e-08 |
| EMD-GARCH 混合 | −6.566 | 1.08e-07 |
| 滚动 std(63) | −6.665 | 1.27e-07 |

**诚实披露：EMD-GARCH 在样本外预测上并没有赢，反而略微落后纯 GARCH 和 EWMA。** 为什么？两个原因：

1. **参数「更真」不等于预测「更准」**。持续性 0.99 虽然是偏误，但在这个数据上，高持续性反而让纯 GARCH 的方差预测更「黏」、更平滑，恰好接近真实的慢变水平。EMD-GARCH 剥离了慢水平后，$\hat g_t$ 的估计误差（尤其端点）会直接进入最终方差，引入额外噪声。
2. **EWMA 是个强基线**。RiskMetrics 的 EWMA 没有任何参数拟合，靠 λ=0.94 的指数遗忘，在「慢变水平 + 聚集」的数据上天然自适应，很难被打败。

![四种方法的样本外方差预测：MSE 与 QLIKE 对比](/images/garch-emd-hybrid/oos_metrics.png)

这不是失败，而是波动率建模里反复出现的真相：**分解方法的价值在「解释」和「参数无偏」，不一定在「点预测精度」。** 如果你的目的是估计真实的波动持续性、判断冲击到底有多长记忆（比如做期权定价、风险资本计量），EMD-GARCH 的 0.95 比纯 GARCH 的 0.99 更可信；如果你只要一个样本外方差数字喂给风控，EWMA 可能就够了。

## 五、两处真实翻车

### 翻车一：剥太多 IMF，把 GARCH 聚集也当趋势吸走

「剥离最低几层」的「几」是超参数。我扫描了剥离层数，看重构的 $\hat g_t$ 与真实慢方差的相关：

![左：剥离层数敏感性；右：EMD 端点效应](/images/garch-emd-hybrid/pitfalls.png)

左图很清楚：剥离层数从 1 增到 2~3 时，$\hat g_t$ 与真实慢方差的相关升到最高；**再往上剥（4 层、5 层……），相关反而掉下来——因为 EMD 开始把属于短期 GARCH 的波动聚集也当成「趋势」吸进低频层，慢方差被高频信息污染。** 剥离层数不是越多越好，存在一个甜点。

### 翻车二：EMD 端点样条外推不稳

右图是 EMD 剥出的慢波动与真实慢波动的偏差。**中段贴合很好，但首尾 60 天（红色区）误差明显放大**——这就是三次样条包络在数据边界外推导致的端点效应。即便我做了端点 clamp，也只能缓解不能消除。

这个坑在**在线滚动预测**里尤其致命：EMD 每次都要重算，而你最关心的「最新一天」恰好落在误差最大的右端点。**这也是 EMD-GARCH 样本外没赢的隐藏原因之一**——测试段的 $\hat g_t$ 右端点估计本身就带偏。

## 六、结果解读与适用边界

**收益归因**：EMD-GARCH 的核心贡献是把持续性从 0.99 修回 0.95，这是「参数无偏」的胜利，量级上是 4 个百分点的持续性差异——对期权 term structure、方差风险溢价这类对持续性敏感的应用，这个差异不小。

**什么时候用它**：
- ✅ 你怀疑样本里有 regime/结构变化，且**关心波动持续性的真实值**（风险资本、长期期权）；
- ✅ 你要做**波动分解叙事**：多少来自慢变宏观、多少来自短期聚集；
- ❌ 你只要一个样本外方差点预测喂风控 —— 先试 EWMA/HAR，别高估分解的增益；
- ❌ 你在做**严格因果的在线预测** —— EMD 端点效应会咬你，考虑用因果版分解（如在线 EMD 或滚动 HP 滤波）。

**五个必须拆穿的陷阱**：

1. **「分解一定能提升预测」**：错。本文实测 EMD-GARCH 样本外 QLIKE 输给纯 GARCH 和 EWMA。分解修的是参数偏误，不保证点预测。
2. **「持续性越低越好」**：错。0.95 比 0.99 更接近真实数据生成过程，但在这个数据上 0.99 的「黏性」反而让点预测更平滑。低不等于优，要看目的。
3. **「混合模型必胜」**：错。多一个 EMD 环节，就多一层估计误差（尤其端点），可能把修参数的收益又吐回去。
4. **「EMD 是因果的」**：错。EMD 的样条包络用到全序列极值，天然是**双向**的。做样本外预测必须只在训练段拟合、或用因果变体，否则就是 look-ahead。
5. **「端点效应无所谓」**：错。滚动预测里你最关心的最新点恰在误差最大的右端点，端点效应直接污染你的实时波动估计。

---

**方法论一句话**：EMD-GARCH 是「先剥慢水平、再拟快聚集」的乘性分解。它擅长把被结构变化污染的持续性估计拉回真值，但别指望它在样本外点预测上碾压——诚实的量化研究，要分清「参数无偏」和「预测精度」是两件不同的事。

> 本文所有数值来自纯 numpy 从零实现的 EMD + GARCH MLE，在乘性成分合成数据上跑出，代码逻辑与图表一一对应。合成数据非真实行情，结论用于方法演示，不构成任何投资建议。
