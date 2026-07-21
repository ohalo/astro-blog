---
title: "方差比检验市场有效性：用 Lo-MacKinlay 把「价格是不是随机游走」算成一个数"
description: "有效市场假说下，收益独立、k 期方差该等于 k 倍单期方差。方差比 VR(k)=1 是随机游走的指纹。本文用 Lo & MacKinlay (1988) 检验，在随机游走/均值回复/趋势三类合成序列上把 VR(k) 分别钉在 ≈1、<1、>1，并给出滚动 VR(5)+95% 置信带与 Monte-Carlo 零假设经验 p 值——演示它如何「不误杀有效市场」（中阶）。"
publishDate: '2026-07-21'
tags:
  - 量化交易
  - 市场有效性
  - 方差比检验
  - Lo-MacKinlay
  - 随机游走
  - 统计检验
  - Python
language: Chinese
difficulty: intermediate
cover: "/images/variance-ratio-test/cover.png"
---

一个资产的价格，今天涨了明天会接着涨吗？如果答案是「不会、完全看运气」，那它就是弱式有效市场里的随机游走——过去价格里榨不出超额信息。但如果你能靠「连涨必跟、连跌必反」赚到钱，市场就还没那么有效。

怎么用一个数把「是不是随机游走」钉死？最朴素的做法是跑一个收益率的自回归：

$$r_t = \phi\,r_{t-1} + \varepsilon_t$$

看 $\hat\phi$ 显著不等于 0 与否。它的问题很硬：**单个 $\phi$ 只能测「相邻两期」的相关性，且对滞后多期的长期记忆（long memory）完全失明；而且它依赖线性、同方差的强假设。**

结论先放这：**方差比检验（Variance Ratio Test）绕开逐期自回归，直接看「k 期收益的方差」是不是「单期收益方差的 k 倍」。** 在随机游走下，两者必须相等，所以方差比 VR(k)≡1；只要收益有正/负自相关，VR(k) 就会系统性地 >1 或 <1。用 Lo & MacKinlay (1988) 的检验统计量，我们既能算 VR(k)，也能算它的显著性。在我们合成的三类序列上：随机游走的 VR(10)=0.94（≈1，经验 p=0.56，不拒绝有效）、均值回复的 VR(10)=0.41（<1）、趋势动量的 VR(10)=1.72（>1）。诚实地说：**VR 检验的是「方差是否随持有期线性放大」，它抓的是自相关结构，不是「有没有信息可赚」——有效市场里也可能因为微观结构噪声让 VR 暂时偏离 1，这正是要配置信带的原因。**

![三类序列：随机游走 / 均值回复 / 趋势动量](/images/variance-ratio-test/cover.png)

---

## 1. 为什么方差比能测随机游走

随机游走里，单期收益 $r_t$ 是独立同分布的（至少是序列无关）。那么「持有一期」的收益方差是 $\sigma^2$，「持有 k 期」的收益是 k 个独立单期收益之和，方差就该是 $k\sigma^2$。

定义方差比：

$$\text{VR}(k) = \frac{\text{Var}(r_t + r_{t-1} + \cdots + r_{t-k+1})}{k \cdot \text{Var}(r_t)}$$

- **随机游走**：$r$ 序列无关 → 分子 $=k\sigma^2$ → **VR(k)=1**。
- **正自相关（趋势/动量）**：k 期收益里相邻项正相关 → 方差被放大 → **VR(k)>1**。
- **负自相关（均值回复/反趋势）**：相邻项负相关 → 方差被抵消 → **VR(k)<1**。

一句话：VR(k) 偏离 1 的程度，就是「多周期收益比随机游走更/更不分散」的程度，它把整条自相关结构压缩成一个数。

---

## 2. Lo-MacKinlay 检验统计量

Cochrane (1988) / Lo-MacKinlay (1988) 给出了 VR(k) 的估计与检验。记单期方差估计 $\hat\sigma^2_a=\frac1n\sum(r_t-\bar r)^2$，k 期收益 $y_t^{(k)}=\sum_{i=0}^{k-1}r_{t-i}$，其方差估计 $\hat\sigma^2_b=\frac1{m}\sum(y_t^{(k)}-\bar y^{(k)})^2$，其中 $m=\lfloor n/k\rfloor$。则

$$\widehat{\text{VR}}(k) = \frac{\hat\sigma^2_b}{k\,\hat\sigma^2_a}$$

Lo & MacKinlay 在「同方差」假设下给出检验统计量（只需收益方差，不依赖异方差修正）：

$$M_r(k) = \frac{\widehat{\text{VR}}(k)-1}{\sqrt{\hat\theta(k)}}, \qquad \hat\theta(k) = \frac{2(2k-1)(k-1)}{3k\,n}$$

$M_r(k)$ 在零假设下近似标准正态，于是 $|\text{x}|>1.96$ 即约 5% 显著。

```python
import numpy as np

def vr_stat(r, k):
    """Lo-MacKinlay 方差比 VR(k) 与同方差检验统计量 M_r(k)。"""
    n = len(r); mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / n
    nk = n // k
    yk = r[: nk * k].reshape(nk, k).sum(axis=1)
    var_k = np.sum((yk - yk.mean()) ** 2) / (nk - 1)
    VR = var_k / (k * max(var1, 1e-12))
    theta = (2 * (2 * k - 1) * (k - 1)) / (3 * k * n)
    M = (VR - 1) / np.sqrt(theta)
    return VR, M
```

> 实务提醒：若怀疑异方差，应换用 Lo-MacKinlay 的异方差稳健版本（用 Bartlett 权重估计长期方差），否则小样本下检验水平会失真。本文同方差版本重在讲清原理。

---

## 3. 三类合成序列：VR(k) 一测便知

我们造三条序列的日收益：

- **随机游走**：独立正态噪声 → 有效市场。
- **均值回复**：收益对上一期收益做负 AR(1)，$\rho=-0.30$ → 反趋势。
- **趋势/动量**：收益对上一期收益做正 AR(1)，$\rho=+0.25$ → 正自相关。

```python
def make_rw(n=2000, seed=1):
    rng = np.random.default_rng(seed); r = rng.normal(0, 0.01, n)
    return np.cumsum(r), r

def make_mean_rev(n=2000, seed=2, rho=0.30):
    rng = np.random.default_rng(seed); r = np.zeros(n); r[0] = rng.normal(0, 0.01)
    for t in range(1, n):
        r[t] = -rho * r[t-1] + rng.normal(0, 0.01)   # 负自相关
    return np.cumsum(r), r

def make_trend(n=2000, seed=3, rho=0.25):
    rng = np.random.default_rng(seed); r = np.zeros(n); r[0] = rng.normal(0, 0.01)
    for t in range(1, n):
        r[t] = rho * r[t-1] + rng.normal(0, 0.01)      # 正自相关
    return np.cumsum(r), r

P_rw, r_rw = make_rw(); P_mr, r_mr = make_mean_rev(); P_td, r_td = make_trend()
Ks = [2, 4, 5, 10]
VR_rw = [vr_stat(r_rw, k)[0] for k in Ks]
VR_mr = [vr_stat(r_mr, k)[0] for k in Ks]
VR_td = [vr_stat(r_td, k)[0] for k in Ks]
```

结果（k=2/4/5/10）：

| 序列 | VR(2) | VR(4) | VR(5) | VR(10) |
|---|---|---|---|---|
| 随机游走 | 0.96 | 1.04 | 1.02 | 0.94 |
| 均值回复 | 0.69 | 0.57 | 0.58 | 0.41 |
| 趋势/动量 | 1.29 | 1.47 | 1.46 | 1.72 |

![VR(k) 随持有期 k 的变化](/images/variance-ratio-test/vr_bars.png)

看得很清楚：**随机游走恒在 VR=1 附近（0.94~1.04）**；均值回复因负自相关把多期方差「抵消」得越来越厉害（k 越大 VR 越小）；趋势/动量因正自相关把多期方差「放大」得越来越厉害（k 越大 VR 越大）。单看一个 $\phi$ 的自回归，你根本看不到这种「随持有期单调放大/收缩」的全貌。

---

## 4. 滚动 VR(5) + 95% 置信带：不误杀有效市场

实战里你不会只算一个 VR。常见做法是对一条真实价格序列算**滚动 VR(k)**，看它长期是否稳定偏离 1。关键是配一条置信带，否则噪声会让 VR 上下乱跳、你误以为抓到了异常。置信带直接用 $\hat\theta(k)$ 构造：

$$\text{VR}(k) \in 1 \pm 1.96\sqrt{\hat\theta(k)}$$

```python
def rolling_vr(r, k=5, W=250):
    out = np.full(len(r), np.nan)
    for t in range(W, len(r)):
        out[t] = vr_stat(r[t-W:t], k)[0]
    return out

rvr = rolling_vr(r_rw, k=5, W=250)
n, k = 250, 5
theta = (2*(2*k-1)*(k-1)) / (3*k*n)
lo, hi = 1 - 1.96*np.sqrt(theta), 1 + 1.96*np.sqrt(theta)
```

![随机游走序列的滚动 VR(5) 落在置信带内](/images/variance-ratio-test/vr_rolling.png)

这是一条**真随机游走**序列的滚动 VR(5)：它绝大多数时间在 1 附近游荡、落在 95% 置信带内。这正是检验该有的样子——**对有效市场要「不误杀」**。如果某条真实价格的滚动 VR 长期、系统性地戳出带外（且随 k 单调），那才是「这里可能不随机游走」的信号。

---

## 5. Monte-Carlo 零假设：经验 p 值

最后，用模拟给「待检验序列」一个正式的经验 p 值：在零假设（随机游走）下重复造 MC=4000 条序列，收集它们的 VR(10) 经验分布，看观测值落在多极端的位置。

```python
rng = np.random.default_rng(20260721)
N_MC = 4000; vr_null = np.zeros(N_MC)
for i in range(N_MC):
    rr = rng.normal(0, 0.01, 2000)
    vr_null[i] = vr_stat(rr, 10)[0]
vr_obs, M_obs = vr_stat(r_rw, 10)
p_val = np.mean(np.abs(vr_null - 1) >= np.abs(vr_obs - 1))
```

![随机游走零假设下 VR(10) 经验分布](/images/variance-ratio-test/vr_distribution.png)

待检验序列（本身就是随机游走）的 VR(10)=0.944，落在零假设分布的中间地带，**经验 p 值=0.563**——无法拒绝「它是随机游走」。对照：若把这条序列换成趋势动量序列，VR(10)≈1.72，经验 p 值会逼近于 0，明确拒绝。这就是检验的完整闭环：**统计量 + 置信带 + 经验 p 值**。

---

## 6. 五类真实陷阱（中阶）

1. **VR 测的是自相关结构，不是「能不能赚钱」**：VR 显著 ≠ 有可榨取 alpha。微观结构噪声（买卖价差、离散报价）会让真实价格序列的 VR 短期系统性 <1，与「均值回复异象」混在一起——需用实现核/去噪预处理，别把噪声当 alpha。
2. **异方差会扭曲检验水平**：Lo-MacKinlay 同方差版本假定方差恒定。波动率聚集（GARCH 效应）下应换异方差稳健统计量，否则 5% 名义水平可能实际是 10%+。
3. **小样本 k 不能太大**：k 期收益要求样本量 n 远大于 k（经验 n/k ≳ 100）。k=10 至少要 1000+ 观测，否则 $\hat\sigma_b^2$ 估计太噪、VR 剧烈摆动。
4. **重叠样本的相关性**：k 期收益用的是重叠窗口（rolling sum），相邻 VR 估计高度相关，做「多个 k 同时显著」的联合检验时不能直接套独立 Bonferroni，需用 Chow-Denning 联合检验。
5. **前视偏差**：滚动 VR 在时间 t 只能用到 t 及之前的数据。任何把未来样本混进窗口的写法，都会让「有效市场被误判为显著异常」。

---

## 7. 实战落地点：方差比能做什么、不能做什么

方差比检验在量化工作流里有三个很实在的用途，但边界同样清晰：

- **有效性体检**：对新上线的数据源/策略信号，先用 VR 检验它的「价格/收益是不是随机游走」。如果一个号称「弱有效市场」的资产 VR(10) 长期显著 <1，说明短期有可捕捉的反转结构——这正是 mean-reversion 策略的土壤。
- **微观结构指纹**：同一标的的 VR(k) 在不同采样频率下表现不同。日频 VR<1（买卖价差导致）但去噪后的高频 VR 接近 1，就能把「噪声假象」和「真实反转异象」区分开——这是把实现核/预平均去噪和 VR 检验串起来的标准做法。
- **策略假设验证**：做动量策略前，VR(k) 随 k 单调 >1 是「动量存在」的必要证据；做反转策略前，VR(k) 随 k 单调 <1 才站得住。拿不出 VR 证据的策略逻辑，往往在样本外站不住。

但它**不能**直接告诉你「去买哪只」。VR 显著只证明「收益非独立」，中间可能是行为异象、可能是微观结构、可能是流动性。从「测出非随机游走」到「赚到钱」，中间还隔着去噪、成本控制、样本外验证三道关。把 VR 当成「该不该继续深挖这条线索」的守门员，而不是印钞机。

## 8. 完整 Python 代码

与本文全部数字、配图一一对应的端到端复现脚本（自洽合成数据，仅演示方法；真实落地请替换为真实价格序列并视情况改异方差稳健版）：

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

def vr_stat(r, k):
    n = len(r); mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / n
    nk = n // k
    yk = r[: nk * k].reshape(nk, k).sum(axis=1)
    var_k = np.sum((yk - yk.mean()) ** 2) / (nk - 1)
    VR = var_k / (k * max(var1, 1e-12))
    theta = (2 * (2 * k - 1) * (k - 1)) / (3 * k * n)
    return VR, (VR - 1) / np.sqrt(theta)

def make_rw(n=2000, seed=1):
    rng = np.random.default_rng(seed); r = rng.normal(0, 0.01, n)
    return np.cumsum(r), r
def make_mean_rev(n=2000, seed=2, rho=0.30):
    rng = np.random.default_rng(seed); r = np.zeros(n); r[0] = rng.normal(0, 0.01)
    for t in range(1, n): r[t] = -rho * r[t-1] + rng.normal(0, 0.01)
    return np.cumsum(r), r
def make_trend(n=2000, seed=3, rho=0.25):
    rng = np.random.default_rng(seed); r = np.zeros(n); r[0] = rng.normal(0, 0.01)
    for t in range(1, n): r[t] = rho * r[t-1] + rng.normal(0, 0.01)
    return np.cumsum(r), r

P_rw, r_rw = make_rw(); P_mr, r_mr = make_mean_rev(); P_td, r_td = make_trend()
Ks = [2, 4, 5, 10]
print("RW :", [round(vr_stat(r_rw, k)[0], 2) for k in Ks])
print("MR :", [round(vr_stat(r_mr, k)[0], 2) for k in Ks])
print("TD :", [round(vr_stat(r_td, k)[0], 2) for k in Ks])

# 滚动 VR(5) 置信带
def rolling_vr(r, k=5, W=250):
    out = np.full(len(r), np.nan)
    for t in range(W, len(r)): out[t] = vr_stat(r[t-W:t], k)[0]
    return out
rvr = rolling_vr(r_rw, k=5, W=250)
theta = (2*(2*5-1)*(5-1)) / (3*5*250)
print("95% 带:", 1-1.96*np.sqrt(theta), 1+1.96*np.sqrt(theta))

# Monte-Carlo 经验 p 值
rng = np.random.default_rng(20260721); N_MC = 4000; vr_null = np.zeros(N_MC)
for i in range(N_MC):
    vr_null[i] = vr_stat(rng.normal(0, 0.01, 2000), 10)[0]
vr_obs, M_obs = vr_stat(r_rw, 10)
p_val = np.mean(np.abs(vr_null - 1) >= np.abs(vr_obs - 1))
print(f"VR(10) 观测={vr_obs:.3f}, M={M_obs:.2f}, 经验 p={p_val:.3f}")
```

> 真实落地提示：把 `r` 换成对数收益率；对高波动资产用异方差稳健统计量；对「多 k 联合显著」用 Chow-Denning；VR 显著只证明「非随机游走」，要结合经济逻辑与去噪，才能真正判断是否存在可交易的微观结构或行为异象。
