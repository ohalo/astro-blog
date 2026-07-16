---
title: "转移熵因果识别：用信息流方向分清『谁带动谁』"
description: "相关系数只说『一起动』，不说『谁先动』。转移熵(Transfer Entropy, Schreiber 2000)用条件互信息量，量的是『加入 X 的历史后，Y 的下一刻不确定性下降了多少 bits』——天然非对称。本文用符号化估计在合成领先-滞后链 X→Y→Z 上复现：TE(X→Y)=0.214 对 TE(Y→X)=0.030（比值 7×）、TE(Y→Z)=0.218 对 TE(Z→Y)=0.037；而共同因驱动的 U/V 虽然相关 0.32，双向 TE 却几乎对称(0.045 vs 0.041)——正是相关说不清、TE 能分开的那类结构。再沿信息流方向下注，Sharpe 5.05 碾压反向 −5.05；信息流方向就是 alpha 方向。附完整 Python 与六类真实陷阱（高阶）。"
publishDate: '2026-07-17'
tags:
  - 量化交易
  - 转移熵
  - 因果识别
  - 信息传递
  - 非线性
  - 领先滞后
  - 信息论
  - Python
language: Chinese
difficulty: advanced
---

你盯盘时常有一种直觉：「好像每次 A 一动，B 过一会儿才跟。」但当你掏出相关系数一算——`corr(A,B)=0.6`，然后把 `corr(B,A)` 也算一遍，还是 0.6。**相关系数天生对称，它永远分不清谁先动、谁后动。**

这在量化里是个大坑。比如你想知道「是美股带领 A 股，还是 A 股带动美股」「是 VIX 带领股指期货，还是反过来」「是行业内龙头带领跟涨小弟，还是小弟先动龙头才反应」。这些问题里，**方向就是一切**——你顺着信息流建仓吃的是领先收益，逆着流就是接盘。

结论先放这：**转移熵（Transfer Entropy, Schreiber 2000）是给「谁带动谁」这件事装了个方向箭头。** 它量的是条件互信息：加入 $X$ 的过去后，$Y$ 的下一刻不确定性下降了多少 bits。下降得多，说明 $X$ 真的在「喂」信息给 $Y$。

## 一、直觉：相关是「一起动」，转移熵是「先动的那边在带」

相关系数 $\rho(A,B)$ 对称，是因为它只描述**联合分布的形状**：两者如何一起上下。但它对**时间箭头**完全失明——无论 A 领先 B 一秒还是 B 领先 A 一秒，$\rho$ 一样。

转移熵换了一个问题。定义 $TE(X \to Y)$：

$$TE(X\to Y) = \sum p(y_{t+1}, y_t, x_t)\,\log_2 \frac{p(y_{t+1}\mid y_t, x_t)}{p(y_{t+1}\mid y_t)}$$

白话：先看「只用 $Y$ 的过去 $y_t$，能多确定 $Y$ 的下一刻 $y_{t+1}$」（这是 $H(y_{t+1}|y_t)$）；再看「加了 $X$ 的过去 $x_t$ 之后，能多确定多少」。两者之差（以 bits 计）就是 $X$ 额外带来的信息量。

关键性质有两个：

1. **非对称**。$TE(X\to Y)$ 和 $TE(Y\to X)$ 一般不相等。若 $X$ 领先 $Y$，前者显著大于后者。
2. **非线性友好**。它是基于概率分布的条件互信息，不要求线性关系——$X$ 用非线性方式影响 $Y$，相关可能接近 0，但 TE 照样能抓到。

> 和相关不同，TE 给的是一条**有向边**：$X \xrightarrow{TE} Y$ 表示「信息从 X 流向 Y」。把一堆资产的 TE 拼起来，就是一张「信息传导网络」，比相关矩阵那张无向毛线团有用得多。

## 二、估计：符号化 / 分箱转移熵（可自检的实现）

Kraskov 的 k-NN 估计是学界金标准，但对连续序列依赖近邻搜索、不易自检。本文用**符号化分箱估计**——把每个序列等宽分箱成离散符号，再用联合/边缘计数近似条件概率。它的好处是**数值必然合理**（TE 不可能超过 $H(y_{t+1}|y_t)$，单变量高斯熵只有零点几 bit），且每一步都能手算核对：

```python
import numpy as np

def binned_te(x, y, k=8, lag=1, base=2.0, eps=1e-9):
    """TE(X->Y): x 源(领先), y 目标(滞后). 返回 bits."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    xb = np.digitize(x, np.linspace(x.min(), x.max(), k+1)[1:-1])
    yb = np.digitize(y, np.linspace(y.min(), y.max(), k+1)[1:-1])
    # 对齐: y_{t+1}, y_t, x_t
    Yf, Yp, Xp = yb[lag:], yb[:-lag], xb[:-lag]
    n = len(Yf)
    # 联合 (Yf,Yp,Xp) 计数, reshape 成 (k,k,k)
    keys = Yf*(k*k) + Yp*k + Xp
    c_joint = np.bincount(keys, minlength=k*k*k).astype(float).reshape(k,k,k) + eps
    c_ypxp  = np.bincount(Yp*k+Xp, minlength=k*k).astype(float).reshape(k,k) + eps
    c_yp    = np.bincount(Yp, minlength=k).astype(float) + eps
    c_yfyp  = c_joint.sum(axis=2) + eps          # 对 Xp 求和 -> (Yf,Yp)
    # p(yf|yp,xp)/p(yf|yp) = p(yf,yp,xp)*p(yp) / [p(yp,xp)*p(yf,yp)]
    ratio = (c_joint * c_yp[None,:,None]) / (c_ypxp[None,:,:] * c_yfyp[:,:,None])
    logr = np.log2(np.clip(ratio, eps, None))
    return float(np.sum(c_joint / n * logr))
```

注意 `k`（箱数）是偏差-方差的旋钮：太小信息被抹平（TE 偏低），太大每个箱样本稀疏（噪声大）。`k=8` 在数千样本下通常稳。

## 三、仿真验证一：X→Y→Z 领先-滞后链

造一条真实的领先-滞后因果链——$Y$ 受 $X$ 的过去带动，$Z$ 受 $Y$ 的过去带动，$X$ 与 $Z$ 之间**没有直接因果**，只通过 $Y$ 间接相连。如果 TE 真能还原方向，应当看到 $TE(X\to Y) \gg TE(Y\to X)$、$TE(Y\to Z) \gg TE(Z\to Y)$，而 $TE(X\to Z)$ 只反映弱间接传递：

```python
rng = np.random.default_rng(20260717)
T = 4000
X = np.zeros(T)
for t in range(1, T):
    X[t] = 0.6*X[t-1] + rng.normal()*0.8
Y = np.zeros(T)
for t in range(1, T):
    Y[t] = 0.5*Y[t-1] + 0.4*X[t-1] + rng.normal()*0.6    # Y 受 X 领先带动
Z = np.zeros(T)
for t in range(1, T):
    Z[t] = 0.5*Z[t-1] + 0.4*Y[t-1] + rng.normal()*0.6    # Z 受 Y 领先带动

te_xy = binned_te(X, Y); te_yx = binned_te(Y, X)
te_yz = binned_te(Y, Z); te_zy = binned_te(Z, Y)
te_xz = binned_te(X, Z); te_zx = binned_te(Z, X)
```

![转移熵矩阵 TE(行→列)：X→Y→Z 链被还原，红色箭头标注真实信息流方向](/images/transfer-entropy-finance/te_chain_recovery.png)

实测（4000 点）：

| 方向 | TE (bits) | 方向 | TE (bits) |
|---|---|---|---|
| **TE(X→Y)** | **0.214** | TE(Y→X) | 0.030 |
| **TE(Y→Z)** | **0.218** | TE(Z→Y) | 0.037 |
| TE(X→Z) | 0.057 | TE(Z→X) | 0.037 |

**比值 $TE(X\to Y)/TE(Y\to X) = 7.0\times$，$TE(Y\to Z)/TE(Z\to Y) = 6.0\times$**——非对称性被干干净净地还原出来。而 $TE(X\to Z)=0.057$ 只有直接边的一半不到，说明间接传递确实弱于直接因果（信息在链上衰减）。这正是 TE 作为「因果探针」最漂亮的演示：它把一条无向相关毛线，拆成了一串有向箭头。

## 四、仿真验证二：共同因 vs 真因果，相关分不清、TE 能分开

这是 TE 比相关强的最关键场景。**共同因（confounder）**：一个隐藏变量 $W$ 同时驱动 $U$ 和 $V$，但 $U$、$V$ 之间**没有因果**。

```python
W = rng.normal(0, 1, T)
U = 0.7*W + rng.normal(0, 1, T)
V = 0.7*W + rng.normal(0, 1, T)
te_uv = binned_te(U, V); te_vu = binned_te(V, U)
corr_uv = np.corrcoef(U, V)[0, 1]
```

![左：相关系数(对称) U↔V=0.32；右：转移熵(非对称) TE(U→V)=0.045 vs TE(V→U)=0.041——双向几乎对称，证伪直接因果](/images/transfer-entropy-finance/te_vs_corr.png)

实测：**corr(U,V) = 0.32（相关显著）**，但 **TE(U→V)=0.045、TE(V→U)=0.041，几乎对称**。换句话说——两者明明「一起动得很欢」，TE 却一眼看穿「它们之间没有信息流，只是被同一个第三者牵着走」。

> 这正是实盘里最容易栽的坑：你看到 A 股和美股「高度相关」，以为找到了领先指标，其实两者都被美联储政策这个共同因驱动，谁也不带谁。相关告诉你「绑一起」，TE 告诉你「别顺着谁建仓，因为那是假领先」。

## 五、仿真验证三：信息流方向就是 alpha 方向

如果 TE 还原的领先关系是真的，那「顺着信息流下注」应该能吃到领先收益。构造一个明确领先关系：$B_{t+1}$ 里含有 $A_t$ 的信息（信息流 $A\to B$），用 $A$ 的当期方向作为 $B$ 下期变化的领先信号：

```python
M = 2000
A = rng.normal(0, 1, M)
B = np.zeros(M)
for t in range(1, M):
    B[t] = 0.5*A[t-1] + rng.normal()*0.8      # A 领先 B 一个 tick
sig = np.sign(A[:-1])                          # t 时刻 A 方向(领先 B 下期)
realized = B[1:] - B[:-1]                       # B 下期真实变化
ret_follow  = sig * realized                   # 沿信息流: 跟 A 方向押 B 下期
ret_against = -sig * realized                  # 反向押注
ret_bh      = realized.copy()                  # 买入持有 B 变化

def sharpe(r):
    r = np.asarray(r)
    return r.mean() / (r.std()+1e-9) * np.sqrt(252)
s_follow, s_against, s_bh = sharpe(ret_follow), sharpe(ret_against), sharpe(ret_bh)
```

![沿信息流下注(蓝, Sharpe 5.05) vs 反向押注(红, −5.05) vs 买入持有(灰, −0.02)：方向就是收益](/images/transfer-entropy-finance/te_trading_edge.png)

实测：**沿信息流 Sharpe = 5.05，反向押注 Sharpe = −5.05，买入持有 ≈ 0（−0.02）**。顺着 TE 指出的信息流方向，你吃的是真实的领先差；逆着流，你等于在给领先方送钱；而买入持有（不判断方向）几乎不赚——因为领先收益在 `B_t` 内部被噪声抵消了，只有「用 A 的已知方向去押 B 的下一刻」才能把它抠出来。

## 六、真实陷阱（别直接照抄）

1. **分箱数 k 是偏差-方差旋钮，不能乱设**。k 太小把非线性关系抹成阶梯（TE 系统性偏低），k 太大每箱样本不足（估计噪声爆炸、小样本下 TE 可能虚高）。实盘日频数据至少数千样本才稳，k 在 6–12 之间做敏感性扫描。

2. **TE 非对称 ≠ 统计显著非对称**。本文比值 7× 是因为合成数据「按真实方向造」；实盘里 TE(X→Y) 和 TE(Y→X) 可能都只比噪声高一点点，差值不显著。务必做**置换检验 / bootstrap 置信区间**，剔除「伪方向」。

3. **非平稳市场会让 TE 退化成相关**。危机期所有资产一起跳，领先滞后结构被「共同冲击」淹没，TE 在危机窗口会失真。应按 regime 分段（或滚动窗口）估计，别拿全样本一张静态矩阵当圣旨。

4. **共同因 / 隐藏变量仍可能骗过 TE**，尤其是**部分**共同因（U、V 既被 W 驱动、又有微弱直接因果）。此时双向 TE 都偏高但非对称，容易误判。严谨做法是用**条件转移熵** $TE(X\to Y \mid Z)$ 控制第三变量后再判方向。

5. **tick 级信息流 ≠ 可交易领先**。微观结构噪声、买卖价差、不同标的交易时段错位，会让「A 动 B 才动」的领先只有毫秒级——你看到箭头，却因延迟和滑点吃不到。TE 指出的领先必须**大于你的执行延迟**才有用（通常要日频/小时频的研究级信号）。

6. **TE 测的是统计因果，不是机制因果**。它告诉你「信息从 X 流向 Y」，但不解释为什么。监管变更、指数调样、算法共振都可能导致假领先。落地前要结合机制叙事，别把纯统计箭头当交易圣杯。

## 七、小结

相关系数回答「谁和谁绑一起」，转移熵回答「谁先动、带着谁动」——这是两个完全不同的问题。本文用符号化估计在合成数据上复现：

- 领先-滞后链 **X→Y→Z 被还原**：TE 比值 7× / 6×，间接边明显弱于直接边；
- **共同因 U/V 虽相关 0.32，双向 TE 却几乎对称（0.045 vs 0.041）**——相关说不清的，TE 一眼拆穿；
- 沿信息流方向下注 **Sharpe 5.05**，反向 −5.05——信息流方向就是 alpha 方向。

从 Granger 因果（线性、对称友好但漏非线性）到本文的转移熵（非线性、非对称、信息论原生），再到条件转移熵（控隐藏变量）——一条主线贯穿：**金融市场里「谁带动谁」不是对称问题，给它装个方向箭头，你才看得见钱往哪流**。把一张相关系数毛线团换成 TE 有向网络，很多「假联动」会当场现形。
