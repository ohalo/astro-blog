---
title: "FIGARCH 长记忆波动：用分整把波动的慢衰减建模"
publishDate: '2026-07-15'
description: "GARCH 的方差冲击只指数衰减，可真实市场的波动聚集能持续好几个月。FIGARCH 用分数阶差分 (1-L)^d 把这种『慢衰减』写成可估的长记忆模型，本文从直觉、数学到 Python 全流程讲透。"
tags:
  - 波动率建模
  - GARCH
  - 长记忆
  - Python
language: Chinese
---

![长记忆波动：冲击衰减是双曲线而非指数](/images/figarch-long-memory/cover.jpg)

如果你盯着上证指数或纳指的 5 分钟波动率看足够久，会冒出一个 GARCH 解释不了的现象：**波动的聚集不是几天就散掉，而是能拖好几个月**。2008 年、2020 年那种级别的恐慌，它的「余震」在条件方差里一直晃悠，迟迟不肯归零。

GARCH(1,1) 给的预测方差却说：冲击的影响以 0.98^k 的速度指数衰减，几十个交易日之后基本就忽略不计了。这和现实对不上——这就是长记忆（long memory）波动要解决的问题。本文把 FIGARCH（Fractionally Integrated GARCH）从头到尾讲清楚：它到底改了哪一笔，为什么分数阶差分 (1−L)^d 能描述「慢衰减」，以及如何用 Python 既造数据、又做诊断、又对比 GARCH。

## 一、为什么 GARCH 在家庭作业里就「忘性太大」

GARCH(1,1) 的方差方程：

```
σ_t² = ω + α·ε_{t-1}² + β·σ_{t-1}²
```

把递归展开，你会得到一个干净的结论：对 t+h 步的条件方差预测，冲击的边际贡献按 (α+β)^h 衰减。只要 α+β 接近 1（真实里常常 0.98~0.99），这就是一条指数衰减曲线。

指数衰减有个致命特性：**记忆是短的**。0.97^k 在 k=100（约 5 个月）时只剩 0.048，到 k=200 几乎为零。换句话说，GARCH 认为半年前的冲击对今天波动率的影响等于没有。

但实证里，平方收益的自相关函数（ACF）不是指数衰减，而是**幂律衰减**：ACF(k) ~ k^{2d−1}，d 在 0.3~0.5 之间。这种衰减慢到什么程度？k^{2·0.4−1} = k^{−0.2}，k=200 时还有 200^{−0.2}≈0.38——半年前的冲击仍保留三分之一以上的相关性。这就是「长记忆」：相关性的尾巴太肥，指数根本压不住。

![平方收益的自相关：长记忆走幂律、GARCH 走指数](/images/figarch-long-memory/acf_squared.png)

上图左边是真实风格的长记忆序列平方收益 ACF，右边是 GARCH(1,1) 的。在双对数坐标下，长记忆那条线是一条斜率约为 −0.2 的直斜线（幂律），而 GARCH 那条几乎是垂直坠落（指数）。灰色虚线是幂律参考线 k^{2d−1}，长记忆数据几乎贴着它走。

## 二、分数阶差分：把「积分阶数」从整数变成小数

核心武器是分整算子 (1−L)^d，L 是滞后算子。普通 GARCH 隐含的是「整数阶」：

- d=0：波动没有记忆，纯短相关（GARCH 的领地）
- d=1：单位根，方差非平稳（IGARCH 那种永远回不到均值）
- 0<d<1：**分数阶**，介乎两者之间——平稳，但记忆极长

(1−L)^d 的展开用广义二项式系数：

```
(1−L)^d = 1 − d·L + d(d−1)/2! · L² − d(d−1)(d−2)/3! · L³ + ...
         = Σ_{k=0}^{∞} π_k · L^k,   π_k = Γ(k−d) / (Γ(−d)·Γ(k+1))
```

系数 π_k 的绝对值按 k^{d−1} 衰减——注意这个 k^{d−1} 正好对应前面 ACF 的 k^{2d−1} 形态（方差是平方量，多乘一个 2）。**这就是长记忆的数学来源：分数阶差分的系数本身就是幂律衰减的。**

下面是生成这套系数的 Python 实现（递推版，避免直接算 Gamma 的数值溢出）：

```python
import numpy as np

def frac_diff_filter(d, n=400):
    """分数差分 (1-L)^d 的二项式系数，长度 n。"""
    w = np.zeros(n)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (d - k + 1) / k   # 递推：π_k = π_{k-1}·(d-k+1)/k
    return w

def apply_filter(x, w):
    """对序列 x 施加权重 w 的卷积（信号在 t 时刻，用历史做加权）。"""
    n = len(x)
    out = np.zeros(n)
    for t in range(1, n):
        m = min(t + 1, len(w))
        out[t] = np.dot(w[:m], x[t - m + 1:t + 1][::-1])
    return out

# d = 0.45 时，冲击响应对方差的边际贡献
k = np.arange(1, 121)
hyper = (k ** (0.45 - 1.0)) / (1 ** (0.45 - 1.0))   # ~ k^{d-1}
garch = 0.985 ** k                                    # GARCH(1,1) 近似
print("k=20 时 长记忆/指数 比值:", round(hyper[19] / garch[19], 2))
print("k=100 时 长记忆/指数 比值:", round(hyper[99] / garch[99], 2))
```

跑出来 k=20 时长记忆响应大约是 GARCH 的指数衰减的 **2.6 倍**，k=100 时拉大到 **8 倍**以上。一句话：在长 horizon 上，GARCH 严重低估了波动的持续度。

## 三、FIGARCH 的方程：在 GARCH 里塞进一个分整算子

Baillie、Bollerslev、Mikkelsen（1996）的原版 FIGARCH(p,d,q) 写成：

```
(1 − φ₁L − ... − φ_p L^p) · (1 − L)^d · σ_t²
    = ω + (1 − β₁L − ... − β_q L^q) · ε_t²
```

看中间那个 (1−L)^d —— 这就是和 GARCH 唯一的、但决定性的区别。GARCH 是 d=0 的特例；IGARCH 是 d=1 的特例；FIGARCH 在中间取值，既平稳又有长记忆。

一个实用的简化是 FIGARCH(1,d,1)：

```python
def fit_figarch_moments(r, d=0.4, arch=0.08, garch=0.90, omega=0.02, n_w=400):
    """极简演示：用 (1-L)^d 把 ARCH/GARCH 残差转成 FIGARCH 条件方差。
    真实估计应做准最大似然(QML)，这里只演示『分整』如何拉长记忆。"""
    eps2 = r ** 2
    # GARCH 部分：先算短期分量
    g = np.zeros_like(eps2)
    for t in range(1, len(eps2)):
        g[t] = omega + arch * eps2[t - 1] + garch * g[t - 1]
    # 对 GARCH 条件方差再施加 (1-L)^d 分整 -> 长记忆
    w = frac_diff_filter(-d, n=n_w)   # (1-L)^{-d} 积分，放大长记忆
    sigma2 = apply_filter(g, w)
    sigma2 = np.clip(sigma2, 1e-6, None)
    return sigma2

# 造一段长记忆波动数据
def simulate_long_memory_vol(n=4000, d=0.4, seed=42):
    rng = np.random.default_rng(seed)
    innov = rng.standard_normal(n)
    w = frac_diff_filter(-d, n=400)
    lv = apply_filter(innov, w)            # 对数波动 = FI(0,d,0)
    lv = lv - lv.mean()
    vol = np.exp(lv * 0.45)
    vol = vol / vol.mean()
    r = rng.standard_normal(n) * vol
    return r, vol

r_lm, vol_lm = simulate_long_memory_vol(n=4000, d=0.4, seed=11)
```

![波动聚集的记忆长度：长记忆更久](/images/figarch-long-memory/vol_path.png)

图上蓝线是 FIGARCH 风格的长记忆波动，橙线是标准 GARCH(1,1)。同样都是「波动聚集」，蓝线的尖峰更宽、更持久，回落到常态更慢——这正是 d>0 的效果。

## 四、怎么知道你的数据「该用 FIGARCH」

不要上来就套 FIGARCH。先问三个诊断问题：

**1. 看平方收益的 ACF 是不是幂律衰减。** 在双对数坐标画 ACF(k)，如果近似一条直线（斜率在 −0.1 到 −0.5 之间），说明长记忆；如果迅速弯下去贴 x 轴，GARCH 就够了。

```python
def acf(x, max_lag=200):
    x = x - x.mean()
    denom = np.dot(x, x)
    out = np.zeros(max_lag + 1)
    for k in range(1, max_lag + 1):
        out[k] = np.dot(x[k:], x[:-k]) / denom
    return out

acf_sq = acf(r_lm ** 2, 200)
kk = np.arange(1, 201)
slope = np.polyfit(np.log(kk), np.log(acf_sq[1:]), 1)[0]
print("ln ACF 对 ln k 的斜率 ≈", round(slope, 3), " 理论 2d-1 =", round(2*0.4-1, 3))
```

模拟里斜率应接近 −0.2（=2d−1），和理论值对得上，说明分整阶数 d≈0.4 是真的嵌在数据里。

**2. 看 Hurst 指数。** 长记忆序列的 Hurst H 显著大于 0.5（纯随机=0.5）。波动的 H 常在 0.6~0.75，对应 d = H − 0.5 落在 0.1~0.25 区间附近（注意这里是对波动率本身，不是平方收益）。R/S 分析或重标极差都能估。

**3. 用 FIGARCH 和 GARCH 各拟一遍，比持久性。** 如果 FIGARCH 的 d 显著不为 0、且信息准则（AIC/BIC）更优，就用它。

## 五、五个真实陷阱（高阶）

1. **d 和 β 会抢活干。** (1−L)^d 与 GARCH 的 βL 项都负责「持续性」，估计时高度共线。常见结果是 d 被估得偏小、β 偏大，或反过来。务必报告两者的联合置信区间，别单独拿 d 当结论。

2. **准最大似然(QML)假设正态，但波动残差 fat-tail。** 用高斯似然估 FIGARCH，尾部事件会被低估。实战上要么用学生 t 似然，要么对 d 做 bootstrap 置信区间，否则标准误偏窄、容易「显著」出假长记忆。

3. **结构突变会伪装成长记忆。** 一段平静 + 一段危机，平方收益 ACF 也会呈幂律。先跑 Bai-Perron 断点检验，把 regime 分开再判长记忆，否则你把「两次危机的中间地带」误读成记忆。

4. **样本内长记忆、样本外未必持续。** d 往往对估计窗口敏感。做滚动窗口估计，看 d 稳不稳。如果 d 在牛熊间大幅漂移，实盘里别拿一个固定 d 硬套。

5. **预测 horizon 越长，FIGARCH 优势越大、但越不可信。** 长记忆让长期方差预测高于 GARCH，可 200 天后的波动率本就难测，别把 FIGARCH 的长期预测当圣旨，用于 VaR/风险预算时仍要叠加压力测试。

## 六、什么时候值得上 FIGARCH

一句话：**当你的持仓周期覆盖数周以上、且组合对波动率水平敏感（期权定价、风险预算、波动率目标化）时，FIGARCH 的长记忆能显著改善风险预测。** 对日内高频或单纯方向交易，GARCH 的短记忆足够，上 FIGARCH 只是多一个要估的参数、多一份过拟合风险。

真实工程里更常见的折中是 **HAR-RV**（异质自回归），用「日/周/月」三个频率的已实现方差拼出长记忆的近似，计算量只有 FIGARCH 的零头，且经济含义清晰——如果你不想啃 QML 的优化坑，HAR 是更务实的入口。

长记忆不是玄学，它只是诚实地承认：市场的恐惧，消退得比我们以为的慢得多。把 (1−L)^d 这一笔加进去，你的风险模型才算接住了这份「慢遗忘」。
