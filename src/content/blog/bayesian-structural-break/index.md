---
title: "贝叶斯结构断点检测：把「数据什么时候变了」变成一个概率分布"
description: "因子 IC 突然塌了、波动率悄无声息地翻倍了——这些「数据在哪一步变了」的问题，朴素的两段 t 检验只能回答「变没变」，答不出「在哪一刻」。本文用贝叶斯框架把变点（changepoint）建模成随机变量，给出它的后验分布：在我们合成的两段均值+一段方差突变序列上，精确把均值断点定位在 t=121（真值 120，偏差仅 1 步），检测率上限 0.87 远超朴素 t 检验的 0.47。附完整 Python 与五类真实陷阱（高阶）。"
publishDate: '2026-07-21'
tags:
  - 量化交易
  - 贝叶斯统计
  - 结构断点
  - 变点检测
  - 概念漂移
  - 模型监控
  - 时间序列
  - Python
language: Chinese
difficulty: advanced
cover: "/images/bayesian-structural-break/cover.png"
---

因子的 IC 上周还好好的 0.06，这周突然塌到 0.02；波动率悄无声息地从 15% 翻到 30%。这些「数据在哪一刻变了 regime」的问题，是量化里最容易被忽视、却最致命的一类——你以为是策略失效，其实可能是市场在某个时点切了状态。

最朴素的诊断，是把样本拦腰切成两段，做个 Welch t 检验看「前后均值一不一样」。它的问题很硬：**它只能回答「变没变」，答不出「在哪一刻变的」**；而且一旦断点不在正中，它把断点前后的数据混在一起算方差，功效（power）被严重稀释。

结论先放这：**贝叶斯结构断点检测把「断点位置 τ」当成随机变量，直接输出它的后验分布**。在我们合成的序列（均值在 t=120 跳变、方差在 t=260 跳变）上，对包含 t=120 的窗口做检测，精确把均值断点定位在 **t=121（真值 120，偏差仅 1 步）**；在信噪比扫描里，贝叶斯检测率上限达 **0.87**，而把全样本拦腰切开做 t 检验的方法上限只有 **0.47**——贝叶斯既更准（localization）又更强（power）。诚实地说：当 SNR≈0（压根没均值跳变）时，两者都几乎判不出（贝叶斯 0.02 vs 朴素 0.03），**模型不会凭空造出断点，它只是把真实存在的断点精确定位出来。**

---

## 1. 均值断点 vs 方差断点

「数据变了」至少有两类，诊断方法完全不同：

- **均值断点（mean break）**：序列的「中心」在某一步跳了。比如因子从年化 15% 的预期收益掉到 5%。
- **方差断点（variance break）**：中心没动，但「散度」变了。比如波动率从 15% 翻到 30%，均值还是零。

![合成序列与两个真实断点](/images/bayesian-structural-break/cover.png)

本文的主模型针对**均值断点**——它假设变点前后各是平稳高斯，用贝叶斯因子（Bayes factor）比较「有一个断点」相对「没有断点」的证据。方差断点（t=260 那次）我们用它来划清方法边界：它不在本文模型的探测范围内，这恰恰说明——**好模型知道自己能测什么、不能测什么。**

---

## 2. 贝叶斯变点模型：把 τ 变成随机变量

考虑一个滑动窗口 `[t0, t]`。我们假设窗口里**至多存在一个变点 τ ∈ (t0, t]**：

- 变点之前，观测来自 N(μ_pre, σ²)
- 变点之后，观测来自 N(μ_post, σ²)
- σ² 用窗口样本方差近似（已知方差的高斯共轭，闭式可得）

对每一个候选 τ，我们比较两种假设的边际似然比。已知方差下，均值从 μ_pre 跳到 μ_post 的贝叶斯因子近似为：

$$\text{BF}(\tau) \approx \frac{k(N-k)}{N} \cdot \frac{(\bar x_{pre} - \bar x_{post})^2}{2\sigma^2}$$

其中 k 是变点前样本数、N 是窗口总长。BF 越大，说明「这里有个断点」的证据越强。把所有候选 τ 的 BF 取 log、减最大、指数化、归一化，就得到 τ 的**后验分布** p(τ)。

```python
import numpy as np

def posterior_changepoint(y, t0, t):
    """返回后验 p[tau] (tau in [t0+1, t]) 与最大后验估计 tau_hat。"""
    win = y[t0 : t + 1]
    s2 = max(np.var(win), 1e-6)
    n = t - t0
    p = np.zeros(max(n - 1, 1))
    for k in range(1, n):               # 变点前长度 = k, tau = t0 + k
        pre = win[:k]
        post = win[k:]
        m_pre, m_post = pre.mean(), post.mean()
        bf = k * (n - k) / n * (m_pre - m_post) ** 2 / (2 * s2)
        p[k - 1] = bf
    p = np.exp(p - p.max())            # 数值稳定后归一化
    p /= p.sum()
    return p, t0 + 1 + np.argmax(p)
```

---

## 3. 合成数据 + 定位结果

我们生成 T=400 个观测：均值在 t=120 从 0.0 跳到 1.2，方差在 t=260 从 1.0 跳到 2.2。

```python
rng = np.random.default_rng(20260721)
T, tau1, tau2 = 400, 120, 260
mu1, mu2, sig1, sig2 = 0.0, 1.2, 1.0, 2.2
y = np.zeros(T)
for t in range(T):
    mu = mu1 if t < tau1 else mu2
    sig = sig1 if t < tau2 else sig2
    y[t] = rng.normal(mu, sig)

# 取一个包含 tau1 的窗口，做检测
t_end = 160
p, tau_hat = posterior_changepoint(y, t_end - 80, t_end)
print(tau_hat)          # -> 121  (真值 120, 偏差 1 步)
```

![窗口内 τ 的后验分布](/images/bayesian-structural-break/posterior_single_window.png)

上图是窗口结束于 t=160 时的后验分布：概率质量几乎全压在 t=121 附近（红色虚线是真实断点 120，绿色点线是检测到的 121），**偏差仅 1 步**。这就是贝叶斯方法相对朴素 t 检验的本质优势——它不仅说「变了」，还说「在 t≈121 变的」。

---

## 4. 滚动后验：断点处的质量会「跳」一下

把上面的检测对每个时点 t 用长度 W=80 的滑动窗口重做，得到随时间的滚动后验。当窗口扫过真实断点 t=120 时，后验质量会瞬间聚集到窗口内的对应偏移；窗口驶离后，质量随距离衰减。

![滚动后验热力图](/images/bayesian-structural-break/rolling_posterior.png)

这张热力图里，白色虚线标出两个真实断点。可以看到：均值断点 t=120 处，后验质量明显「亮」起来（因为均值跳变正是模型在测的量）；而方差断点 t=260 处，模型**没有**给出同等强度的定位信号——它本来就不探测纯方差跳变，这是方法的诚实边界，不是 bug。

---

## 5. 与朴素两段 t 检验对比

朴素做法：把全样本（或长窗口）从正中切开，对前后两段做 Welch t 检验，用 |t| > 2 当「有断点」。问题：**断点不在正中时，它把断点两侧的异质数据混在一起，方差被高估、功效被稀释；而且它只给一个「变没变」的 yes/no，不给位置。**

我们用信噪比（mean jump 大小 / σ）扫描，对比两种方法在 60 次重复下的检测率（|τ_hat − 120| ≤ 3 算命中）：

![贝叶斯 vs 朴素 t 检验](/images/bayesian-structural-break/bayes_vs_naive.png)

- 贝叶斯检测率随 SNR 上升，上限达 **0.87**
- 朴素 t 检验上限仅 **0.47**，且全程落后
- 关键诚实结论：**当 SNR≈0（压根没有均值跳变）时，贝叶斯 0.02、朴素 0.03，两者都几乎判不出**。这说明模型不会「无中生有」地制造断点——它只是把真实存在的断点找出来并精确定位。它能赢，是因为它给了断点一个位置参数 τ 并随之积分，而不是把窗口生硬劈成两半。

> 注：朴素 t 检验在「变点恰在中点」的特殊情形下会很强，但实战中断点位置未知且任意，这正是它系统性失效的地方。贝叶斯方法把「τ 在哪儿」作为待估变量，天然规避了这个陷阱。

---

## 6. 五类真实陷阱（高阶）

1. **方差断点盲区**：本文模型只测均值跳变。波动率翻倍（纯方差断点）它定位不到——实战中必须配一个独立测方差/二阶矩变化的模型（如 BCP、或对边际似然用方差也未知的完整共轭），否则会漏掉最重要的「波动率 regime 切换」。
2. **窗口长度 W 主观**：W 太小噪声大、W 太大对近期断点迟钝。W 是超参数，没有银弹——通常按「你关心的最小 regime 长度」来设，并做敏感性扫描。
3. **多断点假设不成立**：模型假设窗口内「至多一个」断点。遇到 2020 年 3 月那种连续 regime 切换（流动性断点 + 相关性断点紧挨着），单断点假设会把两个断点平滑成一个模糊的伪断点。生产环境应改用 PELT / 贝叶斯在线变点（BOCPD）这类多断点框架。
4. **方差未知时的校准**：我们用窗口样本方差近似 σ²。若窗口内恰好横跨断点，σ² 被高估，BF 被压低、漏检。更严谨的做法是让前后段方差也各自未知（完整正态共轭），代价是 BF 不再是闭式、需要数值积分。
5. **前视偏差**：滚动检测必须在「只看 t 及之前」的窗口里做。任何把 t 之后的数据放进窗口的写法，都会把未来信息偷偷算进断点判定，回测里给出虚假的「完美定位」。

---

## 7. 完整 Python 代码

下面是本文全部数字与配图的端到端复现脚本（与图一一对应）：

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260721)
T, tau1, tau2 = 400, 120, 260
mu1, mu2, sig1, sig2 = 0.0, 1.2, 1.0, 2.2
y = np.zeros(T)
for t in range(T):
    mu = mu1 if t < tau1 else mu2
    sig = sig1 if t < tau2 else sig2
    y[t] = rng.normal(mu, sig)

def posterior_changepoint(y, t0, t):
    win = y[t0 : t + 1]
    s2 = max(np.var(win), 1e-6)
    n = t - t0
    p = np.zeros(max(n - 1, 1))
    for k in range(1, n):
        pre, post = win[:k], win[k:]
        bf = k * (n - k) / n * (pre.mean() - post.mean()) ** 2 / (2 * s2)
        p[k - 1] = bf
    p = np.exp(p - p.max()); p /= p.sum()
    return p, t0 + 1 + np.argmax(p)

# 单窗口定位
t_end = 160
p, tau_hat = posterior_changepoint(y, t_end - 80, t_end)
print("detected tau_hat =", tau_hat, "(true", tau1, ")")

# 滚动后验热力图
W = 80
heat = np.zeros((T - W, W))
for i, t in enumerate(range(W, T)):
    pp, _ = posterior_changepoint(y, t - W, t)
    heat[i, : len(pp)] = pp
plt.figure(figsize=(9, 4.6))
plt.imshow(heat.T, aspect="auto", origin="lower",
           extent=[W, T, 0, W - 1], cmap="viridis")
plt.axvline(tau1, color="white", ls="--"); plt.axvline(tau2, color="white", ls="--")
plt.xlabel("Time t"); plt.ylabel("Offset of most recent CP")
plt.title("Rolling posterior over the last changepoint")
plt.savefig("rolling_posterior.png", dpi=130)

# 贝叶斯 vs 朴素 t 检验（SNR 扫描，节选）
def naive_t_p(y, t0, t):
    win = y[t0 : t + 1]; k = (t - t0) // 2
    pre, post = win[:k], win[k:]
    return abs((post.mean() - pre.mean()) /
               np.sqrt(pre.var(ddof=1)/len(pre) + post.var(ddof=1)/len(post)))

snrs = np.linspace(0.0, 1.4, 20)
bayes_hit, naive_hit = [], []
for snr in snrs:
    hb = hn = 0
    for _ in range(60):
        yy = np.array([rng.normal(mu1 if tt < tau1 else snr,
                                  sig1 if tt < tau2 else sig2) for tt in range(T)])
        _, th = posterior_changepoint(yy, W, T - 1)
        hb += abs(th - tau1) <= 3
        hn += naive_t_p(yy, W, T - 1) > 2.0
    bayes_hit.append(hb/60); naive_hit.append(hn/60)

plt.figure(figsize=(9, 4.6))
plt.plot(snrs, bayes_hit, "-o", label="Bayesian CP")
plt.plot(snrs, naive_hit, "-s", label="Naive t-test")
plt.xlabel("SNR (mean jump / sigma)"); plt.ylabel("Detection rate")
plt.title("Bayesian: higher ceiling + exact localization")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("bayes_vs_naive.png", dpi=130)
```

---

**一句话总结**：结构断点检测的价值不在「有没有变」这个 yes/no，而在「在哪一刻变的」这个精确位置——它决定了你该在哪个时点切换模型、缩减敞口、重启训练。贝叶斯框架把 τ 变成随机变量，给出的后验分布既是告警信号，也是定位坐标；但记住它的边界：它测均值跳变、默许多断点要升级框架、且永远别让未来信息漏进窗口。
