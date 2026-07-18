---
title: "粒子滤波状态估计：用SIR滤波从噪声观测中还原隐含波动率"
description: "你只能观测到带噪声的市场报价,却想知道藏在背后的「真实隐含波动率」——这是个标准的状态估计问题。卡尔曼滤波要求状态方程和观测方程都是线性高斯的,可真实波动的噪声明明是异方差、非高斯的。本文用一套合成数据,从零实现 SIR 粒子滤波(序贯重要性重采样),从被噪声淹没的观测里还原隐含波动率路径,并对照线性高斯卡尔曼滤波,证明粒子滤波 RMSE 低 36%,把整套可复现 Python 给你(中阶)。"
publishDate: '2026-07-19'
tags:
  - 量化交易
  - 状态估计
  - 粒子滤波
  - SIR滤波
  - 隐含波动率
  - 卡尔曼滤波
  - 非线性滤波
  - Python
language: Chinese
difficulty: intermediate
---

你手上有期权报价,想实时知道**市场此刻的隐含波动率(IV)到底是多少**。

麻烦在于:报价本身就是**带噪声的观测**。同一时刻,不同行权价、不同期限的期权报出来的 IV 各不相同,加上买卖价差、流动性、报价延迟,你看到的每一个数都是「真实 IV + 噪声」。而且这个噪声**不是同方差**——波动越大的时候,报价乱得越离谱。

这正是一个经典的**状态估计(state estimation)**问题:

- 隐藏状态 $x_t$:真实隐含波动率(你想知道的);
- 观测 $y_t$:带噪声的报价(你能看到的);
- 状态转移:IV 缓慢游走,偶尔漂移;
- 观测方程:噪声随状态变大(异方差)。

**卡尔曼滤波(KF)**能解这类问题——但它要求「状态转移线性 + 噪声高斯 + 观测方程线性」。真实 IV 的噪声是异方差、非高斯的,KF 的线性高斯假设会**系统性失真**。这时候该上 **粒子滤波(Particle Filter)**。

> 本文用合成数据从零实现 **SIR 粒子滤波(Sequential Importance Resampling)**,从被噪声淹没的观测里还原 IV 路径,并和线性高斯卡尔曼滤波对照,看非线性方法到底强在哪。

## 一、为什么卡尔曼滤波在这里「不够用」

标准卡尔曼滤波的两步假设:

$$
\begin{aligned}
x_t &= F\,x_{t-1} + w_t,\quad w_t\sim\mathcal N(0,Q) \quad\text{(状态转移线性高斯)}\\
y_t &= H\,x_t + v_t,\quad v_t\sim\mathcal N(0,R) \quad\text{(观测线性高斯)}
\end{aligned}
$$

问题在于我们的观测噪声 $v_t$ **随状态变大**:波动高时报价噪声大,波动低时噪声小。这不是「固定 $R$ 的高斯」,而是 $v_t\sim\mathcal N(0,\,(\kappa x_t)^2)$ 的**异方差**。KF 用固定 $R$ 去近似,会在高波动段欠惩罚、低波动段过惩罚——估计被拉偏。

粒子滤波不依赖这些漂亮假设:它用**一堆带权重的粒子**去近似状态的完整后验分布 $p(x_t\mid y_{1:t})$,线性、高斯、异方差通通无所谓。

## 二、造一段「真有结构」的隐含波动率

真实 IV 不是纯随机游走——它常带慢趋势、偶有 regime 切换。我们造 800 个时点的真实 IV:前 60% 缓慢上行、后段缓慢下行,叠加小幅随机游走,并夹在 $[0.05,0.45]$ 之间。观测用 IV 加上**异方差噪声**(噪声幅度正比于当前 IV):

```python
import numpy as np

rng = np.random.default_rng(20260719)
T = 800
true_iv = np.zeros(T); true_iv[0] = 0.20
for t in range(1, T):
    drift = 0.00002 if t < T*0.6 else -0.00003
    true_iv[t] = np.clip(true_iv[t-1] + drift + 0.004*rng.normal(0,1), 0.05, 0.45)

obs = true_iv + true_iv * 0.06 * rng.normal(0, 1, T)   # 异方差噪声: 噪声随 IV 变大
```

因为我们自己造数据,**真实 IV 已知**——这正是检验滤波好坏的金标准(真实研究里你拿不到它,只能用样本外预测误差替代)。

## 三、SIR 粒子滤波:预测 → 更新 → 重采样

核心三步循环,每个时刻 $t$:

1. **预测**:把每个粒子按状态转移推进一步(加过程噪声);
2. **更新**:用当前观测 $y_t$ 给每个粒子算**重要性权重**(似然),异方差噪声下权重用 $\mathcal N(0, (\kappa x^{(i)})^2)$;
3. **重采样**:按权重重新抽一遍粒子,权重重置均匀——防止「粒子退化」(少数粒子权重塌成 1,其余≈0)。

```python
def sir_filter(y, N=2000, proc_sd=0.004, obs_sd_scale=0.06):
    rng = np.random.default_rng(20260719)
    particles = np.full(N, 0.20) + 0.002*rng.normal(0, 1, N)
    mean_est = np.zeros(len(y)); var_est = np.zeros(len(y)); ll = 0.0
    for t in range(len(y)):
        # 1) 预测: 状态转移
        particles = particles + proc_sd*rng.normal(0, 1, N)
        particles = np.clip(particles, 0.02, 0.6)
        # 2) 更新: 异方差似然权重
        obs_sd = obs_sd_scale * particles
        logw = -0.5*((y[t]-particles)/obs_sd)**2 - np.log(obs_sd)
        maxw = logw.max(); w = np.exp(logw - maxw); w /= w.sum()
        # 3) 重采样(系统重采样)
        idx = np.searchsorted(np.cumsum(w), rng.uniform(0, 1, N))
        particles = particles[idx]
        mean_est[t] = particles.mean(); var_est[t] = particles.var()
        ll += np.log(np.exp(logw).mean()) + maxw
    return mean_est, var_est, ll

mean_est, var_est, ll = sir_filter(obs)
```

注意第 2 步的 `obs_sd = obs_sd_scale * particles`:**噪声标准差正比于粒子值**——这就是把「异方差」写进滤波的关键。固定 `obs_sd` 的写法会退化成近似 KF。

## 四、和线性高斯卡尔曼滤波对照

为公平比较,我们给 KF 一个合理的固定 $R$:用观测的总体离散度近似。KF 标准递归:

```python
def kalman(y, q=0.004**2, r=(np.std(obs)*0.06)**2):
    x, P = 0.20, 0.01; est = np.zeros(len(y)); var = np.zeros(len(y))
    for t in range(len(y)):
        x = x; P = P + q                      # 预测
        S = P + r; K = P / S                  # 更新
        x = x + K*(y[t]-x); P = (1-K)*P
        est[t], var[t] = x, P
    return est, var

kf_est, kf_var = kalman(obs)
```

结果:

| 方法 | 还原 RMSE(对真实 IV) | 对数似然 |
|---|---|---|
| 粒子滤波(SIR) | **0.00551** | 6794.7 |
| 卡尔曼(线性高斯近似) | 0.00855 | — |

**粒子滤波的 RMSE 比 KF 低约 36%**。差异来源正是异方差:KF 用固定 $R$ 在波动高段把噪声当太小、在波动低段当太大,估计被系统性拉偏;粒子滤波让每个粒子的噪声随自身大小缩放,在高波动段自动「少信一点观测」。

![粒子滤波从噪声观测中还原隐含波动率路径](/images/particle-filter-state-estimation/pf_state_trajectory.png)

## 五、局部放大:非线性方法更贴真实状态

把前段放大看(图 2),真实 IV(绿)、观测(灰)、粒子滤波(红)、卡尔曼(蓝)叠在一起:

![局部放大：粒子滤波在非高斯噪声下更贴真实状态](/images/particle-filter-state-estimation/pf_local_zoom.png)

在 IV 快速上行或下行的拐点上,卡尔曼(蓝虚线)的估计**滞后且过冲**——因为它假定固定观测噪声,在噪声剧变段反应迟钝;粒子滤波(红)的线更紧地咬住真实值。这正是非线性/非高斯状态估计的价值:不是「更平滑」,而是**在结构变化时更跟手**。

## 六、粒子云:单时刻的后验分布长什么样

粒子滤波最大的额外福利:它不光给一个均值,还给你**整个后验分布**。在时刻 $t=300$ 看粒子云:

```python
# 重跑并在 t=300 保存(粒子, 权重)
parts, wts = sir_filter_snapshot(obs, t_snap=300)
```

![粒子云：滤波在单个时刻给出的后验分布](/images/particle-filter-state-estimation/pf_particle_cloud.png)

实测:真实 IV@300 = **0.1980**,滤波均值@300 = **0.1889**,95% 置信半宽 = **0.0123**。后验是个以均值中心的钟形——你拿到的不只是一个点估计,而是一句「我现在有 95% 把握 IV 在 [0.1766, 0.2012] 之间」。这对**风险预算和期权对冲的置信区间**非常有用:你可以基于后验分位数,而不是直接用点估计去做敞口决策。

## 七、五个必须知道的坑

1. **粒子退化(degeneracy)**。若不做重采样,几步之后权重会塌成「一个粒子≈1、其余≈0」,滤波失效。本文每步都重采样,但重采样太频繁又会**样本贫化**(粒子多样性耗尽)。实务上用 **ESS(有效样本量)阈值**触发重采样更稳。
2. **粒子数 N 是精度/算力的天平。** N=2000 在 800 点上跑得动;实盘高频(每秒数千 tick)要 N=5000+ 并上 `numba`/GPU。N 太小,后验被抽样噪声污染,RMSE 会反超 KF。
3. **过程噪声 `proc_sd` 调参是艺术。** 太大→估计抖动、过拟合观测噪声;太小→跟踪迟钝、跟不上 regime 切换。用样本外预测误差做网格搜索,别拍脑袋。
4. **异方差假设要对。** 本文假设「噪声正比于 IV」。若真实噪声是别的结构(比如正比于已实现波动、或含跳跃),要把 `obs_sd` 换成对应形式,否则粒子滤波只会比 KF 好一点点。
5. **初始分布要覆盖真实状态。** 初始粒子若全集中在错误区间,前期要靠很久才能「游」回真实值,开头一段估计会偏。用一段历史数据的分位数初始化更稳。

## 八、完整可复现脚本

本文全部图表与数字由 `gen_particle_filter.py` 生成(已附仓库)。三件事:① `simulate` 造带异方差噪声的 IV 观测;② `sir_filter` 做预测-更新-重采样;③ 与 `kalman` 对照算 RMSE。把 `obs` 换成你真实的期权 IV 序列(或任何隐藏状态+噪声观测的场景:资金流入、真实杠杆率、潜在违约概率……),就能直接套用。

> 想再进一步:把「状态转移」从随机游走升级成**隐马尔可夫(regime 切换)**,或把粒子滤波接成**自适应观测噪声**——那就是更贴近实盘的非线性状态估计栈了。
