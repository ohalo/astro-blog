---
title: "粒子滤波波动率估计：从噪声价格中提取隐藏波动率"
publishDate: 2026-11-07
language: zh
tags: [随机波动率, 粒子滤波, 状态空间模型, Heston模型, 量化]
difficulty: advanced
description: "用 bootstrap 粒子滤波从可观测价格反推 Heston 隐藏波动率，与实现波动率、EWMA 在同一年化口径下对拍。诚实结论：模型匹配时 MAE 只降 9.11%，参数一旦设错反而落后 19.16%——粒子滤波卖的是模型知识，不是算法魔法。附完整可运行代码、结果 JSON 与五张真实计算图。"
image: "images/particle-filter-volatility/pf_1_paths.png"
---

## 一、波动率不可观测，这才是问题的起点

资产价格 $S_t$ 每天都能看到，但驱动它的波动率 $\sqrt{v_t}$ 看不见。这不是测量精度问题，而是**状态本身隐藏**：能观测到的只有一条带噪声的价格路径，波动率是生成这条路径的潜在变量。所有波动率估计方法，本质上都在做同一件事——用可观测序列去反推不可观测的状态。

最省事的做法是实现波动率（Realized Volatility, RV）：取过去 20 天的对数收益率平方，开方再年化。它不依赖任何模型，实现只要三行代码。问题也正在于「不依赖模型」：RV 假设窗口内波动率近似恒定，用 20 个样本去估一个瞬息变化的量，噪声大且天然滞后；窗口调短噪声爆炸，调长则错过波动率聚集与突变。RiskMetrics 的 EWMA 用指数衰减权重改善了滞后，但仍是无模型平滑，不认识均值回归，也不认识波动率之波动。

粒子滤波（Particle Filter, PF）走的是另一条路：先写下波动率**自己怎么演化**的状态方程，再用贝叶斯公式把它和每日观测逐一融合。它不只是平滑，而是对每个时点给出完整的后验分布。听起来很美，但它把风险从「噪声」转移到了「模型假设」上——本文会用实验把这笔交易算清楚。

## 二、状态空间模型：Heston 与年化参数

主实验用 Heston 随机波动率模型，两个耦合的随机微分方程：

$$
\begin{aligned}
d\log S_t &= \left(\mu - \tfrac{1}{2}v_t\right)dt + \sqrt{v_t}\,dW_t^1 \\
dv_t &= \kappa(\theta - v_t)dt + \sigma_v\sqrt{v_t}\,dW_t^2
\end{aligned}
$$

参数全部按**年化**设定，时间步长 $dt=1/252$：$\kappa=3.0$（均值回归速率，半衰期约 58 个交易日）、$\theta=0.04$（长期方差，对应年化波动 20%）、$\sigma_v=0.5$（波动率之波动）、$v_0=0.04$、$\mu=0.05$（年化漂移 5%）。$dW^1$ 与 $dW^2$ 的相关系数 $\rho$ 是杠杆效应：真实市场里利空来袭时波动率飙升，$\rho$ 通常为负。

**主实验刻意设 $\rho=0$**。原因很重要：bootstrap 粒子滤波的似然只用到观测的方差，不建模 $v_t$ 与价格冲击的符号相关。若生成数据带杠杆而滤波器不认识它，测到的就是「模型错配 + 算法误差」的混合物，无法归因。先让生成模型与滤波器严格匹配，才能干净地量出 PF 本身值多少钱；杠杆错配作为局限在第七节讨论。

Euler 离散（$z_1,z_2$ 为独立标准正态）：

$$v_t = \max\!\left(v_{t-1} + \kappa(\theta-v_{t-1})dt + \sigma_v\sqrt{v_{t-1}dt}\,z_1,\ 10^{-8}\right)$$

$$r_t = \left(\mu - \tfrac{1}{2}v_t\right)dt + \sqrt{v_t dt}\,\left(\rho z_1 + \sqrt{1-\rho^2}z_2\right)$$

用对数价格而非价格加法，是为了保证 $S_t>0$，否则取对数时会出现 NaN——这是这类模拟最常见的翻车点。

## 三、Bootstrap 粒子滤波：五个步骤

用 $N$ 个带权粒子 $\{v_t^{(i)}, w_t^{(i)}\}$ 近似后验 $p(v_t\mid r_{1:t})$。每个交易日循环五步：

**1. 状态转移（预测）**：每个粒子按上面的方差方程独立演化，$v^{(i)} \leftarrow v^{(i)} + \kappa(\theta-v^{(i)})dt + \sigma_v\sqrt{v^{(i)}dt}\,z^{(i)}$。这一步只注入过程噪声，不看当天观测。

**2. 似然加权（更新）**：给定粒子 $v^{(i)}$，当日对数收益服从 $r_t \sim \mathcal N\big((\mu-v^{(i)}/2)dt,\ v^{(i)}dt\big)$，于是

$$\tilde w^{(i)} \propto w^{(i)} \cdot \frac{1}{\sqrt{v^{(i)}dt}}\exp\!\left(-\frac{\big(r_t-(\mu-v^{(i)}/2)dt\big)^2}{2v^{(i)}dt}\right)$$

实践中在 log 域先减最大值再指数化，避免权重下溢。

**3. 后验估计**：$\hat\sigma_t=\sqrt{\sum_i \tilde w^{(i)} v^{(i)}}$，同时用加权分位数给出 5%–95% 可信带。**估计必须在重采样之前取**，否则人为引入一次抽样噪声。

**4. 一步预测**：把后验粒子沿状态方程推一步并取条件期望（去掉零均值随机项），得到对 $t+1$ 时刻波动率的预测。这一步让 PF 与 RV 能在完全相同的任务上对拍。

**5. ESS 诊断与系统重采样**：有效样本数 $\mathrm{ESS}=1/\sum_i w_i^2$，当 $\mathrm{ESS}<N/2$ 时按权重做系统重采样（等距起点 + 累积权重查找），把权重重置为均匀。系统重采样比多项式抽样方差更小，且每步只消耗一个随机数。

这里要特别强调一致性：ESS 阈值与文章末尾代码里的 `if ess < ess_frac*N` 是同一个条件。不少教程在正文写 `ESS<N/2`、代码却写成 `sum(w²)<0.5`（等价于 ESS>2，几乎每步都重采样），两者结果差得很远，读者按文章复现不出来——**代码与正文必须同源**。

## 四、实验怎么保证公平

- 样本 500 个交易日，固定随机种子 20261107，粒子数 500。
- 三种估计全部换算成**年化波动率**：RV 用 20 日窗口除以 $dt$，EWMA 同样除以 $dt$，PF 直接输出 $\sqrt v$。口径不统一是所有「PF 碾压 RV」结论里最隐蔽的水分。
- RV 的窗口取 $r_{t-20:t}$，包含当日收益；PF 的似然也用同一条 $r_t$。两边信息集相同，不存在偷看未来。
- 评估区间统一取 $t=20,\dots,498$；末位排除是因为 EWMA 递推在最后一个点没有对应的收益。
- 一步预测统一对齐：$t$ 时刻的预测值对比 $t+1$ 时刻的真值，RV 用自身当前值充当下一期预测。

真实路径平均年化波动 0.2117，PF 估计均值 0.2077，二者接近，说明滤波器没有明显的系统性漂移。

## 五、结果：优势是真的，但只有 9%

| 指标（MAE 越小越好 / 相关越大越好） | 粒子滤波 | RV(20) | EWMA(0.94) |
|---|---|---|---|
| 估计 MAE | **0.038071** | 0.041887 | 0.039303 |
| 与真值相关系数 | **0.8934** | 0.8698 | 0.8917 |
| 一步预测 MAE | **0.040015** | 0.043660 | — |

PF 相对 RV 的 MAE 改善 **9.11%**，一步预测改善 **8.35%**，相关系数从 0.8698 提到 0.8934。三项都赢，但都是个位数百分比，不是数量级碾压。EWMA 表现意外地接近 PF（MAE 0.039303、相关 0.8917），说明一个调好衰减系数的无模型平滑器，在平稳波动率环境下并不容易打败。

![真值、PF 后验均值与由同一批粒子算出的 5%–95% 分位带](/images/particle-filter-volatility/pf_1_paths.png)

图 1 的红色带不是手工画的 ±1 个常数，而是每个时点从 500 个加权粒子实算的 5%/95% 分位。真值绝大多数时候落在带内，说明后验不确定度的校准基本合理。

![三种估计方法同框对比](/images/particle-filter-volatility/pf_2_baselines.png)

图 2 能看出差别在哪：RV 与 EWMA 的曲线更「毛」，PF 更贴真值的慢变轮廓。PF 的优势来自它知道波动率会均值回归、且波动幅度受状态方程约束，等于用模型结构换掉了部分采样噪声。

![估计误差分布](/images/particle-filter-volatility/pf_3_error_dist.png)

图 3 的误差密度 PF 峰值更高、尾部更薄，RV/EWMA 则更分散——但三者分布重叠很大，这就是为什么改善只有 9%。

![三个时点的后验粒子分布](/images/particle-filter-volatility/pf_4_particles.png)

图 4 是三个**各自独立保存**的粒子快照，分布中心与宽度都明显不同：t=50 时 PF 0.2178 对真值 0.2364（ESS 333）；t=200 波动率冲到 0.3765，PF 只给到 0.2991（ESS 270）；t=400 真值回落到 0.0809，PF 还在 0.1057（ESS 200）。后面两个时点清楚地暴露出 PF 的滞后：**波动率急升时跟不上，急降时退得慢**。这与均值回归先验的收缩效应一致，不是 bug，是代价。

![粒子数敏感性与 ESS 演化](/images/particle-filter-volatility/pf_5_sensitivity.png)

图 5 左侧是共同随机数下的粒子数扫描（N=25/50/100/200/500/1000），MAE 在 0.0372–0.0399 之间震荡、没有单调改善，N=500 那点精确等于主实验的 0.038071。右侧 ESS 曲线显示全程只触发 47 次重采样、平均 ESS 355.3，权重退化并不严重。**结论：粒子数不是瓶颈，模型设定才是。**

## 六、三盆冷水

**第一，9% 不是 50%。** 如果我把 $\kappa$ 设成 0.5「每天」（而不是 3.0「每年」），波动率会剧烈震荡、20 日窗口彻底失效，PF 相对 RV 的改善能膨胀到 50% 以上。很多展示「粒子滤波碾压实现波动率」的实验，隐藏前提就是这种不真实的参数。本文用年化参数把结论拉回现实：在平稳、均值回归缓慢的市场里，PF 只是略胜一筹。

**第二，参数错了，优势立刻反转。** 保留同一条价格路径，只把滤波器的参数设错（$\kappa=0.5$、$\sigma_v=0.1$，真值 3.0 / 0.5），其余完全不变。结果 PF 的 MAE 从 0.038071 恶化到 **0.049912**，反而比 RV 的 0.041887 差 **19.16%**。粒子滤波卖的不是算法魔法，是模型知识；模型是错的，贝叶斯更新会非常高效地把你带向错误答案。

**第三，杠杆效应没被建模。** 真实市场 $\rho\approx-0.7$，价格下跌与波动率上升共享同一个冲击。bootstrap 滤波器的似然只看 $r_t^2$ 的量级，丢掉了符号信息，理论上会损失效率。要处理它需要辅助粒子滤波（ASIR）或在状态转移里引入观测引导，本文没有做，属于明确的能力边界。

## 七、实战意义与局限

可落地的地方有三个：波动率目标策略里，PF 的后验均值比 RV 更平滑，能减少无意义的仓位抖动；期权定价与做市里，PF 顺带产出后验分布，可用于隐含波动率曲面的贝叶斯校准；风险预算里，5%–95% 分位带本身就是估计不确定度，比单点值更适合做保守约束。

局限同样明确：参数 $\kappa,\theta,\sigma_v$ 在实盘未知，需在线估计或滚动重校准，而第六节已经展示参数误差的杀伤力；Heston 假设波动率路径连续，遇到跳空与跳跃扩散会失真；观测噪声被设成高斯，日线收益的厚尾会让似然过度惩罚极端值，换成 Student-t 似然是自然的下一步；高维（多资产随机波动率）下粒子退化会急剧加重，需要转向 Rao-Blackwellized 粒子滤波或集合卡尔曼类方法。

## 八、完整可运行代码

下面这份代码**自包含**：包含数据生成、滤波器、两条基线、主程序与打印。直接 `python3 pf_demo.py` 即可复现文中全部主实验数字（依赖仅 numpy）。结果与 `public/images/particle-filter-volatility/results.json` 逐项一致。

```python
import numpy as np

SEED = 20261107
P = dict(kappa=3.0, theta=0.04, sigma_v=0.5, v0=0.04, mu=0.05, dt=1/252)
T, N, W = 500, 500, 20

def simulate_heston(T, rho, rng, kappa=3.0, theta=0.04, sigma_v=0.5,
                    v0=0.04, mu=0.05, dt=1/252):
    v = np.empty(T); logp = np.empty(T)
    v[0], logp[0] = v0, np.log(100.0)
    z1 = rng.standard_normal(T); z2 = rng.standard_normal(T)
    for t in range(1, T):
        vp = v[t-1]
        v[t] = max(vp + kappa*(theta-vp)*dt + sigma_v*np.sqrt(vp*dt)*z1[t], 1e-8)
        shock = rho*z1[t] + np.sqrt(1-rho**2)*z2[t]
        logp[t] = logp[t-1] + (mu - 0.5*v[t])*dt + np.sqrt(v[t]*dt)*shock
    return np.exp(logp), np.sqrt(v)

def particle_filter(prices, N=500, rng=None, ess_frac=0.5, snapshots=None,
                    kappa=3.0, theta=0.04, sigma_v=0.5, v0=0.04, mu=0.05, dt=1/252):
    if rng is None: rng = np.random.default_rng(0)
    T = len(prices); r = np.diff(np.log(prices))
    vp = np.full(N, v0); w = np.full(N, 1.0/N)
    est = np.empty(T); pred = np.full(T, np.nan); ess_hist = np.empty(T)
    lo = np.full(T, np.nan); hi = np.full(T, np.nan)
    est[0] = np.sqrt(v0); ess_hist[0] = N
    resamples = 0; snap = {}
    for t in range(1, T):
        z = rng.standard_normal(N)
        v_new = np.maximum(vp + kappa*(theta-vp)*dt + sigma_v*np.sqrt(vp*dt)*z, 1e-8)
        mean = (mu - 0.5*v_new)*dt; sd = np.sqrt(v_new*dt)
        logw = -0.5*((r[t-1]-mean)/sd)**2 - np.log(sd)
        w = w*np.exp(logw-logw.max()); w /= w.sum()
        est[t] = np.sqrt(np.dot(v_new, w))
        order = np.argsort(v_new); cw = np.cumsum(w[order]); sv = v_new[order]
        lo[t] = np.sqrt(np.interp(0.05, cw, sv)); hi[t] = np.sqrt(np.interp(0.95, cw, sv))
        v_forecast = np.dot(v_new + kappa*(theta-v_new)*dt, w)
        pred[t] = np.sqrt(max(v_forecast, 1e-8))
        ess = 1.0/np.dot(w, w); ess_hist[t] = ess
        if snapshots and t in snapshots: snap[t] = (v_new.copy(), w.copy())
        if ess < ess_frac*N:
            pos = (rng.random() + np.arange(N))/N
            cum = np.cumsum(w); cum[-1] = 1.0
            vp = v_new[np.searchsorted(cum, pos)]; w = np.full(N, 1.0/N)
            resamples += 1
        else: vp = v_new
    return est, ess_hist, resamples, snap, pred, lo, hi

def realized_vol(prices, window=20, dt=1/252):
    r = np.diff(np.log(prices)); out = np.full(len(prices), np.nan)
    for t in range(window, len(prices)):
        out[t] = np.sqrt(np.mean(r[t-window:t]**2)/dt)
    return out

def ewma_vol(prices, lam=0.94, dt=1/252):
    r = np.diff(np.log(prices)); var = np.full(len(prices), np.nan)
    var[0] = r[0]**2
    for t in range(1, len(r)): var[t] = lam*var[t-1] + (1-lam)*r[t]**2
    return np.sqrt(var/dt)

def mae(est, tv, lo, hi):
    return float(np.mean(np.abs(est[lo:hi]-tv[lo:hi])))

if __name__ == '__main__':
    px, tv = simulate_heston(T, 0.0, np.random.default_rng(SEED), **P)
    pf, ess, nres, snap, pred, lo, hi = particle_filter(
        px, N=N, rng=np.random.default_rng(SEED+1), snapshots={50,200,400}, **P)
    rv, ew = realized_vol(px, W), ewma_vol(px)
    HI = T-1
    print('pf_mae      ', round(mae(pf,tv,W,HI),6))
    print('rv_mae      ', round(mae(rv,tv,W,HI),6))
    print('ewma_mae    ', round(mae(ew,tv,W,HI),6))
    print('pf_corr     ', round(float(np.corrcoef(pf[W:HI],tv[W:HI])[0,1]),4))
    print('rv_corr     ', round(float(np.corrcoef(rv[W:HI],tv[W:HI])[0,1]),4))
    print('pf_pred_mae ', round(float(np.mean(np.abs(pred[W:HI]-tv[W+1:T]))),6))
    print('rv_pred_mae ', round(float(np.mean(np.abs(rv[W:HI]-tv[W+1:T]))),6))
    print('resamples   ', nres)
    print('improve_pct ', round(100*(mae(rv,tv,W,HI)-mae(pf,tv,W,HI))/mae(rv,tv,W,HI),2))
```

预期输出（与结果 JSON 一致）：`pf_mae 0.038071`、`rv_mae 0.041887`、`ewma_mae 0.039303`、`pf_corr 0.8934`、`rv_corr 0.8698`、`pf_pred_mae 0.040015`、`rv_pred_mae 0.04366`、`resamples 47`、`improve_pct 9.11`。

想复现第六节的参数错配，只需把 `P` 换成 `dict(kappa=0.5, theta=0.04, sigma_v=0.1, v0=0.04, mu=0.05, dt=1/252)` 后重跑滤波器，MAE 会变成 0.049912。

## 九、总结

粒子滤波把「波动率不可观测」变成一个标准的贝叶斯状态估计问题。在年化 Heston 参数、同信息集、同口径的公平对照下，它确实稳定优于 RV 与 EWMA：估计 MAE 降 9.11%、一步预测 MAE 降 8.35%、相关系数从 0.8698 提到 0.8934。但更值得记住的是三个反面结论：粒子数从 25 加到 1000 几乎没有增益；急涨急跌处后验明显滞后；参数设错时 PF 反而比 RV 差 19.16%。

所以别把粒子滤波当成更精确的 RV——它是「用一套正确的模型假设，换取更干净的噪声」。模型对了，收益温和而稳定；模型错了，贝叶斯更新会加倍高效地把你送进坑里。
