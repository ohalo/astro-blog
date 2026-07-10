---
title: "贝叶斯资产配置：用共轭先验把观点变成权重"
description: "经典均值-方差在小样本下会被噪声牵着走。用正态-逆威沙特(NIW)共轭先验把「主观观点」编码成数学，后验均值自动向先验收缩，后验协方差带可信带，最终权重更稳健、更可解释。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 资产配置
  - 贝叶斯
  - 共轭先验
  - 均值方差
  - Python
language: Chinese
difficulty: advanced
---

经典均值-方差模型有个尴尬的现实：**它把样本均值和样本协方差当成「真值」直接塞进优化器**。但真实投资里，你往往只有 3~5 年的月度数据（30~60 个观测），用这么少的样本去估计 5 个资产的期望收益和 15 个独立的协方差项，噪声大到离谱——优化器会聪明地把权重塞给「样本均值最高、但可能只是运气最好」的那个资产。

结论先放这：**贝叶斯资产配置不把估计值当真值，而是把「$(μ,Σ)$ 本身就是随机变量」这件事显式建模进来。用正态-逆威沙特(NIW)共轭先验，后验期望收益会自动向我们预设的观点收缩，后验协方差自带可信带，最终算出来的权重既用了数据，又尊重了经验，还能量化「我有多不确定」。**

![样本均值 vs 后验均值：共轭先验把估计向观点收缩](/images/bayesian-asset-allocation/bayes_shrinkage.png)

## 一、为什么经典均值-方差在小样本会翻车

均值-方差有效前沿长这样（用样本均值 $x̄$ 和样本协方差 $S$）：

$$\min_w\; w^\top S w \quad\text{s.t.}\quad w^\top x̄ = \mu_p,\; w^\top \mathbf{1}=1$$

问题出在 $x̄$ 和 $S$ 的估计误差。当样本短，单个资产的样本均值可能大幅偏离真实期望：

- 某股票真实年化期望 10%，但 120 个月样本里恰好前半年涨得好，样本均值只有 2.7%——优化器会嫌弃它；
- 另一资产真实期望 5%，样本里碰巧走出 6%——优化器反而重仓。

这就是「误差最大化」(error maximization)：均值-方差对估计误差极度敏感，小样本下权重剧烈摆动。贝叶斯的解法是：**别假装知道 $μ,Σ$，给它们一个分布**。

## 二、共轭先验：把观点变成数学

我们假定收益 $r_t \sim N(μ, Σ)$。对 $(μ,Σ)$ 同时建模，最干净的组合是**正态-逆威沙特(Normal-Inverse-Wishart, NIW) 先验**——它是 $(μ,Σ)$ 的共轭先验，后验还能写成 NIW，闭式可解：

- 先验：$\mu|\Sigma \sim N(\mu_0, \Sigma/\kappa_0)$，$\Sigma \sim IW(\Lambda_0, \nu_0)$
- 观测 $T$ 期后，后验仍是 NIW，参数为

$$\kappa_n = \kappa_0 + T,\qquad \mu_n = \frac{\kappa_0\mu_0 + T\,\bar r}{\kappa_0 + T}$$

$$\nu_n = \nu_0 + T,\qquad \Lambda_n = \Lambda_0 + (T-1)S + \frac{\kappa_0 T}{\kappa_0+T}(\bar r-\mu_0)(\bar r-\mu_0)^\top$$

注意看 $\mu_n$ 这条公式——**它就是先验观点 $\mu_0$ 与样本均值 $\bar r$ 的加权平均，权重由先验强度 $\kappa_0$（等效样本量）和数据量 $T$ 决定**。$\kappa_0$ 越大，越信任数据；$\kappa_0$ 越小，越信任观点。这就是「把观点变成权重」的核心机制。

```python
import numpy as np

def niw_posterior(xbar, S, T, mu0, kappa0, nu0, Lambda0):
    """正态-逆威沙特后验的闭式参数"""
    kappa_n = kappa0 + T
    mu_n = (kappa0 * mu0 + T * xbar) / kappa_n          # 均值向先验收缩
    nu_n = nu0 + T
    diff = (xbar - mu0).reshape(-1, 1)
    Lambda_n = Lambda0 + (T - 1) * S + (kappa0 * T / kappa_n) * (diff @ diff.T)
    return mu_n, kappa_n, nu_n, Lambda_n
```

后验期望协方差（即我们要喂给优化器的 $\mathbb{E}[\Sigma|\text{data}]$）是：

$$\mathbb{E}[\Sigma|\text{data}] = \frac{\Lambda_n}{\nu_n - p - 1}$$

其中 $p$ 是资产个数。它同样被先验「正则化」了——不会像样本协方差那样在大样本缺失时变成病态矩阵。

## 三、从先验到后验：一次真实的贝叶斯更新

下面是一次完整模拟：5 个资产（3 股票 + 债券 + 商品），真实年化期望分别是 10/7/12/5/9%，我们只有 120 期（10 年月度）观测，先验观点设成「所有资产年化期望 7%」。

```python
rng = np.random.default_rng(20260711)
p = 5
true_mu = np.array([0.10, 0.07, 0.12, 0.05, 0.09])
true_cov = np.array([   # 真实月度协方差（略）
    [0.040, 0.018, 0.010, 0.006, 0.014],
    [0.018, 0.030, 0.008, 0.004, 0.010],
    [0.010, 0.008, 0.055, 0.012, 0.016],
    [0.006, 0.004, 0.012, 0.020, 0.006],
    [0.014, 0.010, 0.016, 0.006, 0.035],
])
T = 120
R = rng.multivariate_normal(true_mu/12.0, true_cov/12.0, size=T)
xbar, S = R.mean(axis=0), np.cov(R, rowvar=False)

mu0 = np.full(p, 0.07/12.0)        # 先验：年化 7%，换算成月度
kappa0, nu0 = 12.0, p + 2.0
Lambda0 = np.diag([0.04, 0.03, 0.05, 0.02, 0.035]) * nu0

mu_n, kappa_n, nu_n, Lambda_n = niw_posterior(xbar, S, T, mu0, kappa0, nu0, Lambda0)
print("样本均值(年化):", np.round(xbar*12, 3))
print("后验均值(年化):", np.round(mu_n*12, 3))   # 介于样本与 7% 之间
```

跑出来样本均值年化是 `[0.027, 0.060, 0.097, 0.060, 0.064]`，而后验均值年化变成 `[0.031, 0.061, 0.094, 0.061, 0.064]`——**极端值被往 7% 拉回，但又不至于完全压平**。这就是收缩的本质：数据弱时听观点的，数据强时听数据的。

后验预测分布（下期收益的分布）是自由度为 $\nu_n-p+1$ 的多元 $t$，比先验更窄、更贴近数据：

![后验预测分布：观测后比先验更紧、更贴近数据](/images/bayesian-asset-allocation/bayes_predictive.png)

## 四、后验均值-方差前沿与可信带

有了后验参数，前沿就不用样本 $(x̄, S)$，而用 $(\mu_n, \mathbb{E}[\Sigma|\text{data}])$。更妙的是，我们可以从 NIW 后验抽样一堆 $\Sigma$，每条都算一条前沿，包络出**可信带**——它真实地量化了「估计不确定性有多大」。

```python
from scipy.stats import wishart

def eff_frontier(mu, cov, npoints=60):
    inv = np.linalg.inv(cov); ones = np.ones(len(mu))
    A, B, C = ones@inv@ones, ones@inv@mu, mu@inv@mu
    D = A*C - B**2
    rets = np.linspace(mu.min()*1.2, mu.max()*1.2, npoints)
    vols = np.sqrt((A*rets**2 - 2*B*rets + C) / D)
    return rets, vols

E_Sigma_post = Lambda_n / (nu_n - p - 1)
rets_b, vols_b = eff_frontier(mu_n*12, E_Sigma_post*12)        # 贝叶斯前沿
rets_s, vols_s = eff_frontier(xbar*12, S*12)                   # 经典前沿

# 从 IW 后验抽样 Σ，画前沿可信带
Sigma_samples = [np.linalg.inv(
    wishart(df=int(round(nu_n)), scale=np.linalg.inv(Lambda_n)).rvs(random_state=rng))
    for _ in range(400)]
```

![贝叶斯前沿比经典前沿更靠内，可信带量化不确定性](/images/bayesian-asset-allocation/bayes_frontier.png)

直观上：经典前沿因为用了「乐观的样本均值」，会被推到更激进、更靠左上角的地方，但那是估计误差画出的海市蜃楼；贝叶斯前沿收缩后更保守、更靠内，可信带告诉你「真实前沿大概率落在这条带子里」。

## 四·五、收缩为什么是「最优」而非「妥协」

有人会问：把样本均值往先验上拉，不是人为引入了偏差吗？在数学上，这恰恰是**对付估计风险的最优解**。经典均值-方差把样本均值当真值，本质是「插件式估计器」(plug-in estimator)，它在平方损失下对期望收益的方差极敏感；而贝叶斯后验均值，是在「后验期望平方损失」意义下的最优估计——给定 NIW 先验，没有任何别的估计器能系统性地比它更低误差。

直觉上可以这样想：当你只有 120 期数据，单个资产期望收益的估计标准误可能有 2~3%（年化），而你的先验说「大概 7% 左右」。后验均值把两者按精度加权，**等价于你潜意识里承认「我的样本还不够确信」**。在估计误差巨大的小样本区，这种收缩带来的方差下降，远大于它引入的那点偏差——净效果是样本外表现更好。这也是为什么「收缩估计器」(shrinkage estimator)在组合优化文献里是被严格证明占优( dominated )插件估计器的少数几招之一。

更彻底的做法是直接**对后验期望效用最优化**，而不是把后验均值塞回经典优化器：

```python
# 后验预期效用：在 NIW 后验下对权重 w 求期望（以二次效用为例）
#  E[ w'μ - (λ/2) w'Σ w | data ] = w'μ_n - (λ/2) w' E[Σ|data] w
#  → 直接得到「贝叶斯均值-方差」权重，天然含收缩
lambda_risk = 3.0
def bayes_mv_weights(mu_n, E_Sigma, lam):
    inv = np.linalg.inv(lam * E_Sigma)
    return inv @ mu_n / (np.ones(p) @ inv @ mu_n)
w_bayes = bayes_mv_weights(mu_n*12, E_Sigma_post*12, lambda_risk)
```

这条路径把「不确定性」直接写进目标函数：风险项用的是后验期望协方差，而非某个点估计。它和上一节的收缩是同一枚硬币的两面——收缩只是这个更一般框架在均值上的投影。

## 五、先验强度如何控制权重

用全局最小方差组合（对预期收益不敏感，便于看协方差收缩）看先验强度 $\kappa_0$ 的影响：

```python
def min_var_weights(mu, cov):
    inv = np.linalg.inv(cov); w = inv @ np.ones(len(mu))
    return w / w.sum()

def weights_for_kappa(k0):
    mn, kn, nn, Ln = niw_posterior(xbar, S, T, mu0, k0, nu0, Lambda0)
    return min_var_weights(mn*12, Ln/(nn-p-1)*12)

ks = [1, 3, 6, 12, 30, 60, 120]
Wk = np.array([weights_for_kappa(k0) for k0 in ks])
```

![先验强度把权重从数据独断平滑拉向稳健区](/images/bayesian-asset-allocation/bayes_weights.png)

$\kappa_0$ 很小时（强观点），权重几乎由先验协方差和观点决定；$\kappa_0\to\infty$（完全信任数据），权重才滑向样本最小方差解。实务上，$\kappa_0$ 等于「你认为了解这个资产等效于多少期历史观测」——对不熟悉的新资产，给小一点；对数据长的宽基指数，给大一点。

**一个经验取值**：若你的观点来自扎实的基本面研究，可设 $\kappa_0$ 相当于 1~2 年（12~24 期）；若只是「市场长期股权溢价约 7%」这种弱信念，$\kappa_0$ 设 6~12 期即可，让数据主导。关键是不把 $\kappa_0$ 设到 0（纯观点，无视一切数据），也别设到 $\infty$（纯样本，重蹈误差最大化覆辙）。

## 六、它和 Black-Litterman 不是一回事

很多人会问：这不就是 Black-Litterman 吗？**不是。** Black-Litterman 是在市场均衡收益 $Π$ 上，用「观点矩阵 $P$ + 信心 $Ω$」做一次（通常是高斯）更新，本质仍是点估计，最终喂给均值-方差的还是「一个确定的后验收益向量」。而这里的 NIW 框架：

1. **同时建模了 $μ$ 和 $Σ$ 的不确定性**，BL 通常把 $Σ$ 当已知；
2. **来自严格的共轭贝叶斯推导**，先验强度有清晰的频率解释（$\kappa_0$ = 等效样本量）；
3. **天然给出可信带**，能回答「这个权重我有多大把握」，BL 给不出分布。

两者可以结合：把 BL 的均衡收益当成这里的 $\mu_0$，就得到一个既尊重市场、又尊重观点、还带不确定性的完整贝叶斯配置。

## 七、真实陷阱（别假装没看见）

- **先验不能乱设**：$\mu_0$ 设得太极端（比如坚信某资产 30%），后验会被观点绑架，等于放弃了数据。先验强度要反映你真正的置信度。
- **NIW 假设联合正态**：收益有肥尾时，正态似然会低估极端风险，后验协方差偏窄。可以用学生 $t$ 似然或尾部更厚的先验。
- **收缩不是免费的午餐**：收缩降低了方差，但引入了偏差（bias）。样本越长，$T$ 越大，收缩自动减弱，偏差也越小——这是 NIW 自带的「自适应」，不用手动调。
- **共轭是为了可解**：NIW 的优雅来自「共轭」，但它强制了特定的先验结构。若你的观点无法写成 NIW（比如非对称的「要么涨要么跌」），就得上 MCMC，那就超出本文范围了。

## 结语

贝叶斯资产配置的核心不是「更玄的数学」，而是**诚实地承认我们不知道真值，并把这种不知道量化出来**。NIW 共轭先验让「观点→收缩→后验→权重」变成一条可解释、可调参、可带可信带的流水线。在小样本、高噪声的实盘环境里，这种「先收缩、再优化」的纪律，往往比直接把样本均值塞进优化器稳健得多。

下一篇我们会聊另一个「把不确定性建模进来」的工具——GARCH 波动率模型族，看它如何把「波动的波动」也做成可预测、可交易的对象。
