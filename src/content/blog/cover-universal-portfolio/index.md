---
title: "Cover 通用投资组合：不预测市场也能逼近最优常数再平衡"
description: "Thomas Cover 1991 年提出的通用投资组合（Universal Portfolio）是在线学习在金融里最漂亮的定理之一：不做任何统计假设、不预测任何收益，只靠把资金摊到所有常数再平衡组合（CRP）上做『财富加权平均』，就能保证长期增长率逼近事后最优 CRP——最坏情况下的财富比值只以多项式速度衰减，而财富本身是指数增长的。2000 日双资产合成实验完整复现：事后最优 b*=0.65 的 CRP 终值 1.98，UP 无预测拿到 1.85，权重轨迹从均匀先验 0.5 被财富加权逐渐吸向 b*；附赠 Shannon's Demon 演示——一个长期不涨的资产加不生息的现金，50/50 每日再平衡照样把组合从 1.0 做到 1.47，波动本身就是收益来源。同时诚实交代三盆冷水：网格维度灾难、交易成本会吃掉再平衡收益、以及 UP 的保证是渐近的——有限样本内它可以跑输朴素 50/50。"
publishDate: '2026-07-30'
tags:
  - 量化交易
  - 在线学习
  - 投资组合
  - 通用组合
  - Python
language: Chinese
difficulty: advanced
---

## 一句话版本

**Cover 通用投资组合（Universal Portfolio, UP）**：把初始资金按均匀先验摊到*所有*常数再平衡组合（CRP）上，让每个 CRP 各自滚动，组合权重自动等于「财富加权平均」。Cover (1991) 证明：不管市场怎么走——甚至是对抗性设计的价格序列——UP 的长期指数增长率都收敛到**事后最优 CRP** 的增长率。不预测、无假设、有下界保证。

---

## 一、问题设定：什么叫「事后最优 CRP」

先固定语言。市场有 $m$ 个资产，第 $t$ 天的**价格相对**（price relative）向量为

$$
\mathbf{x}_t = (x_{t,1}, \dots, x_{t,m}), \qquad x_{t,i} = \frac{P_{t,i}}{P_{t-1,i}}
$$

即今天收盘价除以昨天收盘价。一个**常数再平衡组合（Constant Rebalanced Portfolio, CRP）**由权重向量 $\mathbf{b}$（非负、和为 1）刻画：每天收盘后把持仓调回固定比例 $\mathbf{b}$。$n$ 天后的财富是

$$
S_n(\mathbf{b}) = \prod_{t=1}^{n} \mathbf{b}^\top \mathbf{x}_t
$$

**事后最优 CRP** 就是让 $S_n(\mathbf{b})$ 最大的那个 $\mathbf{b}^*$——注意它需要看完全部 $n$ 天数据才能算出来，实盘上没人提前知道。它是一个很强的基准：

- 它自动包含「全仓持有最好的单个资产」（$\mathbf{b}$ 取角点）；
- 它还包含所有买入持有做不到的**再平衡收益**——高抛低吸的机械化版本。

Cover 的问题是：**存在不存在一个不偷看未来的策略，长期增长率追平 $\mathbf{b}^*$？** 答案是存在，而且构造出奇地简单。

## 二、UP 的构造：财富加权的贝叶斯混合

UP 的第 $t+1$ 天权重定义为

$$
\hat{\mathbf{b}}_{t+1} = \frac{\int_{\Delta_m} \mathbf{b}\, S_t(\mathbf{b})\, d\mathbf{b}}{\int_{\Delta_m} S_t(\mathbf{b})\, d\mathbf{b}}
$$

其中 $\Delta_m$ 是权重单纯形。直觉一句话：**想象把 1 块钱均匀分给单纯形上所有 CRP，各自独立滚动；UP 在第 $t+1$ 天的持仓，恰好就是这堆「CRP 基金」当前总持仓的合计**。表现好的 CRP 财富涨得快，自然在平均里占更大话语权——这就是财富加权。

从这个「基金的基金」视角立刻得到 UP 的财富恒等式：

$$
S_n^{UP} = \int_{\Delta_m} S_n(\mathbf{b})\, d\mathbf{b} \Big/ \int_{\Delta_m} d\mathbf{b}
$$

即 UP 的财富 = 所有 CRP 财富的均匀平均。这一步没有任何近似。

**核心定理（Cover 1991；Cover & Ordentlich 1996）**：对任意收益序列，

$$
\frac{S_n(\mathbf{b}^*)}{S_n^{UP}} \le C \cdot n^{(m-1)/2}
$$

财富比值最多以多项式 $n^{(m-1)/2}$ 增长，而财富本身是 $e^{\Theta(n)}$ 指数增长的。取对数除以 $n$：**UP 与事后最优的年化增长率之差以 $O(\frac{m \log n}{n}) \to 0$ 消失**。这是最坏情况保证——不需要 i.i.d.，不需要平稳性，价格序列哪怕是对手故意设计的都成立。

为什么可能？因为 $S_n(\mathbf{b})$ 作为 $\mathbf{b}$ 的函数是 $n$ 个线性函数的乘积，在 $\mathbf{b}^*$ 附近有一个体积约 $n^{-(m-1)/2}$ 的邻域，其中的 CRP 财富都与峰值同数量级。均匀平均至少能拿到「峰值 × 峰邻域体积」，于是只输一个多项式因子。**这和贝叶斯混合模型的 regret 界是同一个数学**——UP 本质上是 Krichevsky–Trofimov 估计的投资版。

## 三、代码：双资产市场上的完整复现

双资产时单纯形是一维区间 $b \in [0,1]$（$b$ 给资产 A），积分退化成一维网格求和，可以精确到没有实现借口：

```python
import numpy as np

rng = np.random.default_rng(7)
T = 2000

# ---- 合成两资产市场：负相关，样本内漂移受控 ----
z1, z2 = rng.normal(size=T), rng.normal(size=T)
rho = -0.35
za = z1 - z1.mean()
zb = (rho * z1 + np.sqrt(1 - rho**2) * z2)
zb -= zb.mean()
xa = np.exp(0.00030 + 0.022 * za)   # 资产A：高波动高漂移
xb = np.exp(0.00020 + 0.009 * zb)   # 资产B：低波动低漂移

# ---- 所有 CRP 的财富曲面：logS[t, j] = log S_t(b_j) ----
bs = np.linspace(0, 1, 101)
port_rel = bs[None, :] * xa[:, None] + (1 - bs[None, :]) * xb[:, None]
logS = np.cumsum(np.log(port_rel), axis=0)          # (T, 101)

# 事后最优 CRP
j_star = logS[-1].argmax()
b_star = bs[j_star]                                  # 0.65

# ---- UP 财富：所有 CRP 财富的均匀平均（log-sum-exp 防溢出）----
m_ = logS.max(axis=1, keepdims=True)
S_up = np.exp(m_.squeeze()) * np.mean(np.exp(logS - m_), axis=1)

# ---- UP 每日权重：财富加权平均的 b ----
S_prev = np.vstack([np.ones((1, len(bs))),
                    np.exp(logS[:-1] - logS[:-1].max(axis=1, keepdims=True))])
b_up = (S_prev * bs).sum(axis=1) / S_prev.sum(axis=1)

print(f"b* = {b_star:.2f}")
print(f"终值  A: {np.exp(logS[-1, -1]):.2f}  B: {np.exp(logS[-1, 0]):.2f}")
print(f"     CRP*: {np.exp(logS[-1, j_star]):.2f}   UP: {S_up[-1]:.2f}")
```

三处实现细节值得强调：

1. **log-sum-exp**：2000 天的财富乘积在 log 域动辄 ±几十，直接 `exp` 再平均会溢出或下溢，必须先减去按行最大值；
2. **权重用的是 $S_{t-1}$ 不是 $S_t$**：第 $t$ 天开盘调仓时只能用截至 $t-1$ 收盘的财富信息——这里放错一格就是 look-ahead；
3. **网格积分的偏差**：101 个点的黎曼和对一维单纯形绰绰有余（财富曲面是光滑单峰的），但资产数上去后网格点数是 $O(k^{m-1})$，见第五节。

### 实验结果

![四种策略的累积财富对比](/images/cover-universal-portfolio/up-wealth-comparison.png)

2000 个交易日跑完：资产 A 单独持有终值 **1.82**，资产 B 终值 **1.49**，事后最优 CRP（$b^*=0.65$）终值 **1.98**，UP 无预测拿到 **1.85**。UP 曲线几乎贴着最优 CRP 走，中途资产 A 深度回撤时 UP 的回撤也明显小于全仓 A。

![CRP 财富地形与 UP 终值](/images/cover-universal-portfolio/crp-landscape.png)

CRP 终值作为 $b$ 的函数是一条单峰曲线，峰在 $b^*=0.65$——**注意这个峰值位置只有看完全部数据才知道**。UP 的终值（绿色虚线）已经顶到了峰值的 93%，而它每一天的决策都不依赖未来。

![UP 权重从 0.5 漂向 b*](/images/cover-universal-portfolio/up-weight-evolution.png)

权重轨迹是整个算法最直观的一张图：从均匀先验 $b=0.5$ 出发，随着高 $b$ 区域的 CRP 财富领先，财富加权平均被逐渐「吸」向 $b^*$。注意它不是单调逼近——资产 A 每次回撤，权重就往回缩一点。这正是贝叶斯后验随证据摇摆的样子。

## 四、再平衡为什么本身就赚钱：Shannon's Demon

UP 追的基准是 CRP 而不是买入持有，原因是 CRP 有一块买入持有拿不到的收益——**波动收割（volatility pumping）**。极端演示：一个长期不涨的高波动资产（日 log 收益零漂移，2000 日后价格回到原点附近）加上不生息的现金，50/50 每日再平衡：

```python
lc = rng.normal(0, 0.045, 1500)
lc -= lc.mean()                       # 强制资产本身 1500 日后归零涨幅
xc = np.exp(lc)
S_rebal = np.cumprod(0.5 * xc + 0.5)  # 与现金 50/50 每日再平衡
print(np.cumprod(xc)[-1], S_rebal[-1])   # 1.00 vs 1.47
```

![Shannon's Demon 波动收割演示](/images/cover-universal-portfolio/volatility-harvesting.png)

资产本身终值 **1.00**（不赚钱），现金收益为零，但再平衡组合终值 **1.47**。钱从哪来？每次资产暴跌，再平衡强迫你低位补货；每次暴涨，强迫你高位减仓。数学上，组合的 log 增长率约为 $\frac{1}{2}\mu_{arith} - \frac{1}{8}\sigma^2 + \frac{1}{2}\cdot\frac{\sigma^2}{4}$ 里多出一项**方差红利** $\approx b(1-b)\sigma^2/2$。这就是 Shannon 上世纪在 MIT 讲座里演示的「Shannon's Demon」。波动率越大、资产间相关性越低（甚至为负），这块收益越厚——也解释了为什么本文实验刻意把两资产相关系数设成 $-0.35$。

## 五、三盆冷水：UP 在实战中的真实地位

**冷水一：维度灾难。** 均匀网格的点数按 $O(k^{m-1})$ 爆炸，$m=10$ 个资产、每维 20 个格点就是 $20^9 \approx 5 \times 10^{11}$。Kalai & Vempala (2002) 给出了基于 MCMC 采样的多项式时间实现，但工程复杂度直线上升。实践里更常见的替代品是同一族在线学习算法的轻量成员：指数梯度 EG（Helmbold et al. 1998，每步 $O(m)$）和在线牛顿步 ONS（Agarwal et al. 2006，regret 界更紧到 $O(m \log n)$）。

**冷水二：交易成本。** CRP 和 UP 都要求每日调仓，而波动收割的单日收益量级是 $b(1-b)\sigma^2/2$——日波动 2% 时约 0.5bp。单边成本超过几个 bp，方差红利就被啃穿。Blum & Kalai (1999) 证明了带比例交易成本版本的 UP 依然有通用性保证，但常数明显变差。实盘处理通常是降频（周/月再平衡）或加不交易带，代价是收割效率下降。

**冷水三：渐近保证 ≠ 有限样本优势。** 定理说的是增长率**渐近**追平，衰减速度 $O(\frac{m\log n}{n})$ 在 $n=2000$、$m=2$ 时仍是可感知的拖累。本文实验里 UP 终值 1.85，其实略低于朴素 50/50 再平衡的 1.95——因为 UP 把一部分资金「浪费」在了 $b$ 接近 0 和 1 的差 CRP 上，这是它为最坏情况保险支付的保费。**UP 的价值主张是下界，不是上界**：它保证你永远不会离事后最优太远，而不保证你击败任何具体对手。市场如果恰好平稳且你恰好猜对了权重，固定 CRP 当然更好——问题是你事前不知道。

## 六、和相邻方法的关系图谱

| 方法 | 更新方式 | Regret 界 | 单步计算 |
|---|---|---|---|
| Cover UP | 财富加权贝叶斯混合 | $O(m \log n)$ | 指数（网格）/ 多项式（采样） |
| 指数梯度 EG | 乘性梯度更新 | $O(\sqrt{n \log m})$ | $O(m)$ |
| ONS 在线牛顿步 | 二阶信息投影 | $O(m \log n)$ | $O(m^2)$ |
| Kelly 准则 | 需要收益分布 | 无对抗保证 | — |

一个常见误读要澄清：UP 不是 Kelly 的替代品。Kelly 假设你**知道**收益分布，在分布内最优；UP 假设你**什么都不知道**，在对抗序列上有保证。市场平稳时 Kelly（等价于直接持有 $\mathbf{b}^*$）更强；市场结构漂移或你对分布没信心时，UP 一族的免疫性才体现价值。两者的连接点在于：i.i.d. 市场里 UP 的权重会收敛到 Kelly 最优点——它是「不知道分布时的 Kelly 学习器」。

## 七、结论

- **定理层面**：UP 用一个不含任何预测的构造，把「追平事后最优常数再平衡」从愿望变成了带 $O(\frac{m\log n}{n})$ 速率的最坏情况保证，是在线学习进入投资组合理论的开山之作；
- **实验层面**：2000 日双资产复现里 UP 终值 1.85 vs 事后最优 1.98，权重轨迹清晰展示了财富加权后验从 0.5 漂向 $b^*=0.65$ 的全过程；
- **机制层面**：CRP 基准的含金量来自波动收割——零漂移资产加现金的 50/50 再平衡 1500 日做到 1.47，方差红利 $b(1-b)\sigma^2/2$ 是实打实的收益项；
- **实战层面**：维度灾难、交易成本、有限样本保费三盆冷水都泼在正确的位置上。UP 在今天的价值更多是作为 EG / ONS 等可实现在线组合算法的理论锚点，以及「不预测也有下界」这一思维方式本身。

## 参考文献

- Cover, T. M. (1991). Universal Portfolios. *Mathematical Finance*, 1(1), 1-29.
- Cover, T. M., & Ordentlich, E. (1996). Universal Portfolios with Side Information. *IEEE Transactions on Information Theory*, 42(2), 348-363.
- Blum, A., & Kalai, A. (1999). Universal Portfolios With and Without Transaction Costs. *Machine Learning*, 35, 193-205.
- Kalai, A., & Vempala, S. (2002). Efficient Algorithms for Universal Portfolios. *Journal of Machine Learning Research*, 3, 423-440.
- Helmbold, D. P., Schapire, R. E., Singer, Y., & Warmuth, M. K. (1998). On-Line Portfolio Selection Using Multiplicative Updates. *Mathematical Finance*, 8(4), 325-347.
- Agarwal, A., Hazan, E., Kale, S., & Schapire, R. E. (2006). Algorithms for Portfolio Management Based on the Newton Method. *ICML 2006*.
