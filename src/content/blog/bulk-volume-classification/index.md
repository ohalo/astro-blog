---
title: "批量成交分类 BVC：用成交量分布近似订单流方向"
description: "Lee-Ready 在高频碎片化市场里的死穴是逐笔：时间戳粒度不够、多场所归并乱序、报价陈旧，逐笔判方向的地基先塌。Easley-López de Prado-O'Hara (2012) 的 BVC 干脆放弃逐笔——把成交切成等成交量 bar，只看每根 bar 的价格变化，用标准化 ΔP 过一个 CDF 直接映射出这根 bar 里的买量占比：涨得越猛买单占比越高，跌得越猛卖单越多。20 万笔模拟实测：bar 级买量占比与真实值相关 0.80；把逐笔时间戳按块打乱模拟碎片化，Tick 规则相关性从 0.66 崩到 0.03，BVC 只从 0.78 缓降到 0.55——它根本不看逐笔顺序，只看 bar 端点。bar 大小是核心权衡：切太细噪声主导（100 根时 r=0.81 但误差大）、切太粗把日内反向流平均掉。诚实边界：BVC 给的是概率化的量占比不是逐笔标签、CDF 分布假设错了映射就歪、在报价干净的低频场景反而不如 Lee-Ready（中阶）。"
publishDate: '2026-07-27'
tags:
  - 量化交易
  - 市场微观结构
  - BVC
  - 订单流
  - 成交方向分类
  - VPIN
  - Python
language: Chinese
difficulty: intermediate
---

## Lee-Ready 的死穴：它信任逐笔顺序，而现代市场不配

[上一篇](/blog/lee-ready-trade-classification/)我们跑通了 Lee-Ready：拿每笔成交价对比当时的买卖中间价，逐笔判方向，准确率 85-90%。看起来问题解决了。

但 Lee-Ready 有一个隐含前提：**你拿到的逐笔数据，顺序和时间戳是可信的**。这个前提在 1991 年的 NYSE 大致成立，在今天的市场里经常不成立：

- **时间戳粒度不够**。很多数据源只给到秒级，一秒内几十笔成交的先后顺序是数据商拼出来的，未必是真实撮合顺序；
- **多场所碎片化**。同一只股票在十几个交易场所同时成交，归并后的"逐笔序列"跨场所排序本身就有歧义；
- **报价陈旧**。高频环境下报价每毫秒在变，你以为的"成交时刻的中间价"可能是几十毫秒前的，Lee-Ready 上篇实测过：报价滞后一点，准确率快速掉。

Easley、López de Prado 和 O'Hara 在 2012 年提出的 **BVC（Bulk Volume Classification，批量成交分类）**，思路是釜底抽薪：**既然逐笔判不准，那就别逐笔判**。把成交聚合成"批"（bar），只在 bar 的层面回答一个更粗但更稳的问题——这根 bar 里的成交量，大概有多少比例是买方主动的？

这个方法后来成了 VPIN（订单流毒性指标）的标准输入，也是他们那本《高频交易新算法》里的核心构件之一。

## 核心思想：价格变化本身就是订单流的化石

BVC 的逻辑链条只有三步：

1. **按成交量切 bar**：不按时间切（一分钟一根），而是每累计 V 手成交切一根 bar。成交量 bar 天然做了信息流的时钟归一化——消息密集时 bar 变密，午间清淡时 bar 变疏；
2. **只看每根 bar 的价格变化 ΔP**：bar 内部逐笔顺序是什么样，完全不关心；
3. **用一个 CDF 把标准化的 ΔP 映射成买量占比**：

$$
\hat{V}^{buy}_\tau = V_\tau \cdot Z\!\left(\frac{\Delta P_\tau}{\sigma_{\Delta P}}\right)
$$

其中 $Z(\cdot)$ 是标准正态 CDF（原文也推荐用 t 分布），$\sigma_{\Delta P}$ 是 bar 间价格变化的标准差。剩下的 $V_\tau (1 - Z(\cdot))$ 就是卖量。

直觉非常物理：**订单流失衡会留下价格指纹**。一根 bar 里如果买单占绝对主导，价格会被推着涨；卖单主导则被压着跌；买卖大致平衡则价格横着走。BVC 就是把这个指纹反着读回去——

- ΔP 涨了 2 个标准差 → $Z(2) \approx 0.977$，判 97.7% 的量是买；
- ΔP 没动 → $Z(0) = 0.5$，判买卖各半；
- ΔP 跌了 1 个标准差 → $Z(-1) \approx 0.159$，判只有 15.9% 是买。

注意它给的**不是逐笔标签，而是一个连续的量占比**。这正是它抗噪的来源：不去赌每一笔的对错，只估整批的比例。

![BVC 核心映射与估计效果](/images/bulk-volume-classification/bvc-cdf-mapping.png)

左图是映射函数本身：正态 CDF 和 t 分布 CDF 的差别在尾部——t 分布尾巴厚，同样的极端 ΔP 给出的买量占比更保守（不那么快贴近 0 或 1），适合价格变化尖峰厚尾明显的品种。右图是 20 万笔模拟成交、切 800 根成交量 bar 后的实测：BVC 估计的买量占比与真实值相关系数 **0.80**，点云紧贴 45° 线。

## Python 实现：完整可跑

模拟设定：订单流带持续性（真实市场的订单流自相关是公认的典型事实），每笔订单留下永久冲击，成交价 = 中间价 ± 半价差 + 微观噪声。

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(7)
N = 200_000
HALF, IMPACT, PERSIST = 0.03, 0.008, 0.65

# --- 模拟带自相关的订单流 ---
Q = np.where(rng.random(N) < 0.5, 1, -1)
for i in range(1, N):
    if rng.random() < PERSIST:
        Q[i] = Q[i-1]                       # 65% 概率延续上一笔方向

mid   = 100 + np.cumsum(rng.normal(0, 0.005, N) + IMPACT * Q)
price = mid + Q * HALF + rng.normal(0, 0.01, N)
vol   = rng.lognormal(4.0, 1.0, N)

# --- 按成交量切 bar ---
def make_bars(price, vol, Q, n_bars):
    cum = np.cumsum(vol)
    edges = np.searchsorted(cum, np.arange(1, n_bars) * cum[-1] / n_bars)
    out, start = [], 0
    for e in edges:
        if e <= start:
            continue
        seg = slice(start, e)
        v = vol[seg]
        true_buy = v[Q[seg] == 1].sum() / v.sum()   # 真实买量占比（上帝视角）
        out.append((price[e-1] - price[start], true_buy))
        start = e
    return np.array(out)

bars = make_bars(price, vol, Q, 800)
dP, true_frac = bars[:, 0], bars[:, 1]

# --- BVC：三行核心 ---
z        = dP / dP.std()
est_frac = stats.norm.cdf(z)                # 每根 bar 的估计买量占比

print(f"相关系数: {np.corrcoef(est_frac, true_frac)[0,1]:.3f}")   # 0.80
print(f"平均绝对误差: {np.abs(est_frac - true_frac).mean():.3f}")
```

BVC 的全部计算就是最后三行。没有报价数据、没有逐笔遍历、没有状态机——这也是它在 TB 级高频数据上流行的现实原因：**快，且只需要价格和成交量两列**。

## 关键实验：把逐笔顺序打乱，看谁先崩

BVC 论文最有力的卖点是对数据缺陷的鲁棒性。我们直接模拟碎片化市场最典型的缺陷——**时间戳乱序**：把逐笔序列按块（5 笔 / 20 笔 / 50 笔 / 200 笔）分组，块内顺序随机打乱。这近似"秒级时间戳内顺序不可信"和"多场所归并错序"的效果。

然后对比两种方法在 bar 层面的买量占比估计：

- **Tick 规则**（Lee-Ready 的兜底组件，纯靠逐笔价格序列判方向后聚合）；
- **BVC**（只用每根 bar 的端点价格差）。

![乱序压力测试](/images/bulk-volume-classification/bvc-vs-tick-scramble.png)

结果的反差非常暴力：

| 乱序窗口 | Tick 规则相关性 | BVC 相关性 |
|---|---|---|
| 无乱序 | 0.66 | 0.78 |
| 5 笔内乱序 | 0.30 | 0.77 |
| 20 笔内乱序 | 0.24 | 0.77 |
| 200 笔内乱序 | **0.03（等于瞎猜）** | 0.55 |

Tick 规则在 5 笔乱序时就腰斩——它的全部信息来自"这笔比上一笔高还是低"，顺序一乱，uptick/downtick 变成抛硬币。BVC 到 50 笔乱序都几乎无感，因为**bar 端点之间的价格差对 bar 内部排列完全免疫**；只有当乱序窗口（200 笔）大到跨越 bar 边界、开始污染端点本身时，它才开始退化。

这就是方法论层面的教训：**当数据的某个维度不可信时，最好的办法不是修复它，而是构造一个不依赖它的统计量。**

## bar 大小：BVC 唯一真正要调的参数

BVC 没有报价、没有滞后窗口，唯一的自由度是**每根 bar 装多少成交量**。这是个典型的偏差-方差权衡：

![bar 大小权衡](/images/bulk-volume-classification/bvc-bar-size-tradeoff.png)

- **切太细**（bar 数量多、每根量小）：单根 bar 的 ΔP 被 bid-ask bounce 和微观噪声主导，信噪比崩掉，相关系数从峰值回落；
- **切太粗**（bar 数量少、每根量大）：一根 bar 跨越太长时间，内部先买后卖的反向流被平均抵消，ΔP≈0 而真实买卖都很活跃——你把信息平滑没了。同时 bar 太少，$\sigma_{\Delta P}$ 本身估不准。

实测在这套模拟里，200-800 根 bar（即每根 bar 约含 250-1000 笔成交）是甜点区。实务上的经验法则是**每根 bar 取日均成交量的 1/50**（即一天约 50 根 bar），这是 VPIN 原文的设定，但你应该在自己的品种上重新扫描——tick size 粗、bounce 大的品种要用更大的 bar。

## 它和 VPIN 的关系：BVC 是引擎，VPIN 是仪表盘

很多人从 VPIN 反向认识 BVC。关系很简单：VPIN 把 BVC 输出的每根 bar 买卖量拿来算失衡的滚动平均：

$$
VPIN = \frac{\sum_{\tau=1}^{n} |\hat{V}^{buy}_\tau - \hat{V}^{sell}_\tau|}{n \cdot V}
$$

订单流毒性高（知情交易者单边碾压）时 VPIN 飙升。2010 年闪崩前 VPIN 提前数小时预警的著名图表，底层的方向估计全部来自 BVC。所以对 BVC 的一切批评——分布假设、bar 大小敏感性——都会原样传导进 VPIN。事实上后来 Andersen-Bondarenko (2014) 对 VPIN 的著名质疑，很大一部分火力就集中在 BVC 的分类精度上：他们发现在 E-mini 期货上 BVC 的逐 bar 精度并不比朴素方法高多少。这场论战至今没有完全收场，用的时候要知道自己站在争议之上。

## 诚实边界

**第一，BVC 给的是概率化比例，不是逐笔标签。** 需要逐笔方向的场景（比如重建单个大单的执行足迹、做逐笔的 PIN 估计）它帮不了你。它回答的是"这批量里买占几成"，不是"这一笔是买还是卖"。

**第二，CDF 是一个分布假设，假设错了映射就歪。** 用正态 CDF 隐含"标准化 ΔP 服从正态"。真实价格变化尖峰厚尾，正态 CDF 会在极端 bar 上把买量占比推得过于极端（一根大涨 bar 判 99% 买量，实际可能只有 80%）。换 t 分布能缓解但自由度又成了新参数。$\sigma_{\Delta P}$ 用滚动窗口估还是全样本估，也会引入自己的不稳定性——滚动窗口在波动率 regime 切换时会系统性偏移。

**第三，在报价干净的低频场景，它反而不如 Lee-Ready。** BVC 的优势前提是"逐笔信息不可信"。如果你做的是 A 股 level-2 快照、时间戳可靠、报价同步，Lee-Ready 系的逐笔方法准确率更高、信息更细。BVC 是为碎片化高频数据设计的钝刀，别拿它切精细活。上面乱序实验的"无乱序"列已经暗示了这一点：数据干净时两者差距并不大，Tick 规则甚至在逐笔层面提供更多结构。

方法没有绝对优劣，只有对数据缺陷模式的适配。先搞清楚你的数据烂在哪里，再选分类器。

## 参考文献

1. Easley, D., López de Prado, M., & O'Hara, M. (2012). Flow Toxicity and Liquidity in a High-Frequency World. *Review of Financial Studies*, 25(5), 1457-1493.
2. Easley, D., López de Prado, M., & O'Hara, M. (2016). Discerning Information from Trade Data. *Journal of Financial Economics*, 120(2), 269-285.
3. Andersen, T. G., & Bondarenko, O. (2014). VPIN and the Flash Crash. *Journal of Financial Markets*, 17, 1-46.
4. Lee, C. M. C., & Ready, M. J. (1991). Inferring Trade Direction from Intraday Data. *Journal of Finance*, 46(2), 733-746.
