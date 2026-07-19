---
title: "巴黎期权与部分触及障碍：给障碍加一段观察窗，敲出就不再靠运气"
description: "普通障碍期权只要价格「瞬时」碰到障碍线就当场敲出，太容易被毛刺和瞬时跳空误杀。巴黎期权(Parisian option)改了规则：必须「连续」低于/高于障碍累计达到一段观察窗 D 才算敲出。这一个「持续时长」维度，让障碍从瞬时事件变成累计过程，定价、对冲和行为含义都完全不同。本文用蒙特卡洛从零实现标准障碍 vs 巴黎式障碍，验证价格介于标准与欧式之间，扫描触及窗 D 的敏感性，并指出累计计时、离散化与路径依赖三类真实陷阱。"
publishDate: '2026-07-20'
tags:
  - 量化交易
  - 奇异期权
  - 巴黎期权
  - 障碍期权
  - Python
language: Chinese
difficulty: advanced
---

普通障碍期权（Barrier Option）有个反直觉的脆弱性：只要标的**瞬时**碰到障碍线，期权就当场作废（敲出）或生效（敲入）。问题是真实价格序列充满毛刺——一根瞬间下影线、一次夜盘跳空、一笔错单，都能把障碍触发。这对卖方是免费的「意外之喜」，对买方则是「还没反应过来就归零」。

巴黎期权（Parisian option，由 Chesney、Cornaglia 等在上世纪 90 年代提出）给障碍加了一道**观察窗（excursion window）**：价格必须**连续**低于（或高于）障碍累计达到一段时长 $D$，才算真正敲出。这就把「瞬时触碰」变成了「持续停留」——一个更贴近工程直觉、也更难被毛刺误杀的合约。

![标准障碍 vs 巴黎式障碍 vs 普通欧式：价格随障碍水平 B 的变化](/images/parisian-barrier-options/parisian_vs_standard.png)

## 一、为什么需要「观察窗」

设想一个下行敲出看涨期权（Down-and-Out Call, DOC）：约定标的跌破 $B=90$ 就作废。

- **标准障碍**：某天盘中瞬间插针到 89.9，收盘回到 95——期权当场敲出归零。买方毫无缓冲。
- **巴黎式障碍**：必须**连续**低于 90 累计满 $D$（比如 10 个交易日）才敲出。上面那次瞬插针只累积了很短的「触及计时」，远不到 $D$，期权安然无恙。

数学上，巴黎式引入一个**累计触及计时（excursion clock）** $C_t$：

$$C_t = \int_0^t \mathbf{1}_{\{S_u < B\}}\,du \quad \text{（下行巴黎式）}$$

敲出条件变为 $C_T \ge D$。$D=0$ 退化为标准障碍，$D\to\infty$ 退化为永不敲出的普通欧式。所以巴黎式期权是「标准障碍」与「欧式」之间的**连续过渡**——这给了我们一个漂亮的单调性检验。

## 二、蒙特卡洛定价框架

巴黎式没有通用闭式（连续监测版本可用可逆反射思想，但实务多用 MC）。核心是在每条模拟路径上维护「触及计时」：

- 每步若 $S_t < B$，计时器 $+dt$；否则清零（连续型 Parisian）。
- 一旦计时器 $\ge D$，标记该路径敲出，期末 payoff 直接为 0。
- 未敲出的路径，payoff = 普通看涨 $\max(S_T-K,0)$。

```python
import numpy as np
from scipy.stats import norm

S0, K, r, sig, T = 100.0, 100.0, 0.05, 0.25, 1.0
B = 90.0                      # 下行障碍
n_steps = 252; dt = T/n_steps # 日频

def mc_price(option="parisian", D=5*dt, n_paths=40000, seed=0):
    rg = np.random.default_rng(seed)
    Z = rg.standard_normal((n_paths, n_steps))
    log_s = np.log(S0) + np.cumsum(((r-0.5*sig**2)*dt + sig*np.sqrt(dt)*Z), axis=1)
    S = np.exp(log_s); ST = S[:, -1]
    if option == "vanilla":
        return np.exp(-r*T)*np.mean(np.maximum(ST-K, 0.0))
    below = S < B
    if option == "standard":
        knocked = below.any(axis=1)                 # 任一时点触障即敲出
    else:
        D_steps = max(1, int(round(D/dt)))
        knocked = np.zeros(n_paths, dtype=bool)
        run = np.zeros(n_paths)
        for j in range(n_steps):
            run = np.where(below[:,j], run+dt, 0.0)  # 连续低于才累加
            knocked = knocked | (run >= D)
    payoff = np.where(knocked, 0.0, np.maximum(ST-K, 0.0))
    return np.exp(-r*T)*np.mean(payoff)

van = mc_price("vanilla", seed=1)
std = mc_price("standard", seed=1)
par1 = mc_price("parisian", D=5*dt, seed=1)
par2 = mc_price("parisian", D=21*dt, seed=1)
print(f"欧式看涨 MC          = {van:.4f}")
print(f"标准下行敲出         = {std:.4f}  (最便宜)")
print(f"巴黎式(D=1周=5日)    = {par1:.4f}")
print(f"巴黎式(D=1月=21日)   = {par2:.4f}")
```

自洽校验结果：欧式 MC = 12.46，标准障碍 = 9.69，巴黎式（D=5日）= 10.62，巴黎式（D=21日）= 11.64。**单调性完美成立**：标准障碍（最易敲出）最便宜，巴黎式居中，欧式（永不敲出）最贵。

## 三、触及窗 D 的敏感性：一个连续旋钮

$D$ 是巴黎式期权的灵魂参数。它控制「多容易被敲出」：

- $D$ 越小，越接近标准障碍，期权越便宜（敲出越容易）。
- $D$ 越大，越难敲出，期权越贵，逼近欧式。

下面扫描 $D$ 从 0 到 63 个交易日，画出价格曲线——它应当是一条从「标准障碍下限」平滑爬升到「欧式上限」的单调曲线。

```python
D_days = np.array([0,1,3,5,10,15,21,42,63])
price_D = np.array([mc_price("parisian", D=d*dt, n_paths=30000, seed=3) for d in D_days])
# 绘制: price_D 随 D_days 上升, 下界 std、上界 van
```

![巴黎式障碍的触及窗敏感性：D=0→标准障碍，D→∞→欧式](/images/parisian-barrier-options/parisian_window.png)

这条曲线的实务意义是：**$D$ 不是技术参数，而是风险偏好旋钮**。一个担心被毛刺误杀的长线买方，会要求更大的 $D$（付更高的期权费，换取不被瞬时插针清掉）；一个相信价格会「真跌破并停留」的卖方，则偏好小 $D$。定价就是给这段「停留时长」标价。

## 四、路径可视化：累计计时如何决定生死

光看价格不够，得看一条真实路径上「触及计时」的演化。下面画一条模拟路径：它数次瞬时跌破 $B=90$（红色区间），但每次停留都不足 $D=10$ 日，所以计时器始终没到门槛——期权最终**未敲出**，白忙一场的插针没有杀死它。

```python
rg = np.random.default_rng(42)
S_path = S0*np.exp(np.cumsum((r-0.5*sig**2)*dt + sig*np.sqrt(dt)*rg.standard_normal(n_steps)))
D_demo = 10*dt
timer = np.zeros(n_steps)
for j in range(n_steps):
    timer[j] = timer[j-1] + dt if S_path[j] < B else 0.0
knocked = timer >= D_demo
# 上轴画 S_t 与障碍 B; 下轴画累计触及计时与门槛 D
```

![巴黎式障碍：仅当连续低于障碍≥D 才敲出，瞬时插针不足为惧](/images/parisian-barrier-options/parisian_excursion.png)

这张图把巴黎式的本质讲透了：**敲出不是看「有没有碰到」，而是看「停留够不够久」**。这也是它比标准障碍更难对冲的原因——Delta 在「计时器快到门槛」时会剧烈变化。

## 五、收敛性与路径数

MC 定价必须用足够路径数保证稳定。下面用 4 组不同路径数重复估计巴黎式价格，看误差棒随 $N$ 收缩：

```python
n_list = [2000,5000,10000,20000,40000,80000]
reps = [np.mean([mc_price("parisian",D=10*dt,n_paths=n,seed=s) for s in range(5,9)]) for n in n_list]
```

![巴黎式障碍 MC 收敛：价格随路径数稳定](/images/parisian-barrier-options/parisian_convergence.png)

实务建议：$N=40000$ 量级下，巴黎式 DOC 价格的 95% 置信区间已收窄到约 ±0.1，足够报价使用。比标准障碍略贵是因为「连续型累计计时」需要逐条路径逐时点维护 $C_t$，计算量线性于 $N\times n_{\text{steps}}$，但可以向量化这批循环提速。

## 六、两类巴黎式：连续型 vs 累计型

文献里巴黎式有两大变体，常被混淆：

- **连续型（Parisian / continuous）**：计时器一旦价格回到障碍之上就**清零**，必须「一口气」连续待够 $D$。本文用的就是这种。
- **累计型（Parisian window / cumulative）**：只要「总停留时长」相加够 $D$ 就敲出，中途离开不清零（更像「过去 $D$ 天内累计低于障碍的时间」）。

两者定价差异显著：累计型更容易敲出（因为停留可累积），所以比连续型便宜。签合约时务必看清是哪一种——这是结构性产品里最容易踩的语义坑。

## 七、真实陷阱

- **离散化陷阱（连续监测 vs 离散监测）**：本文用日频 252 步近似连续监测。真实合约多为「收盘价监测」或「日内连续监测」，离散监测会系统性**低估**敲出概率（毛刺更难触发），标准障碍偏差尤其大（可达几个百分点）。解法：要么用连续型解析校正，要么把 MC 步长加到日内级。
- **计时器清零逻辑陷阱**：连续型必须「离开即清零」，代码里写成累计型（不清零）会严重高估敲出概率、低估期权费。这是巴黎式实现最常见的 bug。
- **路径依赖对冲陷阱**：巴黎式的 Delta 在「计时器临界」附近极陡——价格贴近障碍且停留快到 $D$ 时，一个微小变动就能从「未敲出」跳到「已敲出」，Delta 瞬间塌缩。静态 Delta 对冲在这里会频繁巨额交易。解法：用障碍敏感度（巴黎式版「梯度/Greeks」）而非普通 Delta。
- **D 的单位陷阱**：合约写的是「10 个交易日」还是「10 个日历日」？涉及节假日、半日交易，必须和蒙特卡洛里 $dt$ 的口径对齐，否则价格差几个百分点。
- **双向巴黎式陷阱**：上敲出（up-and-out）的巴黎式需要维护「高于障碍」的计时器，且 $B>K$ 时边界条件与下行完全不同，直接复用下行代码会错。

## 八、落地清单

1. 先确认合约是**连续型**还是**累计型**巴黎式——这决定计时器清零逻辑，差之毫厘谬以千里。
2. MC 定价默认 $N\ge40000$、日频 252 步起；若要报价级精度，上到 $N=10^5$ 并向量化。
3. 做单调性自检：巴黎式价格必须严格介于标准障碍与欧式之间，否则计时器逻辑有 bug。
4. 对冲用巴黎式专属 Greeks（计时器敏感度），别拿普通 Delta 硬扛临界段的陡变。
5. 离散监测合约：先用连续监测 MC 定价，再按监测频率做离散偏差校正，不要直接把日频结果当报价。

## 结语

巴黎期权是障碍期权家族里「最讲道理」的一员：它把敲出从「瞬时运气」改成「持续停留」，用一段观察窗 $D$ 给买方喘息、给卖方定价一个连续旋钮。实现上它无非是在 MC 路径里多维护一个累计计时器——但就是这个计时器的「清零与否、口径对齐、临界对冲」，藏着巴黎式真正的工程陷阱。把它和上一篇的有限差分法结合起来，你就能处理更广义的「带停留约束的路径依赖期权」：那是奇异期权定价里最考验功底的一小块。

> 本文数值均用 Python 从零实跑自洽校验：欧式 MC = 12.46 / 标准障碍 = 9.69 / 巴黎式（D=5日）= 10.62 / 巴黎式（D=21日）= 11.64，单调性严格成立。代码可复现上述全部图表。
