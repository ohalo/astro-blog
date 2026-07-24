---
title: "向量化回测引擎：用 NumPy 广播把万次模拟压进一次矩阵运算"
description: "for 循环回测一根 K 线一根 K 线地跑，参数扫描时慢到让人放弃——十万根 K 线、两百组参数，循环版可能跑几分钟。向量化换个思路：把信号、持仓、收益全部写成 NumPy 数组运算，让 C 层的广播一次性算完所有 K 线甚至所有参数组合，同样的活压进一次矩阵乘法，实测提速 300~500 倍。本文从零手写一个向量化回测引擎：先讲循环版慢在哪，再用累积和算移动均线、用布尔数组生成信号、用 shift 严守次日执行不偷看未来、用三维广播一次算出 200 组参数的 Sharpe 热图，最后跑 2000 条蒙特卡洛路径压力测试。附速度换内存的取舍分析——向量化不是免费午餐，参数组合爆炸时内存会先崩（中阶）。"
publishDate: '2026-07-25'
tags:
  - 量化交易
  - 向量化
  - NumPy
  - 回测引擎
  - 性能优化
  - 参数扫描
  - Python
language: Chinese
difficulty: intermediate
---

先说结论：**如果你的回测还在用 `for i in range(len(df))` 一根一根 K 线地跑，参数扫描时你正在浪费 99% 的时间。** 把整个回测改写成 NumPy 数组运算，让广播（broadcasting）在 C 层一次算完所有 K 线、甚至所有参数组合，实测能提速 300 到 500 倍。本文从零手写这样一个引擎。

但向量化不是魔法，它用**内存换速度**。等你想同时评估一万组参数时，内存会先于时间崩掉。文章最后会讲这个取舍。

## 循环版回测慢在哪

先看一个最朴素的双均线策略循环回测，感受一下它的问题：

```python
import numpy as np
import pandas as pd

def backtest_loop(prices, fast=10, slow=30):
    """朴素循环版：一根 K 线一根 K 线地推进"""
    n = len(prices)
    ma_fast = np.full(n, np.nan)
    ma_slow = np.full(n, np.nan)
    position = np.zeros(n)
    equity = np.ones(n)

    for i in range(n):
        # 每根 K 线重新切片算均线 —— O(N*W) 的灾难
        if i >= fast - 1:
            ma_fast[i] = prices[i - fast + 1 : i + 1].mean()
        if i >= slow - 1:
            ma_slow[i] = prices[i - slow + 1 : i + 1].mean()
        # 信号：快线上穿慢线满仓，否则空仓
        if i >= slow:
            if ma_fast[i] > ma_slow[i]:
                position[i] = 1.0
        # 收益：昨天的持仓吃今天的涨跌（次日执行的雏形）
        if i > 0:
            ret = prices[i] / prices[i - 1] - 1
            equity[i] = equity[i - 1] * (1 + position[i - 1] * ret)

    return equity, position
```

这段代码有两层慢：

1. **Python 解释器的逐行开销**。每次循环都要做类型检查、边界检查、对象装箱，单根 K 线的常数开销是 C 的几十到上百倍。
2. **重复计算**。`prices[i-fast+1:i+1].mean()` 每根 K 线都从头加一遍窗口内的值，把 O(N) 的移动均线做成了 O(N×W)。

单次回测跑十万根 K 线也许能忍（几百毫秒），但**参数扫描才是真正的噩梦**：快线取 5~30、慢线取 30~120，就是两百多组组合，每组都从头跑一遍循环——几分钟起步。

![循环 vs 向量化的耗时随数据量变化，双对数坐标下差距稳定在数百倍](/images/vectorized-backtest-engine/speed_comparison.png)

上图是两种实现的耗时对比（双对数坐标）。注意随着 K 线数量增长，向量化版本始终稳定快 200~500 倍，且斜率更平——因为它把逐行的解释器开销彻底消掉了。

## 第一步：向量化移动均线

移动均线的向量化关键是**累积和（cumulative sum）**。相邻两个前缀和相减，就是窗口内的和，一次 `cumsum` 把 O(N×W) 压成 O(N)：

```python
def sma_vectorized(x, window):
    """用累积和 O(N) 算移动均线，无 Python 循环"""
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out = (csum[window:] - csum[:-window]) / window
    # 前 window-1 个位置没有完整窗口，填 NaN
    return np.concatenate([np.full(window - 1, np.nan), out])
```

`np.insert(x, 0, 0.0)` 在头部补一个 0，是为了让 `csum[window:] - csum[:-window]` 的下标对齐到「窗口和」。这一步没有任何显式循环，整段在 NumPy 的 C 实现里跑完。

> ⚠️ **浮点累积误差**：`cumsum` 在几十万根长序列上会累积浮点误差。对价格（量级 10²）通常无伤大雅，但如果你在收益率（量级 10⁻³）上做超长累积和，考虑用 `np.cumsum(x, dtype=np.float64)` 或分段求和。

## 第二步：布尔数组生成信号

有了两条均线，信号就是一次**逐元素比较**，得到一个布尔数组，再转成 0/1 持仓：

```python
def generate_signal(prices, fast, slow):
    ma_f = sma_vectorized(prices, fast)
    ma_s = sma_vectorized(prices, slow)
    # 布尔数组：快线在慢线之上 -> True
    long_mask = ma_f > ma_s
    # NaN 段（均线还没算出来）比较结果为 False，正好当空仓处理
    position = long_mask.astype(float)
    return position
```

这里有个隐藏的正确性红利：`ma_f > ma_s` 在任一操作数是 NaN 时结果为 `False`，所以 warmup 段（均线还没热身完）自动被当成空仓，不需要额外写边界判断。

## 第三步：shift 严守次日执行，不偷看未来

这是回测里**最容易翻车**的地方。信号在第 `i` 根 K 线收盘后才确认（你算均线要用到 `close[i]`），所以你最早只能在第 `i+1` 根 K 线执行。如果你用 `position[i] * ret[i]`，就是用今天收盘的信号吃今天的涨跌——**这是 look-ahead，回测会好看得离谱**。

向量化里，「次日执行」就是把持仓数组整体后移一格：

```python
def compute_equity(prices, position):
    ret = np.zeros_like(prices)
    ret[1:] = prices[1:] / prices[:-1] - 1        # 日收益率
    # 关键：position 后移一格 —— 今天的仓位由昨天的信号决定
    pos_shifted = np.empty_like(position)
    pos_shifted[0] = 0.0
    pos_shifted[1:] = position[:-1]
    strat_ret = pos_shifted * ret                  # 逐元素相乘，一次算完
    equity = np.cumprod(1 + strat_ret)             # 累积净值
    return equity, strat_ret
```

`pos_shifted[1:] = position[:-1]` 是整个引擎的信号执行时点。记住这条铁律：**信号在 bar i 确认，仓位在 bar i+1 生效**。少移一格，你的漂亮 Sharpe 就是幻觉。

## 第四步：三维广播，一次算完 200 组参数

到这里都还是单组参数。真正的威力在于**参数扫描**——把「所有快线」和「所有慢线」的组合摊成一个网格，用广播一次算完。

思路：先把每个候选窗口的均线预计算好，存成一个 `(候选窗口数, N)` 的矩阵；再用 NumPy 的广播规则，让「快线矩阵」和「慢线矩阵」在参数维度上对齐相乘，得到一个 `(快线数, 慢线数, N)` 的三维持仓张量。

```python
def scan_params(prices, fast_list, slow_list):
    n = len(prices)
    # 预计算所有候选均线，避免重复
    ma_fast_mat = np.stack([sma_vectorized(prices, f) for f in fast_list])  # (F, N)
    ma_slow_mat = np.stack([sma_vectorized(prices, s) for s in slow_list])  # (S, N)

    # 广播：(F,1,N) > (1,S,N) -> (F,S,N) 布尔持仓张量，一次成型
    pos = (ma_fast_mat[:, None, :] > ma_slow_mat[None, :, :]).astype(float)

    ret = np.zeros(n)
    ret[1:] = prices[1:] / prices[:-1] - 1

    # 次日执行：沿时间轴 shift
    pos_shift = np.zeros_like(pos)
    pos_shift[:, :, 1:] = pos[:, :, :-1]

    strat = pos_shift * ret[None, None, :]         # (F,S,N) 广播乘
    mean = strat.mean(axis=2)
    std = strat.std(axis=2)
    sharpe = np.where(std > 0, mean / std * np.sqrt(252), 0)  # (F,S) 每格一个 Sharpe

    # 屏蔽 fast >= slow 的非法组合
    mask = fast_list[:, None] >= slow_list[None, :]
    sharpe[mask] = np.nan
    return sharpe
```

关键就一行：`ma_fast_mat[:, None, :] > ma_slow_mat[None, :, :]`。`[:, None, :]` 把快线矩阵变成 `(F, 1, N)`，`[None, :, :]` 把慢线矩阵变成 `(1, S, N)`，NumPy 广播规则会自动把它们撑成 `(F, S, N)` 再逐元素比较。**200 多组参数的全部持仓，在这一行里同时生成，没有任何 Python 循环。**

![一次广播算出的 200+ 组参数 Sharpe 热图，绿色区域是快慢线搭配的甜区](/images/vectorized-backtest-engine/param_grid_heatmap.png)

上图就是这次扫描的产物：横轴慢线、纵轴快线，颜色是每组的年化 Sharpe。整张热图是**一次矩阵运算的结果**。你能一眼看到甜区（绿色）和雷区（红色），而循环版要跑两百多遍才能画出同样的图。

> ⚠️ **热图读法陷阱**：不要直接挑颜色最深那格当最优参数。孤立的高分格子通常是过拟合噪声，真正稳健的是一整片连成区域的高分区（对参数不敏感）。这是向量化扫描顺手带来的鲁棒性直觉——单点回测给不了。

## 加餐：2000 条蒙特卡洛路径压力测试

向量化的另一个杀手锏是**批量模拟**。想知道策略在随机市场里的净值分布？生成一个 `(路径数, 期数)` 的收益矩阵，一次 `cumprod` 出全部路径：

```python
def monte_carlo(mu, sigma, n_paths=2000, horizon=252, seed=42):
    rng = np.random.default_rng(seed)
    # 一次抽出 2000×252 个随机冲击
    shocks = rng.normal(mu, sigma, size=(n_paths, horizon))
    paths = 100 * np.cumprod(1 + shocks, axis=1)      # 沿时间轴累乘
    return paths

paths = monte_carlo(0.0004, 0.012)
q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)  # 每期的分位数
```

`np.cumprod(..., axis=1)` 沿时间轴一次算完 2000 条净值曲线。取分位数就能画出「扇形图」，直观看到策略的尾部风险。

![2000 条蒙特卡洛路径的扇形分布图，一次矩阵乘法生成](/images/vectorized-backtest-engine/monte_carlo_fan.png)

深色带是 25~75% 分位，浅色带是 5~95% 分位。这种图对判断「策略最差情况能烂到什么程度」极有价值，而生成它只需要一次 `cumprod`。

## 天下没有免费的午餐：速度换内存

到这里向量化看起来完美，但有个硬约束必须讲：**它把时间开销转移成了内存开销。**

那个 `(F, S, N)` 的持仓张量，元素个数是 `快线数 × 慢线数 × K线数`。假设你要扫 100×100 组参数、10 万根 K 线，那就是 `100 × 100 × 100000 = 10⁹` 个 float64——**80 GB**。你的机器会直接 OOM。

![向量化的代价：随参数组合增长，耗时线性上升的同时内存急剧膨胀](/images/vectorized-backtest-engine/memory_speed_tradeoff.png)

上图展示了这个取舍：蓝线是耗时（很平），红线是峰值内存（快速攀升）。**向量化把墙从"时间墙"换成了"内存墙"。**

实战里的解法是**分块（chunking）**：不要一次把所有参数组合摊平，而是切成能装进内存的批次，逐批向量化：

```python
def scan_params_chunked(prices, fast_list, slow_list, chunk=20):
    results = []
    for i in range(0, len(fast_list), chunk):
        sub = fast_list[i:i + chunk]
        results.append(scan_params(prices, sub, slow_list))  # 每批仍全向量化
    return np.vstack(results)
```

这样每批内部依然享受广播的速度，批次之间用一层轻量循环控制内存峰值。**外层循环 + 内层向量化**是工程上的黄金组合。

## A. 实现细节

- **信号判定字段**：均线全部基于 `close` 计算，信号在收盘后确认。
- **执行时点**：严格次日执行，通过持仓数组 `shift(1)` 实现——bar i 的信号，bar i+1 生效。
- **收益口径**：日简单收益率 `close[i]/close[i-1]-1`，净值用 `cumprod(1+strat_ret)` 复利累积。
- **warmup 处理**：均线未热身段为 NaN，比较运算自动归为空仓，不污染评估。
- **参数扫描口径**：`(F,S,N)` 三维广播，Sharpe 年化系数取 √252；`fast>=slow` 的非法组合置 NaN 屏蔽。

## B. 已知偏差

- **无摩擦成本**：本引擎未建模手续费、滑点、冲击成本。双均线换手不算高，但把它当实盘预期会高估。
- **满仓 0/1 假设**：持仓只有满仓/空仓两态，没建模仓位管理和资金约束，实际敞口会因取整、保证金而偏离。
- **蒙特卡洛的正态假设**：路径模拟用的是独立正态收益，忽略了真实市场的肥尾、波动聚集和序列相关——它给的是"温和世界"的下界，真实尾部更糟。

## C. 结果解读

- **提速来源**：300~500 倍的加速主要来自两处——消掉 Python 逐行解释开销，以及用 `cumsum` 把移动均线从 O(N×W) 降到 O(N)。数据量越大，优势越稳定。
- **参数热图的正确用法**：向量化让你能在秒级扫完整个参数空间，但要挑「成片的高分区域」而非「孤立的最高格」。前者对参数不敏感、更可能样本外存活；后者八成是过拟合。
- **向量化的适用边界**：策略逻辑一旦引入**路径依赖的状态机**（比如"持仓 N 天后才允许加仓""触发止损后冷却 5 天"），纯向量化会变得极其别扭甚至不可能——因为今天的动作依赖昨天动作的结果，无法一次性广播。这类策略要么用 `numba` 给循环加速，要么接受混合写法。**别为了向量化而扭曲策略逻辑。**
- **内存是新的瓶颈**：单标的参数扫描享受向量化红利；但多标的 × 多参数 × 长历史的三重乘积会让内存爆炸。工程上的答案永远是"外层分块循环 + 内层向量化"，而不是追求一行代码算完一切。
