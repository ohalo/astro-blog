---
title: "向量化回测的四个陷阱：一次可复现的双均线审计"
description: "把回测改写成 NumPy 广播，三万根 K 线、20 日窗口的基础运算量从 63 万次降到 6 万次，快了 10.5 倍。但速度不是重点——正因为扫描快到几乎免费，过拟合更容易发生。用固定种子的 1800 天市场把前视偏差、参数选择、交易成本、样本切分四个陷阱逐个量化，并给出 VectorBT 的正确 API 对照。"
publishDate: '2026-11-24T21:40:00Z'
tags:
  - 量化
  - 量化交易
  - 向量化
  - NumPy
  - VectorBT
  - 回测陷阱
  - Python
language: zh
difficulty: advanced
---

结论先行：把双均线回测从 Python 循环改写成 NumPy 广播，同样三万根 K 线、20 日窗口，基础运算量从 63 万次降到 6 万次，快 10.5 倍，墙钟差距更大（图 1）。但速度不是本文的重点——正因为参数扫描快到几乎免费，**过拟合比循环时代更容易发生**。本文用固定种子的 1800 天合成市场，把四个最常被忽略的陷阱逐个量化，所有数字出自同一段可复跑代码。

![循环版与向量化移动均线的实测耗时对比，双对数坐标](/images/quant-backtesting-vectorbt-guide/fig1_speed.png)

## 先搭引擎：三块积木

向量化回测的核心不是「用 NumPy 重写一遍」，而是把策略拆成三块可以独立广播的张量。

**第一块是移动均线。** 逐行切片求均值是 O(N×W)，用累积和相减把它压成 O(N)：

```text
def sma(x, window):
    csum = np.concatenate([[0.0], np.cumsum(x)])
    out = (csum[window:] - csum[:-window]) / window
    return np.concatenate([np.full(window - 1, np.nan), out])
```

头部补零是为了让 `csum[window:] - csum[:-window]` 的下标正好对齐窗口和。前 `window-1` 根没有完整窗口，填 NaN——后面 `NaN > NaN` 结果为 `False`，warmup 段自动变成空仓，不需要额外的边界判断。

**第二块是布尔信号。** 快线在慢线之上就是满仓，一次逐元素比较生成整个持仓数组。

**第三块是执行时点。** 这是全引擎最容易被写错的一行。信号在第 i 根收盘后确认（算均线用到了 `close[i]`），最早只能在第 i+1 根成交。所以持仓必须整体后移一格：`shifted[1:] = pos[:-1]`。少移一格，回测就是在偷看未来；多移一格，则白白丢掉一天收益。

有了这三块，参数扫描只要一行广播：`(F,1,N) > (1,S,N)` 得到 `(F,S,N)` 的持仓张量，再与收益张量相乘，一次算出全部参数组合的 Sharpe。本文扫 6 个快线 × 7 个慢线 = 42 组（剔除非法组合后 36 组），整张热图是一次矩阵运算的产物。

## VectorBT 对照：把常见错误写在旁边

自己手写引擎是为了看清每一步，实战里 VectorBT 更省事。但它的 API 有几处非常容易写错，这里对照列出，本文的数值结论用纯 NumPy 得到，不依赖任何第三方回测库：

| 意图 | 正确写法 | 常见错误 |
|---|---|---|
| 均线交叉 | `fast.ma_crossed_above(slow)` | `fast.ma > slow` 是「快线在区间内」的状态，不是交叉事件 |
| 多参数组合 | `vbt.MA.run_combs(price, window=windows, r=2, short_names=['fast','slow'])` | 分别 `run` 再 `rename` 后比较：两组列名没有交集，结果是全 NaN |
| 业绩指标 | `pf.total_return()`、`pf.sharpe_ratio()`、`pf.trades.expectancy()` | 当属性写 `f"{pf.total_return:.2%}"`，直接 `TypeError` |
| 盈亏比 | `pf.trades.profit_factor()` | `expectancy()` 是平均每笔盈亏金额，不是盈亏比 |
| 滚动切分 | `price.vbt.rolling_split(window_len=126, set_lens=(1,1))` | `set_lens=(84,42)` 填满 126，测试集长度为零，且直接报 `ValueError` |

最后一行尤其致命：把训练段和验证段填满了整个窗口，还声称做了样本外验证，等于自己给自己发了一张合格证。

## 陷阱一：前视偏差，一行 shift 值 24 个百分点

把执行时点从「次日」改成「同日」，其余完全不变（同样的 10/50 均线、同样的 5bps 成本），全样本累计收益从 **-0.17%** 变成 **+23.45%**，Sharpe 从 **0.067** 变成 **0.282**。策略代码一个字没改，只挪了一格数组。

![次日执行与同日执行的净值曲线对比，测试段之后差距持续扩大](/images/quant-backtesting-vectorbt-guide/fig2_lookahead.png)

图 2 里两条线在训练段还贴在一起，越过测试段起点后彻底分道扬镳。这就是为什么「回测收益很高」本身不构成任何证据——你得先问一句：信号是在收盘前还是收盘后算出来的？

## 陷阱二：参数冠军是过拟合的平均值

在训练段（前 900 天）扫完 36 组参数，Sharpe 最高的是快线 10、慢线 60，训练 Sharpe **0.829**。它的八邻域平均只有 **0.484**，已经提示这个格子是孤立尖峰而非成片高原。但更直接的证据在测试段：这个冠军的 Sharpe 是 **-0.347**。

![训练段参数热图，星号标出样本内冠军](/images/quant-backtesting-vectorbt-guide/fig3_param_grid.png)

训练段前五名的 Sharpe 依次为 0.829、0.757、0.734、0.719、0.710，把它们原封不动搬到测试段，对应 Sharpe 是 -0.347、-0.506、-0.895、-0.331、-0.853——**五个全部为负**。参数排名在样本内越靠前，样本外并没有更好；所谓「最优参数」只是对训练段噪声的拟合。

固定用 10/50 参数也一样：训练段累计 +14.8%，验证段 -4.5%，测试段 -19.5%。三段属于同一条合成价格序列，只是时间先后不同，结论就翻转了。

## 陷阱三：成本不是细节，是方向

对样本内冠军做成本扫描（单边 0 到 40bps），测试段 Sharpe 从 **-0.330** 一路降到 **-0.467**，累计收益从 **-13.1%** 降到 **-17.1%**。平均换手率约 2%，在 5bps 下每期成本吃掉 1 个基点，看似不起眼，复利 600 天后就是两个百分点的差距。

![单边成本从 0 到 40bps 时测试段 Sharpe 与累计收益的变化](/images/quant-backtesting-vectorbt-guide/fig4_cost.png)

这里要分清两件事：成本不会把一个负 Sharpe 的策略变成正 Sharpe，但它会把一个「勉强能看」的策略变成确定亏损。任何声称高换手策略有效的回测，如果没填 `fees` 和 `slippage`，结论都不成立。

## 陷阱四：滚动样本外，六次全负

把 1800 天切成 6 个滚动窗口，每个窗口用前 300 天选参数、后 120 天验证。六个窗口的训练 Sharpe 分别是 1.58、2.19、2.06、2.06、1.88、0.66——看着都很漂亮；对应测试 Sharpe 是 -0.14、-1.39、-0.06、-1.62、-1.97、-1.30，**六次全部为负**。

这不是某一组参数选坏了，而是这类策略在这条价格序列上没有可兑现的 edge。把所有窗口的测试段收益收集起来做 2000 条 block bootstrap 路径，期末净值的中位数是 **0.938**，5% 分位 **0.742**，95% 分位 **1.190**——中位数以下，尾部薄得可怜。

![2000 条 block bootstrap 路径的扇形分布](/images/quant-backtesting-vectorbt-guide/fig5_bootstrap_fan.png)

> ⚠️ 这里的 bootstrap 重采样的是日收益，保留了收益序列的经验分布，但不保留波动聚集和长程依赖。它给的是「如果每天独立抽一次」的温和世界，真实尾部通常更厚。

## 天下没有免费的午餐：内存墙

`(F,S,N)` 张量的元素数是 `快线数 × 慢线数 × K线数`。本文 6×7×600 只有两万多个数，毫无压力；换成 100×100 组参数、10 万根 K 线，就是 10⁹ 个 float64——**8 GB**，再多一个维度就 OOM。

工程上的解法是外层分块、内层广播：把快线切成每批 20 个，逐批做全向量化，再纵向拼接结果。批次之间那层轻量 Python 循环换来的是可控的峰值内存，而每批内部依然享受广播的速度。别为了「一行算完」把机器搞崩。

还有一类策略根本不适合纯向量化：今天是否加仓取决于昨天触发止损后的冷却计时——状态机天然是路径依赖的，广播表达不了。这种时候老老实实用 `numba` 给循环加速，别为了向量化扭曲策略逻辑。

```python
"""向量化回测引擎 + 四类回测陷阱审计：文章单一代码源。

纯 NumPy 实现，固定种子，exec 输出与 results.json 深度相等。
生成 5 张图：速度对比、前视偏差、参数热图、成本扫描、蒙特卡洛扇形。
"""
import json
import os
import time

import numpy as np

SEED = 20261124
N_TOTAL = 1800
N_TRAIN = 900
N_VALID = 300
COST_BPS = 5.0
FREQ = 252
FAST_LIST = np.array([5, 10, 15, 20, 25, 30])
SLOW_LIST = np.array([20, 30, 40, 50, 60, 70, 80])
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "public", "images", "quant-backtesting-vectorbt-guide")


# ------------------------------------------------------------------ 数据与引擎
def make_price(seed=SEED, n=N_TOTAL):
    """带两段波动状态的合成日线：温暖期 +  turbulent 期。"""
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(n) < n // 2, 0.009, 0.015)
    drift = np.where(np.arange(n) < n // 2, 0.0005, -0.0002)
    shock = rng.normal(drift, 1.0, size=n) * vol
    return 100.0 * np.exp(np.cumsum(shock))


def sma(x, window):
    """累积和 O(N) 移动均线，warmup 段为 NaN。"""
    csum = np.concatenate([[0.0], np.cumsum(x)])
    out = (csum[window:] - csum[:-window]) / window
    return np.concatenate([np.full(window - 1, np.nan), out])


def sma_matrix(x, windows):
    return np.stack([sma(x, int(w)) for w in windows])


def make_position(price, fast, slow, lookahead=False):
    """双均线持仓：信号在 bar i 收盘确认，默认 bar i+1 生效。"""
    pos = (sma(price, fast) > sma(price, slow)).astype(np.float64)
    if lookahead:
        return pos
    shifted = np.zeros_like(pos)
    shifted[1:] = pos[:-1]
    return shifted


def returns(price):
    r = np.zeros_like(price)
    r[1:] = price[1:] / price[:-1] - 1.0
    return r


def apply_cost(strategy_ret, position, cost_bps):
    """按换手扣单边成本：换手率 = |今日仓位 - 昨日仓位|。"""
    turnover = np.abs(np.diff(position, prepend=0.0))
    return strategy_ret - turnover * (cost_bps / 1e4), turnover


def stats(strategy_ret, position):
    r = np.asarray(strategy_ret, dtype=np.float64)
    mean = float(np.mean(r))
    std = float(np.std(r))
    sharpe = mean / std * np.sqrt(FREQ) if std > 0 else 0.0
    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0))
    return {
        "sharpe": round(sharpe, 6),
        "return": round(total_return, 6),
        "max_dd": round(max_dd, 6),
        "turnover": round(float(np.mean(np.abs(np.diff(position, prepend=0.0)))), 6),
        "exposure": round(float(np.mean(position)), 6),
        "equity": equity,
    }


def backtest(price, fast, slow, start=0, end=None, cost_bps=COST_BPS, lookahead=False):
    end = len(price) if end is None else end
    seg = price[start:end]
    pos = make_position(seg, fast, slow, lookahead=lookahead)
    strat = pos * returns(seg)
    net, turn = apply_cost(strat, pos, cost_bps)
    s = stats(net, pos)
    s["turnover"] = round(float(np.mean(turn)), 6)
    return s


def scan_grid(price, start, end, cost_bps=COST_BPS):
    """(F,S,N) 广播一次算完全部参数组合的净收益 Sharpe。"""
    seg = price[start:end]
    n = len(seg)
    mf = sma_matrix(seg, FAST_LIST)   # (F,N)
    ms = sma_matrix(seg, SLOW_LIST)   # (S,N)
    pos = (mf[:, None, :] > ms[None, :, :]).astype(np.float64)  # (F,S,N)
    shifted = np.zeros_like(pos)
    shifted[:, :, 1:] = pos[:, :, :-1]
    r = returns(seg)
    net, turn = apply_cost(shifted * r[None, None, :], shifted, cost_bps)
    mean = net.mean(axis=2)
    std = net.std(axis=2)
    sharpe = np.zeros_like(mean)
    np.divide(mean, std, out=sharpe, where=std > 0)
    sharpe *= np.sqrt(FREQ)
    illegal = FAST_LIST[:, None] >= SLOW_LIST[None, :]
    sharpe[illegal] = np.nan
    turnover = turn.mean(axis=2)
    return sharpe, turnover


# ------------------------------------------------------------------ 图
def fig_speed(price):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [1000, 3000, 10000, 30000]
    loop_t, vec_t = [], []
    base = price[:500]
    for n in sizes:
        x = np.tile(base, int(np.ceil(n / len(base))))[:n]
        t0 = time.perf_counter()
        ma_f = np.full(n, np.nan)
        for i in range(n):
            if i >= 19:
                ma_f[i] = x[i - 19:i + 1].mean()
        loop_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _ = sma(x, 20)
        vec_t.append(time.perf_counter() - t0)
    # 墙钟时间只用于画图；JSON 只存确定性操作量，保证 exec 复跑一致
    window = 20
    loop_ops = [n * window + n for n in sizes]
    vec_ops = [2 * n + window for n in sizes]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.loglog(sizes, loop_t, "o-", color="#b23a2b", lw=2, label="Python loop (slice mean)")
    ax.loglog(sizes, vec_t, "s-", color="#1f4e79", lw=2, label="NumPy cumsum")
    ax.set_xlabel("number of bars")
    ax.set_ylabel("seconds (log scale)")
    ax.set_title("Loop vs vectorized moving average")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_speed.png"), dpi=140)
    plt.close(fig)
    return loop_t, vec_t, loop_ops, vec_ops


def fig_lookahead(price):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    honest = backtest(price, 10, 50, 0, N_TOTAL, COST_BPS)
    cheat = backtest(price, 10, 50, 0, N_TOTAL, COST_BPS, lookahead=True)
    t = np.arange(N_TOTAL)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot(t, honest["equity"], color="#1f4e79", lw=1.7,
            label="next-bar execution (honest)")
    ax.plot(t, cheat["equity"], color="#b23a2b", lw=1.7,
            label="same-bar execution (look-ahead)")
    ax.axvline(N_TRAIN, color="#888", ls=":", lw=1.0)
    ax.axvline(N_TRAIN + N_VALID, color="#555", ls="--", lw=1.1)
    ax.text(N_TRAIN + N_VALID + 12, ax.get_ylim()[1] * 0.92, "test starts",
            fontsize=9, color="#444")
    ax.set_xlabel("trading day")
    ax.set_ylabel("equity (start = 1)")
    ax.set_title("One-bar shift changes the whole story")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_lookahead.png"), dpi=140)
    plt.close(fig)
    return honest, cheat


def fig_grid(sharpe, best_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    data = np.ma.masked_invalid(sharpe)
    im = ax.imshow(data, origin="lower", aspect="auto", cmap="RdYlGn",
                   extent=[SLOW_LIST[0] - 5, SLOW_LIST[-1] + 5,
                           FAST_LIST[0] - 2.5, FAST_LIST[-1] + 2.5])
    ax.scatter([SLOW_LIST[best_idx[1]]], [FAST_LIST[best_idx[0]]],
               marker="*", s=230, color="black", edgecolor="white", lw=1.1,
               label="in-sample champion")
    ax.set_xlabel("slow window")
    ax.set_ylabel("fast window")
    ax.set_title("In-sample Sharpe grid (5bps cost, next-bar execution)")
    fig.colorbar(im, ax=ax, label="annualised Sharpe")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_param_grid.png"), dpi=140)
    plt.close(fig)


def fig_cost(price, best_f, best_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    costs = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    rows = {}
    for c in costs:
        rows[str(int(c))] = backtest(price, best_f, best_s,
                                     N_TRAIN + N_VALID, N_TOTAL, cost_bps=c)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(costs, [rows[str(int(c))]["sharpe"] for c in costs], "o-",
            color="#1f4e79", lw=2, label="Sharpe (left)")
    ax2 = ax.twinx()
    ax2.plot(costs, [rows[str(int(c))]["return"] for c in costs], "s--",
             color="#b23a2b", lw=1.8, label="test return (right)")
    ax.set_xlabel("single-side cost (bps)")
    ax.set_ylabel("annualised Sharpe", color="#1f4e79")
    ax2.set_ylabel("cumulative test return", color="#b23a2b")
    ax.set_title("Cost sensitivity of the in-sample champion")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_cost.png"), dpi=140)
    plt.close(fig)
    return rows


def fig_mc(strategy_ret, seed=SEED):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n_paths, horizon = 2000, FREQ
    idx = rng.integers(0, len(strategy_ret), size=(n_paths, horizon))
    draws = strategy_ret[idx]
    paths = np.cumprod(1.0 + draws, axis=1)
    perc = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    t = np.arange(1, horizon + 1)
    ax.fill_between(t, perc[0], perc[4], color="#9ecae1", alpha=0.45,
                    label="5–95% (block bootstrap)")
    ax.fill_between(t, perc[1], perc[3], color="#3182bd", alpha=0.45,
                    label="25–75%")
    ax.plot(t, perc[2], color="#08306b", lw=1.8, label="median")
    ax.axhline(1.0, color="#777", ls="--", lw=1)
    ax.set_xlabel("trading day")
    ax.set_ylabel("equity")
    ax.set_title("2000 bootstrap paths of the daily strategy return")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_bootstrap_fan.png"), dpi=140)
    plt.close(fig)
    return [round(float(x), 6) for x in perc[:, -1]]


# ------------------------------------------------------------------ 主流程
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    price = make_price()

    honest = backtest(price, 10, 50, 0, N_TRAIN)
    cheat = backtest(price, 10, 50, 0, N_TRAIN, lookahead=True)
    honest_test = backtest(price, 10, 50, N_TRAIN + N_VALID, N_TOTAL)
    honest_valid = backtest(price, 10, 50, N_TRAIN, N_TRAIN + N_VALID)

    train_sharpe, _ = scan_grid(price, 0, N_TRAIN)
    flat = np.argwhere(~np.isnan(train_sharpe))
    best_flat = flat[np.nanargmax(train_sharpe[~np.isnan(train_sharpe)])]
    bi, sj = int(best_flat[0]), int(best_flat[1])
    best_f, best_s = int(FAST_LIST[bi]), int(SLOW_LIST[sj])
    best_train_sharpe = float(train_sharpe[bi, sj])
    # 高原判定：八邻域（去掉自身）的均值
    neigh = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = bi + di, sj + dj
            if (di, dj) == (0, 0) or not (0 <= ii < len(FAST_LIST) and 0 <= jj < len(SLOW_LIST)):
                continue
            if not np.isnan(train_sharpe[ii, jj]):
                neigh.append(float(train_sharpe[ii, jj]))
    plateau_mean = float(np.mean(neigh)) if neigh else 0.0

    test_sharpe, _ = scan_grid(price, N_TRAIN + N_VALID, N_TOTAL)
    champ_test_sharpe = float(test_sharpe[bi, sj]) if not np.isnan(test_sharpe[bi, sj]) else 0.0
    ranked = np.argsort(-np.nan_to_num(train_sharpe, nan=-1e9), axis=None)
    top5_train = []
    top5_test = []
    for k in ranked[:5]:
        i, j = divmod(int(k), len(SLOW_LIST))
        top5_train.append(round(float(train_sharpe[i, j]), 6))
        tv = test_sharpe[i, j]
        top5_test.append(round(float(tv), 6) if not np.isnan(tv) else None)

    # walk-forward：6 段，train 300 / test 120，滚动选参数
    wf = []
    for w in range(6):
        tr_end = N_TRAIN + w * 60
        te_end = tr_end + 120
        if te_end > N_TOTAL:
            break
        sh, _ = scan_grid(price, tr_end - 300, tr_end)
        valid = np.argwhere(~np.isnan(sh))
        b = valid[np.nanargmax(sh[~np.isnan(sh)])]
        wi, wj = int(b[0]), int(b[1])
        s_tr = backtest(price, int(FAST_LIST[wi]), int(SLOW_LIST[wj]),
                        tr_end - 300, tr_end)["sharpe"]
        s_te = backtest(price, int(FAST_LIST[wi]), int(SLOW_LIST[wj]),
                        tr_end, te_end)["sharpe"]
        wf.append({"window": w + 1, "fast": int(FAST_LIST[wi]),
                   "slow": int(SLOW_LIST[wj]), "train_sharpe": s_tr,
                   "test_sharpe": s_te})

    loop_t, vec_t, loop_ops, vec_ops = fig_speed(price)
    lo, ch = fig_lookahead(price)
    fig_grid(train_sharpe, (bi, sj))
    cost_rows = fig_cost(price, best_f, best_s)
    pos_best = make_position(price[N_TRAIN + N_VALID:N_TOTAL], best_f, best_s)
    r_best = pos_best * returns(price[N_TRAIN + N_VALID:N_TOTAL])
    net_best, _ = apply_cost(r_best, pos_best, COST_BPS)
    mc_final = fig_mc(net_best)

    result = {
        "setup": {
            "seed": SEED, "n_total": N_TOTAL, "n_train": N_TRAIN,
            "n_valid": N_VALID, "n_test": N_TOTAL - N_TRAIN - N_VALID,
            "cost_bps": COST_BPS, "freq": FREQ,
            "fast_list": FAST_LIST.tolist(), "slow_list": SLOW_LIST.tolist(),
        },
        "speed": {"bars": [1000, 3000, 10000, 30000],
                  "loop_elementary_ops": loop_ops,
                  "vector_elementary_ops": vec_ops,
                  "work_ratio": round(float(loop_ops[-1] / vec_ops[-1]), 6)},
        "lookahead": {"honest": {k: lo[k] for k in ("sharpe", "return", "max_dd", "turnover")},
                      "cheat": {k: ch[k] for k in ("sharpe", "return", "max_dd", "turnover")}},
        "fixed": {"train": {k: honest[k] for k in ("sharpe", "return", "max_dd")},
                  "valid": {k: honest_valid[k] for k in ("sharpe", "return", "max_dd")},
                  "test": {k: honest_test[k] for k in ("sharpe", "return", "max_dd")}},
        "grid": {
            "champion": {"fast": best_f, "slow": best_s,
                         "train_sharpe": round(best_train_sharpe, 6),
                         "test_sharpe": round(champ_test_sharpe, 6),
                         "neighbour_mean": round(plateau_mean, 6)},
            "top5_train_sharpe": top5_train,
            "top5_matched_test_sharpe": top5_test,
        },
        "cost": {c: {k: cost_rows[c][k] for k in ("sharpe", "return", "turnover")}
                 for c in cost_rows},
        "walk_forward": wf,
        "bootstrap_final_equity": {
            "p5": mc_final[0], "p25": mc_final[1], "p50": mc_final[2],
            "p75": mc_final[3], "p95": mc_final[4],
        },
    }
    out = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
```


## A. 实现细节

- **信号口径**：均线全部基于收盘价，信号在 bar i 收盘确认，bar i+1 生效，通过 `shifted[1:] = pos[:-1]` 实现。
- **收益口径**：日简单收益率，净值 `cumprod(1 + net_ret)` 复利累积；单边成本按 `|当日仓位 − 昨日仓位|` 扣除。
- **参数扫描**：`(F,S,N)` 三维广播，Sharpe 年化系数 √252，`fast >= slow` 的非法组合置 NaN 屏蔽。
- **确定性**：全部随机性来自固定种子 20261124；速度图用墙钟计时，但写入结果文件的只有确定性操作量，保证任何机器复跑逐项一致。

## B. 已知偏差

- **合成市场**：价格由两段不同波动状态的正态冲击生成，没有涨跌停、跳空和真实流动性，结论只适用于说明方法学，不代表任何真实标的。
- **满仓 0/1**：未建模部分仓位、资金约束与取整，实盘敞口会偏离。
- **不含冲击成本**：手续费和滑点按固定 bps 建模，大额资金的冲击成本是非线性的，本文没有覆盖。

## C. 结果解读

- **速度红利是真的**：三万根 K 线、20 日窗口，基础运算量从 63 万降到 6 万，快 10.5 倍；墙钟差距更大，因为 Python 解释器的逐行开销被整块消掉了。
- **速度红利也是陷阱**：快到能秒扫全部参数，意味着「挑最高格」这个坏习惯的成本变得极低。样本内前五名样本外全负，就是对这种习惯的直接定价。
- **判断稳健性的正确姿势**：看参数邻域是不是连成高原，看滚动窗口是否多数为正，看成本翻倍后结论是否翻转。单一回测数字无论多漂亮，都不构成策略有效的证据。
- **本文的结论是负的**：在这条合成市场上，双均线策略没有样本外 edge。诚实地报告这一点，比拿训练段的 0.829 Sharpe 去讲故事有价值得多。
