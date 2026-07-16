---
title: "ADX 趋势强度：用定向运动指数把『有趋势』和『无趋势』分开"
description: "均线金叉在震荡市里会被反复打脸，问题不是方向判断错，而是根本不该在无趋势时下注。ADX（Wilder 1978）只回答一个问题——现在有没有趋势——把 +DI/-DI 管方向、ADX 管强度分工。本文从零复现 Wilder 平滑，用自洽合成数据证明 ADX>25 过滤能把裸 DI 交叉的 Sharpe 从 -0.01 拉到 0.40，并诚实指出它在强趋势 beta 行情里仍会跑输买入持有，附完整 Python 与六类真实陷阱（高阶）。"
publishDate: '2026-07-16'
tags:
  - 量化交易
  - ADX
  - 定向运动指数
  - 趋势跟随
  - 技术指标
  - Wilder
  - 趋势过滤
  - Python
language: Chinese
difficulty: advanced
---

做趋势跟随的人早晚会撞上同一堵墙：均线金叉、MACD 上穿，方向判断本身没错，可一旦市场进入横盘震荡，这些信号就开始反复打脸——上穿买、下穿卖、再上穿再买，来回被割。问题的根源不是"方向判断错了"，而是**根本不该在没有趋势的时候下注**。

结论先放这：**ADX（Average Directional Index，Wilder 1978）不判断涨跌，只回答一个问题——现在到底有没有趋势。** 它把工作拆成两半：`+DI` 和 `-DI` 管方向（谁强谁主导），`ADX` 管强度（这个方向够不够结实）。在自洽合成数据里，用 `ADX>25` 作为开关去过滤裸 DI 交叉信号，能把策略 Sharpe 从 −0.01 拉到 0.40、年化从 −0.1% 拉到 +2.1%；但在同一段包含大趋势的行情里，它仍然跑不赢单纯买入持有（Sharpe 0.49）——这正是趋势过滤器的本质边界。附完整 Python 与六类真实陷阱（高阶）。

![ADX 定向运动系统：上图价格、下图 +DI/-DI/ADX 三线。ADX 抬头说明趋势在增强，与方向无关](/images/adx-trend-strength/adx_components.png)

## 一、为什么需要一个"只测强度、不测方向"的指标

绝大多数技术指标都在回答"往哪走"：均线看斜率、MACD 看动量方向、RSI 看超买超卖。但它们都默认了一个前提——**市场此刻正处在一个可交易的趋势里**。这个前提在牛市熊市成立，在震荡市里彻底失效。

Wilder 的洞察是：把"方向"和"强度"彻底解耦。一个市场可以是"强烈上涨"（+DI 高、ADX 高）、"强烈下跌"（−DI 高、ADX 高），也可以是"没方向但在乱窜"（DI 纠缠、ADX 低）。ADX 这条线**永远是正的、永远不告诉你涨跌**，它只告诉你：现在这股劲儿，够不够你压上趋势策略。

这就是 ADX 的核心用法——**不是入场信号，而是入场许可证**。方向由别的指标（或 DI 交叉）给，ADX 决定"这个方向值不值得信"。

## 二、从 +DM/-DM 到 ADX：三级递推

ADX 的计算是一条清晰的加工链，每一步都建立在前一步之上：

**第一步：方向运动（Directional Movement）。** 比较今天和昨天的高低点，谁走得更远：

```
上升动量 up   = High_today - High_yesterday
下降动量 down = Low_yesterday - Low_today

+DM = up   （当 up > down 且 up > 0，否则 0）
-DM = down （当 down > up 且 down > 0，否则 0）
```

注意这里的"排他"逻辑：同一天里 +DM 和 −DM 至多有一个非零。今天如果创了新高但没破新低，就只记 +DM；反之只记 −DM；内包线（高更低、低更高）两者都是 0。

**第二步：用真实波幅归一化，得到方向指标 DI。** 原始的 DM 是绝对点数，不同价位不可比，所以要除以真实波幅 ATR：

```
+DI = 100 × Wilder平滑(+DM) / ATR
-DI = 100 × Wilder平滑(-DM) / ATR
```

**第三步：从 DI 差异到 DX，再平滑成 ADX。** 两条 DI 差得越远，说明方向越"一边倒"：

```
DX  = 100 × |+DI - -DI| / (+DI + -DI)
ADX = Wilder平滑(DX)
```

关键在那个 **Wilder 平滑（也叫 RMA）**——它不是普通移动平均，而是一种带记忆的指数式递推，这也是很多人自己实现 ADX 时算错的地方：

```python
import numpy as np

def wilder_smooth(x, period):
    """Wilder 平滑（RMA）：第一个值用简单均值播种，其后指数递推。"""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if len(x) < period:
        return out
    out[period - 1] = np.nanmean(x[:period])          # 播种
    for i in range(period, len(x)):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period   # 递推
    return out
```

完整的 ADX 计算：

```python
def compute_adx(high, low, close, period=14):
    n = len(close)
    tr = np.zeros(n); plus_dm = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i]  = up   if (up > down and up > 0)   else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    atr      = wilder_smooth(tr[1:], period)
    sm_plus  = wilder_smooth(plus_dm[1:], period)
    sm_minus = wilder_smooth(minus_dm[1:], period)
    # 对齐、算 DI/DX/ADX（略，见下方仓库版本）
    plus_di  = 100 * sm_plus  / atr
    minus_di = 100 * sm_minus / atr
    dx  = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)
    return plus_di, minus_di, adx
```

## 三、ADX 阈值：把行情切成两半

约定俗成的读数（Wilder 原版）：

- **ADX < 20**：无趋势，震荡市，趋势策略应该关机。
- **ADX 20~25**：模糊地带，趋势在酝酿或衰竭。
- **ADX > 25**：趋势确立，值得跟随。
- **ADX > 40~50**：强趋势，但也要警惕见顶（ADX 掉头往往是趋势衰竭的早期信号）。

把 `ADX>25` 的时段在价格图上涂成阴影，能非常直观地看到它切出了什么：

![ADX>25 标出的『趋势段』（蓝色阴影）与震荡段（留白）。趋势策略只应在阴影区里工作](/images/adx-trend-strength/adx_regime_split.png)

注意一个反直觉的点：**ADX 高不代表价格高，甚至不代表价格在涨。** 一段猛烈的下跌同样会把 ADX 顶上去。ADX 只是"劲儿大不大"，方向永远要回头看 +DI 和 −DI 谁在上面。

## 四、回测：ADX 过滤到底救了什么

我们做一个最朴素的对照实验。基础信号是 **DI 交叉**：`+DI > -DI` 做多，`-DI > +DI` 空仓（A 股语境只做多）。然后对比两个版本——裸交叉，以及加上 `ADX>25` 才允许持仓的过滤版：

```python
def backtest(use_adx_filter, adx_th=25):
    pos = np.zeros(n); cur = 0
    for i in range(1, n):
        strong = adx[i] > adx_th if use_adx_filter else True
        # 信号在 i 判定，i+1 执行 —— 用前一日 DI 避免 look-ahead
        if plus_di[i-1] > minus_di[i-1] and strong:
            cur = 1
        elif minus_di[i-1] > plus_di[i-1] and strong:
            cur = 0
        elif use_adx_filter and adx[i] <= adx_th:
            cur = 0          # 趋势消失就离场
        pos[i] = cur
    strat_ret = pos * ret
    return np.cumprod(1 + strat_ret), pos, strat_ret
```

合成 900 天、趋势段与震荡段交替出现的数据上，结果是：

![ADX 过滤把『无趋势期的反复被割』挡在门外：过滤版净值更平稳](/images/adx-trend-strength/adx_filter_equity.png)

| 策略 | 年化收益 | Sharpe |
|---|---|---|
| 裸 DI 交叉 | −0.1% | −0.01 |
| ADX>25 过滤 | +2.1% | 0.40 |
| 买入持有 | +6.8% | 0.49 |

三个数字讲了一个诚实的故事：

1. **裸 DI 交叉基本是白干**（Sharpe −0.01）——震荡市里的反复交叉把趋势段赚的钱全吐回去了。
2. **ADX 过滤确实有效**：它没有提高方向判断的准确率，而是**把那些"方向判断在无趋势期毫无意义"的交易直接删掉了**，Sharpe 从 −0.01 抬到 0.40。
3. **但它跑不赢买入持有**——这不是 bug，是这段合成数据里恰好含有大段单边趋势时，"择时"天然干不过"一直在场"。趋势过滤器的价值在**控制回撤和减少无效交易**，不是无脑增厚收益。

## 五、阈值敏感性：25 不是圣数

很多人把 ADX=25 当成金科玉律，其实它只是 Wilder 的经验值。扫一遍 10~40 的阈值：

![ADX 阈值敏感性：年化收益与 Sharpe 随阈值变化，25 附近并非唯一甜点](/images/adx-trend-strength/adx_threshold_scan.png)

在这份数据里，Sharpe 随阈值抬高整体走高（更严格的过滤 = 更少但更干净的交易），最优点落在 36 附近。但**别把这个数字外推到实盘**——它是对这段特定合成数据的过拟合。真正稳健的做法是：在你的标的、你的周期上做样本外验证，看阈值在一个宽区间（比如 20~30）里是否都不太差，而不是去抠那个最高点。

## 六、六类真实陷阱

1. **Wilder 平滑 ≠ SMA/EMA。** 最常见的错误是用 `rolling(14).mean()` 代替 Wilder 平滑，算出来的 ADX 会系统性偏离。Wilder 的 RMA 等价于 α=1/period 的 EMA，记忆更长、更平滑。

2. **ADX 有双重滞后。** DX 已经是平滑后 DI 的差，ADX 又对 DX 再平滑一次——两层 Wilder 平滑意味着 ADX 对趋势变化的反应很慢。等 ADX 涨过 25，趋势往往已经走了一截。它是"确认器"不是"预警器"。

3. **ADX 不给方向，单用必翻车。** ADX=45 可能是暴涨也可能是暴跌。任何"ADX 高就买"的逻辑都是错的，必须配 +DI/−DI 或其他方向指标。

4. **ATR 归一化在低波动标的上会放大噪声。** 当 ATR 很小时，DI = 100×DM/ATR 的分母接近零，DI 会剧烈跳动，DX 随之失真。低波动、低成交的标的上 ADX 可信度下降。

5. **look-ahead 泄漏。** 信号在第 i 天用当天收盘后才能算出的 DI/ADX 判定，就必须在第 i+1 天执行（本文用 `plus_di[i-1]` 严格错开）。如果拿当天 ADX 去决定当天开盘就买，就偷看了未来。

6. **趋势衰竭的假信号。** ADX 见顶回落时，趋势往往还在延续但正在减速。有人把"ADX 掉头"当离场信号，结果频繁在趋势中段被震出。ADX 回落只说明"劲在减弱"，不等于"要反转"。

## 结语

ADX 不是让你赚更多，而是让你**在不该出手的时候把手揣兜里**。它把趋势跟随策略里最烧钱的那部分——震荡市里的反复被割——用一个"趋势强度开关"过滤掉。理解它的边界比记住 25 这个数字更重要：它是趋势策略的守门员，不是印钞机。在真正的大趋势里，最好的策略永远是"一直在场"；ADX 的价值，是帮你在没有大趋势的漫长中间地带活下来。
