---
title: "Bootstrap 回测显著性：用重抽样给你的夏普率算出 p 值"
description: "回测跑出 1.5 的夏普很爽，但它真的不是运气吗？经典 t 检验在收益非正态、自相关、样本短时会严重失真。本文用 iid bootstrap 重抽样,在「零假设=无 alpha」下重建夏普的分布,直接算出 p 值:真 alpha 策略 p=0.015 当场现形、纯噪声策略 p=0.47 无所遁形。并附功效分析与四类真实陷阱(中阶)。"
publishDate: '2026-07-18'
tags:
  - 量化交易
  - 回测
  - 夏普比率
  - Bootstrap
  - 显著性检验
  - p 值
  - 过拟合
  - Python
language: Chinese
difficulty: intermediate
---

你跑完一个策略回测，年化夏普 1.5，净值曲线漂亮得像教科书。同事问了一句：「这玩意儿……是真有 alpha，还是只是运气好？」

你卡住了。因为**没人教过你怎么给一条回测结果算 p 值**。

传统做法是用 t 检验 `t = Sharpe × √(T)` 判断「夏普是否显著不为 0」。但这条公式有致命前提：**收益必须近似独立同分布正态**。而真实策略收益——尖峰厚尾、有自相关、样本只有 120 个月——每一条前提都被打破。直接套 t 检验，等于用尺子量温度。

**Bootstrap（自助法）**走另一条路：不假设收益长什么样，直接从你**自己的回测残差**里反复重抽样，在「零假设：根本没有 alpha」的世界里，把「夏普=0」的分布硬重建出来。然后问一句：**在我这 120 个月里，纯靠随机抽到 1.5 夏普的概率有多大？** 这个概率，就是 p 值。

本文用自包含合成数据，把整套逻辑从方程推到可运行代码，并诚实拆穿四类真实陷阱。

## 一、问题：为什么 t 检验救不了你

经典检验的零假设是「真实夏普 SR=0」，检验统计量：

```
t = SR_obs × √T        # T 为样本期数（如 120 个月）
p 来自 t 分布(t = t_{T-1})
```

它漂亮，但只在三件事同时成立时可靠：

1. **独立**：月度收益互不相关（动量/反转策略直接破防）。
2. **正态**：收益近似高斯（市场崩溃时 −20% 的尾巴，高斯几乎给零概率）。
3. **样本够长**：T 大才能保证 √T 近似起效（多数策略只有几年数据）。

现实里这三条常常全破。于是你会看到：一个纯随机的策略（SR 真=0），t 检验也可能给你「p=0.03 显著」。这叫**假阳性**——你信了一个不存在的 alpha。

## 二、Bootstrap 思路：用你的残差重建「无 alpha 的世界」

核心直觉非常朴素：

> 如果你的策略根本没 alpha，那它的收益里**只有噪声**。而噪声的「形状」就藏在你的残差里。

所以我们：

1. 拿回测收益 `r_t`，去均值得到**零假设残差** `r0_t = r_t − mean(r)`（去均值 = 把「真 alpha」抽掉，留下纯噪声）。
2. 对 `r0_t` 做 **iid 重抽样** B 次（每次抽 T 个、可重复），得到 B 条「假如没有 alpha，收益会是什么样」的伪样本。
3. 对每条伪样本算夏普，得到**零假设下的夏普分布**。
4. p 值 = 这个分布里 ≥ 你观测夏普的比例：

```
p = (1/B) · Σ I( SR_bootstrap_b >= SR_obs )
```

p 越小，说明「纯噪声抽到你这水平」越不可能——你的 alpha 越可信。

关键：**我们重抽样的是残差（去均值后的噪声），不是原始收益**。这样 bootstrap 世界严格满足 H0，p 值才有意义。

## 三、从零复现：两个看起来都不错的净值

我们先造两个 120 个月的策略：一个是**真有 alpha**（年化 SR=0.73），一个是**纯噪声**（SR=0.03）。光看净值，俩都往上爬，肉眼分不出谁真谁假。

```python
import numpy as np

def sharpe(x, ann=12.0):
    x = np.asarray(x, float)
    if x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / x.std(ddof=1) * np.sqrt(ann)

def gen_returns(true_sharpe, T=120, ann=12.0, seed=0):
    r = np.random.default_rng(seed)
    sd = 0.04                                  # 月度波动
    mu = true_sharpe / np.sqrt(ann) * sd      # 反推月均收益
    return r.normal(mu, sd, size=T)

ret_alpha = gen_returns(0.8, T=120, seed=1)    # 真 alpha（年化 SR≈0.73）
ret_noise = gen_returns(0.0, T=120, seed=21)   # 纯噪声
```

![两个净值曲线：真 alpha 与纯噪声肉眼难分](/images/bootstrap-backtest-pvalue/strategy_equity_compare.png)

## 四、Bootstrap p 值：让运气当场现形

重抽样残差，重建零假设分布，算 p 值：

```python
def bootstrap_pvalue(ret, B=2000, seed=1):
    r = np.random.default_rng(seed)
    obs = sharpe(ret)                 # 观测夏普（年化）
    resid = ret - ret.mean()          # 去均值 -> H0 残差（纯噪声）
    idx = r.integers(0, len(ret), size=(B, len(ret)))
    boots = resid[idx]                # 重抽样 B 条伪样本
    m = boots.mean(axis=1)
    s = boots.std(axis=1, ddof=1)
    nulls = np.where(s > 0, m / s * np.sqrt(12), 0.0)
    p = np.mean(nulls >= obs)         # 纯噪声抽到 >= 观测 的比例
    return obs, nulls, p

obs_a, nulls_a, p_a = bootstrap_pvalue(ret_alpha,  seed=3)
obs_n, nulls_n, p_n = bootstrap_pvalue(ret_noise, seed=9)
print(obs_a, p_a)   # 0.73  0.015   <- alpha 显著
print(obs_n, p_n)   # 0.03  0.473   <- 噪声 不显著
```

跑出来的结果一针见血：

- **真 alpha 策略**：观测年化 SR=0.73，p=0.015。在「无 alpha」世界里，重抽样 2000 次只有 1.5% 能抽到这么高——**基本可以排除运气**。
- **纯噪声策略**：观测年化 SR=0.03，p=0.473。接近一半的随机样本都比它高——**彻底暴露为噪声**。

![零假设下的夏普 bootstrap 分布](/images/bootstrap-backtest-pvalue/bootstrap_null_distribution.png)

注意右图红线和左图红线的区别：左图观测值（红线）远远甩在零假设分布右侧，所以 p 极小；右图观测值几乎落在分布中央，p 自然很大。**p 值不是夏普有多高，而是「纯噪声要多努力才能摸到你」**。

## 五、功效分析：真 alpha 越强，bootstrap 越容易抓到你

p 值只能告诉你「不像噪声」。但反过来问：如果一个策略**真有** SR=0.6，bootstrap 有多大概率认出它？这个概率叫**功效（power）**。

我们对真 SR ∈ {0, 0.3, 0.6, 0.9, 1.2} 各造 60 个策略，统计被 5% 阈值拒绝的比例：

```python
true_srs = [0.0, 0.3, 0.6, 0.9, 1.2]
reject = []
for ts in true_srs:
    rr = 0
    for k in range(60):
        ret = gen_returns(ts, T=120, seed=1000 + k)
        _, _, p = bootstrap_pvalue(ret, seed=2000 + k)
        if p < 0.05:
            rr += 1
    reject.append(rr / 60.0)
# [0.083, 0.283, 0.617, 0.883, 0.983]
```

![功效分析：真夏普越强，5% 显著性下被拒绝比例越高](/images/bootstrap-backtest-pvalue/bootstrap_power_analysis.png)

读这张图的三层含义：

1. **真 SR=0（纯噪声）**：拒绝率 8.3%，略高于名义 5%——这是 bootstrap 的**一类错误**基线，说明即使零假设成立也有小概率误报，属正常。
2. **真 SR=0.3（弱 alpha）**：只有 28% 被认出——**样本只有 120 个月时，弱 alpha 极难检出**。这本身就是结论：别指望靠 10 年数据证明一个薄 alpha。
3. **真 SR=0.9+（强 alpha）**：98% 被认出——**够强的 alpha，bootstrap 一抓一个准**。

## 六、置信区间：不止 p 值，还要知道夏普的误差带

p 值只回答「像不像噪声」，但**实战更需要知道「我的真实夏普大概在什么区间」**。同一个 bootstrap 重抽样，顺手就能给出**百分位置信区间**：把 B 条伪样本算出的夏普排序，取 2.5% 和 97.5% 分位即可。

```python
# 复用上一节的 nulls(零假设下分布);这里我们要的是「观测样本的 bootstrap 分布」
# 对原始收益(保留均值)重抽样,得到 SR 自身的抽样分布
def sharpe_ci(ret, B=2000, seed=5, alpha=0.05):
    r = np.random.default_rng(seed)
    T = len(ret)
    idx = r.integers(0, T, size=(B, T))
    boots = ret[idx]
    s = boots.std(axis=1, ddof=1)
    srs = np.where(s > 0, boots.mean(axis=1) / s * np.sqrt(12), 0.0)
    lo, hi = np.percentile(srs, [100*alpha/2, 100*(1-alpha/2)])
    return srs.mean(), lo, hi

mean_sr, lo, hi = sharpe_ci(ret_alpha, seed=5)
print("SR 点估计=%.2f  95%% CI=[%.2f, %.2f]" % (mean_sr, lo, hi))
# 真 alpha 策略: 95% CI 整体 > 0 -> 更踏实的结论
```

对一个**真 alpha 策略**，这条 95% 置信区间通常**整体位于 0 上方**——你不仅知道「不像噪声」，还知道「乐观/悲观情景下夏普也至少是 0.2 而不是 −0.5」。而对**纯噪声策略**，区间会厚厚地跨过 0，提醒你「上下都有可能」。一句话：**点估计告诉你中心在哪，置信区间告诉你这个中心有多不靠谱**。

## 七、四类真实陷阱

**陷阱 1：块 bootstrap 的缺失（最重要）**
本文用的是 **iid bootstrap**（逐点重抽样），它假设收益独立。但**有自相关/动量的策略，逐点打乱会破坏序列结构，p 值失真**。正确做法是用 **块 bootstrap（block bootstrap）**——把连续的 k 期作为一个块整体重抽样，保留短期依赖。本文为演示方法用 iid；实盘若策略有月度自相关，必须换块 bootstrap，否则你会系统性地**高估**显著性。

**陷阱 2：多重检验与 p-hacking**
你筛了 50 个参数组合，挑 p 值最小的那个报——这叫**数据挖掘**。50 次独立检验里，哪怕全是无 alpha，也有约 `1−0.95⁵⁰ ≈ 92%` 概率出现至少一个 p<0.05。补救：用 Bonferroni / False Discovery Rate 校正，或只在**样本外**数据上算一次 p 值。

**陷阱 3：幸存者偏差进 bootstrap**
如果你的回测收益来自「已剔除退市股票」的价表，那残差本身就是被美化过的，bootstrap 只会把美化后的噪声当真相，p 值虚低。必须在含退市的全样本上算。

**陷阱 4：短样本下零假设的脆弱**
T=120 时，即使真有 alpha，块/bootstrap 的零假设残差也只能「绕着你这 120 个点打转」。如果这 120 个月恰好是牛市，去均值后的残差仍带着牛市的波动结构，p 值会偏乐观。**跨多个市场状态、拉长样本**，是唯一的硬解。

## 七、落地路径

- **数据源**：用 `westock-data` 取真实日/月收益序列替代合成 `gen_returns`，记得前视处理（信号在 t 生成，t+1 以开盘价执行）。
- **检验升级**：把 iid bootstrap 换成**块 bootstrap**（如 `arch.bootstrap.StationaryBootstrap`），块长用 ACF 衰减估计。
- **报告规范**：p 值永远配合**置信区间**与**样本外**结论一起报，绝不单甩一个 0.015。
- **多重检验**：参数扫描后报 FDR 校正后的 q 值，而非原始 p。

## 八、要点速记

- **零假设残差**：`resid = r − mean(r)`，重抽样的是它，不是原始收益——否则 p 值无效。
- **p 值含义**：纯噪声重抽样里 ≥ 你观测夏普的比例；小才可信，不是夏普越高 p 越小。
- **iid vs 块**：有自相关的策略必须换**块 bootstrap**，否则显著性被高估。
- **永远配置信区间**：点估计 + 95% CI + 样本外，三件套一起报。
- **防 p-hacking**：参数扫描后做 FDR / Bonferroni 校正，或只在样本外算一次。

## 结论

回测的夏普告诉你「赚了多少」，bootstrap 的 p 值告诉你「这钱值不值得信」。一句口诀：**高夏普 + 低 p 值 = 真 alpha；高夏普 + 高 p 值 = 好运气的马甲**。在把真金白银压上去之前，先让你的残差重抽 2000 次——它会比你更诚实。

> 文中所有图表与数字均由 `gen_bootstrap_backtest_pvalue.py` 用合成数据真实计算生成；数字仅用于演示方法，不构成任何投资建议。
