---
title: "多重检验夏普折扣：用 Bonferroni/BHY 给数据挖掘出来的夏普打折"
publishDate: '2026-07-27'
description: "多重检验夏普折扣：用 Bonferroni/BHY 给数据挖掘出来的夏普打折 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

假设你写了个回测框架，一晚上跑了 500 个策略变体，第二天早上发现其中一个夏普率高达 2.3。你会不会立刻上仓位？如果会，那你很可能正踩在量化研究最大的一个坑上——**多重检验偏差**（multiple testing bias）。跑得越多，找到"看起来很牛"却纯属运气的策略的概率就越高。这篇文章讲的就是怎么给这种数据挖掘出来的夏普率打个合理的折扣，还它一个真实面目。

## 问题的本质：搜索得越多，运气越容易伪装成技能

先做个思想实验。假设你测试的所有策略其实**都没有真实 alpha**，真实夏普率都是 0。由于噪声，每个策略的样本夏普率会随机分布在 0 上下。如果你只测一个，得到夏普率 2 的概率很小；但如果你测 500 个，那么"至少有一个偶然达到 2"几乎是必然事件。

用统计语言说：单次检验的显著性水平 α（比如 5%）在多次检验下会失控。测 N 次独立检验，至少犯一次一类错误（把噪声当成 alpha）的概率是 `1 − (1 − α)^N`。N = 20 时这个概率就飙到 64%，N = 100 时是 99.4%。也就是说，只要你搜的策略够多，几乎必然会"发现"一堆假信号。

Campbell Harvey 和 Yan Liu 在他们著名的《…and the Cross-Section of Expected Returns》和《Backtesting》系列论文里反复强调这一点：金融学文献里发表的几百个"因子"，绝大多数没有经过多重检验校正，因此其中很大比例其实是数据挖掘的产物。他们提出的解决方案就是——给夏普率打折（haircut）。

![多重检验下的假发现](/images/haircut-sharpe-multiple-testing/multiple-testing.jpg)

## 从夏普率到 t 统计量

要做多重检验校正，先得把夏普率翻译成统计检验的语言，也就是 t 统计量。对一段长度为 T（年）、每年观测 N 次的收益序列，夏普率 SR 和 t 值的关系近似为：

```
t = SR × √(观测期总年数)
```

更精确地，若有 `n` 个收益观测，年化夏普率 `SR_annual`，则：

```
t ≈ SR_annual × √(n / 每年观测数)  =  SR_annual × √(年数)
```

举例：一个策略年化夏普 1.0，回测了 10 年，那么 `t ≈ 1.0 × √10 ≈ 3.16`。有了 t 值，就能算出对应的单次检验 p 值，然后再用多重检验方法去调整这个 p 值。

```python
import numpy as np
from scipy import stats

def sharpe_to_tstat(sharpe_annual, years):
    """年化夏普率 -> t 统计量"""
    return sharpe_annual * np.sqrt(years)

def tstat_to_pvalue(t, two_sided=True):
    """t 统计量 -> p 值（用正态近似，大样本足够）"""
    p = stats.norm.sf(abs(t))
    return 2 * p if two_sided else p

sr, years = 1.0, 10
t = sharpe_to_tstat(sr, years)
p = tstat_to_pvalue(t)
print(f"夏普 {sr} / {years}年 => t = {t:.2f}, p = {p:.4f}")
```

## 三种校正方法：Bonferroni、Holm、BHY

假设你一共尝试了 M 个策略（这个 M 要包括所有你试过的、没发表的变体，这是关键也是最难诚实面对的地方）。有三种主流的多重检验校正：

**1. Bonferroni（最保守）。** 直接把显著性门槛除以 M：要求 `p < α / M`。等价地，把每个 p 值乘以 M。它控制的是 FWER（族错误率，即至少犯一次错的概率），假设检验间独立或任意相关都成立，但因为太保守，会拒绝掉很多真实信号。

**2. Holm（逐步 Bonferroni）。** 把 M 个 p 值从小到大排序，第 k 小的 p 值乘以 `(M − k + 1)`。它同样控制 FWER，但比 Bonferroni 更有力（拒绝更多），且无需额外假设。几乎总是应该用 Holm 替代原始 Bonferroni。

**3. BHY（Benjamini-Hochberg-Yekutieli）。** 控制的是 FDR（错误发现率，即被判为显著的里面假的比例），比 FWER 宽松。Harvey-Liu 特别推荐 BHY，因为它考虑了检验间的相关性（金融因子高度相关），在"太松"和"太严"之间取得平衡。

```python
def haircut_sharpe(sharpe, years, M, method='holm', alpha=0.05):
    """
    对数据挖掘出来的夏普率打折。
    返回：原始 p、校正后 p、是否仍显著、折扣后的夏普率
    """
    t = sharpe * np.sqrt(years)
    p_raw = 2 * stats.norm.sf(abs(t))

    if method == 'bonferroni':
        p_adj = min(p_raw * M, 1.0)
    elif method == 'holm':
        # 单个策略的 Holm 上界近似等于 Bonferroni（最悲观位置）
        p_adj = min(p_raw * M, 1.0)
    elif method == 'bhy':
        # BHY 带相关性修正项 c(M) = sum(1/i)
        cM = np.sum(1.0 / np.arange(1, M + 1))
        p_adj = min(p_raw * M * cM / 1.0, 1.0)
    else:
        raise ValueError(method)

    # 由校正后的 p 反推"折扣后"的 t 和夏普
    t_adj = stats.norm.isf(p_adj / 2)
    sr_adj = t_adj / np.sqrt(years)
    haircut_pct = (1 - sr_adj / sharpe) * 100 if sharpe else 0

    return {
        'p_raw': p_raw,
        'p_adj': p_adj,
        'significant': p_adj < alpha,
        'sharpe_raw': sharpe,
        'sharpe_adj': max(sr_adj, 0),
        'haircut_pct': haircut_pct,
    }

# 那个"闪闪发光"的夏普 2.3，其实是 500 次搜索的产物
res = haircut_sharpe(sharpe=2.3, years=5, M=500, method='bhy')
for k, v in res.items():
    print(f"{k:>12}: {v}")
```

跑一下你会发现，即便原始夏普 2.3 看起来很唬人，一旦考虑到它是 500 次搜索里挑出来的，折扣后的夏普率会大幅缩水，折扣比例经常在 30%–60%。有时候校正后 p 值甚至越过了显著性门槛——意味着这个"发现"在统计上根本站不住脚。

## 一个批量评估的例子

真实场景往往是一批策略一起评估。下面模拟"一堆纯噪声策略 + 少数真 alpha"，看多重检验校正能不能把它们区分开：

```python
np.random.seed(7)
years = 8

# 45 个纯噪声策略（真夏普=0）+ 5 个真 alpha（真夏普≈1.2）
true_sr = np.r_[np.zeros(45), np.full(5, 1.2)]
# 观测夏普 = 真夏普 + 抽样噪声（噪声标准差≈1/√years）
obs_sr = true_sr + np.random.normal(0, 1/np.sqrt(years), size=50)
M = len(obs_sr)

t_all = obs_sr * np.sqrt(years)
p_all = 2 * stats.norm.sf(np.abs(t_all))

# BHY / FDR 校正
order = np.argsort(p_all)
cM = np.sum(1.0 / np.arange(1, M + 1))
p_sorted = p_all[order]
# BHY 阈值：p_(k) <= (k / (M*cM)) * alpha
alpha = 0.10
thresh = (np.arange(1, M+1) / (M * cM)) * alpha
passed = p_sorted <= thresh
k_max = np.max(np.where(passed)[0]) + 1 if passed.any() else 0

selected = order[:k_max]
print(f"校正前 p<0.05 的策略数: {(p_all < 0.05).sum()}")
print(f"BHY 校正后入选策略数:   {len(selected)}")
print(f"其中真 alpha (index>=45) 命中: "
      f"{(selected >= 45).sum()} / 5")
print(f"其中假阳性 (index<45):        "
      f"{(selected < 45).sum()}")
```

这个例子能直观展示：不做校正时，会有一堆噪声策略偶然穿过 5% 门槛；做了 BHY 之后，绝大多数假阳性被过滤掉，真 alpha 得以保留。这正是多重检验校正的价值——它不是让你什么都不敢信，而是帮你把"运气"和"实力"分开。

![夏普率折扣](/images/haircut-sharpe-multiple-testing/sharpe-haircut.jpg)

## 实践中的几个关键点

**M 到底该取多少？** 这是最难也最诚实的一环。M 不只是你"记录在案"的回测数，而是包括所有隐性尝试：换参数、换窗口、换标的池、换止损规则……每一次"再试试"都算一次检验。Harvey 建议对已发表的因子研究，M 的量级应该是数百甚至上千。对自己的研究，保守起见宁可高估 M。

**别只看夏普。** 多重检验校正只解决了"统计显著性"，它不能告诉你策略是否有经济逻辑、是否稳健、交易成本吃掉多少。校正后依然显著只是必要条件，不是充分条件。

**样本外才是终极裁判。** 任何样本内的校正都比不过一段干净的样本外验证。多重检验折扣是在你没法做真正样本外测试时的"事前打折"，但如果条件允许，留出一段从未看过的数据做最终确认，永远是最有说服力的。

**报告要透明。** 如果你写研究报告，务必披露你一共试了多少个变体。只报告最好的那个而隐瞒搜索空间，本质上就是在制造统计谎言。

## 小结

夏普率会骗人，尤其是当它是从大量搜索里挑出来的时候。多重检验校正——无论是保守的 Holm 还是更平衡的 BHY——给了我们一个量化的手段，把"跑了 500 次终于蒙对一次"和"真的找到了 alpha"区分开。核心心法只有一句：**你搜索得越多，就该对结果越怀疑，对夏普率打的折扣就该越狠。** 下次看到一个惊艳的回测，先问一句"这是从多少次尝试里选出来的？"，再决定要不要相信它。
