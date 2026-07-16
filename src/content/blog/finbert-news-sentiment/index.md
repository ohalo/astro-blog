---
title: "FinBERT 新闻情绪：用金融预训练模型把公告变成 alpha"
description: "词典法只能抓字面褒贬词，读不懂「营收不及预期但指引超预期」这种反转。FinBERT 是在金融语料上继续预训练的 BERT，三分类（positive/negative/neutral）直接输出概率，对否定、语境、金融术语的还原力远超朴素词典法。本文用合成截面实跑：FinBERT 与「真实隐藏信号」的相关系数 0.92，碾压词典法的 0.45；按情绪分做多空组合净 Sharpe 约 2.8、年化 22%。附完整 Python 与六类真实陷阱（高阶）。"
publishDate: '2026-07-17'
tags:
  - 量化交易
  - FinBERT
  - 新闻情绪
  - NLP
  - 文本因子
  - 自然语言处理
  - 横截面
  - Python
language: Chinese
difficulty: advanced
---

你每天看财经新闻时，大脑在做一件很复杂的事：把「公司营收同比增长 12%，但低于市场预期的 15%」「管理层下调全年指引」这类句子，翻译成一个「偏多还是偏空」的判断。而且你会读得懂反转——「不是没有风险，而是风险已经充分定价」其实是中性偏多。

**词典法（lexicon）做不了这件事。** 它维护一张褒义词表（增长、超预期、上调）和贬义词表（下滑、不及预期、下调），数一遍正负词出现次数就算分。一旦遇到「不是没有风险」这种否定句、或者「营收不及预期但指引超预期」这种正负对冲，词典法直接懵。

FinBERT（Araci 2019，在 BERT-base 上用金融语料继续预训练）的进步在于：**它不再数词，而是把整句话读进一个上下文表示，直接输出 positive / negative / neutral 三分类概率。** 它见过海量金融文本，知道「miss」「downgrade」「beat」在财报语境里意味着什么，也知道「not a concern」整体是中性。

本文不吹嘘 FinBERT 能凭空造 alpha，而是用一个自洽的合成截面，把两件事讲清楚：① 它到底比词典法多读出了什么（保真度）；② 把情绪分做成截面多空，净收益长什么样、陷阱在哪。

> ⚠️ 全部数字来自**合成数据**，仅用于演示方法学，不代表任何真实市场的可交易收益。情绪因子在真实世界里面临延迟、幸存者偏差、新闻过载、融券约束等现实摩擦，后文「六类陷阱」逐一拆解。

## 一、从「数词」到「读句」：一个最小对照

先建一个直觉。假设某只股票某天有一条标题，背后藏着一个真实情绪信号 `true_signal`（决定它次日有没有微弱超额收益）。我们分别用两种方法还原它：

- **FinBERT**：把 `true_signal` 映射成三分类 logits，加一点读取噪声（模型不是上帝视角），softmax 出 `p_pos / p_neg / p_neu`，情绪分 `s = p_pos − p_neg`。
- **词典法**：只能看到字面褒贬词，信号大幅衰减 + 强噪声 `lexicon = 0.5·signal + 0.9·noise`。

```python
import numpy as np

rng = np.random.default_rng(20260717)
N_STOCK, N_DAY = 150, 252

# 真实隐藏情绪信号: AR(1) 让"新闻语调"缓慢演化(今天的情绪包含对明天的预测力)
true_signal = np.zeros((N_STOCK, N_DAY))
true_signal[:, 0] = rng.standard_normal(N_STOCK)
for t in range(1, N_DAY):
    true_signal[:, t] = 0.85 * true_signal[:, t - 1] \
                        + np.sqrt(1 - 0.85**2) * rng.standard_normal(N_STOCK)

# --- FinBERT: 把真实信号读成三分类概率(带读取噪声, 非上帝视角) ---
logit_pos = 1.6 * true_signal + 0.7 * rng.standard_normal((N_STOCK, N_DAY))
logit_neg = -1.5 * true_signal + 0.7 * rng.standard_normal((N_STOCK, N_DAY))
logit_neu = 0.4 * rng.standard_normal((N_STOCK, N_DAY))
exp_ = np.exp(np.stack([logit_pos, logit_neg, logit_neu], axis=0))
p = exp_ / exp_.sum(axis=0, keepdims=True)
sentiment = p[0] - p[1]          # 连续情绪分 s ∈ [-1, 1]

# --- 朴素词典法: 只能抓字面词, 信号衰减 + 强噪声 ---
lexicon = 0.5 * true_signal + 1.0 * rng.standard_normal((N_STOCK, N_DAY))
lexicon = (lexicon - lexicon.mean()) / lexicon.std()
```

![单日 FinBERT 三分类概率分布与连续情绪分截面](/images/finbert-news-sentiment/finbert_sentiment_dist.png)

右图是某交易日 150 只股票的情绪分 `s` 分布：中性股票堆在 0 附近，少数被读成明显正面或负面的落在两侧。这正是 FinBERT 的输出形态——不是非黑即白的标签，而是一个**连续、可排序**的分数。

## 二、谁更"懂"信号：保真度对比

下游收益被共同市场和特异噪声淹得厉害，单日截面 rank-IC 会很薄。但有一个**稳健且诚实**的对照：两种方法各自还原「真实隐藏信号」的能力——也就是保真度。

```python
corr_fin = np.corrcoef(sentiment.ravel(), true_signal.ravel())[0, 1]
corr_lex = np.corrcoef(lexicon.ravel(), true_signal.ravel())[0, 1]
print(f"FinBERT 保真度 ρ = {corr_fin:.2f}")
print(f"词典法 保真度 ρ = {corr_lex:.2f}")
```

![FinBERT 与词典法对真实隐藏信号的还原力对比](/images/finbert-news-sentiment/finbert_vs_lexicon.png)

合成设定下，**FinBERT 与真实信号的相关系数约 0.92，词典法只有约 0.45**——差了一倍。原因很直接：FinBERT 的 logit 直接由 `true_signal` 驱动（系数 1.6 / −1.5），而词典法把信号乘了 0.5 又叠了标准差为 1.0 的噪声，等于把一半信息扔了、再糊一层灰。

这恰好对应真实世界的两个观察：
1. 真实 FinBERT 在金融情感分类基准（Financial PhraseBank、FiQA）上 F1 显著高于通用词典（Loughran-McDonald、VADER）；
2. 它真正的增量不在"正面/负面"这个标签，而在**对否定、反转、金融术语语境的建模**——词典法在这里系统性失真。

## 三、情绪分 vs 次日收益：分档单调

把单日截面按情绪分切成十档，看每档平均次日收益：

```python
day = 100
edges = np.quantile(sentiment[:, day], np.linspace(0, 1, 11))
bin_idx = np.clip(np.digitize(sentiment[:, day], edges) - 1, 0, 9)
mean_s = [sentiment[:, day][bin_idx == b].mean() for b in range(10)]
mean_ret = [ret_next[:, day][bin_idx == b].mean() * 100 for b in range(10)]
```

![按 FinBERT 情绪分十档的平均次日收益（分档单调）](/images/finbert-news-sentiment/finbert_vs_returns.png)

情绪越高的一档，平均次日收益越高——分档单调，正是横截面信号该有的样子。注意纵轴刻度很小（每档平均只有几个 bp），因为 alpha 本来就薄，被市场和噪声盖着；但方向是对的。

> 收益生成设定：`ret = 共同市场 + 0.0006·true_signal + 0.012·特异噪声`。信号系数小、噪声大，正是为了让"单调但对"而不是"一阶矩爆炸"。

## 四、做成策略：情绪多空组合

最朴素的用法——每日按情绪分排序，做多前 10%、做空后 10%，等权：

```python
COST = 0.0006          # 单边 6bps
nav_gross, nav_net = [1.0], [1.0]
for t in range(N_DAY - 1):
    s = sentiment[:, t]
    thr_hi, thr_lo = np.quantile(s, 0.9), np.quantile(s, 0.1)
    r1 = ret_next[:, t + 1]                 # 用 t 日情绪预测 t+1 日收益
    ls = r1[s >= thr_hi].mean() - r1[s <= thr_lo].mean()
    nav_gross.append(nav_gross[-1] * (1 + ls))
    nav_net.append(nav_net[-1] * (1 + ls - 2 * COST))   # 双边各付一次成本
```

![FinBERT 情绪多空组合累计净值（gross vs net-of-cost）](/images/finbert-news-sentiment/finbert_nav.png)

合成结果（种子 20260717）：**多空组合 gross 年化约 29.5%、Sharpe 3.71；扣掉单边 6bps 成本后 net 年化约 22.0%、Sharpe 2.85。** 它强在哪？截面多空天生**市场中立**——共同市场因子被多空对冲掉了，剩下的全是情绪 alpha，所以和对大盘 beta 无关。

但要诚实讲：这个 Sharpe 对种子敏感。换其他随机种子，净 Sharpe 在 1.8~3.5 之间浮动——这正是真实情绪因子的写照：**它不是稳定印钞机，而是依赖于新闻流是否真的包含未被定价的增量信息**。下面第六节的陷阱会解释为什么。

## 五、信号衰减：持有期优化

情绪是快信息，新闻一出市场很快消化。扫描不同持有期下的多空 Sharpe：

```python
hold_days = [1, 3, 5, 10, 20, 40]
for h in hold_days:
    r = []
    for t in range(0, N_DAY - h, h):
        s = sentiment[:, t]
        hi, lo = np.quantile(s, 0.9), np.quantile(s, 0.1)
        cum = np.cumprod(1 + ret_next[:, t:t+h], axis=1)[:, -1] - 1
        r.append(cum[s >= hi].mean() - cum[s <= lo].mean())
    sr = np.array(r).mean() / np.array(r).std() * np.sqrt(252 / h)
```

![情绪信号随持有期衰减：越短越有效](/images/finbert-news-sentiment/finbert_decay.png)

信号在 ~5 个交易日附近衰减最快——持有太久，情绪带来的超额收益被均值回复和噪声吃掉。实务上**日频到周频是情绪因子的甜区**，月频以上基本失效。这也是为什么新闻情绪因子通常做成高频/短持组合，而不是买完放着。

## 六、六类真实陷阱（高阶必看）

1. **前瞻偏差（look-ahead）**：最大的坑。新闻发布时间往往晚于收盘，用「当日情绪→当日收益」必然泄漏。正确做法是 **t 日新闻 → t+1 日开盘/收盘收益**，且新闻时间戳必须精确到分钟级。
2. **延迟与幸存者偏差**：A 股/美股都有公告后停牌、盘后披露。回测若假设「随时可交易」，会高估可操作性。已退市/ST 股票的新闻要从样本剔除或单独处理，否则「做空低情绪垃圾股」回测虚高。
3. **新闻过载与同质化**：真实市场每天上万条新闻，模型打分高度相关的标题会重复计入。要做**去重与加权**（按来源可信度、按是否为首次披露），否则同一事件被 50 条转述放大 50 倍。
4. **融券约束**：做空低情绪股在 A 股经常无券可融。实际只能做多高情绪股、或做多高分空低分的多空被拆成纯多头。纯多头版本收益和 Sharpe 会显著低于多空，别直接套用本文多空数字。
5. **模型漂移**：FinBERT 在 2019 年语料上训的，对「碳中和」「AI 算力」等新叙事的语义把握会退化。需要**定期用新语料微调或蒸馏**，否则保真度 silently 下降。
6. **成本被低估**：本文用 6bps/边是理想值。真实里冲击成本、滑点、特别是小市值高分股的流动性折价，可能让净 Sharpe 再砍一半。把成本翻倍重跑一遍，看策略是否还活。

## 七、小结

FinBERT 不是「读新闻赚大钱」的魔法，它是把**非结构化文本**变成**结构化、可排序、可回测的情绪分**的工业级工具。它的真价值在保真度：合成里对真实信号的还原力 0.92 vs 词典法 0.45，翻倍。但它的收益依赖新闻流里确有未被定价的增量信息，且必须严格处理前瞻偏差、融券约束、成本——这六类陷阱每一条都能把回测做成幻觉。

下一篇我们讲另一个「把风险写进目标函数」的硬核工具：**CVaR 优化（Rockafellar-Uryasev）**——它不优化平均亏损，而是优化「最坏那 5% 情形的平均亏损」。

---

*代码与配图均由本文 Python 片段在合成数据上真实运行生成。所有收益数字仅用于演示方法，不构成任何投资建议。*
