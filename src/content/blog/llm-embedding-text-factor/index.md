---
title: "大语言模型的嵌入作为文本因子：把新闻和公告变成可回测的截面信号"
publishDate: '2026-07-11'
description: "LLM 的句子/文档嵌入把整段金融文本压成稠密向量，天然保留语序、否定与上下文，是连接「文本」与「截面模型」的桥。本文用可复现模拟讲清两种用法（检索式 vs 监督式）、为什么无监督 PCA 几乎挖不出信号而监督映射能、以及嵌入漂移/前视/维度灾难等真实陷阱，并给出落地 pipeline 与完整 Python 实现。"
tags:
  - 量化交易
  - 文本因子
  - LLM嵌入
  - 自然语言处理
  - 因子研究
  - Python
language: Chinese
difficulty: advanced
---

很多团队做文本因子，第一步是用 `jieba` 分词 + 情感词典，或者 TF-IDF 把新闻压成几万维稀疏词频。这类方法能用，但有两个硬伤：**丢语序**（"业绩不及预期但指引向好" 和 "指引向好但业绩不及预期" 在词频上几乎一样）、**吃不动长语境**（一份 50 页年报里的语义，不是几个关键词能代表的）。

大语言模型的**嵌入（embedding）**正好补上这块：它把任意长度的文本映射成一个固定维度的稠密向量，语义相近的文本在向量空间里距离近。本文不聊"用大模型预测涨跌"那种玄学，而是把嵌入当成一类**新的特征工程手段**——怎么把它接进我们熟悉的截面因子框架里，并用可复现的方式验证它到底有没有用。

## 一、嵌入当因子的两种用法

**用法 A：检索 / 相似度式。** 把"历史上发布后大涨的公告"的嵌入算出来，每天对新公告算相似度，相似度高就打分。直觉好，但有一个致命前视风险：你拿"后视镜"里的涨幅去定义"相似"，等于把未来信息泄露进了当下信号。要做也得用**滚动历史窗口**定义相似锚，且只在样本内建锚。

**用法 B：监督式（本文重点）。** 把嵌入直接当特征，训练一个模型 `f(embedding) → 下期收益`，模型的输出（或中间表征）就是因子。这条路的优点是可 OOS 检验、可行业中性化、可和量价因子合并，缺点是嵌入维度高（常 384~1536 维），小样本上极易过拟合。

下面用一套可复现模拟把用法 B 跑通，并正面回答一个问题：**嵌入里到底有没有"收益方向"的信息？还是只是一堆语义结构？**

## 二、可复现实验：在嵌入里埋一个真实信号

我们模拟 500 只股票 × 250 个交易日，每只每日一条"新闻嵌入"（384 维）。真实世界里这条嵌入来自 `sentence-transformers`；为了可复现、不依赖下载模型，这里用**合成嵌入**：在一个随机 384 维向量里，把 8 个特定维度方向作为"真实信号方向"，未来收益由信号方向的投影加噪声生成。

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

rng = np.random.default_rng(42)
N, T, d = 500, 250, 384
signal_dims = rng.choice(d, 8, replace=False)
w = np.zeros(d); w[signal_dims] = rng.normal(0, 1, 8); w /= np.linalg.norm(w)

# 合成嵌入：每条 (股票, 日) 一个 384 维向量，信号埋在 w 方向上
E = rng.normal(0, 1, (N * T, d))
s = (E @ w); s = (s - s.mean()) / s.std()
b = 0.04                                   # 信号强度
fwd_ret = b * s + rng.normal(0, 1.0, N * T)   # 次日收益(%)，与信号弱相关

# 生产路径（注释）：真实嵌入用 sentence-transformers 一步得到
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
# E = model.encode(texts, normalize_embeddings=True)   # 维度 768
```

> 生产环境请固定模型版本并**缓存嵌入**：同一段文本重新编码会得到不同向量（不同 checkpoints），历史因子不可比。合成嵌入只是为了让下面的数字可被你一键复现。

### 2.1 无监督 PCA 几乎挖不出信号

如果"语义结构"直接等于"收益结构"，那对嵌入做 PCA 取前几个主成分当因子就该有效。我们验证一下：

```python
E_day, R_day = E.reshape(T, N, d), fwd_ret.reshape(T, N)
ic_pca = []
for t in range(T):
    c = PCA(n_components=1).fit_transform(E_day[t])[:, 0]
    ic_pca.append(np.corrcoef(c, R_day[t])[0, 1])
print("PCA-top1 平均 IC:", np.nanmean(ic_pca))          # ≈ 0.0024 (≈0.2%，几乎无信号)

pca10 = PCA(n_components=10).fit(E.reshape(-1, d))
print("前10主成分解释方差:", pca10.explained_variance_ratio_[:10].sum())  # ≈ 2.9%
```

结果很说明问题：**嵌入前 10 个主成分只解释 2.9% 的方差，而 PCA 第一主成分与收益的 IC 只有约 0.2%**。也就是说，嵌入的"能量"主要分布在语义主轴上（谁在聊新能源、谁在聊并购），这些方向和"明天涨不涨"几乎正交。直接拿无监督降维当因子，基本是浪费。

![文本嵌入的 2D 投影：颜色=次日收益（信息藏在方向里，不在主轴上）](/images/llm-embedding-text-factor/embedding_pca_projection.png)

### 2.2 监督映射才能把信号"对齐"出来

真正有效的做法是用收益做监督，把嵌入空间**对齐**到收益方向。最朴素的实现是岭回归 `embedding → 收益`，预测值本身就是因子；为防前视，用**时间序列切分**（用前 150 天训练，预测第 t 天），而不是随机抽样。

```python
def oos_factor_and_ic(E_all, R_all, cal_days=150):
    pred = np.full(R_all.shape, np.nan)
    for t in range(cal_days, T):
        ridx = rng.permutation(t)[:cal_days]
        m = Ridge(alpha=1.0).fit(E_all[ridx].reshape(-1, d), R_all[ridx].ravel())
        pred[t] = m.predict(E_all[t])
    ic = [np.corrcoef(pred[t], R_day[t])[0, 1] for t in range(cal_days + 1, T)
          if np.std(pred[t]) > 1e-9]
    return pred, np.nanmean(ic)

pred, ic_ridge = oos_factor_and_ic(E_day, R_day)
print("监督 ridge 因子 OOS IC:", ic_ridge)     # ≈ 0.017 (1.7%)
```

监督映射把 OOS IC 从 0.2% 拉到 **约 1.7%**。1.7% 看着不大，但这是单因子、零额外信息、纯文本来源的数字——而且它证明了一件事：**嵌入里确实存在可被监督学习提取的收益方向，只是它不在无监督主轴里**。

### 2.3 检索式用法（带滚动锚，避免前视）

监督式之外，检索式也有用武之地，关键是锚必须只来自**历史**。下面用滚动窗口建「利好公告嵌入库」，只对历史涨幅打分，绝不混入未来：

```python
# 用截至 t-1 日的样本建锚，计算当日新公告与历史大涨公告的相似度
def retrieval_score(E_day, R_day, t, k=50):
    hist = E_day[:t].reshape(-1, d)
    hist_ret = R_day[:t].ravel()
    anchor = hist[np.argsort(hist_ret)[-k:]]            # 历史涨幅最高的 k 条嵌入
    sim = E_day[t] @ anchor.mean(0) / (np.linalg.norm(E_day[t]) * np.linalg.norm(anchor.mean(0)))
    return sim                                          # 越高 = 越像「历史利好」
# 注意：相似度本身也需 OOS 检验 IC，不能直接当信号用
```

检索式的价值在对齐「叙事模板」——当一只股票出现与历史暴涨公告高度相似的文本时，提示研究员关注；但它对收益的线性预测力通常弱于监督映射，更适合做**事件触发**而非**连续打分**。另外，嵌入模型不是越大越好：768 维的 `bge-base` 往往已足够，1536 维的模型边际信息有限却成倍增加存储与推理成本，小团队优先把算力花在「固定版本 + 缓存 + 严格 OOS」上，而非追新模型。

## 三、多空回测（样本外、零成本假设）

把 OOS 预测值每天横截面排序，做多前 10%、做空后 10%，每日再平衡：

```python
ls = []
for t in range(160, T):
    f = pred[t]
    if np.all(np.isnan(f)):
        continue
    order = np.argsort(f)
    long_r = np.mean(R_day[t, order[-int(N * 0.1):]])
    short_r = np.mean(R_day[t, order[:int(N * 0.1)]])
    ls.append(long_r - short_r)
ls = np.array(ls)
cum = np.cumprod(1 + ls / 100)
ann = cum[-1] ** (252 / len(ls)) - 1
sharpe = ls.mean() / ls.std() * np.sqrt(252)
print(f"多空年化(零成本): {ann*100:.0f}%, Sharpe: {sharpe:.1f}")
```

在我们的设定下得到**多空年化约 21%、Sharpe 约 6.0**。

![嵌入因子多空净值与 IC 对比](/images/llm-embedding-text-factor/factor_ic_and_ls.png)

**务必诚实看待这个数字**：它是 (1) 零交易成本、(2) 零冲击、(3) 单因子线性 Gaussian 设定、(4) 已知信号只埋在 8 个维度里的理想结果。IC 只有 1.7% 却换来 21% 年化，是因为多空吃的是两端尾部价差、且在线性设定下被放大；真实文本嵌入是非线性、噪声更大的，扣掉手续费与冲击后净值会大幅缩水。**本文重点是方法而非这个漂亮数字**——它证明了"嵌入→监督因子"这条路在原理上成立。

## 四、工程落地 checklist

1. **固定模型版本 + 缓存嵌入**：嵌入是"模型版本的函数"，升级模型必须重算全部历史嵌入并做可比性检验。
2. **降维**：384/768 维直接进模型在小样本上必过拟合。先用 PCA 压到 32~64 维，或接一个轻量编码器。
3. **正则 + 时间切分**：岭回归 / LightGBM + 严格的时间序列验证，绝不做随机 train/test split。
4. **截面标准化 + 行业/市值中性化**：嵌入因子常隐含风格暴露，回归取残差再入组合。
5. **增量更新**：新公告只编码新增文本，别每天全量重算。
6. **与量价因子合并**：嵌入因子的价值往往在"补全量价看不到的叙事"，而非单独跑出高夏普。

## 五、真实陷阱（按危害排序）

1. **前视偏差（最致命）**：用 T 日收盘后发布的公告预测 T 日收益，或检索锚里混入了未来涨幅。文本时间戳必须与预测目标严格错位。
2. **嵌入漂移**：同一个词在不同模型/不同版本下的向量不同，跨版本拼接历史会制造假因子。模型一旦升级，历史全部作废重算。
3. **语义 ≠ 预测**：嵌入抓住的是"文本相似"，不是"收益方向"。两个都聊"减产"的公告，一个利好一个利空，嵌入可能给它们相似打分。
4. **维度灾难与过拟合**：384 维对 250×500 的样本量而言极度过参数化，不做正则和时间切分会得到虚假高 IC。
5. **容量与合规**：嵌入因子一旦被很多人用同一模型生成，会趋同、拥挤；另涉及文本数据版权与合规使用边界。

## 结语

LLM 嵌入不是"点石成金的 alpha"，而是把非结构化文本接进截面模型的一根更稳的桥——它保留语序与上下文，胜过词典法和 TF-IDF。但它不是无监督就能用的：信号藏在需要收益监督才能对齐的方向里，且被前视、漂移、过拟合三座大山包围。把它当"特征工程升级"而非"预测神器"，配合严格 OOS 检验和成本建模，才是正路。
