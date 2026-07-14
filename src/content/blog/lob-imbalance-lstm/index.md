---
title: "订单簿失衡 LSTM：用深度模型预测秒级价格方向"
description: "订单簿失衡(Order Flow Imbalance)是高频价格预测里最有信息量的特征之一。本文从零构造 LOB 失衡特征，用 LSTM 捕捉其时序依赖预测秒级价格方向，并用混淆矩阵、训练曲线与含成本回测检验它的真实价值。附完整 PyTorch 风格 Python 与六类真实陷阱（高频）。"
publishDate: '2026-07-14'
tags:
  - 量化交易
  - 高频交易
  - 订单簿
  - 订单流失衡
  - LSTM
  - 深度学习
  - 微观结构
  - Python
language: Chinese
difficulty: advanced
cover: "/images/lob-imbalance-lstm/cover.png"
---

在秒级、甚至毫秒级的世界里，K 线图那套均线、MACD 几乎失效——价格的下一跳，更多取决于**订单簿此刻的供需失衡**。买盘挂单远多于卖盘时，价格有向上的压力；反之亦然。这个直觉，量化里叫做**订单流失衡（Order Flow Imbalance, OFI）**，它是高频价格预测里信息量最高的特征之一。

问题在于：失衡和价格之间不是简单的线性关系，而是有**时序记忆**——过去几秒的失衡累积、衰减、反转，共同决定下一步。这正是 LSTM（长短期记忆网络）擅长的地方。

结论先放这：**订单簿失衡确实领先秒级价格方向，LSTM 能把这种时序依赖学出来（本文合成里三分类验证准确率 65.6%，远超 33.3% 随机基线）。但残酷的现实是——交易成本会吃掉大部分微弱 alpha：不含成本时净值微亏 1.15%，含手续费+冲击后净值跌到 0.913（亏 8.7%）。** 预测得准，不等于赚得到钱，这是高频策略最容易被忽视的鸿沟。

---

## 1. 什么是订单簿失衡

限价订单簿（Limit Order Book, LOB）在每个时刻都有多档买卖挂单。以 5 档为例：

![订单簿失衡快照](/images/lob-imbalance-lstm/lob_snapshot.png)

上图中，买盘（绿）总量 2340，卖盘（红）总量 1680。定义**订单簿失衡指标**：

$$OFI = \frac{V_{bid} - V_{ask}}{V_{bid} + V_{ask}}$$

这里 OFI = (2340−1680)/(2340+1680) = **+0.164**，为正，说明买方力量占优，价格有向上压力。OFI 的取值范围是 [−1, +1]：+1 是纯买单、−1 是纯卖单、0 是完全平衡。

**更精细的变体：**

- **加权失衡**：给近端档位（第1档）更高权重，因为它们更可能成交；
- **变化率失衡**：不看存量，看 ΔV_bid − ΔV_ask 的增量，捕捉「新进订单」的方向；
- **深度加权中价（Micro-price）**：$P_{micro} = \frac{V_{ask} \cdot P_{bid} + V_{bid} \cdot P_{ask}}{V_{bid} + V_{ask}}$，用失衡加权调整中价。

本文用最基础的存量 OFI 演示，实盘中通常组合多种变体。

---

## 2. 失衡为什么能领先价格

先直观验证：失衡真的领先价格吗？

![失衡领先价格](/images/lob-imbalance-lstm/imbalance_leads_price.png)

上图上半部分是中价走势，下半部分是 OFI（绿正红负）。可以看到：**OFI 转正的区间，中价往往随后上行；OFI 转负则中价下压。** 这不是巧合——它背后有微观结构机制：

1. **信息交易者**在下单前会先在有利方向堆积挂单，失衡是他们意图的泄漏；
2. **做市商库存管理**：当一侧挂单被大量吃掉，做市商会调整报价，推动价格朝失衡方向移动；
3. **成交压力**：买盘挂单多，意味着有更多潜在买方，市价买单更容易把价格顶上去。

但注意：这种领先关系**极其短暂**（秒级到分钟级），且信噪比很低。单看 OFP 的即时值预测方向，准确率只比随机好一点点。真正的信号藏在**时序结构**里——这就是为什么要上 LSTM。

---

## 3. 特征工程与标签构造

### 3.1 输入特征

对每个时间步，构造一个特征向量：

```python
import numpy as np
import pandas as pd

def build_lob_features(lob_df, levels=5):
    """
    lob_df: 每行一个 tick, 列包含 bid_px_1..5, bid_sz_1..5, ask_px_1..5, ask_sz_1..5
    返回特征 DataFrame
    """
    feats = pd.DataFrame(index=lob_df.index)

    bid_sz = lob_df[[f"bid_sz_{i}" for i in range(1, levels+1)]]
    ask_sz = lob_df[[f"ask_sz_{i}" for i in range(1, levels+1)]]

    # 1) 多档失衡
    tot_bid, tot_ask = bid_sz.sum(axis=1), ask_sz.sum(axis=1)
    feats["ofi"] = (tot_bid - tot_ask) / (tot_bid + tot_ask + 1e-9)

    # 2) 近端(第1档)失衡
    feats["ofi_l1"] = ((bid_sz.iloc[:, 0] - ask_sz.iloc[:, 0]) /
                       (bid_sz.iloc[:, 0] + ask_sz.iloc[:, 0] + 1e-9))

    # 3) 价差(以 tick 计)
    feats["spread"] = lob_df["ask_px_1"] - lob_df["bid_px_1"]

    # 4) 中价与微价差异(micro-price 偏移)
    mid = (lob_df["ask_px_1"] + lob_df["bid_px_1"]) / 2
    micro = ((ask_sz.iloc[:, 0] * lob_df["bid_px_1"] +
              bid_sz.iloc[:, 0] * lob_df["ask_px_1"]) /
             (bid_sz.iloc[:, 0] + ask_sz.iloc[:, 0] + 1e-9))
    feats["micro_dev"] = (micro - mid) / mid

    # 5) 失衡的变化率(增量方向)
    feats["ofi_delta"] = feats["ofi"].diff().fillna(0)

    return feats.fillna(0)
```

### 3.2 标签：k 步后的价格方向

高频预测通常做**三分类**（涨/平/跌），而不是回归——因为绝对价格变动太小、噪声太大，方向比幅度更可学：

```python
def make_labels(mid_price, horizon=20, threshold=0.0002):
    """
    horizon: 向前看多少个 tick
    threshold: 判定涨跌的最小相对变动(过滤噪声)
    返回标签: +1(涨) / 0(平) / -1(跌)
    """
    future_ret = mid_price.shift(-horizon) / mid_price - 1
    label = np.zeros(len(mid_price), dtype=int)
    label[future_ret > threshold] = 1
    label[future_ret < -threshold] = -1
    # 转成 0/1/2 供分类器使用
    return label + 1  # -1/0/1 -> 0/1/2
```

**threshold 的选择很关键**：太小则大量微小波动被标成涨/跌（其实是噪声），太大则大部分样本被标成「平」，类别极度不平衡。实践中常用「未来收益的滚动标准差」动态设阈值。

---

## 4. LSTM 模型：捕捉时序依赖

LSTM 的核心是用「门控」机制记住有用的历史、遗忘无关的噪声。对 LOB 序列，我们把过去 `seq_len` 个 tick 的特征喂进去，预测下一步方向：

```python
import torch
import torch.nn as nn

class LOBLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, n_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]        # 取序列最后一步的隐状态
        return self.head(last)      # (batch, n_classes)
```

### 4.1 构造序列样本

LSTM 需要滑动窗口切出的序列：

```python
def make_sequences(features, labels, seq_len=50):
    """把逐 tick 特征切成 (N, seq_len, n_feat) 的序列"""
    X, y = [], []
    F = features.values
    L = labels
    for i in range(seq_len, len(F)):
        X.append(F[i-seq_len:i])
        y.append(L[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
```

### 4.2 训练循环（含关键防过拟合措施）

```python
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # 类别不平衡时用加权交叉熵
    crit = nn.CrossEntropyLoss()
    best_val, patience, wait = 1e9, 5, 0

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            opt.step()

        # 验证 + 早停
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += crit(model(xb), yb).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), "best_lob_lstm.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"早停于 epoch {ep}")
                break
    return model
```

![训练曲线](/images/lob-imbalance-lstm/training_curves.png)

上图左边是损失曲线：训练损失持续下降，但验证损失在 epoch 18 后开始抬头——这是**过拟合的典型信号**，早停机制在这里及时刹车。右边是准确率：验证准确率稳定在 55%~57%（这里指整体三分类），远超 33.3% 的随机基线。

---

## 5. 评估：混淆矩阵告诉你真相

准确率是个笼统的数字，**混淆矩阵**才揭示模型到底在哪类样本上有效：

![混淆矩阵](/images/lob-imbalance-lstm/confusion_matrix.png)

从上图（验证准确率 65.6%）能读出几个关键信息：

1. **对角线占优**：三类的正确率都明显高于错误率，说明模型确实学到了方向信号；
2. **「平」类召回最高**（0.67）：因为「平」样本最多，也最容易预测；
3. **涨/跌之间的误判较少**：把「涨」误判成「跌」（0.10）或反之（0.12）的比例低——**这很重要**，因为方向完全反了才是真正致命的错误。混淆多发生在「涨/跌 vs 平」之间，代价相对可控。

这种「方向不易搞反、但边界模糊」的模式，正是 LOB 失衡信号的典型特征：**它对强失衡时刻很灵敏，对弱失衡（接近平衡）时刻则模棱两可。**

---

## 6. 最关键的一步：含成本回测

预测准 ≠ 赚钱。高频策略的成败几乎完全取决于**交易成本**。用模型信号驱动一个玩具多空组合，对比含成本前后：

```python
def backtest_signal(signals, mid_price, fee_bps=0.8, impact_bps=0.0):
    """
    signals: 模型输出的方向 (-1/0/+1)
    fee_bps: 单边手续费(基点)
    返回含成本前后的净值
    """
    ret = mid_price.pct_change().shift(-1).fillna(0)  # 下一步收益
    pnl_gross = signals * ret

    # 换仓成本: 只在仓位变化时收取
    turnover = np.abs(np.diff(np.concatenate([[0], signals])))
    cost = turnover * (fee_bps + impact_bps) / 1e4
    pnl_net = pnl_gross - cost

    eq_gross = (1 + pnl_gross).cumprod()
    eq_net = (1 + pnl_net).cumprod()
    return eq_gross, eq_net
```

![含成本回测](/images/lob-imbalance-lstm/backtest_cost.png)

这张图是全文最该记住的。**不含成本（绿线）时策略只是微亏（净值 0.988）——单靠这个 OFI 信号的 edge 本就微弱；一旦加上手续费和冲击成本（红线），净值跌到 0.913，亏损扩大到 8.7%。**

原因很直接：秒级策略换手极高，每次换仓都要付 spread + 手续费 + 冲击。哪怕单次成本只有 0.8 个基点，成千上万次交易累积起来就是天文数字。**这就是「预测准确率」和「实盘盈利」之间的鸿沟。**

要跨过这道鸿沟，只有几条路：

1. **降低换手**：加入信号平滑、最小持仓时间、失衡强度门槛，只在强信号时交易；
2. **提升单次 edge**：组合多种失衡特征、拉长预测 horizon、用 micro-price 而非 mid-price 计 PnL；
3. **做市而非吃单**：用限价单赚 spread 而不是付 spread，把成本符号反转——但这引入成交不确定性和逆向选择风险。

---

## 7. 六类真实陷阱

### 陷阱 1：用未来信息构造标签（look-ahead）
标签用了未来 k 步价格，训练时必须保证特征只用到 t 时刻及之前的数据。**特征和标签的时间边界一旦错位，回测准确率会虚高到离谱**。

### 陷阱 2：训练/测试按随机切分
时序数据**绝不能随机 shuffle 后切分**！相邻 tick 高度相关，随机切分会让测试集「泄漏」进训练集。必须按时间**严格前后切分**（前 70% 训练、后 30% 测试）。

### 陷阱 3：忽略交易成本
如本文所示，成本能把一个「看起来能用」的策略直接打成亏损。**任何高频回测不含成本都是自欺欺人**。

### 陷阱 4：类别不平衡
「平」类往往占 40%~60%，模型可能退化成「永远猜平」也有不错的准确率。要用**加权交叉熵**或**动态阈值**平衡类别，并重点看涨/跌类的 F1 而非整体准确率。

### 陷阱 5：特征未做时点归一化
不同时段的挂单量级差异巨大（开盘 vs 午盘）。用全样本统计量归一化会引入 look-ahead，必须用**滚动/扩展窗口**归一化，且只用历史数据。

### 陷阱 6：把合成 edge 当真实 edge
真实 LOB 数据的信噪比远低于合成数据，且存在**订单簿操纵、幌骗（spoofing）、撤单闪烁**等对抗行为。合成里跑通的模型，接真实数据后准确率通常大幅下滑——本文的准确率数字仅用于演示机制，不代表实盘水平。

---

## 8. 从合成到实盘

本文用合成 LOB 数据（失衡驱动的中价 + 噪声）是为了让「失衡→价格」的因果机制清晰可复现。接真实数据时：

1. **数据源**：需要逐笔 tick 级的 L2 行情（多档挂单），这类数据昂贵且量巨大（单只股票一天可达 GB 级）；
2. **时钟对齐**：交易所时间戳有微秒级抖动，事件排序要极其小心；
3. **延迟建模**：从收到行情、模型推理、到订单到达交易所有几毫秒延迟，回测必须把这段延迟内的价格变动算进去；
4. **成本细化**：区分 maker/taker 费率、冲击成本随下单量非线性增长、以及最关键的**逆向选择成本**（你能成交，往往是因为对手方信息比你多）。

---

## 9. 总结

订单簿失衡 LSTM 展示了高频量化的两个真相：

- **好消息**：微观结构里确实有信息。订单簿失衡领先秒级价格方向，LSTM 能把这种时序依赖学出来，三分类准确率显著超过随机基线；
- **坏消息**：预测准 ≠ 赚钱。微弱的统计 edge，会被交易成本、延迟、逆向选择层层侵蚀。本文合成里，一个看似「学会了」的模型，含成本后依然亏损。

高频交易的真正壁垒，从来不在「能不能预测」，而在**「预测的 edge 是否大到能覆盖成本」**。这也是为什么这个领域是少数几家有极致低延迟基础设施和最优执行能力的机构的战场。

对普通量化研究者而言，本文的价值在于方法论：**先验证信号领先性（别自欺）→ 用时序模型捕捉依赖（别随机切分）→ 最后死磕成本回测（别忽略摩擦）**。三步都过了，你才算真正理解了一个高频信号的价值边界。
