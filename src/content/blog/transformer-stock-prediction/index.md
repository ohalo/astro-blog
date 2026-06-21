---
title: "Transformer模型股价预测：Attention机制在金融时序中的应用"
description: "深入探讨Transformer模型在股价预测中的应用，从Attention机制到时间序列建模，带你用PyTorch构建端到端的股价预测系统"
pubDate: 2026-06-22
tags: ["机器学习", "深度学习", "Transformer", "股价预测", "PyTorch", "量化交易"]
category: "机器学习应用"
difficulty: "🔴 高阶"
featured: true
---

# Transformer模型股价预测：Attention机制在金融时序中的应用

## 引言

在量化交易领域，时间序列预测一直是核心问题。从传统的ARIMA、GARCH模型，到机器学习方法如随机森林、LSTM，再到最近的Transformer架构，预测方法不断演进。Transformer模型最初为自然语言处理设计，但其Attention机制在处理长序列依赖方面表现出色，这使其在金融时间序列预测中具有重要价值。

本文将深入探讨Transformer模型在股价预测中的应用，从理论到实战，带你用PyTorch构建完整的股价预测系统。

## 一、Transformer模型基础

### 1.1 Attention机制原理

Attention机制的核心思想是让模型能够关注输入序列中不同位置的信息。在股价预测中，这意味着模型可以自动学习哪些历史时间点对当前预测最重要。

**Self-Attention计算公式：**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

其中：
- Q（Query）：当前时刻的查询向量
- K（Key）：历史时刻的键向量  
- V（Value）：历史时刻的值向量
- d_k：键向量的维度

**Multi-Head Attention：**

多头注意力允许模型同时关注来自不同表示子空间的信息：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # 应用mask（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attention = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attention, V)
        
        return output, attention
    
    def split_heads(self, x):
        # 将最后一维分割成 (num_heads, d_k)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        # 合并多头
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        # 线性变换
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        output = self.combine_heads(attention_output)
        
        # 输出线性变换
        output = self.W_o(output)
        
        return output, attention_weights
```

### 1.2 Positional Encoding

由于Transformer不包含递归或卷积结构，需要显式地注入位置信息。在股价预测中，时间顺序至关重要。

**正弦余弦位置编码：**

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度
        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model)
        
        # 注册为buffer（不视为模型参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        return x + self.pe[:, :x.size(1), :]
```

**可学习位置编码：**

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, d_model) * 0.02)
        
    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]
```

## 二、Transformer股价预测模型构建

### 2.1 数据预处理

股价预测需要构造合适的输入特征。常用的特征包括：

- 价格特征：开盘价、最高价、最低价、收盘价
- 成交量特征：成交量、成交额
- 技术指标：MA、RSI、MACD、布林带等
- 衍生特征：收益率、波动率

**数据预处理流程：**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, df, feature_columns, target_column, seq_length=60, pred_length=5):
        """
        股价预测数据集
        
        Args:
            df: 包含股价数据的DataFrame
            feature_columns: 特征列名列表
            target_column: 目标列名（通常是收盘价或收益率）
            seq_length: 输入序列长度（使用过去seq_length天的数据）
            pred_length: 预测序列长度（预测未来pred_length天）
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # 标准化特征
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        features = df[feature_columns].values
        target = df[[target_column]].values
        
        self.features = self.feature_scaler.fit_transform(features)
        self.target = self.target_scaler.fit_transform(target)
        
    def __len__(self):
        return len(self.features) - self.seq_length - self.pred_length + 1
    
    def __getitem__(self, idx):
        # 输入序列
        X = self.features[idx:idx+self.seq_length]
        
        # 目标序列
        y_start = idx + self.seq_length
        y = self.target[y_start:y_start+self.pred_length]
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def inverse_transform_target(self, scaled_target):
        """将标准化的目标值转换回原始尺度"""
        return self.target_scaler.inverse_transform(scaled_target)
```

### 2.2 Transformer编码器模型

构建一个用于股价预测的Transformer编码器模型：

```python
class TransformerStockPredictor(nn.Module):
    def __init__(self, input_dim, d_model=512, num_heads=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, pred_length=5):
        super().__init__()
        
        self.d_model = d_model
        self.pred_length = pred_length
        
        # 输入投影层（将特征维度映射到d_model）
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 输入输出格式为 (batch, seq, feature)
        )
        
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层（预测未来pred_length天的数据）
        self.output_projection = nn.Linear(d_model, pred_length)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_dim)
            
        Returns:
            output: (batch_size, pred_length)
        """
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # 使用最后一个时间步的输出进行预测
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # 输出投影
        output = self.output_projection(x)  # (batch_size, pred_length)
        
        return output
```

### 2.3 带解码器的Transformer模型

对于更复杂的序列预测任务，可以使用编码器-解码器结构：

```python
class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, pred_length=5):
        super().__init__()
        
        self.d_model = d_model
        self.pred_length = pred_length
        
        # 编码器部分
        self.encoder_input_projection = nn.Linear(input_dim, d_model)
        self.encoder_positional_encoding = PositionalEncoding(d_model)
        self.encoder_dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, 
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # 解码器部分
        self.decoder_input_projection = nn.Linear(1, d_model)  # 预测时输入为标量
        self.decoder_positional_encoding = PositionalEncoding(d_model)
        self.decoder_dropout = nn.Dropout(dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, src, tgt):
        """
        Args:
            src: (batch_size, src_seq_length, input_dim) 源序列（历史数据）
            tgt: (batch_size, tgt_seq_length, 1) 目标序列（预测数据，训练时为空或部分的）
        """
        # 编码器前向传播
        src = self.encoder_input_projection(src)
        src = self.encoder_positional_encoding(src)
        src = self.encoder_dropout(src)
        memory = self.transformer_encoder(src)  # (batch_size, src_seq_length, d_model)
        
        # 解码器前向传播
        tgt = self.decoder_input_projection(tgt)
        tgt = self.decoder_positional_encoding(tgt)
        tgt = self.decoder_dropout(tgt)
        
        # 生成后续掩码（防止解码器看到未来信息）
        tgt_seq_length = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_length).to(tgt.device)
        
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # 输出投影
        output = self.output_projection(output)  # (batch_size, tgt_seq_length, 1)
        
        return output.squeeze(-1)  # (batch_size, tgt_seq_length)
```

## 三、模型训练与评估

### 3.1 训练流程

```python
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

def train_transformer_model(model, train_loader, val_loader, device, 
                           num_epochs=100, learning_rate=0.001):
    """
    训练Transformer模型
    
    Args:
        model: Transformer模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备（CPU/GPU）
        num_epochs: 训练轮数
        learning_rate: 学习率
    """
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('transformer_training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return train_losses, val_losses
```

### 3.2 模型评估指标

在股价预测中，常用的评估指标包括：

- **MSE（均方误差）**：衡量预测值与真实值的平方误差
- **RMSE（均方根误差）**：MSE的平方根，与原数据同量纲
- **MAE（平均绝对误差）**：衡量预测值与真实值的平均绝对差异
- **方向准确率**：预测涨跌方向的正确率
- **信息系数（IC）**：预测值与真实值的相关系数

**评估代码实现：**

```python
def evaluate_model(model, test_loader, device, dataset):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 评估设备
        dataset: 数据集对象（用于反标准化）
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            
            # 反标准化
            output_inv = dataset.inverse_transform_target(output.cpu().numpy())
            y_inv = dataset.inverse_transform_target(y.cpu().numpy())
            
            predictions.append(output_inv)
            actuals.append(y_inv)
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # 计算评估指标
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # 方向准确率
    pred_direction = np.sign(predictions[:, 0])  # 预测第一天方向
    actual_direction = np.sign(actuals[:, 0])    # 实际第一天方向
    direction_accuracy = np.mean(pred_direction == actual_direction)
    
    # 信息系数（IC）
    ic = np.corrcoef(predictions.flatten(), actuals.flatten())[0, 1]
    
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Direction Accuracy: {direction_accuracy:.4f}")
    print(f"Information Coefficient (IC): {ic:.4f}")
    
    # 绘制预测vs实际图
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:, 0], label='Actual', alpha=0.7)
    plt.plot(predictions[:, 0], label='Predicted', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('Return')
    plt.title('Transformer Model: Predicted vs Actual')
    plt.legend()
    plt.grid(True)
    plt.savefig('transformer_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'direction_accuracy': direction_accuracy,
        'ic': ic,
        'predictions': predictions,
        'actuals': actuals
    }
```

## 四、实战案例：沪深300指数预测

### 4.1 数据准备

```python
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

# 设置tushare pro API token
ts.set_token('your_tushare_token')
pro = ts.pro_api()

def load_stock_data(stock_code='000300.SH', start_date='20180101', end_date='20231231'):
    """
    加载股票数据
    
    Args:
        stock_code: 股票代码（沪深300指数代码：000300.SH）
        start_date: 开始日期
        end_date: 结束日期
    """
    # 获取日线数据
    df = pro.index_daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
    
    # 按日期升序排列
    df = df.sort_values('trade_date')
    
    # 计算技术指标
    df = calculate_technical_indicators(df)
    
    return df

def calculate_technical_indicators(df, ma_periods=[5, 10, 20, 60]):
    """计算技术指标"""
    
    # 移动平均线
    for period in ma_periods:
        df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    
    # 收益率
    df['return_1d'] = df['close'].pct_change()
    df['return_5d'] = df['close'].pct_change(periods=5)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 布林带
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # 成交量指标
    df['volume_ma5'] = df['vol'].rolling(window=5).mean()
    df['volume_ratio'] = df['vol'] / df['volume_ma5']
    
    # 删除NaN值
    df = df.dropna().reset_index(drop=True)
    
    return df

# 加载数据
df = load_stock_data()
print(f"数据加载完成，共{len(df)}条记录")
print(f"数据时间范围：{df['trade_date'].iloc[0]} 至 {df['trade_date'].iloc[-1]}")
```

### 4.2 模型训练与预测

```python
# 定义特征列
feature_columns = [
    'open', 'high', 'low', 'close', 'vol', 'amount',
    'ma5', 'ma10', 'ma20', 'ma60',
    'return_1d', 'return_5d',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
    'volume_ratio'
]

target_column = 'return_1d'  # 预测次日收益率

# 创建数据集
dataset = StockDataset(
    df, 
    feature_columns=feature_columns, 
    target_column=target_column,
    seq_length=60,  # 使用过去60天的数据
    pred_length=5   # 预测未来5天
)

# 划分训练集、验证集、测试集
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型
input_dim = len(feature_columns)
model = TransformerStockPredictor(
    input_dim=input_dim,
    d_model=256,
    num_heads=8,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.2,
    pred_length=5
)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")

train_losses, val_losses = train_transformer_model(
    model, train_loader, val_loader, device,
    num_epochs=100,
    learning_rate=0.001
)

# 评估模型
metrics = evaluate_model(model, test_loader, device, dataset)
```

## 五、模型优化与改进

### 5.1 注意力可视化

理解Transformer模型的决策过程非常重要。通过可视化注意力权重，我们可以发现模型关注的时间点。

```python
def visualize_attention(model, sample_data, feature_names, save_path='attention_visualization.png'):
    """
    可视化Transformer的注意力权重
    
    Args:
        model: 训练好的Transformer模型
        sample_data: 样本数据 (1, seq_length, input_dim)
        feature_names: 特征名称列表
    """
    model.eval()
    
    with torch.no_grad():
        # 获取注意力权重
        # 注意：这里需要修改模型以返回注意力权重
        # 为简化，这里展示概念性代码
        
        # 假设我们可以获取最后一层的注意力权重
        # attention_weights: (num_heads, seq_length, seq_length)
        
        # 可视化每个头的注意力
        num_heads = 8
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for head in range(num_heads):
            ax = axes[head]
            
            # 绘制注意力热力图
            im = ax.imshow(attention_weights[head], cmap='YlOrRd', aspect='auto')
            
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            ax.set_title(f'Head {head+1}')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('Transformer Attention Weights Visualization', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"注意力可视化已保存至：{save_path}")

# 使用示例
sample_input = torch.FloatTensor(dataset[0][0]).unsqueeze(0).to(device)
visualize_attention(model, sample_input, feature_columns)
```

### 5.2 时序分解与残差连接

股价数据通常包含趋势、季节性和噪声。在Transformer中引入时序分解可以增强模型性能。

```python
class TimeSeriesDecomposition(nn.Module):
    """时间序列分解模块"""
    def __init__(self, seq_length, d_model):
        super().__init__()
        
        # 趋势提取（使用移动平均）
        self.trend_extractor = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.Linear(seq_length, seq_length)
        )
        
        # 季节性提取（使用傅里叶变换）
        self.seasonal_extractor = nn.Sequential(
            nn.Linear(seq_length, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, d_model)
        """
        # 提取趋势
        trend = self.trend_extractor(x.transpose(1, 2)).transpose(1, 2)
        
        # 提取季节性
        seasonal = self.seasonal_extractor(x)
        
        # 残差（噪声）
        residual = x - trend - seasonal
        
        return trend, seasonal, residual

class TransformerWithDecomposition(nn.Module):
    """带时序分解的Transformer模型"""
    def __init__(self, input_dim, d_model=512, num_heads=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, pred_length=5):
        super().__init__()
        
        # 分解模块
        self.decomposition = TimeSeriesDecomposition(seq_length=60, d_model=d_model)
        
        # 分别处理趋势、季节性、残差
        self.trend_encoder = TransformerStockPredictor(input_dim, d_model//3, num_heads, num_layers, 
                                                       dim_feedforward//3, dropout, pred_length)
        self.seasonal_encoder = TransformerStockPredictor(input_dim, d_model//3, num_heads, num_layers,
                                                          dim_feedforward//3, dropout, pred_length)
        self.residual_encoder = TransformerStockPredictor(input_dim, d_model//3, num_heads, num_layers,
                                                         dim_feedforward//3, dropout, pred_length)
        
        # 融合层
        self.fusion = nn.Linear(pred_length * 3, pred_length)
        
    def forward(self, x):
        # 分解
        trend, seasonal, residual = self.decomposition(x)
        
        # 分别编码
        trend_out = self.trend_encoder(trend)
        seasonal_out = self.seasonal_encoder(seasonal)
        residual_out = self.residual_encoder(residual)
        
        # 融合
        combined = torch.cat([trend_out, seasonal_out, residual_out], dim=-1)
        output = self.fusion(combined)
        
        return output
```

### 5.3 集成学习

单一模型可能存在过拟合风险。通过集成多个Transformer模型，可以提高预测的稳健性。

```python
class TransformerEnsemble(nn.Module):
    """Transformer集成模型"""
    def __init__(self, input_dim, num_models=5, **kwargs):
        super().__init__()
        
        # 创建多个不同的Transformer模型
        self.models = nn.ModuleList([
            TransformerStockPredictor(input_dim, **kwargs)
            for _ in range(num_models)
        ])
        
        # 每个模型使用不同的初始化
        for model in self.models:
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 获取每个模型的预测
        predictions = [model(x) for model in self.models]
        
        # 平均集成
        ensemble_prediction = torch.stack(predictions).mean(dim=0)
        
        return ensemble_prediction
    
    def predict_with_uncertainty(self, x, num_samples=100):
        """
        带不确定性的预测（使用MC Dropout）
        
        Args:
            x: 输入数据
            num_samples: 采样次数
            
        Returns:
            mean: 平均预测
            std: 预测标准差（衡量不确定性）
        """
        self.train()  # 启用Dropout
        
        predictions = []
        for _ in range(num_samples):
            pred = self.forward(x)
            predictions.append(pred.detach())
        
        predictions = torch.stack(predictions)
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        self.eval()  # 恢复评估模式
        
        return mean, std
```

## 六、实战策略构建

### 6.1 基于预测的择时策略

有了股价预测模型，我们可以构建量化择时策略：

```python
class TransformerTimingStrategy:
    """基于Transformer预测的择时策略"""
    
    def __init__(self, model, dataset, threshold=0.001, holding_period=5):
        """
        Args:
            model: 训练好的Transformer模型
            dataset: 数据集对象
            threshold: 预测收益率阈值（超过此值才交易）
            holding_period: 持仓周期（天）
        """
        self.model = model
        self.dataset = dataset
        self.threshold = threshold
        self.holding_period = holding_period
        
    def generate_signals(self, test_df):
        """
        生成交易信号
        
        Args:
            test_df: 测试集数据
            
        Returns:
            signals: 交易信号（1：买入，-1：卖出，0：持有）
            predictions: 模型预测值
        """
        self.model.eval()
        
        signals = []
        predictions = []
        
        with torch.no_grad():
            for i in range(len(test_df) - self.dataset.seq_length):
                # 准备输入数据
                X = test_df.iloc[i:i+self.dataset.seq_length][self.dataset.feature_columns].values
                X = self.dataset.feature_scaler.transform(X)
                X = torch.FloatTensor(X).unsqueeze(0).to(next(self.model.parameters()).device)
                
                # 预测
                pred = self.model(X)
                pred = pred.cpu().numpy()
                
                # 反标准化
                pred = self.dataset.inverse_transform_target(pred)
                predictions.append(pred[0, 0])  # 预测第一天收益率
                
                # 生成信号
                if pred[0, 0] > self.threshold:
                    signals.append(1)  # 买入信号
                elif pred[0, 0] < -self.threshold:
                    signals.append(-1)  # 卖出信号
                else:
                    signals.append(0)  # 无信号
        
        return signals, predictions
    
    def backtest(self, test_df, initial_capital=1000000):
        """
        策略回测
        
        Args:
            test_df: 测试集数据
            initial_capital: 初始资金
            
        Returns:
            returns: 策略收益率序列
            portfolio_value: 组合价值序列
        """
        signals, predictions = self.generate_signals(test_df)
        
        # 初始化
        capital = initial_capital
        position = 0  # 持仓数量
        portfolio_value = []
        returns = []
        
        for i, signal in enumerate(signals):
            current_price = test_df.iloc[i + self.dataset.seq_length]['close']
            
            if signal == 1 and position == 0:  # 买入
                position = capital / current_price
                capital = 0
            elif signal == -1 and position > 0:  # 卖出
                capital = position * current_price
                position = 0
            
            # 计算当前组合价值
            current_value = capital + position * current_price
            portfolio_value.append(current_value)
            
            # 计算收益率
            if i > 0:
                daily_return = (current_value - portfolio_value[-2]) / portfolio_value[-2]
                returns.append(daily_return)
        
        return returns, portfolio_value
```

### 6.2 风险控制

任何量化策略都必须考虑风险控制。以下是一些关键措施：

1. **止损止盈**：设置固定的止损止盈阈值
2. **仓位管理**：根据预测置信度动态调整仓位
3. **最大回撤控制**：当回撤超过阈值时暂停交易
4. **交易成本**：考虑手续费和滑点

```python
def backtest_with_risk_control(strategy, test_df, initial_capital=1000000,
                               stop_loss=0.05, take_profit=0.10,
                               max_position=0.95, transaction_cost=0.001):
    """
    带风险控制的回测
    
    Args:
        strategy: 策略对象
        test_df: 测试数据
        initial_capital: 初始资金
        stop_loss: 止损比例
        take_profit: 止盈比例
        max_position: 最大仓位比例
        transaction_cost: 交易成本比例
    """
    signals, predictions = strategy.generate_signals(test_df)
    
    capital = initial_capital
    position = 0
    entry_price = 0
    portfolio_value = []
    trades = []
    
    for i, signal in enumerate(signals):
        current_price = test_df.iloc[i + strategy.dataset.seq_length]['close']
        
        # 风险控制检查
        if position > 0:
            # 计算当前收益率
            current_return = (current_price - entry_price) / entry_price
            
            # 止损
            if current_return < -stop_loss:
                capital = position * current_price * (1 - transaction_cost)
                trades.append({'type': 'stop_loss', 'price': current_price, 'return': current_return})
                position = 0
            
            # 止盈
            elif current_return > take_profit:
                capital = position * current_price * (1 - transaction_cost)
                trades.append({'type': 'take_profit', 'price': current_price, 'return': current_return})
                position = 0
        
        # 执行交易信号
        if signal == 1 and position == 0 and capital > 0:  # 买入
            # 仓位管理：根据预测置信度调整仓位
            confidence = abs(predictions[i]) / 0.01  # 预测收益率除以1%
            position_size = min(max_position, confidence)
            
            position = (capital * position_size) / current_price * (1 - transaction_cost)
            capital = capital * (1 - position_size)
            entry_price = current_price
            trades.append({'type': 'buy', 'price': current_price, 'confidence': predictions[i]})
            
        elif signal == -1 and position > 0:  # 卖出
            capital = position * current_price * (1 - transaction_cost)
            trades.append({'type': 'sell', 'price': current_price, 'return': (current_price - entry_price) / entry_price})
            position = 0
        
        # 记录组合价值
        current_value = capital + position * current_price
        portfolio_value.append(current_value)
    
    # 计算策略表现
    total_return = (portfolio_value[-1] - initial_capital) / initial_capital
    sharpe_ratio = calculate_sharpe_ratio(portfolio_value)
    max_drawdown = calculate_max_drawdown(portfolio_value)
    
    print(f"总收益率：{total_return:.4f}")
    print(f"夏普比率：{sharpe_ratio:.4f}")
    print(f"最大回撤：{max_drawdown:.4f}")
    print(f"交易次数：{len(trades)}")
    
    return portfolio_value, trades

def calculate_sharpe_ratio(portfolio_value, risk_free_rate=0.03):
    """计算夏普比率"""
    returns = pd.Series(portfolio_value).pct_change().dropna()
    excess_returns = returns - risk_free_rate / 252  # 假设252个交易日
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(portfolio_value):
    """计算最大回撤"""
    portfolio_series = pd.Series(portfolio_value)
    cumulative_max = portfolio_series.cummax()
    drawdown = (portfolio_series - cumulative_max) / cumulative_max
    return drawdown.min()
```

## 七、总结与展望

本文系统介绍了Transformer模型在股价预测中的应用，从Attention机制原理到PyTorch实战，构建了一个完整的股价预测系统。关键要点包括：

1. **Attention机制优势**：能够自动学习历史时间点的重要性权重，捕捉长期依赖关系
2. **位置编码必要性**：为模型注入时间顺序信息
3. **数据预处理关键**：特征工程、标准化、序列构造直接影响模型性能
4. **模型评估多维**：除了传统误差指标，还应关注方向准确率、信息系数等
5. **风险控制必须**：任何量化策略都必须有完善的风险管理措施

**未来改进方向：**

1. **多资产联合预测**：同时预测多个相关资产，捕捉联动效应
2. **高频数据应用**：将Transformer应用于分钟级或秒级数据
3. **因果注意力**：引入因果推断，提高模型可解释性
4. **在线学习**：实现模型的持续学习和自适应更新
5. **多模态融合**：结合文本（新闻、财报）、图像（K线图）等多模态数据

Transformer模型为量化交易开辟了新的可能性，但其复杂性也带来了挑战。在实际应用中，需要平衡模型复杂度与可解释性、计算成本与预测精度。希望本文能为读者在量化交易中使用Transformer模型提供有价值的参考。

---

**免责声明**：本文所有策略、代码和案例仅用于学术交流，不构成任何投资建议。量化交易涉及高风险，请谨慎决策。

## 参考文献

1. Vaswani, A., et al. (2017). "Attention is All You Need". NeurIPS.
2. Li, S., et al. (2019). "Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting". NeurIPS.
3. Lim, B., & Zohren, S. (2021). "Time-series Forecasting with Deep Learning: A Survey". Philosophical Transactions of the Royal Society A.
4. Zhang, K., et al. (2020). "Stock Price Prediction Using Attention-based Multi-Input LSTM". ICMLA.
5. Xu, H., et al. (2021). "Transformer-Based Deep Learning Model for Stock Price Prediction". IEEE Access.
