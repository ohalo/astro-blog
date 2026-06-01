---
title: "量化仓位管理：凯利公式、固定比例与波动率调整"
publishDate: '2026-06-01'
description: "量化仓位管理：凯利公式、固定比例与波动率调整 - halo的技术博客"
tags:
  - 量化交易
language: Chinese
---

# 量化仓位管理：凯利公式、固定比例与波动率调整

很多人专注于策略研发，却忽视了**仓位管理**这一决定生死的关键环节。再好的策略，仓位管理不当也会爆仓；平庸的策略，科学的仓位管理也能稳健盈利。

## 为什么仓位管理比策略更重要？

### 数学本质：盈亏不对称

- 亏损50%需要盈利100%才能回本
- 亏损90%需要盈利900%才能回本
- **仓位管理的核心是控制最大回撤**

```python
# 计算回本所需收益率
def recovery_rate(loss_rate):
    """计算亏损后回本所需收益率"""
    remaining = 1 - loss_rate
    required_return = 1 / remaining - 1
    return required_return

print(f"亏损10%后回本需要: {recovery_rate(0.1)*100:.1f}%")
print(f"亏损50%后回本需要: {recovery_rate(0.5)*100:.1f}%")
print(f"亏损90%后回本需要: {recovery_rate(0.9)*100:.1f}%")
# 输出:
# 亏损10%后回本需要: 11.1%
# 亏损50%后回本需要: 100.0%
# 亏损90%后回本需要: 900.0%
```

## 经典仓位管理方法

### 1. 固定比例法（Fixed Fractional）

每次用固定比例的资金交易（如2%）。

**优点**：简单，风险可控  
**缺点**：未考虑策略胜率和赔率

```python
def fixed_fractional(position_size, capital, risk_fraction=0.02):
    """固定比例仓位管理"""
    risk_amount = capital * risk_fraction
    shares = risk_amount / position_size['stop_loss_distance']
    return shares

# 示例
capital = 1000000  # 100万
risk_fraction = 0.02  # 2%风险
stop_loss_distance = 5  # 止损距离5元

position = fixed_fractional(
    {'stop_loss_distance': stop_loss_distance},
    capital,
    risk_fraction
)
print(f"应买入股数: {position:.0f}")
print(f"占用资金: {position * 100:.0f}元")  # 假设股价100元
```

### 2. 凯利公式（Kelly Criterion）

最优仓位公式：  
**f* = (p × b - q) / b**

其中：
- f* = 最优仓位比例
- p = 胜率
- q = 败率 (1-p)
- b = 盈亏比（平均盈利/平均亏损）

```python
def kelly_criterion(win_rate, win_loss_ratio):
    """计算凯利公式最优仓位"""
    lose_rate = 1 - win_rate
    kelly = (win_rate * win_loss_ratio - lose_rate) / win_loss_ratio
    return max(0, kelly)  # 凯利可能为负，此时不应下注

# 示例策略
win_rate = 0.55  # 55%胜率
avg_win = 1200
avg_loss = 800
win_loss_ratio = avg_win / avg_loss

kelly = kelly_criterion(win_rate, win_loss_ratio)
print(f"凯利公式建议仓位: {kelly*100:.1f}%")
print(f"半凯利（更保守）: {kelly*50:.1f}%")
```

**凯利公式的问题**：
- 假设胜率和赔率固定（现实中会变化）
- 全凯利会导致巨大波动（可能90%回撤）
- 实战中常用**半凯利**或**四分之一凯利**

### 3. 波动率调整法（Volatility Targeting）

根据市场波动率动态调整仓位。

**逻辑**：高波动时降低仓位，低波动时增加仓位

```python
import numpy as np
import pandas as pd

def volatility_targeting(position_value, volatility, target_vol=0.10):
    """
    波动率目标仓位管理
    position_value: 当前持仓市值
    volatility: 当前市场波动率（年化）
    target_vol: 目标波动率（默认10%）
    """
    if volatility == 0:
        return position_value
    
    # 调整系数
    adjustment = target_vol / volatility
    
    # 限制调整范围（0.5到2倍）
    adjustment = np.clip(adjustment, 0.5, 2.0)
    
    new_position = position_value * adjustment
    return new_position

# 示例
current_vol = 0.20  # 当前波动率20%
target_vol = 0.10   # 目标波动率10%
current_position = 500000  # 当前持仓50万

new_position = volatility_targeting(current_position, current_vol, target_vol)
print(f"当前波动率: {current_vol*100:.1f}%")
print(f"建议调整仓位: {new_position/10000:.1f}万 ({(new_position/current_position - 1)*100:+.1f}%)")
```

## 实战：多策略组合仓位管理

当有多个策略同时运行时，需要**组合仓位管理**：

```python
class PortfolioPositionManager:
    """组合仓位管理器"""
    
    def __init__(self, total_capital, max_portfolio_risk=0.20):
        self.total_capital = total_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.strategies = {}
        
    def add_strategy(self, name, win_rate, avg_win, avg_loss, correlation=0):
        """添加策略"""
        kelly = kelly_criterion(win_rate, avg_win/avg_loss)
        self.strategies[name] = {
            'kelly': kelly,
            'win_rate': win_rate,
            'correlation': correlation
        }
    
    def calculate_positions(self):
        """计算各策略仓位（考虑相关性）"""
        positions = {}
        total_kelly = sum(s['kelly'] for s in self.strategies.values())
        
        for name, strategy in self.strategies.items():
            # 基础仓位（按凯利比例分配）
            base_position = strategy['kelly'] / total_kelly
            
            # 相关性调整（高相关性降低仓位）
            correlation_penalty = 1 - strategy['correlation'] * 0.5
            
            # 最终仓位
            final_position = base_position * correlation_penalty
            positions[name] = final_position
            
        # 归一化，确保总仓位不超过最大风险
        total_position = sum(positions.values())
        if total_position > self.max_portfolio_risk:
            scale = self.max_portfolio_risk / total_position
            positions = {k: v * scale for k, v in positions.items()}
        
        return positions

# 使用示例
pm = PortfolioPositionManager(total_capital=1000000, max_portfolio_risk=0.25)

pm.add_strategy('均线策略', win_rate=0.52, avg_win=800, avg_loss=600, correlation=0.3)
pm.add_strategy('动量策略', win_rate=0.48, avg_win=1200, avg_loss=800, correlation=0.5)
pm.add_strategy('均值回归', win_rate=0.55, avg_win=600, avg_loss=500, correlation=0.2)

positions = pm.calculate_positions()
for name, pos in positions.items():
    print(f"{name}: {pos*100:.1f}% 仓位")
```

## 动态仓位调整策略

### 1. 移动平均仓位（Moving Average Position Sizing）

根据策略近期表现动态调整仓位。

```python
def moving_average_position_size(recent_returns, lookback=20, base_size=0.10):
    """基于近期表现调整仓位"""
    if len(recent_returns) < lookback:
        return base_size
    
    # 计算近期胜率
    recent = recent_returns[-lookback:]
    win_rate = sum(1 for r in recent if r > 0) / len(recent)
    
    # 根据胜率调整仓位
    if win_rate > 0.6:
        return base_size * 1.5  # 表现好加仓
    elif win_rate < 0.4:
        return base_size * 0.5  # 表现差减仓
    else:
        return base_size

# 示例
returns = [0.02, -0.01, 0.03, 0.01, -0.02, 0.015, 0.025, -0.005, 0.01, 0.02]
position_size = moving_average_position_size(returns, lookback=5, base_size=0.10)
print(f"建议仓位: {position_size*100:.1f}%")
```

### 2. 最大回撤控制法

根据当前回撤程度调整仓位。

```python
def drawdown_based_position(current_drawdown, max_allowed_drawdown=0.20, base_size=0.10):
    """基于回撤调整仓位"""
    if current_drawdown >= max_allowed_drawdown:
        return 0  # 达到最大回撤，清仓
    
    # 回撤越大，仓位越小
    drawdown_ratio = current_drawdown / max_allowed_drawdown
    position_size = base_size * (1 - drawdown_ratio)
    
    return max(0, position_size)

# 示例
current_dd = 0.15  # 当前回撤15%
max_dd = 0.20      # 最大允许回撤20%

position = drawdown_based_position(current_dd, max_dd, base_size=0.10)
print(f"当前回撤: {current_dd*100:.1f}%")
print(f"建议仓位: {position*100:.1f}%")
```

## 风控规则集成

### 单笔止损 + 日内止损 + 最大回撤止损

```python
class RiskManager:
    """风控管理器"""
    
    def __init__(self, max_single_loss=0.02, max_daily_loss=0.05, max_drawdown=0.20):
        self.max_single_loss = max_single_loss
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        
        self.daily_pnl = 0
        self.peak_capital = 0
        self.current_capital = 0
        
    def check_single_trade(self, loss_amount, capital):
        """检查单笔止损"""
        loss_rate = loss_amount / capital
        if loss_rate > self.max_single_loss:
            return False, f"单笔亏损超限: {loss_rate*100:.1f}% > {self.max_single_loss*100:.1f}%"
        return True, "OK"
    
    def check_daily_loss(self, loss_amount):
        """检查日内止损"""
        self.daily_pnl -= loss_amount
        if abs(self.daily_pnl) > self.max_daily_loss * self.peak_capital:
            return False, f"日内亏损超限: {abs(self.daily_pnl)/self.peak_capital*100:.1f}%"
        return True, "OK"
    
    def check_drawdown(self, current_capital):
        """检查最大回撤"""
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        if drawdown > self.max_drawdown:
            return False, f"最大回撤超限: {drawdown*100:.1f}%"
        return True, "OK"

# 使用示例
rm = RiskManager(max_single_loss=0.02, max_daily_loss=0.05, max_drawdown=0.20)

# 检查交易
can_trade, msg = rm.check_single_trade(loss_amount=3000, capital=100000)
print(f"单笔交易检查: {msg}")

# 检查日内亏损
can_trade, msg = rm.check_daily_loss(loss_amount=4000)
print(f"日内亏损检查: {msg}")
```

## Python实战：完整仓位管理系统

```python
import pandas as pd
import numpy as np

class PositionSizingSystem:
    """完整仓位管理系统"""
    
    def __init__(self, capital, method='kelly', risk_per_trade=0.02):
        self.capital = capital
        self.method = method
        self.risk_per_trade = risk_per_trade
        self.trades = []
        
    def calculate_position(self, entry_price, stop_loss, atr=None, volatility=None):
        """计算仓位"""
        if self.method == 'fixed':
            return self._fixed_position(entry_price, stop_loss)
        elif self.method == 'kelly':
            return self._kelly_position(entry_price, stop_loss)
        elif self.method == 'volatility':
            return self._volatility_position(entry_price, stop_loss, volatility)
        elif self.method == 'atr':
            return self._atr_position(entry_price, atr)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _fixed_position(self, entry, stop):
        """固定比例法"""
        risk_amount = self.capital * self.risk_per_trade
        price_risk = abs(entry - stop)
        shares = risk_amount / price_risk
        return int(shares)
    
    def _kelly_position(self, entry, stop):
        """凯利公式法（需要历史交易记录）"""
        if len(self.trades) < 30:
            return self._fixed_position(entry, stop)  # 数据不足用固定比例
        
        wins = [t for t in self.trades if t > 0]
        losses = [t for t in self.trades if t < 0]
        
        win_rate = len(wins) / len(self.trades)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        
        b = avg_win / avg_loss if avg_loss > 0 else 1
        kelly = (win_rate * b - (1 - win_rate)) / b
        
        # 使用半凯利
        kelly = max(0, kelly) * 0.5
        
        risk_amount = self.capital * kelly
        price_risk = abs(entry - stop)
        shares = risk_amount / price_risk
        return int(shares)
    
    def record_trade(self, pnl):
        """记录交易结果"""
        self.trades.append(pnl)
        if len(self.trades) > 100:  # 只保留最近100笔
            self.trades.pop(0)

# 使用示例
pos_system = PositionSizingSystem(capital=1000000, method='kelly')

# 计算仓位
entry_price = 100
stop_loss = 95
position = pos_system.calculate_position(entry_price, stop_loss)
print(f"建议买入: {position} 股")
print(f"占用资金: {position * entry_price} 元")
print(f"风险暴露: {position * (entry_price - stop_loss)} 元")
```

## 仓位管理常见陷阱

### 1. 马丁格尔陷阱（Martingale）

亏钱后加倍下注，最终必然爆仓！

```python
# 错误示范：马丁格尔策略
def martingale_strategy(loss_streak, base_bet=100):
    """马丁格尔：亏钱后加倍"""
    return base_bet * (2 ** loss_streak)

# 连续亏10次后，下注 = 100 * 2^10 = 102,400
# 再亏一次就归零
```

### 2. 满仓进出

A股T+1制度下，满仓意味着丧失灵活性。

### 3. 忽视相关性

多个"独立"策略可能高度相关，导致隐性满仓。

```python
# 检查策略相关性
def check_strategy_correlation(returns_dict):
    """检查多策略收益相关性"""
    df = pd.DataFrame(returns_dict)
    corr_matrix = df.corr()
    
    print("策略相关性矩阵:")
    print(corr_matrix)
    
    # 高相关性警告
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if corr > 0.7:
                print(f"⚠️ {corr_matrix.columns[i]} 与 {corr_matrix.columns[j]} 相关性过高: {corr:.2f}")
```

## 实战建议

1. **从固定比例开始**：新手用2%风险比例，简单有效
2. **逐步引入凯利**：有30笔以上交易记录后再用凯利公式
3. **波动率调整**：高波动市场（如暴跌）自动降仓
4. **多策略分散**：相关性低的策略可以加总，但不能超过总风险上限
5. **定期回顾**：每月检查仓位管理效果，调整参数

## 总结

仓位管理是量化交易的"安全带"：
- **固定比例**：简单实用，适合新手
- **凯利公式**：理论最优，但需保守使用（半凯利）
- **波动率调整**：适应市场环境变化
- **组合管理**：多策略需考虑相关性

记住：**活着比盈利更重要**。控制好仓位，市场在，机会就在。

---

*下期预告：另类数据在量化中的应用——卫星图像、社交媒体与信用卡数据*

![仓位管理示意图](/images/2026-06-01-position-sizing-kelly/position-sizing.jpg)

![凯利公式仓位曲线](/images/2026-06-01-position-sizing-kelly/kelly-curve.jpg)
