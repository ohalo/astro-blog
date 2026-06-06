---
title: 强化学习交易Agent：从DQN到PPO的完整实战
publishDate: '2026-06-06'
description: 强化学习交易Agent实战 - halo的技术博客
tags:
  - 量化交易
language: Chinese
difficulty: advanced
---

## 引言

传统量化策略依赖人工设计的因子和规则，而强化学习（Reinforcement Learning, RL）让Agent能够**从市场数据中自主学习交易策略**。

近年来，深度学习与强化学习的结合（Deep RL）在游戏AI（AlphaGo、OpenAI Five）中取得巨大成功，这激发了量化研究者将RL应用于金融市场的兴趣。

本文将手把手教你构建基于PPO（Proximal Policy Optimization）算法的交易Agent，并在A股市场中验证其性能。

## 强化学习基础概念

### Markov决策过程（MDP）

交易问题可以建模为MDP：

- **状态空间 $S$**：市场观测（价格、成交量、技术指标等）
- **动作空间 $A$**：交易决策（买入、卖出、持仓）
- **状态转移 $P(s'|s,a)$**：市场从状态 $s$ 执行动作 $a$ 后转移到 $s'$ 的概率
- **奖励函数 $R(s,a)$**：交易动作带来的收益或损失

### 核心算法对比

| 算法 | 类型 | 适用场景 | 优缺点 |
|------|------|---------|--------|
| DQN | 价值-based | 离散动作 | 稳定但保守 |
| DDPG | 策略-based | 连续动作 | 训练不稳定 |
| **PPO** | 策略优化 | 连续+离散 | **稳定且高效** ✅ |

**本文选择PPO**，因其在训练稳定性和样本效率之间取得了良好平衡。

## 环境构建：量化交易gym环境

### 自定义TradingEnv

```python
import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    """A股交易环境"""
    
    def __init__(self, price_data, initial_cash=1e6, transaction_cost=0.001):
        super(TradingEnv, self).__init__()
        
        self.price_data = price_data  # DataFrame: [open, high, low, close, volume]
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost  # 双边手续费+滑点
        
        # 动作空间：[-1, 1]，表示目标仓位（归一化）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # 状态空间：过去60个交易日的收盘价+成交量+技术指标
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(60 * 5 + 10,),  # 60天*5个特征 + 10个技术指标
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_step = 60  # 从第60天开始（需要历史数据）
        self.cash = self.initial_cash
        self.position = 0  # 持仓股数
        self.portfolio_value = self.initial_cash
        
        return self._get_observation()
    
    def step(self, action):
        """
        执行交易动作
        action: [-1, 1]，目标仓位比例（-1表示满仓空头，1表示满仓多头）
        """
        # 当前价格
        current_price = self.price_data.iloc[self.current_step]['close']
        
        # 目标持仓市值
        target_value = action[0] * self.portfolio_value
        
        # 当前持仓市值
        current_value = self.position * current_price
        
        # 交易股数（A股100股为整数倍）
        trade_value = target_value - current_value
        trade_shares = int(trade_value / (current_price * 100)) * 100
        
        # 执行交易
        if trade_shares > 0:  # 买入
            cost = trade_shares * current_price * (1 + self.transaction_cost)
            if cost <= self.cash:
                self.cash -= cost
                self.position += trade_shares
        
        elif trade_shares < 0:  # 卖出
            sell_shares = min(-trade_shares, self.position)
            revenue = sell_shares * current_price * (1 - self.transaction_cost)
            self.cash += revenue
            self.position -= sell_shares
        
        # 更新组合价值
        self.portfolio_value = self.cash + self.position * current_price
        
        # 计算奖励（日收益率）
        reward = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        self.prev_portfolio_value = self.portfolio_value
        
        # 移动到下一步
        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1
        
        # 额外奖励：夏普比率（可选）
        if done:
            sharpe = self._calculate_sharpe_ratio()
            reward += sharpe * 0.1  # 加权
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """构造状态向量"""
        # 获取过去60天的数据
        start = self.current_step - 60
        end = self.current_step
        
        price_window = self.price_data.iloc[start:end]
        
        # 特征工程
        features = []
        for col in ['close', 'volume']:
            values = price_window[col].values
            normalized = (values - values.mean()) / (values.std() + 1e-8)
            features.extend(normalized)
        
        # 技术指标（示例：MACD, RSI）
        tech_indicators = self._calculate_technical_indicators(price_window)
        features.extend(tech_indicators)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_technical_indicators(self, price_window):
        """计算技术指标"""
        close = price_window['close'].values
        
        # RSI
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd = ema12 - ema26
        
        return [rsi, macd, ...]  # 返回10个技术指标
```

## PPO算法实现

### Actor-Critic网络

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class Actor(nn.Module):
    """策略网络（Actor）"""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 动作标准差
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))  # 动作均值在[-1, 1]
        std = torch.exp(self.log_std)  # 动作标准差（正值）
        
        return mean, std
    
    def sample_action(self, state):
        """采样动作"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.detach().numpy(), log_prob

class Critic(nn.Module):
    """价值网络（Critic）"""
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 输出状态价值
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        
        return value
```

### PPO训练循环

```python
class PPO:
    """PPO算法实现"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=lr
        )
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        
    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.actor.sample_action(state)
        
        return action[0], log_prob
    
    def update(self, memory):
        """PPO更新"""
        # 从memory中提取数据
        states = torch.FloatTensor(memory.states)
        actions = torch.FloatTensor(memory.actions)
        log_probs_old = torch.FloatTensor(memory.log_probs).unsqueeze(1)
        rewards = memory.rewards
        dones = memory.dones
        
        # 计算回报和优势函数
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 归一化
        
        # 计算优势函数
        state_values = self.critic(states).squeeze()
        advantages = returns - state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新（多次epoch）
        for _ in range(10):  # K_epochs
            # 重新计算新策略的log概率
            mean, std = self.actor(states)
            dist = Normal(mean, std)
            log_probs_new = dist.log_prob(actions)
            
            # 计算概率比
            ratios = torch.exp(log_probs_new - log_probs_old)
            
            # PPO裁剪目标函数
            surr1 = ratios * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.unsqueeze(1)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic损失
            critic_loss = nn.MSELoss()(self.critic(states).squeeze(), returns)
            
            # 总损失
            loss = actor_loss + 0.5 * critic_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
```

## 完整训练流程

```python
# 主训练脚本
def train_ppo_trader(stock_code='600519.SH', episodes=1000):
    """训练PPO交易Agent"""
    
    # 1. 加载数据
    price_data = load_stock_data(stock_code, start='2018-01-01', end='2025-12-31')
    
    # 2. 创建环境
    env = TradingEnv(price_data)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 3. 初始化PPO
    ppo = PPO(state_dim, action_dim)
    memory = Memory()
    
    # 4. 训练循环
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for t in range(1000):  # 每个episode最多1000步
            # 选择动作
            action, log_prob = ppo.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储到memory
            memory.states.append(state)
            memory.actions.append(action)
            memory.log_probs.append(log_prob.detach().numpy())
            memory.rewards.append(reward)
            memory.dones.append(done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # 更新PPO
        loss = ppo.update(memory)
        memory.clear()
        
        # 打印进度
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Loss: {loss:.4f}")
    
    # 5. 保存模型
    torch.save(ppo.actor.state_dict(), 'ppo_trader_actor.pth')
    torch.save(ppo.critic.state_dict(), 'ppo_trader_critic.pth')
    
    return ppo

# 运行训练
ppo_agent = train_ppo_trader()
```

## 回测与评估

### 性能指标

```python
def evaluate_agent(agent, test_data, initial_cash=1e6):
    """评估训练好的Agent"""
    env = TradingEnv(test_data, initial_cash=initial_cash)
    state = env.reset()
    done = False
    
    portfolio_values = []
    
    while not done:
        # 使用训练好的Actor选择动作（无探索）
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, _ = agent.actor(state_tensor)
            action = mean.numpy()[0]
        
        # 执行动作
        state, _, done, _ = env.step(action)
        portfolio_values.append(env.portfolio_value)
    
    # 计算指标
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    total_return = (portfolio_values[-1] - initial_cash) / initial_cash
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'portfolio_values': portfolio_values
    }

# 评估结果
test_data = load_stock_data('600519.SH', start='2024-01-01', end='2025-12-31')
results = evaluate_agent(ppo_agent, test_data)

print(f"总收益率: {results['total_return']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
```

### 实盘注意事项

1.  **过拟合风险**：RL容易在回测中过拟合，必须使用样本外数据验证
2.  **交易成本**：A股手续费+滑点约0.1-0.2%，RL必须考虑
3.  **市场环境变化**：RL策略可能在市场制度变化后失效
4.  **风控机制**：必须设置止损、仓位上限等风控规则

## 完整代码示例

```python
# main.py - 完整训练+回测脚本
import torch
import gym
import numpy as np
from trading_env import TradingEnv
from ppo import PPO

def main():
    # 配置
    STOCK_CODE = '600519.SH'
    TRAIN_START = '2018-01-01'
    TRAIN_END = '2023-12-31'
    TEST_START = '2024-01-01'
    TEST_END = '2025-12-31'
    
    # 训练
    print("开始训练PPO交易Agent...")
    ppo_agent = train_ppo_trader(
        stock_code=STOCK_CODE,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        episodes=2000
    )
    
    # 回测
    print("\n开始回测...")
    test_data = load_stock_data(STOCK_CODE, TEST_START, TEST_END)
    results = evaluate_agent(ppo_agent, test_data)
    
    # 可视化
    plot_results(results)
    
    # 保存结果
    save_results(results, filename='ppo_trader_results.json')

if __name__ == "__main__":
    main()
```

## 总结

强化学习为量化交易提供了全新的范式：

1.  ✅ **端到端学习**：从原始数据直接学习交易策略
2.  ✅ **适应性强**：能够适应市场环境变化
3.  ✅ **风险可控**：可以在奖励函数中引入风险惩罚

但也要注意：
-  ⚠️ 训练不稳定，需要大量调参
-  ⚠️ 容易过拟合，必须严格样本外测试
-  ⚠️ 实盘表现可能显著差于回测

**建议**：将RL作为传统量化策略的补充，而非完全替代。在小资金上验证效果后，再逐步扩大规模。

---

**参考资料**
- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.
- Deng, Y., et al. (2016). "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading." *IEEE Transactions on Neural Networks and Learning Systems*.
- 张伟, 李明 (2024). "基于PPO算法的A股日内交易策略研究." *量化投资*.
