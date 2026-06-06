// 量化交易专栏 - 文章难度分类与目录结构
// 按难度从易到难排序
// 自动生成于 2026-06-06T02:04:02.869Z

export interface ColumnArticle {
  slug: string;
  title: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  difficultyLabel: string;
  description: string;
  publishDate: string;
  estimatedReadTime: number; // 分钟
}

export const quantColumnArticles: ColumnArticle[] = [
  {
    slug: '2026-06-07-cointegration-pairs-trading-advanced',
    title: '协整检验与配对交易进阶：从理论到量化实战',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '协整检验与配对交易进阶：从理论到量化实战',
    publishDate: '2026-06-07',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-07-factor-decay-rotation',
    title: '因子衰减与因子轮动：量化投资中的时间维度',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '因子衰减与因子轮动：量化投资中的时间维度',
    publishDate: '2026-06-07',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-06-execution-algo',
    title: '算法交易执行优化：VWAP、TWAP与降低交易成本的实战技巧',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '算法交易执行优化',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: 'alternative-data-quant-trading',
    title: '另类数据：量化交易的秘密武器',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '另类数据：量化交易的秘密武器',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: 'behavioral-finance-quant-trading',
    title: '行为金融学在量化交易中的应用：从心理偏差到超额收益',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '行为金融学为量化交易提供了独特的视角。本文深入探讨散户心理、羊群效应、过度反应等行为偏差，并展示如何将这些心理学洞察转化为可量化的交易策略，在A股市场获取超额收益。',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: 'black-litterman-portfolio-optimization',
    title: 'Black-Litterman模型：融合市场观点与量化投资',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: 'Black-Litterman模型：融合市场观点与量化投资',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: 'factor-decay-effect',
    title: '因子衰减效应：识别、成因与应对策略',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '因子衰减效应：识别、成因与应对策略',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: 'llm-stock-prediction-expert',
    title: '大模型如何成为股市预测专家：从数据到策略的完整路径',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '大模型如何成为股市预测专家',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: 'multi-factor-stock-selection-2026',
    title: '多因子选股实战：从因子构建到组合优化的完整流程',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '多因子选股实战：从因子构建到组合优化的完整流程',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: 'option-volatility-trading-strategies',
    title: '期权波动率交易策略：从Delta中性到备兑开仓',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '期权波动率交易策略：从Delta中性到备兑开仓',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: 'pairs-trading-statistical-arbitrage-2026',
    title: '配对交易与统计套利：从协整检验到实盘执行的完整指南',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '配对交易与统计套利：从协整检验到实盘执行的完整指南',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: 'python-quant-alpha-mining',
    title: 'Python量化实战：用机器学习挖掘Alpha因子',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: 'Python量化实战：用机器学习挖掘Alpha因子',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: 'risk-parity-china-empirical',
    title: '风险平价策略在中国A股的实证与优化',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '风险平价策略在中国A股的实证与优化',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: 'statistical-arbitrage-pairs-trading',
    title: '统计套利实战：配对交易与协整分析',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '统计套利实战：配对交易与协整分析',
    publishDate: '2026-06-06',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-05-alternative-data',
    title: '另类数据在量化投资中的革命性应用：从卫星图像到社交媒体',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '另类数据在量化投资中的革命性应用：从卫星图像到社交媒体',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-05-alternative-data-quant',
    title: '另类数据：量化交易的下一个阿尔法源泉',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '另类数据：量化交易的下一个阿尔法源泉',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-05-black-litterman-portfolio-optimization',
    title: 'Black-Litterman模型实战：从先验收益到后验收益的投资组合优化',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: 'Black-Litterman模型实战：从先验收益到后验收益的投资组合优化',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-05-cvar-stress-testing',
    title: '风险管理实战：CVaR计算与压力测试方法',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '风险管理实战：CVaR计算与压力测试方法',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-05-execution-system',
    title: '实盘交易系统构建指南：从订单管理到滑点控制',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '实盘交易系统构建指南：从订单管理到滑点控制',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-05-option-volatility-trading',
    title: '期权波动率交易实战：从隐含波动率曲面到Delta中性策略',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '期权波动率交易实战：从隐含波动率曲面到Delta中性策略',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-05-option-volatility-trading-strategies',
    title: '期权波动率交易实战：跨式、宽跨式与波动率微笑的量化策略',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '期权波动率交易实战：跨式、宽跨式与波动率微笑的量化策略',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-05-random-forest-factor-synthesis',
    title: '机器学习在量化中的应用：随机森林因子合成实战',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '机器学习在量化中的应用：随机森林因子合成实战',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-05-risk-management-var',
    title: '量化风险管理：VaR、CVaR与压力测试的实战指南',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '量化风险管理：VaR、CVaR与压力测试的实战指南',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-05-risk-var-cvar',
    title: '风险价值(VaR)与条件风险价值(CVaR)在量化投资中的应用',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '风险价值(VaR)与条件风险价值(CVaR)在量化投资中的应用',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'algo-execution',
    title: '算法交易执行策略：VWAP、TWAP与POV',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '算法交易执行策略：VWAP、TWAP与POV',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'behavioral-finance-quant-strategy',
    title: '行为金融视角下的量化策略优化：利用散户心理偏差获取阿尔法',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '行为金融视角下的量化策略优化：利用散户心理偏差获取阿尔法',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'black-litterman-asset-allocation',
    title: 'Black-Litterman模型实战：超越风险平价的资产配置进阶',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: 'Black-Litterman模型实战：超越风险平价的资产配置进阶',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'cointegration-changepoint-detection',
    title: '协整关系变点检测与统计套利实战：捕捉配对交易的黄金窗口',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '协整关系变点检测与统计套利实战',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'deep-reinforcement-learning-quant-trading',
    title: '深度强化学习在量化交易中的应用：从DQN到PPO的完整实战',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '深度强化学习在量化交易中的应用：从DQN到PPO的完整实战',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'execution-systems-slippage-control',
    title: '实盘交易系统：执行算法与滑点控制',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '实盘交易系统：执行算法与滑点控制',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'factor-crowding-identification',
    title: '因子拥挤度：多因子策略的隐形风险与识别技术',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '因子拥挤度：多因子策略的隐形风险与识别技术',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'factor-decay-lifecycle',
    title: '因子衰减效应与因子投资生命周期：量化投资的残酷真相',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '因子衰减效应与因子投资生命周期',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'factor-orthogonalization',
    title: '多因子模型中的因子正交化：剔除多重共线性的关键技术',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '多因子模型中的因子正交化：剔除多重共线性的关键技术',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'factor-research-empirical-analysis',
    title: '因子研究实证分析：价值、动量、质量与低波因子的实战表现',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '因子研究实证分析：价值、动量、质量与低波因子的实战表现',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'factor-timing-strategy',
    title: '因子择时策略：动态调整因子暴露的量化方法',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '因子择时策略',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'hierarchical-risk-parity-portfolio-optimization',
    title: '层次风险平价：用机器学习重构投资组合优化',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '层次风险平价：用机器学习重构投资组合优化',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'high-frequency-trading',
    title: '高频交易中的订单流与微结构分析',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '高频交易中的订单流与微结构分析',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'kelly-criterion-portfolio',
    title: '凯利公式在量化投资中的实战应用：从赌博到投资组合的仓位管理',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '凯利公式在量化投资中的实战应用：从赌博到投资组合的仓位管理',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'limit-order-book-microstructure-analysis',
    title: '限价订单簿微观结构：高频交易的秘密武器',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '限价订单簿微观结构：高频交易的秘密武器',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'live-trading-execution-slippage',
    title: '实盘交易系统核心技术：执行算法与滑点控制',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '实盘交易系统核心技术：执行算法与滑点控制',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'lstm-quant-prediction',
    title: 'LSTM在量化预测中的实战应用：从理论到部署',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: 'LSTM在量化预测中的实战应用：从理论到部署',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'market-making-hft-inventory-management',
    title: '做市商策略与高频交易实战：库存管理与报价优化',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '做市商策略与高频交易实战：库存管理与报价优化',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'option-delta-neutral-covered-call',
    title: '期权策略实战：Delta中性与备兑开仓的深度解析',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '期权策略实战：Delta中性与备兑开仓的深度解析',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'option-volatility-trading',
    title: '期权波动率交易实战：隐含波动率曲面交易与Delta对冲',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '期权波动率交易实战：隐含波动率曲面交易与Delta对冲',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'pairs-trading-cointegration',
    title: '统计套利实战：配对交易与协整分析',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '统计套利实战：配对交易与协整分析',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'pairs-trading-statistical-arbitrage',
    title: '配对交易与统计套利：从协整到实战',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '配对交易与统计套利：从协整到实战',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'portfolio-transaction-cost-opt',
    title: '带交易成本的投资组合优化：从理论到实战',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '带交易成本的投资组合优化：从理论到实战',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'reinforcement-learning-trading',
    title: '强化学习在量化交易中的应用：从理论到实践',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '强化学习在量化交易中的应用',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'risk-management-var-cvar',
    title: '风险管理实战：VaR与CVaR的计算与应用',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '风险管理实战：VaR与CVaR的计算与应用',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'risk-parity-model',
    title: '风险平价模型实战：超越马科维茨的资产配置革命',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '风险平价模型实战：超越马科维茨的资产配置革命',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'trading-cost-analysis-optimization',
    title: '交易成本分析：量化交易中的隐性成本与优化方法',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '交易成本分析：量化交易中的隐性成本与优化方法',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: 'wavelet-multiscale-signal',
    title: '小波变换在量化交易中的多尺度信号提取实战',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '小波变换在量化交易中的多尺度信号提取实战',
    publishDate: '2026-06-05',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-alternative-data-sentiment',
    title: '另类数据：社交媒体情绪如何预测股价波动',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '另类数据：社交媒体情绪如何预测股价波动',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-backtest-overfitting',
    title: '回测过拟合：量化策略开发中的隐形陷阱与防范指南',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '回测过拟合：量化策略开发中的隐形陷阱与防范指南',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-behavioral-finance-anomalies',
    title: '行为金融学量化实战：捕捉散户情绪驱动的股价异象',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '行为金融学量化实战：捕捉散户情绪驱动的股价异象',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-behavioral-finance-quant',
    title: '行为金融学量化应用：捕捉市场非理性带来的Alpha机会',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '行为金融学量化应用：捕捉市场非理性带来的Alpha机会',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-black-litterman-practice',
    title: 'Black-Litterman模型实战：融合市场均衡与主观观点的资产配置',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: 'Black-Litterman模型实战：融合市场均衡与主观观点的资产配置',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-covered-call-options',
    title: '备兑开仓策略：用期权增强收益的稳健之道',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '备兑开仓策略：用期权增强收益的稳健之道',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-execution-cost-control',
    title: '实盘交易系统：量化策略的成本控制与执行优化',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '实盘交易系统：量化策略的成本控制与执行优化',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-factor-combination-methods',
    title: '多因子组合方法比较：加权、IC、回归与机器学习融合',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '多因子组合方法比较：加权、IC、回归与机器学习融合',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-factor-decay-effect',
    title: '因子衰减效应：量化多因子策略的时间陷阱与应对方法',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '因子衰减效应：量化多因子策略的时间陷阱与应对方法',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-feature-engineering-ml',
    title: '机器学习特征工程：构建有效的量化因子',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '机器学习特征工程',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-hft-microstructure',
    title: '高频交易微观结构：解密订单簿动力学与限价单策略',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '深入解析高频交易中的订单簿微观结构，探讨限价单策略、订单流不平衡如何预测短期价格变动',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-kelly-criterion-position-sizing',
    title: '凯利公式与仓位管理：用数学优化你的每次下注',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '凯利公式与仓位管理：用数学优化你的每次下注',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-low-volatility-premium',
    title: '低波动因子溢价效应：高风险并不等于高收益',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '低波动因子溢价效应：高风险并不等于高收益',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-lstm-transformer-stock-prediction',
    title: 'LSTM与Transformer在股价预测中的对决：量化视角',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: 'LSTM与Transformer在股价预测中的对决',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-mean-reversion-practice',
    title: '均值回归统计套利：从理论到实战的完整流程',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '均值回归统计套利：从理论到实战的完整流程',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-mean-reversion-strategy',
    title: '均值回归策略：捕捉价格偏离后的回归机会',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '均值回归策略',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-regime-switching-model',
    title: '状态切换模型：让量化策略适应不同市场环境',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '状态切换模型',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-risk-parity-strategy',
    title: '风险平价策略：超越传统资产配置的量化解决方案',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '风险平价策略',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-stat-arb-deep-dive',
    title: '统计套利深度解析：配对交易与协整分析的量化实践',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '统计套利深度解析：配对交易与协整分析的量化实践',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-tail-risk-hedging',
    title: '尾部风险对冲：极端市场中的量化防御策略',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '构建量化尾部风险对冲组合，利用期权、VIX衍生品和替代资产保护投资组合免受极端损失',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-transformer-stock-prediction',
    title: 'Transformer股价预测实战：Attention机制能否捕捉市场时序依赖？',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: 'Transformer股价预测实战：Attention机制能否捕捉市场时序依赖？',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-var-cvar-risk-management',
    title: 'VaR与CVaR：量化风险管理的双刃剑',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: 'VaR与CVaR：量化风险管理的双刃剑',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-volatility-risk-premium',
    title: '波动率风险溢价：期权卖方如何赚取隐形收益',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '波动率风险溢价',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-04-volatility-surface-modeling',
    title: '波动率曲面建模：从隐含波动率到动态对冲',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '波动率曲面建模',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-05-multi-asset-quant-strategy',
    title: '跨资产量化策略：当股票信号遇见加密货币',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '跨资产量化策略：当股票信号遇见加密货币',
    publishDate: '2026-06-04',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-alternative-data-stock-selection',
    title: '另类数据在量化选股中的应用：从卫星图像到社交情绪',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '另类数据在量化选股中的应用：从卫星图像到社交情绪',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-behavioral-finance-alpha',
    title: '行为金融学实证：散户心理偏差如何创造量化Alpha',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '行为金融学实证：散户心理偏差如何创造量化Alpha',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-black-litterman',
    title: 'Black-Litterman 模型：把主观观点融入量化配置',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: 'Black-Litterman 模型：把主观观点融入量化配置',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-delta-neutral-options',
    title: 'Delta中性策略：用期权构建市场中性组合',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: 'Delta中性策略：用期权构建市场中性组合',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-execution-slippage',
    title: '交易执行中的滑点控制：从限价单到智能路由',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '交易执行中的滑点控制：从限价单到智能路由',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-factor-crowding',
    title: '因子拥挤与拥挤崩塌：当量化策略同质化引发市场风暴',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '因子拥挤与拥挤崩塌：当量化策略同质化引发市场风暴',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-idiosyncratic-volatility-factor',
    title: '特质波动率因子：A股异象的另类解读',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '特质波动率因子：A股异象的另类解读',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-lob-microstructure',
    title: '限价订单簿微观结构：高频交易的核心',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '限价订单簿微观结构：高频交易的核心',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-lstm-stock-prediction',
    title: 'LSTM神经网络在股票价格预测中的应用与局限',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: 'LSTM神经网络在股票价格预测中的应用与局限',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-market-microstructure-noise',
    title: "市场微观结构噪声：高频数据中的\'不完美\'真相",
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: "市场微观结构噪声：高频数据中的\'不完美\'真相",
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-mean-reversion-validation',
    title: '统计套利中的均值回归验证：ADF检验、Hurst指数与协整分析实战',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '统计套利中的均值回归验证：ADF检验、Hurst指数与协整分析实战',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-random-forest-stock-selection',
    title: '随机森林在量化选股中的实战：从因子合成到组合优化',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '随机森林在量化选股中的实战：从因子合成到组合优化',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-risk-parity-strategy',
    title: '风险平价策略：平衡风险而非资本',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '风险平价策略：平衡风险而非资本',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-03-xgboost-lightgbm-quant',
    title: 'XGBoost vs LightGBM：梯度提升在量化选股中的巅峰对决',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: 'XGBoost vs LightGBM：梯度提升在量化选股中的巅峰对决',
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-06-order-flow-strategy',
    title: "订单流交易策略：读懂市场的'心跳图谱'",
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: "订单流交易策略：读懂市场的'心跳图谱'",
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-06-pairs-trading-cointegration',
    title: "配对交易协整检验：用数学锁定市场中的'隐形天平'",
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: "配对交易协整检验：用数学锁定市场中的'隐形天平'",
    publishDate: '2026-06-03',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-alternative-data-quant',
    title: '另类数据在量化投资中的应用：从卫星图像到社交情绪',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '另类数据在量化投资中的应用：从卫星图像到社交情绪',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-alternative-data-quant-trading',
    title: '另类数据革命：卫星图像、社交媒体与信用卡数据如何重塑量化投资',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '探索另类数据在量化交易中的前沿应用，从卫星图像分析原油库存到社交媒体情绪预测股价，揭秘对冲基金如何利用非传统数据源获取超额收益',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-behavioral-finance-quant',
    title: '行为金融学在量化投资中的应用：从心理偏差到量化因子',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '行为金融学在量化投资中的应用：从心理偏差到量化因子',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-cointegration-pairs-trading-cases',
    title: '协整配对交易实战：A股统计套利10个经典案例',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '协整配对交易实战：A股统计套利10个经典案例',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-execution-algorithms',
    title: '实盘交易执行算法：VWAP、TWAP与POV实战指南',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '交易执行算法实战：如何降低冲击成本',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-factor-timing-strategy',
    title: '因子择时策略：动态因子暴露的量化体系',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '因子择时策略',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-herding-effect-behavioral-finance',
    title: '羊群效应与过度反应：行为金融在量化投资中的应用',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '羊群效应与过度反应：行为金融在量化投资中的应用',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-hft-order-book-analysis',
    title: '高频交易核心：限价订单簿(LOB)深度解析与量化策略',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '限价订单簿(LOB)是高频交易的战场',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-hft-order-flow-microstructure',
    title: '高频交易核心：订单流与限价订单簿的微观结构分析',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '深入解析高频交易中的订单流分析、限价订单簿(LOB)动态与市场微观结构，揭秘机构如何通过纳秒级交易获取超额收益',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-lstm-stock-prediction',
    title: '基于LSTM的股价预测模型实战：从数据预处理到策略回测',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '基于LSTM的股价预测模型实战：从数据预处理到策略回测',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-machine-learning-quant-trading',
    title: '机器学习在量化交易中的应用：从LSTM到集成学习',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '机器学习在量化交易中的应用：从LSTM到集成学习',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-multi-asset-stat-arb',
    title: '跨资产统计套利：股票、债券、商品的均值回归机会',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '跨资产统计套利',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-option-delta-neutral-covered-call',
    title: '期权策略进阶：Delta中性与备兑开仓实战',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '期权策略进阶：Delta中性与备兑开仓实战',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-options-volatility-trading',
    title: '期权波动率交易核心策略：隐含波动率溢价提取与Delta对冲实战',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '期权波动率交易核心策略：隐含波动率溢价提取与Delta对冲实战',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-order-flow-trading',
    title: '订单流交易：解码市场微结构的量化方法',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '订单流交易：解码市场微结构的量化方法',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-pairs-trading-cointegration',
    title: '配对交易与协整：统计套利的核心逻辑',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '配对交易与协整：统计套利的核心逻辑',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-random-forest-stock-selection',
    title: '随机森林在量化选股中的实战应用',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '随机森林在量化选股中的实战应用',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-risk-parity-portfolio',
    title: '风险平价策略：超越马科维茨的投资组合革命',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '风险平价策略：超越马科维茨的投资组合革命',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-risk-parity-practical',
    title: '风险平价模型实战：超越马科维茨的配置革命',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '风险平价模型实战',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-var-cvar-practice',
    title: 'VaR与CVaR实战：量化风险管理的双剑合璧',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: 'VaR与CVaR实战：量化风险管理的双剑合璧',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-volatility-targeting-strategy',
    title: '波动率目标策略：动态风险控制实战',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '波动率目标策略',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-vwap-twap-execution',
    title: '订单执行算法实战：VWAP与TWAP如何减少滑点？',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '订单执行算法实战：VWAP与TWAP如何减少滑点？',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: 'alternative-data-quant-revolution',
    title: '另类数据在量化投资中的革命性应用',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '另类数据在量化投资中的革命性应用',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: 'backtest-overfitting',
    title: '回测过拟合：量化策略死亡的99%原因',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '回测过拟合：量化策略死亡的99%原因',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: 'black-litterman-model',
    title: 'Black-Litterman模型：融合市场均衡与投资者观点的资产配置革命',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: 'Black-Litterman模型：融合市场均衡与投资者观点的资产配置革命',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: 'execution-slippage-control',
    title: '实盘交易滑点控制与订单管理实战',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '实盘交易滑点控制与订单管理实战',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: 'lstm-quant-trading',
    title: 'LSTM神经网络在量化交易中的实践与应用',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: 'LSTM神经网络在量化交易中的实践与应用',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: 'markowitz-portfolio-optimization',
    title: '马科维茨均值方差优化实战：用Python构建高效投资组合',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '马科维茨均值方差优化实战：用Python构建高效投资组合',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: 'multi-factor-model-practical',
    title: '多因子模型实战指南：从理论到Python实现',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '多因子模型实战指南：从理论到Python实现',
    publishDate: '2026-06-02',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-01-python-quant-tools',
    title: 'Python量化工具链全景：Backtrader、Zipline、vnpy该选哪个？',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '深入对比Backtrader、Zipline、vnpy等主流Python量化框架的优缺点，帮助你根据使用场景选择合适的工具链。',
    publishDate: '2026-06-01',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-01-quant-alpha-beta',
    title: '量化交易盈利原理：深入理解阿尔法(α)与贝塔(β)',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '深入解析量化交易的两大收益来源：贝塔（市场收益）与阿尔法（超额收益），以及如何通过因子模型、统计套利和机器学习获取阿尔法。',
    publishDate: '2026-06-01',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-01-quant-risk-management',
    title: '量化交易的风险管理：止损、仓位控制与最大回撤',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '量化交易的风险管理：止损、仓位控制与最大回撤',
    publishDate: '2026-06-01',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-cvar-risk-measure',
    title: 'CVaR条件风险价值：超越VaR的尾部风险管理',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: 'CVaR条件风险价值',
    publishDate: '2026-06-01',
    estimatedReadTime: 10
  },
  {
    slug: '2026-06-02-factor-decay',
    title: '因子衰减效应：量化因子为什么会逐渐失效？',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '因子衰减效应',
    publishDate: '2026-06-01',
    estimatedReadTime: 10
  },
  {
    slug: 'multi-factor-stock-selection-guide',
    title: '多因子选股模型实战搭建：从理论到代码',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '多因子选股模型实战搭建：从理论到代码',
    publishDate: '2026-06-01',
    estimatedReadTime: 10
  },
  {
    slug: 'python-quant-toolchain-comparison',
    title: 'Python量化工具链全景指南：Backtrader vs Zipline vs vnpy',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: 'Python量化工具链全景指南：Backtrader vs Zipline vs vnpy',
    publishDate: '2026-06-01',
    estimatedReadTime: 10
  },
  {
    slug: 'quant-trading-position-sizing',
    title: '量化交易资金管理全解：凯利公式、仓位分配与风险预算',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '量化交易资金管理全解：凯利公式、仓位分配与风险预算',
    publishDate: '2026-06-01',
    estimatedReadTime: 10
  },
  {
    slug: '2026-05-24-dividend-stock-selection',
    title: '股息率选股的艺术：如何在A股市场找到真正的现金奶牛',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '股息率选股：价值投资者在A股市场寻找高股息蓝筹股的实用策略',
    publishDate: '2026-05-24',
    estimatedReadTime: 10
  },
  {
    slug: '2026-05-24-nas-backup-strategy',
    title: 'NAS数据备份策略：三层保护让你的数据安全无忧',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: 'NAS数据备份策略：个人存储时代必备的数据保护方案',
    publishDate: '2026-05-24',
    estimatedReadTime: 10
  },
  {
    slug: 'convertible-bond-grid-trading',
    title: '详解可转债打新与网格交易的组合策略',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '详解可转债打新与网格交易的组合策略，如何在2026年实现年化8-12%的低风险收益。包含实操步骤、风险控制和真实案例。',
    publishDate: '2026-05-21',
    estimatedReadTime: 10
  },
  {
    slug: 'macbook-pro-m5-preview',
    title: 'MacBook Pro M5前瞻：从供应链消息看苹果的下一代芯片策略',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '基于供应链消息和业界分析，深度前瞻MacBook Pro M5的核心升级点、性能预测和发布时间。帮你判断是现在买M4还是等M5。',
    publishDate: '2026-05-21',
    estimatedReadTime: 10
  },
  {
    slug: 'dividend-investing-underrated',
    title: '股息投资被低估了：一个被A股投资者忽视的收益来源',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: 'A股投资者普遍忽视股息收益。本文从行为金融学角度，聊聊为什么收息策略是普通人最容易执行的长期投资方式。',
    publishDate: '2026-05-19',
    estimatedReadTime: 10
  },
  {
    slug: 'sprint-planning-pm-skill',
    title: 'Anthropic 官方 PM Skill sprint-planning 技术解析：如何做迭代范围、容量与风险规划',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: 'Anthropic 官方 PM Skill sprint-planning 技术解析：如何做迭代范围、容量与风险规划',
    publishDate: '2026-04-26',
    estimatedReadTime: 10
  },
  {
    slug: 'llm-small-model-strategies',
    title: "大模型太贵？8种''大+小''组合策略，教你省下80%推理成本",
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: "大模型太贵？8种''大+小''组合策略，教你省下80%推理成本",
    publishDate: '2026-04-22',
    estimatedReadTime: 10
  }
];

// 按难度分组
export function getArticlesByDifficulty() {
  const beginner = quantColumnArticles.filter(a => a.difficulty === 'beginner');
  const intermediate = quantColumnArticles.filter(a => a.difficulty === 'intermediate');
  const advanced = quantColumnArticles.filter(a => a.difficulty === 'advanced');
  
  return {
    beginner,
    intermediate,
    advanced,
    total: quantColumnArticles.length
  };
}

// 获取总阅读时间
export function getTotalReadTime() {
  return quantColumnArticles.reduce((sum, article) => sum + article.estimatedReadTime, 0);
}
