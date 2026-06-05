// 量化交易专栏 - 文章难度分类与目录结构
// 按难度从易到难排序

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
  // ==================== 入门级 (Beginner) ====================
  {
    slug: 'factor-research-empirical-analysis',
    title: '因子研究实证分析：价值、动量、质量与低波因子的实战表现',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '介绍四大经典因子（价值、动量、质量、低波）的基本概念和实证方法',
    publishDate: '2026-06-05',
    estimatedReadTime: 15
  },
  {
    slug: '2026-06-05-behavioral-finance-quant',
    title: '行为金融学在量化策略中的应用：捕捉散户情绪与认知偏差',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '利用行为金融学理论，将散户心理偏差转化为量化策略信号',
    publishDate: '2026-06-05',
    estimatedReadTime: 18
  },
  {
    slug: '2026-06-05-risk-management-var',
    title: '量化风险管理：VaR、CVaR与压力测试的实战指南',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '风险管理基础：VaR、CVaR计算方法与压力测试框架',
    publishDate: '2026-06-05',
    estimatedReadTime: 20
  },
  {
    slug: '2026-06-05-cvar-stress-testing',
    title: '风险管理实战：CVaR计算与压力测试方法',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '深入学习CVaR（条件风险价值）的计算方法与压力测试实战',
    publishDate: '2026-06-05',
    estimatedReadTime: 18
  },
  {
    slug: 'risk-management-var-cvar',
    title: '风险管理实战：VaR与CVaR的计算与应用',
    difficulty: 'beginner',
    difficultyLabel: '入门',
    description: '风险价值（VaR）和条件风险价值（CVaR）的计算方法与实战应用',
    publishDate: '2026-06-05',
    estimatedReadTime: 16
  },

  // ==================== 中级 (Intermediate) ====================
  {
    slug: 'pairs-trading-statistical-arbitrage',
    title: '配对交易与统计套利：从协整到实战',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '统计套利核心：协整检验、配对选择、交易信号生成',
    publishDate: '2026-06-05',
    estimatedReadTime: 22
  },
  {
    slug: 'pairs-trading-cointegration',
    title: '统计套利实战：配对交易与协整分析',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '协整理论基础与配对交易策略的实战实现',
    publishDate: '2026-06-05',
    estimatedReadTime: 20
  },
  {
    slug: 'cointegration-changepoint-detection',
    title: '协整关系变点检测与统计套利实战：捕捉配对交易的黄金窗口',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '进阶统计套利：检测协整关系变点，捕捉配对交易最佳时机',
    publishDate: '2026-06-05',
    estimatedReadTime: 25
  },
  {
    slug: 'factor-decay-lifecycle',
    title: '因子衰减效应与因子投资生命周期：量化投资的残酷真相',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '因子研究进阶：理解因子衰减的原因、生命周期与应对方法',
    publishDate: '2026-06-05',
    estimatedReadTime: 20
  },
  {
    slug: 'factor-crowding-identification',
    title: '因子拥挤度：多因子策略的隐形风险与识别技术',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '识别因子拥挤的方法与因子投资策略的风险管理',
    publishDate: '2026-06-05',
    estimatedReadTime: 22
  },
  {
    slug: 'risk-parity-model',
    title: '风险平价模型实战：超越马科维茨的资产配置革命',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '投资组合理论进阶：从马科维茨均值方差到风险平价模型',
    publishDate: '2026-06-05',
    estimatedReadTime: 22
  },
  {
    slug: '2026-06-05-black-litterman-portfolio-optimization',
    title: 'Black-Litterman模型实战：从先验收益到后验收益的投资组合优化',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '进阶资产配置：Black-Litterman模型结合投资者观点与市场均衡',
    publishDate: '2026-06-05',
    estimatedReadTime: 25
  },
  {
    slug: 'hierarchical-risk-parity-portfolio-optimization',
    title: '层次风险平价：用机器学习重构投资组合优化',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '机器学习在投资组合中的应用：层次聚类风险平价模型',
    publishDate: '2026-06-05',
    estimatedReadTime: 28
  },
  {
    slug: '2026-06-05-option-volatility-trading-strategies',
    title: '期权波动率交易实战：跨式、宽跨式与波动率微笑的量化策略',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '期权交易核心：波动率策略、期权组合与Greeks管理',
    publishDate: '2026-06-05',
    estimatedReadTime: 24
  },
  {
    slug: 'option-delta-neutral-covered-call',
    title: '期权策略实战：Delta中性与备兑开仓的深度解析',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '期权实战策略：Delta中性对冲与备兑开仓策略详解',
    publishDate: '2026-06-05',
    estimatedReadTime: 22
  },
  {
    slug: 'option-volatility-trading',
    title: '期权波动率交易实战：隐含波动率曲面交易与Delta对冲',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '期权波动率交易的高级技术：波动率曲面构造与交易',
    publishDate: '2026-06-05',
    estimatedReadTime: 26
  },
  {
    slug: '2026-06-05-alternative-data-quant',
    title: '另类数据：量化交易的下一个阿尔法源泉',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '数据挖掘：卫星图像、社交媒体、信用卡数据在量化中的应用',
    publishDate: '2026-06-05',
    estimatedReadTime: 20
  },
  {
    slug: 'alternative-data-quant-trading',
    title: '另类数据在量化交易中的应用：从卫星图像到社交情绪',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '另类数据源的获取、处理与在量化策略中的应用案例',
    publishDate: '2026-06-05',
    estimatedReadTime: 22
  },
  {
    slug: 'portfolio-transaction-cost-opt',
    title: '带交易成本的投资组合优化：从理论到实战',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '实战优化：将交易成本纳入投资组合构建框架',
    publishDate: '2026-06-05',
    estimatedReadTime: 23
  },
  {
    slug: 'trading-cost-analysis-optimization',
    title: '交易成本分析：量化交易中的隐性成本与优化方法',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '实战细节：交易成本分解、Market Impact模型与最优执行',
    publishDate: '2026-06-05',
    estimatedReadTime: 22
  },
  {
    slug: 'statistical-arbitrage-pairs-trading',
    title: '统计套利：配对交易与协整策略实战',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '统计套利理论基础与配对交易策略的完整实现流程',
    publishDate: '2026-06-05',
    estimatedReadTime: 24
  },
  {
    slug: 'behavioral-finance-quant-strategy',
    title: '行为金融视角下的量化策略优化：利用散户心理偏差获取阿尔法',
    difficulty: 'intermediate',
    difficultyLabel: '中级',
    description: '行为金融学进阶：如何利用散户心理偏差构建量化策略',
    publishDate: '2026-06-05',
    estimatedReadTime: 21
  },

  // ==================== 高级 (Advanced) ====================
  {
    slug: '2026-06-05-ml-quant-trading',
    title: '机器学习在量化交易中的应用：从LSTM到Transformer的完整实战指南',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: 'AI量化：LSTM、Transformer等深度学习模型在量化预测中的应用',
    publishDate: '2026-06-05',
    estimatedReadTime: 30
  },
  {
    slug: 'lstm-quant-prediction',
    title: 'LSTM在量化预测中的实战应用：从理论到部署',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '深度学习实战：LSTM模型构建、训练、验证与实盘部署',
    publishDate: '2026-06-05',
    estimatedReadTime: 28
  },
  {
    slug: '2026-06-05-random-forest-factor-synthesis',
    title: '机器学习在量化中的应用：随机森林因子合成实战',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '机器学习因子挖掘：随机森林在因子合成与非线性建模中的应用',
    publishDate: '2026-06-05',
    estimatedReadTime: 25
  },
  {
    slug: 'wavelet-multiscale-signal',
    title: '小波变换在量化交易中的多尺度信号提取实战',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '信号处理：小波变换在多尺度金融时间序列分析中的应用',
    publishDate: '2026-06-05',
    estimatedReadTime: 27
  },
  {
    slug: 'limit-order-book-microstructure-analysis',
    title: '限价订单簿微观结构：高频交易的秘密武器',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '高频交易基础：限价订单簿(LOB)数据处理与微观结构分析',
    publishDate: '2026-06-05',
    estimatedReadTime: 26
  },
  {
    slug: 'order-flow-high-freq',
    title: '订单流分析：高频交易的微观结构与交易信号',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '高频策略核心：订单流(Order Flow)分析技术与交易信号提取',
    publishDate: '2026-06-05',
    estimatedReadTime: 28
  },
  {
    slug: 'execution-systems-slippage-control',
    title: '实盘交易系统：执行算法与滑点控制',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '实盘技术：VWAP/TWAP执行算法、智能订单路由与滑点优化',
    publishDate: '2026-06-05',
    estimatedReadTime: 24
  },
  {
    slug: 'live-trading-execution-slippage',
    title: '实盘交易系统核心技术：执行算法与滑点控制',
    difficulty: 'advanced',
    difficultyLabel: '高级',
    description: '实盘交易系统深度解析：算法执行、滑点控制、订单管理',
    publishDate: '2026-06-05',
    estimatedReadTime: 26
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
