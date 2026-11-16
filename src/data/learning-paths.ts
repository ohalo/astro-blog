/**
 * 量化学习路线图数据（/learn 页面用）
 *
 * 每条路径从 src/content/blog 里手工挑选、按建议学习顺序排列。
 * posts 引用文章目录名（即文章 URL /blog/<slug> 的 slug）。
 * 找不到对应文章时页面会自动跳过并告警（见 LearningPathsContent.astro）。
 */
import { getCollection } from 'astro:content'

export interface LearningPath {
  /** localStorage 进度 key 的一部分，勿改 */
  id: string
  emoji: string
  title: string
  titleEn: string
  /** 一句话：这条线讲什么、学完能干什么 */
  blurb: string
  blurbEn: string
  posts: string[]
}

export const learningPaths: LearningPath[] = [
  {
    id: 'intro',
    emoji: '🌱',
    title: '量化入门线',
    titleEn: 'Getting Started with Quant',
    blurb: '从零理解量化交易靠什么赚钱、工具链长什么样，以及回测为什么会骗你。',
    blurbEn: 'How quant trading makes money, what the toolchain looks like, and why backtests lie to you.',
    posts: [
      '2026-06-10-stock-basics-complete',
      '2026-06-01-quant-alpha-beta',
      '2026-02-26-ema-calculation-formula',
      '2026-06-01-quant-risk-management',
      '2026-06-01-python-quant-tools',
      '2026-06-10-quant-developer-tech-stack',
      'backtest-overfitting',
      '2026-06-04-kelly-criterion-position-sizing'
    ]
  },
  {
    id: 'timeseries',
    emoji: '🧮',
    title: '时间序列与统计基础线',
    titleEn: 'Time Series & Statistical Foundations',
    blurb: '平稳性、自相关、GARCH、协整、卡尔曼——量化建模的地基，检验先行。',
    blurbEn: 'Stationarity, autocorrelation, GARCH, cointegration, Kalman filters — the bedrock of quant modeling, tests first.',
    posts: [
      'white-noise-ljung-box',
      'adf-kpss-stationarity',
      '2026-06-03-mean-reversion-validation',
      'threshold-autoregression-tar',
      'vector-autoregression-var',
      'garch-volatility-family',
      'egarch-gjr-leverage-effect',
      'pairs-trading-cointegration',
      'kalman-filter-tracking'
    ]
  },
  {
    id: 'ml',
    emoji: '🧠',
    title: '机器学习量化线',
    titleEn: 'Machine Learning for Quant',
    blurb: '从特征工程到树模型与深度学习，再到把过拟合检测做严格——别让模型自欺。',
    blurbEn: 'Feature engineering, tree models, deep nets — plus rigorous overfitting checks so your model can’t fool you.',
    posts: [
      '2026-06-02-machine-learning-quant-trading',
      '2026-06-04-feature-engineering-ml',
      '2026-06-03-xgboost-lightgbm-quant',
      '2026-06-03-random-forest-stock-selection',
      '2026-06-04-lstm-transformer-stock-prediction',
      'feature-importance-mdi-mda',
      'walk-forward-optimization',
      'combinatorial-purged-cv',
      'deflated-sharpe-ratio',
      'white-reality-check'
    ]
  },
  {
    id: 'risk',
    emoji: '🛡',
    title: '风险管理与组合优化线',
    titleEn: 'Risk Management & Portfolio Optimization',
    blurb: '均值方差、VaR/CVaR、风险平价、Black-Litterman、波动率目标——管住组合的下行。',
    blurbEn: 'Mean-variance, VaR/CVaR, risk parity, Black-Litterman, volatility targeting — cap the downside of your portfolio.',
    posts: [
      '2026-06-11-portfolio-theory-markowitz',
      '2026-06-02-var-cvar-practice',
      '2026-06-02-cvar-risk-measure',
      '2026-06-05-cvar-stress-testing',
      '2026-06-02-risk-parity-practical',
      '2026-06-03-black-litterman',
      '2026-06-02-volatility-targeting-strategy',
      'kelly-criterion-portfolio'
    ]
  },
  {
    id: 'factor',
    emoji: '📈',
    title: '因子投资线',
    titleEn: 'Factor Investing',
    blurb: '经典因子从哪来、怎么组合打分，以及最容易被忽略的：因子会衰减、会拥挤。',
    blurbEn: 'Where classic factors come from, how to combine and score them — and the catch: factors decay and crowd.',
    posts: [
      '2026-06-14-value-factor-research',
      '2026-07-26-barra-factor-model',
      '2026-06-04-factor-combination-methods',
      'lightgbm-feature-selection',
      '2026-06-03-idiosyncratic-volatility-factor',
      'amihud-illiquidity-premium',
      '2026-06-02-factor-decay',
      '2026-06-03-factor-crowding',
      '2026-06-07-factor-decay-rotation'
    ]
  },
  {
    id: 'microstructure',
    emoji: '⚡',
    title: '市场微观结构与高频线',
    titleEn: 'Market Microstructure & HFT',
    blurb: '订单簿、订单流、毒性、价差估计、做市与执行——tick 级别的市场长什么样。',
    blurbEn: 'Order books, order flow, toxicity, spread estimation, market making and execution — the tick-level market.',
    posts: [
      '2026-06-02-hft-order-book-analysis',
      '2026-06-03-market-microstructure-noise',
      'volume-clock-sampling',
      '2026-06-02-order-flow-trading',
      'vpin-toxicity',
      'corwin-schultz-spread',
      'avellaneda-stoikov-market-making',
      'algorithmic-execution-slippage-control',
      '2026-06-02-vwap-twap-execution'
    ]
  },
  {
    id: 'options',
    emoji: '🎭',
    title: '期权与波动率交易线',
    titleEn: 'Options & Volatility Trading',
    blurb: '从 Greeks 地图到波动率曲面与 VRP 收割——交易波动率本身，而不是方向。',
    blurbEn: 'From the Greeks map to the vol surface and VRP harvesting — trade volatility itself, not direction.',
    posts: [
      'greeks-hedging-map',
      'option-delta-neutral-covered-call',
      'covered-call-strategy-options',
      'option-volatility-trading',
      '2026-06-04-volatility-surface-modeling',
      'volatility-smile-svi',
      '2026-06-04-volatility-risk-premium',
      'variance-risk-premium',
      'vix-mean-reversion',
      'gamma-scalping-options'
    ]
  }
]

/** 反查：文章 slug → 所属路径（一条文章可能属于多条线） */
export function pathsForPost(slug: string): { path: LearningPath; index: number }[] {
  const hits: { path: LearningPath; index: number }[] = []
  for (const path of learningPaths) {
    const index = path.posts.indexOf(slug)
    if (index >= 0) hits.push({ path, index })
  }
  return hits
}

/** 预计阅读分钟：按去空白后的正文字数 ~400 字/分钟估算 */
export function readingMinutes(body: string | undefined): number {
  if (!body) return 1
  const text = body
    .replace(/```[\s\S]*?```/g, '')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[#>*_`~|-]/g, '')
    .replace(/\s+/g, '')
  return Math.max(1, Math.round(text.length / 400))
}

/** 汇总统计：/learn 页文案与 meta description 共用 */
export async function learningStats() {
  const byId = new Map((await getCollection('blog')).map((p) => [p.id, p]))
  const resolved = learningPaths
    .map((path) => ({ ...path, resolved: path.posts.filter((s) => byId.has(s)) }))
    .filter((p) => p.resolved.length > 0)
  const posts = resolved.reduce((n, p) => n + p.resolved.length, 0)
  let minutes = 0
  for (const p of resolved)
    for (const s of p.resolved) minutes += readingMinutes(byId.get(s)?.body)
  return { paths: resolved.length, posts, minutes, days: Math.max(1, Math.round(minutes / 45)) }
}
