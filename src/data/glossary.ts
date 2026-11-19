/**
 * 量化术语表 —— 文章正文悬浮卡（GlossaryTooltip）数据源
 *
 * - terms：正文匹配别名（中英混排，最长优先匹配，拉丁词做单词边界校验，大小写不敏感）
 * - zh / en：定义文案（≤120 字），按页面语言显示
 * - 只包裹全文第一个命中的别名；代码块 / 链接 / 标题 / 公式内不生效
 * - 新增术语直接往数组里加，文章页零改动
 */
export interface GlossaryTerm {
  id: string
  terms: string[]
  zh: string
  en: string
}

export const glossaryTerms: GlossaryTerm[] = [
  // ── 时间序列与统计推断 ──
  {
    id: 'adf',
    terms: ['ADF 检验', 'ADF检验', 'ADF test', 'Augmented Dickey-Fuller', 'ADF'],
    zh: '增广迪基-富勒检验：回归序列是否含单位根的单边检验，原假设是「存在单位根（非平稳）」。p 值小于显著性水平才拒绝、判定为平稳。只认「拒绝」，不能证明平稳。',
    en: 'Augmented Dickey-Fuller test: a one-sided unit-root test. The null hypothesis is a unit root (non-stationarity); only a small p-value lets you reject it. Failing to reject is not evidence of stationarity.'
  },
  {
    id: 'kpss',
    terms: ['KPSS 检验', 'KPSS检验', 'KPSS test', 'KPSS'],
    zh: '与 ADF 相反的检验：原假设是「序列平稳」，拒绝才说明非平稳。两者对「弱趋势 / 强自相关」的敏感区互补，实际工作中常同时报告、看是否交叉验证。',
    en: 'KPSS test: null hypothesis is stationarity — the mirror image of ADF. The two disagree in grey zones (weak trends, strong autocorrelation), so practitioners report both.'
  },
  {
    id: 'white-noise',
    terms: ['白噪声', 'white noise'],
    zh: '均值恒定、方差恒定、各期互不相关的序列：可预测的部分已被消化完的残差。Ljung-Box 等检验常用来确认残差是否已退化为白噪声。',
    en: 'A sequence with constant mean, constant variance and zero autocorrelation — noise with nothing left to predict. Ljung-Box style tests check whether residuals have degenerated to white noise.'
  },
  {
    id: 'ljung-box',
    terms: ['Ljung-Box', 'Ljung–Box', 'LB 检验', 'Q 统计量'],
    zh: '对前 m 个自相关系数联合为零的卡方型检验，小样本下比旧式 Q 统计量修正更足。对 GARCH 残差要平方后再检验，否则测不出波动率聚集。',
    en: 'A portmanteau chi-square test that the first m autocorrelations are jointly zero, better adjusted for small samples. For GARCH residuals, apply it to squared residuals to catch volatility clustering.'
  },
  {
    id: 'garch',
    terms: ['GARCH', '广义自回归条件异方差'],
    zh: 'Bollerslev 提出的波动率模型：当期条件方差由过去的冲击平方与自身滞后线性驱动，能刻画波动率聚集与厚尾，是金融波动率建模的事实基准。',
    en: 'Bollerslev’s volatility model: conditional variance is driven linearly by squared past shocks and its own lags. Captures volatility clustering and fat tails; the de-facto baseline in finance.'
  },
  {
    id: 'egarch',
    terms: ['EGARCH', 'GJR', '杠杆效应'],
    zh: '刻画「利空比利好更能推高波动」的不对称族：EGARCH 对数形式免参数约束，GJR 用示性函数直接给负冲击加倍权重——两者都建模杠杆效应。',
    en: 'Asymmetric GARCH families modelling the leverage effect — bad news raises volatility more than good news. EGARCH logs the variance (no positivity constraints); GJR weights negative shocks higher.'
  },
  {
    id: 'kalman',
    terms: ['卡尔曼滤波', 'Kalman filter', 'Kalman Filter'],
    zh: '线性高斯状态空间的最优递推估计：用「预测—更新」两步在噪声观测中追踪不可直接观测的状态，如时变 β、配对交易价差的中枢。',
    en: 'Optimal recursive estimator for linear-Gaussian state spaces: a predict–update pair that tracks unobservable states (time-varying betas, a pairs-spread mean) from noisy observations.'
  },
  {
    id: 'cointegration',
    terms: ['协整', 'cointegration', 'Engle-Granger'],
    zh: '两个非平稳序列的某个线性组合却平稳——存在长期均衡关系。配对交易的理论前提：价差偏离均衡后倾向回归。Engle-Granger 两步法与 Johansen 检验是常用检验。',
    en: 'Two non-stationary series whose linear combination is stationary — a long-run equilibrium. The premise of pairs trading: spreads mean-revert. Tested with Engle-Granger two-step or Johansen.'
  },
  {
    id: 'pairs-trade',
    terms: ['配对交易', 'pairs trading'],
    zh: '找协整资产对，等价差偏离均值 ±2σ 左右时做空价差、回归时平仓。关键风险是协整关系断裂：止损规则与半衰期估计同样重要。',
    en: 'Trade cointegrated pairs: short the spread at roughly ±2σ deviation, close on reversion. The key risk is cointegration breakdown — stop-loss rules and half-life estimates matter as much as entry.'
  },
  {
    id: 'var-model',
    terms: ['VAR 模型', 'VAR model', '向量自回归', 'Vector Autoregression'],
    zh: '多个时间序列互为滞后解释变量的联合模型，不用先验外生结构；配合脉冲响应函数与方差分解分析变量间的动态传导。',
    en: 'Vector Autoregression: many series regressed on each other’s lags with no exogenous structure; interpreted via impulse responses and variance decomposition.'
  },
  {
    id: 'irf',
    terms: ['脉冲响应', '脉冲响应函数', 'impulse response', 'IRF'],
    zh: '给系统一个标准冲击后，各变量随后若干期的平均反应路径。局部投影（Jordà）与结构 VAR 是两条主流估计路线，前者对误设更稳健。',
    en: 'The average path of variables over the following horizons after a one-unit shock. Estimated by structural VAR or Jordà-style local projections, the latter more robust to misspecification.'
  },
  {
    id: 'local-projection',
    terms: ['局部投影', 'local projection', 'Jordà'],
    zh: 'Jordà（2005）提出的脉冲响应估计法：对每个预测期单独回归「未来值 对 冲击及其滞后」，不依赖完整模型设定；误设下更稳健，代价是估计量在长视界上效率较低。',
    en: 'Jordà (2005): estimate each forecast horizon with its own projection regression. Robust to misspecification by construction, at some efficiency cost in long horizons.'
  },
  {
    id: 'stationarity',
    terms: ['平稳性', 'stationarity'],
    zh: '序列的均值、方差与自相关结构不随时间推移改变（弱平稳）。多数计量模型的前提假设；对数价格通常不平稳、对数收益率近似平稳。',
    en: 'Constant mean, variance and autocovariance over time (weak stationarity). A prerequisite of most econometric models; log prices are typically non-stationary, log returns approximately stationary.'
  },
  {
    id: 'hmm',
    terms: ['隐马尔可夫', 'HMM', 'Hidden Markov', '状态切换模型', 'regime switching'],
    zh: '状态不可直接观测、按马尔可夫链转移、各状态生成不同观测分布的模型；量化里用于识别牛熊 / 高低波动状态切换，滤波概率是核心输出。',
    en: 'Hidden states following a Markov chain emit observations from state-dependent distributions; in quant, used to detect bull/bear or high/low volatility regimes via filtered probabilities.'
  },
  {
    id: 'cramer-rao',
    terms: ['Cramér–Rao', 'Cramér-Rao', 'CRLB', '有效信息界'],
    zh: '无偏估计量方差的理论下界：任何无偏估计都不可能比 Fisher 信息的倒数更准。评估估计器是否「有效」就是看它是否达到这条下界。',
    en: 'The lower bound on the variance of any unbiased estimator, the inverse of Fisher information. An estimator is “efficient” when it attains the CRLB.'
  },

  // ── 期权与波动率 ──
  {
    id: 'bsm',
    terms: ['Black-Scholes', 'Black–Scholes', 'BSM', 'B-S 公式'],
    zh: '几何布朗运动假设下的欧式期权解析定价公式：五个输入（价、权、期、利、波动率）定期权价；其隐含的「波动率恒定、价格连续」正是后续一切扩展的靶子。',
    en: 'The closed-form European option price under geometric Brownian motion: five inputs (spot, strike, time, rate, vol) fix the price — and its constant-vol, no-jump assumptions are exactly what later models relax.'
  },
  {
    id: 'iv',
    terms: ['隐含波动率', 'implied volatility'],
    zh: '把市场价代入 BSM 反解出的波动率，是市场对未来波动的共识定价而非预测；期限结构与偏斜（微笑）形态本身携带风险信息。',
    en: 'Volatility backed out of market price through BSM — a market consensus price of future variance, not a forecast. Its term structure and skew carry risk information.'
  },
  {
    id: 'vol-surface',
    terms: ['波动率微笑', 'volatility smile', '波动率曲面', 'volatility surface', '偏斜', 'volatility skew'],
    zh: '隐含波动率随行权价与到期日变化的曲面：股票市场典型形态是左高右低的偏斜（恐慌贵），微笑形多见于外汇。曲面形状直接对应尾概率定价。',
    en: 'The implied-vol surface over strike and maturity: equities show a downward skew (panic is expensive), FX a smile. The shape is a direct price on tail probabilities.'
  },
  {
    id: 'vrp',
    terms: ['波动率风险溢价', 'variance risk premium', 'VRP'],
    zh: '隐含方差对未来已实现方差的系统性高于部分：市场愿意为「波动险」支付超过精算公允价的保费。卖出期权长期为正、但负偏极端，赚的是这个溢价。',
    en: 'The systematic excess of implied over future realised variance: the premium paid for vol insurance. Short-vol harvesting carries a positive mean but severe negative skew.'
  },
  {
    id: 'heston',
    terms: ['Heston', '随机波动率模型', 'stochastic volatility'],
    zh: 'Bates–Heston 族随机波动率模型：方差本身服从 CIR 过程与价格相关，天然生成厚尾与波动率偏斜，可用特征函数半解析定价。',
    en: 'Heston (1993): variance follows a CIR process correlated with price, endogenously generating fat tails and the vol smile; priced semi-analytically via characteristic functions.'
  },
  {
    id: 'greeks',
    terms: ['希腊字母', 'Greeks', 'Delta 对冲', 'delta hedge'],
    zh: '期权价格对各输入的一阶 / 二阶偏导：Delta、Gamma、Vega、Theta、Rho。Delta 对冲即持续调整标的头寸使组合对小幅涨跌免疫，其盈亏由 Gamma 与 Theta 互相支付。',
    en: 'Option price sensitivities: delta, gamma, vega, theta, rho. Delta-hedging keeps the position immunised to small moves; the P&L is a flow between gamma and theta.'
  },
  {
    id: 'monte-carlo',
    terms: ['蒙特卡洛', 'Monte Carlo', '蒙特卡洛模拟'],
    zh: '按模型动态模拟大量随机路径、以路径均值折现估计期望的方法：路径依赖、奇异期权和高维问题的万能工具箱，误差以 O(1/√N) 收敛，可用方差缩减提速。',
    en: 'Simulate many paths from the model dynamics and average discounted payoffs: the universal tool for path-dependent, exotic or high-dimensional problems; error shrinks at O(1/√N), variance reduction speeds it up.'
  },

  // ── 组合、风险与资金管理 ──
  {
    id: 'kelly',
    terms: ['凯利公式', '凯利准则', 'Kelly criterion', 'Kelly'],
    zh: '最大化长期对数增长率的下注比例公式：二元赌局 g(f)=p·ln(1+f)+(1−p)·ln(1−f)，f*=p−q/b 为最优仓位。满凯利波动极大，实务常用半凯利；超额下注会显著抬高破产概率。',
    en: 'The bet size maximising long-run log growth: for a binary bet g(f)=p·ln(1+f)+(1−p)·ln(1−f), with f*=p−q/b. Full Kelly is volatile — half-Kelly is common; over-betting sharply raises the ruin probability.'
  },
  {
    id: 'sharpe',
    terms: ['夏普比率', 'Sharpe ratio'],
    zh: '超额收益均值与波动率之比：单位风险补偿的标尺。假设收益独立同分布，对厚尾、序列相关与择时的收益会系统性高估。',
    en: 'Mean excess return per unit of volatility — return per unit of risk. Assumes i.i.d. returns, so it flatters fat tails, autocorrelation and timing returns.'
  },
  {
    id: 'max-drawdown',
    terms: ['最大回撤', 'maximum drawdown', 'drawdown'],
    zh: '净值从任一历史高点到随后的最低点的最大跌幅：尾部风险的直观度量。与年化收益、波动率并列表述策略风险；小样本下极易低估真实尾部。',
    en: 'The largest peak-to-trough decline in equity: an intuitive tail-risk measure alongside return and vol, badly underestimated in small samples.'
  },
  {
    id: 'var-risk',
    terms: ['风险价值', 'Value at Risk', 'VaR'],
    zh: '给定置信度下、给定持有期内的最大预期损失，如「95% 日 VaR = 2%」。只刻画置信分位点、不刻画分位点之后的尾部损失，可用 ES/CVaR 补全。',
    en: 'The worst expected loss at a given confidence level over a horizon (e.g. 95% daily VaR = 2%). Silent about what lies beyond the quantile; use ES/CVaR to complete the tail.'
  },
  {
    id: 'cvar',
    terms: ['CVaR', 'Expected Shortfall', 'ES', '条件风险价值'],
    zh: '损失超过 VaR 之后的条件均值：比 VaR 更关注尾部的次可加一致性风险度量，常取 97.5% / 99% 分位做组合风控。',
    en: 'The average loss given the loss exceeds VaR — a coherent, tail-focused extension of VaR, commonly taken at the 97.5% / 99% quantile for risk control.'
  },
  {
    id: 'mean-variance',
    terms: ['均值-方差', 'mean-variance', 'Markowitz', '马科维茨'],
    zh: 'Markowitz 框架：用期望收益与协方差矩阵刻画组合，风险厌恶者在有效前沿上选点。对输入的期望与协方差极度敏感，实务靠收缩估计、约束与 Black-Litterman 改造。',
    en: 'Markowitz’s framework: portfolios described by mean and covariance, choosing a point on the efficient frontier. Pathologically sensitive to inputs; fixed in practice by shrinkage, constraints or Black-Litterman.'
  },

  // ── 因子与策略研究 ──
  {
    id: 'ic',
    terms: ['信息系数', 'information coefficient', 'IC'],
    zh: '选股因子截面值与下期收益的截面相关（常用 Rank IC）：因子预测力的直观指标。月频 A 股中 |IC| 常年 0.03–0.05 已属可用，ICIR 再除以 IC 的波动。',
    en: 'Cross-sectional correlation between factor values and next-period returns (usually Rank IC). On monthly A-shares, |IC| of 0.03–0.05 sustained is usable; ICIR standardises it by IC volatility.'
  },
  {
    id: 'factor-decay',
    terms: ['因子衰减', 'factor decay', 'alpha decay'],
    zh: '因子发布或拥挤后预测力下降的现象：套利资金挤入、交易成本上升与数据挖掘偏差共同作用。识别靠样本外分段 IC，应对靠容量估计与轮动。',
    en: 'The drop in a factor’s predictive power after publication or crowding — arbs pile in, costs rise, data-mining bias unwinds. Diagnosed with out-of-sample segmented IC; managed via capacity and rotation.'
  },
  {
    id: 'overfitting',
    terms: ['过拟合', 'overfitting'],
    zh: '参数拟合了历史噪声而非真实结构：样本内亮眼、样本外崩塌。多重检验、前视偏差与幸存者偏差是其常见放大器，walk-forward 与样本外验证是唯一防线。',
    en: 'Parameters that fit historical noise rather than real structure: great in-sample, broken out-of-sample. Amplified by multiple testing, look-ahead and survivor bias; walk-forward OOS is the defence.'
  },
  {
    id: 'walk-forward',
    terms: ['walk-forward', '滚动训练', '滚动样本外验证'],
    zh: '滚动式样本外验证：只用截至 t 的数据训练、对 t 后一小段数据预测，窗口整体前移。更接近实盘时序，但会吃掉数据、且对窗口长度敏感。',
    en: 'Rolling out-of-sample validation: train up to t, test on the next window, slide forward. Closer to live trading, at the cost of lost data and sensitivity to window length.'
  },
  {
    id: 'backtest',
    terms: ['回测', 'backtesting'],
    zh: '用历史数据模拟策略交易、估算业绩指标的过程。四大陷阱：前视偏差、幸存者偏差、过拟合与成本低估（冲击成本、滑点）。',
    en: 'Simulating a strategy over history to estimate performance. Four classic traps: look-ahead, survivorship, overfitting and cost underestimation (impact, slippage).'
  },

  // ── 微观结构与执行 ──
  {
    id: 'microstructure',
    terms: ['市场微观结构', 'market microstructure'],
    zh: '研究撮合机制、订单簿动态、点差形成与订单流信息含量的一级学科：连接「理论价格」与「真实成交价」的桥梁。',
    en: 'The field studying matching engines, order-book dynamics, spread formation and order-flow information — the bridge between theoretical and realised prices.'
  },
  {
    id: 'vwap',
    terms: ['VWAP', '成交量加权平均价格'],
    zh: '成交量加权平均成交价：机构衡量执行质量的基准，也是常见执行算法的跟踪目标；TWAP 则按时间等分下单。',
    en: 'Volume-weighted average price: the benchmark for execution quality and the tracking target of common execution algos; TWAP slices the parent order evenly over time.'
  },
  {
    id: 'almgren-chriss',
    terms: ['Almgren-Chriss', '最优执行'],
    zh: '最优执行框架：在「早成交承担更多冲击成本」与「晚成交承担更多价格风险」之间求效用最优的减仓轨迹，给出冲击成本—方差的折衷前沿。',
    en: 'The optimal-execution framework trading impact cost of trading fast against price risk of trading slow, producing optimal liquidation trajectories and an impact-variance efficient frontier.'
  },
  {
    id: 'market-impact',
    terms: ['冲击成本', 'market impact', '平方根定律'],
    zh: '下单推动价格逆行造成的隐性成本，随规模以约平方根或幂律增长：大单必须拆、延、藏的经验法则的量化根源。',
    en: 'The cost of your own order moving the market against you; grows sub-linearly (roughly square-root) with size — the quantitative basis of slicing, delaying and hiding large orders.'
  },

  // ── 利率与期限结构 ──
  {
    id: 'nelson-siegel',
    terms: ['Nelson-Siegel', 'NS 模型', 'Svensson', '期限结构'],
    zh: '利率期限结构的经典因子参数化：水平、斜率、曲率三个因子拟合国债收益率曲线，系数可解释、可预测，被央行与债券基金广泛用作降维工具。',
    en: 'The classic factor parametrisation of the yield curve: level, slope and curvature fit government yields with interpretable, forecastable coefficients — a dimensionality tool used by central banks and bond desks.'
  },
  {
    id: 'term-premium',
    terms: ['期限溢价', 'term premium'],
    zh: '持有长期债券相对滚动持有短期债的额外补偿：对通胀与利率风险的报酬。时变且难以观测，常由 NS 因子或 DSGE-VAR 联合估计。',
    en: 'The extra expected return of holding a long bond versus rolling short ones — compensation for inflation and rate risk. Time-varying and latent, usually estimated via latent-factor or DSGE-VAR models.'
  }
]
