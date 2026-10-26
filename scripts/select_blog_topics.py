#!/usr/bin/env python3
"""
量化博客主题选择脚本
自动扫描已有文章,选择2个不重复的主题
"""

import os
import re
from datetime import datetime

# 博客目录 — 可移植写法：优先用环境变量，否则相对于脚本自身位置
import os as _os
BLOG_DIR = _os.environ.get("BLOG_DIR") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src", "content", "blog")
)

# 可选主题池(按领域分类)
TOPIC_POOL = {
    # ===== 2026-07-30-B 补充：全库 1250 篇，上午批次已全部发布，新增金融机器学习标注 / 信息驱动采样与风控真正全新主题（2026-07-30 中午全库磁盘扫描确认未发布）=====
    "前沿扩展2026-07-30B-金融机器学习标注": [
        ("元标注 Meta-Labeling：让二级模型决定『信这个信号多少钱』", "meta-labeling-strategy"),
        ("趋势扫描标注 Trend Scanning：用滚动回归 t 值给样本贴上趋势标签", "trend-scanning-labels"),
        ("CUSUM 结构突变检测：用累积和滤波器抓住行情的『变脸』时刻", "cusum-structural-breaks"),
        ("序贯自助法样本权重：修正金融标签重叠带来的伪独立性", "sequential-bootstrap-weights"),
        ("熵特征工程：用信息熵度量价格序列的『可预测性』", "entropy-features-trading"),
    ],
    "前沿扩展2026-07-30B-信息驱动采样与风控": [
        ("美元棒采样 Dollar Bars：按成交金额而非时钟切分K线", "dollar-bars-sampling"),
        ("失衡棒 Imbalance Bars：让信息流的不对称决定采样节奏", "microstructure-bars-imbalance"),
        ("VPIN 订单流毒性：用成交量同步的知情概率预警闪崩", "vpin-toxicity"),
        ("CPPI 组合保险：用动态乘数在保本底线上追逐风险资产", "cppi-portfolio-insurance"),
        ("分数凯利准则：在全凯利的暴涨暴跌与半凯利的稳健之间选择", "fractional-kelly-criterion"),
    ],
    # ===== 2026-07-30 补充：全库 1236 篇，主池+全部旧兜底已用尽，新增在线组合学习 / 时序模式挖掘真正全新主题（2026-07-30 全库磁盘扫描确认未发布）=====
    "前沿扩展2026-07-30-在线组合学习": [
        ("Cover 通用投资组合：不预测市场也能逼近最优常数再平衡", "cover-universal-portfolio"),
        ("在线牛顿步 ONS：用二阶信息加速在线组合选择的遗憾收敛", "ons-online-newton-step"),
        ("CORN 模式匹配组合：用相关性找历史上的『相似市场』再优化", "corn-pattern-matching"),
        ("Anticor 反相关组合：把资金从刚涨过的资产搬去反相关的落后者", "anticor-portfolio"),
        ("指数梯度组合 EG：用乘性更新在线追踪最优资产权重", "exponential-gradient-portfolio"),
    ],
    "前沿扩展2026-07-30-时序模式挖掘": [
        ("Matrix Profile 模体发现：线性时间找出价格序列里的重复形态", "matrix-profile-motif"),
        ("SAX 符号聚合近似：把连续价格序列离散成可检索的字符串", "symbolic-aggregate-sax"),
        ("Shapelet 形态特征：从时间序列里学出最有判别力的子形状", "shapelet-time-series"),
        ("Discord 异常子序列：用最不像别处的片段捕捉市场异常", "discord-anomaly-detection"),
        ("DTW 重心平均：对齐时间扭曲后求出K线形态的『平均脸』", "dtw-barycenter-averaging"),
    ],
    # ===== 2026-07-28-B 补充：全库 1175 篇，主池仅剩个别空闲，新增交易执行 / 价差估计真正全新主题（2026-07-28 全库磁盘扫描确认未发布）=====
    "前沿扩展2026-07-28B-最优执行": [
        ("VWAP 与 TWAP 执行：把大单切成时间与成交量加权的碎片", "vwap-twap-execution"),
        ("执行落差 Implementation Shortfall：从决策价到成交价流失了多少钱", "implementation-shortfall-is"),
        ("Kyle Lambda 价格冲击：从订单流回归里估出市场的『吃单成本』", "kyle-lambda-price-impact"),
        ("平方根冲击定律：为什么成交成本随下单量的平方根增长", "square-root-law-impact"),
        ("传播子模型 Propagator：把每笔成交的冲击衰减写成核函数", "propagator-model-market-impact"),
        ("POV 参与率执行：跟着市场成交量的节奏出货", "pov-participation-execution"),
    ],
    "前沿扩展2026-07-28B-价差与信息交易": [
        ("Glosten-Milgrom 模型：逆向选择如何撑开买卖价差", "glosten-milgrom-spread"),
        ("Roll 价差估计量：只用收盘价的自协方差反推有效价差", "roll-spread-estimator"),
        ("Corwin-Schultz 高低价价差估计：用两日高低价把价差与波动拆开", "corwin-schultz-high-low"),
        ("Abdi-Ranaldo 价差估计：用收盘价与中间价的关系改良 Roll 估计", "abdi-ranaldo-spread"),
        ("PIN 知情交易概率：用买卖单不平衡估计市场里有多少『内幕单』", "pin-probability-informed-trading"),
        ("Easley-O'Hara 信息模型：信息事件如何驱动订单流与价差", "easley-ohara-microstructure"),
    ],
    # ===== 2026-07-28 补充：主池 + 全部旧兜底已全库用尽（1156 篇），新增真正全新主题（置于最前优先选用）=====
    # 以下 slug 已于 2026-07-28 全库磁盘扫描确认【未发布】。协方差估计 / 组合稳健化方向，均可用 numpy/scipy 合成数据演示。
    "前沿扩展2026-07-28-协方差估计": [
        ("Ledoit-Wolf 收缩估计：把样本协方差往结构化目标拉一把", "ledoit-wolf-shrinkage"),
        ("随机矩阵理论去噪：用 Marchenko-Pastur 边界剪掉协方差里的噪声特征值", "random-matrix-marchenko-pastur"),
        ("EWMA 协方差 RiskMetrics：用指数衰减权重让风险估计跟上市场", "ewma-riskmetrics-covariance"),
        ("DCC-GARCH 动态相关：让资产间相关系数随时间自行演化", "dcc-garch-correlation"),
        ("Graphical Lasso 稀疏逆协方差：用 L1 惩罚画出资产间的条件依赖网络", "graphical-lasso-precision"),
        ("Tyler M-估计量：对肥尾免疫的稳健散布矩阵", "tyler-m-estimator-covariance"),
        ("常相关模型 CCM：把整个相关矩阵压成一个平均相关系数", "constant-correlation-model"),
        ("Gerber 统计量：用共动阈值替代皮尔逊相关构建组合", "gerber-statistic-covariance"),
    ],
    "前沿扩展2026-07-28-稳健组合构建": [
        ("Michaud 重抽样有效前沿：用 Bootstrap 平均把优化器的极端权重驯服", "michaud-resampled-frontier"),
        ("Bayes-Stein 收缩期望收益：把各资产均值往全局大盘拉", "bayes-stein-shrinkage-returns"),
        ("嵌套聚类优化 NCO：先聚类再优化的两层组合构建", "nested-clustered-optimization"),
        ("范数约束组合优化：用权重范数上限间接实现收缩", "norm-constrained-portfolio"),
        ("马氏距离金融湍流指数：用历史分布的『异常度』做风险开关", "mahalanobis-turbulence-index"),
        ("吸收比率与市场脆弱性：用主成分集中度预警系统性风险", "absorption-ratio-fragility"),
        ("相关矩阵去噪与去调：把市场模式从相关结构里剥出来再聚类", "denoising-detoning-correlation"),
        ("因子结构协方差 POET：用主成分加稀疏残差拼出高维协方差", "factor-covariance-poet"),
    ],
    # ===== 2026-07-27 补充：主池 + 全部旧兜底已全库用尽（1121 篇），新增真正全新主题（置于最前优先选用）=====
    # 以下 slug 已于 2026-07-27 全库磁盘扫描确认【未发布】。均为可用 numpy/scipy 合成数据演示的量化专题。
    "前沿扩展2026-07-27-风险度量": [
        ("Cornish-Fisher VaR：用偏度峰度修正正态分位数，给肥尾风险一个快速估计", "cornish-fisher-var"),
        ("Johnson SU 分布拟合收益：用四参数分布同时抓住偏度与厚尾", "johnson-su-distribution"),
        ("NIG 正态逆高斯分布：用半重尾分布刻画收益的尖峰厚尾", "nig-distribution-returns"),
        ("谱风险度量：用风险厌恶权重函数把整条尾部写进单一数字", "spectral-risk-measure"),
        ("熵风险度量 EVaR：用指数矩给 CVaR 一个更保守的上界", "entropic-var-evar"),
        ("Rachev 比率：用两端尾部期望之比重新定义收益风险性价比", "rachev-ratio-tail"),
    ],
    "前沿扩展2026-07-27-回测显著性": [
        ("概率夏普率 PSR：给你的夏普率算出一个『真的大于零』的置信度", "probabilistic-sharpe-ratio"),
        ("最小回测长度 MinTRL：多长的样本才够证明策略不是运气", "minimum-track-record"),
        ("多重检验夏普折扣：用 Bonferroni/BHY 给数据挖掘出来的夏普打折", "haircut-sharpe-multiple-testing"),
        ("组合式净化交叉验证 CPCV：用组合分块把回测过拟合概率算出来", "combinatorial-purged-cv"),
        ("回测过拟合概率 PBO：用组合对称检验量化『选出来的最优』有多假", "pbo-overfitting-probability"),
        ("White Reality Check：用自助法检验最优策略是否只是数据窥探", "white-reality-check"),
        ("Hansen SPA 检验：在一堆策略里判断有没有真正跑赢基准的", "hansen-spa-test"),
        ("走向前优化 Walk-Forward：用滚动样本外把参数拟合的水分挤掉", "walk-forward-optimization"),
        ("蒙特卡洛置换检验：用打乱标签给策略收益算经验 p 值", "monte-carlo-permutation-test"),
    ],
    # ===== 2026-07-23 补充：主池 + 旧兜底已全库用尽（976 篇），新增真正全新主题（置于最前优先选用）=====
    "前沿扩展2026-07-23": [
        ("距离相关系数:用 Hilbert-Schmidt 独立性把非线性依赖从噪声里捞出来", "distance-correlation-dependence"),
        ("排列熵与复杂度:用序数模式把『混沌』和『随机』一刀切开", "permutation-entropy-complexity"),
        ("DCov 距离协方差:用特征映射的核把依赖关系写成距离", "dcov-nonlinear-dependence"),
        ("序数模式熵:用涨跌排列的『形状』识别市场状态切换", "ordinal-pattern-entropy"),
        ("多尺度熵与 DFA:用粗粒化把不同时间尺度的不规则性分开", "multiscale-entropy-dfa"),
    ],
    # ===== 2026-07-19 补充：主池 + 旧兜底已全库用尽，新增真正全新主题（置于最前优先选用）=====
    "前沿扩展2026-07-19": [
        ("溢出指数与连通性网络:用 Diebold-Yilmaz 把风险传染画成地图", "diebold-yilmaz-connectedness"),
        ("已实现波动率双幂变差:用 Bipower 检验把连续波动与跳跃剥开", "realized-bipower-variation"),
    ],
    "因子研究": [
        ("因子择时:动态调整因子暴露", "factor-timing"),
        ("因子拥挤度监测与规避", "factor-crowding"),
        ("多因子模型风险分解", "multi-factor-risk-decomposition"),
    ],
    "统计套利": [
        ("配对交易与协整分析", "pair-trading-cointegration"),
        ("统计套利:均值回归策略", "statistical-arbitrage-mean-reversion"),
        ("PCA与因子模型在统计套利中的应用", "pca-statistical-arbitrage"),
    ],
    "机器学习": [
        ("XGBoost与LightGBM在量化选股中的应用", "xgboost-lightgbm-stock-selection"),
        ("Transformer模型股价预测", "transformer-stock-prediction"),
        ("强化学习算法交易", "reinforcement-learning-algo-trading"),
    ],
    "高频交易": [
        ("FPGA在量化交易中的应用", "fpga-quant-trading"),
        ("冰山订单检测与交易策略", "iceberg-order-detection"),
        ("高频交易监管与合规", "hft-regulation-compliance"),
    ],
    "期权策略": [
        ("VIX衍生品交易策略", "vix-derivatives-trading"),
        ("期权做市商策略", "option-market-making"),
        ("波动率风险溢价捕捉", "volatility-risk-premium"),
    ],
    "风险管理": [
        ("流动性风险管理", "liquidity-risk-management"),
        ("模型风险与回测过拟合检测", "model-risk-backtest-overfitting"),
        ("极端风险建模:肥尾与黑天鹅", "extreme-risk-modeling"),
    ],
    "投资组合": [
        ("风险预算模型", "risk-budgeting-model"),
        ("动态资产配置策略", "dynamic-asset-allocation"),
        ("Smart Beta与因子投资", "smart-beta-factor-investing"),
    ],
    "行为金融": [
        ("分析师预期偏差与量化策略", "analyst-bias-quant-strategy"),
        ("社交媒体情绪传播与价格发现", "social-media-sentiment-price-discovery"),
        ("行为组合理论", "behavioral-portfolio-theory"),
    ],
    "另类数据": [
        ("供应链数据在量化投资中的应用", "supply-chain-alternative-data"),
        ("气象数据对大宗商品预测", "weather-data-commodity-prediction"),
        ("网页爬虫与文本挖掘", "web-scraping-text-mining"),
    ],
    "实盘系统": [
        ("量化策略监控系统设计", "quant-strategy-monitoring"),
        ("交易成本控制与优化", "trading-cost-control"),
        ("量化策略合规与审计", "quant-compliance-audit"),
    ],
    # ===== 以下为新增主题（避免池子耗尽后回退到已存在文章）=====
    "高级专题": [
        ("卡尔曼滤波在动态对冲中的应用", "kalman-dynamic-hedging"),
        ("波动率锥：用历史分位给期权定价与择时", "volatility-cone"),
        ("障碍期权与奇异期权在结构化策略中的应用", "exotic-options-structures"),
    ],
    "组合优化进阶": [
        ("遗传算法与进化策略在组合优化中的应用", "genetic-algo-portfolio"),
        ("风险平价与波动率目标的实战融合", "risk-parity-vol-target"),
    ],
    "ESG与信用": [
        ("ESG 因子量化：可持续数据能否转化为 Alpha", "esg-factor-quant"),
        ("信用利差因子：用债券相对价值挖掘 Alpha", "credit-spread-factor"),
    ],
    # ===== 固定收益与利率（新增，避免池子耗尽后回退到已存在文章）=====
    "固定收益与利率": [
        ("利率期限结构 Nelson-Siegel 模型：用少数因子刻画整条收益率曲线", "nelson-siegel-term-structure"),
        ("隐含波动率与已实现波动率套利：波动率风险的双面", "implied-vs-realized-vol"),
        ("方差互换与波动率互换：做空波动率风险溢价的衍生品工具", "variance-swap-trading"),
        ("贝叶斯资产配置：用共轭先验把观点变成权重", "bayesian-asset-allocation"),
    ],
    # ===== 以下为新增主题（避免池子耗尽后回退到已存在文章）=====
    "波动率建模": [
        ("GARCH 波动率模型族：从 ARCH 到 EGARCH 的实战预测", "garch-volatility-family"),
    ],
    "组合优化进阶II": [
        ("最大分散化组合：用相关性加权超越风险平价", "maximum-diversification-portfolio"),
        ("随机波动率模型与 MCMC 估计：给波动率加一层隐状态", "stochastic-volatility-mcmc"),
    ],
    "市场微观结构进阶": [
        ("订单流毒性与 VPIN 指标：识别被逆向选择的高频风险", "order-flow-toxicity-vpin"),
    ],
    "因子工程进阶": [
        ("因子正交化：剔除多重共线性的关键技术", "factor-orthogonalization-methods"),
        ("因子截面对冲：用行业市值中性化剔除风格暴露", "factor-neutralization-industry"),
    ],
    # ===== 主题扩展（2026-07，避免池子耗尽后回退到已存在文章）=====
    "隐马尔可夫": [
        ("隐马尔可夫模型(HMM)市场状态识别：让策略知道现在是牛是熊", "hmm-market-regime"),
    ],
    "资金管理": [
        ("Kelly 公式与资金管理：用分数凯利把收益与破产风险一起优化", "kelly-criterion-sizing"),
    ],
    "期权对冲": [
        ("期权 Gamma Scalping：用动态对冲把 Theta 变成 Alpha", "gamma-scalping-options"),
    ],
    "依赖建模": [
        ("Copula 依赖建模：捕捉相关性之外的尾部联动风险", "copula-dependence-modeling"),
    ],
    "流动性溢价": [
        ("Amihud 非流动性溢价：用交易摩擦挖出稳健的选股因子", "amihud-illiquidity-premium"),
    ],
    "深度学习因子": [
        ("自编码器因子挖掘：用深度学习提取非线性 Alpha", "autoencoder-factor-mining"),
    ],
    "跨期套利": [
        ("日历价差与跨期套利：赚期限结构曲线的钱", "calendar-spread-arbitrage"),
    ],
    "风险模型": [
        ("Barra 风险模型实战拆解：把你持仓的每一个风险都摊开来", "barra-risk-model-practice"),
    ],
    # ===== 前沿扩展（2026-07-15 补充，避免主题池再次耗尽）=====
    # 以下 slug 已于 2026-07-15 全库扫描确认为【未发布】。每次运行消耗 2 个，
    # 剩余可支撑后续多次自动运行；新主题均带真实可合成数据的 Python 配图。
    "前沿扩展2026-07-15": [
        ("Ledoit-Wolf 协方差收缩：把样本协方差的噪声压回信号", "covariance-shrinkage-ledoit-wolf"),
        ("Heston 随机波动率模型：用特征函数 + 校准还原波动率曲面", "heston-model-calibration"),
        ("Breeden-Litzenberger 隐含风险中性密度：从期权价格反推市场预期的崩盘", "breeden-litzenberger-implied-density"),
        ("Merton 违约距离：用期权定价思想给公司违约概率打分", "credit-merton-distance-to-default"),
        ("PCA 特征组合：把相关性矩阵的主成分变成可交易因子", "pca-eigenportfolio"),
        ("高阶矩组合：把偏度与峰度也写进目标函数", "higher-moment-portfolio"),
        ("稀疏 PCA 统计套利：只保留少数正交共性因子", "sparse-pca-stat-arb"),
        ("相关性网络与最小生成树：用图论看清板块联动结构", "correlation-network-mst"),
        ("FIGARCH 长记忆波动：用分整把波动的慢衰减建模", "figarch-long-memory"),
        ("并购套利风险：用价差与完成概率给 deal spread 定价", "merger-arbitrage-risk"),
        ("应计异象：利润表里的现金 vs 纸面盈利谁更可靠", "accruals-anomaly"),
        ("Altman Z-score 财务困境：用五变量线性判别预警破产", "altman-zscore-distress"),
        ("波动率之波动率(VVIX)：恐慌的二阶矩能否预测", "volatility-of-volatility"),
        ("VSTOXX 策略：欧洲恐慌指数的期限结构与交易", "vstoxx-strategy"),
        ("Temporal Fusion Transformer：把多变量时序预测做成可解释黑盒", "temporal-fusion-transformer"),
        ("N-BEATS 神经基展开：无卷积无注意力的纯残差时序预测", "nbeats-forecast"),
        ("Nowcasting 混频实时预测：用高频指标盯住低频 GDP", "nowcasting-midas-gdp"),
        ("净权益发行异象：公司回购越多未来越该涨", "composite-equity-issuance"),
        ("OAS 信用利差收缩：把债券相对价值做成可交易信号", "oas-covariance-shrinkage"),
    ],
    # ===== 2026-07-16 补充：主池 + 旧兜底已全库用尽（794 篇），新增真正全新主题 =====
    "前沿扩展2026-07-16-A": [
        ("层级风险平价(HRP)：用相关性树把马科维茨的协方差病治好", "hierarchical-risk-parity-hrp"),
        ("熵率市场状态识别：用信息熵把『混沌』和『秩序』分开", "entropy-market-regime"),
        ("毛利率因子：Novy-Marx 之后，利润表的现金含量才是硬 alpha", "gross-profitability-factor"),
        ("价值因子复兴：当 cheap 被踩进泥里，用稳健估值捡便宜货", "value-factor-revival"),
        ("时间序列动量：不比截面比自己，用过去收益预测未来收益", "time-series-momentum"),
        ("系统性 Carry 交易：用利差把『被奖励的持险』做成策略", "carry-trade-systematic"),
    ],
    "前沿扩展2026-07-16-B": [
        ("深度对冲：用神经网络取代 BS 公式做期权动态对冲", "deep-hedging-neural"),
        ("最优执行 Almgren-Chriss：把大单拆成最优执行流", "optimal-execution-almgren"),
        ("粗糙波动率模型：用分数布朗运动重写波动率的记忆", "rough-volatility-model"),
        ("转移熵因果识别：用信息流方向分清『谁带动谁』", "transfer-entropy-finance"),
        ("盈余公告漂移 PEAD：用事件后 60 日的惯性捡市场慢半拍的钱", "post-earnings-drift"),
        ("因子动量：当因子本身也有趋势，用因子收益做二阶择时", "factor-momentum-signal"),
        ("FinBERT 新闻情绪：用金融预训练模型把公告变成 alpha", "finbert-news-sentiment"),
        ("CVaR 优化 Rockafellar：把『最坏情况的平均』写进目标函数", "cvar-optimization-rockafellar"),
        ("52 周新高因子：用价格纪录突破捕捉动量拐点", "fifty-two-week-high"),
        ("商品期限结构：用展期收益把 contango/back 写成可交易信号", "commodity-term-structure"),
        ("Q 因子模型：用投资与盈利替代市值与账面做资产定价", "q-factor-model"),
        ("Fama-French 五因子实战拆解：把规模价值再加盈利与投资", "fama-french-five-factor"),
        ("短期反转因子：用隔夜过度反应捡日内均值回归的钱", "short-term-reversal-factor"),
    ],
}

def scan_recent_articles(n=10):
    """扫描最近n篇文章的标题（用于展示日志）"""
    articles = []

    # 获取所有文章目录，按时间排序
    dirs = sorted(
        [d for d in os.listdir(BLOG_DIR) if os.path.isdir(os.path.join(BLOG_DIR, d))],
        key=lambda x: os.path.getmtime(os.path.join(BLOG_DIR, x)),
        reverse=True
    )

    for dir_name in dirs[:n]:
        index_path = os.path.join(BLOG_DIR, dir_name, "index.md")
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 提取标题
                title_match = re.search(r'^title:\s*["\']?(.*?)["\']?$', content, re.MULTILINE)
                if title_match:
                    articles.append({
                        'dir': dir_name,
                        'title': title_match.group(1)
                    })

    return articles


def scan_all_articles():
    """扫描全部已有文章，返回 (dir, title) 列表，用于去重（防止覆盖已有文章）"""
    articles = []
    if not os.path.isdir(BLOG_DIR):
        return articles
    for dir_name in os.listdir(BLOG_DIR):
        if not os.path.isdir(os.path.join(BLOG_DIR, dir_name)):
            continue
        index_path = os.path.join(BLOG_DIR, dir_name, "index.md")
        title = None
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
                title_match = re.search(r'^title:\s*["\']?(.*?)["\']?$', content, re.MULTILINE)
                if title_match:
                    title = title_match.group(1)
        articles.append({'dir': dir_name, 'title': title})
    return articles

def select_topics():
    """选择2个不重复的主题（排除所有已有文章，避免覆盖旧内容）"""
    # 展示最近10篇（仅日志用途）
    recent = scan_recent_articles(10)
    recent_titles = [a['title'] for a in recent if a['title']]

    print(f"📊 最近10篇文章：")
    for i, a in enumerate(recent, 1):
        print(f"  {i}. {a['title']}")

    # 关键修正：去重基于【全部】已有文章（slug + 标题），而非仅最近10篇
    all_articles = scan_all_articles()
    existing_slugs = {a['dir'] for a in all_articles}
    existing_titles = {a['title'] for a in all_articles if a['title']}

    def is_duplicate(title, slug):
        # slug 是唯一物理目录名：只要 slug 未被占用即可安全新建
        if slug in existing_slugs:
            return True
        # 标题采用精确匹配（去除引号/空格），避免子串误杀全新主题
        tn = (title or "").replace('"', '').replace("'", '').strip()
        for t in existing_titles:
            if not t:
                continue
            te = t.replace('"', '').replace("'", '').strip()
            if tn == te:
                return True
        return False

    # 1) 展开全部候选主题，过滤掉已存在的
    candidates = []
    for category, topics in TOPIC_POOL.items():
        for title, slug in topics:
            if not is_duplicate(title, slug):
                candidates.append((category, title, slug))

    # 2) 优先选 2 个不同领域的主题；不足时退而求其次（仍保证 slug 不重复）
    selected = []
    used_categories = set()
    for category, title, slug in candidates:
        if len(selected) >= 2:
            break
        if category in used_categories:
            continue
        selected.append((title, slug, category))
        used_categories.add(category)

    # 3) 若不同领域不足 2 个，放宽到同领域但 slug 必须不同
    if len(selected) < 2:
        for category, title, slug in candidates:
            if len(selected) >= 2:
                break
            if any(slug == s for _, s, _ in selected):
                continue
            selected.append((title, slug, category))

    # 4) 兜底：连一个候选都没有（池子用尽），新增一组【真正全新】的主题
    #    注意：兜底主题必须与主池不同 slug，避免再次被 is_duplicate 全部过滤
    if len(selected) < 2:
        print("⚠️ 警告：主主题池已用尽，使用内置全新主题兜底（已逐一校验 slug 全库不存在）")
        # 以下为 2026-07-14 全库扫描确认【未发布】的前沿专题；slug 唯一、标题精确不重复。
        # 主题池已用尽，这里保留一批可持续使用的新专题，避免以后再次全库命中。
        fallback_pool = [
            ("最优传输(Wasserstein)资产配置：用 Earth Mover's Distance 把分布差异变成权重", "optimal-transport-portfolio"),
            ("高频跳跃检测(Lee-Mykland)：从噪声里揪出真实的瞬时跳变", "high-frequency-jump-detection"),
            ("智能订单路由(SOR)：在多个交易场所之间把大单拆成最优流", "smart-order-routing-venues"),
            ("波动率管理组合(Moreira-Muir)：用已实现波动缩放敞口而非预测方向", "volatility-managed-portfolio"),
            ("杠杆厌恶因子(Frazzini-Pedersen)：被杠杆约束压低的低 Beta 溢价", "leverage-aversion-factor"),
            ("贝叶斯状态空间择时：把『现在是牛是熊』变成后验概率", "bayesian-regime-timing"),
            ("事件研究 CAR/CAAR：用市场模型量化公告的异常收益", "event-study-car-caar"),
            ("订单簿失衡 LSTM：用深度模型预测秒级价格方向", "lob-imbalance-lstm"),
            ("流动性共同因子：Amihud 溢价的系统性来源与危机放大", "liquidity-common-factor"),
            ("波动率风险溢价择时：用 VRP 开关股债配置", "vrp-timing-allocation"),
            ("偏度偏好因子：散户为何为左尾买单、机构如何收割", "skewness-preference-factor"),
            ("做市库存风险：Avellaneda-Stoikov 框架下的库存-对冲博弈", "market-making-inventory-avellaneda"),
            ("已实现半方差(HARQ)：把隔夜跳变与日内连续波动拆开预测", "harq-realized-volatility"),
            ("截面动量的衰减曲线：因子信号逐日『腐烂』的实证与对抗", "cross-sectional-momentum-decay"),
            ("滚动窗口协整：用 DCC 抓住会『断』的长期关系", "rolling-cointegration-break"),
            ("Fama-MacBeth 截面回归：两步法把因子暴露变成风险溢价", "fama-macbeth-cross-section"),
            ("Hawkes 过程与自激点过程：给订单流的 contagion 建模", "hawkes-process-order-flow"),
            ("CoVaR 与系统性风险：一只机构倒下会拖垮多少同行", "covar-systemic-risk"),
            ("LPPL 对数周期幂律：用临界泡沫模型给崩盘点位预警", "lppl-bubble-prediction"),
            ("MIDAS 混频回归：用高频数据预测低频变量", "midas-mixed-frequency"),
            ("TSRV 二次变差：用子采样把微观结构噪声剔出波动率", "tsrv-two-scale-volatility"),
            ("离散度交易：用个股 IV 与指数 IV 的背离做波动率套利", "dispersion-trading-correlation"),
            ("预期短缺回测：用 ES 检验替代 VaR 的后验失序", "expected-shortfall-backtesting"),
            ("动态 Nelson-Siegel：用三因子状态空间跟踪收益率曲线形变", "dynamic-nelson-siegel"),
            ("系统性风险网络：用关联图把传染路径画出来", "systemic-risk-network"),
        # ===== 2026-07-16 新增：主池 + 旧兜底已全库用尽，补充真正全新主题 =====
        ("日历异象与季节性：月初效应、Sell-in-May、周末效应能否变成可交易信号", "calendar-anomaly-seasonality"),
        ("ATR 吊灯退出：用真实波幅把止损变成随波动率呼吸的活防线", "atr-chandelier-exit"),
        ("ADX 趋势强度：用定向运动指数把『有趋势』和『无趋势』分开", "adx-trend-strength"),
        ("抛物线 SAR 与追踪止损：用加速因子把出场点焊在价格上", "parabolic-sar-trailing"),
        ("周内效应再检验：星期几真的有超额收益吗", "day-of-week-effect"),
        ("质量因子质量溢价：用盈利稳定性把『好公司』写成可交易 alpha", "quality-factor-premium"),
        ("低波动异象：低风险股票为何长期跑赢高风险", "low-volatility-anomaly-cn"),
        # ===== 2026-07-17 新增：主池/旧兜底全库用尽（812 篇），补充真正全新主题 =====
        # 以下 slug 已于 2026-07-17 全库扫描确认【未发布】。均为可合成真实数据的量化专题。
        ("Vasicek 利率模型：用 Ornstein-Uhlenbeck 过程给整条收益率曲线定价", "vasicek-rate-model"),
        ("卡尔曼滤波配对交易：用状态空间模型动态追踪对冲比例", "kalman-pairs-trading"),
        ("马尔可夫切换 GARCH：让波动率在高/低波动状态间自行切换", "markov-switching-garch"),
        ("CIR 利率模型：用平方根扩散守住利率的非负底线", "cir-interest-rate-model"),
        ("Hurst 指数与分形市场：用重标极差判断序列是随机游走还是有记忆", "hurst-exponent-regime"),
        ("隔夜收益异象：用收盘到开盘的跳变预测次日方向", "overnight-return-anomaly"),
        ("等风险贡献组合(ERC)：让每个资产对总风险的贡献完全相等", "equal-risk-contribution"),
        ("Beta 中性组合：用市场对冲把方向风险剥离干净只留 alpha", "beta-neutral-portfolio"),
        ("极值理论尾部依赖：用广义帕累托分布给极端联动定价", "evt-tail-dependence"),
        ("跨资产宏观动量：把股、债、商品、外汇的趋势做成单一择时信号", "cross-asset-macro-momentum"),
        ("Put-Call 比率情绪：用期权持仓失衡给市场恐慌测温", "put-call-ratio-sentiment"),
        ("VIX 均值回归：用恐慌指数的回复特性做波动率择时", "vix-mean-reversion"),
        ("ARFIMA 长记忆：用分数差分把波动与收益的慢衰减写进模型", "arfima-long-memory"),
        ("日内成交量季节性：用 U 型曲线预测盘中流动性窗口", "intraday-volume-seasonality"),
        ("反演优化隐含风险：从组合的边际风险反推各大类资产的真实暴露", "reverse-optimization-implied"),
        ("因子模拟组合：用多头空头组合把抽象因子变成可交易资产", "factor-mimicking-portfolio"),
        ("因子跨越检验：用 GRS 检验判断新因子是否真有增量信息", "factor-spanning-test"),
        ("动量反转的持有期边界：用持有期扫描找出动量转反转的拐点", "momentum-reversal-horizon"),
        ("GARCH 跳跃模型：把跳跃成分从连续波动里剥离开来", "garch-jumps"),
        ("LSV 流动性风险：用流动性调整给资产定价加一道折扣", "lsv-liquidity-risk"),
        ("遍历组合：用深套后的均值回复重建最优长期增长路径", "ergodic-portfolio"),
        ("分数布朗运动交易：用 Hurst 依赖重写均值回复策略", "fractional-brownian-trading"),
        ("RSI 背离交易：用价格与动能的背离捕捉趋势衰竭", "rsi-divergence"),
        ("OU 过程均值回复：用 Ornstein-Uhlenbeck 给配对价差做统计套利", "mean-reversion-ou"),
        ("熵池化压力测试：把先验观点与历史分布融合成压力情景", "entropy-pooling-stress"),
        ]
        # ===== 2026-07-18 补充：主池 + 旧兜底已全库用尽（844 篇），补充真正全新主题 =====
        # 以下 slug 已于 2026-07-18 全库扫描确认【未发布】。均为可合成真实数据的量化专题。
        refresh_pool = [
            ("Bootstrap 回测显著性：用重抽样给你的夏普率算出 p 值", "bootstrap-backtest-pvalue"),
            ("动态因子模型(DFM)：用少数共同因子把宏观面板压缩成可预测信号", "dynamic-factor-model-dfm"),
            ("稳健统计在量化中的应用：用中位数与 MAD 替代均值方差抗极端值", "robust-statistics-quant"),
            ("False Discovery Rate：用 BH 程序在多因子检验里控制假阳性", "false-discovery-rate-fdr"),
            ("基本面定律 Grinold-Kahn：用信息比率拆解主动管理的收益上限", "fundamental-law-active-management"),
            ("采样频率偏差：日频 vs 周频回测为何给出不同夏普", "sampling-frequency-bias"),
            ("白噪声检验 Ljung-Box：用自相关把『真信号』和『伪相关』分开", "white-noise-ljung-box"),
        ]
        fallback_pool = refresh_pool + fallback_pool

        # ===== 2026-07-19 补充：主池 + 旧兜底已全库用尽（≥854 篇），补充真正全新主题 =====
        # 以下 slug 已于 2026-07-19 全库扫描确认【未发布】。均为可合成真实数据的量化专题。
        fresh_pool = [
            ("Lasso 与弹性网络变量选择：用 L1 惩罚把几百个因子压成可交易少数", "lasso-elasticnet-variable-selection"),
            ("共形预测在量化中的应用：给预测区间一个可验证的覆盖率保证", "conformal-prediction-quant"),
            ("概念漂移检测：当因子失效，用漂移预警救回你的组合", "concept-drift-factor-failure"),
            ("隐含相关性：从指数期权反推市场对未来相关结构的预期", "implied-correlation-options"),
            ("贝叶斯模型平均：把『哪个模型对』变成权重而非赌注", "bayesian-model-averaging-asset"),
            ("分数差分平稳化：在保留记忆与消除单位根之间取平衡", "fractional-differentiation-stationary"),
            ("卡尔曼滤波动态跟踪：用状态空间模型把噪声里的真值估出来", "kalman-filter-tracking"),
            ("GMM 资产定价：用矩条件代替强假设，让数据自己说话", "gmm-asset-pricing"),
        ]
        fallback_pool = fresh_pool + fallback_pool

        for title, slug in fallback_pool:
            if len(selected) >= 2:
                break
            if is_duplicate(title, slug):
                continue
            if any(slug == s for _, s, _ in selected):
                continue
            selected.append((title, slug, "实盘系统"))

    # ===== ROBUST FALLBACK (运行时扫描磁盘, 不再依赖写死清单) =====
    # 旧版写死的 "guaranteed_free" 清单会随文章发布而失效(全部已发布 -> 去重过滤
    # 后只剩 <2 个 -> assert 失败)。改为: 从一份"新鲜候选池"里逐条检查 slug 是否
    # 真在磁盘上【未发布】, 取前 2 个未被占用的 (标题按 slug 即时生成, 避免重复)。
    # 候选池保持较长, 即使前面用掉一部分也总能找到空闲 slug。
    if len(selected) < 2:
        # ===== 2026-07-19 彻底修复：旧 fresh_pool 全部已发布 -> 去重后 0 个 -> assert 失败。
        # 改为 2026-07-19 全库扫描确认【未发布】的 77 个真正全新主题，覆盖固收/衍生品/
        # 另类数据/加密链上/因子/风险/高频/微观结构/机器学习/配置/技术指标等方向，
        # 顺序即优先级；脚本每次取前 2 个磁盘上未占用的 slug（标题按 slug 即时生成避免重复）。
        fresh_pool = [
            # ===== 2026-08-30 补充：主池+旧兜底+旧 fresh_pool 已全库用尽（1400+ 篇），新增真正全新主题（磁盘扫描逐一确认 slug 全库未发布）=====
            # 以下均为 numpy/scipy 可合成真实数据演示的量化专题；顺序即优先级。
            ("模糊厌恶与最大最小投资组合：用 worst-case 期望把『我可能看错了』写进优化目标", "ambiguity-aversion-maxmin-portfolio"),
            ("主成分套期保值：用少数公因子把组合暴露降到最低方差对冲", "pca-minimum-variance-hedge"),
            ("稳健 PCA 与异常因子剥离：用主成分把离群股票的影响压下来", "robust-pca-outlier-factor"),
            ("Bootstrap IC 置信区间：给你的因子Rank-IC算出一个真显著的上下界", "bootstrap-ic-confidence-interval"),
            ("门限 GARCH 非对称波动：让波动只在『坏消息』上跳", "threshold-garch-asymmetric"),
            ("Merton 跳跃扩散资产定价：把跳变成分从连续波动里剥出来", "merton-jump-diffusion-asset"),
            ("最大熵投资组合：在『什么都不确定』时给出最不偏见的权重", "maximum-entropy-portfolio"),
            ("Gram-Schmidt 因子正交化：把相关因子拆成互不重叠的纯净信号", "gram-schmidt-factor-orthogonalization"),
            ("半方差与下行风险组合：只用亏损那半边方差做优化", "semivariance-downside-portfolio"),
            # ===== 2026-08-28 补充：主池 + 全部旧兜底已全库用尽（1376 篇），新增真正全新主题（前沿 AI/ML 量化方向，磁盘扫描确认未发布）=====
            ("KAN Kolmogorov-Arnold 网络：用可学习一元函数替代线性权重做因子建模", "kan-kolmogorov-arnold-network"),
            ("时间序列基础模型：用预训练大模型做零样本金融预测", "time-series-foundation-model"),
            ("扩散模型合成金融序列：为稀缺行情数据做可控数据增强", "diffusion-financial-synthetic"),
            ("智能体市场微观结构仿真：用多 Agent 重演限价簿价格发现", "agent-based-market-simulation"),
            ("可微分组合优化：把 Markowitz 写进可反向传播的图", "differentiable-portfolio-optimization"),
            ("MoE 混合专家因子模型：用稀疏激活给不同市场状态配不同专家", "moe-expert-factor-model"),
            ("信息几何与组合：把权重单纯形上的优化变成黎曼梯度下降", "information-geometry-portfolio"),
            ("量子退火组合优化：用退火的量子涨落跳出均值方差的局部最优", "quantum-annealing-portfolio"),
            ("联邦学习因子共享：在不交换原始数据的前提下联合训练选股模型", "federated-learning-factor"),
            ("因果发现 PC 算法：用条件独立性把变量间的真实因果图挖出来", "causal-discovery-pc"),
            ("扩散模型收益分布建模：用分数匹配拟合整条尾部", "score-matching-return-distribution"),
            ("大语言模型作为因子矿工：用 prompt 工程把文本变成可交易信号", "llm-factor-miner"),
            ("多智能体强化学习交易：用对手建模学会在博弈中下单", "multi-agent-rl-trading"),
            ("状态空间模型 S4/S5：用结构化状态空间替代注意力做长序列", "state-space-model-s4"),
            ("图神经网络过平滑：在板块关联图上避免深层 GNN 的信号坍塌", "gnn-over-smoothing-finance"),
            ("对比学习负采样策略：金融表征里该选哪些『难负例』", "contrastive-negative-sampling"),
            ("神经组合优化 Pointer Net：用 seq2seq 直接输出排序权重", "pointer-network-portfolio"),
            ("生存分析在违约预测：用 Cox 比例风险把『何时违约』变成风险率", "survival-analysis-default"),
            ("贝叶斯深度学习因子：用 MC Dropout 给神经网络预测配不确定性", "bayesian-deep-learning-factor"),
            ("最优输运再平衡：把组合调整写成分布间的最小搬运费", "optimal-transport-rebalance"),
            ("生成对抗模仿学习：从优秀交易员轨迹里学策略", "gail-imitation-trading"),
            ("时间序列因果推断 DoWhy：用反事实回答『如果当时换策略会怎样』", "time-series-causal-dowhy"),
            ("神经过程 Neural Process：用元学习给出带置信区间的点预测", "neural-process-finance"),
            ("稀疏专家路由因子：让门控网络动态挑选少量有效因子", "sparse-expert-routing-factor"),
            ("市场状态连续表征：用 VAE 隐变量把牛熊震荡编码成一条曲线", "latent-regime-representation"),
            ("Transformer 旋转位置编码金融改进：让周期与相位进入注意力", "rope-rotary-position-finance"),
            ("决策树因子可解释性：用规则集替代黑盒给出合规理由", "decision-tree-interpretable-factor"),
            ("在线学习概念漂移自适应：当因子缓慢失效时自动重加权", "online-concept-drift-adapt"),
            ("多任务因子学习：用硬共享底座同时预测收益与风险", "multi-task-factor-learning"),
            ("元强化学习少样本适应：让交易智能体在新市场快速上手", "meta-rl-fewshot-adapt"),
            ("图对比学习板块表征：用子图扰动学出行业嵌入向量", "graph-contrastive-sector-embedding"),
            ("神经微分方程组合再平衡：用连续时间控制优化调仓节奏", "neural-ode-rebalance"),
            ("可解释 boosting 因子归因：用 SHAP 树解释把收益拆到特征", "interpretable-boosting-shap"),
            # ===== 2026-08-01 补充：全库 1289 篇磁盘扫描确认以下 slug 均未发布（最优执行/微观结构/波动率估计量/协整配对专题，可用 numpy/pandas 合成数据演示）=====
            ("Hawkes 过程与成交到达：用自激点过程建模订单簇发", "hawkes-process-trade-arrival"),
            ("Avellaneda-Stoikov 做市：用库存风险与最优报价拆解买卖价差", "avellaneda-stoikov-market-making"),
            ("Obizhaeva-Wang 最优执行：把限价簿弹性写进下单节奏", "obizhaeva-wang-optimal-execution"),
            ("Parkinson 波动率估计量：用日内高低价榨出更高效的波动估计", "parkinson-volatility-estimator"),
            ("Garman-Klass 波动率：把开高低收四价一起塞进波动估计", "garman-klass-volatility"),
            ("Yang-Zhang 波动率估计量：同时免疫跳空与漂移的波动测量", "yang-zhang-volatility-estimator"),
            ("Hurst 指数与均值回归：用长记忆判别趋势还是反转", "hurst-exponent-mean-reversion"),
            ("Ornstein-Uhlenbeck 配对交易：给价差一个可交易的回归速度", "ornstein-uhlenbeck-pairs-trading"),
            ("Johansen 协整篮子：用多资产协整向量构建可交易价差", "johansen-cointegration-basket"),
            ("卡尔曼滤波动态对冲比率：让套保系数随市场自我更新", "kalman-filter-dynamic-hedge-ratio"),
            ("分数阶差分与平稳性：在保留记忆和通过 ADF 之间求平衡", "fractional-differencing-stationarity"),
            # ===== 2026-07-29C 补充：全库 1217 篇磁盘扫描确认以下 slug 均未发布（金融机器学习 AFML / 高频采样 / 微观结构特征专题，可用 numpy/pandas 合成数据演示）=====
            ("三重障碍标注法：给金融机器学习一个不作弊的标签", "triple-barrier-labeling"),
            ("元标注 Meta-Labeling：把方向和仓位拆成两个模型", "meta-labeling-position-sizing"),
            ("成交量时钟采样：用交易活动而非日历时间切分数据", "volume-clock-sampling"),
            ("美元 Bar 与信息驱动采样：让每根 K 线携带等量信息", "dollar-bars-information-sampling"),
            ("Tick 失衡 Bar：用订单流失衡触发的自适应采样", "tick-imbalance-bars"),
            ("CUSUM 事件过滤器：只在市场偏离足够大时才采样", "cusum-filter-event-sampling"),
            ("样本唯一性加权：重叠标签下的正确样本权重", "sample-weights-uniqueness"),
            ("MDI 与 MDA 特征重要性：识别真正有用的因子特征", "feature-importance-mdi-mda"),
            ("概率仓位管理 Bet Sizing：把模型置信度变成头寸大小", "bet-sizing-probability"),
            ("SADF 结构突变检验：实时监测价格序列的爆炸性泡沫", "structural-breaks-sadf"),
            ("微观结构熵特征：用信息论度量价格序列的可预测性", "entropy-features-microstructure"),
            ("Run Bars 采样：用连续同向成交的长度定义信息事件", "run-bars-sampling"),
            ("队列反应模型：限价簿排队动态的马尔可夫建模", "queue-reactive-lob-model"),
            ("Epps 效应：高频下相关系数为何随采样频率衰减", "epps-effect-highfreq"),
            ("高频领先滞后估计：谁先动，谁跟随", "lead-lag-highfreq-estimation"),
            ("日内动量之尾盘效应：最后半小时的可预测性", "intraday-momentum-last30min"),
            # ===== 2026-07-27 补充：全库 1139 篇，主池+全部旧兜底仅剩个别空闲，新增 VaR 回测检验 / 回撤风险 / 尾部建模真正全新主题（2026-07-27 全库磁盘扫描确认未发布）=====
            ("Kupiec POF 检验：用失败次数的似然比给 VaR 模型做体检", "kupiec-pof-test"),
            ("Christoffersen 独立性检验：VaR 突破是否扎堆出现", "christoffersen-independence-test"),
            ("巴塞尔交通灯检验：监管视角下的 VaR 模型分区", "var-traffic-light-basel"),
            ("Acerbi-Szekely ES 检验：给预期短缺一个可执行的回测", "acerbi-szekely-es-test"),
            ("动态分位数 DQ 检验：把 VaR 突破回归到信息集上", "dynamic-quantile-test-dq"),
            ("CAViaR 条件自回归 VaR：让分位数自己带记忆演化", "caviar-conditional-autoregressive-var"),
            ("分位数交叉与重排：修复多分位预测的次序违背", "quantile-crossing-rearrangement"),
            ("Expectile 风险度量：兼具一致性与可引出性的另一条尾部路线", "expectile-risk-evar"),
            ("GARCH 滤波历史模拟：把波动率标准化后再抽历史", "garch-filtered-historical-simulation"),
            ("Copula-GARCH 组合 VaR：边缘与依赖分开建模的组合风险", "copula-garch-portfolio-var"),
            ("区组极大值 GEV：用广义极值分布刻画最坏月份", "block-maxima-gev-risk"),
            ("超阈值 POT 建模：用广义帕累托分布拟合尾部超额", "peaks-over-threshold-pot"),
            ("状态条件 Kelly：把 regime 概率写进最优杠杆", "regime-conditional-kelly"),
            ("分数 Kelly 与回撤约束：在增长率和最大回撤之间讨价还价", "fractional-kelly-drawdown"),
            ("回撤在险值 CDaR：把回撤分布的尾部写进优化目标", "drawdown-at-risk-cdar"),
            ("条件回撤优化：直接最小化最差回撤路径的组合构建", "conditional-drawdown-optimization"),
            ("回撤的序列依赖：自相关如何放大最大回撤", "serial-dependence-drawdown"),
            # ===== 2026-07-26-C 补充：全库 1100 篇，主池+旧兜底几乎用尽，新增市场微观结构/流动性度量真正全新主题（2026-07-26 全库磁盘扫描确认未发布）=====
            ("Glosten-Milgrom 序贯交易模型：用贝叶斯更新把知情交易写进买卖价差", "glosten-milgrom-model"),
            ("Easley-O'Hara 序贯交易信息模型：从成交序列反推信息事件概率", "easley-ohara-sequential"),
            ("Kyle 连续拍卖模型：把知情者的最优下单速度解出来", "kyle-continuous-auction"),
            ("Stoll 已实现价差分解：把买卖价差拆成信息、库存与订单处理成本", "stoll-realized-spread"),
            ("Huang-Stoll 价差成分分解：用两步回归量化价差的三大来源", "huang-stoll-decomposition"),
            ("Lee-Ready 成交方向分类：只用价与报价推断每笔是买还是卖", "lee-ready-trade-classification"),
            ("批量成交分类 BVC：用成交量分布近似订单流方向", "bulk-volume-classification"),
            ("有效 Tick 估计量：用价格聚集反推隐含买卖价差", "effective-tick-estimator"),
            ("Gibbs 抽样估计买卖价差：用贝叶斯 MCMC 从收盘价里捞出交易成本", "gibbs-sampler-spread"),
            ("报价深度失衡：用限价簿两侧挂单量预测短期价格方向", "quoted-depth-imbalance"),
            ("订单失衡与收益：用成交方向净额预测下一段价格", "order-imbalance-return"),
            ("Amivest 流动性比率：用成交额换价格变动衡量市场深度", "amivest-liquidity-ratio"),
            ("零收益天数流动性度量：用停滞比例给流动性打分", "zeros-liquidity-measure"),
            ("换手率流动性因子：用成交活跃度截面选股", "turnover-liquidity-factor"),
            ("价格冲击衰减：用成交后的价格回复曲线拆临时与永久冲击", "price-impact-decay"),
            ("报价斜率流动性：用限价簿弹性度量瞬时深度", "quote-slope-liquidity"),
            ("加权价格贡献 WPC：多市场交易时谁贡献了价格发现", "weighted-price-contribution"),
            ("Tick Size 与流动性：最小报价单位如何改变价差与深度", "tick-size-liquidity"),
            ("已实现价差分解：用成交后中间价拆临时成本", "realized-spread-decomposition"),
            ("订单流毒性批量估计：用 BVC 近似 VPIN 的流动性预警", "flow-toxicity-bulk"),
            # ===== 2026-07-26-B 补充：上一批 20 主题已发布，全库 ≥1093 篇，新增真正全新主题 =====
            # 以下为 2026-07-26 全库磁盘扫描确认【未发布】的市场微观结构 / 高频 / 统计检验专题，均可用 numpy/scipy 合成数据演示。
            ("DCCA 去趋势互相关分析：在非平稳序列里量化跨市场的标度联动", "dcca-cross-correlation"),
            ("日内波动率季节性 FFF：用柔性傅里叶形式剖开 U 型日内波动曲线", "intraday-seasonality-fff"),
            ("Ho-Stoll 做市商库存模型：把持仓风险写进最优报价", "ho-stoll-dealer"),
            ("Madhavan-Richardson-Roomans 微观结构模型：拆解成交价里的信息与流动性成分", "madhavan-glosten-model"),
            # 2026-07-29 移除 roll-effective-spread：与已发布 roll-spread-estimator 内容重复
            # ===== 2026-07-26 补充：主池 + 全部旧兜底已全库用尽（1067 篇），新增真正全新主题 =====
            # 以下为 2026-07-26 全库磁盘扫描确认【未发布】的量化专题，均可用 numpy/scipy 合成数据演示。
            ("Almgren-Chriss 最优执行：在冲击成本与时间风险之间找平衡点", "almgren-chriss-optimal-execution"),
            ("Amihud 非流动性因子：用单位成交额的价格冲击给流动性定价", "amihud-illiquidity-factor"),
            ("PIN 知情交易概率：从订单流不平衡里估计谁在偷跑", "pin-informed-trading-probability"),
            ("VPIN 订单流毒性：闪崩前的流动性预警指标", "vpin-flow-toxicity"),
            ("Hasbrouck 信息份额：多市场交易时价格发现发生在哪里", "hasbrouck-information-share"),
            ("融券余额因子：卖空者的持仓是不是聪明钱", "short-interest-factor"),
            ("轧空天数 Days-to-Cover：识别潜在逼空行情的量化信号", "days-to-cover-short-squeeze"),
            ("指数调仓前瞻交易：成份股纳入剔除的可预测资金流", "index-rebalance-front-running"),
            ("封闭式基金折价：情绪代理与均值回归套利", "closed-end-fund-discount"),
            ("并购套利价差：把交易完成概率写进仓位", "merger-arbitrage-spread"),
            ("PEAD 盈余公告后漂移：财报惊喜的慢消化", "pead-post-earnings-drift"),
            ("内部人交易信号：高管增减持的信息含量", "insider-trading-signal"),
            ("13F 机构持仓克隆：抄大佬作业的系统化方法", "institutional-13f-cloning"),
            ("基金申赎资金流压力：被动买卖造成的可预测价格偏离", "mutual-fund-flow-pressure"),
            ("Obizhaeva-Wang 执行模型：把限价簿弹性写进最优拆单", "obizhaeva-wang-execution"),
            ("Pastor-Stambaugh 流动性因子：流动性冲击的系统性溢价", "pastor-stambaugh-liquidity"),
            ("Amihud-Mendelson 价差与收益：买卖价差如何进入资产定价", "amihud-mendelson-spread"),
            ("ETF 资金流套利：申赎机制下的一二级价差", "etf-flow-arbitrage"),
            ("ADR 溢价套利：同一家公司两地上市的价差回归", "adr-premium-arbitrage"),
            ("分拆上市事件驱动：Spinoff 后的错误定价窗口", "spinoff-event-driven"),
            # ===== 2026-07-25 补充：主池 + 全部旧兜底已全库用尽（1041 篇），新增真正全新主题 =====
            # 以下为 2026-07-25 全库扫描确认【未发布】的量化专题，均可用 numpy/scipy 合成真实数据演示。
            ("极值理论 EVT 尾部风险：用广义帕累托分布给『黑天鹅』定价", "extreme-value-theory-evt"),
            ("Hill 估计量与尾部指数：量化收益分布到底有多肥", "hill-estimator-tail-index"),
            ("TWAP/VWAP 最优执行：把大单拆细在时间轴上隐身", "optimal-execution-twap-vwap"),
            ("传播子模型价格冲击：把每笔成交的冲击写成核函数衰减", "propagator-model-price-impact"),
            ("Epps 效应与高频相关性：为什么采样越细相关性越假", "epps-effect-correlation"),
            ("Hayashi-Yoshida 领先滞后估计：在异步 tick 里找谁先动", "lead-lag-hayashi-yoshida"),
            ("CPPI 回撤控制：用动态乘数把组合下行锁进保险箱", "drawdown-control-cppi"),
            ("TIPP 组合保险：把保本线钉在历史最高净值上", "tipp-portfolio-insurance"),
            ("Omega 比率优化：用整条收益分布替代均值方差", "omega-ratio-optimization"),
            ("溃疡指数与痛苦指数：把回撤的深度和时长一起计入风险", "pain-index-ulcer"),
            ("熵池化观点融合：用相对熵把主观观点揉进先验分布", "entropy-pooling-views"),
            ("Wasserstein 鲁棒组合优化：让权重对分布扰动免疫", "robust-optimization-wasserstein"),
            ("换手率约束组合优化：把调仓成本写进目标函数", "turnover-constrained-optimization"),
            ("日内动量最后一小时效应：收盘前的方向延续从哪来", "intraday-momentum-last-hour"),
            ("隔夜与日内收益拆分：同一只股票的两副面孔", "overnight-intraday-return-split"),
            ("月末月初效应：日历翻页时的资金流规律", "turn-of-month-effect"),
            ("VIX 期限结构 Carry：在 contango 里滚动收割升水", "vix-term-structure-carry"),
            ("方差风险溢价择时：用隐含减已实现的差做风险开关", "variance-risk-premium-timing"),
            ("偏度风险溢价：卖出崩盘保险的系统化收益", "skew-risk-premium"),
            ("财报电话会语调分析：用词频把管理层情绪变成因子", "earnings-call-tone-analysis"),
            # ===== 2026-07-24 补充：主池 + 全部旧兜底已全库用尽（1004 篇），新增真正全新主题 =====
            # 以下为 2026-07-24 全库扫描确认【未发布】的量化专题，均为 numpy/scipy 可合成真实数据的方向。
            ("门限自回归 TAR 非线性建模：用状态依赖的回归捕捉市场不对称反应", "threshold-autoregression-tar"),
            ("平滑转移自回归 STAR：用连续过渡函数给 regime 加一条软边界", "smooth-transition-star"),
            ("马尔可夫区制转移 VAR：让宏观变量在牛熊状态间自行切换", "markov-switching-var"),
            ("向量误差修正模型 VECM：把协整关系写成向长期均衡的拉力", "vecm-cointegration"),
            ("Johansen 协整检验：用迹统计量与最大特征值锁定多变量长期关系", "johansen-cointegration-test"),
            ("非线性 Granger 因果：用神经网络把『谁预测谁』的单向箭头放宽", "nonlinear-granger-causality"),
            ("谱密度估计与周期分析：用傅里叶把隐藏的周期从噪声里拎出来", "spectral-density-estimation"),
            ("小波阈值去噪：用多尺度分解把高频噪声和真实跳变分开", "wavelet-threshold-denoise"),
            ("稳健协方差 MCD 估计：用最小协方差行列式抗住极端值污染", "robust-covariance-mcd"),
            ("独立成分分析 ICA：把混合信号拆回相互独立的源", "ica-financial-signal"),
            ("非负矩阵分解 NMF：把资产收益矩阵压成可解释的基与权重", "nmf-asset-decomposition"),
            ("流形学习资产表征：用降维把高维因子铺成可看的地图", "manifold-learning-assets"),
            ("高斯混合聚类资产：用软分配把股票分成潜在风格群", "gmm-asset-clustering"),
            ("孤立森林异常检测：用随机孤立路径给极端行情打分", "isolation-forest-anomaly"),
            ("One-Class SVM 异常：用高维边界圈出『不正常』的交易日", "oneclass-svm-anomaly"),
            ("词嵌入金融文本：用向量空间把公告语义变成可计算因子", "word-embedding-finance"),
            ("命名实体识别金融事件：从新闻里自动抽出公司-事件-时间", "ner-financial-events"),
            ("SHAP 因子解释：用博弈论份额拆开每个特征对预测的贡献", "shap-factor-explanation"),
            ("局部依赖图 PDP：把单个因子与收益的边际关系画成曲线", "partial-dependence-plot"),
            ("累积局部效应 ALE：用条件期望修正相关因子下的边际效应", "accumulated-local-effects"),
            ("向量化回测引擎：用 NumPy 广播把万次模拟压进一次矩阵运算", "vectorized-backtest-engine"),
            ("贝叶斯优化超参：用高斯过程代理把调参成本砍到最低", "bayesian-optimization-hyperparam"),
            ("遗传编程因子挖掘：让表达式树自己进化出可交易 alpha", "genetic-programming-factor"),
            ("PPO 强化学习交易：用近端策略优化学会在连续动作空间下单", "ppo-trading-agent"),
            ("元学习量化策略：用 MAML 让模型在少样本上快速适应新市场", "meta-learning-quant"),
            ("图注意力金融网络：用注意力权重重新分配板块间的信息流", "graph-attention-finance"),
            ("路径签名粗糙路径：用迭代积分把价格轨迹写成可学习的特征", "path-signature-rough"),
            ("多重分形 DFA：用多重分形谱刻画不同尺度下的记忆结构", "multifractal-dfa-analysis"),
            ("已实现协方差矩阵：用高频数据拼出盘中多资产联动全景", "realized-covariance-matrix"),
            ("三角套利加密货币：用三个交易对的价差吃掉无风险利润", "triangular-arb-crypto"),
            # ===== 2026-07-23 补充（历史，已基本发布）=====
            ("Vine Copula 依赖建模：用层级藤把高维尾部联动拆成条件二元对", "vine-copula-dependence"),
            ("RNN/LSTM 金融时序预测：用门控记忆捕捉长程依赖", "rnn-financial-trading"),
            ("TCN 时序卷积交易：用膨胀因果卷积替代 RNN", "temporal-conv-trading"),
            ("Neural ODE 连续时间建模：把隐藏状态写成微分方程", "neural-ode-financial"),
            ("分位数回归森林：给出收益分布而非单点预测", "quantile-regression-forest"),
            ("注意力机制金融情绪：用 Transformer 读新闻标题", "attention-omics-sentiment"),
            ("谱图神经网络：把关联矩阵当成拉普拉斯信号", "spectral-gnn-financial"),
            ("因果卷积金融预测：用膨胀卷积抓长程周期", "causal-conv-financial"),
            ("Informer 长序列预测：用概率稀疏注意力砍复杂度", "informer-financial"),
            ("Autoformer 时序分解预测：用序列分解+自相关机制", "autoformer-financial"),
            ("Crossformer 跨变量预测：用维度分段注意力建模变量交互", "crossformer-financial"),
            ("图小波网络金融关联：把图卷积换成图小波", "graph-wave-net-financial"),
            ("深度状态空间交易：用线性 RNN 做高效长程建模", "deep-state-space-trading"),
            ("置换不变集合建模：用 DeepSets 处理无序资产集合", "permutation-invariant-set"),
            ("GARCH-EMD 混合波动：用经验模态分解剥离趋势再做 GARCH", "garch-emd-hybrid"),
            ("得分匹配生成金融序列：用去噪分数匹配合成行情", "score-matching-financial"),
            ("Set Transformer 金融表征：用诱导点注意力聚合多资产", "set-transformer-financial"),
            ("分位数 RNN 风险预测：用分位损失给出收益区间", "qrnn-quantile-forecast"),
            ("离线强化学习交易：用 CQL 保守 Q 学习做策略迁移", "cql-offline-rl-trading"),
            ("共形强化学习交易：给 RL 动作一个可验证的覆盖保证", "conformal-rl-trading"),
            # ===== 以下为历史兜底（已基本全部发布，保留以防个别未占用）=====
            ("距离相关系数:用 Hilbert-Schmidt 独立性把非线性依赖从噪声里捞出来", "distance-correlation-dependence"),
            ("排列熵与复杂度:用序数模式把『混沌』和『随机』一刀切开", "permutation-entropy-complexity"),
            ("DCov 距离协方差:用特征映射的核把依赖关系写成距离", "dcov-nonlinear-dependence"),
            ("序数模式熵:用涨跌排列的『形状』识别市场状态切换", "ordinal-pattern-entropy"),
            ("多尺度熵与 DFA:用粗粒化把不同时间尺度的不规则性分开", "multiscale-entropy-dfa"),
            ("可转债定价：把转股权、回售与赎回装进一棵二叉树", "convertible-bond-pricing"),
            ("信用违约互换 CDS 估值与信用曲线 bootstrap", "cds-valuation-credit-curve"),
            ("通胀掉期与盈亏平衡通胀率：用互换价差拆出通胀预期", "inflation-swap-breakeven"),
            ("利率互换 IRS 定价：用折现曲线做 bootstrap", "interest-rate-swap-pricing"),
            ("国债期货最便宜可交割券 CTD 与基差交易", "bond-futures-ctd-basis"),
            ("巴黎期权与部分触及障碍：给障碍加一段观察窗", "parisian-barrier-options"),
            ("二元/数字期权定价与风险：用现金流复制 exotic payoff", "binary-digital-options"),
            ("碳价格因子：把气候风险写进可交易 alpha", "carbon-price-factor"),
            ("供应链集中度因子：用客户/供应商依赖量化基本面风险", "supply-chain-concentration-factor"),
            ("专利质量因子：用发明专利授权构建创新 alpha", "patent-quality-factor"),
            ("地缘政治风险指数 GPR 与资产定价", "geopolitical-risk-pricing"),
            ("经济政策不确定性 EPU 与股债配置切换", "epu-asset-allocation"),
            ("谷歌趋势作为另类情绪信号：用搜索量预测短期波动", "google-trends-sentiment"),
            ("比特币链上 MVRV 与 NUPL：用链上成本基础做择时", "btc-onchain-mvrv-nupl"),
            ("稳定币供给与加密市场流动性领先指标", "stablecoin-supply-crypto-liquidity"),
            ("加密永续合约资金费率均值回归", "crypto-funding-rate-reversion"),
            ("加密横截面动量因子", "crypto-cross-section-momentum"),
            ("链上 SOPR 与获利盘抛压信号", "onchain-sopr-profit"),
            ("加密市场波动率指数 DVOL 构建", "crypto-volatility-index"),
            ("下跌 Beta 与危机 Alpha 的多空构建", "downside-beta-crisis-alpha"),
            ("特质波动率异象与套利限制", "idiosyncratic-volatility-anomaly"),
            ("博彩偏好与最大日收益因子 MAX", "lottery-max-factor"),
            ("分析师评级修正动量", "analyst-revision-momentum"),
            ("盈余预告惊喜与漂移", "earnings-guidance-drift"),
            ("回购强度因子", "buyback-intensity-factor"),
            ("资本支出效率因子", "capex-efficiency-factor"),
            ("Piotroski F-score 财务健康打分", "piotroski-fscore"),
            ("Ohlson O-score 破产概率建模", "ohlson-oscore-distress"),
            ("杜邦分析 ROE 拆解因子", "dupont-roe-decomposition"),
            ("现金转化周期因子", "cash-conversion-cycle"),
            ("保护性看跌 overlay 与组合下行保护", "protective-put-overlay"),
            ("领口策略 collar 与成本锁定", "collar-cost-lock"),
            ("蝶式/鹰式价差波动率交易", "butterfly-condor-vol-trading"),
            ("日历价差 theta 收割", "calendar-spread-theta"),
            ("系统性风险 SRISK 与边际期望损失 MES", "srisk-mes-systemic"),
            ("网络中心性与系统性重要机构识别", "network-centrality-sifis"),
            ("买卖价差 Corwin-Schultz 估计", "corwin-schultz-spread"),
            ("实现核与预平均去噪：高频波动率估计", "realized-kernel-preaveraging"),
            ("Kyle lambda 与信息含量度量", "kyle-lambda-information"),
            ("限价订单簿队列位置价值", "lob-queue-position-value"),
            ("贝叶斯结构断点检测", "bayesian-structural-break"),
            ("时变参数 TVP-VAR 模型", "tvp-var-time-varying"),
            ("方差比检验市场有效性", "variance-ratio-test"),
            ("BDS 检验非线性依赖", "bds-test-nonlinearity"),
            ("经验模态分解 EMD 趋势提取", "emd-trend-extraction"),
            ("变分模态分解 VMD", "vmd-decomposition"),
            ("奇异谱分析 SSA 周期提取", "singular-spectrum-ssa"),
            ("变分自编码器 VAE 因子挖掘", "vae-factor-mining"),
            ("GAN 生成金融序列与数据增强", "gan-financial-series"),
            ("高斯过程回归在金融预测中的应用", "gaussian-process-finance"),
            ("对比学习金融表征", "contrastive-learning-finance"),
            ("图神经网络 GNN 资产关系建模", "gnn-asset-relations"),
            ("PatchTST 时序分块预测", "patchtst-time-series"),
            ("TimesNet 周期建模", "timesnet-periodic"),
            ("状态空间模型 Mamba 金融时序", "mamba-state-space"),
            ("双重差分 DID 在金融事件评估", "causal-did-finance"),
            ("合成控制法评估政策冲击", "synthetic-control-method"),
            ("双重机器学习 Double ML 因果估计", "double-ml-causal"),
            ("Carhart 四因子模型实战拆解", "carhart-four-factor"),
            ("Stambaugh-Yuan 误定价因子", "stambaugh-yuan-mispricing"),
            ("基本面指数加权 vs 市值加权", "fundamental-index-weighting"),
            ("核心卫星策略配置框架", "core-satellite-strategy"),
            ("桥水全天候风险平价", "all-weather-risk-parity"),
            ("永久组合 Harry Browne 配置", "permanent-portfolio-browne"),
            ("耶鲁捐赠基金模式借鉴", "yale-endowment-model"),
            ("黄金作为尾部对冲资产", "gold-tail-hedge"),
            ("CTA 多品种趋势跟踪", "cta-multi-trend"),
            ("期现套利与基差交易", "cash-futures-basis-arb"),
            ("随机指标 KDJ 实战", "kdj-indicator"),
            ("商品通道指数 CCI", "cci-indicator"),
            ("OBV 能量潮指标", "obv-on-balance-volume"),
            ("Roll 协方差度量买卖成本", "roll-covariance-measure"),
            ("少数者博弈金融市场模型", "minority-game-model"),
            ("伊辛模型金融临界性", "ising-model-finance"),
            ("小波相干与跨资产联动", "wavelet-coherence"),
            ("Stacking 集成因子模型", "stacking-factor-model"),
            ("逆强化学习策略偏好推断", "inverse-rl-preference"),
            # ===== 2026-07-29B 补充：全库 1210 篇磁盘扫描确认以下 slug 均未发布（波动率衍生品/高频估计专题）=====
            ("方差互换复制：用一篮子期权静态复制已实现方差", "variance-swap-replication"),
            ("SABR 随机波动率模型：把波动率微笑写进四个参数", "sabr-volatility-model"),
            ("Gamma Scalping 损益拆解：对冲频率、已实现波动与期权时间价值的博弈", "gamma-scalping-pnl"),
            ("粗糙波动率 rFSV：Hurst 指数远小于 0.5 的波动率新范式", "rough-volatility-rfsv"),
            ("VIX 期货基差交易：期限结构里的滚动收益与崩盘风险", "vix-futures-basis-trading"),
            ("Vanna-Volga 定价：用三个希腊字母修正 BS 的微笑误差", "vanna-volga-pricing"),
            ("高频已实现 Beta：用日内数据估计时变系统性暴露", "realized-beta-highfreq"),
            ("期权隐含 Beta：从期权价格反推前瞻性市场暴露", "option-implied-beta"),
            ("走廊方差互换：只对价格区间内的波动付费", "corridor-variance-swap"),
            ("Tick Rule 与成交量同步概率：用逐笔方向重构知情交易强度", "tick-rule-volume-synchronized"),
            # ===== 2026-07-29 补充：全库 1193 篇磁盘扫描确认以下 slug 均未发布 =====
            ("GEX 伽马敞口：做市商对冲流如何压制与放大波动", "gamma-exposure-gex"),
            ("北向资金流因子：陆股通资金能否预测 A 股收益", "northbound-flow-factor"),
            ("零日期权 0DTE 对市场波动结构的冲击", "zero-dte-options-impact"),
            ("最大痛点理论与期权到期钉盘效应", "max-pain-option-pinning"),
            ("指数纳入效应：被动资金冲击下的价格压力与反转", "index-inclusion-effect"),
            ("融券费率信号：借券成本里的负面信息", "securities-lending-fee-signal"),
            ("涨跌停磁吸效应：价格接近极限时的加速动力学", "limit-up-down-magnet"),
            ("龙虎榜席位因子：营业部数据的信息含量", "dragon-tiger-list-factor"),
            ("IPO 抑价异象：新股首日收益的系统性规律", "ipo-anomaly-underpricing"),
            ("回购公告 Alpha：公司回购事件后的漂移", "buyback-announcement-alpha"),
            ("资产增长异象：扩张越快未来收益越差", "asset-growth-anomaly"),
            ("股票拆分效应：名义价格幻觉与流动性变化", "stock-split-effect"),
            ("收盘集合竞价动力学：尾盘流动性与价格发现", "closing-auction-dynamics"),
            ("碎股交易的信息含量：odd lot 里的知情交易", "odd-lot-trades-information"),
            ("订单流付款 PFOF：零佣金背后的执行质量博弈", "pfof-order-flow-payment"),
            # ===== 2026-07-30 补充：全库 1260 篇磁盘扫描确认以下 slug 均未发布 =====
            ("隐含相关性交易：指数波动率与成分股波动率的价差", "implied-correlation-dispersion"),
            ("久期中性国债期货套利：跨期限曲线交易实战", "duration-neutral-bond-futures"),
            ("宏观因子映射：把资产收益拆解到增长与通胀两轴", "macro-factor-mapping-growth-inflation"),
            ("跨式期权盈亏平衡带：事件驱动波动率多头的成本核算", "straddle-breakeven-event-vol"),
            ("拥挤度指标：用持仓相关性预警因子踩踏", "factor-crowding-metric"),
            ("回购利率与股指期货基差：资金成本如何定价升贴水", "repo-rate-index-futures-basis"),
            ("分层抽样再平衡：降低指数复制的跟踪误差", "stratified-sampling-index-replication"),
            ("日内均值回归的时间衰减：开盘半小时与尾盘的不同世界", "intraday-mean-reversion-decay"),
            # ===== 2026-07-31 补充：全库 1270 篇磁盘扫描确认以下 slug 均未发布 =====
            ("季末粉饰橱窗效应：基金排名压力如何扭曲最后三天的价格", "window-dressing-quarter-end"),
            ("国债招标周期效应：一级市场供给冲击的可交易节奏", "treasury-auction-cycle-effect"),
            ("三巫日到期：股指期货、期权与个股期权同日结算的流动性异常", "futures-expiry-triple-witching"),
            ("可转债双低策略：价格与溢价率的二维打分实战", "convertible-bond-double-low"),
            ("陆股通假期缺口：内外盘休市错位带来的跳空风险", "stock-connect-holiday-gap"),
            ("股息捕获策略：除权日前后的价格调整能不能吃到", "dividend-capture-strategy"),
            ("行业轮动与经济周期：用增长-通胀象限做板块配置", "sector-rotation-business-cycle"),
            ("换手率衰减与 Alpha 半衰期：信号该持有多久", "turnover-decay-alpha-halflife"),
            ("订单簿斜率信号：用挂单密度预测短期价格方向", "order-imbalance-slope-signal"),
            # ===== 2026-08-01 补充：全库 1307 篇磁盘扫描逐一确认以下 slug 均未发布（微观结构执行/事件套利/仓位管理专题）=====
            ("限价单排队位置：同样的价格，为什么你总是最后成交", "queue-position-limit-order"),
            ("Microprice 微观价格：用挂单量加权修正中间价的方向偏差", "midprice-microprice-estimator"),
            ("Hawkes 自激过程：订单到达如何自己点燃自己", "hawkes-process-order-arrival"),
            ("VPIN 知情交易概率：用成交量桶量化逆向选择风险", "adverse-selection-vpin"),
            ("执行落差归因：把 Implementation Shortfall 拆成延迟、冲击与择时", "implementation-shortfall-attribution"),
            ("并购套利价差模型：用成交概率给折价定价", "merger-arb-spread-model"),
            ("分拆上市事件 Alpha：母子公司拆分后的价格漂移", "spinoff-event-alpha"),
            ("可转债 Delta 对冲套利：把转股期权从债底里剥出来", "convertible-arb-delta-hedge"),
            ("轧空与 Gamma 正反馈环：逼空行情的自我强化机制", "short-squeeze-gamma-loop"),
            ("延迟套利与主机托管：微秒级速度差到底值多少钱", "latency-arbitrage-colocation"),
            ("报价填塞检测：用撤单率识别扰乱性高频行为", "quote-stuffing-detection"),
            ("穿价规则与全国最优报价：订单路由的合规约束成本", "trade-through-rule-nbbo"),
            ("除息套利与税收效应：股息落差为什么不等于股息", "ex-dividend-arbitrage-tax"),
            ("Copula 尾部配对交易：相关性在极端行情下会断裂", "pairs-trading-copula-tail"),
            ("HMM 状态切换资产配置：让模型自己识别牛熊震荡", "regime-switching-hmm-allocation"),
            ("波动率目标风险平价：把组合波动锁在固定区间", "risk-parity-vol-targeting"),
            ("分数 Kelly 仓位：为什么全 Kelly 是理论最优却是实战自杀", "kelly-fractional-sizing"),
            ("回撤控制覆盖层：在原策略之上加一层自动减仓", "drawdown-control-overlay"),
            # ===== 2026-08-07 补充：全库 1344 篇磁盘扫描逐一确认以下 slug 均未发布（固收/衍生品/信用/另类专题）=====
            ("国债基差交易：对冲基金 8000 亿美元头寸的脆弱性", "basis-trade-treasury-hedge-fund"),
            ("互换利差为负：无风险利率反而高于国债收益率的十年异象", "swap-spread-negative-anomaly"),
            ("跨货币基差套利：抛补利率平价失效后的美元荒定价", "cross-currency-basis-arbitrage"),
            ("折扣率顺周期性：风控参数如何在危机中自我强化", "collateral-haircut-procyclicality"),
            ("ETF 授权参与人套利：一级市场申赎如何锚定二级价格", "etf-authorized-participant-arbitrage"),
            ("债券 ETF 的流动性转换：日内可交易的底层其实一周才成交一次", "bond-etf-liquidity-transformation"),
            ("波动率 Carry：VIX 期货展期收益的来源与崩塌", "volatility-carry-vix-futures-roll"),
            ("方差风险溢价收割：卖波动为什么长期赚钱又偶尔归零", "variance-risk-premium-harvest"),
            ("尾部对冲的成本拖累：保险费到底吃掉多少长期收益", "tail-hedge-cost-drag"),
            ("CPPI 组合保险：固定比例机制与 1987 年的教训", "portfolio-insurance-cppi"),
            ("展期收益与期货结构：Contango 和 Backwardation 的真实成本", "commodity-roll-yield-contango-backwardation"),
            ("通胀盈亏平衡交易：TIPS 与名义债的价差里有多少流动性噪音", "inflation-breakeven-trade"),
            ("收益率曲线蝶式交易：中段凸性的均值回归", "yield-curve-butterfly-trade"),
            ("远期汇率偏差与 Carry Trade：UIP 失效的持久性", "forward-rate-bias-carry-trade"),
            ("主权 CDS 基差：违约保护为什么比债券便宜", "sovereign-cds-basis"),
            ("信用利差分解：违约补偿、流动性溢价与风险厌恶各占多少", "credit-spread-decomposition-default-liquidity"),
            ("Merton 违约距离模型：用股价反推信用风险", "merton-distance-to-default"),
            ("堕落天使的强制抛售：评级下调触发的机械性卖压", "fallen-angel-forced-selling"),
            ("评级迁移矩阵建模：信用组合的多期损失分布", "rating-migration-matrix-modeling"),
            ("私募信贷的估值平滑：低波动是真的还是没有报价", "private-credit-mark-smoothing"),
            ("陈旧定价与自相关平滑：另类资产 Sharpe 的系统性虚高", "stale-pricing-autocorrelation-smoothing"),
            ("业绩报酬的期权价值：2/20 结构对基金经理的风险激励", "performance-fee-option-value"),
            ("高水位线的激励扭曲：深水基金为什么会赌一把", "high-water-mark-incentive-distortion"),
            ("资金流与业绩反馈环：申赎如何反过来杀死策略", "fund-flow-performance-feedback"),
            ("暗中指数化与 Active Share：你付的主动管理费买到了什么", "closet-indexing-active-share"),
            # ===== 2026-08-28 补充（第八次后全库 1408+ 篇，主池+全部旧兜底已用尽）：新增真正全新主题，磁盘扫描逐一确认 slug 未发布 =====
            ("粒子滤波随机波动率实时校准：用序贯蒙特卡洛在线更新 SV 模型参数", "particle-filter-sv-calibration"),
            ("图神经网络因子扩散：用 GCN 在板块关联图上传播收益信号", "gnn-factor-diffusion-gcn"),
            ("向量误差修正与卡尔曼滤波融合：把协整关系写进状态空间", "vecm-kalman-state-space-fusion"),
            ("对抗训练鲁棒因子：用 FGSM/PGD 给预测模型加噪声免疫力", "adversarial-robust-factor-training"),
            ("分位数回归森林预测条件收益分布：不只点预测还要分位", "quantile-regression-forest-forecast"),
            ("隐马尔可夫做市商库存控制：用 POMDP 平衡存货与价差", "hmm-market-maker-inventory"),
            ("多尺度波动率分解：用最大重叠离散小波把 GARCH 残差拆到尺度", "multiresolution-volatility-modwt"),
            ("傅里叶周期分解择时：把价格拆成可交易的正弦分量", "fourier-periodic-decomposition-timing"),
            ("经验模态分解去趋势：用 EMD 把趋势与周期自适应剥离", "emd-trend-decomposition"),
            ("高斯过程回归波动率曲面：用非参数方法插值期限结构", "gaussian-process-vol-surface"),
            ("深度均衡模型时序预测：用不动点迭代替代层层堆叠", "deep-equilibrium-model-forecast"),
            ("强化学习最优执行：用策略梯度把大单拆成低冲击序列", "rl-optimal-execution-twap"),
            ("稀疏主成分因子模型：用 L1 约束逼出可解释载荷", "sparse-pca-factor-model"),
            ("贝叶斯收缩因子组合：用层次先验把极端权重拉回均值", "bayesian-shrinkage-factor-portfolio"),
            ("已实现核与微观结构噪声：用子采样估计无偏积分波动率", "realized-kernel-microstructure-noise"),
            ("小波相干性板块联动：用交叉小波看板块共振的时频结构", "wavelet-coherence-sector-comovement"),
            # ===== 2026-08-31 补充：主池 + 全部旧兜底 + 旧 fresh_pool 已全库用尽（1438 篇），新增真正全新主题 =====
            # 以下为 2026-08-31 全库磁盘扫描逐一确认 slug 未发布（微观结构/组合优化/套利/机器学习/市场微观结构专题）
            # 顺序即优先级：前 2 个为本次自动发布选定主题。
            ("限价单簿隐马尔可夫建模：用 HMM 把盘口状态分成安静与挤兑", "limit-order-book-hmm"),
            ("鲁棒均值方差优化：让组合对期望收益估计误差免疫", "robust-mean-variance-optimization"),
            ("不平衡 Bar 信息驱动采样：让每根 K 线携带等量信息而非等量时间", "imbalance-bars-information-sampling"),
            ("贝叶斯期权定价 MCMC：用马尔可夫链蒙特卡洛给奇异期权定价", "bayesian-option-pricing-mcmc"),
            ("跳跃扩散神经网络定价：用深度学习逼近含跳资产的价格", "jump-diffusion-neural-pricing"),
            ("最小方差组合解析解：把全局最小风险权重闭式写出来", "minimum-variance-analytical"),
            ("风险平价在中国 A 股：换手与约束下的实战再检验", "risk-parity-a-shares-practice"),
            ("截面动量衰减曲线：因子信号逐日腐烂的实证与对抗", "momentum-decay-cross-section"),
            ("波动率目标择时股债配置：用已实现波动开关风险敞口", "vol-target-timing-equity-bond"),
            ("ETF 折溢价套利：一二级市场价差的可交易窗口", "etf-premium-discount-arb"),
            ("美股隔夜异象：收盘到开盘跳变能否预测次日方向", "us-overnight-anomaly"),
            ("期权 Vega 风险：波动率敞口如何被忽略又如何对冲", "option-vega-risk-management"),
            ("跨资产相关性危机：恐慌时相关性冲向 1 的组合含义", "cross-asset-correlation-crisis"),
            ("流动性螺旋与去杠杆：抛售如何自我强化成危机", "liquidity-spiral-deleveraging"),
            ("强化学习组合配置：用策略梯度学出动态再平衡", "rl-portfolio-allocation"),
            ("协整配对 Kalman 融合：用状态空间模型动态追踪对冲比例", "kalman-pairs-cointegration-fusion"),
            ("图神经网络板块关联：在行业关系图上传播收益信号", "gnn-sector-correlation-network"),
            ("多智能体市场仿真：用 Agent 重演限价簿价格发现", "multi-agent-market-simulation-lob"),
            ("订单流回归 Kyle 连续拍卖：把知情者最优下单速度解出来", "kyle-continuous-auction-flow"),
            ("做市商深度强化学习库存控制：用 POMDP 平衡存货与价差", "market-maker-deep-rl-inventory"),
            ("波动率曲面 Butterfly 风险：中段凸性的均值回归交易", "vol-surface-butterfly-risk"),
            ("生存分析高频违约预测：用 Cox 比例风险建模跳变", "survival-analysis-highfreq-default"),
            ("贝叶斯结构断点检测：用变点把行情的变脸时刻找出来", "bayesian-structural-breakpoint"),
            ("时序卷积因果推断：用膨胀卷积预测滞后传导", "temporal-conv-causal-inference"),
            ("跨资产宏观动量再检验：股债商品外汇趋势的合成信号", "cross-asset-macro-momentum-revisit"),
            ("因子投资 Smart Beta 红利低波：用质量与分红重构宽基", "smart-beta-dividend-lowvol"),
            # ===== 2026-10-26 补充：主池 + 全部旧兜底 + 旧 fresh_pool 已全库用尽（1469 篇），新增真正全新主题 =====
            # 以下为 2026-10-26 全库磁盘扫描逐一确认 slug 未发布的方向（机器学习前沿 / 因果推断 / 拓扑 / 量子 / 最优传输 / 扩散模型 / 神经微分方程 / 对比学习 / MCTS / 分布鲁棒 / 平均场博弈等）
            # 顺序即优先级：前 2 个为本次自动发布选定主题。
            ("随机投影与压缩感知因子降维：用 Johnson-Legendre 定理把高维因子映射到低维", "random-projection-factor-reduction"),
            ("信息瓶颈在序列预测中的应用：压缩预测什么，什么就能被预测", "information-bottleneck-sequence-prediction"),
            ("因果发现在因子选择中的 PC 算法：从条件独立推断因子因果图", "pc-algorithm-causal-factor-discovery"),
            ("核密度估计的条件概率：用 KDE 替代直方图做 regime 概率映射", "kernel-density-conditional-probability"),
            ("高斯混合模型 regime 分类：用 EM 算法把市场分成多态", "gaussian-mixture-regime-classification"),
            ("隐马尔可夫模型市场状态：用 Viterbi 解码最可能的情境路径", "hidden-markov-market-regime"),
            ("变点检测在波动率跳跃中的应用：用 CUSUM 或 PELT 捕捉 regime 切换", "change-point-detection-volatility-jump"),
            ("贝叶斯模型平均预测集成：用后验权重替代简单的等权平均", "bayesian-model-averaging-ensemble"),
            ("量子计算在期权定价中的应用：用 QAMC 实现维度指数加速", "quantum-computing-option-pricing"),
            ("拓扑数据分析持续同调：用持久子图捕捉高维数据的拓扑特征", "topological-data-analysis-persistence"),
            ("最优传输在组合优化中的应用：用 Wasserstein 距离做分布鲁棒", "optimal-transport-portfolio-optimization"),
            ("扩散模型生成市场路径：用 DDPM 合成符合历史分布的价格轨迹", "diffusion-model-market-path-generation"),
            ("神经微分方程时序建模：用 ODE 求解器拟合连续时间动态", "neural-ode-time-series"),
            ("对比学习自监督预训练：用 SimCLR 把时序片段映射到语义空间", "contrastive-self-supervised-pretraining"),
            ("蒙特卡洛树搜索组合优化：用 UCB1 在指数级组合中剪枝", "mcts-portfolio-optimization"),
            ("分布鲁棒优化：用 Wasserstein 球定义最坏情况分布族", "distributionally-robust-optimization"),
            ("随机控制最优执行：用 Hamilton-Jacobi-Bellman 方程解最优拆单", "stochastic-control-optimal-execution"),
            ("平均场博弈多代理人：用 N 人博弈的对称均衡解做做市策略", "mean-field-game-multiaagent"),
        ]
        # 2026-07-27 修复：不再清空 selected（旧代码 `selected = []` 会丢掉前面已选中的主题，
        # 导致前面剩 1 个 + 本池剩 1 个时总数仍凑不齐 2 个而 assert 失败）。保留已选，仅补足到 2 个。
        for title, slug in fresh_pool:
            if len(selected) >= 2:
                break
            if slug in existing_slugs:      # 真在磁盘已发布 -> 跳过(关键修复)
                continue
            if any(slug == s for _, s, _ in selected):
                continue
            if any(slug == s for _, s, _ in selected):
                continue
            selected.append((title, slug, "runtime-free-scan"))

    # 保证最终恰好 2 个，且 slug 互不相同
    selected = selected[:2]
    assert len(selected) == 2, f"未能选出2个主题，仅 {len(selected)} 个"
    assert selected[0][1] != selected[1][1], "选出的两个主题 slug 重复！"

    print(f"\n✅ 选择的主题（已排除全部已有文章，共 {len(existing_slugs)} 篇）：")
    for i, (title, slug, _) in enumerate(selected, 1):
        print(f"  {i}. {title} (slug: {slug})")

    # 输出供AI读取
    print("\n" + "="*50)
    for title, slug, _ in selected:
        print(f"{title}|{slug}")

    return [(t, s) for t, s, _ in selected]

if __name__ == "__main__":
    select_topics()
