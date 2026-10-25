/**
 * 股票网址导航数据
 * 所有链接均经过存活状态实测（2026-10），403/412 属站点反爬但域名有效
 */

export interface StockLink {
  name: string
  url: string
  desc: string
  /** 访问条件标签：免费 / 开源免费 / 需开户 / 付费 等 */
  badge?: string
}

export interface StockLinkCategory {
  id: string
  name: string
  emoji: string
  desc: string
  links: StockLink[]
}

export const stockLinkCategories: StockLinkCategory[] = [
  {
    id: 'quotes',
    name: '行情与交易',
    emoji: '📈',
    desc: '盯盘、报价、K 线与交易入口，国内外主要市场覆盖',
    links: [
      { name: '东方财富网', url: 'https://www.eastmoney.com', desc: 'A 股 / 港股 / 美股行情、财报、资金流向一站式' },
      { name: '东方财富行情', url: 'https://quote.eastmoney.com', desc: '实时报价、K 线与盘口数据' },
      { name: '同花顺', url: 'https://www.10jqka.com.cn', desc: '行情 + 财报 + 问财选股的综合平台' },
      { name: '同花顺量化', url: 'https://quant.10jqka.com.cn', desc: '免费在线量化投研平台，含策略回测' },
      { name: '问财选股', url: 'https://www.iwencai.com', desc: '自然语言选股，如「近三年 ROE>20 且毛利率>40%」' },
      { name: '恒生聚源', url: 'https://www.gildata.com', badge: '付费', desc: '金融数据库与量化数据服务' },
      { name: '腾讯自选股', url: 'https://gu.qq.com', desc: '轻量行情与盯盘提醒' },
      { name: '和讯网', url: 'https://www.hexun.com', desc: '老牌财经门户，全球指数与外盘一览' },
      { name: '上海证券交易所', url: 'https://www.sse.com.cn', desc: '沪市规则、公告与市场统计' },
      { name: '深圳证券交易所', url: 'https://www.szse.cn', desc: '深市规则、上市公司与市场数据' },
      { name: '香港交易所', url: 'https://www.hkex.com.hk', desc: '港股行情、上市与交易规则' },
      { name: 'Nasdaq', url: 'https://www.nasdaq.com', desc: '美股行情、IPO 与 ETF 数据' },
      { name: 'NYSE', url: 'https://www.nyse.com', desc: '纽交所官方交易与公司信息' },
      { name: 'TradingView', url: 'https://www.tradingview.com', desc: '专业技术图表，内置 Pine Script 策略脚本' },
      { name: 'Investing.com 中文', url: 'https://cn.investing.com', desc: '全球品种行情、经济日历与财经新闻' },
      { name: 'Yahoo Finance', url: 'https://finance.yahoo.com', desc: '全球行情、财报与历史数据下载' },
      { name: 'Google Finance', url: 'https://www.google.com/finance', desc: '多市场行情快速对比' },
      { name: '富途牛牛', url: 'https://www.futunn.com', badge: '需开户', desc: '港美股行情与交易，社区活跃' },
      { name: '长桥 Longbridge', url: 'https://www.longbridge.com', badge: '需开户', desc: '港美股券商与行情，量化 API 友好' },
      { name: '老虎证券', url: 'https://www.itiger.com', badge: '需开户', desc: '港美股交易与期权平台' },
      { name: '盈透证券 IBKR', url: 'https://www.interactivebrokers.com', badge: '需开户', desc: '全球专业券商，为量化交易设计' }
    ]
  },
  {
    id: 'data',
    name: '数据与研究平台',
    emoji: '🗂',
    desc: '行情数据接口、量化研究平台与宏观数据库',
    links: [
      { name: 'Tushare Pro', url: 'https://tushare.pro', badge: '需积分', desc: '金融数据接口，A 股数据最全的免费方案之一' },
      { name: 'AKShare', url: 'https://akshare.akfamily.xyz', badge: '完全免费', desc: '开源财经数据接口，一行代码拉全市场数据' },
      { name: '聚宽 JoinQuant', url: 'https://www.joinquant.com', badge: '免费', desc: '在线量化研究平台，社区活跃、教程丰富' },
      { name: '米筐 Ricequant', url: 'https://www.ricequant.com', badge: '免费', desc: '量化投研平台，内置多风格因子库' },
      { name: 'BigQuant', url: 'https://bigquant.com', badge: '免费', desc: '无代码机器学习选股与策略平台' },
      { name: '优矿 Uqer', url: 'https://uqer.datayes.com', badge: '免费', desc: '通联数据的量化研究平台' },
      { name: '萝卜投研', url: 'https://robo.datayes.com', desc: '智能投研、因子挖掘与基本面数据' },
      { name: '理杏仁', url: 'https://www.lixinger.com', badge: '免费', desc: '估值分位数与财务数据可视化，做择时判断' },
      { name: '慧博投研', url: 'https://www.hibor.com.cn', badge: '免费', desc: '中文研报与数据聚合平台' },
      { name: 'Alpha Vantage', url: 'https://www.alphavantage.co', badge: '免费额度', desc: '免费全球股票 / 外汇 / 加密行情 API' },
      { name: 'Wind 万得', url: 'https://www.wind.com.cn', badge: '付费', desc: '金融终端，机构标配数据源' },
      { name: '东方财富 Choice', url: 'https://choice.eastmoney.com', badge: '付费', desc: '金融终端与量化数据接口' },
      { name: '金十数据', url: 'https://www.jin10.com', desc: '宏观数据、财经日历与突发快讯' },
      { name: '国家统计局', url: 'https://www.stats.gov.cn', desc: '官方宏观与经济统计原始数据' },
      { name: '中国人民银行', url: 'http://www.pbc.gov.cn', desc: '货币政策、利率与金融统计数据' },
      { name: '中国债券信息网', url: 'https://www.chinabond.com.cn', desc: '债券收益率曲线与发行数据' },
      { name: '中国货币网', url: 'https://www.chinamoney.com.cn', desc: '银行间市场利率与货币市场数据' },
      { name: '上海黄金交易所', url: 'https://www.sge.com.cn', desc: '黄金与贵金属现货价格' },
      { name: 'FRED', url: 'https://fred.stlouisfed.org', badge: '免费', desc: '美联储宏观数据库，利率与经济指标' },
      { name: 'CFTC', url: 'https://www.cftc.gov', desc: '美国商品期货持仓报告（COT）' },
      { name: 'CME Group', url: 'https://www.cmegroup.com', desc: '股指、利率、能源等期货与期权行情' },
      { name: 'Trading Economics', url: 'https://tradingeconomics.com', desc: '全球宏观指标与历史数据' },
      { name: '世界银行数据', url: 'https://data.worldbank.org', desc: '各国经济与发展指标' }
    ]
  },
  {
    id: 'academic',
    name: '学术与论文',
    emoji: '📚',
    desc: '量化金融论文、期刊与研究方法的一手来源',
    links: [
      { name: 'arXiv 量化金融', url: 'https://arxiv.org/list/q-fin/recent', badge: '免费', desc: '最前沿的量化金融预印本，免费全文' },
      { name: 'Alpha101 原始论文', url: 'https://arxiv.org/abs/1601.00991', badge: '免费', desc: 'WorldQuant 101 Alphas 因子原文' },
      { name: 'NBER', url: 'https://www.nber.org', badge: '部分免费', desc: '经济学工作论文与宏观研究' },
      { name: 'SSRN', url: 'https://www.ssrn.com', badge: '免费', desc: '社会科学研究网络，海量论文预印本' },
      { name: 'IEEE Xplore', url: 'https://ieeexplore.ieee.org', desc: '电子工程与金融工程期刊论文' },
      { name: 'Taylor & Francis', url: 'https://www.tandfonline.com', desc: 'Quantitative Finance 等核心期刊' },
      { name: 'JSTOR', url: 'https://www.jstor.org', desc: '经济学与金融学历史文献档案' },
      { name: 'Google 学术', url: 'https://scholar.google.com', badge: '免费', desc: '论文检索与引用追踪' },
      { name: 'Semantic Scholar', url: 'https://www.semanticscholar.org', badge: '免费', desc: 'AI 驱动的论文检索与引用图谱' },
      { name: 'ResearchGate', url: 'https://www.researchgate.net', desc: '研究者社区与论文互助' },
      { name: 'DOI 解析', url: 'https://doi.org', badge: '免费', desc: '论文永久链接，一键跳转到出版商页面' }
    ]
  },
  {
    id: 'frameworks',
    name: '量化框架与工具',
    emoji: '⚙️',
    desc: '回测引擎、因子分析、机器学习与高性能计算',
    links: [
      { name: 'Backtrader', url: 'https://www.backtrader.com', badge: '开源免费', desc: 'Python 回测框架经典，指标库完善' },
      { name: 'Backtesting.py', url: 'https://kernc.github.io/backtesting.py/', badge: '开源免费', desc: '轻量回测框架，几十行写完策略' },
      { name: 'Zipline', url: 'https://zipline.ml4trading.io', badge: '开源免费', desc: 'Quantopian 系事件驱动回测引擎' },
      { name: 'vn.py', url: 'https://www.vnpy.com', badge: '开源免费', desc: '国内全市场交易系统框架，接入实盘' },
      { name: 'Microsoft Qlib', url: 'https://github.com/microsoft/qlib', badge: '开源免费', desc: '微软开源的 AI 量化投研平台' },
      { name: 'RQAlpha', url: 'https://github.com/ricequant/rqalpha', badge: '开源免费', desc: '事件驱动回测引擎，与米筐生态打通' },
      { name: 'QUANTAXIS', url: 'https://github.com/yutiansut/QUANTAXIS', badge: '开源免费', desc: '纯 Python 全市场量化框架' },
      { name: 'Catalyst', url: 'https://github.com/enigmampc/catalyst', badge: '开源免费', desc: '加密资产量化交易框架' },
      { name: 'PyAlgotrade', url: 'https://github.com/gbeced/pyalgotrade', badge: '开源免费', desc: '事件驱动算法交易框架' },
      { name: 'TA-Lib', url: 'https://github.com/TA-Lib/ta-lib-python', badge: '开源免费', desc: '150+ 技术指标的官方 Python 封装' },
      { name: 'pandas-ta', url: 'https://pypi.org/project/pandas-ta/', badge: '开源免费', desc: '130+ 指标与形态识别，纯 pandas 实现' },
      { name: 'vectorbt', url: 'https://github.com/polakowo/vectorbt', badge: '开源免费', desc: '向量化大规模回测，参数扫描极快' },
      { name: 'MLFinLab', url: 'https://github.com/hudson-and-thames/mlfinlab', badge: '开源免费', desc: '金融机器学习工具库，含样本唯一性检验' },
      { name: 'Alphalens', url: 'https://github.com/stefan-jansen/alphalens-reloaded', badge: '开源免费', desc: '因子有效性分析，分层收益与 IC 拆解' },
      { name: 'PyBroker', url: 'https://github.com/edtechre/pybroker', badge: '开源免费', desc: '走机器学习优化的回测框架' },
      { name: 'FinRL', url: 'https://github.com/AI4Finance-Foundation/FinRL', badge: '开源免费', desc: '金融强化学习框架，含多种交易环境' },
      { name: 'Machine Learning for Trading', url: 'https://github.com/stefan-jansen/machine-learning-for-trading', badge: '开源免费', desc: '配套书籍《机器学习交易》代码全集' },
      { name: 'quantstats', url: 'https://github.com/ranaroussi/quantstats', badge: '开源免费', desc: '组合绩效指标与报告一键生成' },
      { name: 'PyPortfolioOpt', url: 'https://github.com/robertmartin8/PyPortfolioOpt', badge: '开源免费', desc: '现代组合优化与风险模型' },
      { name: '金融数值方法', url: 'https://github.com/cantaro86/Financial-Models-Numerical-Methods', desc: '金融定价的数值方法实现（FFT、蒙特卡洛）' },
      { name: 'Optuna', url: 'https://optuna.org', badge: '开源免费', desc: '超参数自动优化，TPE / SAMER 效率高' },
      { name: 'Numba', url: 'https://numba.pydata.org', badge: '开源免费', desc: 'JIT 加速，让回测循环提速几十倍' },
      { name: 'Polars', url: 'https://pola.rs', badge: '开源免费', desc: 'Rust 引擎的高性能 DataFrame' },
      { name: 'Dask', url: 'https://github.com/dask/dask', badge: '开源免费', desc: '分布式并行计算，处理大体量行情' },
      { name: 'Ray', url: 'https://www.ray.io', badge: '开源免费', desc: '分布式训练与超参搜索' },
      { name: 'uv', url: 'https://docs.astral.sh/uv', badge: '开源免费', desc: '极速 Python 依赖管理，装环境省时' },
      { name: 'QuantConnect', url: 'https://www.quantconnect.com', badge: '免费+付费', desc: '云端量化平台，LEAN 引擎完全开源' },
      { name: 'yfinance', url: 'https://github.com/ranaroussi/yfinance', badge: '开源免费', desc: 'Yahoo 行情的 Python 封装' },
      { name: 'AKShare 源码', url: 'https://github.com/akfamily/akshare', badge: '开源免费', desc: '数据接口实现与最新数据源跟踪' },
      { name: 'Tushare 源码', url: 'https://github.com/waditu/tushare', badge: '开源免费', desc: '数据接口实现与字段文档' }
    ]
  },
  {
    id: 'factors',
    name: '因子与策略库',
    emoji: '🧮',
    desc: '因子挖掘、Alpha 库与系统化策略实现',
    links: [
      { name: 'WorldQuant BRAIN', url: 'https://platform.worldquant.com', badge: '需申请', desc: '在线因子挖掘与 Alpha 回测平台' },
      { name: 'awesome-quant', url: 'https://github.com/wilsonfreitas/awesome-quant', badge: '开源免费', desc: '量化领域资源总索引，找工具神器' },
      { name: 'Systematic Trading Examples', url: 'https://github.com/robcarver17/systematictradingexamples', desc: '系统化策略的完整代码实现' },
      { name: 'TradingView Pine Script', url: 'https://www.tradingview.com/pine-script-docs/', desc: '图表平台上的可执行策略脚本语言' }
    ]
  },
  {
    id: 'media',
    name: '财经媒体',
    emoji: '📰',
    desc: '快讯、深度报道与市场解读',
    links: [
      { name: '华尔街见闻', url: 'https://wallstreetcn.com', desc: '全球市场快讯与深度解读' },
      { name: '财新网', url: 'https://www.caixin.com', desc: '调查报道与政策解读' },
      { name: '财联社', url: 'https://www.cls.cn', desc: '实时财经电报，盘面异动提示' },
      { name: '第一财经', url: 'https://www.yicai.com', desc: '电视财经与深度报道' },
      { name: '证券时报', url: 'https://www.stcn.com', desc: '证监会指定信息披露媒体' },
      { name: '上海证券报', url: 'https://www.cnstock.com', desc: '上市公司与政策动态' },
      { name: '中国证券报', url: 'https://www.cs.com.cn', desc: '资本市场主流媒体' },
      { name: '经济观察报', url: 'https://www.eeo.com.cn', desc: '商业与宏观深度观察' },
      { name: '全景网', url: 'https://www.p5w.net', desc: '上市公司路演与投资者互动平台' },
      { name: '新浪财经', url: 'https://finance.sina.com.cn', desc: '综合行情与财经门户' },
      { name: 'FT 中文网', url: 'https://www.ftchinese.com', desc: '国际财经深度与专栏' },
      { name: 'Reuters', url: 'https://www.reuters.com', desc: '全球财经快讯第一梯队' },
      { name: 'Bloomberg', url: 'https://www.bloomberg.com', desc: '全球市场数据与终端新闻' },
      { name: 'MarketWatch', url: 'https://www.marketwatch.com', desc: '市场数据与专栏文章' },
      { name: "Barron's", url: 'https://www.barrons.com', desc: '投资观点与市场分析' },
      { name: 'Investing.com', url: 'https://www.investing.com', desc: '行情、财经日历与分析师评级' }
    ]
  },
  {
    id: 'regulatory',
    name: '监管与公告',
    emoji: '🏛',
    desc: '监管规则、法定披露与交易所官方数据',
    links: [
      { name: '中国证监会', url: 'http://www.csrc.gov.cn', desc: '监管规则、公告与行政处罚' },
      { name: '巨潮资讯网', url: 'http://www.cninfo.com.cn', desc: '上市公司法定信息披露平台' },
      { name: '港交所披露易', url: 'https://www.hkexnews.hk', desc: '港股公告、年报与财报' },
      { name: '中国金融期货交易所', url: 'http://www.cffex.com.cn', desc: '股指期货合约、持仓与规则' },
      { name: '上海期货交易所', url: 'https://www.shfe.com.cn', desc: '商品期货行情与合约' },
      { name: '郑州商品交易所', url: 'https://www.czce.com.cn', desc: '农产品与化工期货数据' },
      { name: '大连商品交易所', url: 'http://www.dce.com.cn', desc: '大宗商品期货与持仓' },
      { name: '国家外汇管理局', url: 'https://www.safe.gov.cn', desc: '外汇政策与国际收支数据' },
      { name: 'SEC', url: 'https://www.sec.gov', desc: '美国证监会，规则与执法公告' },
      { name: 'EDGAR', url: 'https://www.sec.gov/edgar', desc: '美股公司原始文件全文检索' }
    ]
  },
  {
    id: 'brokers',
    name: '券商与开户',
    emoji: '🏦',
    desc: '境内外券商开户与交易通道',
    links: [
      { name: '中信证券', url: 'https://www.citics.com', badge: '需开户', desc: '头部综合券商' },
      { name: '华泰证券', url: 'https://www.htsc.com.cn', badge: '需开户', desc: '大型综合券商，涨乐财富通' },
      { name: '国泰海通', url: 'https://www.gtja.com', badge: '需开户', desc: '综合券商（原国泰君安）' },
      { name: '东方财富证券', url: 'https://www.dfzq.com.cn', badge: '需开户', desc: '互联网券商，佣金低' },
      { name: '招商证券', url: 'https://www.cmschina.com', badge: '需开户', desc: '综合券商，机构服务完善' },
      { name: '广发证券', url: 'https://www.gf.com.cn', badge: '需开户', desc: '大型综合券商' },
      { name: '中国银河证券', url: 'https://www.chinastock.com.cn', badge: '需开户', desc: '综合券商，零售与机构并重' },
      { name: '富途牛牛', url: 'https://www.futunn.com', badge: '需开户', desc: '港美股线上券商' },
      { name: '老虎证券', url: 'https://www.itiger.com', badge: '需开户', desc: '港美股与期权交易' },
      { name: '盈透证券 IBKR', url: 'https://www.interactivebrokers.com', badge: '需开户', desc: '全球券商，量化 API 最成熟' },
      { name: '长桥 Longbridge', url: 'https://www.longbridge.com', badge: '需开户', desc: '港美股券商，API 交易友好' }
    ]
  },
  {
    id: 'communities',
    name: '社区与交流',
    emoji: '💬',
    desc: '投资者社区，观点碰撞与实盘交流',
    links: [
      { name: '雪球', url: 'https://xueqiu.com', desc: '最大投资者社区，个股讨论与持仓分析' },
      { name: '韭研公社', url: 'https://www.jiuyangongshe.com', desc: 'A 股题材与产业逻辑研究' },
      { name: '集思录', url: 'https://www.jisilu.cn', badge: '免费注册', desc: '可转债、分级基金与套利数据' },
      { name: '果仁网', url: 'https://www.guorn.com', desc: '组合策略回测与基金投资社区' },
      { name: '淘股吧', url: 'https://www.taoguba.com.cn', desc: '短线题材与实盘讨论' },
      { name: '知乎', url: 'https://www.zhihu.com', desc: '泛财经与职业发展讨论' }
    ]
  },
  {
    id: 'crypto',
    name: '加密货币',
    emoji: '🪙',
    desc: '数字资产行情、链上数据与交易接口',
    links: [
      { name: 'CoinMarketCap', url: 'https://coinmarketcap.com', desc: '加密资产市值与行情排名' },
      { name: 'CoinGecko', url: 'https://www.coingecko.com', desc: '加密行情与研究数据，免费额度充足' },
      { name: 'Binance', url: 'https://www.binance.com', desc: '全球最大加密交易所' },
      { name: 'OKX', url: 'https://www.okx.com', desc: '交易所与 Web3 钱包' },
      { name: 'Deribit', url: 'https://www.deribit.com', desc: '加密期权与永续合约，波动率数据源' },
      { name: 'OKLink', url: 'https://www.oklink.com', desc: '链上数据浏览器与地址标签' },
      { name: 'Glassnode', url: 'https://glassnode.com', badge: '付费', desc: '链上宏观与交易所资金流分析' },
      { name: 'CCXT', url: 'https://github.com/ccxt/ccxt', badge: '开源免费', desc: '上百家交易所的统一 API' }
    ]
  },
  {
    id: 'learning',
    name: '学习资源',
    emoji: '🎓',
    desc: '课程、教材、百科与 Python 生态入门',
    links: [
      { name: 'Quantpedia', url: 'https://quantpedia.com', badge: '免费', desc: '学术策略数据库，按指标查历史回测' },
      { name: 'MIT OpenCourseWare', url: 'https://ocw.mit.edu', badge: '免费', desc: '公开课，金融理论、概率与数理' },
      { name: 'Khan Academy', url: 'https://www.khanacademy.org', badge: '免费', desc: '金融与统计基础，配套练习' },
      { name: 'CFA Institute', url: 'https://www.cfainstitute.org', badge: '付费', desc: 'CFA 教材、考纲与研究资源' },
      { name: 'GARP', url: 'https://www.garp.org', badge: '付费', desc: 'FRM 风险管理标准与备考资源' },
      { name: 'Wall Street Prep', url: 'https://www.breakingdownwalls.com', badge: '付费', desc: '投行与量化岗位面试题库' },
      { name: 'Investopedia', url: 'https://www.investopedia.com', badge: '免费', desc: '金融术语百科与研究方法说明' },
      { name: 'Coursera', url: 'https://www.coursera.org', badge: '部分免费', desc: '在线课程平台，量化与金融数学方向' },
      { name: 'DeepLearning.AI', url: 'https://www.deeplearning.ai', badge: '免费', desc: 'AI 短课，含金融应用专题' },
      { name: 'Kaggle', url: 'https://www.kaggle.com', badge: '免费', desc: '金融数据集与量化竞赛' },
      { name: 'Hugging Face Datasets', url: 'https://huggingface.co/datasets', badge: '免费', desc: '文本情绪与行情数据集' },
      { name: 'Python 官网', url: 'https://www.python.org', badge: '免费', desc: '语言下载与生态入口' },
      { name: 'Python 文档', url: 'https://docs.python.org', badge: '免费', desc: '标准库与语法权威参考' },
      { name: 'pandas', url: 'https://pandas.pydata.org', badge: '免费', desc: '金融时间序列处理的事实标准' },
      { name: 'NumPy', url: 'https://numpy.org', badge: '免费', desc: '数值计算基础，回测提速的前提' }
    ]
  }
]

/** 汇总统计 */
export const stockLinkStats = {
  total: stockLinkCategories.reduce((sum, c) => sum + c.links.length, 0),
  categories: stockLinkCategories.length,
  /** 去掉重复出现的链接后的实际去重数 */
  unique: new Set(stockLinkCategories.flatMap((c) => c.links.map((l) => l.url))).size
}
