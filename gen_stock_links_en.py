import re

# 中文站点名 → (英文名, 英文说明)
T = {
# 行情与交易
'东方财富网': ('Eastmoney', 'A-share, HK and US quotes with filings and fund flows'),
'东方财富行情': ('Eastmoney Quotes', 'Real-time quotes, candlesticks and order-book data'),
'同花顺': ('10jqka', 'Quotes, filings and natural-language stock screening in one place'),
'同花顺量化': ('10jqka Quant', 'Free online quant research platform with strategy backtesting'),
'问财选股': ('iWencai', 'Natural-language screening, e.g. "ROE above 20% for three years"'),
'恒生聚源': ('Gildata', 'Financial database and quantitative data services'),
'腾讯自选股': ('Tencent Quotes', 'Lightweight quotes and price alerts'),
'和讯网': ('Hexun', 'Long-running finance portal with global indices'),
'上海证券交易所': ('Shanghai Stock Exchange', 'SSE rules, filings and market statistics'),
'深圳证券交易所': ('Shenzhen Stock Exchange', 'SZSE rules, listed companies and market data'),
'香港交易所': ('HKEX', 'Hong Kong quotes, listings and trading rules'),
'Nasdaq': ('Nasdaq', 'US quotes, IPOs and ETF data'),
'NYSE': ('NYSE', 'Official listings and exchange trading data'),
'TradingView': ('TradingView', 'Professional technical charts with Pine Script strategies'),
'Investing.com 中文': ('Investing.com (CN)', 'Global quotes, economic calendar and market news'),
'Yahoo Finance': ('Yahoo Finance', 'Global quotes, earnings and historical data downloads'),
'Google Finance': ('Google Finance', 'Quick cross-market quote comparison'),
'富途牛牛': ('Futu', 'HK/US quotes and trading with an active community'),
'长桥 Longbridge': ('Longbridge', 'HK/US broker with a quant-friendly API'),
'老虎证券': ('Tiger Brokers', 'HK/US trading and options platform'),
'盈透证券 IBKR': ('Interactive Brokers', 'Global professional broker built for quant traders'),
# 数据与研究平台
'Tushare Pro': ('Tushare Pro', 'Financial data API, one of the richest free A-share sources'),
'AKShare': ('AKShare', 'Open-source finance data interface; market data in one line of code'),
'聚宽 JoinQuant': ('JoinQuant', 'Online quant research platform with an active community'),
'米筐 Ricequant': ('Ricequant', 'Quant research platform with multi-style factor libraries'),
'BigQuant': ('BigQuant', 'No-code machine-learning stock selection and strategy platform'),
'优矿 Uqer': ('Uqer', 'Quantitative research platform by DataYes'),
'萝卜投研': ('Robo Research', 'AI-assisted research, factor mining and fundamentals'),
'理杏仁': ('Lixinger', 'Valuation percentiles and financial data visualisation'),
'慧博投研': ('Hibor', 'Chinese research-report and data aggregator'),
'Alpha Vantage': ('Alpha Vantage', 'Free global stock, FX and crypto API'),
'Wind 万得': ('Wind', 'Institutional data terminal, the industry standard'),
'东方财富 Choice': ('Eastmoney Choice', 'Financial terminal and quant data feeds'),
'金十数据': ('Jin10', 'Macro data, economic calendar and breaking news'),
'国家统计局': ('National Bureau of Statistics', 'Official macro and economic statistics'),
'中国人民银行': ('People’s Bank of China', 'Monetary policy, rates and financial statistics'),
'中国债券信息网': ('Chinabond', 'Bond yield curves and issuance data'),
'中国货币网': ('ChinaMoney', 'Interbank rates and money-market data'),
'上海黄金交易所': ('Shanghai Gold Exchange', 'Spot gold and precious-metal prices'),
'FRED': ('FRED', 'Federal Reserve macro database'),
'CFTC': ('CFTC', 'US commodity futures positioning reports'),
'CME Group': ('CME Group', 'Equity, rates and energy futures and options'),
'Trading Economics': ('Trading Economics', 'Global macro indicators and historical data'),
'世界银行数据': ('World Bank Data', 'Economic and development indicators by country'),
# 学术与论文
'arXiv 量化金融': ('arXiv q-fin', 'Cutting-edge quantitative finance preprints, free full text'),
'Alpha101 原始论文': ('The Alpha101 Paper', 'Original paper behind WorldQuant’s 101 Alphas'),
'NBER': ('NBER', 'Economics working papers and macro research'),
'SSRN': ('SSRN', 'Social science research network with a huge preprint archive'),
'IEEE Xplore': ('IEEE Xplore', 'Electrical and financial engineering journals'),
'Taylor & Francis': ('Taylor & Francis', 'Journals including Quantitative Finance'),
'JSTOR': ('JSTOR', 'Archive of economics and finance literature'),
'Google 学术': ('Google Scholar', 'Paper search and citation tracking'),
'Semantic Scholar': ('Semantic Scholar', 'AI-driven paper search and citation graphs'),
'ResearchGate': ('ResearchGate', 'Researcher community and paper sharing'),
'DOI 解析': ('DOI Resolver', 'Permanent links straight to the publisher'),
# 量化框架与工具
'Backtrader': ('Backtrader', 'Classic Python backtesting framework with a rich indicator set'),
'Backtesting.py': ('Backtesting.py', 'Lightweight backtester; a strategy in a few dozen lines'),
'Zipline': ('Zipline', 'Quantopian-style event-driven backtesting engine'),
'vn.py': ('vn.py', 'Full-market Chinese trading framework that connects to live accounts'),
'Microsoft Qlib': ('Microsoft Qlib', 'Microsoft’s open-source AI quant research platform'),
'RQAlpha': ('RQAlpha', 'Event-driven engine integrated with the Ricequant ecosystem'),
'QUANTAXIS': ('QUANTAXIS', 'Pure-Python full-market quant framework'),
'Catalyst': ('Catalyst', 'Quant trading framework for crypto assets'),
'PyAlgotrade': ('PyAlgotrade', 'Event-driven algorithmic trading framework'),
'TA-Lib': ('TA-Lib', 'Official Python wrapper for 150+ technical indicators'),
'pandas-ta': ('pandas-ta', '130+ indicators and pattern detection in pure pandas'),
'vectorbt': ('vectorbt', 'Vectorised large-scale backtesting and parameter sweeps'),
'MLFinLab': ('MLFinLab', 'Financial machine-learning toolbox'),
'Alphalens': ('Alphalens', 'Factor efficacy analysis with layered returns and IC breakdown'),
'PyBroker': ('PyBroker', 'Event-driven backtesting with machine-learning optimisation'),
'FinRL': ('FinRL', 'Reinforcement-learning framework for finance'),
'Machine Learning for Trading': ('Machine Learning for Trading', 'Full code for the companion book'),
'quantstats': ('quantstats', 'One-click portfolio performance reports'),
'PyPortfolioOpt': ('PyPortfolioOpt', 'Modern portfolio optimisation and risk models'),
'金融数值方法': ('Financial Numerical Methods', 'Numerical methods for financial pricing'),
'Optuna': ('Optuna', 'Automated hyperparameter optimisation (TPE / SAMER)'),
'Numba': ('Numba', 'JIT compilation that speeds up backtest loops by orders of magnitude'),
'Polars': ('Polars', 'Rust-powered high-performance DataFrame'),
'Dask': ('Dask', 'Distributed parallel computing for large market datasets'),
'Ray': ('Ray', 'Distributed training and hyperparameter search'),
'uv': ('uv', 'Blazing-fast Python dependency management'),
'QuantConnect': ('QuantConnect', 'Cloud quant platform built on the open-source LEAN engine'),
'yfinance': ('yfinance', 'Python wrapper for Yahoo market data'),
'AKShare 源码': ('AKShare on GitHub', 'Interface source and latest data-source tracking'),
'Tushare 源码': ('Tushare on GitHub', 'Interface source and field documentation'),
# 因子与策略库
'WorldQuant BRAIN': ('WorldQuant BRAIN', 'Online factor mining and alpha backtesting platform'),
'awesome-quant': ('awesome-quant', 'Master index of quant resources and tooling'),
'Systematic Trading Examples': ('Systematic Trading Examples', 'Complete implementations of systematic strategies'),
'TradingView Pine Script': ('TradingView Pine Script', 'Executable strategy scripting on the charting platform'),
# 财经媒体
'华尔街见闻': ('Wallstreetcn', 'Global market flashes and in-depth analysis'),
'财新网': ('Caixin', 'Investigative reporting and policy interpretation'),
'财联社': ('CLS', 'Real-time financial telegrams and flow alerts'),
'第一财经': ('Yicai', 'TV finance and in-depth reporting'),
'证券时报': ('Securities Times', 'CSRC-designated disclosure media'),
'上海证券报': ('Shanghai Securities News', 'Listed-company and policy developments'),
'中国证券报': ('China Securities Journal', 'Mainstream capital-markets media'),
'经济观察报': ('Economic Observer', 'Business and macro deep dives'),
'全景网': ('P5W', 'Listed-company roadshows and investor interaction'),
'新浪财经': ('Sina Finance', 'Composite quotes and finance portal'),
'FT 中文网': ('FT Chinese', 'International finance analysis and columns'),
'Reuters': ('Reuters', 'Top-tier global financial wire service'),
'Bloomberg': ('Bloomberg', 'Global market data and terminal news'),
'MarketWatch': ('MarketWatch', 'Market data and analyst columns'),
'Barron’s': ('Barron’s', 'Investment views and market analysis'),
'Investing.com': ('Investing.com', 'Quotes, economic calendar and analyst ratings'),
# 监管与公告
'中国证监会': ('CSRC', 'Regulatory rules, notices and enforcement actions'),
'巨潮资讯网': ('CNINFO', 'Statutory disclosure platform for listed companies'),
'港交所披露易': ('HKEXnews', 'HK filings, annual reports and results'),
'中国金融期货交易所': ('CFFEX', 'Index-futures contracts, positions and rules'),
'上海期货交易所': ('SHFE', 'Commodity futures quotes and contracts'),
'郑州商品交易所': ('CZCE', 'Agricultural and chemical futures data'),
'大连商品交易所': ('DCE', 'Bulk commodity futures and positioning'),
'国家外汇管理局': ('SAFE', 'FX policy and balance-of-payments data'),
'SEC': ('SEC', 'US regulator: rules and enforcement notices'),
'EDGAR': ('EDGAR', 'Full-text search of US company filings'),
# 券商与开户
'中信证券': ('CITIC Securities', 'Top-tier full-service broker'),
'华泰证券': ('Huatai Securities', 'Large full-service broker with a retail trading app'),
'国泰海通': ('Guotai Haitong', 'Full-service broker (formerly Guotai Junan)'),
'东方财富证券': ('Eastmoney Securities', 'Online-first broker with low commissions'),
'招商证券': ('China Merchants Securities', 'Full-service broker with strong institutional service'),
'广发证券': ('GF Securities', 'Large full-service broker'),
'中国银河证券': ('China Galaxy Securities', 'Retail and institutional brokerage'),
# 社区与交流
'雪球': ('Xueqiu', 'Largest investor community: stock discussion and portfolio analysis'),
'韭研公社': ('Jiuyangongshe', 'A-share theme and industry-chain research'),
'集思录': ('Jisilu', 'Convertible bonds, graded funds and arbitrage data'),
'果仁网': ('Guorn', 'Portfolio backtesting and fund-investing community'),
'淘股吧': ('Taoguba', 'Short-term themes and live trading talk'),
'知乎': ('Zhihu', 'General finance discussion and career advice'),
# 加密货币
'CoinMarketCap': ('CoinMarketCap', 'Crypto market capitalisation and rankings'),
'CoinGecko': ('CoinGecko', 'Crypto quotes and research data with a generous free tier'),
'Binance': ('Binance', 'Largest global crypto exchange'),
'OKX': ('OKX', 'Exchange and Web3 wallet'),
'Deribit': ('Deribit', 'Crypto options and perpetuals; a volatility data source'),
'OKLink': ('OKLink', 'On-chain explorer and address labels'),
'Glassnode': ('Glassnode', 'On-chain macro and exchange flow analysis'),
'CCXT': ('CCXT', 'Unified API for a hundred exchanges'),
# 学习资源
'Quantpedia': ('Quantpedia', 'Academic strategy database searchable by metric'),
'MIT OpenCourseWare': ('MIT OpenCourseWare', 'Open courses in finance, probability and maths'),
'Khan Academy': ('Khan Academy', 'Finance and statistics basics with exercises'),
'CFA Institute': ('CFA Institute', 'CFA curriculum, syllabus and research'),
'GARP': ('GARP', 'FRM risk-management standard and study resources'),
'Wall Street Prep': ('Wall Street Prep', 'Investment banking and quant interview question banks'),
'Investopedia': ('Investopedia', 'Financial term encyclopedia and methodology notes'),
'Coursera': ('Coursera', 'Online courses in quant and financial mathematics'),
'DeepLearning.AI': ('DeepLearning.AI', 'Short AI courses including finance tracks'),
'Kaggle': ('Kaggle', 'Financial datasets and quant competitions'),
'Hugging Face Datasets': ('Hugging Face Datasets', 'Text sentiment and market datasets'),
'Python 官网': ('Python.org', 'Language downloads and ecosystem entry point'),
'Python 文档': ('Python Docs', 'Authoritative standard library and syntax reference'),
'pandas': ('pandas', 'De facto standard for financial time-series handling'),
'NumPy': ('NumPy', 'Numerical foundation that makes backtests fast'),
}

CATS = {
 'quotes': ('Quotes & Trading', 'Watchlists, quotes, candlesticks and trading entry points across major global markets'),
 'data': ('Data & Research', 'Market data APIs, quant research platforms and macro databases'),
 'academic': ('Academic & Papers', 'Primary sources for quant finance papers, journals and research methods'),
 'frameworks': ('Quant Frameworks & Tools', 'Backtesting engines, factor analysis, machine learning and high-performance computing'),
 'factors': ('Factors & Strategy Libraries', 'Factor mining, alpha libraries and systematic strategy implementations'),
 'media': ('Financial Media', 'Breaking news, in-depth reporting and market interpretation'),
 'regulatory': ('Regulators & Filings', 'Regulatory rules, statutory disclosure and official exchange data'),
 'brokers': ('Brokers & Accounts', 'Chinese and international brokers for account opening and execution'),
 'communities': ('Communities', 'Investor communities for idea exchange and live trading talk'),
 'crypto': ('Crypto', 'Digital asset quotes, on-chain data and exchange APIs'),
 'learning': ('Learning Resources', 'Courses, textbooks, encyclopedias and the Python ecosystem'),
}

src = open('src/data/stock-links.ts', encoding='utf-8').read()

missing = []
for zh_name in T:
    if f"name: '{zh_name}'" not in src:
        missing.append(zh_name)
if missing:
    print('⚠️ 未在数据文件中找到的条目：', missing)

# 提取每个分类的 links 块，按分类生成英文版
cats = re.findall(
    r"\{\s*\n\s*id: '([a-z]+)',\s*\n\s*name: '([^']+)',\s*\n\s*emoji: '([^']+)',\s*\n\s*desc: '([^']+)',\s*\n\s*links: \[(.*?)\n\s*\]\n\s*\}",
    src, re.S)

out = []
for cid, cname, emoji, cdesc, links_block in cats:
    en_name, en_desc = CATS[cid]
    entries = re.findall(
        r"\{ name: '([^']+)', url: '([^']+)'(?:, badge: '([^']*)')?, desc: '([^']*)' \}", links_block)
    en_entries = []
    for zn, url, badge, desc in entries:
        if zn not in T:
            missing.append(zn)
            continue
        en_n, en_d = T[zn]
        badge_part = f", badge: '{badge}'" if badge else ''
        en_entries.append(f"      {{ name: '{en_n}', url: '{url}'{badge_part}, desc: '{en_d}' }}")
    links_out = ',\n'.join(en_entries)
    out.append(f"""  {{
    id: '{cid}',
    name: '{en_name}',
    emoji: '{emoji}',
    desc: '{en_desc}',
    links: [
{links_out}
    ]
  }}""")

en_block = "export const stockLinkCategoriesEn: StockLinkCategory[] = [\n" + ",\n".join(out) + "\n]\n"

# 追加到文件末尾并补充统计导出
extra = f"""
/** 英文版分类数据 */
{en_block}
/** 英文版统计 */
export const stockLinkStatsEn = {{
  total: stockLinkCategoriesEn.reduce((sum, c) => sum + c.links.length, 0),
  categories: stockLinkCategoriesEn.length,
  unique: new Set(stockLinkCategoriesEn.flatMap((c) => c.links.map((l) => l.url))).size
}}
"""
open('src/data/stock-links.ts', 'a', encoding='utf-8').write(extra)
print(f'英文版分类已生成：{len(cats)} 类')
if missing:
    print('⚠️ 缺失映射：', missing)
else:
    print('全部条目均已翻译')
