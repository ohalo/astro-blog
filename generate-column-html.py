#!/usr/bin/env python3
# Generate complete quant-column.html with proper structure

html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>量化交易实战专栏 - 从零开始学量化</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1200px; margin: 0 auto; padding: 2rem; background: #f8f9fa; line-height: 1.6; }
    
    .column-header { text-align: center; margin-bottom: 3rem; padding: 3rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white; }
    .column-header h1 { font-size: 2.5rem; margin-bottom: 1rem; }
    .column-subtitle { font-size: 1.3rem; opacity: 0.95; margin-bottom: 2rem; }
    
    .intro-section { background: white; padding: 2.5rem; border-radius: 12px; margin-bottom: 3rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .intro-section h2 { color: #333; margin-bottom: 1.5rem; font-size: 1.8rem; }
    .intro-section p { color: #555; margin-bottom: 1rem; font-size: 1.05rem; }
    .intro-section ul { margin: 1rem 0 1rem 2rem; color: #555; }
    .intro-section li { margin-bottom: 0.5rem; }
    .highlight-box { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; border-left: 4px solid #667eea; }
    
    .column-stats { display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 2rem; }
    .stat-item { background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 20px; font-size: 1rem; }
    
    .column-toc { background: white; padding: 2rem; border-radius: 8px; margin-bottom: 3rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .column-toc h2 { margin-bottom: 1.5rem; color: #333; font-size: 1.5rem; }
    .column-toc ul { list-style: none; padding: 0; display: flex; gap: 2rem; flex-wrap: wrap; }
    .column-toc a { color: #667eea; text-decoration: none; font-weight: 500; font-size: 1.05rem; }
    .column-toc a:hover { text-decoration: underline; }
    
    .difficulty-section { margin-bottom: 4rem; }
    .section-header { margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 3px solid; }
    .beginner .section-header { border-color: #51cf66; }
    .intermediate .section-header { border-color: #ff922b; }
    .advanced .section-header { border-color: #fa5252; }
    .section-header h2 { font-size: 1.8rem; margin-bottom: 0.5rem; color: #333; }
    .section-description { color: #666; font-size: 1.05rem; }
    
    .articles-list { display: flex; flex-direction: column; gap: 1.5rem; }
    .article-item { display: flex; gap: 1.5rem; padding: 1.5rem; border: 1px solid #e9ecef; border-radius: 8px; background: white; transition: all 0.3s ease; }
    .article-item:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); transform: translateY(-2px); }
    .article-index { font-size: 2rem; font-weight: bold; color: #dee2e6; min-width: 50px; display: flex; align-items: center; }
    .article-info { flex: 1; }
    .difficulty-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.75rem; }
    .beginner-badge { background: #d3f9d8; color: #2b8a3e; }
    .intermediate-badge { background: #fff3bf; color: #e67700; }
    .advanced-badge { background: #ffe3e3; color: #c92a2a; }
    .article-info h3 { margin-bottom: 0.75rem; font-size: 1.15rem; }
    .article-info h3 a { color: #333; text-decoration: none; }
    .article-info h3 a:hover { color: #667eea; }
    .article-info p { color: #666; margin-bottom: 0.75rem; line-height: 1.5; }
    .read-time { color: #868e96; font-size: 0.9rem; }
    
    .learning-tips { margin-top: 4rem; padding: 2.5rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .learning-tips h2 { text-align: center; margin-bottom: 2rem; color: #333; font-size: 1.8rem; }
    .tips-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; }
    .tip-card { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #667eea; }
    .tip-card h3 { margin-bottom: 0.75rem; color: #333; font-size: 1.1rem; }
    .tip-card p { color: #666; line-height: 1.5; }
    
    footer { text-align: center; margin-top: 3rem; padding: 2rem; color: #999; font-size: 0.9rem; }
    footer a { color: #667eea; }
    
    @media (max-width: 768px) {
      body { padding: 1rem; }
      .column-header h1 { font-size: 1.8rem; }
      .column-stats { flex-direction: column; align-items: center; }
      .column-toc ul { flex-direction: column; gap: 0.75rem; }
      .article-item { flex-direction: column; }
      .article-index { font-size: 1.5rem; }
    }
  </style>
</head>
<body>
  <div class="quant-column-container">
    <!-- 专栏标题区 -->
    <header class="column-header">
      <h1>📊 量化交易实战专栏</h1>
      <p class="column-subtitle">从零开始，系统学习量化投资</p>
      <div class="column-stats">
        <span class="stat-item">📚 共 25 篇文章</span>
        <span class="stat-item">⏱️ 预计阅读 558 分钟</span>
        <span class="stat-item">🎯 从入门到实战</span>
      </div>
    </header>

    <!-- 引言：量化是什么？为什么要做量化？ -->
    <section class="intro-section">
      <h2>🤔 量化交易是什么？为什么要学它？</h2>
      
      <p><strong>想象一下：</strong></p>
      <p>你是一个基金经理，每天要面对海量的市场数据：股价、成交量、财报、新闻... 靠人工分析，根本看不过来。更糟的是，人性是情绪化的——恐惧、贪婪、侥幸心理，往往导致追涨杀跌、割肉在地板。</p>
      
      <p><strong>量化交易就是答案：</strong></p>
      <p>用量化的方法（数学、统计、编程）来研究市场、制定策略、自动执行交易。它不靠感觉，不靠小道消息，只相信数据和回测结果。</p>
      
      <div class="highlight-box">
        <p><strong>📈 量化的核心优势：</strong></p>
        <ul>
          <li><strong>纪律性</strong>：严格执行策略，不受情绪干扰</li>
          <li><strong>系统性</strong>：可以同时监控上千只股票，发现人工看不到的规律</li>
          <li><strong>可验证</strong>：用历史数据回测，策略好不好，数据说了算</li>
          <li><strong>可扩展</strong>：策略写好后，可以7×24小时自动运行</li>
        </ul>
      </div>
      
      <p><strong>🎯 这个专栏能帮你什么？</strong></p>
      <p>本专栏系统梳理了量化交易的核心知识体系，从基础概念到实战策略，从因子研究到机器学习应用。无论你是：</p>
      <ul>
        <li>📖 <strong>完全新手</strong>：想了解量化交易是什么</li>
        <li>📊 <strong>有一定基础</strong>：想系统学习量化策略</li>
        <li>💻 <strong>程序员</strong>：想把编程技能应用到金融</li>
        <li>📈 <strong>交易员</strong>：想用量化方法提升业绩</li>
      </ul>
      <p>都能在这里找到适合自己的学习路径。</p>
      
      <p><strong>📚 学习建议：</strong></p>
      <p>建议按照<strong>入门→中级→高级</strong>的顺序学习。每篇文章都包含Python代码示例，建议边学边敲代码，用历史数据回测验证。记住：<strong>量化不是圣杯，风险管理和资金安全永远是第一位的</strong>。</p>
    </section>

    <!-- 目录导航 -->
    <nav class="column-toc">
      <h2>📑 快速导航</h2>
      <ul>
        <li><a href="#beginner">🌱 入门级：基础概念 (5 篇)</a></li>
        <li><a href="#intermediate">🚀 中级：策略与应用 (18 篇)</a></li>
        <li><a href="#advanced">💎 高级：技术与实战 (7 篇)</a></li>
      </ul>
    </nav>

    <!-- 入门级文章 -->
    <section id="beginner" class="difficulty-section beginner">
      <div class="section-header">
        <h2>🌱 入门级：基础概念</h2>
        <p class="section-description">适合量化交易新手，建立基础认知框架。学完这部分，你会明白：因子是什么、风险如何度量、行为金融如何应用于量化。</p>
      </div>
      <div class="articles-list">
        <div class="article-item">
          <div class="article-index">01</div>
          <div class="article-info">
            <span class="difficulty-badge beginner-badge">入门</span>
            <h3><a href="/blog/factor-research-empirical-analysis/">因子研究实证分析：价值、动量、质量与低波因子的实战表现</a></h3>
            <p>介绍四大经典因子（价值、动量、质量、低波）的基本概念和实证方法</p>
            <span class="read-time">⏱️ 15 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">02</div>
          <div class="article-info">
            <span class="difficulty-badge beginner-badge">入门</span>
            <h3><a href="/blog/2026-06-05-behavioral-finance-quant/">行为金融学在量化策略中的应用：捕捉散户情绪与认知偏差</a></h3>
            <p>利用行为金融学理论，将散户心理偏差转化为量化策略信号</p>
            <span class="read-time">⏱️ 18 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">03</div>
          <div class="article-info">
            <span class="difficulty-badge beginner-badge">入门</span>
            <h3><a href="/blog/2026-06-05-risk-management-var/">量化风险管理：VaR、CVaR与压力测试的实战指南</a></h3>
            <p>风险管理基础：VaR、CVaR计算方法与压力测试框架</p>
            <span class="read-time">⏱️ 20 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">04</div>
          <div class="article-info">
            <span class="difficulty-badge beginner-badge">入门</span>
            <h3><a href="/blog/2026-06-05-cvar-stress-testing/">风险管理实战：CVaR计算与压力测试方法</a></h3>
            <p>深入学习CVaR（条件风险价值）的计算方法与压力测试实战</p>
            <span class="read-time">⏱️ 18 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">05</div>
          <div class="article-info">
            <span class="difficulty-badge beginner-badge">入门</span>
            <h3><a href="/blog/risk-management-var-cvar/">风险管理实战：VaR与CVaR的计算与应用</a></h3>
            <p>风险价值（VaR）和条件风险价值（CVaR）的计算方法与实战应用</p>
            <span class="read-time">⏱️ 16 分钟</span>
          </div>
        </div>
      </div>
    </section>

    <!-- 中级文章 -->
    <section id="intermediate" class="difficulty-section intermediate">
      <div class="section-header">
        <h2>🚀 中级：策略与应用</h2>
        <p class="section-description">掌握核心策略，开始实盘应用。学完这部分，你会掌握：配对交易、期权策略、投资组合优化、另类数据应用。</p>
      </div>
      <div class="articles-list">
        <div class="article-item">
          <div class="article-index">01</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/pairs-trading-statistical-arbitrage/">配对交易与统计套利：从协整到实战</a></h3>
            <p>统计套利核心：协整检验、配对选择、交易信号生成</p>
            <span class="read-time">⏱️ 22 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">02</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/pairs-trading-cointegration/">统计套利实战：配对交易与协整分析</a></h3>
            <p>协整理论基础与配对交易策略的实战实现</p>
            <span class="read-time">⏱️ 20 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">03</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/cointegration-changepoint-detection/">协整关系变点检测与统计套利实战：捕捉配对交易的黄金窗口</a></h3>
            <p>进阶统计套利：检测协整关系变点，捕捉配对交易最佳时机</p>
            <span class="read-time">⏱️ 25 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">04</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/factor-decay-lifecycle/">因子衰减效应与因子投资生命周期：量化投资的残酷真相</a></h3>
            <p>因子研究进阶：理解因子衰减的原因、生命周期与应对方法</p>
            <span class="read-time">⏱️ 20 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">05</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/factor-crowding-identification/">因子拥挤度：多因子策略的隐形风险与识别技术</a></h3>
            <p>识别因子拥挤的方法与因子投资策略的风险管理</p>
            <span class="read-time">⏱️ 22 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">06</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/risk-parity-model/">风险平价模型实战：超越马科维茨的资产配置革命</a></h3>
            <p>投资组合理论进阶：从马科维茨均值方差到风险平价模型</p>
            <span class="read-time">⏱️ 22 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">07</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/2026-06-05-black-litterman-portfolio-optimization/">Black-Litterman模型实战：从先验收益到后验收益的投资组合优化</a></h3>
            <p>进阶资产配置：Black-Litterman模型结合投资者观点与市场均衡</p>
            <span class="read-time">⏱️ 25 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">08</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/hierarchical-risk-parity-portfolio-optimization/">层次风险平价：用机器学习重构投资组合优化</a></h3>
            <p>机器学习在投资组合中的应用：层次聚类风险平价模型</p>
            <span class="read-time">⏱️ 28 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">09</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/2026-06-05-option-volatility-trading-strategies/">期权波动率交易实战：跨式、宽跨式与波动率微笑的量化策略</a></h3>
            <p>期权交易核心：波动率策略、期权组合与Greeks管理</p>
            <span class="read-time">⏱️ 24 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">10</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/option-delta-neutral-covered-call/">期权策略实战：Delta中性与备兑开仓的深度解析</a></h3>
            <p>期权实战策略：Delta中性对冲与备兑开仓策略详解</p>
            <span class="read-time">⏱️ 22 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">11</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/option-volatility-trading/">期权波动率交易实战：隐含波动率曲面交易与Delta对冲</a></h3>
            <p>期权波动率交易的高级技术：波动率曲面构造与交易</p>
            <span class="read-time">⏱️ 26 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">12</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/2026-06-05-alternative-data-quant/">另类数据：量化交易的下一个阿尔法源泉</a></h3>
            <p>数据挖掘：卫星图像、社交媒体、信用卡数据在量化中的应用</p>
            <span class="read-time">⏱️ 20 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">13</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/alternative-data-quant-trading/">另类数据在量化交易中的应用：从卫星图像到社交情绪</a></h3>
            <p>另类数据源的获取、处理与在量化策略中的应用案例</p>
            <span class="read-time">⏱️ 22 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">14</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/portfolio-transaction-cost-opt/">带交易成本的投资组合优化：从理论到实战</a></h3>
            <p>实战优化：将交易成本纳入投资组合构建框架</p>
            <span class="read-time">⏱️ 23 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">15</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/trading-cost-analysis-optimization/">交易成本分析：量化交易中的隐性成本与优化方法</a></h3>
            <p>实战细节：交易成本分解、Market Impact模型与最优执行</p>
            <span class="read-time">⏱️ 22 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">16</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/statistical-arbitrage-pairs-trading/">统计套利：配对交易与协整策略实战</a></h3>
            <p>统计套利理论基础与配对交易策略的完整实现流程</p>
            <span class="read-time">⏱️ 24 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">17</div>
          <div class="article-info">
            <span class="difficulty-badge intermediate-badge">中级</span>
            <h3><a href="/blog/behavioral-finance-quant-strategy/">行为金融视角下的量化策略优化：利用散户心理偏差获取阿尔法</a></h3>
            <p>行为金融学进阶：如何利用散户心理偏差构建量化策略</p>
            <span class="read-time">⏱️ 21 分钟</span>
          </div>
        </div>
      </div>
    </section>

    <!-- 高级文章 -->
    <section id="advanced" class="difficulty-section advanced">
      <div class="section-header">
        <h2>💎 高级：技术与实战</h2>
        <p class="section-description">深入技术细节，掌握实战核心。学完这部分，你会掌握：机器学习量化、小波变换、高频交易、实盘系统搭建。</p>
      </div>
      <div class="articles-list">
        <div class="article-item">
          <div class="article-index">01</div>
          <div class="article-info">
            <span class="difficulty-badge advanced-badge">高级</span>
            <h3><a href="/blog/2026-06-05-ml-quant-trading/">机器学习在量化交易中的应用：从LSTM到Transformer的完整实战指南</a></h3>
            <p>AI量化：LSTM、Transformer等深度学习模型在量化预测中的应用</p>
            <span class="read-time">⏱️ 30 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">02</div>
          <div class="article-info">
            <span class="difficulty-badge advanced-badge">高级</span>
            <h3><a href="/blog/lstm-quant-prediction/">LSTM在量化预测中的实战应用：从理论到部署</a></h3>
            <p>深度学习实战：LSTM模型构建、训练、验证与实盘部署</p>
            <span class="read-time">⏱️ 28 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">03</div>
          <div class="article-info">
            <span class="difficulty-badge advanced-badge">高级</span>
            <h3><a href="/blog/2026-06-05-random-forest-factor-synthesis/">机器学习在量化中的应用：随机森林因子合成实战</a></h3>
            <p>机器学习因子挖掘：随机森林在因子合成与非线性建模中的应用</p>
            <span class="read-time">⏱️ 25 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">04</div>
          <div class="article-info">
            <span class="difficulty-badge advanced-badge">高级</span>
            <h3><a href="/blog/wavelet-multiscale-signal/">小波变换在量化交易中的多尺度信号提取实战</a></h3>
            <p>信号处理：小波变换在多尺度金融时间序列分析中的应用</p>
            <span class="read-time">⏱️ 27 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">05</div>
          <div class="article-info">
            <span class="difficulty-badge advanced-badge">高级</span>
            <h3><a href="/blog/limit-order-book-microstructure-analysis/">限价订单簿微观结构：高频交易的秘密武器</a></h3>
            <p>高频交易基础：限价订单簿(LOB)数据处理与微观结构分析</p>
            <span class="read-time">⏱️ 26 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">06</div>
          <div class="article-info">
            <span class="difficulty-badge advanced-badge">高级</span>
            <h3><a href="/blog/execution-systems-slippage-control/">实盘交易系统：执行算法与滑点控制</a></h3>
            <p>实盘技术：VWAP/TWAP执行算法、智能订单路由与滑点优化</p>
            <span class="read-time">⏱️ 24 分钟</span>
          </div>
        </div>
        
        <div class="article-item">
          <div class="article-index">07</div>
          <div class="article-info">
            <span class="difficulty-badge advanced-badge">高级</span>
            <h3><a href="/blog/live-trading-execution-slippage/">实盘交易系统核心技术：执行算法与滑点控制</a></h3>
            <p>实盘交易系统深度解析：算法执行、滑点控制、订单管理</p>
            <span class="read-time">⏱️ 26 分钟</span>
          </div>
        </div>
      </div>
    </section>

    <!-- 学习建议 -->
    <section class="learning-tips">
      <h2>💡 学习建议</h2>
      <div class="tips-grid">
        <div class="tip-card">
          <h3>📖 循序渐进</h3>
          <p>建议按照入门→中级→高级的顺序学习，每篇文章都建立在前文基础上。</p>
        </div>
        <div class="tip-card">
          <h3>💻 动手实践</h3>
          <p>每篇文章都包含Python代码示例，建议边学边敲代码加深理解。</p>
        </div>
        <div class="tip-card">
          <h3>📊 回测验证</h3>
          <p>不要盲目相信策略，用历史数据回测验证每个策略的有效性。</p>
        </div>
        <div class="tip-card">
          <h3>⚠️ 风险第一</h3>
          <p>量化交易不是圣杯，始终将风险管理和资金安全放在首位。</p>
        </div>
      </div>
    </section>
    
    <footer>
      <p>📊 量化交易实战专栏 | 持续更新中...</p>
      <p style="margin-top: 0.5rem;">想要完整PDF版本？<a href="/quant-column.pdf">点击下载</a></p>
    </footer>
  </div>
</body>
</html>'''

# Write to file
with open('/Users/halo/workspace/astro-blog/public/quant-column.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✅ Generated complete quant-column.html")
print(f"📊 File size: {len(html_content)} bytes")
