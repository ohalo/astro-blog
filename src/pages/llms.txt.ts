import type { APIRoute } from 'astro'

import { getBlogCollection, sortMDByDate } from 'astro-pure/server'

const SITE = new URL('/', import.meta.env.SITE).href.replace(/\/$/, '')

/**
 * llms.txt：给 LLM / AI 搜索引擎的站点导览（llmstxt.org 格式）。
 * 数字与标题都从内容集合实时取，不手写、不会说谎。
 * 只列真正能代表本站的长文，不倾倒全部文章清单。
 */
const PILLARS: { section: string; ids: string[] }[] = [
  {
    section: '量化策略与风险管理',
    ids: [
      'johansen-cointegration-basket',
      'black-litterman-practical',
      '2026-06-04-kelly-criterion-position-sizing',
      'expectile-risk-evar',
      'order-flow-toxicity-vpin',
      'gamma-scalping-options',
      'convertible-bond-double-low',
      'execution-slippage-control'
    ]
  },
  {
    section: '机器学习 × 量化',
    ids: [
      'quant-backtesting-vectorbt-guide',
      'llm-quant-backtesting-practice',
      'time-series-foundation-model',
      'kan-kolmogorov-arnold-network',
      'deep-reinforcement-learning-quant-trading',
      'llm-financial-nlp-revolution'
    ]
  },
  {
    section: 'AI 工程实战',
    ids: ['openclaw-complete-guide']
  }
]

export const GET: APIRoute = async () => {
  const posts = sortMDByDate(await getBlogCollection())
  const byId = new Map(posts.map((post) => [post.id, post]))

  const sections = PILLARS.map(({ section, ids }) => {
    const lines = ids
      .map((id) => byId.get(id))
      .filter((post): post is (typeof posts)[number] => Boolean(post))
      .map((post) => `- [${post.data.title}](${SITE}/blog/${post.id})`)
    return `## ${section}\n\n${lines.join('\n')}`
  }).join('\n\n')

  const latest = posts.reduce(
    (max, post) => (post.data.updatedDate ?? post.data.publishDate) > max
      ? (post.data.updatedDate ?? post.data.publishDate)
      : max,
    posts[0]?.data.publishDate ?? new Date()
  )

  // 简介也用实测值：旧文案写「每篇通常 4000 字以上，含可运行代码」，
  // 实际中文字数中位数不到 2000，且约 1/4 文章没有代码块。
  const zhLength = (raw: string | undefined) =>
    (raw ?? '')
      .replace(/```[\s\S]*?```/g, '')
      .replace(/\$\$[\s\S]*?\$\$/g, '')
      .match(/[\u4e00-\u9fff]/g)?.length ?? 0
  const lengths = posts.map((post) => zhLength(post.body)).sort((a, b) => a - b)
  const medianZh = lengths[Math.floor(lengths.length / 2)] ?? 0
  const withCode = posts.filter((post) => /```/.test(post.body ?? '')).length

  const body = `# halo 的技术博客

> 中文个人技术博客，主线是「量化交易 × 大模型」：因子与统计套利、风险管理、
> 期权衍生品、回测工程，以及 LLM 在金融研究中的落地。
> 共 ${posts.length} 篇文章（正文中文字符中位数约 ${medianZh} 字，${withCode} 篇含代码块），最近更新于 ${latest.toISOString().slice(0, 10)}。
> 正文语言：简体中文；代码注释中英混合。

${sections}

## 在线工具

- [凯利公式在线计算器](${SITE}/tools/kelly): 输入胜率与盈亏比，输出全凯利/半凯利/四分之一凯利仓位、单笔几何增长率，以及蒙特卡洛爆仓概率模拟。

## 索引与栏目

- [全部文章](${SITE}/blog/)
- [量化专栏导航](${SITE}/quant-column)
- [按年归档](${SITE}/archives)
- [全部标签](${SITE}/tags)
- [关于作者](${SITE}/about)

## 机器可读入口

- RSS: ${SITE}/rss.xml
- Sitemap: ${SITE}/sitemap-index.xml

## 引用与转载

- 允许在引用片段的同时注明出处并链接到原文 URL。
- 文中代码示例默认 MIT 许可（文内另有声明者除外）。
- 请如实标注发布日期与作者（halo）。
`.trimEnd() + '\n'

  return new Response(body, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' }
  })
}
