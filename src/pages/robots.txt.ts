import type { APIRoute } from 'astro'

// 独立的 User-agent 行属于同一组，必须紧跟规则才生效；
// 旧版三组后面直接空行接 User-agent:* Allow:/，等价于允许 AI 爬虫抓取（规则失效）。
const robotsTxt = `
# 训练类爬虫（用于模型训练）继续按站点意愿屏蔽；
# 下面显式放行的是「搜索发现类」爬虫，它们只决定 AI 搜索引擎能不能收录/引用本站，
# 与训练无关，屏蔽会直接损失 ChatGPT / Perplexity / Claude 的引用机会。
User-agent: OAI-SearchBot
Allow: /

User-agent: PerplexityBot
Allow: /

User-agent: Claude-SearchBot
Allow: /

User-agent: GPTBot
User-agent: ClaudeBot
User-agent: Claude-Web
Disallow: /

User-agent: *
Allow: /

Sitemap: ${new URL('sitemap-index.xml', import.meta.env.SITE).href}
`.trim()

export const GET: APIRoute = () =>
  new Response(robotsTxt, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8'
    }
  })
