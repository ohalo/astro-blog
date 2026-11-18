import type { APIRoute } from 'astro'

// 独立的 User-agent 行属于同一组，必须紧跟规则才生效；
// 旧版三组后面直接空行接 User-agent:* Allow:/，等价于允许 AI 爬虫抓取（规则失效）。
const robotsTxt = `
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
