/**
 * 股票搜索 API（Vercel serverless 代理，模仿 /api/stock 的零 Python 直接查询模式）
 * 路由：/api/stock-search?q=茅台 | maotai | 600519
 * 数据源：腾讯 smartbox（支持中文、拼音、代码），返回 [{code, market, name}]
 *
 * 解析说明：腾讯返回形如 v_hint="sh~600519~\u8d35\u5dde\u8305\u53f0~gzmt~GP-A"
 * 以 ~ 分隔，字段依次为 market代码前缀 / 6位代码 / 名称(unicode转义) / 拼音 / 类型。
 * 我们只取 A 股（类型 GP-A / GP-AB / 指数等），过滤掉基金/债券。
 */
export const prerender = false

import type { APIRoute } from 'astro'

interface Hit { code: string; market: string; name: string; kind: string }

const UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/124.0'

// 类型后缀过滤：只保留股票/指数/ETF，排除基金(LC/OF)、债券(ZQ)等噪声
const KEEP = new Set(['GP-A', 'GP-AB', 'GP-B', 'ZS-A', 'JJ-LOF', 'JJ-ETF', 'JJ-OF', 'LC'])

export const GET: APIRoute = async ({ url }) => {
  const q = (url.searchParams.get('q') || '').trim()
  if (!q) return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'content-type': 'application/json; charset=utf-8' } })

  try {
    const r = await fetch(`https://smartbox.gtimg.cn/s3/?q=${encodeURIComponent(q)}&t=all`, {
      headers: { 'User-Agent': UA, Referer: 'https://stockapp.finance.qq.com/' },
      signal: AbortSignal.timeout(8000)
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    const body = await r.text()
    const m = body.match(/v_hint="([^"]*)"/)
    if (!m || !m[1] || m[1] === 'N') return new Response(JSON.stringify({ items: [] }), { status: 200, headers: { 'content-type': 'application/json; charset=utf-8' } })

    // 多组结果以 ^ 分隔（腾讯 smartbox），每组内部以 ~ 分隔
    const items: Hit[] = m[1]
      .split('^')
      .map((seg) => seg.split('~'))
      .filter((p) => p.length >= 5)
      .filter((p) => KEEP.has(p[4]))
      .map((p) => ({ market: p[0], code: p[1], name: JSON.parse(`"${p[2]}"`), kind: p[4] }))
      // 去重（smartbox 偶尔重复）
      .filter((h, i, arr) => arr.findIndex((x) => x.code === h.code) === i)
      .slice(0, 12)

    return new Response(JSON.stringify({ items }), {
      status: 200,
      headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'public, max-age=300, s-maxage=300' }
    })
  } catch (e: any) {
    return new Response(JSON.stringify({ items: [], error: String(e?.message || e) }), {
      status: 200,
      headers: { 'content-type': 'application/json; charset=utf-8' }
    })
  }
}
