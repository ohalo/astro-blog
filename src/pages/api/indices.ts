/**
 * 指数行情带 API（Vercel serverless 代理，零 Python 直接查询）
 * 路由：/api/indices
 * 数据源：腾讯 qt.gtimg.cn（与 /api/stock 同族），一次聚合 5 大指数实时报价。
 * 替代原先需要 baostock 入库 + 发布 index/daily.json 的链路。
 */
export const prerender = false

import type { APIRoute } from 'astro'

interface Idx { code: string; market: string; name: string; price: number; chg: number; chgPct: number }

// 5 大指数：腾讯指数代码 = market前缀 + 6位，market 用 sh/sz
const INDICES = [
  { code: '000001', market: 'sh', name: '上证指数' },
  { code: '399001', market: 'sz', name: '深证成指' },
  { code: '399006', market: 'sz', name: '创业板指' },
  { code: '000300', market: 'sh', name: '沪深300' },
  { code: '000905', market: 'sh', name: '中证500' }
]

const UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/124.0'

async function fetchBuf(url: string): Promise<Buffer> {
  const r = await fetch(url, { headers: { 'User-Agent': UA, Referer: 'https://gu.qq.com/' }, signal: AbortSignal.timeout(9000) })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return Buffer.from(await r.arrayBuffer())
}

async function getOne(code: string, market: string): Promise<Idx> {
  const buf = await fetchBuf(`https://qt.gtimg.cn/q=${market}${code}`)
  const text = new TextDecoder('gbk').decode(buf)
  const m = text.match(/="([^"]+)"/)
  if (!m) throw new Error('解析失败')
  const a = m[1].split('~')
  const num = (s: string | undefined, d = 0) => (s === undefined || s === '' ? d : Number(s))
  if (!a[3]) throw new Error('为空')
  return {
    code,
    market,
    name: a[1]?.trim() || code,
    price: num(a[3]),
    chg: num(a[31]),
    chgPct: num(a[32])
  }
}

export const GET: APIRoute = async () => {
  const results = await Promise.allSettled(INDICES.map((i) => getOne(i.code, i.market)))
  const items = results
    .map((r, i) => (r.status === 'fulfilled' ? r.value : null))
    .filter((x): x is Idx => x !== null)
  return new Response(JSON.stringify({ items, asOf: new Date().toISOString() }), {
    status: 200,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'public, max-age=30, s-maxage=30' }
  })
}
