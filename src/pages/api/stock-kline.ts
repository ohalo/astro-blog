/**
 * 深度 K线代理（按需加载，避免默认请求过重）。
 * 路由：/api/stock-kline?code=600519&n=1000
 *   n = 60 / 250 / 500 / 1000（默认 250，上限 1000）
 * 数据源：新浪（scale=240 日线，必须带 ma=no，否则只返回 60 根）；
 *         腾讯 fqkline 作为备源（可能被 WAF 限流）。
 * 用途：/dash 图表切到 3年/全部区间时按需拉取更深历史。
 */
export const prerender = false

import type { APIRoute } from 'astro'

interface Bar {
  t: string
  o: number
  h: number
  l: number
  c: number
  v: number
}

const UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'

function marketOf(code: string): string {
  if (!/^\d{6}$/.test(code)) throw new Error('代码需为 6 位数字')
  if (code[0] === '6') return 'sh'
  if (code[0] === '0' || code[0] === '3') return 'sz'
  if (['4', '8', '9'].includes(code[0])) return 'bj'
  throw new Error('无法识别市场')
}

/** 新浪日线（带 ma=no 才能返回满 datalen 根） */
async function fromSina(prefix: string, code: string, n: number): Promise<Bar[]> {
  const url = `https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=${prefix}${code}&scale=240&ma=no&datalen=${n}`
  const r = await fetch(url, {
    headers: { 'User-Agent': UA, Referer: 'https://finance.sina.com.cn/' },
    signal: AbortSignal.timeout(10000)
  })
  if (!r.ok) throw new Error(`sina HTTP ${r.status}`)
  const j = await r.json()
  if (!Array.isArray(j) || !j.length) throw new Error('sina empty')
  return j
    .map((k: any) => ({
      t: String(k.day).slice(0, 10),
      o: Number(k.open),
      h: Number(k.high),
      l: Number(k.low),
      c: Number(k.close),
      v: Number(k.volume) / 100 // 股 → 手
    }))
    .filter((b: Bar) => Number.isFinite(b.c) && b.c > 0)
}

/** 腾讯前复权日线（备源） */
async function fromTencent(prefix: string, code: string, n: number): Promise<Bar[]> {
  const p = `${prefix}${code}`
  const r = await fetch(
    `https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=${p},day,,,${n},qfq`,
    { headers: { 'User-Agent': UA, Referer: 'https://gu.qq.com/' }, signal: AbortSignal.timeout(10000) }
  )
  if (!r.ok) throw new Error(`tencent HTTP ${r.status}`)
  const j = await r.json()
  const node = j?.data?.[p]
  const arr: any[] = node?.qfqday || node?.hfqday || node?.day || []
  if (!Array.isArray(arr) || !arr.length) throw new Error('tencent empty')
  return arr
    .map((k: any) => ({
      t: String(k[0]),
      o: Number(k[1]),
      c: Number(k[2]),
      h: Number(k[3]),
      l: Number(k[4]),
      v: Number(k[5])
    }))
    .filter((b: Bar) => Number.isFinite(b.c) && b.c > 0)
}

export const GET: APIRoute = async ({ url }) => {
  const code = url.searchParams.get('code') || ''
  const n = Math.min(1000, Math.max(30, Number(url.searchParams.get('n')) || 250))
  const json = (obj: unknown, status = 200) =>
    new Response(JSON.stringify(obj), {
      status,
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'public, max-age=300' }
    })
  try {
    const prefix = marketOf(code)
    let bars: Bar[] = []
    let source = 'sina'
    try {
      bars = await fromSina(prefix, code, n)
    } catch {
      bars = await fromTencent(prefix, code, n)
      source = 'tencent'
    }
    return json({ code, n, source, bars, count: bars.length })
  } catch (e: any) {
    return json({ code, n, source: 'none', bars: [], count: 0, error: String(e?.message || e) })
  }
}
