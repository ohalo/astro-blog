/**
 * 股票查询 API（Vercel 服务端代理）
 * 解决浏览器直接请求东方财富/腾讯接口的 CORS 与 GBK 编码问题。
 * 路由：/api/stock?code=000002&market=sz
 * 数据源：
 *  - 腾讯 qt.gtimg.cn  （核心行情，GBK → UTF-8 解码）
 *  - 东方财富 push2 / push2his（K线、资金流、行业、52周高低，UTF-8 JSON）
 */
export const prerender = false

import type { APIRoute } from 'astro'

interface KlineDaily {
  t: string // 日期 YYYY-MM-DD
  o: number // 开盘
  c: number // 收盘
  h: number // 最高
  l: number // 最低
  v: number // 成交量（手）
}

interface StockResult {
  code: string
  name: string
  market: string
  price: number
  prevClose: number
  open: number
  high: number
  low: number
  change: number
  changePct: number
  volume: number // 手
  amount: number // 元
  turnover: number // 换手率 %
  amplitude: number // 振幅 %
  floatCap: number // 元
  totalCap: number // 元
  pb: number
  peTTM: number | null
  limitUp: number
  limitDown: number
  volumeRatio: number
  totalShares: number // 股
  week52High: number | null
  week52Low: number | null
  industry: string | null
  time: string // YYYYMMDDHHMMSS
  kline: KlineDaily[] // 近 60 日 K 线（OHLC + 成交量）
  fundFlow: {
    main: number // 主力净流入 元
    huge: number // 超大单 元
    big: number // 大单 元
    mid: number // 中单 元
    small: number // 小单 元
  } | null
}

function resolveMarket(code: string, market?: string) {
  const c = code.trim()
  let prefix = 'sz'
  let secidMarket = '0'
  if (market === 'sh') {
    prefix = 'sh'
    secidMarket = '1'
  } else if (market === 'bj') {
    prefix = 'bj'
    secidMarket = '0'
  } else if (market === 'sz') {
    prefix = 'sz'
    secidMarket = '0'
  } else {
    // 自动识别
    if (/^(6|9)/.test(c)) {
      prefix = 'sh'
      secidMarket = '1'
    } else if (/^(0|2|3)/.test(c)) {
      prefix = 'sz'
      secidMarket = '0'
    } else if (/^(4|8)/.test(c)) {
      prefix = 'bj'
      secidMarket = '0'
    }
  }
  return { prefix, secid: `${secidMarket}.${c}` }
}

async function fetchText(url: string, referer: string) {
  const r = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0', Referer: referer },
    signal: AbortSignal.timeout(8000)
  })
  return Buffer.from(await r.arrayBuffer())
}

async function getQuote(code: string, prefix: string) {
  const buf = await fetchText(
    `https://qt.gtimg.cn/q=${prefix}${code}`,
    'https://gu.qq.com/'
  )
  // 腾讯接口返回 GBK，Node 22 内置 TextDecoder 支持 gbk
  const text = new TextDecoder('gbk').decode(buf)
  const m = text.match(/="([^"]+)"/)
  if (!m) throw new Error('无法解析行情数据')
  const a = m[1].split('~')
  const num = (s: string | undefined, d = 0) =>
    s === undefined || s === '' ? d : Number(s)
  const price = num(a[3])
  const totalCap = num(a[45]) * 1e8 // 亿 → 元
  const totalShares = price > 0 ? totalCap / price : 0
  return {
    name: a[1]?.trim() || '',
    code: a[2] || code,
    price,
    prevClose: num(a[4]),
    open: num(a[5]),
    high: num(a[33]),
    low: num(a[34]),
    change: num(a[31]),
    changePct: num(a[32]),
    volume: num(a[6]), // 手
    amount: num(a[37]) * 1e4, // 万 → 元
    turnover: num(a[38]),
    amplitude: num(a[43]),
    floatCap: num(a[44]) * 1e8, // 亿 → 元
    totalCap,
    pb: num(a[46]),
    peTTM: a[39] === '' || a[39] === undefined ? null : num(a[39]),
    limitUp: num(a[47]),
    limitDown: num(a[48]),
    volumeRatio: num(a[49]),
    totalShares,
    time: a[30] || ''
  }
}

async function getKline(secid: string): Promise<KlineDaily[]> {
  const url =
    `https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=${secid}` +
    `&klt=101&fqt=0&lmt=60&end=20500101` +
    `&fields1=f1,f2,f3,f4,f5&fields2=f51,f52,f53,f54,f55,f56`
  const r = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0', Referer: 'https://quote.eastmoney.com/' },
    signal: AbortSignal.timeout(8000)
  })
  const j = await r.json()
  const klines: string[] = j?.data?.klines || []
  // 字段顺序：日期,开盘,收盘,最高,最低,成交量
  return klines.map((k) => {
    const p = k.split(',')
    return {
      t: p[0],
      o: Number(p[1]),
      c: Number(p[2]),
      h: Number(p[3]),
      l: Number(p[4]),
      v: Number(p[5])
    }
  })
}

async function getFundFlow(secid: string) {
  const url =
    `https://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=1&klt=1&secid=${secid}` +
    `&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65`
  const r = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0', Referer: 'https://quote.eastmoney.com/' },
    signal: AbortSignal.timeout(8000)
  })
  const j = await r.json()
  const k = j?.data?.klines?.[0]
  if (!k) return null
  const p = k.split(',')
  return {
    main: Number(p[1] || 0),
    huge: Number(p[2] || 0),
    big: Number(p[3] || 0),
    mid: Number(p[4] || 0),
    small: Number(p[5] || 0)
  }
}

async function getEnrich(secid: string) {
  try {
    const url =
      `https://push2.eastmoney.com/api/qt/stock/get?secid=${secid}` +
      `&fields=f100,f174,f175`
    const r = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0', Referer: 'https://quote.eastmoney.com/' },
      signal: AbortSignal.timeout(8000)
    })
    const j = await r.json()
    const d = j?.data || {}
    return {
      industry: d.f100 ? String(d.f100).replace(/[\[\]"]/g, '') : null,
      week52High: d.f174 != null ? Number(d.f174) / 100 : null,
      week52Low: d.f175 != null ? Number(d.f175) / 100 : null
    }
  } catch {
    return { industry: null, week52High: null, week52Low: null }
  }
}

export const GET: APIRoute = async ({ url }) => {
  const code = (url.searchParams.get('code') || '').trim()
  const market = url.searchParams.get('market') || undefined
  if (!/^\d{4,6}$/.test(code)) {
    return new Response(JSON.stringify({ error: '请输入有效的股票代码（4-6 位数字）' }), {
      status: 400,
      headers: { 'content-type': 'application/json; charset=utf-8' }
    })
  }

  const { prefix, secid } = resolveMarket(code, market)
  try {
    const [quote, kline, fundFlow, enrich] = await Promise.all([
      getQuote(code, prefix),
      getKline(secid).catch(() => []),
      getFundFlow(secid).catch(() => null),
      getEnrich(secid)
    ])

    const result: StockResult = {
      ...quote,
      market: prefix,
      week52High: enrich.week52High,
      week52Low: enrich.week52Low,
      industry: enrich.industry,
      kline,
      fundFlow
    }

    return new Response(JSON.stringify(result), {
      status: 200,
      headers: {
        'content-type': 'application/json; charset=utf-8',
        'cache-control': 'public, max-age=30, s-maxage=30'
      }
    })
  } catch (e) {
    return new Response(
      JSON.stringify({ error: '行情获取失败，请稍后重试', detail: String(e) }),
      {
        status: 502,
        headers: { 'content-type': 'application/json; charset=utf-8' }
      }
    )
  }
}
