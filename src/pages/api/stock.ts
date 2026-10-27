/**
 * 股票查询 API（Vercel 服务端代理）
 * 解决浏览器直接请求上游接口的 CORS 与 GBK 编码问题。
 * 路由：/api/stock?code=000002&market=sz
 *
 * 2026-10-27 数据源重构：
 *   东方财富 push2 / push2his 已从服务端网络不可达（连 Vercel 与本地都拿不到响应），
 *   导致 kline=[] / fundFlow=null，页面上 K线与资金流向整体消失。
 *   现改为多源冗余：
 *     - K线    ：腾讯 web.ifzq.gtimg.cn（主） → 新浪 money.finance.sina.com.cn（备）
 *     - 资金流 ：新浪 vip.stock.finance.sina.com.cn（主） → 东方财富 push2（备）
 *     - 52周高低：由 250 日 K线本地计算（不再依赖上游字段）
 *     - 行业   ：东方财富 datacenter-web（主，尽力而为，失败降级为 null）
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
  sources: Record<string, string> // 各字段实际命中的数据源（便于排障）
}

async function fetchJSON(url: string, referer: string, timeout = 9000) {
  const r = await fetch(url, {
    headers: {
      'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
      Referer: referer,
      Accept: '*/*'
    },
    signal: AbortSignal.timeout(timeout)
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

async function fetchBuf(url: string, referer: string, timeout = 9000) {
  const r = await fetch(url, {
    headers: {
      'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
      Referer: referer
    },
    signal: AbortSignal.timeout(timeout)
  })
  return Buffer.from(await r.arrayBuffer())
}

function resolveMarket(code: string, market?: string) {
  const c = code.trim()
  let prefix = 'sz'
  let secidMarket = '0'
  let exchange = 'SZ' // 东方财富 datacenter 后缀
  if (market === 'sh') {
    prefix = 'sh'
    secidMarket = '1'
    exchange = 'SH'
  } else if (market === 'bj') {
    prefix = 'bj'
    secidMarket = '0'
    exchange = 'BJ'
  } else if (market === 'sz') {
    prefix = 'sz'
    secidMarket = '0'
    exchange = 'SZ'
  } else {
    // 自动识别
    if (/^(6|9)/.test(c)) {
      prefix = 'sh'
      secidMarket = '1'
      exchange = 'SH'
    } else if (/^(0|2|3)/.test(c)) {
      prefix = 'sz'
      secidMarket = '0'
      exchange = 'SZ'
    } else if (/^(4|8)/.test(c)) {
      prefix = 'bj'
      secidMarket = '0'
      exchange = 'BJ'
    }
  }
  return { prefix, secid: `${secidMarket}.${c}`, secucode: `${c}.${exchange}` }
}

/** 核心行情：腾讯 qt.gtimg.cn（GBK） */
async function getQuote(code: string, prefix: string) {
  const buf = await fetchBuf(`https://qt.gtimg.cn/q=${prefix}${code}`, 'https://gu.qq.com/')
  const text = new TextDecoder('gbk').decode(buf)
  const m = text.match(/="([^"]+)"/)
  if (!m) throw new Error('无法解析行情数据')
  const a = m[1].split('~')
  const num = (s: string | undefined, d = 0) =>
    s === undefined || s === '' ? d : Number(s)
  if (!a[1] || !a[3]) throw new Error('行情返回为空，代码可能不存在')
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

/**
 * K线（主源）：腾讯 web.ifzq.gtimg.cn
 * 复权参数不同返回 key 不同：qfq→qfqday、hfq→hfqday、nfq→day，
 * 指数/港股一律返回 day —— 三种 key 都要兜住。
 * 画图用 qfq（前复权，除权不断崖），52 周高低用 nfq（不复权，与行情软件口径一致）。
 */
async function getKlineTencent(
  code: string,
  prefix: string,
  n: number,
  fq: 'qfq' | 'nfq'
): Promise<KlineDaily[]> {
  const p = `${prefix}${code}`
  const j = await fetchJSON(
    `https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=${p},day,,,${n},${fq}`,
    'https://gu.qq.com/'
  )
  const node = j?.data?.[p]
  const arr: any[] = node?.qfqday || node?.hfqday || node?.day || []
  if (!Array.isArray(arr) || arr.length === 0) return []
  return arr
    .map((k: any) => ({
      t: String(k[0]),
      o: Number(k[1]),
      c: Number(k[2]),
      h: Number(k[3]),
      l: Number(k[4]),
      v: Number(k[5])
    }))
    .filter((d) => d.c > 0)
}

/** K线（备源）：新浪 */
async function getKlineSina(code: string, prefix: string, n: number): Promise<KlineDaily[]> {
  const j = await fetchJSON(
    `https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=${prefix}${code}&scale=240&ma=5&datalen=${n}`,
    'https://finance.sina.com.cn/'
  )
  if (!Array.isArray(j) || j.length === 0) return []
  return j.map((k: any) => ({
    t: String(k.day).slice(0, 10),
    o: Number(k.open),
    c: Number(k.close),
    h: Number(k.high),
    l: Number(k.low),
    v: Number(k.volume) / 100 // 股 → 手
  }))
}

async function getKline(code: string, prefix: string, n: number, fq: 'qfq' | 'nfq' = 'qfq') {
  try {
    const k = await getKlineTencent(code, prefix, n, fq)
    if (k.length >= 2) return { data: k, source: `tencent-${fq}` }
  } catch {
    /* fall through */
  }
  try {
    const k = await getKlineSina(code, prefix, n)
    if (k.length >= 2) return { data: k, source: 'sina' }
  } catch {
    /* fall through */
  }
  return { data: [] as KlineDaily[], source: 'none' }
}

/**
 * 资金流（主源）：新浪
 * r0=超大单 r1=大单 r2=中单 r3=小单，*_in / *_out 为流入流出额（元）
 */
async function getFundFlowSina(code: string, prefix: string) {
  const j = await fetchJSON(
    `https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssi_ssfx_flzjtj?daima=${prefix}${code}`,
    'https://finance.sina.com.cn/'
  )
  if (!j || typeof j !== 'object' || j.r0_in == null) return null
  const n = (k: string) => Number(j[k] || 0)
  const huge = n('r0_in') - n('r0_out')
  const big = n('r1_in') - n('r1_out')
  const mid = n('r2_in') - n('r2_out')
  const small = n('r3_in') - n('r3_out')
  return { main: huge + big, huge, big, mid, small }
}

/** 资金流（备源）：东方财富 push2（字段序：日期,主力,小单,中单,大单,超大单） */
async function getFundFlowEM(secid: string) {
  const j = await fetchJSON(
    `https://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=1&klt=1&secid=${secid}` +
      `&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65`,
    'https://quote.eastmoney.com/'
  )
  const k = j?.data?.klines?.[0]
  if (!k) return null
  const p = String(k).split(',')
  const main = Number(p[1] || 0)
  const small = Number(p[2] || 0)
  const mid = Number(p[3] || 0)
  const big = Number(p[4] || 0)
  const huge = Number(p[5] || 0)
  if (main === 0 && small === 0 && mid === 0 && big === 0 && huge === 0) return null
  return { main, huge, big, mid, small }
}

async function getFundFlow(code: string, prefix: string, secid: string) {
  try {
    const f = await getFundFlowSina(code, prefix)
    if (f) return { data: f, source: 'sina' }
  } catch {
    /* fall through */
  }
  try {
    const f = await getFundFlowEM(secid)
    if (f) return { data: f, source: 'eastmoney' }
  } catch {
    /* fall through */
  }
  return { data: null, source: 'none' }
}

/** 所属行业：东方财富 datacenter-web（尽力而为） */
async function getIndustry(secucode: string): Promise<string | null> {
  try {
    const j = await fetchJSON(
      `https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_F10_BASIC_ORGINFO` +
        `&columns=SECUCODE,INDUSTRYCSRC1&filter=(SECUCODE%3D%22${encodeURIComponent(secucode)}%22)`,
      'https://data.eastmoney.com/'
    )
    const v = j?.result?.data?.[0]?.INDUSTRYCSRC1
    return v ? String(v) : null
  } catch {
    return null
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

  const { prefix, secid, secucode } = resolveMarket(code, market)

  try {
    // 报价是主数据，失败即整体失败
    const quote = await getQuote(code, prefix)

    // K线取 250 日：最后 60 日用于画图，整段用于算 52 周高低
    const [kQfq, kNfq, fundFlowRes, industry] = await Promise.all([
      getKline(code, prefix, 250, 'qfq'),
      getKline(code, prefix, 250, 'nfq'),
      getFundFlow(code, prefix, secid),
      getIndustry(secucode)
    ])

    const chart = (kQfq.data.length >= 2 ? kQfq.data : kNfq.data).slice(-60)
    // 52 周高低优先用不复权口径（与行情软件一致），取不到就退回前复权
    const base = kNfq.data.length >= 2 ? kNfq.data : kQfq.data
    const highs = base.map((d) => d.h).filter((x) => x > 0)
    const lows = base.map((d) => d.l).filter((x) => x > 0)

    const result: StockResult = {
      ...quote,
      market: prefix,
      week52High: highs.length ? Math.max(...highs) : null,
      week52Low: lows.length ? Math.min(...lows) : null,
      industry,
      kline: chart,
      fundFlow: fundFlowRes.data,
      sources: {
        quote: 'tencent',
        kline: kQfq.data.length >= 2 ? kQfq.source : kNfq.source,
        week52: kNfq.data.length >= 2 ? kNfq.source : kQfq.source,
        fundFlow: fundFlowRes.source,
        industry: industry ? 'eastmoney-datacenter' : 'none'
      }
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
      JSON.stringify({ error: '行情获取失败，请检查代码或稍后重试', detail: String(e) }),
      {
        status: 502,
        headers: { 'content-type': 'application/json; charset=utf-8' }
      }
    )
  }
}
