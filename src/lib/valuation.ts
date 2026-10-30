/**
 * 股票内在价值估值计算（移植自 stock-intrinsic-value-calc 技能）
 *
 * 所有金额单位为元，股本单位为股，价格为每股。
 * 估值必须给出有假设支撑的区间，而非单一目标价：PE / PB / DCF 至少两法交叉验证。
 *
 * 用法示例（与技能脚本一致）：
 *   peScenario(shares, profit, peLow, peHigh)
 *   pbScenario(navps, pbLow, pbHigh)
 *   pegCheck(pe, growth)
 *   dcfScenario(shares, fcff, wacc, g, netDebt, minority)
 */

export interface PeScenario {
  eps: number
  low: number
  mid: number
  high: number
}

/** 市盈率法：合理股价 = 预期EPS × 合理PE。EPS = 归母净利润 / 总股本。 */
export function peScenario(
  shares: number,
  profit: number,
  peLow: number,
  peHigh: number
): PeScenario {
  if (!(shares > 0)) throw new Error('总股本必须为正')
  if (profit === 0) throw new Error('PE 法不适用于零利润公司')
  const lo = Math.min(peLow, peHigh)
  const hi = Math.max(peLow, peHigh)
  if (!(lo > 0) || !(hi > 0)) throw new Error('合理 PE 必须为正')
  const eps = profit / shares
  return { eps, low: eps * lo, mid: (eps * (lo + hi)) / 2, high: eps * hi }
}

export interface PbScenario {
  bvps: number
  low: number
  mid: number
  high: number
}

/** 市净率法：合理股价 = 每股净资产(BVPS) × 合理PB。适用于银行/保险/周期/重资产。 */
export function pbScenario(navps: number, pbLow: number, pbHigh: number): PbScenario {
  const lo = Math.min(pbLow, pbHigh)
  const hi = Math.max(pbLow, pbHigh)
  if (!(lo > 0) || !(hi > 0)) throw new Error('合理 PB 必须为正')
  return { bvps: navps, low: navps * lo, mid: (navps * (lo + hi)) / 2, high: navps * hi }
}

export interface PegResult {
  peg: number
  verdict: string
}

/** PEG 校验（PE 的辅助，非独立估值法）：PEG = PE / 净利润增速%。 */
export function pegCheck(pe: number, growth: number): PegResult {
  if (!(growth > 0)) throw new Error('增速需为正')
  if (!(pe > 0)) throw new Error('PE 需为正')
  const peg = pe / growth
  let verdict: string
  if (peg < 0.8) verdict = 'PEG 偏低：成长与估值匹配度占优，仍需验证增速可持续性'
  else if (peg <= 1.2) verdict = 'PEG 接近 1：成长股常见合理区间，不算明显便宜'
  else verdict = 'PEG 偏高：估值对增长预期敏感，需警惕增速下修'
  return { peg, verdict }
}

export interface DcfScenario {
  perShare: number
  terminalPct: number
  pvExplicit: number
  pvTerminal: number
  equityValue: number
  enterpriseValue: number
}

/**
 * 现金流折现 DCF（FCFF 两阶段）：
 *   EV = Σ FCFF_t/(1+WACC)^t + FCFF_n(1+g)/((WACC-g)(1+WACC)^n)
 *   每股价值 = (EV - 净债务 - 少数股东权益) / 总股本
 * 约束：g < WACC。终值占比 >70% 时估值对假设高度敏感，应给多情景。
 */
export function dcfScenario(
  shares: number,
  fcff: number[],
  wacc: number,
  g: number,
  netDebt = 0,
  minority = 0
): DcfScenario {
  if (!fcff.length) throw new Error('FCFF 序列不能为空')
  if (!(wacc > 0)) throw new Error('WACC 必须为正')
  if (!(g < wacc)) throw new Error('永续增长率 g 必须小于 WACC')
  if (!(shares > 0)) throw new Error('总股本必须为正')

  const pvExplicit = fcff.reduce((s, cf, i) => s + cf / Math.pow(1 + wacc, i + 1), 0)
  const terminal = (fcff[fcff.length - 1] * (1 + g)) / (wacc - g)
  const pvTerminal = terminal / Math.pow(1 + wacc, fcff.length)
  const enterpriseValue = pvExplicit + pvTerminal
  const equityValue = enterpriseValue - netDebt - minority
  const perShare = equityValue / shares

  return {
    perShare,
    terminalPct: (pvTerminal / enterpriseValue) * 100,
    pvExplicit,
    pvTerminal,
    equityValue,
    enterpriseValue
  }
}

/** DCF 三情景：对 WACC 与 g 做 ± 调整，悲观收紧、乐观放松。保证 g<WACC。 */
export interface DcfTrio {
  low: number
  mid: number
  high: number
}

export function dcfTrio(
  shares: number,
  fcff: number[],
  wacc: number,
  g: number,
  netDebt = 0,
  minority = 0
): DcfTrio {
  const waccP = Math.min(wacc + 0.015, (wacc + g) / 2 - 0.001) // 悲观：折现率更高
  const gP = Math.max(g - 0.005, 0)
  const waccO = Math.max(wacc - 0.015, 0.03) // 乐观：折现率更低
  const gO = g + 0.005
  const mid = dcfScenario(shares, fcff, wacc, g, netDebt, minority).perShare
  const low = dcfScenario(shares, fcff, waccP, gP, netDebt, minority).perShare
  const high = dcfScenario(shares, fcff, waccO, gO, netDebt, minority).perShare
  return { low: Math.min(low, mid, high), mid, high: Math.max(low, mid, high) }
}

export interface Zone {
  low: number
  mid: number
  high: number
}

/**
 * 综合估值区间：把多种方法的 [low, high] 合并为统一的悲观/中性/乐观。
 * 悲观取各法下限的最小值，乐观取各法上限的最大值，中性取各法中点均值。
 */
export function combineZones(zones: Zone[]): Zone {
  if (!zones.length) throw new Error('至少需要一种估值方法')
  const low = Math.min(...zones.map((z) => z.low))
  const high = Math.max(...zones.map((z) => z.high))
  const mid = zones.reduce((s, z) => s + z.mid, 0) / zones.length
  return { low, mid, high }
}

export type ValuationBand = 'undervalued' | 'fair' | 'overvalued'

/** 判断现价落在哪个估值区：低于悲观=低估，高于乐观=高估，之间=合理。 */
export function bandOf(zone: Zone, price: number): ValuationBand {
  if (price < zone.low) return 'undervalued'
  if (price > zone.high) return 'overvalued'
  return 'fair'
}
