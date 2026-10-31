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

/** DCF 三情景：对 WACC 与 g 做 ± 调整，悲观收紧（WACC↑、g↓）、乐观放松（WACC↓、g↑）。保证 g<WACC。 */
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
  const spread = 0.015
  const waccP = wacc + spread // 悲观：折现率更高 → 价值更低
  const gP = Math.max(g - spread, 0)
  const waccO = Math.max(wacc - spread, 0.02) // 乐观：折现率更低 → 价值更高
  const gO = Math.min(g + spread, waccO - 0.001)
  const mid = dcfScenario(shares, fcff, wacc, g, netDebt, minority).perShare
  const low = dcfScenario(shares, fcff, waccP, gP, netDebt, minority).perShare
  const high = dcfScenario(shares, fcff, waccO, gO, netDebt, minority).perShare
  return {
    low: Math.min(low, mid, high),
    mid,
    high: Math.max(low, mid, high)
  }
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

/* MARKER_ANCHOR_V1 */

/* =========================================================================
 * 基本面锚定（价格无关）
 *
 * 关键原则：内在价值区间【不能】由当日行情 PE/PB 反推。
 * 若把「合理 PE」取成当日 PE，则 合理价 = EPS × 当日PE = (价/PE) × PE = 现价，
 * 估值区间就退化成现价 ±20%，每天随股价漂移，且判定恒为「合理」——毫无信息量。
 *
 * 正确的锚点是价格无关的基本面：
 *   EPS  = 价 / PE_TTM    （价与 PE 同向，比值只随财报更新变动）
 *   BVPS = 价 / PB        （只随净资产更新变动）
 *   ROE  = EPS / BVPS = PB / PE_TTM （价格无关）
 * 再由 ROE 用戈登增长/剩余收益口径推合理倍数：
 *   PB = (ROE - g) / (COE - g)
 *   PE = PB / ROE
 * 三者都不含当日股价，因此区间日间恒定；只有财报、净资产、股本变动才会移动。
 * ========================================================================= */

export interface Fundamentals {
  eps: number
  bvps: number
  roe: number // 小数，如 0.203 表示 20.3%
  shares: number
}

/** 由行情推导价格无关的基本面。PE/PB 缺失或为负时对应字段为 0（表示不可用）。 */
export function deriveFundamentals(
  price: number,
  peTTM: number | null | undefined,
  pb: number,
  totalShares: number
): Fundamentals {
  const eps = price > 0 && peTTM != null && peTTM > 0 ? price / peTTM : 0
  const bvps = price > 0 && pb > 0 ? price / pb : 0
  const roe = bvps > 0 && eps > 0 ? eps / bvps : 0
  return { eps, bvps, roe, shares: totalShares || 0 }
}

/**
 * 合理 PB（戈登增长 / 剩余收益）：PB = (ROE - g) / (COE - g)
 * COE-g 过小、或结果非正时压到下限 0.2（ROE<g 属价值毁损）。
 */
export function justifiedPb(roe: number, coe: number, g: number): number {
  const denom = coe - g
  if (!(denom > 0.005)) return 0
  const v = (roe - g) / denom
  if (!isFinite(v)) return 0
  if (v <= 0.2) return 0.2
  if (v > 20) return 20
  return v
}

/** 合理 PE = 合理 PB / ROE（与 PB 口径内部一致）。 */
export function justifiedPe(roe: number, coe: number, g: number): number {
  const pb = justifiedPb(roe, coe, g)
  if (!(roe > 0) || !(pb > 0)) return 0
  return pb / roe
}

/**
 * 三情景合理倍数：悲观(COE↑、g↓)→倍数更低→价值更低；乐观(COE↓、g↑)→价值更高。
 * 返回的 low/high 即悲观/乐观，用于构造估值区间。
 */
export function justifiedBands(
  roe: number,
  coe: number,
  g: number,
  spread = 0.015
): { pbLow: number; pbHigh: number; peLow: number; peHigh: number } {
  const gP = Math.max(g - spread / 2, 0)
  const pbP = justifiedPb(roe, coe + spread, gP)
  const pbM = justifiedPb(roe, coe, g)
  const coeO = Math.max(coe - spread, 0.02)
  const gO = Math.min(g + spread / 2, Math.max(coeO - 0.01, 0))
  const pbO = justifiedPb(roe, coeO, gO)
  const toPe = (pb: number) => (roe > 0 && pb > 0 ? pb / roe : 0)
  const pes = [toPe(pbP), toPe(pbM), toPe(pbO)].filter((x) => x > 0)
  const pbs = [pbP, pbM, pbO].filter((x) => x > 0)
  return {
    pbLow: pbs.length ? Math.min(...pbs) : 0,
    pbHigh: pbs.length ? Math.max(...pbs) : 0,
    peLow: pes.length ? Math.min(...pes) : 0,
    peHigh: pes.length ? Math.max(...pes) : 0
  }
}

/** PEG=1 锚点（ROE 不可用时的回退）：合理 PE ≈ 净利增速%，悲观/乐观 ±15%。 */
export function pegBands(growthPct: number): { peLow: number; peHigh: number } {
  if (!(growthPct > 0)) return { peLow: 0, peHigh: 0 }
  return { peLow: growthPct * 0.85, peHigh: growthPct * 1.15 }
}

export type AnchorKind = 'roe' | 'peg' | 'manual' | 'none'

/**
 * 依据基本面推导合理倍数区间（自动锚定）。
 * 优先 ROE 推导；ROE 不可用则退回 PEG=1；都不行返回 none，需用户手填假设。
 */
export function deriveBands(
  f: Fundamentals,
  coe: number,
  g: number,
  growthPct: number
): { kind: AnchorKind; peLow: number; peHigh: number; pbLow: number; pbHigh: number } {
  if (f.roe > 0) {
    const b = justifiedBands(f.roe, coe, g)
    if (b.peLow > 0 && b.peHigh > 0) {
      return { kind: 'roe', peLow: b.peLow, peHigh: b.peHigh, pbLow: b.pbLow, pbHigh: b.pbHigh }
    }
  }
  const p = pegBands(growthPct)
  if (p.peLow > 0) {
    return { kind: 'peg', peLow: p.peLow, peHigh: p.peHigh, pbLow: 0, pbHigh: 0 }
  }
  return { kind: 'none', peLow: 0, peHigh: 0, pbLow: 0, pbHigh: 0 }
}

/** 相对变化率；(old,new) 都为 0 视为未变。 */
export function relChange(oldV: number, newV: number): number {
  if (oldV === 0 && newV === 0) return 0
  if (oldV === 0) return 1
  return (newV - oldV) / Math.abs(oldV)
}

/** 基本面是否在容差内未变（默认 0.5%）。用于判断估值区间是否应保持不变。 */
export function fundamentalsEqual(a: Fundamentals, b: Fundamentals, tol = 0.005): boolean {
  const eq = (x: number, y: number) => Math.abs(relChange(x, y)) <= tol
  return eq(a.eps, b.eps) && eq(a.bvps, b.bvps) && eq(a.roe, b.roe) && eq(a.shares, b.shares)
}
