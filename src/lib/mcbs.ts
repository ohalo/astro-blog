/**
 * 欧洲期权蒙特卡洛定价的纯数学内核（无 DOM 依赖，可在浏览器与 node 里跑）
 *
 * 被两处使用：
 *  - QuantLabMC.astro：浏览器实时参数实验（<quant-lab-mc>）
 *  - scripts/test-mcbs.ts：数值核对（BSM 解析值 vs 已知基准、两种估计量的无偏与方差缩减）
 *
 * 约定：所有模拟用固定种子 mulberry32 + Box-Muller，结果稳定可复现。
 */

/** 确定性伪随机源（与 QuantLab 凯利实验同款） */
export const mulberry32 = (seed: number) => {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 标准正态分布函数，Abramowitz–Stegun 7.1.26，最大误差 ~1.5e-7 */
export function normCdf(x: number): number {
  if (x < 0) return 1 - normCdf(-x)
  const t = 1 / (1 + 0.2316419 * x)
  const pdf = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
  const poly =
    t *
    (0.319381530 +
      t *
        (-0.356563782 +
          t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
  return 1 - pdf * poly
}

export interface BsmParams {
  s: number
  k: number
  r: number
  vol: number
  t: number
  /** 1 看涨 / -1 看跌 */
  cp: 1 | -1
}

/** Black-Scholes-Merton 解析价（真值锚点） */
export function bsmPrice(p: BsmParams): number {
  const { s, k, r, vol, t, cp } = p
  if (t <= 0 || vol <= 0) return Math.max(0, cp * (s - k))
  const sq = vol * Math.sqrt(t)
  const d1 = (Math.log(s / k) + (r + 0.5 * vol * vol) * t) / sq
  const d2 = d1 - sq
  const df = Math.exp(-r * t)
  return cp === 1
    ? s * normCdf(d1) - k * df * normCdf(d2)
    : k * df * normCdf(-d2) - s * normCdf(-d1)
}

export interface McPaths {
  /** 终点收益（已贴现，按生成顺序）：crude 每路径一个 */
  crude: Float64Array
  /** 对偶变量配对的平均收益（长度 = crude 的一半），同生成顺序 */
  anti: Float64Array
}

/**
 * 在风险中性测度下按 GBM 模拟终点价格并取欧式收益。
 * pairs 条“随机线”，crude 视角 = 2·pairs 条独立路径；anti 把 z 与 −z 配对平均。
 * 两种估计量共用同一批 z，因此收敛曲线可直接对比。
 */
export function simulateEuropean(p: BsmParams, pairs: number, steps: number, seed: number): McPaths {
  const { s, k, r, vol, t, cp } = p
  const run = mulberry32(seed)
  const dt = t / steps
  const drift = (r - 0.5 * vol * vol) * dt
  const shock = vol * Math.sqrt(dt)
  const df = Math.exp(-r * t)

  const crude = new Float64Array(pairs * 2)
  const anti = new Float64Array(pairs)

  for (let i = 0; i < pairs; i++) {
    // 一条布朗线 W = Σ z_h（Box-Muller 第一分量）；对偶路径用 −W，
    // 因 −W 与 W 同分布，两条路径都无偏，而配对显著压低估计方差。
    let w = 0
    for (let h = 0; h < steps; h++) {
      let u1 = run()
      const u2 = run()
      if (u1 < 1e-12) u1 = 1e-12
      w += Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    }
    const base = drift * steps
    const stA = s * Math.exp(base + shock * w)
    const stB = s * Math.exp(base - shock * w)
    const payA = df * Math.max(0, cp * (stA - k))
    const payB = df * Math.max(0, cp * (stB - k))
    crude[i * 2] = payA
    crude[i * 2 + 1] = payB
    anti[i] = (payA + payB) / 2
  }
  return { crude, anti }
}

export interface ConvergePoint {
  n: number
  mean: number
  se: number
}

/** 在 log 间隔的检查点上回看累计均值与标准误（用于收敛曲线） */
export function convergeTrace(values: Float64Array, points = 24): ConvergePoint[] {
  const n = values.length
  const out: ConvergePoint[] = []
  for (let i = 1; i <= points; i++) {
    const m = Math.max(2, Math.round(Math.pow(n, i / points)))
    if (i > 1 && m === out[out.length - 1].n) continue
    let s = 0
    for (let j = 0; j < m; j++) s += values[j]
    const mean = s / m
    let v = 0
    for (let j = 0; j < m; j++) {
      const d = values[j] - mean
      v += d * d
    }
    out.push({ n: m, mean, se: Math.sqrt(v / (m - 1) / m) })
  }
  return out
}

/** 估计量的均值与标准误 */
export function meanSe(values: Float64Array): { mean: number; se: number } {
  const n = values.length
  let s = 0
  for (let i = 0; i < n; i++) s += values[i]
  const mean = s / n
  let v = 0
  for (let i = 0; i < n; i++) {
    const d = values[i] - mean
    v += d * d
  }
  return { mean, se: Math.sqrt(v / (n - 1) / n) }
}
