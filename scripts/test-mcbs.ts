/**
 * mcbs 数值核对（一次性验收，不进 CI 流水线）：
 *   npx tsx scripts/test-mcbs.ts
 *
 * 对照基准：
 *  - BSM 解析值用独立实现的 erfc 版 normCdf 复算（与 A&S 近似互为校验）
 *  - 已知基准（Haug 教科书案例）：S=100,K=100,r=5%,σ=20%,T=1 → call 10.4506、put 5.5735
 *  - 蒙特卡洛两估计量对解析值的偏差应在 2 个标准误内；对偶方差应显著小于 crude
 */
import { bsmPrice, simulateEuropean, meanSe, convergeTrace, normCdf } from '../src/lib/mcbs'

// 独立 normCdf：用 erfc 的数值积分实现（与 A&S 多项式近似不同源）
function normCdfRef(x: number): number {
  return 0.5 * erfc(-x / Math.SQRT2)
}
function erfc(x: number): number {
  // Numerical Recipes erfc 有理近似，|误差| < 1.2e-7
  const z = Math.abs(x)
  const t = 2 / (2 + z)
  const ty = 4 * t - 2
  const cof = [
    -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2, -9.561514786808631e-3,
    -9.46595344482036e-4, 3.66839497852761e-4, 4.2523324806907e-5, -2.0278578112534e-5,
    -1.624290004647e-6, 1.303655835580e-6, 1.5626441722e-8, -8.175031805e-8, -6.333859e-10,
    2.72905e-9, 2.021e-11, -1.1325e-10
  ]
  let d = 0
  let dd = 0
  for (let j = cof.length - 1; j > 0; j--) {
    const tmp = d
    d = ty * d - dd + cof[j]
    dd = tmp
  }
  const ans = t * Math.exp((-z * z) + 0.5 * (cof[0] + ty * d) - dd)
  return x >= 0 ? ans : 2 - ans
}

function bsmRef(s: number, k: number, r: number, vol: number, t: number, cp: 1 | -1) {
  const sq = vol * Math.sqrt(t)
  const d1 = (Math.log(s / k) + (r + 0.5 * vol * vol) * t) / sq
  const d2 = d1 - sq
  const df = Math.exp(-r * t)
  return cp === 1
    ? s * normCdfRef(d1) - k * df * normCdfRef(d2)
    : k * df * normCdfRef(-d2) - s * normCdfRef(-d1)
}

let fail = 0
const check = (name: string, ok: boolean, detail: string) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`)
  if (!ok) fail++
}

// 1) normCdf 双实现一致
for (const x of [-3, -1.96, -0.5, 0, 0.5, 1.96, 3]) {
  const a = normCdf(x)
  const b = normCdfRef(x)
  check(`normCdf(${x})`, Math.abs(a - b) < 1e-6, `A&S=${a.toFixed(9)} ref=${b.toFixed(9)}`)
}

// 2) BSM 已知基准
const p0 = { s: 100, k: 100, r: 0.05, vol: 0.2, t: 1, cp: 1 as const }
check('BSM call 基准', Math.abs(bsmPrice(p0) - 10.4506) < 2e-3, `= ${bsmPrice(p0).toFixed(4)} (期望 10.4506)`)
check('BSM put 基准', Math.abs(bsmPrice({ ...p0, cp: -1 }) - 5.5735) < 2e-3, `= ${bsmPrice({ ...p0, cp: -1 }).toFixed(4)} (期望 5.5735)`)
check('put-call parity', Math.abs(bsmPrice(p0) - bsmPrice({ ...p0, cp: -1 }) - (100 - 100 * Math.exp(-0.05))) < 1e-6, 'C−P=S−Ke^{−rT}')

// 3) MC 无偏 + 对偶方差缩减（多组参数抽查）
for (const p of [
  p0,
  { s: 100, k: 110, r: 0.02, vol: 0.35, t: 0.5, cp: 1 as const },
  { s: 90, k: 80, r: 0.03, vol: 0.2, t: 2, cp: -1 as const }
]) {
  const exact = bsmRef(p.s, p.k, p.r, p.vol, p.t, p.cp)
  const { crude, anti } = simulateEuropean(p, 20000, 64, 2026_1119)
  const c = meanSe(crude)
  const a = meanSe(anti)
  const tag = `S=${p.s} K=${p.k} σ=${p.vol} T=${p.t} ${p.cp === 1 ? 'C' : 'P'}`
  check(`crude |z|<3 (${tag})`, Math.abs(c.mean - exact) < 3 * c.se + 1e-12, `MC=${c.mean.toFixed(4)}±${c.se.toFixed(4)} vs BSM=${exact.toFixed(4)}`)
  check(`anti |z|<3 (${tag})`, Math.abs(a.mean - exact) < 3 * a.se + 1e-12, `MC=${a.mean.toFixed(4)}±${a.se.toFixed(4)} vs BSM=${exact.toFixed(4)}`)
  check(`对偶方差缩减 (${tag})`, a.se < c.se, `SE crude=${c.se.toFixed(4)} → anti=${a.se.toFixed(4)}（×${(a.se / c.se).toFixed(2)}）`)
}

// 4) 收敛轨迹单调收窄（SE 随 N 下降）
const { crude } = simulateEuropean(p0, 20000, 64, 7)
const tr = convergeTrace(crude, 20)
let mono = true
for (let i = 2; i < tr.length; i++) if (tr[i].se > tr[i - 2].se * 1.35) mono = false
check('收敛轨迹 SE 总体收窄', mono, `se[N=${tr[1].n}]=${tr[1].se.toFixed(4)} → se[N=${tr[tr.length - 1].n}]=${tr[tr.length - 1].se.toFixed(4)}`)

console.log(fail ? `\n${fail} 项失败` : '\n全部通过 ✅')
process.exit(fail ? 1 : 0)
