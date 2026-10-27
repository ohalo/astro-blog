/**
 * 生成黑金主题的金币头像（¥ 符号）与全套 favicon
 *
 * 设计要点：
 * - ¥ 符号用 SVG 路径绘制而非字体，渲染不依赖系统字体
 * - 金属质感靠多层渐变 + 压铸细圈 + 斜面高光叠加
 * - 头像留透明边距（避免被 rounded-full 裁切），favicon 满幅
 *
 * 用法：node gen_gold_coin.mjs
 */
import sharp from 'sharp'
import { writeFileSync, mkdirSync } from 'node:fs'

const S = 512

// ---- ¥ 符号几何（画布中心 256,256）----
const V_LEFT = 'M166,152 L256,274'
const V_RIGHT = 'M346,152 L256,274'
const STEM = 'M256,274 L256,384'
const STROKE_W = 30
const BARS = [
  { x: 180, y: 300, w: 152, h: 21 },
  { x: 180, y: 342, w: 152, h: 21 }
]

const glyphPaths = `
    <path d="${V_LEFT}" fill="none" stroke-width="${STROKE_W}" />
    <path d="${V_RIGHT}" fill="none" stroke-width="${STROKE_W}" />
    <path d="${STEM}" fill="none" stroke-width="${STROKE_W}" />
    ${BARS.map((b) => `<rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="3" />`).join('')}
`

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${S}" height="${S}" viewBox="0 0 ${S} ${S}">
  <defs>
    <radialGradient id="face" cx="34%" cy="26%" r="92%">
      <stop offset="0%" stop-color="#FDF0BC"/>
      <stop offset="22%" stop-color="#F0D584"/>
      <stop offset="48%" stop-color="#DDB849"/>
      <stop offset="72%" stop-color="#C39A28"/>
      <stop offset="90%" stop-color="#9C7718"/>
      <stop offset="100%" stop-color="#7A5C11"/>
    </radialGradient>
    <linearGradient id="rim" x1="18%" y1="0%" x2="82%" y2="100%">
      <stop offset="0%" stop-color="#FBE9A8"/>
      <stop offset="22%" stop-color="#D9B348"/>
      <stop offset="50%" stop-color="#8A6A16"/>
      <stop offset="74%" stop-color="#C79B2B"/>
      <stop offset="100%" stop-color="#6A5010"/>
    </linearGradient>
    <linearGradient id="glyph" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FFF8DC"/>
      <stop offset="38%" stop-color="#F4DC8E"/>
      <stop offset="70%" stop-color="#D9B040"/>
      <stop offset="100%" stop-color="#B98D1C"/>
    </linearGradient>
    <radialGradient id="innerShade" cx="50%" cy="50%" r="50%">
      <stop offset="82%" stop-color="#6B4E0C" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#6B4E0C" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="gloss" x1="12%" y1="4%" x2="72%" y2="62%">
      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.42"/>
      <stop offset="45%" stop-color="#FFFFFF" stop-opacity="0.10"/>
      <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0"/>
    </linearGradient>
  </defs>

  <circle cx="256" cy="256" r="256" fill="url(#rim)"/>
  <circle cx="256" cy="256" r="233" fill="#7A5C11"/>
  <circle cx="256" cy="256" r="226" fill="url(#face)"/>
  <circle cx="256" cy="256" r="226" fill="url(#innerShade)"/>
  <circle cx="256" cy="256" r="205" fill="none" stroke="#7A5C11" stroke-opacity="0.5" stroke-width="3"/>
  <circle cx="256" cy="256" r="197" fill="none" stroke="#FFF3C4" stroke-opacity="0.32" stroke-width="1.6"/>

  <g stroke-linecap="butt" stroke-linejoin="miter">
    <g transform="translate(0,4)" stroke="#6B4E0C" stroke-opacity="0.5" fill="#6B4E0C">${glyphPaths}</g>
    <g transform="translate(0,-2)" stroke="#FFF6D0" stroke-opacity="0.42" fill="#FFF6D0">${glyphPaths}</g>
    <g stroke="url(#glyph)" fill="url(#glyph)">${glyphPaths}</g>
  </g>

  <circle cx="256" cy="256" r="226" fill="url(#gloss)"/>
</svg>`

// 头像：金币内缩 10px，留透明边距
const avatarSvg = svg
  .replace('r="256" fill="url(#rim)"', 'r="246" fill="url(#rim)"')
  .replace('r="233"', 'r="225"')
  .replace(/r="226"/g, 'r="218"')
  .replace('r="205"', 'r="197"')
  .replace('r="197" fill="none" stroke="#FFF3C4"', 'r="189" fill="none" stroke="#FFF3C4"')

/**
 * favicon 专用：小尺寸下 ¥ 笔画会糊成一片，所以做一版加粗变体
 * —— 笔画加粗到 46、放大到 1.12 倍、去掉 1.6px 的高光细圈
 */
const faviconSvg = svg
  .replace('stroke-width="30"', 'stroke-width="46"')
  .replace('translate(0,4)', 'translate(0,5)')
  .replace('translate(0,-2)', 'translate(0,-3)')
  .replace(
    '<g stroke="url(#glyph)" fill="url(#glyph)">',
    '<g transform="translate(256,256) scale(1.12) translate(-256,-256)" stroke="url(#glyph)" fill="url(#glyph)">'
  )
  .replace('<circle cx="256" cy="256" r="197" fill="none" stroke="#FFF3C4" stroke-opacity="0.32" stroke-width="1.6"/>', '')

const render = (source, size, opts = {}) =>
  sharp(Buffer.from(source), { density: 384 })
    .resize(size, size, { fit: 'cover', background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png(opts)
    .toBuffer()

mkdirSync('tmp', { recursive: true })

// 1. 头像（调色板 PNG，视觉无损但体积小 3 倍）
const avatar = await render(avatarSvg, S, { palette: true, quality: 92, compressionLevel: 9 })
writeFileSync('src/assets/avatar.png', avatar)
console.log(`src/assets/avatar.png            ${(avatar.length / 1024).toFixed(1)} KB`)

// 2. favicon 满幅版（加粗字形）
const favicons = {
  'public/favicon/favicon-16x16.png': 16,
  'public/favicon/favicon-32x32.png': 32,
  'public/favicon/apple-touch-icon.png': 180,
  'public/favicon/android-chrome-192x192.png': 192,
  'public/favicon/android-chrome-512x512.png': 512
}
for (const [path, size] of Object.entries(favicons)) {
  // 大尺寸（≥64）用常规字形，小尺寸用加粗字形
  const src = size >= 64 ? svg : faviconSvg
  const buf = await render(src, size, { palette: true, quality: 92, compressionLevel: 9 })
  writeFileSync(path, buf)
  console.log(`${path.padEnd(42)} ${(buf.length / 1024).toFixed(1)} KB`)
}

// 3. favicon.ico（内嵌 16/32/64 三张 PNG，小尺寸用加粗字形）
const icoSizes = [16, 32, 64]
const icoImages = []
for (const size of icoSizes)
  icoImages.push(await render(size >= 64 ? svg : faviconSvg, size))

const header = Buffer.alloc(6)
header.writeUInt16LE(0, 0) // reserved
header.writeUInt16LE(1, 2) // type: icon
header.writeUInt16LE(icoSizes.length, 4)

const entries = []
let offset = 6 + icoSizes.length * 16
icoImages.forEach((img, i) => {
  const e = Buffer.alloc(16)
  e.writeUInt8(icoSizes[i] === 256 ? 0 : icoSizes[i], 0) // width
  e.writeUInt8(icoSizes[i] === 256 ? 0 : icoSizes[i], 1) // height
  e.writeUInt8(0, 2) // palette
  e.writeUInt8(0, 3) // reserved
  e.writeUInt16LE(1, 4) // color planes
  e.writeUInt16LE(32, 6) // bits per pixel
  e.writeUInt32LE(img.length, 8) // bytes
  e.writeUInt32LE(offset, 12) // offset
  entries.push(e)
  offset += img.length
})

writeFileSync('public/favicon/favicon.ico', Buffer.concat([header, ...entries, ...icoImages]))
console.log(
  `public/favicon/favicon.ico         ${(Buffer.concat([header, ...entries, ...icoImages]).length / 1024).toFixed(1)} KB`
)
