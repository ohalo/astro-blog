/**
 * 生成黑金主题的社交分享卡片（OG Image 1200×630）
 * 用法：node gen_og_card.mjs
 */
import sharp from 'sharp'
import { readFileSync, writeFileSync } from 'node:fs'

// 复用金币定义
const coinSvg = readFileSync('tmp/coin.svg', 'utf8')
// 取出金币内部内容，缩放后嵌入卡片
const inner = coinSvg
  .replace(/^[\s\S]*?<defs>/, '<defs>')
  .replace(/<\/svg>\s*$/, '')
  .replace(/<defs>([\s\S]*?)<\/defs>/, '<defs>$1</defs>')

const CARD_W = 1200
const CARD_H = 630
const COIN = 250
const COIN_X = 92
const COIN_Y = (CARD_H - COIN) / 2

const cardSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="${CARD_W}" height="${CARD_H}" viewBox="0 0 ${CARD_W} ${CARD_H}">
  <defs>
    <radialGradient id="bg" cx="22%" cy="18%" r="95%">
      <stop offset="0%" stop-color="#221D14"/>
      <stop offset="45%" stop-color="#12100D"/>
      <stop offset="100%" stop-color="#080706"/>
    </radialGradient>
    <radialGradient id="glow" cx="24%" cy="50%" r="42%">
      <stop offset="0%" stop-color="#C9A227" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#C9A227" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="line" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#C9A227" stop-opacity="0"/>
      <stop offset="30%" stop-color="#C9A227" stop-opacity="0.85"/>
      <stop offset="100%" stop-color="#C9A227" stop-opacity="0"/>
    </linearGradient>
  </defs>

  <rect width="${CARD_W}" height="${CARD_H}" fill="url(#bg)"/>
  <rect width="${CARD_W}" height="${CARD_H}" fill="url(#glow)"/>

  <!-- 金币 -->
  <g transform="translate(${COIN_X}, ${COIN_Y}) scale(${COIN / 512})">
    ${inner}
  </g>

  <!-- 文字区 -->
  <text x="400" y="248" font-family="PingFang SC, Hiragino Sans GB, Heiti SC, sans-serif"
        font-size="62" font-weight="600" fill="#F2ECDF" letter-spacing="2">halo 的技术博客</text>

  <rect x="400" y="278" width="420" height="2" fill="url(#line)"/>

  <text x="400" y="340" font-family="PingFang SC, Hiragino Sans GB, Heiti SC, sans-serif"
        font-size="27" fill="#C9A227" letter-spacing="3">AI 工具 · 投资理财 · 阅读思考 · 生活方式</text>

  <text x="400" y="404" font-family="PingFang SC, Hiragino Sans GB, Heiti SC, sans-serif"
        font-size="23" fill="#8A7A55" letter-spacing="1.5">blog.halo26812.eu.org</text>

  <!-- 金色细边框 -->
  <rect x="24" y="24" width="${CARD_W - 48}" height="${CARD_H - 48}" rx="14"
        fill="none" stroke="#C9A227" stroke-opacity="0.32" stroke-width="1.5"/>
</svg>`

const buf = await sharp(Buffer.from(cardSvg), { density: 144 })
  .resize(CARD_W, CARD_H, { fit: 'cover' })
  .png({ compressionLevel: 9 })
  .toBuffer()

writeFileSync('public/images/social-card.png', buf)
console.log('public/images/social-card.png', (buf.length / 1024).toFixed(1), 'KB')
