import type { Element, Root } from 'hast'
import { visit } from 'unist-util-visit'

/**
 * 给 Markdown 正文图片加原生懒加载。
 *
 * 背景：本站正文图片全在 public/images/ 下，走不到 astro:assets 优化管线，
 * 文章页的所有大图（最大 2.8MB）会在首屏一次性拉取，拖累 LCP 与流量。
 *
 * 策略：
 *  - 每篇文章第一张图保持 eager + fetchpriority="high"，它通常在首屏内，
 *    作为 LCP 候选不该被延迟；其余图片 lazy。
 *  - 尊重作者显式写的 loading 属性（如手动 eager），不覆盖。
 *  - 不碰图片 src / class，避免影响 Medium Zoom 的 zoomable 钩子。
 */
export default function rehypeImageLazy() {
  return (tree: Root) => {
    let imageIndex = 0
    visit(tree, 'element', (node: Element) => {
      if (node.tagName !== 'img') return
      const idx = imageIndex++
      const props = (node.properties ??= {})
      if (props.loading != null) return // 作者已显式指定
      if (idx === 0) {
        props.loading = 'eager'
        props.fetchpriority = 'high'
      } else {
        props.loading = 'lazy'
      }
      props.decoding = 'async'
    })
  }
}
