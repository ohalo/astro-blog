/**
 * 语言切换可用性判定。
 *
 * 背景：Header 以前无条件把中文路径拼上 /en 前缀，但本站只有少数页面有真实英文对照：
 *   - 文章：只有 frontmatter.language === 'English' 才产出 /en/blog/{id} 正文页，
 *     其余是 meta refresh 弹回中文的跳转壳（静态输出拿不到真 301）；
 *   - /en/blog（列表与分页）、/en/tools/*、/en/terms/* 细则、/en/docs/* 根本不存在。
 * 结果「English」按钮在 1800+ 页面上要么原地回弹、要么 404。
 *
 * 这里集中维护「确实存在英文版」的路由，Header 只在可用时渲染可点链接。
 * 对照表从内容集合自动生成，将来真有英文文章会自动生效，不用改这里。
 */
import { getCollection } from 'astro:content'

import { getLocale } from './index'

/** 与 src/pages/en/** 一一对应的静态页（UI 有完整英文文案）。 */
const TRANSLATED_STATIC = new Set([
  '/',
  // /404 不在此列：英文 404 页已移除，切过去只会得到真正的 404（见 src/pages/en/404.astro 的删除记录）
  '/about',
  '/archives',
  '/dash',
  '/learn',
  '/links',
  '/projects',
  '/quant-column',
  // /search 不在此列：Pagefind 只索引中文页，英文搜索框只会返回中文结果（见 en/search/index.astro）
  '/stock',
  '/stock-links',
  '/tags',
  '/terms'
])

let idsCache: Promise<Set<string>> | undefined

/** 真正有英文正文的文章 id（当前为空；新增英文文章后自动生效）。 */
export function getTranslatedPostIds(): Promise<Set<string>> {
  idsCache ??= getCollection('blog').then((posts) => {
    const ids = posts
      .filter((post) => post.data.language === 'English' && !post.data.draft)
      .map((post) => post.id)
    return new Set(ids)
  })
  return idsCache
}

/**
 * 当前路径是否允许切换到另一种语言。
 * en → zh 恒可用：当前英文页既然存在，其中文原页必然存在。
 * zh → en 严格按上面的静态白名单 / 文章 language 字段判定。
 */
export function canSwitchLocale(pathname: string, translatedPostIds: Set<string>): boolean {
  const path = pathname.replace(/\/+$/, '') || '/'
  if (getLocale(path) === 'en') return true
  if (TRANSLATED_STATIC.has(path)) return true
  // /en/tags/<tag> 由全部文章的 tag 生成，列表页自身有完整英文 UI
  if (path.startsWith('/tags/')) return true
  if (path.startsWith('/blog/')) return translatedPostIds.has(path.slice('/blog/'.length))
  return false
}
