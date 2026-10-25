import { en } from './locales/en'
import { zh } from './locales/zh'

export type Locale = 'zh' | 'en'

export const DEFAULT_LOCALE: Locale = 'zh'
export const LOCALES: Locale[] = ['zh', 'en']
export const LOCALE_LABEL: Record<Locale, string> = { zh: '中文', en: 'EN' }
export const LOCALE_FULL: Record<Locale, string> = { zh: '简体中文', en: 'English' }

const DICTIONARIES = { zh, en }

/** 取指定语言的文案表 */
export function dict(locale: Locale) {
  return DICTIONARIES[locale]
}

/**
 * 依据当前请求路径判定语言
 * /en/... → en；其余 → zh（默认语言不加前缀，保护已有中文 URL 与外链）
 */
export function getLocale(pathname: string): Locale {
  return pathname === '/en' || pathname.startsWith('/en/') ? 'en' : DEFAULT_LOCALE
}

/** 给指定语言拼接路径 */
export function localePath(locale: Locale, path = '/'): string {
  const clean = path === '/' ? '' : path
  if (locale === DEFAULT_LOCALE) return clean || '/'
  return `/en${clean}`
}

/**
 * 生成语言切换的目标路径
 * 中文 → 英文：在路径前加 /en；英文 → 中文：去掉 /en 前缀
 */
export function switchLocalePath(pathname: string): { locale: Locale; path: string } {
  if (getLocale(pathname) === 'en') {
    const stripped = pathname.replace(/^\/en/, '') || '/'
    return { locale: 'zh', path: stripped }
  }
  return { locale: 'en', path: localePath('en', pathname) }
}

/**
 * 模板变量替换：把 {name} 占位符换成实际值
 * 例：t('home.quantDesc', locale, { count: 100 })
 */
export function fill(template: string, vars?: Record<string, string | number>): string {
  if (!vars) return template
  return template.replace(/\{(\w+)\}/g, (match, key) =>
    key in vars ? String(vars[key]) : match
  )
}

/**
 * 按点号路径取值并填充变量
 * 例：t('common.search', locale)
 */
export function t(path: string, locale: Locale, vars?: Record<string, string | number>): string {
  const value = path
    .split('.')
    .reduce<unknown>(
      (acc, key) => (acc && typeof acc === 'object' ? (acc as Record<string, unknown>)[key] : undefined),
      DICTIONARIES[locale]
    )
  if (typeof value !== 'string') {
    // 数组类型（如 hobbiesItems）直接返回第一项做兜底提示
    if (Array.isArray(value)) return value.join('、')
    return path
  }
  return fill(value, vars)
}
