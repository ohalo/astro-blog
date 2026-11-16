/**
 * 黑金内页统一页头元数据。
 * 按 pathname 前缀匹配，覆盖静态页与分页 / 标签 / 条款等动态路由；中英文各一套文案。
 * 首页 / 量化专栏 / 文章详情走 frame='blackgold'，不使用这里的元数据。
 */
export interface HgPageMeta {
  eyebrow: string
  title: string
  desc: string
  crumb: string
  no: string
}

type Rule = [prefix: string, zh: HgPageMeta, en: HgPageMeta]

const RULES: Rule[] = [
  [
    '/blog',
    {
      eyebrow: 'Article Index',
      title: '文章索引',
      desc: '按发布时间倒序浏览全部研究文章，从入门笔记到可复现的数值实验。',
      crumb: '博客',
      no: '01'
    },
    {
      eyebrow: 'Article Index',
      title: 'Article Index',
      desc: 'Every research note, newest first — from introductory notes to reproducible numerical experiments.',
      crumb: 'Blog',
      no: '01'
    }
  ],
  [
    '/learn',
    {
      eyebrow: 'Learning Paths',
      title: '量化学习路线',
      desc: '把散落的文章按主题串成由浅入深的主线，进度保存在本机浏览器。',
      crumb: '学习路线',
      no: '02'
    },
    {
      eyebrow: 'Learning Paths',
      title: 'Learning Paths',
      desc: 'Thematic reading routes from fundamentals to advanced research, with local progress tracking.',
      crumb: 'Learning Paths',
      no: '02'
    }
  ],
  [
    '/archives',
    {
      eyebrow: 'Chronology',
      title: '文章归档',
      desc: '按年份回看每一次更新，全部文章按发布时间排列。',
      crumb: '归档',
      no: '03'
    },
    {
      eyebrow: 'Chronology',
      title: 'Archives',
      desc: 'A year-by-year record of every published article.',
      crumb: 'Archives',
      no: '03'
    }
  ],
  [
    '/tags',
    {
      eyebrow: 'Topics',
      title: '主题标签',
      desc: '按话题检索研究文章，从量化交易、市场微观结构到工具与工程实践。',
      crumb: '标签',
      no: '04'
    },
    {
      eyebrow: 'Topics',
      title: 'Topics',
      desc: 'Browse research articles by topic and tag.',
      crumb: 'Tags',
      no: '04'
    }
  ],
  [
    '/stock-links',
    {
      eyebrow: 'Directory',
      title: '网址导航',
      desc: '量化研究、数据源与工具站点分类索引，标注开放与受限访问条件。',
      crumb: '网址导航',
      no: '05'
    },
    {
      eyebrow: 'Directory',
      title: 'Directory',
      desc: 'A curated index of quant research, data and tooling sites.',
      crumb: 'Directory',
      no: '05'
    }
  ],
  [
    '/links',
    {
      eyebrow: 'Friends',
      title: '友情链接',
      desc: '朋友们的站点与友链申请说明。',
      crumb: '友链',
      no: '06'
    },
    {
      eyebrow: 'Friends',
      title: 'Links',
      desc: 'Sites from friends, and how to apply for a link exchange.',
      crumb: 'Links',
      no: '06'
    }
  ],
  [
    '/projects',
    {
      eyebrow: 'Projects',
      title: '项目',
      desc: '开源项目、学习项目与一些还在进行中的实验。',
      crumb: '项目',
      no: '07'
    },
    {
      eyebrow: 'Projects',
      title: 'Projects',
      desc: 'Open-source work, learning projects and experiments in progress.',
      crumb: 'Projects',
      no: '07'
    }
  ],
  [
    '/search',
    {
      eyebrow: 'Search',
      title: '全站检索',
      desc: '输入关键词，检索标题与正文中的研究笔记。',
      crumb: '搜索',
      no: '08'
    },
    {
      eyebrow: 'Search',
      title: 'Search',
      desc: 'Search article titles and full text by keyword.',
      crumb: 'Search',
      no: '08'
    }
  ],
  [
    '/about',
    {
      eyebrow: 'About',
      title: '关于',
      desc: '关于我、常用工具栈，以及这个站点是怎么搭起来的。',
      crumb: '关于',
      no: '09'
    },
    {
      eyebrow: 'About',
      title: 'About',
      desc: 'About me, the tool stack, and how this site is built.',
      crumb: 'About',
      no: '09'
    }
  ],
  [
    '/dash',
    {
      eyebrow: 'Terminal',
      title: '量化工作台',
      desc: '实时行情、自选异动、多周期 K 线与量化指标的复盘终端。',
      crumb: '量化工作台',
      no: '10'
    },
    {
      eyebrow: 'Terminal',
      title: 'Quant Terminal',
      desc: 'Live quotes, watchlist, multi-timeframe charts and computed metrics.',
      crumb: 'Terminal',
      no: '10'
    }
  ],
  [
    '/stock',
    {
      eyebrow: 'Quotes',
      title: '股票查询',
      desc: '输入股票代码，实时查询行情、K 线、资金流向与基本资料。',
      crumb: '股票查询',
      no: '11'
    },
    {
      eyebrow: 'Quotes',
      title: 'Stock Quotes',
      desc: 'Look up live quotes, charts and company fundamentals by ticker.',
      crumb: 'Quotes',
      no: '11'
    }
  ],
  [
    '/docs',
    {
      eyebrow: 'Documentation',
      title: '文档',
      desc: '主题功能说明与站点文档索引。',
      crumb: '文档',
      no: '12'
    },
    {
      eyebrow: 'Documentation',
      title: 'Docs',
      desc: 'Theme documentation and site reference.',
      crumb: 'Docs',
      no: '12'
    }
  ],
  [
    '/terms',
    {
      eyebrow: 'Policy',
      title: '站点政策',
      desc: '版权、免责声明、隐私政策与使用条款。',
      crumb: '站点政策',
      no: '13'
    },
    {
      eyebrow: 'Policy',
      title: 'Policies',
      desc: 'Copyright, disclaimer, privacy policy and terms of use.',
      crumb: 'Policies',
      no: '13'
    }
  ],
  [
    '/404',
    {
      eyebrow: 'Not Found',
      title: '页面未找到',
      desc: '这个地址没有对应的内容，回到首页或去看看文章索引。',
      crumb: '404',
      no: '14'
    },
    {
      eyebrow: 'Not Found',
      title: 'Page Not Found',
      desc: 'Nothing lives at this address. Head home or browse the article index.',
      crumb: '404',
      no: '14'
    }
  ]
]

const FALLBACK_ZH: HgPageMeta = {
  eyebrow: 'Research Journal',
  title: 'halo 的技术博客',
  desc: '把市场的噪声，写成可复现的结论。',
  crumb: '站内',
  no: '00'
}
const FALLBACK_EN: HgPageMeta = {
  eyebrow: 'Research Journal',
  title: 'halo. Research Journal',
  desc: 'Turning market noise into reproducible conclusions.',
  crumb: 'Site',
  no: '00'
}

export function getHgPageMeta(pathname: string): HgPageMeta & { locale: 'zh' | 'en' } {
  const isEn = /^\/en(?:\/|$)/.test(pathname)
  const path = (pathname.replace(/^\/en/, '') || '/').replace(/\/+$/, '') || '/'
  const rule = RULES.find(([prefix]) => path === prefix || path.startsWith(`${prefix}/`))
  const meta = rule ? (isEn ? rule[2] : rule[1]) : isEn ? FALLBACK_EN : FALLBACK_ZH
  return { ...meta, locale: isEn ? 'en' : 'zh' }
}
