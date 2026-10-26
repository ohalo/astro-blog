/**
 * 中文文案（默认语言）
 * 结构即类型定义，en.ts 以 typeof zh 约束，保证两种语言键完全对齐
 */
export const zh = {
  nav: {
    blog: '博客',
    quant: '量化专栏',
    links: '网址导航',
    stock: '股票查询',
    about: '关于',
    home: '首页'
  },
  common: {
    search: '搜索',
    menu: '菜单',
    darkMode: '深色主题',
    lightMode: '浅色主题',
    back: '返回',
    backHome: '返回首页',
    morePosts: '更多文章',
    viewCount: '次浏览',
    commentCount: '条评论',
    sourceCode: '源代码',
    location: '中国上海',
    readTime: '阅读',
    lastUpdated: '更新于',
    publishedAt: '发布于',
    tags: '标签',
    tableOfContents: '目录',
    prevPage: '← 上一页',
    nextPage: '下一页 →',
    viewByYear: '按年份查看 →',
    allTags: '所有标签',
    noPosts: '暂无文章。',
    noTags: '暂无标签。',
    language: '语言'
  },
  home: {
    aboutTitle: '关于我',
    aboutRole: '创业者 / 开发者',
    aboutDesc: '一个 37 岁的创业者，正在探索 AI 时代的工作方式。',
    aboutMore: '了解更多',
    quantTitle: '量化专栏',
    quantDesc:
      '系统化的量化交易学习路径：{count} 篇文章按「入门 / 进阶 / 高阶」三级分类，覆盖股票基础、回测框架、因子研究、统计套利、高频微观结构与机器学习选股。',
    quantMore: '进入量化专栏',
    linksTitle: '网址导航',
    linksDesc:
      '股票与量化投资相关的常用站点导航：{count} 个链接按 {cats} 个主题分类整理（行情交易、数据研究、学术论文、量化框架、因子策略、财经媒体、监管公告、券商开户、社区交流、加密货币、学习资源），支持站内搜索。',
    linksMore: '打开网址导航',
    latestPosts: '最新文章',
    siteList: '网站列表',
    browserExt: '浏览器扩展',
    skills: '技能',
    moreCategories: '+{count} 个分类',
    visitorTitle: '当前访客位置与天气',
    skillLanguages: '编程语言',
    skillFrontend: '前端技术',
    skillBackend: '后端技术',
    techBlog: '技术博客',
    techBlogDesc: '分享 AI 工具、投资理财、阅读思考、生活方式',
    hafw: 'HAFW',
    hafwDesc: 'AI 智能开发协作平台，覆盖需求分析到部署运维',
    bookmarks: 'Bookmarks Topic',
    bookmarksDesc: '智能书签管理器，支持 WebDAV 同步、断链检测',
    cybermap: 'Cybermap',
    cybermapDesc: '网络安全地图与资源导航'
  },
  about: {
    title: '关于我',
    intro: '一个 37 岁的创业者，正在探索 AI 时代的工作方式。',
    motto: 'Motto: Stay hungry, Stay foolish.',
    hobbies: '兴趣爱好',
    hobbiesItems: [
      'AI 工具探索与应用',
      '创业思考与商业洞察',
      '技术实践与效率提升',
      '社会现象观察与分析'
    ],
    tools: '工具栈',
    toolsIntro: '我日常使用的工具：',
    toolGroups: {
      ai: 'AI 工具',
      dev: '开发工具',
      productivity: '生产力'
    },
    social: '社交网络',
    socialIntro: '欢迎在各种平台找到我：',
    aboutBlog: '关于博客',
    aboutBlogIntro: '这个博客的发展历程：',
    supportIntro: '博客的正常运行依赖于以下优秀项目/服务提供者的资源和技术支持：',
    supportItems: {
      domain: '域名：',
      framework: '框架与主题：',
      hosting: '网站托管：',
      cicd: 'CI/CD：',
      comment: '评论系统：',
      analytics: '访问统计：'
    },
    timeline: {
      created: '博客建立，使用 Astro + Astro Theme Pure 主题',
      migrated: '迁移到新主题，优化搜索功能和页面性能',
      ai: '使用 OpenClaw AI 助手自动生成和发布文章'
    },
    toolDescs: {
      openclaw: 'AI 助手与自动化',
      hafw: 'AI 开发流程管理',
      claude: 'AI 对话与编程',
      chatgpt: 'AI 对话与创作',
      vscode: '代码编辑器',
      github: '代码托管与协作',
      astro: '静态网站生成器',
      notion: '笔记与知识管理',
      typora: 'Markdown 编辑器'
    }
  },
  quant: {
    title: '量化专栏',
    subtitle: '系统化的量化投资学习路径：从入门到高阶',
    intro:
      '系统化的量化投资学习路径：{count} 篇文章按「入门 / 进阶 / 高阶」三级分类，覆盖股票基础、回测框架、因子研究、统计套利、高频微观结构与机器学习选股。',
    stats: {
      total: '篇文章',
      beginner: '入门',
      intermediate: '进阶',
      advanced: '高阶'
    },
    categories: {
      beginner: '入门：量化交易基础',
      beginnerDesc: '适合初学者的量化交易概念和工具',
      intermediate: '进阶：策略开发与回测',
      intermediateDesc: '因子挖掘、策略设计、风险控制',
      advanced: '高阶：实盘交易与高级策略',
      advancedDesc: '高频交易、机器学习、组合优化',
      uncategorized: '未分类',
      uncategorizedDesc: '等待分类的量化文章（请在 frontmatter 中添加 difficulty 字段）'
    }
  },
  links: {
    title: '股票网址导航',
    intro:
      '收录 {total} 个股票与量化投资相关的常用站点（{unique} 个独立域名），覆盖 {cats} 个分类。所有链接均经过存活状态实测，标签标注了免费 / 需注册 / 付费等访问条件。',
    stats: {
      total: '收录站点',
      categories: '覆盖分类',
      free: '免费 / 开源',
      gated: '需注册 / 付费'
    },
    searchPlaceholder: '搜索站点名称、功能或域名…',
    matchCount: '匹配 {count} 个站点',
    empty: '没有匹配的站点，换个关键词试试。',
    unit: '个'
  },
  blog: {
    title: '博客',
    description: '博客文章归档',
    listTitle: '博客',
    archive: '归档',
    archiveDesc: '按日期浏览所有文章',
    archiveEmpty: '暂无文章。',
    archiveCount: '{count} 篇文章',
    tagsTitle: '标签',
    tagsDesc: '博客涉及的所有话题标签',
    tagPage: '标签：{tag}',
    tagDesc: '查看标签「{tag}」下的所有文章',
    pageInfo: '第 {page} 页 · 显示 {shown} / {total} 篇文章',
    searchTitle: '搜索',
    searchDesc: '搜索博客文章',
    searchIntro: '输入关键词搜索博客文章。',
    searchDisabled: '搜索功能未启用。'
  },
  projects: {
    title: '项目',
    headings: {
      programs: '开源项目',
      learnings: '学习项目',
      others: '其他'
    },
    activity: 'GitHub 活动图：',
    intro: '如果你对我的项目感兴趣，或者想要合作，欢迎联系我讨论。'
  },
  terms: {
    title: '站点政策',
    backHome: '返回首页'
  },
  linksPage: {
    title: 'Links',
    commonLinks: 'Common Links',
    specialLinks: 'Special Links',
    applyLinks: 'Apply Links',
    randomNote: "They're sorted randomly.",
    badStatus: 'Links with Bad Status',
    history: 'Link History Book',
    applyIntro: '本站信息（点击复制）The site information (click to copy)：',
    applyOutro: '可通过下方留言或 提交 PR 申请友链。申请说明：',
    applyRules: [
      "推荐了本站 You've recommended our link on your website",
      '站点活着且有正式域名 Make sure your site is alive with a formal domain',
      '至少五篇原创文章且最近一年更新过 At least 5 original articles and exists articles released in recent year',
      '内容不违反国家法律 Content does not violate laws'
    ]
  },
  notFound: {
    title: '页面未找到',
    desc: '糟糕，出了点问题。',
    body: '抱歉，找不到您要的页面。',
    backHome: '返回首页'
  },
  footer: {
    sitePolicy: '站点政策',
    themeBy: '主题',
    drivenBy: '驱动'
  },
  search: {
    placeholder: '搜索文章…',
    noResult: '没有找到相关内容',
    resultCount: '找到 {count} 条结果'
  },
  post: {
    originalNotice: '本文为中文原创，暂无英文版本。',
    relatedTitle: '相关文章',
    prevPost: '上一篇',
    nextPost: '下一篇'
  }
}

export type Dictionary = typeof zh
