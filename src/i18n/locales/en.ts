import type { Dictionary } from './zh'

/** 英文文案 —— 结构与 zh.ts 完全一致（由 Dictionary 类型约束） */
export const en: Dictionary = {
  nav: {
    blog: 'Blog',
    quant: 'Quant Column',
    links: 'Site Map',
    stock: 'Stocks',
    about: 'About',
    home: 'Home'
  },
  common: {
    search: 'Search',
    menu: 'Menu',
    darkMode: 'Dark theme',
    lightMode: 'Light theme',
    back: 'Back',
    backHome: 'Back to Home',
    morePosts: 'More posts',
    viewCount: 'views',
    commentCount: 'comments',
    sourceCode: 'Source code',
    location: 'Shanghai, China',
    readTime: 'read',
    lastUpdated: 'Updated on',
    publishedAt: 'Published on',
    tags: 'Tags',
    tableOfContents: 'Table of Contents',
    prevPage: '← Previous',
    nextPage: 'Next →',
    viewByYear: 'Browse by year →',
    allTags: 'All tags',
    noPosts: 'No posts yet.',
    noTags: 'No tags yet.',
    language: 'Language'
  },
  home: {
    aboutTitle: 'About Me',
    aboutRole: 'Entrepreneur / Developer',
    aboutDesc: 'A 37-year-old entrepreneur exploring how to work in the age of AI.',
    aboutMore: 'Learn more',
    quantTitle: 'Quant Column',
    quantDesc:
      'A structured path into quantitative trading: {count} articles split into Beginner / Intermediate / Advanced, covering stock basics, backtesting frameworks, factor research, statistical arbitrage, high-frequency microstructure and ML-based stock selection.',
    quantMore: 'Enter the Quant Column',
    linksTitle: 'Site Map',
    linksDesc:
      'A curated directory of {count} stock and quant-investing sites across {cats} categories (quotes, data research, academic papers, quant frameworks, factor libraries, financial media, regulators, brokers, communities, crypto, learning resources), with built-in search.',
    linksMore: 'Open the Site Map',
    latestPosts: 'Latest Posts',
    siteList: 'My Sites',
    browserExt: 'Browser Extensions',
    skills: 'Skills',
    moreCategories: '+{count} more categories',
    visitorTitle: 'Your Location & Weather',
    skillLanguages: 'Languages',
    skillFrontend: 'Frontend',
    skillBackend: 'Backend',
    techBlog: 'Tech Blog',
    techBlogDesc: 'AI tools, investing, reading notes and life',
    hafw: 'HAFW',
    hafwDesc: 'AI-assisted development platform from requirements to deployment',
    bookmarks: 'Bookmarks Topic',
    bookmarksDesc: 'Smart bookmark manager with WebDAV sync and broken-link detection',
    cybermap: 'Cybermap',
    cybermapDesc: 'Cybersecurity map and resource navigator'
  },
  about: {
    title: 'About Me',
    intro: 'A 37-year-old entrepreneur exploring how to work in the age of AI.',
    motto: 'Motto: Stay hungry, Stay foolish.',
    hobbies: 'Interests',
    hobbiesItems: [
      'Exploring and building AI tools',
      'Thinking about startups and business insight',
      'Hands-on engineering and productivity',
      'Observing and analysing society'
    ],
    tools: 'Toolkit',
    toolsIntro: 'Tools I use daily:',
    toolGroups: {
      ai: 'AI Tools',
      dev: 'Dev Tools',
      productivity: 'Productivity'
    },
    social: 'Social',
    socialIntro: 'Find me around the web:',
    aboutBlog: 'About This Blog',
    aboutBlogIntro: 'How this blog evolved:',
    supportIntro:
      'This blog runs on the resources and support of the following projects and providers:',
    supportItems: {
      domain: 'Domain:',
      framework: 'Framework & theme:',
      hosting: 'Hosting:',
      cicd: 'CI/CD:',
      comment: 'Comments:',
      analytics: 'Analytics:'
    },
    timeline: {
      created: 'Blog launched with Astro + Astro Theme Pure',
      migrated: 'Migrated to a new theme, improved search and page performance',
      ai: 'Started generating and publishing articles with the OpenClaw AI assistant'
    },
    toolDescs: {
      openclaw: 'AI assistant & automation',
      hafw: 'AI development workflow',
      claude: 'AI chat & coding',
      chatgpt: 'AI chat & creation',
      vscode: 'Code editor',
      github: 'Code hosting & collaboration',
      astro: 'Static site generator',
      notion: 'Notes & knowledge management',
      typora: 'Markdown editor'
    }
  },
  quant: {
    title: 'Quant Column',
    subtitle: 'A structured path into quantitative investing, from beginner to advanced',
    intro:
      'A structured path into quantitative investing: {count} articles split into Beginner / Intermediate / Advanced, covering stock basics, backtesting frameworks, factor research, statistical arbitrage, high-frequency microstructure and ML-based stock selection.',
    stats: {
      total: 'articles',
      beginner: 'Beginner',
      intermediate: 'Intermediate',
      advanced: 'Advanced'
    },
    categories: {
      beginner: '🌱 Beginner: Foundations',
      beginnerDesc: 'Concepts and tools for newcomers to quantitative trading',
      intermediate: '📈 Intermediate: Strategy & Backtesting',
      intermediateDesc: 'Factor mining, strategy design, risk control',
      advanced: '🚀 Advanced: Live Trading & Advanced Strategies',
      advancedDesc: 'High-frequency trading, machine learning, portfolio optimisation',
      uncategorized: '📚 Uncategorized',
      uncategorizedDesc: 'Articles awaiting a difficulty tag (add `difficulty` in the frontmatter)'
    }
  },
  links: {
    title: 'Stock Site Directory',
    intro:
      '{total} curated sites for stock and quant investing ({unique} unique domains) across {cats} categories. Every link was checked for availability, and tags mark whether it is free, needs registration or is paid.',
    stats: {
      total: 'Sites listed',
      categories: 'Categories',
      free: 'Free / Open source',
      gated: 'Registration / Paid'
    },
    searchPlaceholder: 'Search by name, feature or domain…',
    matchCount: '{count} matching sites',
    empty: 'No matching site — try a different keyword.',
    unit: ''
  },
  blog: {
    title: 'Blog',
    description: 'Article archive',
    listTitle: 'Blog',
    archive: 'Archive',
    archiveDesc: 'Browse every article by date',
    archiveEmpty: 'No posts yet.',
    archiveCount: '{count} articles',
    tagsTitle: 'Tags',
    tagsDesc: 'Every topic tag used across the blog',
    tagPage: 'Tag: {tag}',
    tagDesc: 'All articles tagged “{tag}”',
    pageInfo: 'Page {page} · Showing {shown} / {total} articles',
    searchTitle: 'Search',
    searchDesc: 'Search blog articles',
    searchIntro: 'Type a keyword to search the blog.',
    searchDisabled: 'Search is not enabled.'
  },
  projects: {
    title: 'Projects',
    headings: {
      programs: 'Open Source',
      learnings: 'Learning',
      others: 'Others'
    },
    activity: 'GitHub activity:',
    intro: 'If you are interested in my projects, or would like to collaborate, get in touch.'
  },
  terms: {
    title: 'Site Policies',
    backHome: 'Back to Home'
  },
  linksPage: {
    title: 'Links',
    commonLinks: 'Common Links',
    specialLinks: 'Special Links',
    applyLinks: 'Apply Links',
    randomNote: "They're sorted randomly.",
    badStatus: 'Links with Bad Status',
    history: 'Link History Book',
    applyIntro: 'Site information (click to copy):',
    applyOutro: 'Apply for a link via the comments below or by submitting a PR. Requirements:',
    applyRules: [
      "You've recommended this site on your own site",
      'Your site is alive and uses a proper domain',
      'At least 5 original posts, with at least one published in the past year',
      'Content does not violate applicable laws'
    ]
  },
  notFound: {
    title: 'Page not found',
    desc: 'Oops, something went wrong.',
    body: "Sorry, we couldn't find the page you're looking for.",
    backHome: 'Back to Home'
  },
  footer: {
    sitePolicy: 'Site Policies',
    themeBy: 'theme',
    drivenBy: 'powered by'
  },
  search: {
    placeholder: 'Search articles…',
    noResult: 'Nothing found',
    resultCount: '{count} results found'
  },
  post: {
    originalNotice: 'This article is originally written in Chinese; no English version yet.',
    relatedTitle: 'Related posts',
    prevPost: 'Previous',
    nextPost: 'Next'
  }
}
