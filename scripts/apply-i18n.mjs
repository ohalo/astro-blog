import { readFileSync, writeFileSync } from 'fs'
import { join } from 'path'

const root = process.cwd()

const replacements = [
  // Header.astro
  {
    file: 'node_modules/astro-pure/components/basic/Header.astro',
    changes: [
      ["title='Search'", "title='搜索'"],
      ["<span class='sr-only'>Search</span>", "<span class='sr-only'>搜索</span>"],
      ["<span class='sr-only'>Dark Theme</span>", "<span class='sr-only'>深色主题</span>"],
      ["<span class='sr-only'>Menu</span>", "<span class='sr-only'>菜单</span>"],
      ['showToast({ message: `Set theme to ${newTheme}` })', 'showToast({ message: `已切换主题到 ${newTheme}` })'],
    ]
  },
  // Footer.astro
  {
    file: 'node_modules/astro-pure/components/basic/Footer.astro',
    changes: [
      ['&\n              <a', '主题\n              <a'],
      ['theme powered', '驱动'],
    ]
  },
  // BackToTop.astro
  {
    file: 'node_modules/astro-pure/components/pages/BackToTop.astro',
    changes: [
      ["aria-label='Back to Top'", "aria-label='回到顶部'"],
    ]
  },
  // Copyright.astro - simple replacements
  {
    file: 'node_modules/astro-pure/components/pages/Copyright.astro',
    changes: [
      ['<span>Copyright</span>', '<span>版权声明</span>'],
    ]
  },
  // Paginator.astro
  {
    file: 'node_modules/astro-pure/components/pages/Paginator.astro',
    changes: [
      ["'Previous'", "'上一页'"],
      ["'Next'", "'下一页'"],
    ]
  },
  // PostPreview.astro
  {
    file: 'node_modules/astro-pure/components/pages/PostPreview.astro',
    changes: [
      ["title='Tags'", "title='标签'"],
      ["aria-label='Tags'", "aria-label='标签'"],
      ["aria-labelledby='Tags'", "aria-labelledby='标签'"],
    ]
  },
  // TOC.astro
  {
    file: 'node_modules/astro-pure/components/pages/TOC.astro',
    changes: [
      ['<h2 class=\'font-medium\'>TABLE OF CONTENTS</h2>', '<h2 class=\'font-medium\'>目录</h2>'],
    ]
  },
]

let total = 0
for (const { file, changes } of replacements) {
  const path = join(root, file)
  let content = readFileSync(path, 'utf-8')
  let count = 0
  for (const [from, to] of changes) {
    if (content.includes(from)) {
      content = content.replaceAll(from, to)
      count++
    }
  }
  
  // Special handling for Copyright.astro - replace "Buy me a cup of coffee" block
  if (file.includes('Copyright.astro')) {
    const oldBlock = /  <div class='mx-6 rounded-b-xl border border-t-0 px-3 pb-1\.5 pt-1 sm:mx-8 sm:px-4'>\n    <a\n      href='\/projects#sponsorship'\n      class='flex items-center justify-between text-muted-foreground no-underline'\n      target='_blank'\n    >\n      <span>Buy me a cup of coffee ☕\.<\/span>\n      <Icon class='box-content size-5 p-1' name='receive-money' \/>\n    <\/a>\n  <\/div>/s
    const newBlock = `  <div class='mx-6 rounded-b-xl border border-t-0 px-3 pb-1.5 pt-1 text-center text-sm text-muted-foreground sm:mx-8 sm:px-4'>\n    <span>学而时习之，不亦说乎</span>\n  </div>`
    
    if (oldBlock.test(content)) {
      content = content.replace(oldBlock, newBlock)
      count++
      console.log(`  ✓ Replaced "Buy me a cup of coffee" block`)
    }
  }
  
  if (count > 0) {
    writeFileSync(path, content)
    console.log(`✓ ${file} (${count} changes)`)
    total += count
  }
}

console.log(`\nApplied ${total} i18n replacements!`)
