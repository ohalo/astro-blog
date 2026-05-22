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
  // Copyright.astro
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
  if (count > 0) {
    writeFileSync(path, content)
    console.log(`✓ ${file} (${count} changes)`)
    total += count
  }
}

console.log(`\nApplied ${total} i18n replacements!`)
