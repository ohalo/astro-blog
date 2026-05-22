#!/usr/bin/env python3
"""
修复 Astro 博客文章日期
从 Jekyll 博客的 HTML 文件名提取日期，更新 Astro 文章的 frontmatter
"""

import os
import re
import sys
from pathlib import Path

# 配置
JEKYLL_BLOG = Path("/Users/halo/workspace/blog-repo")
ASTRO_BLOG = Path("/Users/halo/workspace/astro-blog")
ASTRO_CONTENT = ASTRO_BLOG / "src/content/blog"

def extract_date_from_filename(filename):
    """从 Jekyll 文件名提取日期 (YYYY-MM-DD-title.html)"""
    match = re.match(r'^(\d{4}-\d{2}-\d{2})-(.+)\.html$', filename)
    if match:
        date = match.group(1)
        title = match.group(2)
        return date, title
    return None, None

def find_jekyll_articles():
    """扫描 Jekyll 博客，提取所有文章的日期和标题"""
    articles = {}
    
    # 扫描 posts/ 目录（递归）
    for root, dirs, files in os.walk(JEKYLL_BLOG / "posts"):
        for file in files:
            if file.endswith('.html'):
                date, title = extract_date_from_filename(file)
                if date and title:
                    articles[title] = date
                    print(f"✓ 找到: {title} → {date}")
    
    return articles

def find_astro_articles():
    """扫描 Astro 博客，获取所有文章目录"""
    articles = {}
    
    if not ASTRO_CONTENT.exists():
        print(f"❌ 错误: Astro 内容目录不存在: {ASTRO_CONTENT}")
        sys.exit(1)
    
    for dir_path in ASTRO_CONTENT.iterdir():
        if dir_path.is_dir():
            index_file = dir_path / "index.md"
            if index_file.exists():
                articles[dir_path.name] = index_file
                print(f"✓ 找到 Astro 文章: {dir_path.name}")
    
    return articles

def update_frontmatter(file_path, new_date):
    """更新 Markdown 文件的 frontmatter 日期"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换 publishDate（支持多种格式）
        # 格式1: publishDate: 'YYYY-MM-DD'
        # 格式2: publishDate: "YYYY-MM-DD"
        # 格式3: publishDate: YYYY-MM-DD
        pattern = r"^(publishDate:\s*['\"]?)[\d-]+(['\"]?)$"
        
        # 使用 MULTILINE 模式，^ 匹配每一行的开头
        new_content = re.sub(
            pattern,
            f"\g<1>{new_date}\g<2>",
            content,
            flags=re.MULTILINE
        )
        
        if new_content == content:
            print(f"  ⚠️  未找到 publishDate，跳过: {file_path}")
            return False
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  ✓ 更新日期: {new_date}")
        return True
    
    except Exception as e:
        print(f"  ❌ 错误: {file_path}: {e}")
        return False

def main():
    print("=" * 60)
    print("修复 Astro 博客文章日期")
    print("=" * 60)
    print()
    
    # 1. 扫描 Jekyll 博客
    print("📖 扫描 Jekyll 博客...")
    jekyll_articles = find_jekyll_articles()
    print(f"✓ 找到 {len(jekyll_articles)} 篇 Jekyll 文章")
    print()
    
    # 2. 扫描 Astro 博客
    print("📘 扫描 Astro 博客...")
    astro_articles = find_astro_articles()
    print(f"✓ 找到 {len(astro_articles)} 篇 Astro 文章")
    print()
    
    # 3. 匹配并更新
    print("🔧 开始修复日期...")
    updated = 0
    not_found = 0
    not_matched = []
    
    for title, index_file in astro_articles.items():
        if title in jekyll_articles:
            new_date = jekyll_articles[title]
            if update_frontmatter(index_file, new_date):
                updated += 1
        else:
            not_matched.append(title)
            not_found += 1
    
    print()
    print("=" * 60)
    print("修复完成！")
    print("=" * 60)
    print(f"✓ 成功更新: {updated} 篇文章")
    print(f"⚠️  未匹配: {not_found} 篇文章")
    print()
    
    if not_matched:
        print("未匹配的文章（Jekyll 中未找到）:")
        for title in not_matched[:10]:
            print(f"  - {title}")
        if len(not_matched) > 10:
            print(f"  ... 还有 {len(not_matched) - 10} 篇")
        print()
    
    print("下一步:")
    print("  1. 检查修复结果: npm run dev")
    print("  2. 提交并推送: git add -A && git commit -m 'fix: correct all article publish dates' && git push origin main")

if __name__ == "__main__":
    main()
