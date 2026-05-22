#!/usr/bin/env python3
"""
修复剩余 11 篇文章的日期
设置近似日期为 2026-05-22
"""

import os
import re
from pathlib import Path

ASTRO_CONTENT = Path("/Users/halo/workspace/astro-blog/src/content/blog")

# 需要修复的文章列表
ARTICLES = [
    "deep-work-focus",
    "senior-engineer-promotion",
    "interview-beyond-coding",
    "coffee-nap-pomodoro",
    "mcp-ai-tool-connector",
    "digital-connection-emptiness",
    "thematic-reading-classics",
    "obsidian-knowledge-system",
    "reading-fast-and-deep",
    "claude-code-vs-cursor",
    "ai-art-human-aesthetic",
]

def main():
    print("=" * 60)
    print("修复剩余 11 篇文章的日期")
    print("=" * 60)
    print()
    
    updated = 0
    not_found = 0
    
    for article in ARTICLES:
        index_file = ASTRO_CONTENT / article / "index.md"
        
        if not index_file.exists():
            print(f"⚠️  未找到: {article}/index.md")
            not_found += 1
            continue
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换 publishDate: '2024-01-01' → publishDate: '2026-05-22'
            pattern = r"publishDate:\s*['\"]2024-01-01['\"]"
            replacement = "publishDate: '2026-05-22'"
            
            new_content = re.sub(pattern, replacement, content)
            
            if new_content == content:
                print(f"⚠️  未找到 2024-01-01: {article}")
                not_found += 1
                continue
            
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✓ 已更新: {article} → 2026-05-22")
            updated += 1
            
        except Exception as e:
            print(f"❌ 错误: {article}: {e}")
            not_found += 1
    
    print()
    print("=" * 60)
    print("修复完成！")
    print("=" * 60)
    print(f"✓ 成功更新: {updated} 篇文章")
    print(f"⚠️  未找到: {not_found} 篇文章")
    print()
    print("下一步:")
    print("  1. 检查修复结果: npm run dev")
    print("  2. 提交并推送: git add -A && git commit -m 'fix: correct remaining 11 article dates' && git push origin main")

if __name__ == "__main__":
    main()
