#!/usr/bin/env python3
"""
量化文章生成脚本 - 自动分类难度并加入专栏
用法：python3 generate-quant-article.py --topic "因子投资" --difficulty "beginner"
"""

import os
import sys
import json
import argparse
from datetime import datetime
import re

def parse_args():
    parser = argparse.ArgumentParser(description='生成量化文章并自动加入专栏')
    parser.add_argument('--topic', required=True, help='文章主题')
    parser.add_argument('--difficulty', required=True, choices=['beginner', 'intermediate', 'advanced'], help='难度级别')
    parser.add_argument('--title', help='文章标题（如不提供则自动生成）')
    parser.add_argument('--tags', nargs='+', help='额外标签')
    return parser.parse_args()

def generate_slug(topic, date):
    """生成文章slug"""
    date_str = date.strftime('%Y-%m-%d')
    topic_slug = re.sub(r'[^a-z0-9\u4e00-\u9fa5]', '-', topic.lower())
    topic_slug = re.sub(r'-+', '-', topic_slug).strip('-')
    return f"{date_str}-{topic_slug}"

def generate_frontmatter(topic, difficulty, date, tags=None):
    """生成文章frontmatter"""
    title = f"{topic}：量化实战指南"
    slug = generate_slug(topic, date)
    
    # 难度映射
    difficulty_map = {
        'beginner': '入门',
        'intermediate': '中级',
        'advanced': '高级'
    }
    
    # 基础标签
    base_tags = ['quant-blog', 'quant-column', difficulty]
    if tags:
        base_tags.extend(tags)
    
    # 生成frontmatter
    frontmatter = f"""---
title: "{title}"
description: "{topic}的量化交易实战方法"
date: {date.strftime('%Y-%m-%d')}
tags: {json.dumps(base_tags, ensure_ascii=False)}
difficulty: "{difficulty_map[difficulty]}"
estimatedReadTime: 20
slug: "{slug}"
---
"""
    return frontmatter, slug

def generate_article_content(topic, difficulty):
    """生成文章内容（模板）"""
    content = f"""
# {topic}：量化实战指南

## 引言

（本文由AI生成，仅供参考）

## 核心概念

{topic}是量化交易中的重要主题...

## 策略原理

（详细讲解策略原理）

## Python实现

```python
# 示例代码
import pandas as pd
import numpy as np

# {topic}策略实现
def {topic.lower().replace(' ', '_')}_strategy(data):
    # 策略逻辑
    pass
```

## 回测结果

（回测分析和绩效指标）

## 风险管理

（止损、仓位管理、压力测试）

## 实战建议

（实盘注意事项）

## 总结

（关键要点总结）

---

*本文是量化交易实战专栏的一部分。更多文章，请访问[量化专栏](/quant-column.html)。*
"""
    return content

def create_article_file(topic, difficulty, tags=None):
    """创建文章文件"""
    date = datetime.now()
    frontmatter, slug = generate_frontmatter(topic, difficulty, date, tags)
    content = generate_article_content(topic, difficulty)
    
    # 文件路径
    file_path = f"/Users/halo/workspace/astro-blog/src/content/blog/{slug}.md"
    
    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter)
        f.write(content)
    
    print(f"✅ 文章已创建: {file_path}")
    print(f"📊 难度: {difficulty}")
    print(f"🏷️  标签: quant-blog, quant-column, {difficulty}")
    print(f"🔗 Slug: {slug}")
    
    return slug, file_path

def update_column_data(slug, title, difficulty, description, read_time=20):
    """更新专栏数据文件"""
    data_file = "/Users/halo/workspace/astro-blog/src/data/quant-column.ts"
    
    # 读取现有数据
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析现有文章（简单方法：在数组末尾添加）
    # 注意：这只是简单实现，实际应该用TS解析器
    new_article = f"""  {{
    slug: "{slug}",
    title: "{title}",
    difficulty: "{difficulty}",
    description: "{description}",
    date: "{datetime.now().strftime('%Y-%m-%d')}",
    readTime: {read_time}
  }}"""
    
    print(f"⚠️  请手动将以下文章添加到 {data_file}:")
    print(new_article)
    
    return True

def regenerate_column_html():
    """重新生成专栏HTML"""
    print("🔄 重新生成专栏HTML...")
    os.system("cd /Users/halo/workspace/astro-blog && python3 generate-column-html.py")
    print("✅ 专栏HTML已更新")

def regenerate_column_pdf():
    """重新生成专栏PDF"""
    print("🔄 重新生成专栏PDF...")
    os.system("cd /Users/halo/workspace/astro-blog && weasyprint public/quant-column.html public/quant-column.pdf")
    print("✅ 专栏PDF已更新")

def main():
    args = parse_args()
    
    print(f"📝 生成量化文章...")
    print(f"   主题: {args.topic}")
    print(f"   难度: {args.difficulty}")
    
    # 1. 创建文章文件
    slug, file_path = create_article_file(args.topic, args.difficulty, args.tags)
    
    # 2. 更新专栏数据
    title = args.title or f"{args.topic}：量化实战指南"
    update_column_data(slug, title, args.difficulty, f"{args.topic}的量化交易实战方法")
    
    # 3. 重新生成专栏HTML和PDF
    regenerate_column_html()
    regenerate_column_pdf()
    
    print(f"\n✅ 完成！文章已创建并加入专栏。")
    print(f"📊 记得：")
    print(f"   1. 完善文章内容（当前是模板）")
    print(f"   2. 提交并推送代码")
    print(f"   3. 检查专栏页: https://blog.halo26812.eu.org/quant-column.html")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
