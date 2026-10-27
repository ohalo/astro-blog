#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os

os.chdir("/Users/halo123456/workspace/astro-blog")
for slug in ["rolling-cointegration-bre", "low-volatility-anomaly-cn"]:
    p = f"src/content/blog/{slug}/index.md"
    txt = open(p, encoding="utf-8").read()
    lines = txt.split("\n")
    ok_start = lines[0] == "---"
    fm_end = None
    for i in range(1, len(lines)):
        if lines[i] == "---":
            fm_end = i
            break
    body = "\n".join(lines[fm_end + 1:])
    body = re.sub(r"```.*?```", "", body, flags=re.S)
    body = re.sub(r"!\[.*?\]\(.*?\)", "", body)
    cn = len(re.findall(r"[\u4e00-\u9fff]", body))
    imgs = sorted(set(re.findall(r"/images/([^)]+?)/", txt)))
    missing = [i for i in imgs if not os.path.exists(f"public/images/{slug}/{i}")]
    # frontmatter 关键字段
    fm = "\n".join(lines[1:fm_end])
    has_title = "title:" in fm
    has_date = "publishDate:" in fm
    has_diff = "difficulty:" in fm
    desc_ok = True
    m = re.search(r'description: "(.*)"', fm)
    if m:
        desc_ok = len(m.group(1)) <= 600
    print(f"{slug}: 起始---={ok_start} 闭合---={fm_end is not None} 中文字数={cn} "
          f"引用图={len(imgs)} 缺失={missing} title={has_title} date={has_date} "
          f"difficulty={has_diff} desc<=600={desc_ok}")
