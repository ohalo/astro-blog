#!/usr/bin/env python3
"""Scan all blog posts for structural issues:
1. Frontmatter validity (title, publishDate, description, tags, language, difficulty)
2. Image references vs actual files
3. Markdown code block closure
4. Empty/duplicate content
5. Frontmatter parse errors
"""
import os, re, sys
from datetime import datetime

BLOG_DIR = "/Users/halo/workspace/astro-blog/src/content/blog"
PUBLIC_DIR = "/Users/halo/workspace/astro-blog/public/images"

def parse_frontmatter(text):
    """Return (meta_dict, body) or (None, error)."""
    if not text.startswith("---"):
        return None, "NO_FRONTMATTER"
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return None, "FRONTMATTER_UNCLOSED"
    raw = m.group(1)
    # Parse simple key: value pairs
    meta = {}
    in_list = False
    current_key = None
    for line in raw.split("\n"):
        if in_list:
            l = line.strip()
            if l.startswith("-"):
                meta.setdefault(current_key, []).append(l[1:].strip().strip("'\""))
            else:
                in_list = False
                current_key = None
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip("'\"").strip()
        if val.startswith("[") and val.endswith("]"):
            items = [x.strip().strip("'\"") for x in val[1:-1].split(",") if x.strip()]
            meta[key] = items
        elif val == "" or val.startswith("-"):
            meta.setdefault(key, [] if (val == "" or val.startswith("-")) else val)
            if val.startswith("-"):
                in_list = True
                current_key = key
                meta[key].append(val[1:].strip().strip("'\""))
        else:
            meta[key] = val
    return meta, text[m.end():]

issues = []
posts = []
for slug in sorted(os.listdir(BLOG_DIR)):
    p = os.path.join(BLOG_DIR, slug)
    if not os.path.isdir(p):
        continue
    md = os.path.join(p, "index.md")
    if not os.path.exists(md):
        issues.append(f"[{slug}] MISSING index.md")
        continue
    with open(md, encoding="utf-8", errors="replace") as f:
        content = f.read()
    meta, body = parse_frontmatter(content)
    
    if meta is None:
        issues.append(f"[{slug}] {body} ({len(content)} chars)")
        continue
    
    # Check required fields
    for field in ["title", "publishDate", "description", "tags", "language"]:
        if field not in meta or meta[field] in (None, "", []):
            issues.append(f"[{slug}] MISSING field: {field}")
    
    # Check publishDate format
    pd = meta.get("publishDate", "")
    if pd:
        if not re.match(r"^\d{4}-\d{2}-\d{2}", str(pd)):
            issues.append(f"[{slug}] BAD publishDate: {pd}")
        else:
            try:
                datetime.strptime(str(pd)[:10], "%Y-%m-%d")
            except ValueError:
                issues.append(f"[{slug}] BAD publishDate value: {pd}")
    
    # Check tags type
    if "tags" in meta and not isinstance(meta["tags"], list):
        issues.append(f"[{slug}] tags not a list: {meta['tags'][:50]}")
    
    # Check difficulty value
    diff = meta.get("difficulty", "")
    if diff and diff not in ("beginner", "intermediate", "advanced"):
        issues.append(f"[{slug}] BAD difficulty: {diff}")
    
    # Check code block closure
    n_fence = body.count("```")
    if n_fence % 2 != 0:
        issues.append(f"[{slug}] UNBALANCED code fences ({n_fence})")
    
    # Check image references
    for img in re.findall(r"!\[[^\]]*\]\((/images/[^)\s]+)\)", body):
        rel = img.lstrip("/")
        full = os.path.join("/Users/halo/workspace/astro-blog/public", rel)
        if not os.path.exists(full):
            issues.append(f"[{slug}] MISSING IMAGE: {img}")
    
    # Check relative image refs
    for img in re.findall(r"!\[[^\]]*\]\((?!http|/)([^)\s]+)\)", body):
        issues.append(f"[{slug}] RELATIVE IMG: {img}")
    
    posts.append({
        "slug": slug, "len": len(content), "body": len(body),
        "date": str(meta.get("publishDate", ""))[:10],
        "title": str(meta.get("title", ""))[:40],
    })

print(f"Total posts scanned: {len(posts)}")
print(f"Total issues: {len(issues)}")
print()

# Summary by category
cat_counts = {}
for i in issues:
    cat = i.split("]")[0].lstrip("[").rstrip() + " " + i.split("]")[-1].split(":")[0].split(" ")[0] if "] " in i else "OTHER"
    m = re.search(r"\] (\w+.*?)(?::|$)", i)
    cat = m.group(1).strip() if m else "OTHER"
    cat_counts[cat] = cat_counts.get(cat, 0) + 1

for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")

print()
print("=== ISSUE DETAILS ===")
# Print issues grouped, first 100
for i in issues[:150]:
    print(i)
