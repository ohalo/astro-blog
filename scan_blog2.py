#!/usr/bin/env python3
"""Reliable scan of all blog posts using PyYAML for frontmatter."""
import os, re
import yaml

BLOG_DIR = "/Users/halo/workspace/astro-blog/src/content/blog"

REQUIRED = ["title", "publishDate", "description", "tags", "language"]
DIFF_ENUM = {"beginner", "intermediate", "advanced"}

def parse_frontmatter(text):
    if not text.startswith("---"):
        return None, "NO_FRONTMATTER"
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return None, "FRONTMATTER_UNCLOSED"
    try:
        meta = yaml.safe_load(m.group(1))
    except Exception as e:
        return None, f"YAML_ERROR: {e}"
    if not isinstance(meta, dict):
        return None, "META_NOT_DICT"
    return meta, text[m.end():]

issues = []
posts = []

for slug in sorted(os.listdir(BLOG_DIR)):
    p = os.path.join(BLOG_DIR, slug)
    if not os.path.isdir(p):
        continue
    md = os.path.join(p, "index.md")
    if not os.path.exists(md):
        issues.append(f"[{slug}] MISSING_INDEX_MD")
        continue
    with open(md, encoding="utf-8", errors="replace") as f:
        content = f.read()
    meta, body = parse_frontmatter(content)
    if meta is None:
        issues.append(f"[{slug}] FM: {body[:60]}")
        continue

    for field in REQUIRED:
        val = meta.get(field)
        if val is None or val == "" or val == []:
            issues.append(f"[{slug}] MISSING_FIELD: {field}")

    pd = meta.get("publishDate")
    if pd is not None:
        try:
            from datetime import datetime
            if hasattr(pd, "strftime"):
                pass
            else:
                datetime.strptime(str(pd)[:10], "%Y-%m-%d")
        except Exception as e:
            issues.append(f"[{slug}] BAD_PUBLISHDATE: {pd}")

    tags = meta.get("tags")
    if isinstance(tags, list) and len(tags) == 0:
        issues.append(f"[{slug}] EMPTY_TAGS")

    diff = meta.get("difficulty")
    if diff is not None and diff not in DIFF_ENUM:
        issues.append(f"[{slug}] BAD_DIFFICULTY: {diff}")

    # Code fence balance
    lines = body.splitlines(keepends=True)
    in_code = False
    fence_issues = 0
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("```"):
            in_code = not in_code
    if in_code:
        issues.append(f"[{slug}] UNCLOSED_CODE_FENCE")

    # Image refs vs files
    for img in re.findall(r"!\[[^\]]*\]\((/images/[^)\s]+)\)", body):
        full = os.path.join("/Users/halo/workspace/astro-blog/public", img.lstrip("/"))
        if not os.path.exists(full):
            issues.append(f"[{slug}] BROKEN_IMG: {img}")

    posts.append({"slug": slug, "len": len(content), "date": str(pd)[:10] if pd else ""})

# Categorize
cat = {}
for i in issues:
    key = re.search(r"\]\s+(\w+)", i)
    k = key.group(1) if key else "OTHER"
    cat[k] = cat.get(k, 0) + 1

print(f"Posts scanned: {len(posts)}")
print(f"Total issues: {len(issues)}")
print("By category:")
for k, n in sorted(cat.items(), key=lambda x: -x[1]):
    print(f"  {k}: {n}")

# Save full issue list
with open("/Users/halo/workspace/astro-blog/_scan_issues.txt", "w") as f:
    f.write("\n".join(issues))
print("\nSaved issue list to _scan_issues.txt")
print("\n=== Broken images & structural issues only ===")
for i in issues:
    if any(k in i for k in ["BROKEN_IMG", "UNCLOSED", "MISSING_INDEX", "BAD_DIFF", "YAML_ERROR", "EMPTY_TAGS", "BAD_PUB"]):
        print(i)
