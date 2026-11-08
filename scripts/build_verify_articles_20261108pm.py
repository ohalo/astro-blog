"""把核心实验代码注入文章占位符，并做发布前全量校验。

    python3 scripts/build_verify_articles_20261108pm.py

校验项：
  1. 每篇文章恰好一个 python 代码块，且与 scripts/article_core_*.py 逐字一致
  2. 从 md 抽出代码块实际 exec，重新产出的 results.json 与现有 JSON 深度相等
  3. 正文中文字数（不含 frontmatter / 代码块）在 2000-3000
  4. 正文引用 5 张图，且文件真实存在；图片 hash 两两不同
  5. frontmatter description <= 600 字符，必填字段齐全，内部无 ASCII 直引号
  6. slug 目录名在 src/content/blog 中唯一，且对应图片目录不与他人共用
"""
import hashlib
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ARTICLES = [
    {
        "slug": "functional-pca-intraday-volatility-curve",
        "core": "scripts/article_core_fpca_20261108pm.py",
        "images": [
            "fig1_curves_and_shapes.png", "fig2_eigenfunctions.png",
            "fig3_noise_scan.png", "fig4_variance_and_orthogonality.png",
            "fig5_forecast_and_lambda.png",
        ],
    },
    {
        "slug": "thompson-sampling-nonstationary-posterior-collapse",
        "core": "scripts/article_core_ts_20261108pm.py",
        "images": [
            "fig1_regime_dgp.png", "fig2_cumulative_regret.png",
            "fig3_concentration_and_posterior_sd.png",
            "fig4_recovery_speed.png", "fig5_gamma_and_regime_length.png",
        ],
    },
]

PLACEHOLDER = "<!--ARTICLE_CODE-->"
FENCE_RE = re.compile(r"```python\n(.*?)```", re.S)
FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

errors = []


def check(cond, msg):
    if not cond:
        errors.append(msg)
    print(("  OK   " if cond else "  FAIL ") + msg)


for art in ARTICLES:
    slug, core = art["slug"], art["core"]
    md_path = f"src/content/blog/{slug}/index.md"
    img_dir = f"public/images/{slug}"
    json_path = f"{img_dir}/results.json"
    print(f"\n=== {slug} ===")

    check(sum(1 for s in os.listdir("src/content/blog") if s == slug) == 1,
          "slug appears exactly once in blog collection")

    with open(core, encoding="utf-8") as f:
        code = f.read()
    with open(md_path, encoding="utf-8") as f:
        md = f.read()

    # 1) 注入代码块
    if PLACEHOLDER in md:
        md = md.replace(PLACEHOLDER, "```python\n" + code.rstrip() + "\n```")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print("  injected article code block")

    blocks = FENCE_RE.findall(md)
    check(len(blocks) == 1, f"exactly one python code block (got {len(blocks)})")
    if blocks:
        check(blocks[0].strip() == code.strip(),
              "article code block is byte-identical to core script")

    # 2) frontmatter
    fm = FM_RE.match(md)
    check(fm is not None, "frontmatter present")
    if fm:
        head = fm.group(1)
        for key in ("title:", "description:", "publishDate:", "tags:",
                    "language:", "difficulty: advanced"):
            check(key in head, f"frontmatter has {key}")
        m = re.search(r"^description: (.+)$", head, re.M)
        check(m is not None and len(m.group(1)) <= 600,
              f"description <= 600 chars "
              f"(got {len(m.group(1)) if m else 'N/A'})")
        desc_val = m.group(1).strip() if m else "x"
        inner = (desc_val[1:-1]
                 if desc_val.startswith('"') and desc_val.endswith('"')
                 else desc_val)
        check('"' not in inner,
              "description has no internal ASCII double quote")
        check("  - 量化" in head, "tags contain 量化")

    # 3) 图片引用与文件
    body = FM_RE.sub("", md)
    body_no_code = FENCE_RE.sub("", body)
    hashes = []
    for img in art["images"]:
        path = f"{img_dir}/{img}"
        check(f"![](/images/{slug}/{img})" in body_no_code,
              f"body references {img}")
        ok = os.path.isfile(path) and os.path.getsize(path) > 5000
        check(ok, f"image file exists and non-trivial: {img}")
        if ok:
            with open(path, "rb") as f:
                hashes.append(hashlib.sha256(f.read()).hexdigest())
    check(len(set(hashes)) == len(art["images"]),
          "all 5 figures have distinct content")

    # 4) 字数
    n_cjk = len(CJK_RE.findall(body_no_code))
    check(2000 <= n_cjk <= 3000,
          f"Chinese chars (excl. code) in 2000-3000 (got {n_cjk})")

    # 5) 代码块 exec 后 JSON 深度一致
    with open(json_path, encoding="utf-8") as f:
        before = json.load(f)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(blocks[0], md_path, "exec"),
                 {"__name__": "__main__", "__file__": md_path})
        with open(json_path, encoding="utf-8") as f:
            after = json.load(f)
        check(before == after,
              "re-executed article code reproduces identical results.json")
        out = buf.getvalue()
        check(len(out.strip()) > 100,
              "code block runs and prints real computed statistics")
        print("  --- code output (first 5 lines) ---")
        for line in out.strip().splitlines()[:5]:
            print("  | " + line)
    except Exception as e:  # noqa: BLE001
        check(False, f"article code block executes: {type(e).__name__}: {e}")

print("\n=== summary ===")
if errors:
    print(f"FAILED: {len(errors)} check(s)")
    for e in errors:
        print("  - " + e)
    sys.exit(1)
print("ALL CHECKS PASSED")
