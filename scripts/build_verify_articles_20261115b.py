# -*- coding: utf-8 -*-
"""2026-11-15 傍晚批次：拼接代码块 + 全量校验两篇文章。"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTICLES = [
    {
        "slug": "kalman-smoother-hp-filter-lookahead",
        "core": "scripts/article_core_hp_smooth_20261115b.py",
    },
    {
        "slug": "optimal-stopping-trade-exit",
        "core": "scripts/article_core_optimal_stopping_20261115b.py",
    },
]


def stitch(md_path, core_path):
    txt = open(md_path, encoding="utf-8").read()
    code = open(core_path, encoding="utf-8").read().rstrip("\n")
    block = "```python\n" + code + "\n```"
    if "<!--CODE-->" in txt:
        txt = txt.replace("<!--CODE-->", block + "\n")
        open(md_path, "w", encoding="utf-8").write(txt)
        print("  [stitch] 代码块已注入")
    else:
        blocks = re.findall(r"```python\n(.*?)\n```", txt, re.S)
        assert len(blocks) == 1, f"代码块数量 {len(blocks)}"
        if blocks[0].rstrip("\n") != code:
            txt = txt.replace("```python\n" + blocks[0] + "\n```", block, 1)
            open(md_path, "w", encoding="utf-8").write(txt)
            print("  [stitch] 代码块已更新为最新 core")
        else:
            print("  [stitch] 代码块已就位（幂等跳过）")


def deep_equal(a, b, path="root"):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            print(f"  keys differ at {path}: {set(a) ^ set(b)}")
            return False
        return all(deep_equal(a[k], b[k], f"{path}.{k}") for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            print(f"  len differ at {path}")
            return False
        return all(deep_equal(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a == b
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def main():
    ok = True
    for art in ARTICLES:
        slug, core_rel = art["slug"], art["core"]
        print(f"=== {slug} ===")
        md_path = os.path.join(ROOT, "src", "content", "blog", slug, "index.md")
        core_path = os.path.join(ROOT, core_rel)
        img_dir = os.path.join(ROOT, "public", "images", slug)

        stitch(md_path, core_path)

        txt = open(md_path, encoding="utf-8").read()
        m = re.match(r"^---\n(.*?)\n---\n", txt, re.S)
        assert m, "frontmatter 解析失败"
        fm = m.group(1)
        desc = re.search(r'description: "(.*)"', fm).group(1)
        assert len(desc) <= 600, f"desc 超长 {len(desc)}"
        assert '"' not in desc, "desc 内含 ASCII 直引号"
        assert "publishDate:" in fm and "language: zh" in fm
        assert "difficulty: advanced" in fm
        assert re.search(r"^\s+- 量化$", fm, re.M), "缺 量化 tag"
        print(f"  [fm] desc={len(desc)} tags/difficulty/language OK")

        body = txt[m.end():]
        prose = re.sub(r"```python\n.*?\n```", "", body, flags=re.S)
        cjk = len(re.findall(r"[\u4e00-\u9fff]", prose))
        assert 2000 <= cjk <= 3000, f"正文中文字数（不含代码块）{cjk} 不在 2000-3000"
        print(f"  [len] 正文中文字数（不含代码块）{cjk}")

        blocks = re.findall(r"```python\n(.*?)\n```", txt, re.S)
        assert len(blocks) == 1, f"代码块数量 {len(blocks)}"
        core_src = open(core_path, encoding="utf-8").read().rstrip("\n")
        assert blocks[0].rstrip("\n") == core_src, "代码块与 core 不一致"
        print("  [code] 与 core 逐字一致")

        refs = re.findall(r"!\[.*?\]\(/images/" + slug + r"/([^)]+)\)", txt)
        assert len(refs) == 5, f"图片引用 {len(refs)}"
        hashes = set()
        for f in refs:
            p = os.path.join(img_dir, f)
            assert os.path.isfile(p), f"缺图 {f}"
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            assert h not in hashes, f"重复图 {f}"
            hashes.add(h)
        print("  [img] 5 张引用齐全且 hash 各不相同")

        import io
        import contextlib
        buf = io.StringIO()
        results_before = open(os.path.join(img_dir, "results.json"), encoding="utf-8").read()
        g = {"__file__": core_path, "__name__": "__main__"}
        with contextlib.redirect_stdout(buf):
            exec(compile(core_src, core_path, "exec"), g)
        out = buf.getvalue()
        fresh = json.loads(out[out.index("{"):])
        saved = json.loads(results_before)
        if not deep_equal(fresh, saved):
            ok = False
            print("  [exec] 复跑输出与 results.json 不一致")
        else:
            print("  [exec] 复跑输出与 results.json 深度相等")
        results_after = open(os.path.join(img_dir, "results.json"), encoding="utf-8").read()
        assert json.loads(results_after) == saved, "results.json 被改写"
        low = prose.lower()
        for w in ["占位", "placeholder", "todo", "fixme", "fake", "mock数据"]:
            assert w not in low, f"禁用词 {w}"
        print("  [words] 禁用词零残留")

    print("ALL CHECKS PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
