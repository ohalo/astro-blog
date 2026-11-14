# -*- coding: utf-8 -*-
"""通用量化文章校验器（dsh 机械流水线用，替代每批次手写 build_verify_*.py）。

对每篇文章做与线上规约一致的全套硬校验：
  1. 把 core 实验脚本注入 md 的 <!--CODE--> 占位符（或核对已有代码块逐字一致）；
  2. frontmatter：description ≤600 字符且无 ASCII 直引号、含 publishDate / language: zh /
     difficulty: advanced / 「量化」tag；
  3. 正文中文字数（不含代码块）2000–3000；
  4. 代码块与 core 脚本逐字一致；
  5. 正文引用 5 张图、文件均存在、5 张 hash 各不相同；
  6. exec 复跑 core，输出 JSON 与 results.json 深度相等；
  7. 禁用词零残留（占位/placeholder/TODO/fake/mock数据 等）。

用法：
  python3 scripts/quant_verify.py <slug1> <core1.py> [<slug2> <core2.py> ...]

退出码 0 = 全部 PASS，1 = 有失败。
"""
import hashlib
import json
import os
import re
import sys
import io
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANNED = ["占位", "placeholder", "todo", "fixme", "fake", "mock数据", "伪造"]


def stitch(md_path, core_path):
    """把 core 脚本注入 md：优先替换 <!--CODE--> 占位符；否则核对唯一 python 代码块一致。"""
    txt = open(md_path, encoding="utf-8").read()
    code = open(core_path, encoding="utf-8").read().rstrip("\n")
    block = "```python\n" + code + "\n```"
    if "<!--CODE-->" in txt:
        txt = txt.replace("<!--CODE-->", block + "\n")
        open(md_path, "w", encoding="utf-8").write(txt)
        print("  [stitch] 代码块已注入 <!--CODE-->")
    else:
        blocks = re.findall(r"```python\n(.*?)\n```", txt, re.S)
        assert len(blocks) == 1, f"python 代码块数量 {len(blocks)}（应为 1）"
        if blocks[0].rstrip("\n") != code:
            txt = txt.replace("```python\n" + blocks[0] + "\n```", block, 1)
            open(md_path, "w", encoding="utf-8").write(txt)
            print("  [stitch] 代码块已更新为最新 core")
        else:
            print("  [stitch] 代码块已就位（幂等跳过）")


def deep_equal(a, b, path="root"):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            print(f"  [exec] keys differ at {path}: {set(a) ^ set(b)}")
            return False
        return all(deep_equal(a[k], b[k], f"{path}.{k}") for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            print(f"  [exec] len differ at {path}: {len(a)} vs {len(b)}")
            return False
        return all(deep_equal(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a == b
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def verify(slug, core_rel):
    print(f"=== {slug} ===")
    md_path = os.path.join(ROOT, "src", "content", "blog", slug, "index.md")
    core_path = os.path.join(ROOT, core_rel)
    img_dir = os.path.join(ROOT, "public", "images", slug)
    assert os.path.isfile(md_path), f"缺文章 {md_path}"
    assert os.path.isfile(core_path), f"缺 core {core_path}"

    stitch(md_path, core_path)

    txt = open(md_path, encoding="utf-8").read()
    m = re.match(r"^---\n(.*?)\n---\n", txt, re.S)
    assert m, "frontmatter 解析失败"
    fm = m.group(1)
    desc = re.search(r'description: "(.*)"', fm).group(1)
    assert len(desc) <= 600, f"description 超长 {len(desc)}"
    assert '"' not in desc, "description 内含 ASCII 直引号（会提前闭合 YAML 标量）"
    assert "publishDate:" in fm and "language: zh" in fm
    assert "difficulty: advanced" in fm
    assert re.search(r"^\s+- 量化$", fm, re.M), "缺「量化」tag（专栏页不渲染）"
    print(f"  [fm] description={len(desc)} 字符，tags/difficulty/language OK")

    body = txt[m.end():]
    cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
    assert 2000 <= cjk <= 3000, f"正文中文字数 {cjk} 不在 2000–3000"
    print(f"  [len] 正文中文字数 {cjk}")

    blocks = re.findall(r"```python\n(.*?)\n```", txt, re.S)
    assert len(blocks) == 1, f"python 代码块数量 {len(blocks)}"
    core_src = open(core_path, encoding="utf-8").read().rstrip("\n")
    assert blocks[0].rstrip("\n") == core_src, "代码块与 core 逐字不一致"
    print("  [code] 与 core 逐字一致")

    refs = re.findall(r"!\[.*?\]\(/images/" + re.escape(slug) + r"/([^)]+)\)", txt)
    assert len(refs) == 5, f"图片引用 {len(refs)} 张（应为 5）"
    hashes = set()
    for f in refs:
        p = os.path.join(img_dir, f)
        assert os.path.isfile(p), f"缺图 {f}"
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert h not in hashes, f"重复图（hash 碰撞）{f}"
        hashes.add(h)
    print("  [img] 5 张引用齐全、文件存在、hash 各不相同")

    buf = io.StringIO()
    results_before = open(os.path.join(img_dir, "results.json"), encoding="utf-8").read()
    g = {"__file__": core_path, "__name__": "__main__"}
    with contextlib.redirect_stdout(buf):
        exec(compile(core_src, core_path, "exec"), g)
    out = buf.getvalue()
    fresh = json.loads(out[out.index("{"):])
    saved = json.loads(results_before)
    assert deep_equal(fresh, saved), "exec 复跑输出与 results.json 不一致"
    results_after = open(os.path.join(img_dir, "results.json"), encoding="utf-8").read()
    assert json.loads(results_after) == saved, "results.json 被 exec 改写"
    print("  [exec] 复跑输出与 results.json 深度相等")

    body_low = body.lower()
    for w in BANNED:
        assert w not in body_low, f"禁用词残留：{w}"
    print("  [words] 禁用词零残留")
    return True


def main():
    args = sys.argv[1:]
    assert len(args) >= 2 and len(args) % 2 == 0, "用法: quant_verify.py <slug> <core.py> [<slug> <core.py> ...]"
    ok = True
    for i in range(0, len(args), 2):
        try:
            ok &= verify(args[i], args[i + 1])
        except AssertionError as e:
            ok = False
            print(f"  ✗ FAIL: {e}")
        except Exception as e:  # noqa
            ok = False
            print(f"  ✗ ERROR: {e}")
    print("ALL CHECKS PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
