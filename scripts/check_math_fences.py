# -*- coding: utf-8 -*-
"""站内数学公式围栏检查 / 修复（remark-math v6 + rehype-katex 的实际行为为准）。

为什么需要：remark-math v6 关不掉「$$ 开头、闭合的 $$ 在后面某行」这种块。它不会
只影响那一个公式——从那一行起整篇正文被吞进一个 math 节点，KaTeX 报
"Can't use function '$' in math mode"，后半篇的表格、图片、行内公式全部掉成裸
markdown（cramer-rao-mle-finite-sample 就是这么线上翻车的：85 个公式只出图 22 个）。

两类问题：
  HARD  unclosed     `$$content` 开头、闭合在后续行  -> 整篇后半边被吞
  HARD  unbalanced   正文 `$$` 出现次数为奇数        -> 有块没关上
  SOFT  inline_math  独占一行但写成单行 `$$…$$`      -> 渲染成行内公式（不居中、
                     不进 display 块，`\\tag` 直接报错），应拆成独立围栏

站内的正确写法（现 159+ 篇在用）：`$$` 单独占一行。

用法：
  python3 scripts/check_math_fences.py                # 扫全站文章，只报告
  python3 scripts/check_math_fences.py <slug|路径> ...  # 只查指定文章
  python3 scripts/check_math_fences.py --fix [同上]    # 就地修（幂等）
  python3 scripts/check_math_fences.py --quiet          # 只打印有问题的

退出码 0 = 无 HARD 问题，1 = 有 HARD 问题。
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FENCE = "```"


def scan_math(text):
    """返回 [(kind, lineno, snippet)]；kind ∈ unclosed / unbalanced / inline_math。"""
    lines = text.split("\n")
    prose, infence = [], False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith(FENCE):
            infence = not infence
            prose.append((i, None))
            continue
        prose.append((i, None if infence else line))

    problems = []
    opened = None
    fences = 0
    for lineno, line in prose:
        if line is None:
            continue
        fences += line.count("$$")
        if line.startswith("$$") and line[2:].strip():
            if not re.match(r"^\$\$.*\$\$", line):          # 同行没闭合
                problems.append(("unclosed", lineno, line[:90]))
            elif re.match(r"^\$\$\s*\S.*\s*\$\$$", line):   # 整行就一对 -> 行内
                problems.append(("inline_math", lineno, line[:90]))
            opened = None
        elif opened is None and "$$" in line:
            opened = lineno
    if fences % 2:
        problems.append(("unbalanced", 0, f"正文 $$ 出现 {fences} 次（奇数）"))
    return problems


def fix_math(text):
    """把两类问题改成独立成行的 $$ 围栏。代码块内不动。返回 (新文本, 修复数)。"""
    lines = text.split("\n")
    out, infence, n = [], False, 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith(FENCE):
            infence = not infence
            out.append(line)
            i += 1
            continue
        if infence:
            out.append(line)
            i += 1
            continue

        m = re.match(r"^\$\$\s*(\S.*?)\s*\$\$$", line)       # SOFT：单行一对
        if m:
            if out and out[-1].strip():
                out.append("")
            out += ["$$", m.group(1), "$$"]
            n += 1
            i += 1
            continue

        if line.startswith("$$") and line[2:].strip() and not re.match(r"^\$\$.*\$\$", line):
            head = line[2:].rstrip()                          # HARD：跨行块
            if out and out[-1].strip():
                out.append("")
            out.append("$$")
            body = [head] if head else []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.lstrip().startswith(FENCE) or nxt.startswith("$$"):
                    break
                if "$$" in nxt:
                    cut = nxt.rfind("$$")
                    if nxt[:cut].strip():
                        body.append(nxt[:cut].rstrip())
                    body.append("$$")
                    tail = nxt[cut + 2:].strip()
                    if tail:
                        body += ["", tail]
                        n += 1
                    i += 1
                    break
                body.append(nxt.rstrip())
                i += 1
            if body[-1] != "$$":
                body.append("$$")
            out += body
            if i < len(lines) and lines[i].strip():
                out.append("")
            n += 1
            continue

        out.append(line)
        i += 1
    return "\n".join(out), n


def targets(args):
    if not args:
        return sorted(glob.glob(os.path.join(ROOT, "src/content/blog/*/index.md")))
    out = []
    for a in args:
        p = a if os.path.sep in a else os.path.join(ROOT, "src/content/blog", a, "index.md")
        p = os.path.abspath(p if p.endswith(".md") else os.path.join(p, "index.md"))
        if not os.path.isfile(p):
            sys.exit(f"找不到文章：{p}")
        out.append(p)
    return out


def main():
    argv = sys.argv[1:]
    do_fix = "--fix" in argv
    quiet = "--quiet" in argv
    quiet = quiet or do_fix
    pos = [a for a in argv if not a.startswith("--")]

    hard = soft = fixed = 0
    corpus = not pos
    soft_by_file = {}
    for p in targets(pos):
        txt = open(p, encoding="utf-8").read()
        probs = scan_math(txt)
        if probs and do_fix:
            new, n = fix_math(txt)
            if n:
                open(p, "w", encoding="utf-8").write(new)
                fixed += n
                probs = scan_math(new)
        h = [x for x in probs if x[0] != "inline_math"]
        s = [x for x in probs if x[0] == "inline_math"]
        hard += len(h)
        soft += len(s)
        for kind, lineno, snip in h:
            print(f'HARD  {kind:12} {"/".join(p.split(os.sep)[-3:-1])}:{lineno}  {snip}')
        if s and (not corpus or "--all" in argv):
            for kind, lineno, snip in s:
                print(f'SOFT  {kind:12} {"/".join(p.split(os.sep)[-3:-1])}:{lineno}  {snip}')
        elif s:
            soft_by_file["/".join(p.split(os.sep)[-3:-1])] = len(s)

    if do_fix:
        print(f"已修复 {fixed} 处围栏")
    if corpus and soft_by_file:
        for name, cnt in sorted(soft_by_file.items(), key=lambda kv: -kv[1])[:10]:
            print(f"SOFT  inline_math  {name}  ×{cnt}")
        if len(soft_by_file) > 10:
            print(f"      …另 {len(soft_by_file) - 10} 篇，共 {soft} 处；--all 看全部")
    if not quiet or hard or soft:
        print(f"扫描 {'全站' if corpus else str(len(pos)) + ' 篇'}：HARD {hard} 处，SOFT {soft} 处"
              + ("（SOFT = 单行 $$…$$ 会渲染成行内公式，不居中）" if soft else ""))
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
