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
  7. 禁用词零残留（占位/placeholder/TODO/fake/mock数据 等）；
  8. 数学围栏：`$$` 必须独立成行——跨行块会被 remark-math 吞成整篇 KaTeX 报错（拦）；
     单行 `$$…$$` 只渲染成行内公式（历史遗留 929 处，只警示不拦，免得卡死 cron）。
     要改就跑 `python3 scripts/check_math_fences.py --fix <slug>`。

用法：
  python3 scripts/quant_verify.py <slug1> <core1.py> [<slug2> <core2.py> ...]   单篇全套校验
  python3 scripts/quant_verify.py --all-images                                 全库正文图片引用存在性巡检（Markdown ![]() + 原生 <img>）
  python3 scripts/quant_verify.py --all-links                                  全库站内链接（md 链接 + <a href>）死链巡检
  python3 scripts/quant_verify.py --all-content                                全库只读内容契约报表（默认不失败，存量不硬拦）
  python3 scripts/quant_verify.py --all-content --since 2026-11-25             只查指定日期之后发布的文章，违规即失败
  python3 scripts/quant_verify.py --all-content --git-added [base]             只查相对 base 新增的文章（发布闸门），违规即失败
  python3 scripts/quant_verify.py --all-content slug-a slug-b                  只查显式 slug，违规即失败

退出码 0 = 全部 PASS，1 = 有失败。
"""
import hashlib
import json
import os
import re
import sys
import io
import contextlib
import subprocess
import glob as globmod

from check_math_fences import scan_math

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


def _is_timing(path):
    """壁钟计时类字段：跨运行天然抖动，放宽容差。

    历史规则只认 `t_` 前缀和 `speedup`，导致 `exp5_t_exact_ms` / `ms_exact` /
    `runtime_s` 这类带实验前缀的计时字段被当成精确统计量做 1e-9 比对，
    每次 exec 复跑都会随机失败（11-25 曾靠改字段名规避）。这里按语义匹配：
    键名含独立的时间/计时词片段（t、time、ms、runtime、elapsed、wall、speedup）。
    """
    key = path.rsplit(".", 1)[-1].rsplit("[", 1)[0].lower()
    if "speedup" in key:
        return True
    # `t` 单独作为键名常是确定性的时间网格，不能放宽；只认带后缀/前缀的计时命名
    if re.search(r"(^|_)t_", key) or re.search(r"_t($|_|\d)", key):
        return True
    return any(
        re.search(rf"(^|_){word}(_|$|\d)", key)
        for word in ("time", "ms", "runtime", "elapsed", "wall")
    )


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
        if _is_timing(path):  # 计时字段：同数量级即可（相对 50%）
            return abs(a - b) <= 0.5 * max(1.0, abs(a), abs(b))
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

    probs = scan_math(body)
    hard = [p for p in probs if p[0] != "inline_math"]
    soft = [p for p in probs if p[0] == "inline_math"]
    assert not hard, (
        "数学围栏会把后半篇整篇吞进 KaTeX 报错块："
        + "；".join(f"L{ln} {k}" for _, ln, k in hard)
        + f"（先跑 python3 scripts/check_math_fences.py --fix {slug}）"
    )
    if soft:
        # 历史遗留写法（全站 929 处），只警示不拦：拦了会把 cron 发布卡死。
        # 批量批改完可以把它也升成 assert。
        print(f"  [math] ⚠ {len(soft)} 处单行 $$…$$ 只渲染成行内公式（不居中）："
              + " ".join(f"L{ln}" for _, ln, _ in soft)
              + f"（要改：python3 scripts/check_math_fences.py --fix {slug}）")
    else:
        print("  [math] $$ 围栏均独立成行，无跨行吞并 / 行内降级")

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
    assert deep_equal(json.loads(results_after), saved), \
        "results.json 被 exec 改写到超出容差"
    print("  [exec] 复跑输出与 results.json 深度相等")

    body_low = body.lower()
    for w in BANNED:
        assert w not in body_low, f"禁用词残留：{w}"
    print("  [words] 禁用词零残留")
    return True


def verify_all_images():
    """全库扫描：每篇 md 正文引用的 /images/ 路径都必须在 public 下真实存在。

    同时覆盖 Markdown `![]()` 与原生 HTML `<img src="...">` 两种写法——
    历史上 digital-minimalism 一文用 <img> 标签引用了 3 张不存在的图，
    只扫 Markdown 语法会漏检（单篇 verify() 只检查传入 slug 的 5 张图，
    历史文章不走 cron，破图会长期漏检）。
    """
    import glob

    blog_dir = os.path.join(ROOT, "src", "content", "blog")
    missing = []
    ref_count = 0
    img_re = re.compile(
        r"!\[[^\]]*\]\((/images/[^)]+)\)"      # Markdown 图片
        r"|<img[^>]+src=\"(/images/[^\"]+)\""   # 原生 HTML <img>
    )
    for md_path in glob.glob(os.path.join(blog_dir, "**", "*.md"), recursive=True):
        txt = open(md_path, encoding="utf-8", errors="ignore").read()
        for m in img_re.finditer(txt):
            url = m.group(1) or m.group(2)
            ref_count += 1
            rel = url.split("?")[0]
            if not os.path.exists(os.path.join(ROOT, "public", rel.lstrip("/"))):
                slug = os.path.basename(os.path.dirname(md_path))
                missing.append((slug, url))
    print(f"  [all-images] 扫描正文图片引用 {ref_count} 处，缺失 {len(missing)} 处")
    for slug, url in missing:
        print(f"    ✗ {slug}: {url}")
    assert not missing, f"全库有 {len(missing)} 处破图引用（见上）"
    return True


def _collect_routes_and_tags():
    """构建站内有效路由集合（静态页 + 博客 slug）与全量标签集合。"""
    import glob

    routes = set()
    pages_dir = os.path.join(ROOT, "src", "pages")
    for dirpath, _dirnames, filenames in os.walk(pages_dir):
        for fn in filenames:
            if not fn.endswith((".astro", ".md")):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), pages_dir)
            rel = os.path.splitext(rel)[0]
            if os.path.basename(rel) == "index":
                rel = os.path.dirname(rel)
            route = "/" + rel.replace(os.sep, "/")
            if "[" not in route:  # 动态路由单独按内容推导
                routes.add(route)
    blog_dir = os.path.join(ROOT, "src", "content", "blog")
    tags = set()
    for md_path in glob.glob(os.path.join(blog_dir, "**", "*.md"), recursive=True):
        routes.add("/blog/" + os.path.relpath(os.path.dirname(md_path), blog_dir).replace(os.sep, "/"))
        txt = open(md_path, encoding="utf-8", errors="ignore").read()[:1500]
        m = re.search(r"^tags:\s*\[(.*?)\]", txt, re.M)
        lst = [x.strip().strip("'\"") for x in m.group(1).split(",")] if m else []
        m2 = re.search(r"^tags:\n((?:\s+- .*\n)+)", txt, re.M)
        if m2:
            lst = [ln.split("-", 1)[1].strip().strip("'\"") for ln in m2.group(1).strip().splitlines()]
        tags.update(x for x in lst if x)
    return routes, tags


def verify_all_links():
    """全库扫描：md 正文里的站内链接（Markdown 链接 + 原生 <a href>）必须可达。

    覆盖相对路径与同域绝对 URL（https://blog.halo26812.eu.org/...）。
    可达判定：public/ 下存在文件、静态页路由、博客 slug、标签页
    （astro-pure 会把标签统一转小写，故大小写不敏感比较）、/blog 分页数字。
    站点历史上 /posts/... 旧结构链接曾 10 处 404 长期漏检，本命令兜底。
    """
    import glob
    from urllib.parse import unquote

    blog_dir = os.path.join(ROOT, "src", "content", "blog")
    routes, tags = _collect_routes_and_tags()
    tag_low = {t.lower() for t in tags}
    link_re = re.compile(
        r"(?<!\!)\[[^\]]*\]\((/[^)\s]+|https://blog\.halo26812\.eu\.org[^)\s]*)\)"
        r"|<a[^>]+href=\"(/[^\"]+|https://blog\.halo26812\.eu\.org[^\"]*)\""
    )
    missing = []
    ref_count = 0
    for md_path in glob.glob(os.path.join(blog_dir, "**", "*.md"), recursive=True):
        txt = open(md_path, encoding="utf-8", errors="ignore").read()
        txt = re.sub(r"^---\n.*?\n---\n", "", txt, flags=re.S)  # 去 frontmatter
        txt = re.sub(r"```.*?\n```", "", txt, flags=re.S)       # 去代码块
        for m in link_re.finditer(txt):
            raw = m.group(1) or m.group(2)
            if not raw:
                continue
            ref_count += 1
            u = raw.replace("https://blog.halo26812.eu.org", "")
            u = unquote(u.split("#")[0].split("?")[0])
            if not u:
                continue  # 纯锚点
            path = u.rstrip("/") or "/"
            ok = False
            if path == "/":
                ok = True
            elif not path.startswith(("/blog", "/tags")) and os.path.exists(
                os.path.join(ROOT, "public", path.lstrip("/"))
            ):
                ok = True  # public 静态资产/文件
            elif path in routes:
                ok = True
            elif path == "/blog" or path.startswith("/en/blog"):
                ok = True
            elif path.startswith("/blog/") and (path[len("/blog/"):].isdigit() or path.endswith("rss.xml")):
                ok = True
            elif path == "/tags":
                ok = True
            elif path.startswith("/tags/"):
                ok = path[len("/tags/"):].lower() in tag_low  # astro-pure 将标签统一转小写
            if not ok:
                missing.append((os.path.basename(os.path.dirname(md_path)), raw))
    print(f"  [all-links] 扫描站内链接 {ref_count} 处，死链 {len(missing)} 处")
    for slug, url in missing:
        print(f"    ✗ {slug}: {url}")
    assert not missing, f"全库有 {len(missing)} 处死链（见上）"
    return True


def _fm_value(fm, key):
    m = re.search(rf'^{key}:[ \t]*(.*)$', fm, re.M)
    return m.group(1).strip() if m else None


def _fm_tags(fm):
    """兼容 `tags: [a, b]` 与 YAML 列表两种写法。"""
    m = re.search(r'^tags:[ \t]*(.*)$', fm, re.M)
    if not m:
        return []
    inline = m.group(1).strip()
    if inline.startswith('[') and inline.endswith(']'):
        return [x.strip().strip('"\'') for x in inline[1:-1].split(',') if x.strip()]
    tags = []
    for line in fm[m.end():].splitlines():
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*:', line):
            break
        item = re.match(r'^\s*-\s+(.+)$', line)
        if item:
            tags.append(item.group(1).strip().strip('"\''))
    return tags


def content_contract(md_path):
    """只读检查一篇的内容契约，返回 (errors, warnings)。

    与 verify() 的区别：绝不调用 stitch()，不写回任何文件；只做静态契约检查。
    量化高阶文章（difficulty: advanced + 量化类 tag）才检查 5 图 / 单代码块 /
    可运行入口 / results.json，避免把普通生活类文章误判。
    """
    txt = open(md_path, encoding='utf-8').read()
    m = re.match(r'^---\n(.*?)\n---\n', txt, re.S)
    if not m:
        return ['frontmatter 解析失败'], []
    fm, body = m.group(1), txt[m.end():]
    slug = os.path.basename(os.path.dirname(md_path))
    errors, warns = [], []

    desc = re.search(r'^description:\s*"?([^"\n]*)"?\s*$', fm, re.M)
    if not desc or len(desc.group(1)) > 600:
        errors.append(f'description 缺失或超过 600 字符（{len(desc.group(1)) if desc else 0}）')
    if not re.search(r'^publishDate:\s*\S', fm, re.M):
        errors.append('缺 publishDate')
    tags = _fm_tags(fm)
    difficulty = _fm_value(fm, 'difficulty') or ''
    is_quant = difficulty.strip('"\'') == 'advanced' and any(
        t in ('量化', 'quant', '交易', '投资') for t in tags
    )

    if is_quant:
        refs = re.findall(r'!\[.*?\]\(/images/' + re.escape(slug) + r'/([^)]+)\)', txt)
        if len(refs) != 5:
            errors.append(f'图片引用 {len(refs)} 张（应为 5）')
        else:
            img_dir = os.path.join(ROOT, 'public', 'images', slug)
            hashes = set()
            for f in refs:
                p = os.path.join(img_dir, f)
                if not os.path.isfile(p):
                    errors.append(f'缺图 {f}')
                    continue
                h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
                if h in hashes:
                    errors.append(f'重复图 {f}')
                hashes.add(h)
        blocks = re.findall(r'```python\n(.*?)\n```', txt, re.S)
        if len(blocks) != 1:
            errors.append(f'python 代码块 {len(blocks)} 个（应为 1）')
        else:
            if '__main__' not in blocks[0]:
                errors.append('代码块缺 __main__ 入口（不可独立运行）')
            if 'savefig' not in blocks[0]:
                warns.append('代码块无 savefig（图片生成方式不可判定）')
        if not os.path.isfile(os.path.join(ROOT, 'public', 'images', slug, 'results.json')):
            errors.append('缺 results.json')
        cjk = len(re.findall(r'[\u4e00-\u9fff]', body))
        if not 2000 <= cjk <= 3000:
            errors.append(f'正文中文字数 {cjk} 不在 2000–3000')
        hard = [p for p in scan_math(body) if p[0] != 'inline_math']
        if hard:
            errors.append(f'数学围栏 HARD {len(hard)} 处：' + ' '.join(f'L{ln}' for _, ln, _ in hard[:5]))
        body_low = body.lower()
        hit = [w for w in BANNED if w in body_low]
        if hit:
            errors.append(f'禁用词残留 {hit}')
    return errors, warns


def _git_added_slugs(base):
    """取相对 base 新增的文章 slug（含已暂存），刻意不含未跟踪文件——用户 WIP 不参与闸门。"""
    paths = set()
    for extra in (['diff', '--name-only', '--diff-filter=A', f'{base}..HEAD'],
                  ['diff', '--name-only', '--diff-filter=A', '--cached']):
        out = subprocess.run(['git'] + extra, cwd=ROOT, capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if line.startswith('src/content/blog/') and line.endswith('/index.md'):
                paths.add(line)
    return [os.path.basename(os.path.dirname(p)) for p in sorted(paths)]


def verify_all_content(since=None, git_added=None, slugs=None, strict=False):
    """内容契约巡检：全库默认只出报表；限定范围或 --strict 时违规即失败。"""
    if slugs:
        paths = [os.path.join(ROOT, 'src', 'content', 'blog', s, 'index.md') for s in slugs]
    elif git_added is not None:
        paths = [os.path.join(ROOT, 'src', 'content', 'blog', s, 'index.md')
                 for s in _git_added_slugs(git_added)]
    else:
        paths = sorted(globmod.glob(os.path.join(ROOT, 'src', 'content', 'blog', '**', 'index.md'), recursive=True))
    if since:
        def _pub(p):
            txt = open(p, encoding='utf-8').read()
            fm = re.match(r'^---\n(.*?)\n---\n', txt, re.S)
            return (_fm_value(fm.group(1), 'publishDate') or '').strip('"\'')[:10] if fm else ''
        paths = [p for p in paths if os.path.isfile(p) and _pub(p) >= since]

    bad, quant_n = [], 0
    for p in paths:
        if not os.path.isfile(p):
            bad.append((os.path.basename(os.path.dirname(p)), ['文章不存在'], []))
            continue
        errors, warns = content_contract(p)
        slug = os.path.basename(os.path.dirname(p))
        txt = open(p, encoding='utf-8').read()
        fm = re.match(r'^---\n(.*?)\n---\n', txt, re.S)
        fm = fm.group(1) if fm else ''
        if (_fm_value(fm, 'difficulty') or '').strip('"\'') == 'advanced':
            quant_n += 1
        if errors or warns:
            bad.append((slug, errors, warns))
    scope = f'since={since}' if since else (f'git-added({git_added})' if git_added is not None else ('explicit' if slugs else '全库'))
    print(f'  [all-content] 范围 {scope}：扫描 {len(paths)} 篇（advanced {quant_n} 篇），有问题 {len(bad)} 篇')
    for slug, errors, warns in bad[:40]:
        for e in errors:
            print(f'    ✗ {slug}: {e}')
        for w in warns:
            print(f'    ⚠ {slug}: {w}')
    if len(bad) > 40:
        print(f'    ... 另有 {len(bad) - 40} 篇未展开')
    fail = [b for b in bad if b[1]]
    if fail:
        print(f'  [all-content] 硬违规 {len(fail)} 篇（只出报表，不失败）' if not (strict or since or git_added is not None or slugs)
              else f'  [all-content] FAIL：{len(fail)} 篇硬违规')
    else:
        print('  [all-content] 无硬违规')
    return not fail or not (strict or since or git_added is not None or slugs)


def main():
    args = sys.argv[1:]
    # 全库巡检模式：与按 slug 的单篇校验互斥。
    if args and args[0] == '--all-content':
        rest = args[1:]
        strict = '--strict' in rest
        rest = [x for x in rest if x != '--strict']
        since = git_added = None
        if '--since' in rest:
            i = rest.index('--since')
            assert i + 1 < len(rest), '--since 需要一个日期参数'
            since = rest[i + 1]
            del rest[i:i + 2]
        if '--git-added' in rest:
            i = rest.index('--git-added')
            has_base = i + 1 < len(rest) and not rest[i + 1].startswith('-')
            git_added = rest[i + 1] if has_base else 'HEAD~1'
            del rest[i:i + 2 if has_base else 1]
        assert not (since and git_added is not None), '--since 与 --git-added 不能同时用'
        assert not (rest and (since or git_added is not None)), '显式 slug 不能与 --since/--git-added 同时用'
        try:
            ok = verify_all_content(since=since, git_added=git_added, slugs=rest or None, strict=strict)
        except Exception as e:  # noqa
            print(f'  ✗ ERROR: {e}')
            sys.exit(1)
        sys.exit(0 if ok else 1)
    if args and args[0] in ("--all-images", "--all-links"):
        assert len(args) == 1, f"{args[0]} 不接受其他参数"
        try:
            verify_all_images() if args[0] == "--all-images" else verify_all_links()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            sys.exit(1)
        print("ALL CHECKS PASSED")
        sys.exit(0)
    assert len(args) >= 2 and len(args) % 2 == 0, "用法: quant_verify.py <slug> <core.py> [...] | --all-images | --all-links | --all-content [--since DATE | --git-added [base] | slug...] [--strict]"
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
