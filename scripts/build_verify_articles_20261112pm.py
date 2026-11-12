"""Inject both core scripts into the two 2026-11-12 afternoon articles and verify.

Run with system python3 (needs numpy/scipy/matplotlib):

    python3 scripts/build_verify_articles_20261112pm.py
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLES = [
    {
        "slug": "probability-calibration-brier-isotonic",
        "core": "scripts/article_core_calibration_20261112pm.py",
        "md": "src/content/blog/probability-calibration-brier-isotonic/index.md",
    },
    {
        "slug": "rolling-window-bias-variance-drift",
        "core": "scripts/article_core_rolling_window_20261112pm.py",
        "md": "src/content/blog/rolling-window-bias-variance-drift/index.md",
    },
]
PLACEHOLDER = "{{CORE_CODE}}"
errors = []


def check(cond, msg):
    if not cond:
        errors.append(msg)
    print(("  OK   " if cond else "  FAIL ") + msg)
    return cond


def cjk_count(text):
    return len(re.findall(r"[\u4e00-\u9fff]", text))


for a in ARTICLES:
    slug, core_rel, md_rel = a["slug"], a["core"], a["md"]
    print(f"\n==== {slug} ====")
    core_path, md_path = ROOT / core_rel, ROOT / md_rel
    img_dir = ROOT / "public" / "images" / slug
    core_text = core_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")

    if PLACEHOLDER in md_text:
        check(md_text.count(PLACEHOLDER) == 1, "exactly one code placeholder")
        md_text = md_text.replace(PLACEHOLDER, core_text.rstrip("\n"))
        md_path.write_text(md_text, encoding="utf-8")
        print(f"  injected {len(core_text.splitlines())} lines of runnable code")
    else:
        print("  code already injected; verifying existing block instead")

    fm = md_text.split("---", 2)[1]
    title = re.search(r'^title:\s*"(.+)"\s*$', fm, re.M).group(1)
    desc = re.search(r'^description:\s*"(.+)"\s*$', fm, re.M).group(1)
    pub = re.search(r"^publishDate:\s*'([^']+)'", fm, re.M).group(1)
    check(len(title) <= 150, f"title length {len(title)} <= 150")
    check(len(desc) <= 600, f"description length {len(desc)} <= 600")
    check('"' not in desc, "description has no inner ASCII double quote")
    check("量化" in fm, "tags include 量化")
    check("difficulty: advanced" in fm, "difficulty advanced")
    check("language: zh" in fm, "language zh")
    check(pub.startswith("2026-11-12T"), f"publishDate is today's ISO datetime ({pub})")

    blocks = re.findall(r"```python\n(.*?)\n```", md_text, re.S)
    check(len(blocks) == 1, "exactly one python code block")
    check(blocks[0].strip() == core_text.strip(), "code block is verbatim the core script")

    with tempfile.TemporaryDirectory(prefix=f"verify_{slug[:12]}_") as td:
        tmp = Path(td)
        (tmp / "article_code.py").write_text(blocks[0], encoding="utf-8")
        env = dict(os.environ, ARTICLE_OUT=str(tmp), MPLBACKEND="Agg")
        r = subprocess.run([sys.executable, str(tmp / "article_code.py")],
                           capture_output=True, text=True, env=env, timeout=900)
        check(r.returncode == 0, "standalone code block exits 0" +
              ("" if r.returncode == 0 else f" :: {r.stderr[-700:]}"))
        check("NaN" not in r.stdout and "Traceback" not in r.stderr,
              "no NaN / Traceback in standalone run")
        gen_json = tmp / "results.json"
        ref_json = img_dir / "results.json"
        if gen_json.exists() and ref_json.exists():
            g = json.loads(gen_json.read_text(encoding="utf-8"))
            ref = json.loads(ref_json.read_text(encoding="utf-8"))
            check(g == ref, "exec-reproduced results.json deep-equals checked-in JSON")
            pngs = sorted(tmp.glob("fig*.png"))
            check(len(pngs) == 5, f"code block generated 5 PNGs (got {len(pngs)})")
            for p in pngs:
                target = img_dir / p.name
                check(target.exists() and
                      hashlib.sha256(p.read_bytes()).hexdigest() ==
                      hashlib.sha256(target.read_bytes()).hexdigest(),
                      f"generated {p.name} byte-identical to site asset")
        else:
            check(False, "results.json exists in both temp and site dirs")

    body = md_text.split("---", 2)[2]
    body = re.sub(r"```python.*?```", "", body, flags=re.S)
    n = cjk_count(body)
    check(2000 <= n <= 3000, f"body CJK chars {n} in [2000, 3000]")

    refs = re.findall(r"!\[[^\]]*\]\(/images/([^)]+)\)", body)
    check(len(refs) == 5, f"exactly 5 image references (got {len(refs)})")
    check(all(r.startswith(slug + "/") for r in refs), "all refs under the article's image dir")
    files = sorted(img_dir.glob("fig*.png"))
    check(len(files) == 5, f"exactly 5 PNG files (got {len(files)})")
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    check(len(set(hashes)) == len(hashes), "all 5 PNGs have distinct hashes")
    check(all((img_dir / r.split("/", 1)[1]).exists() for r in refs), "every referenced PNG exists")
    check((img_dir / "results.json").exists(), "results.json sits with the figures")

    banned = ["占位图", "示意图，非真实", "伪造", "TODO", "省略代码", "此处省略"]
    check(not any(b in body for b in banned), "no placeholder / omitted-code wording")

print("\n" + "=" * 60)
if errors:
    print("FAILURES:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("ALL CHECKS PASSED (2 articles)")
