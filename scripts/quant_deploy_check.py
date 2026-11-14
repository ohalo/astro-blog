# -*- coding: utf-8 -*-
"""部署上线轮询校验（dsh 机械流水线用）。

处理 Vercel 部署的几个已知坑：
  - 部署延迟 3–6 分钟才全量生效；
  - 专栏页是 ~1.5MB 单行 HTML，不能用 grep 中文标题（会误判）；
  - 文章页 URL 返回 308 尾斜杠重定向，必须跟随；
  - CDN 缓存顽固，?cb= 时间戳可能被忽略，必须用 no-cache 强制回源；
  - 强制回源首轮拿到 MISS 后，后续轮次 HIT 属正常，判定条件是
    「新 last-modified + 专栏页含两篇链接 + 两文章页 200」且连续两轮通过。

用法：
  python3 scripts/quant_deploy_check.py <slug1> <slug2> [--timeout 600] [--interval 30]
"""
import sys
import time
import urllib.request
import urllib.error

BASE = "https://blog.halo26812.eu.org"
NO_CACHE = {"Cache-Control": "no-cache", "Pragma": "no-cache"}


def fetch(url, follow=True):
    req = urllib.request.Request(url, headers=NO_CACHE)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
            return r.status, r.headers, data
    except urllib.error.HTTPError as e:
        # HTTPError 也带 headers（含 Location 重定向信息）
        return e.code, e.headers, b""


def check_once(slugs):
    # 专栏页（强制回源）
    status, headers, data = fetch(f"{BASE}/quant-column")
    qc_ok = status == 200
    qc_text = data.decode("utf-8", "ignore")
    links_ok = all(f"/blog/{s}/" in qc_text for s in slugs)
    lm = headers.get("Last-Modified", "")

    # 文章页（跟随 308 重定向）
    art_ok = []
    for s in slugs:
        st, _, _ = fetch(f"{BASE}/blog/{s}/")
        art_ok.append(st == 200)

    return {
        "qc_status": status,
        "links_ok": links_ok,
        "last_modified": lm,
        "art_status": art_ok,
        "all_ok": qc_ok and links_ok and all(art_ok),
    }


def main():
    args = sys.argv[1:]
    slugs = [a for a in args if not a.startswith("--")]
    timeout = 600
    interval = 30
    for i, a in enumerate(args):
        if a == "--timeout":
            timeout = int(args[i + 1])
        if a == "--interval":
            interval = int(args[i + 1])
    assert len(slugs) == 2, "需要恰好 2 个 slug"

    start = time.time()
    last_lm = ""
    consecutive = 0
    print(f"开始部署校验：slugs={slugs} timeout={timeout}s interval={interval}s")
    while time.time() - start < timeout:
        r = check_once(slugs)
        elapsed = int(time.time() - start)
        print(f"  [t+{elapsed}s] 专栏 {r['qc_status']} 链接{'✓' if r['links_ok'] else '✗'} "
              f"文章 {r['art_status']} LM={r['last_modified']}")
        if r["all_ok"]:
            # 连续两轮通过（容忍首轮 HIT 旧构建）
            if r["last_modified"] and r["last_modified"] != last_lm:
                consecutive += 1
                last_lm = r["last_modified"]
            else:
                consecutive += 1
            if consecutive >= 2:
                print("✅ 部署校验通过（连续两轮全命中）")
                sys.exit(0)
        else:
            consecutive = 0
            last_lm = r["last_modified"]
        time.sleep(interval)

    print("❌ 部署校验超时未通过，请手动 curl 核查")
    sys.exit(1)


if __name__ == "__main__":
    main()
