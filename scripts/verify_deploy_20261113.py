# -*- coding: utf-8 -*-
"""2026-11-13 批次线上部署验证（幂等可重跑）。"""
import json
import subprocess
import time
import urllib.request

BASE = "https://blog.halo26812.eu.org"
SLUGS = ["beta-attenuation-errors-in-variables", "mincer-zarnowitz-vol-forecast-eval"]
TITLES = ["β 测不准，溢价就打折", "连 oracle 都过不了 MZ 检验"]

NC = ["-H", "Cache-Control: no-cache", "-H", "Pragma: no-cache"]


def curl_headers(url):
    r = subprocess.run(
        ["curl", "-s", "-D", "-", "-o", "/dev/null", "-L"] + NC + [url],
        capture_output=True, text=True, timeout=60)
    hdrs = {}
    for line in r.stdout.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            hdrs[k.strip().lower()] = v.strip()
    return hdrs


def fetch(url):
    r = subprocess.run(["curl", "-s", "-L", "--max-time", "60", url],
                       capture_output=True, timeout=90)
    return r.stdout


def check_once(verbose=True):
    # 任务指定的 200 检查
    code = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         f"{BASE}/quant-column"],
        capture_output=True, text=True, timeout=60).stdout
    h = curl_headers(f"{BASE}/quant-column")
    body = fetch(f"{BASE}/quant-column")
    try:
        text = body.decode("utf-8", "ignore")
    except Exception:
        text = ""
    titles_hit = [t for t in TITLES if t in text]
    arts = {}
    for s in SLUGS:
        c = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-L",
             f"{BASE}/blog/{s}/"], capture_output=True, text=True,
            timeout=60).stdout
        arts[s] = c
    imgs = {}
    for s in SLUGS:
        c = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             f"{BASE}/images/{s}/results.json"],
            capture_output=True, text=True, timeout=60).stdout
        imgs[s] = c
    ok = (code == "200" and len(titles_hit) == 2
          and all(v == "200" for v in arts.values())
          and all(v == "200" for v in imgs.values()))
    if verbose:
        print(json.dumps({
            "quant_column_code": code,
            "cache": h.get("x-vercel-cache"),
            "last_modified": h.get("last-modified"),
            "titles_hit": titles_hit,
            "articles": arts,
            "result_json": imgs,
            "ok": ok,
        }, ensure_ascii=False))
    return ok, h.get("last-modified")


def main():
    passed_rounds = 0
    lm_seen = None
    for i in range(14):
        ok, lm = check_once()
        print(f"--- round {i+1} ok={ok} lm={lm}", flush=True)
        if ok and lm:
            if lm == lm_seen:
                passed_rounds += 1
            else:
                lm_seen = lm
                passed_rounds = 1
        else:
            passed_rounds = 0
            lm_seen = None
        if passed_rounds >= 2:
            print("VERIFICATION PASSED (2 consecutive rounds)")
            return 0
        time.sleep(75)
    print("VERIFICATION FAILED after 14 rounds")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
