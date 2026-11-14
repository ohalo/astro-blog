#!/usr/bin/env bash
# =============================================================================
# dsh_entrypoint.sh - quant-blog "mechanical publish" pipeline (called by dsh cron)
#
# What it does (deterministic, idempotent, safe - mechanical half only):
#   1. read scripts/.dsh_pending.json written by the content-production step
#   2. for each slug run quant_verify.py (full hard checks)
#   3. local npm build (.vercel/output moved aside to dodge safe-delete block)
#   4. git add -> commit -> git pull --rebase -> push
#   5. quant_deploy_check.py polls Vercel (CDN cache / 308 / 3-6 min delay)
#   6. on success rename .dsh_pending.json -> .dsh_done_<ts>.json
#
# Prereqs on the dsh runner:
#   - git SSH key for git@github.com:ohalo/astro-blog.git (passwordless push)
#   - node v22 / npm 10 (already in repo)
#   - system python3: numpy scipy matplotlib pandas (no sklearn)
#   - venv python has PyYAML (frontmatter validation)
#   - matplotlib CJK font (Heiti TC etc.) or Chinese glyphs missing in images
#
# The content-production half (topic design + real computations + 5 PNGs +
# 2000-3000 Chinese chars) MUST be done by an LLM/human BEFORE this script,
# and must write scripts/.dsh_pending.json:
#   {"date":"2026-11-14","articles":[
#     {"slug":"some-slug-a","core":"scripts/article_core_a.py"},
#     {"slug":"some-slug-b","core":"scripts/article_core_b.py"}]}
# =============================================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/astro-blog}"
PENDING="${REPO_DIR}/scripts/.dsh_pending.json"
DONE_DIR="${REPO_DIR}/scripts"
VENV_PY="${VENV_PY:-$HOME/.workbuddy/binaries/python/envs/default/bin/python}"

cd "$REPO_DIR"

TS="$(date +%Y%m%d_%H%M%S)"

# Log goes to stdout/stderr; the dsh cron line already appends to cron.log
# via ">> cron.log 2>&1". Do NOT use "exec > >(tee ...)" here - on some
# runners tee is unavailable and the whole script dies before doing anything.
echo "===== dsh quant-publish @ ${TS} ====="
echo "REPO_DIR=${REPO_DIR}"

# ---- 0. precheck -----------------------------------------------------------
if [ ! -f "$PENDING" ]; then
  echo "[SKIP] no $PENDING found - nothing to publish (content step produced no output). Exiting."
  exit 0
fi

# parse pending (use venv python so PyYAML is available)
mapfile -t SLUGS < <("$VENV_PY" - <<'PY'
import json
d = json.load(open("scripts/.dsh_pending.json"))
for a in d["articles"]:
    print(a["slug"])
PY
)
mapfile -t CORES < <("$VENV_PY" - <<'PY'
import json
d = json.load(open("scripts/.dsh_pending.json"))
for a in d["articles"]:
    print(a["core"])
PY
)

if [ "${#SLUGS[@]}" -ne 2 ]; then
  echo "[FATAL] pending article count = ${#SLUGS[@]}, hard rule requires exactly 2. Abort."
  exit 1
fi

echo "Targets: ${SLUGS[0]} (${CORES[0]}) | ${SLUGS[1]} (${CORES[1]})"

# ---- 1. slug dedup (double safety, never overwrite old posts) --------------
echo "---- slug dedup ----"
for s in "${SLUGS[@]}"; do
  if [ -d "src/content/blog/$s" ] && git ls-files --error-unmatch "src/content/blog/$s/index.md" >/dev/null 2>&1; then
    echo "[FATAL] slug '$s' already tracked by git (old post) - refuse to overwrite. Pick another slug."
    exit 1
  fi
  if [ -d "public/images/$s" ] && git ls-files --error-unmatch "public/images/$s" >/dev/null 2>&1; then
    echo "[FATAL] image dir '$s' already tracked by git (old post) - refuse rm -rf. Pick another slug."
    exit 1
  fi
  echo "  OK: $s does not collide with any tracked dir"
done

# ---- 2. verify (any failure -> nonzero exit, never publish) -----------------
echo "---- quant_verify ----"
python3 scripts/quant_verify.py "${SLUGS[0]}" "${CORES[0]}" "${SLUGS[1]}" "${CORES[1]}"
echo "verify exit=$?"

# ---- 3. local build (move .vercel/output aside first) ----------------------
echo "---- npm build ----"
if [ -d .vercel/output ]; then mv .vercel/output "/tmp/vercel_output_bak_${TS}"; fi
npm run build
echo "build exit=$?"

# ---- 4. git commit + rebase pull + push ------------------------------------
echo "---- git ----"
git add -A
if git diff --cached --quiet; then
  echo "[FATAL] no staged changes - content may not have been written. Abort."
  exit 1
fi
git commit -m "feat: add 2 quant articles $(date +'%Y-%m-%d')"
git pull --rebase
git push
echo "push exit=$?"

# ---- 5. deploy poll --------------------------------------------------------
echo "---- deploy check ----"
python3 scripts/quant_deploy_check.py "${SLUGS[0]}" "${SLUGS[1]}"
echo "deploy check exit=$?"

# ---- 6. archive ------------------------------------------------------------
mv "$PENDING" "${DONE_DIR}/.dsh_done_${TS}.json"
echo "===== DONE @ $(date +%Y-%m-%dT%H:%M:%S) ====="
echo "Published: ${SLUGS[0]} / ${SLUGS[1]}"
