#!/usr/bin/env bash
# =============================================================================
# 量化博客机械发布流水线（dsh 定时任务入口）
#
# 前提：两篇文章的内容已生产完毕（core 实验脚本 + index.md + 5 张图 + results.json），
#       即「内容生产」半已由 LLM/人工完成。本脚本只负责 校验 → 构建 → 提交推送 → 部署验证。
#
# 用法（slug 与 core 成对传入）：
#   ./scripts/quant_publish.sh <slug1> <core1.py> <slug2> <core2.py> [YYYY-MM-DD]
#
# 例：
#   ./scripts/quant_publish.sh \
#     beta-attenuation-errors-in-variables scripts/article_core_beta_eiv_20261113.py \
#     mincer-zarnowitz-vol-forecast-eval scripts/article_core_mz_qlike_20261113.py
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$(pwd)
DATE=${5:-$(date +%Y-%m-%d)}

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "用法: $0 <slug1> <core1.py> <slug2> <core2.py> [YYYY-MM-DD]" >&2
  exit 2
fi

SLUG1=$1; CORE1=$2; SLUG2=$3; CORE2=$4
echo "==> 发布批次 $DATE : $SLUG1 / $SLUG2"

# ---------------------------------------------------------------------------
# 1) 全套硬校验（含代码块注入、exec 复跑与 results.json 深度相等）
# ---------------------------------------------------------------------------
echo "==> [1/4] quant_verify"
python3 scripts/quant_verify.py "$SLUG1" "$CORE1" "$SLUG2" "$CORE2"

# ---------------------------------------------------------------------------
# 2) 本地 build（先挪开 .vercel/output 避免 Astro safe-delete 拦截；产物在 dist/client/）
# ---------------------------------------------------------------------------
echo "==> [2/4] npm run build"
if [ -d .vercel/output ]; then
  mv .vercel/output "/tmp/vercel_output_bak_$(date +%s)"
  echo "    已挪开旧 .vercel/output"
fi
npm run build

# ---------------------------------------------------------------------------
# 3) git 工作流：先 commit，再 pull --rebase（解冲突），再 push
#    —— 注意：本流水线【不】修改 src/content/pages/quant-column.md（该文件是死内容，
#       线上专栏页由 frontmatter 的 量化 tag + difficulty: advanced 渲染）。
#        若坚持维护它，pull --rebase 时它极易冲突，需手动合并两个日期分组。
# ---------------------------------------------------------------------------
echo "==> [3/4] git commit / pull --rebase / push"
git add -A
git commit -m "feat: add 2 quant articles $DATE"
git pull --rebase
git push

# ---------------------------------------------------------------------------
# 4) 部署验证（Vercel 通常 3–6 分钟生效；轮询到连续两轮全命中）
# ---------------------------------------------------------------------------
echo "==> [4/4] 部署校验"
python3 scripts/quant_deploy_check.py "$SLUG1" "$SLUG2" --timeout 600 --interval 30

echo "✅ 发布完成：$BASE/quant-column"
