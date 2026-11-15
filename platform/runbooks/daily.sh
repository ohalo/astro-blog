#!/bin/zsh
# 每日 17:10（launchd）执行：数据湖增量入库 -> K线发布 -> git 提交触发 Vercel 重新部署。
# 手动跑一次验证：zsh platform/runbooks/daily.sh
set -e
BASE="/Users/halo123456/workspace/astro-blog"
LOG_DIR="$BASE/platform/runs"
mkdir -p "$LOG_DIR"

echo "===== $(date '+%F %T') daily 开始 ====="
cd "$BASE/platform"
./.venv/bin/python -m cli.qplat daily

cd "$BASE"
git add public/data
if git diff --cached --quiet; then
  echo "[daily] 数据无变化，不提交"
else
  git commit -m "data: 每日K线发布 $(date +%F)"
  git push || echo "[daily] push 失败（离线/冲突？），数据已本地发布，明日重试"
fi
echo "===== $(date '+%F %T') daily 完成 ====="
