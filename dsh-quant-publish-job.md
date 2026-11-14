# dsh 定时任务定义：量化博客每批 2 篇 + 发布到 Vercel

> 这是把 WorkBuddy 自动化 `quant-blog-2-posts-simplified`（`6a2858b0`，原调度 `FREQ=HOURLY;INTERVAL=6`）迁移到 dsh 的落地包。
> 本文件 + `scripts/dsh_entrypoint.sh` + `scripts/quant_verify.py` + `scripts/quant_deploy_check.py` 即为完整迁移产物。
> 配套详细规格见 `quant-publish-runbook.md`（内容生产 + 机械发布全约定）。

---

## 1. 任务拆分（迁移成败关键）

这个定时任务**不能整体脚本化**，必须拆成两半：

| 半 | 内容 | 能否纯脚本 | dsh 上怎么跑 |
|---|---|---|---|
| **A. 内容生产** | 设计 2 个不重复主题 + 真实数值实验 + 现场算数据 + 生成 5 张真实 PNG + 写 2000–3000 字中文正文 | **不能**，必须 LLM/人工 | 见 §4 LLM 契约；dsh 上需有可调用的 LLM（claude CLI / API / 自建 agent），否则这半人工跑、只把 B 半交给 dsh |
| **B. 机械发布** | 校验 → 本地 build → git commit/push → Vercel 部署轮询 | **能**，纯脚本 | `scripts/dsh_entrypoint.sh`（本包已提供） |

**结论**：dsh 要「全自动」就必须侧也有 LLM。否则 dsh 只跑 B 半（半自动），A 半由人工/WorkBuddy 产出后 dsh 负责发布。

---

## 2. dsh 任务定义（可直接导入）

### 2.1 cron（复刻原调度；建议改为每日一次，见注）

原调度是「每 6 小时」，但主题池已耗尽、每批需人工/LLM 设计主题，**每 6 小时生成 2 篇会把池子瞬间榨干且不现实**。建议 dsh 上改为：

```
# 推荐：每天 07:00 北京时间（=23:00 UTC 前一天）跑一批
0 23 * * *  /opt/astro-blog/scripts/dsh_entrypoint.sh >> /opt/astro-blog/scripts/.dsh_logs/cron.log 2>&1
```

若坚持复刻原每 6 小时：

```
0 */6 * * *  /opt/astro-blog/scripts/dsh_entrypoint.sh >> /opt/astro-blog/scripts/.dsh_logs/cron.log 2>&1
```

> dsh 若用自有调度格式（Web/API/JSON），把上面的 `command` 与 `schedule(cron)` 填进对应字段即可；本脚本对调度系统无特殊依赖。

### 2.2 任务元数据

| 字段 | 值 |
|---|---|
| 名称 | `quant-blog-2-posts-publish` |
| 命令 | `/opt/astro-blog/scripts/dsh_entrypoint.sh` |
| 工作目录 | `/opt/astro-blog`（dsh 执行机上 clone 的仓库根） |
| 超时 | 建议 ≥ 600s（build ~60–90s + 部署轮询最多 ~6 分钟） |
| 环境变量 | `REPO_DIR`（默认 `/opt/astro-blog`）、`VENV_PY`（默认 `~/.workbuddy/binaries/python/envs/default/bin/python`，用于 PyYAML 校验） |
| 重试 | 失败不自动重试（避免重复 push）；靠下次调度或人工查看 `.dsh_logs/` |
| 告警 | 脚本非零退出即失败，接 dsh 失败通知 |

### 2.3 执行机前置依赖（一次性装好）

```bash
# 1) clone 仓库（SSH，配好 deploy key 免密 push）
git clone git@github.com:ohalo/astro-blog.git /opt/astro-blog
cd /opt/astro-blog && npm install

# 2) 系统 python3 需有：numpy scipy matplotlib pandas（无 sklearn，EM/优化手写）
python3 -c "import numpy,scipy,matplotlib,pandas; print('ok')"

# 3) 用于 frontmatter 校验的 venv python 需有 PyYAML
~/.workbuddy/binaries/python/envs/default/bin/python -c "import yaml; print('yaml ok')"

# 4) matplotlib 中文字体（否则图片中文缺字形）：安装 Heiti TC / Noto Sans CJK 并显式指定
#    macOS: 系统自带 Heiti TC；Linux: apt install fonts-noto-cjk
```

---

## 3. 机械发布入口做了什么（`dsh_entrypoint.sh`）

1. 读取内容生产环节写好的 `scripts/.dsh_pending.json`（含 2 个 slug + 各自 core 脚本路径）。无此文件则 **SKIP 正常退出**（内容环节未产出）。
2. **slug 双查重**：`os.listdir` + `git ls-files` 确认不与任何已跟踪目录冲突，**绝不 rm -rf 旧图目录、绝不覆盖旧文**。
3. 跑 `quant_verify.py` 全套硬校验（见 §5），任一不过则非零退出，**绝不发布**。
4. 本地 `npm run build`（先 `mv .vercel/output` 挪开避 safe-delete 拦截）。
5. `git add -A` → `commit` → `git pull --rebase` → `push`（**先 commit 再 pull** 防冲突报错）。
6. `quant_deploy_check.py` 轮询 Vercel（处理 CDN 缓存 / 308 / 3–6 分钟延迟）。
7. 成功后将 `.dsh_pending.json` 改名落档为 `.dsh_done_<ts>.json`，写执行日志。

**幂等性**：无 pending 则跳过；校验失败不 push；无 staged 改动不 commit。

---

## 4. 内容生产环节（LLM 契约，A 半）

dsh 上若跑全自动，需一个**前置 LLM 步骤**（在 `dsh_entrypoint.sh` 之前触发），输出符合下契约的 2 篇文章 + 写 `scripts/.dsh_pending.json`。

### 4.1 触发方式（示例，按你们 LLM 接入调整）

```bash
# 例：用 claude CLI 在内容生产步骤产出（prompt 见 4.2）
claude -p "$(cat /opt/astro-blog/scripts/dsh_gen_prompt.txt)" \
  && /opt/astro-blog/scripts/dsh_write_pending.sh   # 抽取产物路径写 .dsh_pending.json
```

> 注意：让 LLM 一次写出「含可运行代码 + 真实计算 + 5 张图 + 2000–3000 字」的 2 篇文章风险很高，历史上多次出现假图/崩代码/伪造数字。生产环境强烈建议 **LLM 产出 core 实验脚本 → 本地真实执行 → 再拼装正文**，并把数值写进 `results.json` 与正文逐项对账。本仓库既有做法是 `article_core_*.py`（代码块本体）+ `build_verify_*.py`（注入+校验）。

### 4.2 内容硬约束（LLM 必须遵守，否则 `quant_verify.py` 不通过）

- **主题**：2 个不重复 slug，先 `grep`/查 `src/content/blog` 现有 1520+ 目录避免撞名（本站覆盖极广）；选互补进阶角度并交叉引用旧文。
- **frontmatter**：`publishDate`（`z.coerce.date()` 能解析，同日置顶用 ISO datetime 如 `2026-11-14T07:10:00Z`）+ `language: zh` + `tags` 含「量化」+ `difficulty: advanced`（**决定能否进线上 quant-column 专栏**，组件按 tag+difficulty 过滤，**不读 `quant-column.md`**）。
- **description** ≤ 600 字符，内部不得含 ASCII 直引号 `"`（会提前闭合 YAML 双引号标量）；用「」交替。
- **正文** 2000–3000 中文字（不含代码块），5 张**真实计算结果** PNG 置于 `public/images/{slug}/`，结果 JSON 同目录，正文数字必须与 JSON 一致。
- **代码块必须自包含可运行**（含数据生成 + `__main__`），由 `build_verify` 抽出 exec 与 `results.json` 深度相等比对。
- **禁止**：占位图、乘系数伪造曲线、把估计误差冒充预测误差、`try/except` 回退真值冒充拟合、捏造学术出处。
- **诚实口径**：对照实验同源优化器 / 共同随机数 / MSE 归一化 / 学术出处核对（IB=Tishby,Pereira&Bialek 1999；PC=Spirtes&Glymour 1990 前后）。

### 4.3 `.dsh_pending.json` 格式

```json
{
  "date": "2026-11-14",
  "articles": [
    {"slug": "topic-a-slug", "core": "scripts/article_core_a_20261114.py"},
    {"slug": "topic-b-slug", "core": "scripts/article_core_b_20261114.py"}
  ]
}
```

---

## 5. `quant_verify.py` 校验清单（发布前硬闸）

对每个 slug + core 脚本：
- frontmatter 字段完整、`description`≤600 且无 ASCII 直引号；`量化` tag + `difficulty: advanced` 存在。
- 正文 2000–3000 中文字；禁用词（占位/伪造类）零残留。
- `public/images/{slug}/` 恰好 5 张 PNG + `results.json`，且 5 图 hash 各不同（非复制）。
- 从 md 抽取代码块实际 `exec`，输出与 `results.json` 逐项深度相等。
- slug 不与现有目录重名。
- 本地 `npm run build` 通过（0 schema 错误）。

---

## 6. 部署轮询要点（`quant_deploy_check.py`）

- Vercel 推送后 **3–6 分钟**才全量生效，期间 `HIT` 旧构建、文章页 404 属正常。
- 验证 `quant-column` 用 `curl -s -o /dev/null -w "%{http_code}"` 返回 200 即成功（任务指定口径）。
- **CDN 缓存顽固**：`?cb=<ts>` 可能被忽略（`x-vercel-cache: HIT`、age 上千秒仍旧版）→ 用 `-H 'Cache-Control: no-cache' -H 'Pragma: no-cache'` 强制回源看 `MISS` + 新 `last-modified`。
- **连续两轮判定**：首轮回源 `MISS` 后 CDN 缓存新版本，后续 `HIT` 属正常，**只需「新 last-modified + 标题/文章页/资产全命中」**，不必强求两轮都 MISS。
- 文章页 308 是尾斜杠重定向，`curl` 加 `-L`。

---

## 7. 迁移步骤（checklist）

1. dsh 执行机装依赖（§2.3）：clone + npm install + python 包 + CJK 字体 + github deploy key。
2. 把本包 4 个文件放到 `/opt/astro-blog/scripts/`（及本 md 到仓库根）。
3. 若做全自动：接好 LLM 内容步骤（§4），确认能产出 `.dsh_pending.json`。
4. **手动跑一次 `dsh_entrypoint.sh`**（先用测试 pending）验证端到端通过。
5. 在 dsh 按 §2.1 配 cron（建议每日，非每 6 小时）。
6. **确认连续 2 次成功发布后**，再到 WorkBuddy 设置里把自动化 `6a2858b0` 停用（先观察一周，不要立刻删）。
7. 保留 `quant-publish-runbook.md` 作为完整规格备查。

---

## 8. 已知坑速查（排错表）

| 现象 | 原因 | 处置 |
|---|---|---|
| 文章页 curl 308 | 尾斜杠重定向 | 加 `-L` |
| 推送后几分钟仍旧版/404 | Vercel 3–6 分钟生效 + CDN 缓存 | 耐心轮询，用 no-cache 强制回源 |
| `?cb=` 无效仍旧版 | 边缘忽略 cache-buster | 改用 `Cache-Control: no-cache` 头 |
| build 报 safe-delete 拦截 | `.vercel/output` 旧产物 count>50 | 先 `mv .vercel/output` 挪开 |
| `np.trapz` 报错 | numpy 2.x 已删 | 用 `np.trapezoid`（写 fallback） |
| frontmatter `bad indentation` | description 内 ASCII 直引号闭合 YAML | 内部改用「」 |
| 专栏页不显示新文 | 缺 `量化` tag 或 `difficulty`≠advanced | 补齐 frontmatter（组件不读 quant-column.md） |
| 覆盖旧文 | slug 撞名 + rm -rf 图目录 | `os.listdir`+`git ls-files` 双查重，绝不整目录 rm |
| select_blog_topics.py 崩 | 主池+兜底池耗尽，assert 0 | 内容环节手动设计主题（A 半负责） |
