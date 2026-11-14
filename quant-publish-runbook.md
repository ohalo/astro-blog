# 量化博客自动发布 · 定时任务 Runbook（可迁移到 dsh）

> 本文件是「每期自动生产并发布 2 篇量化交易博客到 Vercel」这一定时任务的**完整规格与操作手册**。
> 原任务跑在 WorkBuddy 自动化（ID `6a2858b0`）上；本 runbook 把它整理成可自洽执行的文档，
> 以便迁移到 dsh（你自己的调度机 / cron）上运行，卸掉 WorkBuddy 自动化的调度压力。
>
> 最后整理：2026-11-14。仓库根目录：`~/workspace/astro-blog`。

---

## 0. 这个任务到底是什么

- **目标**：每期产出并发布 **2 篇**「量化交易」长文到站点 `https://blog.halo26812.eu.org`。
- **产物**：
  1. 2 个文章目录 `src/content/blog/{slug}/index.md`（含 frontmatter + 中文正文 + 可运行代码块）。
  2. 每篇 **5 张真实计算 PNG** 于 `public/images/{slug}/`，同目录一份 `results.json`（全部统计量的唯一真值源）。
  3. 线上 `/quant-column` 专栏页自动出现这两篇（依赖 frontmatter，不依赖 md 文件）。
- **触发**：原在 WorkBuddy 按固定频率触发；迁移后由 **dsh 的 cron / 调度器**触发。

---

## 1. 任务的两半（最关键的一节）

这个任务**不能**被一个纯 shell 脚本完全自动化，因为它包含不可脚本化的「内容生产」。务必分清：

| 半 | 内容 | 能否脚本化 | 谁来做 |
|---|---|---|---|
| **A. 内容生产** | 选题 → 设计真实数值实验 → 写 2000–3000 字中文长文 + 可运行 core 脚本 | ❌ 不能 | LLM agent（原 WorkBuddy 小编）或人工 |
| **B. 机械发布** | 构建校验 → 本地 build → git 提交推送 → 部署验证 | ✅ 能 | 纯脚本（`quant_verify.py` + `quant_publish.sh`） |

**结论（决定 dsh 怎么搭）：**

- 如果你要 dsh 上**全自动**：dsh 侧也必须有一个可调用的 LLM（Claude CLI / API / 自建 agent）来承担 A 半。本 runbook 第 4.2 节就是给那个 agent 的精确指令。
- 如果你只想要 dsh 跑**半自动**：A 半在别处（WorkBuddy 或人工）产出，dsh 只跑 B 半的发布。此时 dsh 的 cron 直接调 `scripts/quant_publish.sh`。
- **无论哪种**，B 半的脚本（`quant_verify.py` / `quant_publish.sh` / `quant_deploy_check.py`）都已就位、可直接复用。

---

## 2. 环境准备（dsh 一次性）

| 项 | 要求 | 备注 |
|---|---|---|
| Python | 系统 `python3`，需有 `numpy` / `scipy` / `matplotlib` / `pandas` | **不需要** `sklearn`——EM、优化、分解都手写。 |
| PyYAML | 仅在 venv `~/.workbuddy/binaries/python/envs/default/bin/python` 内 | 仅当用 venv 跑 frontmatter 校验时才需要；`quant_verify.py` 用正则解析 frontmatter，不依赖 PyYAML。 |
| Node / npm | 能跑 `npm run build`（Astro 站点构建） | 首次需 `npm install`。 |
| 中文字体 | matplotlib 必须显式设 `Heiti TC` / `PingFang SC` / `Arial Unicode MS` | 默认 DejaVu 下 CJK 全变缺字形方块。core 脚本顶部已设 `plt.rcParams["font.sans-serif"]`。 |
| 仓库 | `git clone` 本仓库并配好能 push 到 GitHub 的凭证 | SSH key 或 token。 |
| 网络 | 能访问 `github.com`（push）与 `blog.halo26812.eu.org`（验证） | Vercel 从 GitHub 自动部署，无需直连 Vercel。 |

> ⚠️ 若 dsh 是 Linux 无 macOS 字体，`Heiti TC` 不存在会回退到下一候选；请确认 dsh 上装有任一中文字体并在 core 脚本字体列表里补上（如 `Noto Sans CJK SC`）。

---

## 3. 硬约束（红线，违反即 build 失败或内容作废）

1. **每期必须 2 篇**；`slug` 不得与 `src/content/blog/` 下任何现有目录重名（见第 8 节查重）。
2. **配图必须来自真实计算**，禁止占位图 / 伪造曲线 / 「乘系数」改图。
3. **正文数字必须来自真实计算**，且与 `results.json` 逐项一致；禁止把估计误差冒充预测误差、禁止 `try/except` 吞掉拟合报错后回退真值冒充结果。
4. **frontmatter 字段名是 `publishDate`**（不是 `date`）+ `language: zh`；`description` ≤ 600 字符且**内部不得含 ASCII 直引号 `"`**（会提前闭合 YAML 双引号标量导致 build 失败）。
5. 文章要进线上专栏并置顶：frontmatter 必须带 **`量化` 类 tag + `difficulty: advanced`**，且 `publishDate` 最新。
6. **正文中文字数（不含代码块）2000–3000**；正文必须真的引用 5 张图（否则图是孤儿）。
7. frontmatter `image` 字段必须指向真实存在的文件；代码块必须自包含可运行（含数据生成 + `__main__`），且能从 md 抽出 **exec 复跑** 与 `results.json` 深度相等。

---

## 4. 完整分步流程

### 4.1 主题选择（脚本辅助，人工/agent 拍板）

```bash
cd ~/workspace/astro-blog
python3 scripts/select_blog_topics.py
```

- 输出格式：`主题标题|slug`（每行一个），末尾 `===` 分隔。
- 脚本内部有**多级兜底池**（TOPIC_POOL → fallback_pool → fresh_pool），运行时扫描磁盘排除已发布 slug，理论上总能吐出 2 个未占用 slug。
- **现实**：截至 2026-11，全库已 >1525 篇，主池 + 旧兜底基本耗尽，脚本最终退到 `fresh_pool` 末尾。如果想保证质量与互补性，**建议 agent 直接指定 2 个互补进阶角度的 slug**（参考第 8 节策略），而不是盲信脚本输出。
- **查重铁律**：用 Python `os.listdir('src/content/blog')` + `git ls-files src/content/blog/<slug>` 双重确认；**绝不用** `subprocess.run(['ls',...], shell=True)`（曾误判已存在 slug 为未占用，覆盖旧文）。

### 4.2 内容生产（LLM/人工，A 半）

对每篇，agent 要产出的交付物：

**(a) core 实验脚本** `scripts/article_core_{slug}_{YYYYMMDD}.py`，约定：

```python
# -*- coding: utf-8 -*-
"""一句话说明本文所有数字的唯一来源。运行会：1) 现场生成模拟数据（固定种子）；
   2) 画 5 张图到 public/images/{slug}/；3) 写同目录 results.json。"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261114
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(os.path.dirname(_HERE), "public", "images", "{slug}")  # ← 改成自己的 slug
os.makedirs(_OUT, exist_ok=True)

def _save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, name), dpi=150); plt.close(fig)

# ... 5 个实验函数，各自返回「可 JSON 化的统计量字典」...

def main():
    results = {"exp1": exp1(), "exp2": exp2(), ...}   # 5 项
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))  # ← 最后打印，供 exec 复跑比对

if __name__ == "__main__":
    main()
```

要点：固定随机种子（可复现）；5 张图文件名与正文 `![...](/images/{slug}/xxx.png)` 一一对应；**最后 `print` 的必须是完整 JSON**（校验脚本从第一个 `{` 解析）；`results.json` 与打印内容必须同源。

**(b) 文章** `src/content/blog/{slug}/index.md`：

```markdown
---
title: "中文标题"
description: "≤600字符，无ASCII直引号，概括核心结论"
publishDate: '2026-11-14T07:10:00Z'
tags:
  - 量化
  - 交易
  - 你的一级领域
  - Python
language: zh
difficulty: advanced
---

## 一、问题：……
（2000–3000 中文字，含 5 张图引用、公式、可运行代码块）

![图1 说明](/images/{slug}/fig1.png)
…

<!--CODE-->
（或手写一个 ```python 代码块，build 时会与 core 逐字对齐）
```

- 同日多批次要置顶：把 `publishDate` 写成 **ISO datetime**（如 `'2026-11-14T09:20:00Z'`），比当天纯日期批次晚即可排到 advanced 区最前。schema 是 `z.coerce.date()`，能正常解析。
- frontmatter 结尾的 `---` 别漏（曾两次犯）。

**(c) 诚实结论口径**（血泪红线，来自历次返工）：
- 对照实验**必须同源优化器**；「训练/留出误差背离」是过拟合直接证据，要写进正文。
- **共同随机数**：对照两组必须共享初始化与 minibatch 序列，否则差异被噪声污染。
- 扫描中若信号强度随参数变化，**必须用 MSE/Var(真值) 归一化**，否则曲线上升只是尺度变化。
- 学术出处必须核对（如 IB=Tishby,Pereira&Bialek 1999；PC=Spirtes&Glymour）；偏相关用精度矩阵 + Fisher-z，不用 `|r|<t_α`。
- 禁止粉饰：方法失效/落后/假设错配要如实写，元数据里记录「诚实边界」。

### 4.3 构建与校验（脚本，B 半第 1 步）

```bash
python3 scripts/quant_verify.py <slug1> <core1.py> <slug2> <core2.py>
```

它会：注入代码块 → 校验 frontmatter/字数/5 图/禁用词 → **exec 复跑 core 比对 `results.json` 深度相等**。全部 PASS（退出码 0）才继续。

### 4.4 本地 build + git 工作流（脚本，B 半第 2–3 步）

```bash
# 本地 build（先挪开 .vercel/output 避 Astro safe-delete 拦截；产物在 dist/client/）
[ -d .vercel/output ] && mv .vercel/output /tmp/vercel_output_bak_$(date +%s)
npm run build

# git：先 commit → pull --rebase（无新提交则快进）→ push
git add -A
git commit -m "feat: add 2 quant articles $(date +%Y-%m-%d)"
git pull --rebase
git push
```

> 本流水线**不**改 `src/content/pages/quant-column.md`——该文件是**死内容**，线上专栏页由 `src/components/pages/QuantColumnContent.astro` 按 frontmatter 的 `量化` tag + `difficulty: advanced` 渲染。若坚持维护它（仅历史一致性），`pull --rebase` 时它极易冲突：解法是保留双方日期分组、按日期降序、删冲突标记后 `git add` + `GIT_EDITOR=true git rebase --continue`。

### 4.5 部署验证（脚本，B 半第 4 步）

```bash
python3 scripts/quant_deploy_check.py <slug1> <slug2> --timeout 600 --interval 30
```

Vercel 通常 3–6 分钟生效。脚本轮询到「专栏页 200 + 含两篇 `/blog/{slug}/` 链接 + 两文章页 200」且连续两轮通过才退出 0。

> 文章页 URL 返回 **308** 尾斜杠重定向，`quant_deploy_check.py` 已跟随；普通 `curl` 验证须加 `-L`。
> CDN 缓存顽固：`?cb=<ts>` 时间戳可能被忽略，脚本已用 `Cache-Control: no-cache` + `Pragma: no-cache` 强制回源。首轮回源 `MISS` 后后续 `HIT` 属正常，**判定条件是新 `Last-Modified` + 链接/页面全命中**，不要求两轮都 `MISS`。

---

## 5. 机械流水线一键入口（dsh 用）

`scripts/quant_publish.sh` 把 4.3–4.5 串成一条命令（内容已生产的前提下）：

```bash
./scripts/quant_publish.sh \
  <slug1> <core1.py> <slug2> <core2.py> [YYYY-MM-DD]
```

它依次执行：校验 → npm build → git commit/pull--rebase/push → 部署轮询。任一环节失败即中止（非零退出），cron 可据此告警。

---

## 6. dsh 定时配置（cron 示例）

内容全自动（dsh 侧有 LLM agent 先产出内容）时，cron 直接调发布脚本：

```cron
# 每天 07:30 跑发布（内容由前置 agent 产出；时间配合 publishDate 置顶策略）
30 7 * * *  cd ~/workspace/astro-blog && ./scripts/quant_publish.sh \
  "$(cat /tmp/quant_batch_slugs.txt | head -1)" "$(cat /tmp/quant_batch_cores.txt | head -1)" \
  "$(cat /tmp/quant_batch_slugs.txt | tail -1)" "$(cat /tmp/quant_batch_cores.txt | tail -1)" \
  >> /var/log/quant_publish.log 2>&1
```

- 频率请对齐原 WorkBuddy 自动化（在 WorkBuddy 自动化设置里查看原 rrule，保持一致）。
- 若 dsh 只跑半自动：cron 只需要在你本地/WorkBuddy 产出内容并 push 后，单独跑 `quant_publish.sh`；或干脆把发布也留在 WorkBuddy、只把「调度触发」挪到 dsh 的 Webhook。
- 日志轮转：加 `logrotate` 或 `>>` 配 `date` 分段，避免单文件无限增长。

---

## 7. 已知坑与排错（高频）

| 现象 | 原因 | 解决 |
|---|---|---|
| `build` 报 `bad indentation of a mapping entry` | `description` 内含 ASCII 直引号 `"` 提前闭合标量 | 外层用 ASCII `"`、内部引号改用「」或删掉 |
| `description` 超 600 字符 | schema 硬上限 | 缩写；中文约 200–500 字即可 |
| 文章页 curl 返回 308 | 尾斜杠重定向 | 加 `-L` |
| 专栏页 grep 中文标题返回空 | 1.5MB 单行 HTML，grep 不可靠 | 查 `/blog/{slug}/` 子串，或用 `quant_deploy_check.py` |
| 推送后 curl 仍旧版（age 上千秒） | CDN 缓存未刷新 | 用 no-cache 强制回源；耐心等 3–6 分钟，至少验证两轮 |
| `npm run build` 被 safe-delete 拦截（count>50） | `.vercel/output` 旧产物 | 先 `mv .vercel/output /tmp/...` 再 build |
| `git pull --rebase` 冲突 | `quant-column.md` 多批次并发改动 | 不维护该死文件；或保留双方分组按日期降序合并 |
| `np.trapz` 报错 | numpy 2.x 已删除 | 用 `np.trapezoid`（写 fallback 兼容旧版） |
| 图里 CJK 是方块 | matplotlib 默认字体无 CJK | 顶部显式设 `Heiti TC` 等；dsh Linux 装 `Noto Sans CJK SC` |
| exec 复跑与 results.json 不一致 | core 末次运行改了随机数/顺序 | 固定种子；保证 `print(json)` 与写文件同源 |
| Kendall/相关矩阵「非正定率」恒为 0 | 等相关矩阵经 `sin(πτ/2)` 后仍 PSD | 换异质相关矩阵（低秩因子+个体差异） |

---

## 8. 主题池耗尽问题与当前策略

- `scripts/select_blog_topics.py` 的多级池（主池 662 + 多个 fallback/fresh 池）截至 2026-11 已被 ~1525 篇占用，**主池 + 旧兜底基本用尽**，脚本最终靠 `fresh_pool` 末尾未被占用的 slug 兜底。
- **当前实操策略**（agent 拍板，质量优先）：
  1. 用 `os.listdir` + `git ls-files` 双重查重，确认候选 slug 全库未占用；
  2. 先按关键词 grep 现有目录（本站覆盖极广：execution / cointegration / bootstrap / kelly / copula 等均有多篇），选**互补进阶角度**并交叉引用旧文，避免制造重复灌水；
  3. 同一天多批次时，用 ISO datetime 的 `publishDate` 控制置顶顺序。
- 写入前**绝整个 `rm -rf` `public/images/{slug}/`**（常被旧文占用）；误删用 `git checkout HEAD -- <dir>` 恢复。

---

## 9. 回滚与恢复

- 发布后发现内容问题：`git revert <commit>` 或 `git checkout HEAD~1 -- src/content/blog/<slug> public/images/<slug>`，重新 push；Vercel 自动重新部署。
- 误覆盖旧文（查重失效时）：立即 `git checkout HEAD -- src/content/blog/<旧slug>/index.md public/images/<旧slug>` 恢复，再重新走流程。
- 旧 `.vercel/output` 被挪到 `/tmp/vercel_output_bak_*`：build 成功后可删，与部署无关。

---

## 10. 迁移到 dsh 的校验清单

- [ ] dsh 装好 Python（numpy/scipy/matplotlib/pandas）+ Node/npm + 中文字体。
- [ ] 仓库 clone 完毕、`npm install` 完成、git push 凭证可用。
- [ ] 内容生产方案确定：dsh 侧 LLM agent（全自动）还是 半自动（别处产出）。
- [ ] `scripts/quant_verify.py` / `quant_publish.sh` / `quant_deploy_check.py` 在 dsh 上跑通一次（手动带两个真实 slug 试跑）。
- [ ] cron 条目按原 WorkBuddy 频率配好，日志路径就绪。
- [ ] **确认 dsh 上连续两次发布都成功、线上专栏页出现新文章**后，再去 WorkBuddy 自动化设置里**停用** ID `6a2858b0`（不要先删，先停用观察一周）。
- [ ] 停用后观察 WorkBuddy 自动化列表压力是否如期下降。

---

## 附：文件清单

| 文件 | 作用 |
|---|---|
| `scripts/select_blog_topics.py` | 主题候选列举（多级兜底池 + 磁盘查重） |
| `scripts/quant_verify.py` | **通用** 构建校验（注入代码块 + 全套硬校验 + exec 复跑比对） |
| `scripts/quant_publish.sh` | 机械发布编排（校验→build→git→部署轮询） |
| `scripts/quant_deploy_check.py` | 部署上线轮询校验（处理缓存/308/延迟） |
| `scripts/article_core_{slug}_{date}.py` | 每篇的数值实验本体（每批次新建，非通用） |
| `src/content/blog/{slug}/index.md` | 文章（每批次新建） |
| `public/images/{slug}/*.png` + `results.json` | 配图与真值 |
| `src/content/pages/quant-column.md` | **死内容**，线上不渲染，无需维护 |
