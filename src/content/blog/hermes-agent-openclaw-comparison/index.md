---
title: "用了一个月 Hermes Agent，我决定回到 OpenClaw"
publishDate: '2024-01-01'
description: "用了一个月 Hermes Agent，我决定回到 OpenClaw - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

## 说清楚，不是因为 Hermes 不好。恰恰相反，它的核心创新非常吸引人。只是用下来发现，**有些场景它真的不如 OpenClaw** 。

## Hermes 的核心创新到底是什么？

先说 Hermes 真正有意思的地方。

它的 README 第一句话就是：**"The agent that grows with you"** —— 跟你一起成长的 Agent。

这句话不是营销口号。它有一个具体的技术实现：

每完成一个复杂任务，Hermes 会自动把解决路径整理成一个 **Skill 文件** （结构化的技能文档）。下次遇到类似任务，它会直接调用这个 Skill，而不是重新摸索。有用户报告，用了三个月后，重复性研究任务的耗时缩短了40%。

这和 OpenClaw 的 Skill 机制完全不同。

**OpenClaw 的 Skill 是"装上去"的** —— 你去 SkillHub 搜索、安装、维护，能力来自外部供给。

**Hermes 的 Skill 是"长出来"的** —— 它根据你的使用习惯自动生成，能力来自内部积累。

这个差异听起来很小，实际用起来感受很明显。OpenClaw 是你告诉它怎么干活；Hermes 是它观察你怎么干活，然后自己学会。

这就是为什么 Hermes 在两个月内拿了5.1万 star。**它解决了一个 OpenClaw 一直没解决好的问题：Agent 能不能自己学习？**

![AI Agent](/images/hermes-agent/ai-agent-compare.jpg)

## 为什么我最后还是回到 OpenClaw

说两个具体场景。

### 场景一：快速任务

我想让 Agent 帮我搜一下今天的科技新闻，写个摘要。

OpenClaw：打开会话，说完需求，30秒出结果。

Hermes：启动常驻服务，等待连接，回答……但我今天就想查个东西，为什么要开一整套后台服务？

Hermes 的设计哲学是"常驻 + 持续记忆"，这对需要长期跟踪的任务很有价值。但如果你只是偶尔问个问题，它的存在感太强了，流程比 OpenClaw 重。

### 场景二：平台接入

我想让 Agent 管理我的 Telegram、Slack 和邮件。

OpenClaw：装三个插件，配置一下 token，完事。SkillHub 有现成的。

Hermes：官方支持这些平台，但插件生态远不如 OpenClaw 丰富，配置文档也没那么完善。有几个平台我找了半天没找到对应的接入方式。

### 场景三：上下文成本

这是最实际的问题。

OpenClaw 把所有历史都塞进上下文，长期运行下来 token 消耗会越来越高。Hermes 的解法是 SQLite 分层记忆 —— 高价值决策固化为 Skill，历史交互存在数据库，需要时再召回。

这在理论上更经济。但实际用下来，Hermes 的召回机制还不够精准，有时候我问一个上周处理过的问题，它记忆里有，但召回出来的上下文不够完整，还是得重新解释一遍。

## 什么时候选 Hermes，什么时候选 OpenClaw

场景 | 推荐  
---|---  
长期项目跟踪、研究任务 | Hermes ✅  
重复性工作流（每周报告、自动化脚本） | Hermes ✅  
多平台消息接入（Telegram/Discord/Slack） | OpenClaw ✅  
偶尔用的临时任务 | OpenClaw ✅  
需要本地文件/浏览器深度控制 | OpenClaw ✅  
需要极低运行成本（5美元VPS） | Hermes ✅  
注重数据隐私（无插件市场风险） | Hermes ✅  
  
说白了：**Hermes 是给"让 Agent 帮我做长期项目"的人用的，OpenClaw 是给"让 Agent 帮我处理各种杂事"的人用的。**

## 一个有意思的信号

Hermes 专门做了一个 `hermes claw migrate` 命令，把 OpenClaw 的配置和数据迁移过去。

这说明什么？

开发者自己也承认，这两个东西在用户上有重叠。他们的策略是：**不回避竞争，而是做无缝切换** —— 你想试试 Hermes？一行命令迁移过来，配置都能带走。

这反而让我觉得这个项目更靠谱。知道自己要什么，也知道对手在哪。

* * *

所以我的结论是：**两个都装，根据任务选工具。**

需要长期跟踪、重复性高、想让它越用越聪明 → Hermes。

需要快速执行、多平台接入、偶尔用一下 → OpenClaw。

这不是非此即彼的选择。它们代表 AI Agent 的两条技术路线：**一条是"装上去"，靠生态；一条是"长出来"，靠学习。** 哪个更好？取决于你要解决什么问题。
