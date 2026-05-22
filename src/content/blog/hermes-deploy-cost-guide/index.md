---
title: "跑一个 AI Agent，每月成本从50块降到5块：Hermes 的部署指南"
publishDate: '2026-04-13'
description: "跑一个 AI Agent，每月成本从50块降到5块：Hermes 的部署指南 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

## Hermes Agent 有个功能很少有人提，但它是我见过最实用的——**Daytona 和 Modal 集成** 。

简单说：让你的 Agent 在没活干的时候自动休眠，有活来了再唤醒。闲置时的成本几乎为零。

配合 $5 一月的 VPS，或者直接用 Daytona/Modal serverless，Hermes 的月账单可以压到十几块。而 OpenClaw 因为是常驻进程，跑一个月怎么也要四五十。

这不是省小钱的问题，是改变了 AI Agent 的使用逻辑——以前总觉得"让 Agent 24 小时跑着"很贵，现在可以重新算这笔账。

![服务器成本](/images/hermes-deploy/vps-cost.jpg)

## 一行命令装完，四个地方可以跑

Hermes 的安装是它最友好的地方：
    
    
    curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
    

Linux、macOS、WSL2、Android Termux 都支持。Windows 不行，需要装 WSL2。

装完之后，命令行有完整工具集：
    
    
    hermes              # 启动对话
    hermes model        # 选模型
    hermes tools        # 配置工具
    hermes config set   # 设置参数
    hermes setup        # 全配置向导
    hermes claw migrate # 从 OpenClaw 迁移（重点，后面说）
    hermes doctor        # 诊断问题
    hermes update       # 更新版本
    

消息平台支持 Telegram、Discord、Slack、WhatsApp、Signal，一套网关进程连接所有平台，会话统一管理。

## 六个后端：VPS、服务器集群、serverless

Hermes 支持六种终端后端，按需选：

后端 | 适合场景 | 成本  
---|---|---  
Local | 本地 macOS/Linux | 设备成本  
Docker | 已有 Docker 环境 | 托管费  
SSH | 远程服务器 | VPS 月费  
Daytona | serverless 自动休眠 | 按调用计费，闲置接近 0  
Singularity | HPC/学术场景 | 集群费  
Modal | serverless + 持久化 | 按调用计费，闲置接近 0  
![serverless架构](/images/hermes-deploy/serverless.jpg)

**Daytona 和 Modal 是最有意思的两个。**

Daytona 是开源的 agent 环境管理平台，Hermes 跑在上面时，闲置自动休眠，有消息才唤醒。你不用管服务器，账单按实际调用算。

Modal 也是类似逻辑，官方的说法是"costs nearly nothing when idle"。

这对个人用户最有意义。以前我们总觉得"让 Agent 24 小时待命"是件奢侈的事，现在可以换个算法——**只为你实际用到的部分付钱** 。

## 从 OpenClaw 迁移：一行命令，原有配置全部带走

如果你已经在用 OpenClaw，想试试 Hermes，迁移成本几乎为零。
    
    
    hermes claw migrate    # 完整迁移（交互式）
    hermes claw migrate --dry-run    # 预览，不实际执行
    hermes claw migrate --preset user-data  # 不迁移密钥
    hermes claw migrate --overwrite  # 覆盖已有配置
    

官方会迁移这些东西：

  * **SOUL.md** — 你的 persona 配置
  * **MEMORY.md / USER.md** — 记忆文件
  * **Skills** — 你自己写的 Skills
  * **API keys** — Telegram、OpenRouter、OpenAI、Anthropic 等
  * **消息平台配置** — 接入平台、允许用户、工作目录



说白了，装完 Hermes，跑一条 migrate 命令，你的 OpenClaw 配置基本都搬过去了。不用重配。

## 选模型：支持 200+ 模型，不想花脑子直接用 Nous Portal

Hermes 的模型接入是它另一个强项。支持：

  * **Nous Portal** — 官方订阅制，零配置即用
  * **Anthropic Claude** — API key 或 Claude Code 授权直连
  * **OpenRouter** — 200+ 模型随便切，`hermes model` 一条命令换模型
  * **DeepSeek、阿里云 DashScope（Qwen）、小米 MiMo** 、GitHub Copilot
  * **Ollama 本地模型** — 纯本地，零 API 费用



最近还有一条新闻：小米大模型 MiMo 已接入 Hermes，4月8日到22日限免两周。

### 怎么选？

如果不想折腾：Nous Portal 订阅制，直接用。

如果想省 API 费用：OpenRouter 按量付费，选个性价比高的模型。

如果完全不想花钱：Ollama 本地跑，token 成本为零。

* * *

## 我的结论

Hermes 的部署成本优势是真实的。$5 VPS + Daytona/Modal serverless，把月账单从四五十压到十几块，这个账很容易算。

加上 `hermes claw migrate` 让迁移成本几乎为零——从 OpenClaw 过来，原有配置全部带走，不需要从头开始——这个切换门槛比我预期的低很多。

下一篇，我会写一个更系统的对比：结合 36氪 的报道和官方文档，完整梳理 Hermes Agent 的核心创新。感兴趣可以关注。
