---
title: "OpenClaw onboard：一条命令，10个步骤，从零到能用"
publishDate: '2026-04-10'
description: "OpenClaw onboard：一条命令，10个步骤，从零到能用 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

刚装完 OpenClaw，终端里敲什么？ 

很多人第一反应是 `openclaw help`。但真正应该敲的是： 
    
    
    openclaw onboard
    

这一个命令，会带你走完 10 个步骤——从检测现有配置到安装守护进程，从选模型到接渠道，最后给你一个 24 小时在线的 AI 助手。 

我自己跑过两次这个流程，第一次选了 OpenAI API + Telegram，第二次换了 Codex 订阅 + 微信。两次体验完全不一样，踩过的坑也不一样。 

* * *

## 第一步：它不会乱删你的东西

如果你之前跑过 `onboard`，再跑一次会怎样？ 

它会问你要不要保留现有配置。选项有三个： 

  * **保留** ：直接跳到下一步
  * **修改** ：保留部分配置，改其他的
  * **重置** ：清空重来



重点：**重置用`trash`，不用 `rm`**。删掉的东西在废纸篓里，还能找回来。 

除非你明确选了"重置"，否则什么都不会丢。 

* * *

## 第二步：选模型，像选手机套餐

这是最容易卡住的一步。 

OpenClaw 支持的模型提供商有十几个：Anthropic、OpenAI、Codex、xAI、Ollama、MiniMax、StepFun、Moonshot…… 

怎么选？我整理了一个简单的对比： 

方案| 适合谁| 成本  
---|---|---  
Codex 订阅| 重度用户，每天用| $20/月，不限量  
OpenAI API| 轻度用户，偶尔用| 按次计费，几毛到几块  
Ollama 本地| 有显卡、重视隐私| 免费，但需要硬件  
MiniMax| 国内用户，中文优先| 按 API 用量计费  
  
**我自己选的是 Codex 订阅** 。原因很简单：每天用，API 费用加起来比订阅还贵。 

如果你有现成的 `ANTHROPIC_API_KEY` 或 `OPENAI_API_KEY` 环境变量，onboard 会自动检测到，省得你手动输入。 

* * *

## 第三步：工作区在哪

默认是 `~/.openclaw/workspace`。 

这个目录里有你的 AGENTS.md、SOUL.md、TOOLS.md，还有 memory 目录——AI 的"长期记忆"存在这里。 

不建议改这个路径，除非你有特殊的目录管理习惯。改了之后容易忘，出了问题排查也麻烦。 

* * *

## 第四步：Gateway 是什么

Gateway 是 OpenClaw 的"大脑"，负责接收消息、调用模型、执行命令。 

配置项不多，但有一个坑： 

**即使是本地运行，也建议设置 Token。**

为什么？因为如果你开了 WebSocket 客户端（比如某些第三方前端），没有 Token 的话，任何人都能连上来。 

Token 模式下，你可以选择： 

  * 明文存储（简单，适合个人电脑）
  * SecretRef（复杂，适合服务器环境）



我自己的 Mac 上用的是明文，服务器上用的是 SecretRef。 

* * *

## 第五步：接渠道，让你的 AI 有"入口"

这是 OpenClaw 真正强大的地方。 

你可以让 AI 接入这些渠道： 

  * **Telegram** ：最简单，找 @BotFather 创建机器人，拿到 Token 就行
  * **Discord** ：类似 Telegram，创建 Bot 拿 Token
  * **WhatsApp** ：扫码登录
  * **Signal** ：需要安装 `signal-cli`
  * **iMessage** ：推荐用 BlueBubbles 方案



我第一次选的是 Telegram，因为最简单——不需要公网 IP，不需要额外配置，几分钟就跑通了。 

**一个细节** ：私聊默认开启"配对模式"。第一次有人私聊你的 Bot，会收到一个验证码。你需要运行： 
    
    
    openclaw pairing approve telegram XXXXXX
    

这样别人就不能随便用你的 Bot 了。 

* * *

## 第六步：Web 搜索（可选）

让 AI 能联网搜索。支持 Brave、DuckDuckGo、Perplexity 等。 

这一步可以跳过。如果你主要用它处理本地任务（写代码、整理文件），不搜网也行。 

* * *

## 第七步：守护进程，让它 24 小时在线

这是很多人容易忽略的一步。 

`onboard` 会问你要不要安装守护进程： 

  * **macOS** ：安装 LaunchAgent，开机自启
  * **Linux** ：安装 systemd 用户单元，登出后继续运行



**强烈建议安装。**

不装的话，每次用之前都要手动 `openclaw gateway start`，很麻烦。装了之后，开机就自动跑，你随时发消息，它随时回。 

* * *

## 第八步：健康检查

最后会跑一次 `openclaw health`，检查所有配置是否正常。 

如果有问题，会直接告诉你怎么修。 

* * *

## 非交互模式：脚本化安装

如果你要在服务器上批量部署，可以用非交互模式： 
    
    
    openclaw onboard --non-interactive \
      --mode local \
      --auth-choice apiKey \
      --anthropic-api-key "$ANTHROPIC_API_KEY" \
      --gateway-port 18789 \
      --gateway-bind loopback \
      --install-daemon \
      --daemon-runtime node \
      --skip-skills
    

一条命令，全自动跑完，适合 CI/CD 或服务器初始化。 

* * *

## 我的建议

如果你刚装完 OpenClaw，不知道从哪开始： 

  1. 先跑 `openclaw onboard`，别自己改配置文件
  2. 模型选 Codex 订阅（重度用户）或 API（轻度用户）
  3. 渠道先接 Telegram，最简单
  4. 守护进程一定要装
  5. 跑完 `health` 检查，全绿就可以用了



这一套走下来，大概 10 分钟。之后你就有了一个 24 小时在线、能帮你干活的 AI 助手。 

* * *

_你跑 onboard 的时候卡在哪一步？欢迎评论区说说。_

### 💬 评论区
