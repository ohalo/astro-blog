---
title: "Hermes Agent 真正创新的地方：不是更会聊，是会自己变强"
publishDate: '2026-04-13'
description: "Hermes Agent 真正创新的地方：不是更会聊，是会自己变强 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

看了我之前写的几篇 Hermes 文章，有读者问我：你把记忆系统、部署成本、迁移工具都单独写了，到底哪篇能让人完整了解 Hermes？

这篇就是答案。

我整合了官方文档和 36氪的报道，加上自己用了一段时间的判断，把 Hermes 的核心创新分成六个维度讲清楚。不是功能罗列，是真正解释——**它凭什么说自己不一样** 。

* * *

## 一、唯一内置"学习循环"的 Agent

![AI学习循环](/images/hermes-innovations/learning-loop.jpg)

这是 Hermes 和所有同类产品本质不同的地方。

大多数 Agent 是一个固定的推理引擎：你输入，它输出，下次重来——完全不保留这次的经验。OpenClaw 就是这样，我用了大半年，每次任务做完，下次遇到类似问题，它完全没有"记忆"。

Hermes 做了什么？它内置了一套触发式学习机制：

**什么情况下会触发学习？**

  * 工具调用超过 5 次还没搞定
  * 用户中途纠正了它的错误
  * 它走了一条"非显性路径"才完成任务
  * 用户明确纠正了它的行为



触发后，Hermes 会自动做两件事：**更新技能文件（Skills）** ，以及**把这次经验写进记忆** 。下次遇到类似场景，它会直接用优化后的路径走。

说白了——它不只完成任务，它从每个任务里吸取教训。这才是"自我改进"agent 该有的样子。

## 二、四层记忆：不是越多越好，是该记的记、不该记的不记

记忆不是越多越聪明。

大多数 Agent 的做法是"全塞进去"——把所有对话、所有上下文都塞进上下文窗口，直到 token 爆炸。效果差，成本高。

Hermes 的做法是分层：

层级 | 内容 | 触发条件  
---|---|---  
SOUL.md | 你的 persona/风格偏好 | 常驻，3575 字符上限  
会话归档 | 跨 session 的历史对话摘要 | FTS5 全文索引，按需召回  
技能文件 | 学会的技能和经验 | 任务类型触发  
Honcho 用户建模 | 你的工作风格、偏好 | 周期性更新  
  
SOUL.md 有 3575 字符上限这件事很有意思——官方刻意限制，逼你做减法：**只放最重要的** ，其余的靠技能文件和会话归档按需召回。

## 三、模型随意换：一条命令切换，不锁定供应商

![多模型切换](/images/hermes-innovations/model-switch.jpg)

大多数 Agent 让你选一个模型，然后用到底。Hermes 的逻辑完全不同——**模型是插件，不是核心** 。

支持：Nous Portal（官方订阅）、OpenRouter（200+ 模型）、Claude（Anthropic 直连）、DeepSeek、Qwen（阿里云 DashScope）、小米 MiMo（限免中）、OpenAI、Ollama（本地）、自己架设的端点。

切换模型只需要一条命令：
    
    
    hermes model

选一个模型，打完。

最近比较值得关注的是小米 MiMo 接入 Hermes，4月8日到22日限免两周。这给国内用户多了一个低门槛选择。

这个模型灵活性带来的实际好处：你可以随时对比不同模型在同任务上的表现，然后选性价比最高的。不用换 Agent，不用重配置，直接切。

## 四、开放标准 agentskills.io：学到的技能不只是 Hermes 能用

这是 Hermes 最被低估的创新。

它支持 agentskills.io 开放标准——简单说，你在 Hermes 里创建的技能，不只属于 Hermes，任何兼容这个标准的 Agent 都能用。

当前支持 agentskills.io 的产品已经有好几个，生态在扩大。这改变了技能开发的逻辑：以前你为一个 Agent 写的工具，Agent 换了就废了；现在写的技能，是可以跨平台复用的资产。

加上 Hermes 本身支持 Skills Hub（社区技能市场），以及用户可以自己创建技能（直接写 SKILL.md 文件），整个技能生态是可积累的。

## 五、Serverless 改变成本逻辑：闲置时几乎不花钱

![serverless架构](/images/hermes-innovations/architecture.jpg)

这是让我真正认真对待 Hermes 的原因。

Daytona 和 Modal 两个后端，让 Hermes 可以跑在完全 serverless 的环境里——没活干的时候自动休眠，有消息来了才唤醒，按实际调用计费。

官方原话： _"costs nearly nothing when idle"_

对比一下：

  * **OpenClaw：** 常驻进程，24 小时跑着，月费四五十起步
  * **Hermes + Daytona/Modal：** 按调用计费，闲置时几乎零成本



这个差距不是"省小钱"，是改变了使用 Agent 的心理门槛。以前我会想"24 小时开着有点浪费"，现在可以换个心态：Agent 在睡觉，不花我钱。

配合 $5 一月的 VPS，或者直接用 Daytona/Modal 的 serverless 模式，Hermes 的月账单可以压到十几块。

## 六、研究友好：从使用到训练下一代模型

这部分对普通用户没什么用，但对 AI 研究者来说很有意思。

Hermes 内置了 Atropos RL 集成，支持 batch trajectory generation——简单说，你可以用它来收集高质量的 tool-calling 数据，用来训练下一代模型。

还支持 trajectory compression，减少存储开销。

如果你在研究 agent 的强化学习方向，Hermes 不只是一个工具，它本身就是一个研究平台。

* * *

## 总结：Hermes 真正创新的地方

创新点 | 核心价值  
---|---  
内置学习循环 | 每次任务后自动改进，不用手动维护  
四层记忆分层 | 该记的记，不浪费 token  
模型随便换 | 不锁定供应商，随时比价  
agentskills.io 开放标准 | 技能跨平台复用  
Daytona/Modal serverless | 改变成本逻辑  
RL 研究集成 | 从工具到平台  
  
Hermes 不是一个"更聪明的聊天机器人"。它的野心是成为一个**会自我改进、可积累经验、跨平台复用、不被供应商锁定的 AI agent 基础设施** 。

这个定位比大多数同类产品高一个层次。

我用了它一段时间，又回到 OpenClaw——这是另一个话题。但 Hermes 展现的方向是值得关注的。它在解决的不只是"怎么让 AI 帮你做事"，而是"怎么让 AI 在帮你做事的过程中变得越来越会帮你做事"。

这是本质不同的两种思路。
