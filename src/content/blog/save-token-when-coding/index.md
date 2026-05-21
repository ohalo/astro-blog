---
title: "用 Claude Code 写代码，怎么省 token？8000万 token 的账我帮你算明白了"
publishDate: '2024-01-01'
description: "用 Claude Code 写代码，怎么省 token？8000万 token 的账我帮你算明白了 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

# 用 Claude Code 写代码，怎么省 token？8000万 token 的账我帮你算明白了

上篇文章写完，有人在评论区说："道理我都懂，但 DeepSeek 也要 160 块，谁用得起？"

我后来算了算，发现一件事：**不是模型贵，是消耗量太大了。**

同样的需求，你让 Claude 读 50 个文件 vs 只读 5 个文件，token 消耗差 10 倍。这不是模型定价问题，是**用法问题** 。

Anthropic 官方文档里其实给了完整方案链。我用了两周，把每个方案都实测了一遍，这篇文章把最有效的 3 个讲清楚。

* * *

## 先说清楚一件事：token 消耗的大头在哪

根据官方文档，一个典型会话的上下文窗口（200K token）是这样填满的：
    
    
    启动时自动消耗（约 7-8K token）：
    ├── System prompt        4,200 token
    ├── Auto memory          680 token（之前会话的笔记）
    ├── 环境信息              280 token
    ├── Skill 描述列表        450 token（只是描述，不是内容）
    ├── ~/.claude/CLAUDE.md   320 token
    └── 项目 CLAUDE.md       1,800 token
    
    对话过程中：
    ├── 你发的每条消息
    ├── Claude 读的每个文件  ← 【最大头，通常 1-3K token/文件】
    ├── Claude 运行的命令输出
    └── 历史对话累积
    

**文件读取才是 token 消耗的主力。** 你让 Claude "帮我看看这个项目有什么问题"，它可能读了 30 个文件；你换成 "帮我看看 src/auth/ 目录下的登录模块有什么问题"，它只读 3 个文件。

省 token 的本质：**别让它读不该读的东西。**

* * *

## 方案一：把长篇参考文档从 CLAUDE.md 移到 Skills

这是最容易上手、效果最立竿见影的方案。

CLAUDE.md 每次会话都会被完整加载到上下文窗口。Anthropic 官方建议**不超过 200 行** ，但很多人的 CLAUDE.md 已经堆到四五百行了——每次启动先烧掉 3-4K token，还不算对话。

而 Skills 的机制完全不同：**只有在用的时候才加载，不用的时候几乎不消耗 token。**
    
    
    CLAUDE.md（每次加载）：
    第1行  →  上下文
    第2行  →  上下文
    ...
    第500行 → 上下文  ← 每次会话都要读，哪怕这次根本不需要
    
    Skill（按需加载）：
    第1行  →  调用时才加载，不用就不读
    

具体怎么操作：

  1. **识别哪些内容可以移走** ：API 设计规范、Docker 使用手册、代码审查 checklist——这些不是每次会话都需要的内容
  2. **创建一个 Skill 文件** ：


    
    
    mkdir -p ~/.claude/skills/api-conventions

在 `~/.claude/skills/api-conventions/SKILL.md` 里写：
    
    
    ---
    name: api-conventions
    description: API 设计规范。调用 /api-conventions 时使用。
    ---
    
    ## 这个项目的 API 规范
    
    - RESTful 命名，用名词不用动词
    - 统一错误格式：{ code, message, data }
    - 所有接口必须带参数校验

  3. **在对话中调用** ：`/api-conventions`



官方文档原话是："Unlike CLAUDE.md content, a skill's body loads only when it's used, so long reference material costs almost nothing until you need it."

**实测效果** ：我的 CLAUDE.md 从 460 行精简到 120 行，每次会话省了约 2,800 token。一个 10 次会话的项目，省了 28,000 token，换算下来便宜了不少。

* * *

## 方案二：用 Sub-agents 做代码探索

这是官方内置的能力，但用的人不多。

Claude Code 有一个内置子代理叫 **Explore** ，专门用来搜索和分析代码库。它的特点：

  * **用 Haiku 模型** （最便宜的那档，价格是 Sonnet 的 1/10）
  * **只读权限** ，不能写文件
  * **独立上下文窗口** ，探索结果不会污染你的主会话



典型场景：你接手一个陌生的代码库，想先摸清结构。以前你会说"帮我了解一下这个项目"，Claude 读了 20 个文件后给出一个概述——这 20 个文件的内容全部留在了你的主会话上下文里，之后越用越慢。

现在改成：先说"/explore 帮我了解一下这个项目"，Explore 用 Haiku 在独立窗口里完成探索，**只返回一个简洁的总结** ，你的主会话上下文完全不受影响。
    
    
    【旧做法】
    你："帮我了解这个项目"
    → Claude 读 20 个文件 → 20 个文件全在主上下文里
    
    【新做法】
    你："/explore 帮我了解这个项目"
    → Explore 读 20 个文件 → 只返回一个总结 → 主上下文干净
    

官方原话："Delegating research to a subagent keeps large file reads out of your main window."

另外，**Explore 子代理默认用 Haiku 模型** ——这个模型价格是 Sonnet 的十分之一，但做代码搜索完全够用。复杂任务需要写代码时才切换回 Sonnet。

* * *

## 方案三：CLAUDE.md 精简到 200 行以内，配合路径规则

这是 Anthropic 官方明确给出的建议，但被很多人忽略了。

文档里写：

> "Keep it under 200 lines. Longer files consume more context and reduce adherence."

CLAUDE.md 超过 200 行之后，Claude 对指令的遵守度反而会下降——因为它需要在更长的内容里找相关指令，注意力被分散了。

精简 CLAUDE.md 的优先级：

**必留（每次都要知道的）：**

  * 构建命令（`npm run dev`、`pytest` 等）
  * 测试命令
  * 分支命名规范
  * 代码风格（2 空格缩进还是 4 空格）



**可移走（不需要每次都知道的）：**

  * 详细的设计规范（移到 Skill）
  * 某个模块的特殊说明（移到路径规则）
  * 团队成员信息



对于只适用于部分文件的规则，用 **`.claude/rules/` 路径规则**：
    
    
    .claude/rules/
    ├── api-conventions.md    ← 只有访问 src/api/** 时才加载
    ├── testing.md            ← 只有访问 *.test.ts 时才加载
    └── security.md           ← 只有访问 **/auth** 时才加载

这样 Claude 每次只需要加载**跟当前任务相关的** 那部分规则，其他内容完全不会进入上下文。

* * *

## 这三个方案加在一起，能省多少？

方案| 每次节省| 难度  
---|---|---  
CLAUDE.md 精简到 200 行| ~2-3K token/次| ⭐ 简单  
Skills 按需加载| ~1-5K token/次| ⭐⭐ 中等  
Explore 子代理做探索| ~5-15K token/次| ⭐ 简单  
  
一个 20 次会话的中型项目，用这三个方案，总共能省 **10-40 万 token** 。

换算成钱：DeepSeek 的话，大约能省 **2-8 块钱** 。听起来不多对吧？但这是同一个需求下的消耗对比。如果你原来用 8000 万 token，优化后可能只需要 2000 万——那就是 **160 块变成 40 块** 的区别。

**问题的本质从来不是模型贵不贵，是你的用法让不让它省着用。**

* * *

## 一个反直觉的提醒

写这篇文章的时候，我一开始想加一节"怎么接入 DeepSeek API 来省钱"。后来没写。

原因是：我之前试过接本地 120B 模型，图的是"token 不要钱"。结果发现本地模型指令跟随能力弱，同样的需求描述扔过去，它先反问我三个问题，再跑偏答非所问，最后花的时间比省的钱多得多。

省 token 不是为了省钱而省钱。**能用 Sonnet 一次搞定的事，不要用 Haiku 跑三遍。** 把精力放在 CLAUDE.md、Skills、Sub-agents 这些"不降质还提效"的做法上。

* * *

你觉得这三个方案里，哪个对你最有帮助？

我自己的感受是 Explore 子代理是最被低估的——很多人不知道它存在，更不知道它默认用 Haiku 模型。如果你用过它，欢迎在评论区说说体验。
