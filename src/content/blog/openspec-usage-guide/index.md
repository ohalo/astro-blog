---
title: "OpenSpec 中文上手指南：让AI按你的规矩写代码"
publishDate: '2026-04-15'
description: "OpenSpec 中文上手指南：让AI按你的规矩写代码 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

AI编程OpenSpec规范驱动开发Claude CodeCursor

你用 Claude Code 或 Cursor 写代码的时候，有没有这种感觉——

聊着聊着，AI 不知道哪去了，代码写得跟你的意图完全不一样。你让它加个深色模式，它把登录流程重写了；你让它修个 bug，它顺手把数据库迁移了。

不是 AI 不好，是你俩没签合同。

**OpenSpec** 干的就是这件事：给 AI 一份结构化的需求文档，让它照着干，别自由发挥。

这是官方中文文档站（radebit.github.io/OpenSpec-Docs-zh）的实操手册，翻译整理自官方文档，配合我的实际使用体验。

![OpenSpec Dashboard](/images/openspec-manual/openspec-dashboard.png)

## 先安装，两分钟的事
    
    
    # 需要 Node.js 20.19.0+
    npm install -g @fission-ai/openspec@latest
    
    # 验证
    openspec --version

进入你的项目目录初始化：
    
    
    cd your-project
    openspec init

初始化会创建 `openspec/` 目录，自动为你的 AI 工具生成配置文件。完成后 AI 工具就能识别 OpenSpec 指令了。

## 核心概念就两个，看完就能干活

### 规范（specs）：系统现在的行为

`openspec/specs/` 里放的是**当前系统的事实** ——不需要描述"想加什么"，只描述"现在是什么"。

比如 `openspec/specs/auth/spec.md`：
    
    
    ## 目的
    应用程序的身份验证和会话管理。
    
    ## 需求
    ### 需求：用户身份验证
    系统 SHALL 在登录成功时颁发 JWT 令牌。
    
    #### 场景：有效凭据
    - GIVEN 用户具有有效凭据
    - WHEN 用户提交登录表单
    - THEN 返回 JWT 令牌
    - AND 用户被重定向到仪表板

注意这里用的是 `SHALL`（RFC 2119 关键词），意思是"必须"。规范描述的是**行为契约** ，不是实现步骤——没有说用什么框架、哪个类、怎么写代码。

### 变更（changes）：你想改什么

`openspec/changes/` 里的每个文件夹，是一个**独立的变更** 。每个变更包含：
    
    
    openspec/changes/add-dark-mode/
    ├── proposal.md    # 为什么做、做什么
    ├── design.md      # 怎么实现
    ├── tasks.md       # 任务清单
    └── specs/        # Delta 规范（正在改的部分）

Delta 规范里只写改了什么：`ADDED`（新增）、`MODIFIED`（修改）、`REMOVED`（删除）。不影响其他部分。

这个设计的好处是什么？**你可以在不影响主线的情况下并行做多个变更** ，最后统一合并。

## 最常用的一组命令

在 AI 对话窗口里直接敲斜杠命令：

### /opsx:propose —— 开启一个新变更
    
    
    你: /opsx:propose add-dark-mode
    
    AI:
    已创建 openspec/changes/add-dark-mode/
    ✓ proposal.md
    ✓ specs/ui/spec.md
    ✓ design.md
    ✓ tasks.md
    
    准备好实现。运行 /opsx:apply。

用自然语言也行：
    
    
    你: /opsx:propose 给后台管理系统加一个深色模式

### /opsx:explore —— 需求不明确，先探索
    
    
    你: /opsx:explore 移动端身份验证方案
    
    AI:
    你想探索什么？
    你: 如何处理移动端身份验证？
    
    AI:
    我看到你在 Web 端用了基于 session 的认证。移动端有三种选择：
    
    1. JWT 令牌（无状态，支持离线）
    2. OAuth2（如需第三方登录）
    3. 扩展当前 session（更简单，但需要连接）
    
    你想哪个方向？

探索阶段不会创建任何产物，适合在拿不定主意的时候先研究。

### /opsx:apply —— 让 AI 照着任务清单实现
    
    
    你: /opsx:apply
    
    AI:
    开始实现 tasks.md 中的任务...
    
    [任务1/5] 添加 CSS 变量定义...
    [任务2/5] 修改组件使用 CSS 变量...
    ...

它会逐条执行 `tasks.md` 里的任务，每完成一条会告诉你。

### /opsx:archive —— 合并变更，关掉它
    
    
    你: /opsx:archive
    
    AI:
    将 Delta 规范合并到主规范。
    变更文件夹已移至 openspec/changes/_archived/add-dark-mode/

Delta 自动合并到 `specs/` 里，变更进入归档状态。

## 进阶用法：更细粒度的控制

如果默认的 propose → apply → archive 不够用，可以开启**扩展工作流** ：
    
    
    openspec config profile
    # 选择 custom
    openspec update

然后多了这些命令：

  * `/opsx:new` —— 只创建变更脚手架，不自动生成产物
  * `/opsx:continue` —— 逐步创建产物（proposal → design → tasks），可以中途修改
  * `/opsx:ff` —— 一次性创建所有规划产物（适合需求已经很清楚的情况）
  * `/opsx:verify` —— 验证实现是否与产物匹配，在合并前做最后检查
  * `/opsx:sync` —— 把 Delta 规范合并到主规范



## 我自己用下来的感受

用了大概两周，说几个真实的感受：

**好的地方：**

  * AI 确实不会乱跑了。它会先看 proposal 和 design，知道你要什么再动手。
  * 多变更并行很方便。我同时在做深色模式和性能优化两个功能，互不干扰。
  * Delta 规范的设计很适合存量项目——不用重写整个系统，只写改了什么。



**需要注意的地方：**

  * 用 `/opsx:propose` 时，你只说一句话，AI 自动生成 proposal/design/tasks 三个文件。听起来很轻，但习惯上还是需要先把需求想清楚再动手——描述越清楚，生成的东西越准确。
  * 协作功能还在开发中，团队场景目前体验一般。
  * 它本身不强制 TDD，如果你想 TDD，需要自己在 tasks.md 里加要求。



**适合谁？**

  * 个人开发者或小团队（2-5人）
  * 正在做需要精细控制 AI 的项目
  * 不习惯 TDD 但想要基本质量保证的



**不适合谁？**

  * 想快速跑通 Demo 的（流程偏重）
  * 已经有严格代码审查流程的团队（OpenSpec 的约束可能和你们的流程冲突）



## 最后

说实话，我最初觉得这东西有点多余——"不就是让 AI 多写几个文件吗？"

但用了一周后意识到，**真正值钱的不是规范文件本身，而是你和 AI 之间的"对齐"过程** 。

你被迫在动手之前把需求想清楚：做什么、为什么做、怎么做。AI 照着文件走，你盯着它，不容易跑偏。

工具本身不复杂，关键是习惯用它——从"直接让 AI 写代码"变成"先说清楚要什么"。

![OpenSpec Logo](/images/openspec-manual/openspec-logo.png)
