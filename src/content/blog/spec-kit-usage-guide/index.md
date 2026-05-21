---
title: "Spec-Kit 上手指南：GitHub官方的AI编程规范框架"
publishDate: '2024-01-01'
description: "Spec-Kit 上手指南：GitHub官方的AI编程规范框架 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

AI编程Spec-KitSDDGitHubClaude Code

你有没有试过让AI写代码，结果它越写越偏，最后完全不是你想要的？

你跟 Claude Code 说"给我加个用户登录功能"，它给你写了一堆，但你发现它没考虑密码加密、没考虑session管理、没考虑登录失败的处理。等你反应过来，它已经改了二十个文件。

这不是AI的问题，是你俩没对齐。

**Spec-Kit** 是 GitHub 官方开源的工具包，用来解决这个问题。它的核心理念叫 **Spec-Driven Development（规范驱动开发）** ——在让AI写代码之前，先把"要做什么、怎么做"写成可执行的规范，AI照着规范实现，而不是自由发挥。

![Spec-Kit Logo](/images/spec-kit-manual/spec-kit-logo.webp)

## 安装：两分钟搞定

**前置要求** ：已安装 Python 3.11+、Git 和 uv（Python 包管理工具）。

Spec-Kit 基于 Python，用 `uv` 管理：
    
    
    # 安装 specify CLI
    uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
    
    # 验证
    specify --version

然后初始化项目：
    
    
    # 新建项目目录并初始化
    specify init my-project --ai claude
    cd my-project
    
    # 或在现有目录初始化
    specify init . --ai claude

`--ai claude` 告诉它你要用 Claude Code。支持 20 多个 AI 工具，包括 Cursor、Copilot、Gemini、Codex 等。

初始化后，你的项目会多一个 `.specify/` 目录，里面有四个核心文件夹：

  * `memory/` \- 项目记忆和上下文
  * `scripts/` \- 自动化脚本
  * `specs/` \- 规范文件存放处
  * `templates/` \- 模板文件



## 五阶段工作流：从想法到实现

Spec-Kit 把整个开发流程拆成五个阶段，每个阶段都有对应的斜杠命令（需在 AI agent 如 Claude Code 中执行）：

阶段 | 命令 | 干什么  
---|---|---  
1\. 定规矩 | `/speckit.constitution` | 创建项目原则，比如"必须用TypeScript"、"必须写测试"  
2\. 定需求 | `/speckit.specify` | 描述你要做什么，只讲"做什么"，不讲"用什么技术"  
3\. 定方案 | `/speckit.plan` | 确定技术栈和实现方案  
4\. 拆任务 | `/speckit.tasks` | 把方案拆成可执行的任务清单  
5\. 执行 | `/speckit.implement` | AI 按任务清单逐项实现  
  
**可选核心命令** ：`/speckit.taskstoissues` \- 将任务清单转为 GitHub Issues 跟踪。

看起来步骤很多，实际用起来很顺。我给你演示一遍（基于 Spec-Kit v0.7.0）：

### 第一步：定规矩
    
    
    你: /speckit.constitution
    
    AI: 已创建 .specify/CONSTITUTION.md
    
    项目原则：
    - 代码必须用 TypeScript
    - 所有功能必须有单元测试
    - 优先使用原生 API，减少依赖
    - 错误处理必须捕获并记录

这个文件是项目的"宪法"，以后所有的规范都要遵守这些原则。

### 第二步：定需求
    
    
    你: /speckit.specify 做一个照片相册应用，可以按日期分组，支持拖拽排序，相册不能嵌套
    
    AI: 已创建 .specify/features/001-photo-albums/spec.md
    
    ## 需求：照片相册管理
    
    ### 场景：创建相册
    - GIVEN 用户在主页面
    - WHEN 点击"新建相册"
    - THEN 弹出对话框输入相册名称
    - AND 创建空相册并显示在列表中
    
    ### 场景：拖拽排序
    - GIVEN 用户有多个相册
    - WHEN 拖拽相册到目标位置
    - THEN 相册顺序更新
    - AND 新顺序持久化保存

注意这里完全没有提技术栈——没说用 React 还是 Vue，没说用 SQLite 还是 PostgreSQL。这是故意的，Spec-Kit 把"做什么"和"怎么做"分开。

### 第三步：定方案
    
    
    你: /speckit.plan 用 Vite + React，尽量减少第三方库。图片存在本地，元数据用 SQLite。
    
    AI: 已创建 .specify/features/001-photo-albums/plan.md
    
    技术方案：
    - 前端：Vite + React + TypeScript
    - 状态管理：React Context（不用 Redux）
    - 存储：SQLite（via sql.js 浏览器版）
    - 拖拽：HTML5 Drag and Drop API（不用库）

### 第四步：拆任务
    
    
    你: /speckit.tasks
    
    AI: 已创建 .specify/features/001-photo-albums/tasks.md
    
    - [ ] 初始化 Vite 项目
    - [ ] 配置 SQLite 数据库
    - [ ] 实现相册列表组件
    - [ ] 实现拖拽排序功能
    - [ ] 实现新建相册对话框
    - [ ] 添加单元测试

### 第五步：执行
    
    
    你: /speckit.implement
    
    AI: 开始执行任务...
    [1/6] 初始化 Vite 项目 ✓
    [2/6] 配置 SQLite 数据库 ✓
    [3/6] 实现相册列表组件 ✓
    ...

AI 会逐项完成任务，每完成一项会告诉你。你可以随时中断，下次继续。

## 扩展生态：不止这五个命令

Spec-Kit 的核心只有五个命令，但社区贡献了近 **50 个扩展** ，覆盖各种场景：

  * **Jira 集成** ：把 tasks 同步到 Jira
  * **代码审查** ：实现完后自动做代码 review
  * **GitHub Issues** ：从 issue 生成 spec
  * **测试生成** ：根据 spec 自动生成测试用例
  * **安全检查** ：实现完后做安全审计



安装扩展很简单：
    
    
    specify extension search jira    # 搜索扩展，确认正确名称
    specify extension add spec-kit-jira  # 安装 Jira 集成（扩展名需与官方列表一致）

## 我自己用下来的感受

基于 Spec-Kit v0.7.0（2026年4月15日发布）用了大概一个月，说几个真实的感受：

**好的地方：**

  * **AI 确实不乱跑了** 。constitution 和 spec 给它划了边界，它知道什么该做、什么不该做。
  * **需求变更有迹可循** 。如果中途想改需求，改 spec.md，AI 会自动同步到 plan 和 tasks。
  * **适合复杂功能** 。越复杂的功能，提前写 spec 的收益越大。



**需要注意的地方：**

  * **前期确实要多一步** 。以前直接说"给我做个登录"，现在要先写 constitution、再写 spec、再写 plan。小功能可能觉得麻烦。
  * **constitution 要写清楚** 。如果 constitution 写得模糊，AI 还是会跑偏。比如"写测试"不够，要写"单元测试覆盖率必须≥80%"。
  * **不是所有场景都适合** 。快速原型、验证想法的阶段，直接 vibe coding 更快。



**适合谁？**

  * 需要精细控制 AI 输出的场景
  * 复杂功能开发（> 3 天的任务）
  * 团队协作（spec 可以 code review）



**不适合谁？**

  * 快速验证想法（用 vibe coding 更快）
  * 已经有严格开发流程的团队（Spec-Kit 可能和你们现有流程冲突）



## 最后

Spec-Kit 不是让 AI 写代码更快，而是让 AI 写代码**更可预测** 。

它的核心理念——"spec 先于 code"——其实和传统软件工程的做法是一致的。只是以前写 spec 是给程序员看的，现在写 spec 是给 AI 看的。

如果你也被 AI "自由发挥"困扰过，可以试试这个框架。

* * *

你现在用 AI 写代码是什么 workflow？直接聊还是也用什么框架？评论区说说。
