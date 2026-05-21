---
title: "Claude Code 记忆机制：六层体系实操指南"
publishDate: '2024-01-01'
description: "Claude Code 记忆机制：六层体系实操指南 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

Claude CodeCLAUDE.mdAuto Memory记忆机制

> 看完这篇，你会彻底搞懂 Claude Code 的记忆分层，不再被"明明告诉过它"的问题困扰。

## 先说痛点

用 Claude Code 的人大概都遇到过：

> 关掉会话，第二天回来，它什么都不记得了。项目用什么包管理器？代码风格怎么约定？上次调试到哪里？全要重新解释一遍。

Anthropic 推出了 **Auto Memory** 功能，让 Claude Code 自己记笔记。听起来很美好，但如果你不理解它的记忆体系是怎么分层的，很容易搞出一堆互相矛盾的指令。

## 一、两套记忆系统

Claude Code 有两套互补的记忆系统：

| CLAUDE.md| Auto Memory  
---|---|---  
**谁写的**|  你| Claude 自己  
**里面是什么**|  指令、规则| 学到的模式、踩过的坑  
**作用范围**|  项目/用户/组织| 单个项目  
**什么时候加载**|  每次会话| 每次会话（前200行）  
**用来干嘛**|  编码规范、工作流、架构决策| 构建命令、调试经验、偏好  
  
简单说：**CLAUDE.md 是你在管它，Auto Memory 是它自己在总结经验。**

## 二、六层记忆结构

CLAUDE.md 不是一个文件，而是一个层级体系。从全局到局部：

层级| 位置| 谁维护| 共享范围  
---|---|---|---  
组织策略| `/Library/Application Support/ClaudeCode/CLAUDE.md`| IT/DevOps| 组织内所有人  
项目记忆| `./CLAUDE.md` 或 `./.claude/CLAUDE.md`| 团队| 通过 Git 共享  
项目规则| `./.claude/rules/*.md`| 团队| 通过 Git 共享  
用户记忆| `~/.claude/CLAUDE.md`| 个人| 所有项目  
项目本地| `./CLAUDE.local.md`| 个人| 仅当前项目  
Auto Memory| `~/.claude/projects/<project>/memory/`| Claude| 仅你自己  
  
**越具体的层级优先级越高。** 项目规则覆盖用户偏好，本地配置覆盖项目配置。这个设计很像 Git 的配置层级：`--system`、`--global`、`--local`。

## 三、CLAUDE.md 怎么写

CLAUDE.md 本质就是 Markdown，用自然语言写指令。

### 一个例子
    
    
    # 项目约定
    - 使用 pnpm，不要用 npm
    - 测试命令：pnpm test
    - 提交前必须跑 lint
    
    # 代码风格
    - TypeScript 严格模式
    - 组件用 PascalCase，工具函数用 camelCase

### 三条实用建议

  1. **把常用命令写进去** \- Claude Code 每次都要翻 package.json 找构建命令，不如直接告诉它。
  2. **写具体的约定，不写模糊的要求** \- ❌ "代码要简洁" → ✅ "函数不超过 30 行"
  3. **用 /init 自动生成** \- Claude Code 会扫描项目结构，生成一份基础的 CLAUDE.md。



## 四、模块化规则：.claude/rules/

当项目变大，一个 CLAUDE.md 会变得又长又杂。`.claude/rules/` 目录解决这个问题——按主题拆分。

### 条件规则：只在特定文件时生效
    
    
    ---
    paths:
    - "src/api/**/*.ts"
    ---
    # API 开发规则
    - 所有端点必须做输入校验
    - 使用标准错误响应格式

带 paths 的规则只在 Claude 实际读写匹配文件时才生效。处理前端代码时这条规则不起作用，只有碰到 `src/api/` 下的 TypeScript 文件时才会生效。

## 五、Auto Memory：让它自己记

这是新功能。Claude Code 会在工作过程中自己记笔记。

### 它记什么

  * **项目模式** ：构建命令、测试约定、代码风格
  * **调试经验** ：遇到过的坑、解决方案
  * **架构笔记** ：关键文件、模块关系
  * **你的偏好** ：沟通风格、工作习惯、工具选择



### 存在哪里
    
    
    ~/.claude/projects/<project>/memory/
    ├── MEMORY.md      # 索引文件，每次启动加载前 200 行
    ├── debugging.md   # 调试相关笔记
    ├── api-conventions.md  # API 设计决策
    └── ...

### 怎么控制

Auto Memory 默认开启。如果不想用：对话里跑 `/memory` 关闭，或设置环境变量 `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`。

你也可以主动让它记住东西：

> "记住我们用 pnpm 不用 npm"  
> "保存到记忆：API 测试需要本地 Redis"

反过来也行：

> "忘掉之前关于 Redis 的记忆"

## 六、导入机制：@path/to/file

CLAUDE.md 支持 `@path/to/file` 语法导入其他文件：
    
    
    参考 @README 了解项目概况，@package.json 查看可用命令。
    
    # 额外指令
    - Git 工作流 @docs/git-instructions.md

相对路径基于当前文件所在目录解析，支持递归导入，最深 5 层。

### 一个实用场景

如果你用 Git worktree，`CLAUDE.local.md` 只存在于一个 worktree 里。把个人配置放到 home 目录，然后导入：
    
    
    # 个人偏好
    - @~/.claude/my-project-instructions.md

## 七、加载时机：为什么没生效？

加载类型| 说明  
---|---  
**启动时全量加载**|  工作目录往上的所有 CLAUDE.md、CLAUDE.local.md、~/.claude/rules/*.md、Auto Memory 的 MEMORY.md 前 200 行  
**按需加载**|  子目录下的 CLAUDE.md 只在 Claude 读取那个目录的文件时才加载；Auto Memory 的主题文件在需要时读取  
**条件加载**| .claude/rules/ 里有 paths 字段的规则只在匹配文件时生效  
  
这意味着你可以在 monorepo 的每个子包里放自己的 CLAUDE.md，不会一开始就把所有指令都塞进上下文。

## 八、最佳实践

  1. **项目 CLAUDE.md 写团队共识，提交到 Git** \- 构建命令、代码规范、架构决策
  2. **CLAUDE.local.md 写个人偏好，自动 .gitignore** \- 你的测试数据路径、沙箱 URL
  3. **Auto Memory 让它自己跑，定期检查** \- 检查 MEMORY.md 有没有记错的
  4. **不要重复** \- 如果 CLAUDE.md 里已经写了"用 pnpm"，Auto Memory 再记一遍就是噪音
  5. **保持精简** \- MEMORY.md 超过 200 行就会被截断，把细节移到独立文件



## 最后说一句

记忆越多不代表越好。

Claude Code 每次启动都会把这些内容塞进 system prompt，占的是上下文窗口。**写得精准、组织得清楚，比写得多更重要。**

* * *
