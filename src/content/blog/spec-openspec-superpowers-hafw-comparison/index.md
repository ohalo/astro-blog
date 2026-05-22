---
title: "三款AI编程工作流横评：Spec-Kit、OpenSpec、Superpowers选哪个"
publishDate: '2026-04-15'
description: "三款AI编程工作流横评：Spec-Kit、OpenSpec、Superpowers选哪个 - halo的技术博客"
tags:
  - AI观察
language: Chinese
---

AI编程工作流框架Spec-KitOpenSpecSuperpowers

2026年了，AI编程工具早就不是"Copilot写代码补全"那个阶段了。

现在大家拼的不是谁生成的代码更好，而是——**谁的开发流程更能管住AI，让它不乱来** 。

于是冒出了一批"AI编程工作流框架"，目标都是同一个：给AI代理一套结构化的约束，让它按你的规矩干活。

但思路完全不一样，有的讲规范，有的讲技能，有的讲变更。

今天把三个最主流的框架放在一起横评，不吹不黑，说清楚每个的适用场景。

![Spec-Kit Logo](/images/ai-workflow-comparison/speckit-logo.webp)

## 一、Spec-Kit：GitHub官方出品，阶段门控最严格

**背景** ：GitHub官方2025年初推出，Den Delimarsky和John Lam主导开发，82.5K Star。

**核心思路** ：规范不只是文档，而是可执行的——规范能直接生成工作代码。

它的7个阶段是一道道"门"：
    
    
    constitution（项目宪法）→ specify（定义需求）→ clarify（澄清模糊）
    → plan（技术计划）→ tasks（任务分解）→ analyze（一致性检查）→ implement（执行实现）

每道门必须关好才能进下一道。好消息是质量可控，坏消息是不够灵活。

![Spec-Kit Specify CLI 演示](/images/ai-workflow-comparison/speckit-specify-cli.gif)

**技术栈** ：Python（需要uv包管理器）

**AI支持** ：11+工具（Claude Code、Copilot、Cursor等）

**特点** ：

  * constitution.md 定义项目级别的不可违反规则
  * spec.md 描述功能需求，不涉及技术栈（只说what和why）
  * plan.md 和 tasks.md 由AI自动生成
  * 支持模板引擎 + 扩展系统，定制性很强



**适合谁** ：多人协作的企业级项目，需要严格质量管控的团队。

**不适合谁** ：想快速跑通一个Demo的独立开发者（流程太重了）。

![OpenSpec Logo](/images/ai-workflow-comparison/openspec-bg.png)

## 二、OpenSpec：轻量级规范层，变更驱动

**背景** ：Fission-AI团队开发，34.5K Star，TypeScript实现。

**核心思路** ：不做"规范生成代码"，而是做一层轻量规范管理，通过提案→执行→归档的循环，让规范成为活文档。
    
    
    /opsx:propose（创建提案）→ /opsx:continue（逐步实施）→ /opsx:apply（执行）→ /opsx:archive（归档）

每个功能变更是一个独立目录，包含proposal.md、design.md、tasks.md和specs/。完成后归档到知识库，成为项目的"记忆"。

![OpenSpec Dashboard 预览](/images/ai-workflow-comparison/openspec-dashboard.png)

**技术栈** ：TypeScript（npm安装）

**AI支持** ：20+工具（覆盖最广，包括Claude Code、Cursor、Windsurf、Codex等）

**特点** ：

  * 无需API Key和MCP，通用性最强
  * Brownfield优先设计——明确面向现有代码库的渐进式改造
  * 团队协作功能目前还在开发中



**适合谁** ：需要快速迭代的个人开发者或小团队，想要轻量约束但不想被流程绑死。

**不适合谁** ：对代码质量有硬性要求的团队（OpenSpec本身不强制TDD）。

## 三、Superpowers：技能触发式，质量门控最硬

**背景** ：Jesse Vincent（obra）出品，115K Star，社区规模最大。

**核心思路** ：不是规范驱动，而是技能驱动——通过一组可组合的"技能"来约束AI代理行为。核心是Hook触发系统：不是手动调用命令，而是根据上下文自动激活相关技能。
    
    
    写代码前 → 自动触发TDD技能
    写完后 → 自动触发code-review技能

它的工作方式很直觉：

  1. 你一开干，它不急着写代码，先退一步问你"你到底想做什么"
  2. 把需求拆成你能消化的规格，一段一段给你看
  3. 你点头之后，它出一份执行计划——写得连"热情但品味差、不爱写测试的初级工程师"都能照着做
  4. 你说"开搞"，它启动子代理分工干活，自己巡检、review、推进



而且技能都是自动触发的，你不用特意做什么，你的coding agent就自动有了Superpowers。

**技术栈** ：Shell/JavaScript

**AI支持** ：主要针对Claude Code优化（5+工具）

**特点** ：

  * 强制RED-GREEN-REFACTOR TDD循环——三家里唯一强制TDD的
  * 核心技能包括：test-driven-development、systematic-debugging、brainstorming、subagent-driven-development
  * 基于Git Worktrees管理变更状态



**适合谁** ：对测试有硬性要求的团队，想让AI自动走TDD流程。

**不适合谁** ：不习惯TDD的开发者，或者只用Cursor/Windsurf不用Claude Code的人。

## 四、横向对比一览

维度| Spec-Kit| OpenSpec| Superpowers  
---|---|---|---  
**核心范式**|  规范可执行化| 轻量规范层| 技能触发  
**技术栈**|  Python| TypeScript| Shell/JS  
**Star数**|  82.5K| 34.5K| 115K  
**AI工具支持**|  11+| 20+| 5+  
**TDD强制**|  ❌| ❌| ✅  
**团队协作**|  ✅ 企业级| 🚧 开发中| ✅ Discord社区  
**Brownfield支持**|  ✅| ✅ 优先| ✅  
**学习曲线**|  中等| 平缓| 平缓  
**工作流风格**|  阶段门控| 流畅迭代| 技能自动触发  
**哲学**|  结构胜过混乱| 迭代胜过瀑布| 流程胜过猜测  
  
## 五、我的选择建议

说白了就一个问题：**你更担心什么？**

  * **担心AI乱来、不按规范走** → Spec-Kit（阶段门控最严格）
  * **担心流程太重、拖慢速度** → OpenSpec（轻量灵活）
  * **担心代码质量、想要强制TDD** → Superpowers（TDD唯一强制）



如果你非要我给一个推荐——**个人开发者先试OpenSpec** ，上手快、通用性强，用一个月你就知道自己缺什么了。

## 六、最后一句

这些框架本质上解决的是同一个问题：**怎么让AI不只是个写代码的工具，而是个听话的合作伙伴** 。

选哪个不重要，重要的是——**先用一个跑起来** 。

用了一个月，你自然知道它哪里不够用，再换也不迟。最怕的是永远在对比，永远不开始。

* * *

你用的是什么AI编程工作流？有没有踩过什么坑？评论区聊聊。
