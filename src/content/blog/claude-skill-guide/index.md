---
title: "Anthropic 官方发布：Claude Skill 完全指南（我的解读）"
publishDate: '2024-01-01'
description: "Anthropic 官方发布：Claude Skill 完全指南（我的解读） - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

ClaudeSkillMCPAI工具

前两天 Anthropic 放了一份完整的 Skill 构建指南，我看完觉得：**这东西比我想的更有用。**

## 先说 Skill 是什么

一句话：**Skill = 教 Claude 一次，处处生效。**

你不用每次对话都重复"我想要什么格式"、"按什么流程来"、"注意什么规范"。写进 Skill 里，Claude 自动按你的规矩办。

就像你带新人，不用每次都从头教——给他一份 SOP，他自己照着做。

## 什么时候该写个 Skill？

官方给了三类场景：

**1\. 有固定产出物的**  
写文档、做 PPT、生成前端、出报告——每次格式差不多，质量要求一致。

**2\. 有固定流程的**  
调研方法、代码审查流程、多步审批——步骤固定，但每次都要重复说。

**3\. 接了 MCP 但不知道怎么用的**  
连上了 Notion、Linear、Figma，但每次都要从头告诉 Claude 怎么配合用——烦。

我自己的经验：**如果一个流程你跟 Claude 解释过三次以上，就该写 Skill 了。**

## 技术门槛：其实很低

一个 Skill 就是一个文件夹，里面必须有一个 `SKILL.md`。
    
    
    your-skill/
    ├── SKILL.md      # 必需，主说明文件
    ├── scripts/      # 可选，放脚本
    ├── references/   # 可选，放详细文档
    └── assets/       # 可选，放模板

`SKILL.md` 开头是 YAML frontmatter，告诉 Claude "什么时候用"：
    
    
    ---
    name: my-skill
    description: 做什么。在用户说"xxx"或"yyy"时使用。
    ---

然后是正文，写清楚步骤、示例、常见错误。

**就这么简单。**

## 一个容易踩的坑：description 写不好

官方强调：description 必须同时写清"做什么"和"什么时候用"。

❌ 太泛：`"帮助处理项目"`  
❌ 缺触发：`"创建复杂文档系统"`  
✅ 正确：`"分析 Figma 设计并生成开发交接文档；在用户上传 .fig、问「设计规格」「组件文档」时使用"`

我理解了：**Claude 是靠 description 判断要不要加载这个 Skill 的。** 写得太泛，就会乱触发；写得太窄，该触发时不触发。

## 和 MCP 的关系

很多人搞混了：

  * **MCP** ：提供"连接"——把 Claude 连到你的工具（Notion、Linear、Figma）
  * **Skill** ：提供"知识"——教 Claude 怎么用好这些工具



光有 MCP，Claude 知道"能调什么 API"，但不知道"按什么顺序调"、"调完怎么处理"。Skill 补的就是这块。

## 测试：三种都要做

  1. **触发测试** ：该触发时触发，不该触发时不触发
  2. **功能测试** ：输出正确、API 成功、错误处理到位
  3. **效果对比** ：和"不用 Skill"比，消息轮数、失败调用数



官方说：**先在一个难任务上迭代到稳定，再扩展用例。** 别一上来就想覆盖所有场景。

## 分发：已经支持组织级部署

2025年12月起，管理员可以工作区级部署 Skill，支持自动更新和集中管理。

如果你想开源：放 GitHub，写清楚 README，说明依赖和用法。

## 我的看法

看完这份指南，我最大的感受是：**Skill 的价值被低估了。**

大多数人还在把 Claude 当"聊天工具"用——每次对话从头解释需求、纠正输出、重复流程。

Skill 把这些"一次性对话"变成了"可复用的能力"。你教一次，以后就不用再教了。

**15-30 分钟搭一个 Skill，换来以后无数次省心。** 这笔账很划算。

* * *

官方 PDF 下载：[The Complete Guide to Building Skills for Claude](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf?hsLang=en)
