---
title: "OpenClaw 有哪些好用的 Skill？装完不知道怎么选，先看完这篇"
publishDate: '2026-04-22'
description: "OpenClaw 有哪些好用的 Skill？装完不知道怎么选，先看完这篇 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

openclawskillsclawhub效率工具

装完 OpenClaw 愣在原地——这是绝大多数人遇到的第一道坎。

大模型是脑子，Gateway 是神经，但 Skill 才是手脚。没有手脚，大脑再强也只是个能聊天的壳。

问题在于：**ClawHub 上有 5700+ 个 Skill，到底该装哪些？** 装错了轻则浪费精力，重则像 2026 年初 ClawHavoc 事件那样，被伪装成热门工具的恶意 Skill 偷走 API Key，境外服务器按小时消耗你的 token 配额。

所以这篇文章不给你一张"装了就是大佬"的清单——**给你一个选 Skill 的框架** 。框架比清单重要，因为 Skill 生态每周都在更新。

## 一、先把"必装"和"危险"说清楚

很多人踩的第一个坑是：把系统内置的技能当成了需要安装的东西。

`system-command`、`file-manager`、`agent-browser` 这三个是 OpenClaw 自带的，不需要装也不能从 ClawHub 安装。装它们只会报 "Skill not found"。

**真正需要装的，是 ClawHub 上的第三方 Skill。**

但装之前必须先装 `skill-vetting`——这不是建议，是必须。

2026 年 1 月底，CVE-2026-25253（CVSS 8.8）漏洞被披露，341 个伪装成热门工具的恶意 Skill 发动了大规模攻击。这些 Skill 名字只差一个字母，图标和正版完全一样，装上去之后会悄悄把你的 API Key 发送到境外服务器。

装任何第三方 Skill 之前，先用 `skill-vetting` 跑一遍。不到 5 分钟。
    
    
    npx clawhub@latest install skill-vetting
    

## 二、按使用场景选：三个配置方案

Skill 不是越多越好。超过 10 个之后，Agent 每次执行任务都要在大量工具中检索，决策变慢，容易相互干扰。

**把数量控制在 10 个以内，按角色选。**

### 方案 A：内容创作者

| Skill | 用途 |  
|-------|------|  
| `skill-vetting` | 必装，安全扫描 |  
| `find-skills` | 不知道装什么？描述需求，它帮你找答案 |  
| `summarize` | 长文章/视频/PDF 一键摘要，40 分钟播客 20 秒出结果 |  
| `humanizer` | 消除 AI 写作痕迹，基于维基百科 24 种 AI 味特征库 |  
| `multi-search-engine` | 集成 17 个搜索引擎（8 个中文 + 9 个全球），国内用户友好 |

这套解决的是**信息过载 + 写作去 AI 味** 两个核心痛点。

### 方案 B：开发者

| Skill | 用途 |  
|-------|------|  
| `skill-vetting` | 必装 |  
| `github` | 手机上远程管代码、查 CI 报错日志 |  
| `proactive-agent` | 把 Agent 从"你让它干才干"变成"它主动找活干" |  
| `self-improving-agent` | 记录每次错误和修正，形成永久记忆，越用越懂你 |  
| `auto-updater-skill` | 每日自动更新所有已装 Skill，配合 cron 静默运行 |

### 方案 C：知识工作者

| Skill | 用途 |  
|-------|------|  
| `skill-vetting` | 必装 |  
| `gog` | 一个 Skill 打通 Gmail、日历、云盘、Docs、Sheets |  
| `ontology` | 构建结构化知识图谱，创建和关联人物、项目、任务等实体 |  
| `obsidian` 或 `notion` | 把笔记库变成 Agent 的知识库 |  
| `tavily-web-search` | 专为 AI Agent 优化的联网搜索，返回干净的结构化数据 |

ClawHub 上 Ontology 的浏览量排第 6，但 15 篇博主文章几乎没人提——这是被集体低估的 Skill。**强烈建议关注。**

## 三、装 Skill 的正确方式

不要看到好文章就直接照抄清单——那些清单里的 Skill 可能已经停更多年、作者换了赛道不再维护、或者针对的 OpenClaw 版本不同。

安装方式：
    
    
    # 方式一：npx（推荐，不需要全局安装）
    npx clawhub@latest install <skill-slug>
    
    # 方式二：全局安装 ClawHub（适合经常用）
    npm i -g clawhub
    clawhub install <skill-slug>
    
    # 方式三：直接把仓库链接发给 OpenClaw
    "请安装这个技能：https://github.com/openclaw/skills/tree/main/skills/steipete/github"
    

## 四、安全这件事说多少遍都不够

ClawHavoc 事件之后，skill-vetting 从"建议装"变成了"必须装"。

装任何 Skill 前：

  1. 用 `skill-vetting` 跑一遍
  2. 看 GitHub 仓库的 Issues 和 PR
  3. 优先选 star 多、更新频繁的
  4. 敏感操作先在小号测试



有博主写过真实经历：装了名字只差一个字母的 Skill，用了两天才发现 API Key 被悄悄提交到了境外服务器，以每小时 1 亿 token 的速度消耗。幸好每周查一次 Key 调用日志，发现异常后立刻吊销了密钥。

**每周查一次 Key 调用日志** ——这句话值一篇完整的安全指南。

## 结尾

回到开头的问题：OpenClaw 有哪些好用的 Skill？

答案是：**取决于你做什么，以及你先保证安全。**

先装 skill-vetting，再用 find-skills 找你要的那个——这才是正确的打开顺序。
