---
title: "OpenClaw 默认 AGENTS.md：装了不会配？看这一篇就够了"
publishDate: '2024-01-01'
description: "OpenClaw 默认 AGENTS.md：装了不会配？看这一篇就够了 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

刚装完 OpenClaw，打开设置界面，提示你配置 AGENTS.md。 

这时候有两个选择：直接用默认模板，或者自己写一套规则。 

我自己踩过坑——一开始觉得默认模板是"随便给的参考"，自己写了一套"更专业"的配置。结果没过两天就发现：文件散落在各处、记忆系统完全没用起来、想加新功能不知道往哪加。 

后来老老实实把默认模板研究了一遍，才发现它不是随便给的——每一条都是有原因的。 

* * *

## AGENTS.md 到底是什么

简单说，这是一个给 AI 助手看的"说明书"。你告诉它"你是谁""你的工作区在哪""遇到问题按什么流程处理"，它照着执行。 

OpenClaw 的默认模板在这里：`~/.openclaw/workspace/AGENTS.md`

装完应用之后，这个文件不一定存在，需要手动创建： 
    
    
    mkdir -p ~/.openclaw/workspace
    

然后把模板复制进去： 
    
    
    cp docs/reference/templates/AGENTS.md ~/.openclaw/workspace/AGENTS.md
    

注意路径是 `~/.openclaw/workspace`，不是 `~/.qclaw/workspace`——这两个不一样，我之前就搞过错。 

* * *

## 默认模板里最值得了解的三个模块

### 1\. 工作区配置

模板默认工作区是 `~/.openclaw/workspace`，但这个可以改。如果你想让不同项目用不同的配置，可以创建多个工作区目录，然后在设置里切换： 
    
    
    {
      agents: { defaults: { workspace: "~/.openclaw/workspace" } }
    }
    

这里有个细节：工作区目录里除了 AGENTS.md，还需要 SOUL.md（定义 AI 的性格和语气）和 TOOLS.md（存放环境相关的工具配置）。三个文件各司其职，不是随便堆一起的。 

### 2\. Memory 系统（我踩坑最多的地方）

默认模板里有一套记忆系统，分两层： 

  * **每日日志** ：`memory/YYYY-MM-DD.md`，记录当天发生的事
  * **长期记忆** ：`MEMORY.md`，提炼持久的事实和偏好



刚用的时候我以为这套系统是"自动的"，AI 会自动记住一切。后来发现不对——它只在会话开始时读取这些文件，内容需要我自己维护。 

所以现在我养成了一个习惯：**重要决策做完就写进 MEMORY.md** ，不然下次开会话它完全不记得。 

### 3\. Skills 系统

Skills 是 OpenClaw 的扩展机制，每个 Skill 是一组工具和使用说明的集合。 

Skill| 用途  
---|---  
mcporter| 管理外部工具后端  
imsg| 发 iMessage  
wacli| WhatsApp 操作  
gog| Google 全家桶  
spotify-player| 控制 Spotify  
  
不需要全部开启，用到什么开什么。默认模板的 Skills 列表挺全的，但实际用的时候我建议先装几个最常用的，其他等有需求再加——不然装一堆用不上的技能，维护成本就上去了。 

* * *

## 默认模板够不够用

说实话，**对大多数用户来说，默认模板够用了** 。 

安全规则、Memory 规范、会话启动流程，这些都已经配好了。直接用，边用边改，比一开始自己写一整套规则要省心得多。 

除非你有特殊需求——比如要给不同项目配不同的工作区、或者需要深度定制 AI 的行为——否则不建议从零重写。先把默认模板跑通，理解每一条规则的原因，再根据自己的情况删改。 

* * *

## 一个我建议现在就去做的操作

打开终端，运行： 
    
    
    ls ~/.openclaw/workspace/
    

看看这个目录里有没有 AGENTS.md、SOUL.md、TOOLS.md 这三个文件。 

如果只有一个空的目录，那就按上面的步骤把模板复制进去，这是让 OpenClaw 真正跑起来的第一步。 

默认模板不是随便给的参考，是一套经过验证的基础配置。先用起来，再慢慢调，比自己重写要靠谱得多。 

* * *

_你用 OpenClaw 的过程中踩过哪些坑？欢迎评论区说说。_

### 💬 评论区
