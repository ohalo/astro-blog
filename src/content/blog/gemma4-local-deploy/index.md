---
title: "我在自己的电脑上跑了一个 AI 模型，token 成本为零"
publishDate: '2026-04-10'
description: "我在自己的电脑上跑了一个 AI 模型，token 成本为零 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

你有没有算过，每月花多少 API 费用？ 

我之前用的是云端 API，每天几十个对话，一个月下来小几百块。直到我把 Gemma 4 装到了自己的电脑上——token 成本直接变成零。 

不是偷着用，是谷歌 Apache 2.0 协议开源的，商用都没问题。 

* * *

## 先看你的内存够不够

Gemma 4 有四个版本，都是 4-bit 量化后的内存占用： 

版本| 参数量| 内存需求| 上下文| 适合谁  
---|---|---|---|---  
E2B| 23亿| 4 GB| 128K| 手机/树莓派玩家  
E4B| 45亿| 5.5 GB| 128K| 日常聊天够用  
26B| 252亿(MoE)| 16-18 GB| 256K| 性价比最高  
31B| 307亿| 17-20 GB| 256K| 跑分最猛  
  
我自己用的是 26B。为什么？因为它是混合专家架构，252 亿参数看起来吓人，但每次推理只激活 38 亿——速度接近小模型，质量接近满血版。 

一句话：4 GB 跑 E2B，6 GB 跑 E4B，18 GB 跑 26B，20 GB 以上跑 31B。 

* * *

## Mac 用户：三步搞定

### 第一步：装 Ollama
    
    
    brew install --cask ollama-app
    

或者去 ollama.com 下载。Ollama 把模型下载、推理引擎、API 服务打包成一个 App，装完就能用。 

### 第二步：拉模型
    
    
    open -a Ollama
    ollama run gemma4:26b
    

26B 大约 18 GB，看网速，快的几分钟，慢的等一会儿。 

### 第三步：验证

随便问一句，看到回答就成功了。 
    
    
    ollama ps
    

这个命令会显示 CPU/GPU 分配比例。Apple Silicon 上基本是 14%/86% CPU/GPU，大部分计算跑在 GPU 上，速度比纯 CPU 快得多。 

三步，搞定。 

* * *

## Windows 用户

同理，先装 Ollama： 
    
    
    irm https://ollama.com/install.ps1 | iex
    

然后： 
    
    
    ollama run gemma4:26b
    

有 NVIDIA 显卡的话，Ollama 自动调用 CUDA 加速。RTX 40 系以上的显卡还有个额外福利：Ollama 0.19 新增了 NVFP4 格式，用更少显存跑模型，精度损失很小。 

* * *

## 和 OpenClaw 接起来，token 费用归零

这是我目前最推荐的用法。 

如果你已经在用 OpenClaw（一个开源的 AI 助手框架），部署就更简单了——连终端都不用碰。 

直接给 OpenClaw 发消息： 

  1. "在服务器上安装 Ollama，运行 `curl -fsSL https://ollama.com/install.sh | sh`"
  2. "下载 Gemma 4 26B 模型：`ollama pull gemma4:26b`"
  3. "测试一下：`ollama run gemma4:26b '你好，你是什么模型？'`"



它会帮你装依赖、拉模型、跑测试，全程不用你动手。 

然后把 OpenClaw 的模型后端切到本地，API 端点指向 `localhost:11434`，从此不再需要云端 API。 

**但有个建议** ：小模型更适合端侧场景（手机、树莓派），主力模型还是建议用满血版或者云端大模型。26B 在纯 CPU 服务器上跑起来还是有点勉强，E4B 会快得多。 

* * *

## 常用命令备忘
    
    
    ollama list             # 查看已下载的模型
    ollama ps               # 查看正在运行的模型和内存占用
    ollama run gemma4:26b   # 启动对话
    ollama stop gemma4:26b  # 卸载模型释放内存
    ollama pull gemma4:26b  # 更新到最新版本
    ollama rm gemma4:26b    # 删除模型
    

* * *

## 我的真实感受

跑了几天下来，26B 在我的 M 系列 Mac 上日常对话完全够用，速度也还行。但复杂推理（比如写代码、数学题）还是不如云端的大模型。 

所以我现在的方式是：**日常闲聊用本地 Gemma 4，干活用云端模型。**

这样 API 费用从每月几百降到每月几十，省下来的钱够再买一台树莓派养虾了。 

* * *

_你在本地跑过 AI 模型吗？用的哪个版本？体验怎么样？评论区聊聊。_

### 💬 评论区
