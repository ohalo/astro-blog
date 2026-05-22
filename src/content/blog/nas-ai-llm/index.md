---
title: "用NAS跑本地大模型：Ollama部署DeepSeek/Qwen体验"
publishDate: '2026-04-08'
description: "用NAS跑本地大模型：Ollama部署DeepSeek/Qwen体验 - halo的技术博客"
tags:
  - 硬件数码
language: Chinese
---

NASAI大模型OllamaDeepSeek本地LLM

ChatGPT 很强大，但每次对话都把数据送到云端，你真的放心吗？把大模型跑在本地 NAS 上，数据不出局域网，隐私问题彻底解决。而且 NAS 24 小时开着，随时唤醒，比电脑跑更方便。聊聊我在飞牛 OS 上跑 Ollama + DeepSeek 的体验。

## 一、为什么要在NAS上跑AI？

![AI大脑](/images/nas-ai-llm/ai-brain.jpg)

这个问题我问过自己很多次，云端 API 这么方便，本地跑图什么？

**隐私是最大的理由** 。我工作上有些敏感文档需要 AI 辅助处理，送到云端 API 就是把数据交给第三方。虽然各厂商都有数据保护承诺，但永远比不上数据在自己硬盘上踏实。

**离线可用** 。NAS 跑的是本地模型，不需要联网。断网的时候也能用 AI（虽然慢一点）。

**成本** 。Ollama 完全免费，没有 API 调用费用。虽然有电费，但 NAS 本来就 24 小时开着，边际成本接近零。

## 二、Ollama安装与部署

![LLM聊天](/images/nas-ai-llm/llm-chat.jpg)

Ollama 是目前最简单的大模型运行工具，Docker 一键安装：
    
    
    docker run -d \
      --name ollama \
      --restart unless-stopped \
      -p 11434:11434 \
      -v ollama:/root/.ollama \
      -v /path/to/models:/root/.ollama/models \
      ollama/ollama:latest

飞牛 OS 直接在 Docker 管理界面里图形化安装更简单，不需要记命令。

装完之后，浏览器打开 http://NAS_IP:11434 可以看到 Ollama 的 Web UI，或者直接用 API：
    
    
    curl http://localhost:11434/api/generate -d '{
      "model": "deepseek-r1:7b",
      "prompt": "用一句话解释量子计算"
    }'

## 三、模型选择与内存门槛

不是所有模型都能跑在你的 NAS 上，内存是关键瓶颈。

模型| 参数量| 最低内存| 适用场景  
---|---|---|---  
Qwen2.5-3B| 30亿| 4GB| N100/J4125入门  
Qwen2.5-7B| 70亿| 8GB| 均衡性能/推荐  
DeepSeek-R1-Distill-Qwen-7B| 70亿| 8GB| 推理能力强/推荐  
Llama-3.1-8B| 80亿| 8GB| 英文为主  
DeepSeek-R1-Distill-Qwen-14B| 140亿| 16GB| 更强推理/内存要求高  
Qwen2.5-72B（量化）| 720亿| 32GB| NAS基本跑不了  
  
我的推荐：**DeepSeek-R1-Distill-Qwen-7B** 。推理能力强，中文支持好，内存门槛大多数 NAS 都能满足。Qwen2.5-7B 也不错，对话流畅性更好一点。

## 四、下载模型
    
    
    ollama pull deepseek-r1:7b

模型文件大概 4GB 左右（7B Q4 量化版），下载时间取决于网速。用 aria2 或者 NAS 下载工具提前下载更省时间。

下载完成后，用命令验证是否正常运行：
    
    
    ollama run deepseek-r1:7b "你好，请自我介绍一下"

## 五、性能实测：响应速度怎么样？

我用 J4125（8GB RAM）+ Ollama 跑了 DeepSeek-R1-Distill-Qwen-7B：

  * **首次响应时间** ：约 3-5 秒开始输出。原因是模型需要加载到内存，首次推理有预热。
  * **生成速度** ：约 8-15 tokens/秒。足够实时对话使用。
  * **长对话** ：超过 20 轮对话后，速度会逐渐下降（上下文窗口被填满）。Ollama 支持上下文截断，用 `/reset` 可以清空上下文。



对比云端 API（GPT-4o Mini）：云端响应更快（50+ tokens/秒），但本地模型的差距其实没有想象的大——日常对话中 8-15 tokens/秒完全可接受。

## 六、实用场景

跑了一段时间后，我发现这几个场景最实用：

  * **写作辅助** ：让 AI 帮忙润色文章、翻译、总结长文。不需要联网，隐私有保障。
  * **代码辅助** ：DeepSeek 的代码能力很强，帮忙写脚本、改 bug 效果不错。
  * **本地知识库** ：配合 Ollama + LangChain，可以把 NAS 上的文档作为知识库，让 AI 回答关于你自己数据的问题。
  * **24小时客服** ：接一个简单的 Web UI，比如 OpenWebUI，当作团队内部的 AI 助手用。



## 七、vs ChatGPT：本地模型能打得过吗？

老实说——**打不过 GPT-4o，也打不过 Claude 3.5** 。本地 7B 模型的能力大概相当于 GPT-3.5 水平，有时候甚至不如。

但它有三个不可替代的优势：完全离线、绝对隐私、零 API 费用。对于隐私敏感的场景，这个优势是压倒性的。

## 八、电费和长期待机

很多人担心跑 AI 的电费问题。我的 J4125 NAS 跑 DeepSeek 7B 时，CPU 占用大概 30-50%，功率增加约 15-20W。每天多耗 0.4 度电左右，每月多花 2-3 块钱。几乎可以忽略不计。

* * *

_数据来源：Ollama 官方文档，DeepSeek 官方论文，Hugging Face 模型卡片，个人实测（2026年4月）_
