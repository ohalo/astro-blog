---
title: "我在公司搭了一套私有AI，同事们以为我花了几十万"
publishDate: '2024-01-01'
description: "我在公司搭了一套私有AI，同事们以为我花了几十万 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

_一行命令启动大模型，三步封装成API服务，最后用Docker打包部署——整个流程我用了不到半天。_

* * *

## 前言

上周公司要做一个内部知识库问答系统，需求很简单：**把公司的文档喂给大模型，让员工直接问问题就行** 。

但问题来了——数据涉及客户隐私和商业机密，不可能调OpenAI或者百度的API。

**必须在本地部署。**

我去网上搜了一圈，发现要么是纯理论科普，要么是上来就甩几十行代码的"极客教程"，小白看完直接劝退。

所以我决定自己踩一遍坑，然后把整个过程用大白话写出来。

先说结论：**一台8G显存的显卡 + 半天时间，就能跑起来一个生产级的本地大模型服务。**

* * *

## 一、先搞清楚：你要哪种部署方式？

很多人一上来就被各种名词搞晕了——llama.cpp、Ollama、vLLM、transformers、TGI……

其实就三种路线：

部署方式| 适合场景| 核心工具| 上手难度  
---|---|---|---  
**本地直接跑**|  个人开发、测试、低并发| Ollama / llama.cpp| ⭐ 一行命令  
**API服务化**|  团队共享、对接业务系统| vLLM / FastAPI| ⭐⭐ 需要写点代码  
**容器化部署**|  生产环境、多服务协同| Docker + docker-compose| ⭐⭐⭐ 需要Docker基础  
  
**我的建议** ：先用 Ollama 10分钟跑起来感受一下，再决定要不要上 vLLM。

别一上来就搞容器化，你会怀疑人生的。

* * *

## 二、方案一：Ollama——一行命令跑大模型

### 为什么推荐Ollama？

我试过transformers原生加载、试过llama.cpp编译、试过text-generation-webui……

最后发现 **Ollama 是最省心的方案** ，没有之一。

它的设计哲学就一个字：**简** 。

  * 安装：一条命令
  * 下载模型：一条命令
  * 启动模型：一条命令
  * 提供API：自动的



### 安装（macOS / Linux）
    
    
    # macOS（推荐用Homebrew）
    brew install ollama
    
    # Linux（一键脚本）
    curl -fsSL https://ollama.com/install.sh | sh
    

### 跑一个模型试试
    
    
    # 下载并启动 Qwen2.5 7B（约4.7GB，首次下载需要几分钟）
    ollama run qwen2.5:7b
    
    # 进入交互模式，直接聊天
    >>> 你好，介绍一下你自己
    

就这？**就这。**

不需要配Python环境，不需要装CUDA，不需要下载GGUF文件手动加载。

### 几个我踩过的坑

**坑1：Mac M芯片的显存共享**

M系列芯片的内存是CPU和GPU共享的。8G内存的MacBook能跑7B模型，但会比较吃力。16G以上比较舒服。

**坑2：模型别贪大**

模型大小| 最低显存| 推荐用途  
---|---|---  
1.5B~3B| 4GB| 轻量问答、分类  
7B~8B| 8GB| 通用对话、代码生成  
14B| 16GB| 复杂推理、长文本  
72B| 4×24GB| 企业级生产（一般用不到）  
  
**坑3：默认模型目录太大**

模型默认存在 `~/.ollama/models/`，如果你系统盘空间紧张，可以改：
    
    
    # 设置环境变量指定存储位置
    export OLLAMA_MODELS=/data/ollama/models
    

### Ollama 的隐藏福利：自动提供API

Ollama 启动后，自动在 `http://localhost:11434` 提供OpenAI兼容的API。
    
    
    # 直接用OpenAI SDK调用本地模型
    from openai import OpenAI
    
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # 随便填
    )
    
    response = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "解释一下什么是K8s"}],
    )
    print(response.choices[0].message.content)
    

**这意味着你的所有现有代码，只需要改一行`base_url` 就能从OpenAI切到本地模型。**

这个设计太优雅了。

* * *

## 三、方案二：vLLM——高性能推理服务

### Ollama不够用吗？

对大多数场景来说，Ollama已经够用了。但如果你有以下需求：

  * **高并发** （几十人同时访问）
  * **低延迟** （响应时间要小于1秒）
  * **批量推理** （一次处理大量请求）
  * **GPU利用率优化** （榨干每一滴显存）



那你就需要 vLLM。

### vLLM 的核心技术：PagedAttention

不扯理论，说人话就是：**vLLM 把显存当硬盘用，需要哪部分就加载哪部分，而不是一股脑全塞进去。**

效果：同样的显卡，vLLM 能多跑 2-4倍的并发请求。

### 安装和启动
    
    
    # 安装（需要 CUDA 11.8+）
    pip install vllm
    
    # 启动推理服务（OpenAI兼容模式）
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
        --served-model-name qwen-14b \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization 0.90
    

### 客户端调用（和Ollama一样用OpenAI SDK）
    
    
    from openai import OpenAI
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )
    
    response = client.chat.completions.create(
        model="qwen-14b",
        messages=[
            {"role": "system", "content": "你是一位资深Python工程师"},
            {"role": "user", "content": "如何优化asyncio的并发性能？"},
        ],
        max_tokens=2048,
    )
    

### vLLM vs Ollama 怎么选？

对比维度| Ollama| vLLM  
---|---|---  
安装难度| 一条命令| pip install + CUDA  
并发能力| 低（单用户）| 高（几十并发）  
显存利用| 一般| PagedAttention优化  
模型切换| 秒切| 需重启  
适合场景| 个人开发、测试| 生产环境、团队服务  
  
**我的选择** ：开发调试用 Ollama，线上部署用 vLLM。

* * *

## 四、方案三：Docker 封装——从"能跑"到"能上线"

### 为什么需要Docker？

当你需要把模型服务部署到服务器上时，会遇到这些问题：

  * 服务器没有Python环境怎么办？
  * CUDA版本不对怎么办？
  * 依赖冲突怎么办？
  * 怎么快速扩容？



Docker 解决了所有这些问题：**把运行环境整个打包，到哪都能跑。**

### Dockerfile（多阶段构建）
    
    
    # 构建阶段：安装依赖
    FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
    
    ENV DEBIAN_FRONTEND=noninteractive
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        && rm -rf /var/lib/apt/lists/*
    
    RUN python3.11 -m venv /opt/venv
    ENV PATH="/opt/venv/bin:$PATH"
    
    COPY requirements.txt /tmp/
    RUN pip install --no-cache-dir -r /tmp/requirements.txt
    
    # 运行阶段：精简镜像
    FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
    
    RUN apt-get update && apt-get install -y --no-install-recommends python3.11 \
        && rm -rf /var/lib/apt/lists/*
    
    COPY --from=builder /opt/venv /opt/venv
    ENV PATH="/opt/venv/bin:$PATH"
    
    WORKDIR /app
    COPY api_server.py .
    EXPOSE 8000
    
    CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
    

**关键技巧：多阶段构建。** 构建阶段装依赖（镜像大），运行阶段只拷贝编译好的包（镜像小）。最终镜像能从 8GB 缩到 3GB。

### docker-compose：一键启动
    
    
    version: "3.9"
    services:
      llm-api:
        build: .
        container_name: llm-api-server
        ports:
          - "8000:8000"
        volumes:
          - ~/.cache/huggingface:/root/.cache/huggingface  # 模型缓存
        environment:
          - NVIDIA_VISIBLE_DEVICES=all
          - MODEL_ID=Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: all
                  capabilities: [gpu]
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
          interval: 30s
          timeout: 10s
          retries: 3
    
      redis:
        image: redis:7-alpine
        ports:
          - "6379:6379"
        restart: unless-stopped
    
    
    
    # 启动
    docker-compose up -d
    
    # 测试
    curl -X POST http://localhost:8000/v1/chat \
      -H "Content-Type: application/json" \
      -d '{"prompt": "解释Docker的多阶段构建", "max_tokens": 512}'
    

* * *

## 五、性能调优：我把速度从 3秒/条 优化到 0.5秒/条

这些参数是我踩了一周坑才摸清的：

参数| 说明| 推荐值  
---|---|---  
`gpu_memory_utilization`| 显存使用上限| 0.85~0.95（越高越快，但要留buffer）  
`max_model_len`| 最大上下文长度| 按需设置（越长越占显存）  
`tensor_parallel_size`| 多GPU并行数| 匹配物理GPU数量  
`quantization`| 量化方式| GPTQ-Int4 / AWQ（精度换速度）  
`enforce_eager`| 禁用CUDA Graph| 调试时开启，生产环境关闭  
  
**最大的坑** ：`max_model_len` 设太大会直接OOM。14B模型配合GPTQ-Int4量化，在24GB显存上，`max_model_len` 设4096刚刚好。

* * *

## 六、我的部署路径总结
    
    
    第一步（10分钟）：Ollama 本地跑起来
             ↓ 确认模型效果OK
    第二步（1小时）：vLLM 搭建API服务
             ↓ 确认并发性能OK
    第三步（2小时）：Docker 封装 + docker-compose
             ↓ 部署到服务器
    第四步（持续）：性能调优 + 监控
    

**别跳步。** 我见过太多人一上来就搞Docker + K8s + 微服务，结果一个模型都没跑通过。

先跑通，再优化。

* * *

## 七、2026年模型推荐

基于我实际测试的效果，给个选型参考：

场景| 推荐模型| 大小| 特点  
---|---|---|---  
中文通用| Qwen2.5-7B-Instruct| 7B| 中文最强开源，速度快  
代码生成| DeepSeek-Coder-V2-Lite| 16B| 代码能力突出  
长文本| Yi-1.5-9B-Chat| 9B| 支持200K上下文  
低资源| Phi-3-mini| 3.8B| 4G显存就能跑  
高性能| Qwen2.5-72B-Instruct-GPTQ| 72B| 量化后24GB可跑  
  
* * *

## 写在最后

2026年了，本地部署大模型已经不是什么高门槛的事。

**真正难的不是技术，是选型。** 选对工具、选对模型、选对路径，半天搞定；选错了，可能折腾一周还在踩坑。

希望这篇文章能帮你在踩坑之前，先看到路。

* * *

_觉得有用的话，点个赞，收藏起来。部署过程中遇到问题可以在评论区交流，我会尽量回复 👍_

### 💬 评论区
