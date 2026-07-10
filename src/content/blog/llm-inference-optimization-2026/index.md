---
title: "大模型推理优化：从vLLM到SGLang的技术演进与实践"
publishDate: '2026-07-10'
description: "大模型推理优化：从vLLM到SGLang的技术演进与实践 - halo的技术博客"
tags:
 - AI工具
language: Chinese
---

2026年，大模型推理已经从"能不能跑起来"进入了"如何做到毫秒级延迟、万级并发"的阶段。这篇文章梳理一下当前主流推理框架的技术演进路线，以及在实际部署中的选择思路。

## 从一次请求说起

假设你接到一个任务：部署一个70B参数的模型，要求首Token延迟不超过500ms，单卡能撑住500 tokens/s的吞吐。这看起来不难，但实际上涉及一整条技术链的选择。

传统的做法是用HuggingFace Transformers直接加载模型做推理。这种方式的问题在于，标准的自回归生成是**逐Token串行计算**的——GPU的算力利用率（MFU）往往只有5%-15%，大量的时间花在了内存搬运和等待上。

## vLLM：PagedAttention 的革命

vLLM在2023年提出了**PagedAttention**，这个想法的核心非常直观：把KV Cache的管理从连续内存分配改为分页管理，就像操作系统的虚拟内存一样。

为什么要这么做？因为不同请求的长度不同，如果预分配连续的KV Cache，碎片化会非常严重——实际利用率可能只有20%-40%。PagedAttention通过页表映射，让内存利用率提升到接近100%，对应的吞吐量提升了2-4倍。

2026年的vLLM已经迭代到了v0.8.x，新增了以下关键特性：

- **Prefix Caching**：当多个请求共享相同的系统提示词时，KV Cache可以直接复用，省去了大量重复计算
- **Chunked Prefill**：把长prompt的prefill阶段拆分，与decode阶段混合调度，避免长prompt阻塞短请求
- **FP8量化推理**：在H100/H200上原生支持FP8，吞吐再提升50%以上

```bash
# vLLM 部署示例
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --enable-chunked-prefill
```

![GPU服务器集群](/images/llm-inference-optimization-2026/gpu-cluster.jpg)

## SGLang：从调度器视角的重新设计

如果说vLLM解决了内存管理问题，那SGLang解决的就是调度问题。

SGLang的核心是**RadixAttention**——一种基于基数树的KV Cache重用机制。与vLLM的Prefix Caching（按完整前缀匹配）不同，RadixAttention可以在任意公共前缀上重用Cache。这在多轮对话、Few-shot prompting等场景下有显著优势。

SGLang的另一大创新是**结构化生成**。通过约束解码（constrained decoding），SGLang可以保证模型输出符合预定义的JSON Schema、正则表达式或语法规则。这在Agent调用、结构化数据提取等场景中非常关键——不需要再靠prompt engineering来"祈祷"模型输出正确的JSON。

## TensorRT-LLM：英伟达的官方答案

如果说vLLM和SGLang是软件优化，那TensorRT-LLM就是**硬件级优化**的极致。

TensorRT-LLM通过图编译优化、算子融合、kernel自动调优等手段，在单卡推理场景下通常能比vLLM快20%-40%。代价是部署流程更复杂——需要预先编译引擎文件，模型格式转换也有不少坑。

2026年，TensorRT-LLM的**In-flight Batching**（连续批处理）已经相当成熟，可以在不中断正在进行的推理的情况下动态加入新请求，大幅降低排队延迟。

## 选择框架的实际思路

实际选型不是看Benchmark谁分高，而是看业务场景：

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| 快速原型/灵活部署 | vLLM | 生态最成熟，兼容OpenAI API |
| 多轮对话/Agent调用 | SGLang | RadixAttention + 结构化生成 |
| 极致性能/单模型长期服务 | TensorRT-LLM | 硬件级优化，延迟最低 |
| 混合负载（训练+推理） | vLLM | 与主流框架兼容性好 |

![推理框架对比](/images/llm-inference-optimization-2026/inference-frameworks.jpg)

## 推理优化的未来方向

展望2026下半年，有几个方向值得关注：

**Speculative Decoding**（投机解码）。用小模型做"草稿"生成，大模型做并行验证，在不损失质量的前提下提升2-3倍生成速度。Google的Gemma系列已经原生支持。

**Disaggregated Prefill-Decode**（分离式预填充）。把Prefill（计算密集）和Decode（内存密集）部署在不同的GPU节点上，各自独立扩缩容。Anthropic和DeepSeek都在大规模使用这种架构。

**MoE推理优化**。随着Mixtral、DeepSeek-V3等MoE模型成为主流，如何高效调度专家网络、减少专家间的通信开销，正在成为推理优化的新课题。

说到底，推理优化本质是在**延迟**、**吞吐**和**成本**三者之间找平衡。没有银弹，只有合不合适。
