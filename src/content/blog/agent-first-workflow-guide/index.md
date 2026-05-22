---
title: "从零开始理解 Agent-First 工作流：不是给 AI 套流程，是让 AI 自己跑流程"
publishDate: '2026-04-15'
description: "从零开始理解 Agent-First 工作流：不是给 AI 套流程，是让 AI 自己跑流程 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

AI Agent工作流自动化Agent-FirstLangChainAutoGen

你用 Zapier 或者 n8n 做过自动化吗？

做了几十个 "如果-那么" 规则，接API、写过滤器、配错误处理。结果呢？用户稍微说点规则外的话，流程就卡住了，你只能在后台干瞪眼，最后还是自己上去手动处理。

这不是你的自动化方案不够好，是**思路从根本上就跑偏了** 。

传统自动化的逻辑是 "把现有流程用代码实现"。但 AI 时代有另一种思路：**不是让自动化适应现有流程，而是让流程适应 AI 的能力** 。

这两种思路有本质区别。

## 两种完全不同的设计逻辑

用一张表说清楚：

维度 | 传统自动化 | Agent-First 自动化  
---|---|---  
核心逻辑 | 固定规则 | 目标和约束  
处理未知情况 | 需要人工介入 | AI 自主决策  
异常处理 | 预设分支 → 死路 | 智能判断 → 继续或升级  
流程特点 | 线性 | 动态  
执行中心 | 人 | AI Agent  
  
![传统自动化 vs Agent-First 对比图](/images/agent-first-workflow/comparison.png)

举个例子更清楚。

**传统方式** 处理客户工单：
    
    
    收到工单 → 分类（规则匹配）→ 分配对应团队 → 人工回复 → 标记完成

这套流程在规则内没问题。但用户说 "我上周下的单还没到，帮我催一下" —— 工单分类识别不了这个 intent，预设的分支走不通，人工介入。

**Agent-First 方式** ：
    
    
    收到工单 → Agent分析意图 → 自主查订单状态 → 生成回复 → 必要时升级人工 → 记录经验

区别在于：Agent 不是在"执行预设步骤"，而是"理解目标并自主达成"。

* * *

## 踩过坑才明白：为什么不能直接套用现有流程

我自己试过把 n8n 的流程直接接 GPT API，结果体验很差。AI 被束缚在僵化的流程里——"先分类，再查数据库，再回复"，每一步都要等上一步完成，AI 的能力根本没发挥出来。

问题出在哪？**流程是给人设计的，不是给 AI 设计的** 。

传统流程有几个典型问题：

  * 到处都是"需要人工确认"的节点
  * 不同系统之间数据不互通，AI 拿不到完整上下文
  * 流程是被动的，有输入才触发，不主动优化
  * 没有学习机制，同样的错误会重复发生



这些问题在有人参与时是正常的——人需要节点来控制风险。但 AI 不需要这种控制感，它需要的是**目标** 和**边界** 。

## 实战：用代码搭一个 Agent-First 客户支持系统

光说概念太虚，我拿一个具体例子来说明怎么做。

### 第一步：先定义目标和约束，不是步骤
    
    
    # agent-config.yaml
    agent:
      name: SupportAgent
      goal: "解决客户问题，提升满意度"
    
    constraints:
      - 不能承诺退款超过 $500
      - 不能访问用户支付密码
      - 复杂问题必须升级人工
      - 回复必须友好专业
    
    capabilities:
      - 查询订单状态
      - 处理退换货请求
      - 解答产品问题
      - 收集用户反馈
    
    tools:
      - order_lookup
      - refund_processor
      - knowledge_base
      - ticket_system

这一步我踩过最大的坑是：**约束写得太模糊** 。比如"不能泄露用户隐私"，这种描述 AI 不知道怎么执行。改成"回复中不能包含用户的完整身份证号和银行卡号"，AI 才能真的做到。

### 第二步：让 Agent 的"感知-思考-行动"跑起来

这是 Agent 的核心循环，三个阶段：

![Agent 核心循环：感知-思考-行动](/images/agent-first-workflow/agent-loop.png)
    
    
    async def process_ticket(self, ticket: dict) -> dict:
        # 1. 感知：理解用户问题
        intent = await self._understand_intent(ticket['content'])
    
        # 2. 思考：规划解决方案
        plan = await self._create_plan(intent, ticket)
    
        # 3. 行动：执行计划
        result = await self._execute_plan(plan, ticket)
    
        # 4. 学习：记录经验
        await self._learn(ticket, result)
    
        return result

实际用的时候，我发现 `_understand_intent` 的质量直接决定后续所有步骤的效果。Prompt 写得好，intent 识别准确率能到 90% 以上；Prompt 写得敷衍，经常出现 intent 误判。

### 第三步：设计好工具接口

这一步很重要，接口设计得好不好，直接影响 Agent 能不能正确执行。
    
    
    async def check_refund_eligibility(order_id: str) -> dict:
        """
        Returns:
          {
            "eligible": bool,
            "reason": str,  # 如果不符合，说明原因
            "max_amount": float  # 最大可退款金额
          }
        """
        ...

接口返回要包含 **eligible** （能不能做）、**reason** （为什么）和 **max_amount** （能做的上限），不能只返回一个布尔值。Agent 需要知道失败的原因，才能决定下一步怎么处理。

### 第四步：保留人工介入点

Agent 不是万能的，关键时刻必须能升级人工。
    
    
    # 检查是否需要升级人工
    if self._should_escalate(result):
        await self.slack.send_message(
            channel='#support-escalations',
            text=f"工单 #{ticket['id']} 需要人工处理，原因: {result.get('reason')}"
        )
        await self.ticket_system.reply(
            ticket['id'],
            "您的问题已转交专业客服，将在1小时内回复您。"
        )

我自己跑这套系统时，升级率大概在 15-20%，属于正常范围。如果超过 30%，说明 intent 理解这块需要优化了。

## 工具生态：不是选框架，是选场景

文章里提到了几个框架，我实际用过其中几个，说一下感受：

  * **LangChain** ：文档最全，社区最大，但上手曲线比较陡。我用下来感觉它的 Agent 模块比较成熟，工具调用这块封装得很好。
  * **AutoGen** ：微软出的，多 Agent 协作场景很强。如果你需要多个 Agent 互相配合（比如一个写代码，一个审查），AutoGen 比 LangChain 更直接。
  * **Temporal** ：适合工作流可靠性要求高的场景。比如支付流程、订单处理这种不能出错的操作，Temporal 的容错机制很强。



![LangChain](/images/agent-first-workflow/langchain-logo.png) ![Prefect](/images/agent-first-workflow/prefect-logo.png) ![Temporal](/images/agent-first-workflow/temporal-logo.png)

如果只是快速验证想法，LangChain + 一个好的 Prompt 就够了。等需求复杂了，再按场景引入其他工具。

## 我自己用下来的真实感受

基于这套方法论跑过几个项目，说几个实在的：

**确实好用的场景：**

  * 客服工单处理（尤其是高频、规则相对清晰的场景）
  * 数据录入和整理（从邮件/文档里提取信息）
  * 多系统协调（比如订单状态同步到多个平台）



**不适合的场景：**

  * 需要强合规约束的操作（比如金融交易、法律文件）
  * 流程经常变化的场景（每次改需求都要重新配 Agent，成本反而高）
  * 需要 100% 可解释性的场景（Agent 的决策链路有时候不太好追溯）



**核心体会** ：Agent-First 不是让 AI 更快，是让 AI **更可预测** 。流程从"你告诉 AI 每一步怎么做"变成"你告诉 AI 要达成什么，它自己决定怎么做"。后者灵活得多，但前提是你把约束写清楚。

## 最后

传统自动化到 Agent-First，是一次思维方式的转变，不是单纯的技术升级。

你现在的自动化方案，遇到过哪些"规则覆盖不到"的场景？有没有试过让 AI 自主决策而不是按预设步骤走？

评论区说说。
