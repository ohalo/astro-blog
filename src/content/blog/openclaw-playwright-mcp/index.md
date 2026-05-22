---
title: "我给 OpenClaw 装了个浏览器，现在它能自己上网了"
publishDate: '2026-04-07'
description: "我给 OpenClaw 装了个浏览器，现在它能自己上网了 - halo的技术博客"
tags:
  - AI工具
language: Chinese
---

用 AI agent 最大的痛点是什么？ 

**它看不到你的屏幕，打不开你的网页，填不了你的表单。**

你想让它帮你在知乎搜个话题、截个图、抓个数据，它只能告诉你"我打不开这个链接，你把内容发给我"。 

直到我给 OpenClaw 装了 **playwright-mcp** ，情况才彻底改变。 

* * *

## 什么是 playwright-mcp？

简单说：**给 AI agent 装一双眼睛和一双手。**

  * **Playwright** ：微软开源的浏览器自动化框架，能控制 Chrome、Firefox、WebKit
  * **MCP** （Model Context Protocol）：Anthropic 推出的 AI 工具调用协议
  * **playwright-mcp** ：把 Playwright 包装成 MCP 服务器，AI 可以直接调用



装上之后，AI 能做的事： 

  * 打开网页、点击按钮、填写表单
  * 提取页面内容、抓取数据
  * 截图、生成 PDF
  * 自动化重复操作（抢票、签到、数据采集）



* * *

## 安装过程（踩坑实录）

### 第一步：安装 npm 包
    
    
    npm install -g @playwright/mcp
    

> ⚠️ 如果报 `command not found: npm`，先装 Node.js。

装完后验证： 
    
    
    playwright-mcp --version
    

### 第二步：安装浏览器内核
    
    
    npx playwright install chromium
    

这步会下载 Chromium 浏览器，大概 150MB。如果你网络不好，可以设镜像： 
    
    
    PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright npx playwright install chromium
    

### 第三步：在 OpenClaw 中启用

如果你已经用 skillhub 安装过 playwright-mcp 技能，只需要在配置中启用： 
    
    
    {
    
    
      "skills": {
        "entries": {
          "playwright-mcp": {
            "enabled": true
          }
        }
      }
    }
    
    
    
    

如果还没安装，一条命令搞定： 
    
    
    skillhub install playwright-mcp
    

### 第四步：重启 Gateway

改完配置后重启 OpenClaw Gateway，新技能就会加载。 

* * *

## 装好了怎么用？

直接用自然语言告诉 AI 就行： 

你说的话| AI 实际做的事  
---|---  
"打开知乎搜索今日热榜"| 启动浏览器 → 导航到知乎 → 搜索 → 提取结果  
"帮我把这个网页截图"| 打开页面 → 截图 → 返回图片  
"登录这个网站，填一下表单"| 打开登录页 → 填账号密码 → 点击登录  
"抓取这个表格的数据"| 打开页面 → 定位表格 → 提取数据 → 返回 JSON  
  
不需要写代码，不需要配置，**用嘴就行** 。 

* * *

## 我踩过的坑

### 坑1：安装后还是不能用

**原因** ：技能安装了但没有在配置中 `enabled: true`。 **解决** ：检查 `openclaw.json`，确保 `skills.entries.playwright-mcp.enabled` 为 `true`。 

### 坑2：浏览器打不开页面

**原因** ：Chromium 没装或者版本不对。 **解决** ：重新运行 `npx playwright install chromium`。 

### 坑3：打开知乎/微信等网站被拦截

**原因** ：OpenClaw 有 SSRF 安全策略，默认不允许访问内网地址。 **解决** ：大部分公网网站不受影响。如果需要访问特定站点，可以在 `browser.ssrfPolicy` 中配置白名单。 

### 坑4：中文网页乱码

**原因** ：Playwright 默认系统编码不是 UTF-8。 **解决** ：启动时加 `--lang zh-CN` 参数。 

* * *

## 更进一步：高级玩法

### 1\. 自动化数据采集

让 AI 每天定时打开某个网页，抓取数据保存到本地： 
    
    
    每天早上 9 点，打开 XX 网站的热榜页面，
    
    
    提取前 20 条数据，保存到 CSV 文件。
    
    
    
    

### 2\. 自动化表单填写

重复性的表单操作，教 AI 一次，以后自动执行： 
    
    
    帮我登录 XX 网站，进入设置页面，
    
    
    把以下信息填进去：...
    
    
    
    

### 3\. 网页截图和对比

监控网页变化： 
    
    
    打开 XX 页面截图，和昨天的截图对比，
    
    
    看看有什么不同。
    
    
    
    

### 4\. 无障碍测试

Playwright 有无障碍快照功能，AI 能"看到"页面的结构化信息： 
    
    
    打开 XX 页面，用无障碍快照分析页面结构，
    
    
    告诉我这个页面的布局是否合理。
    
    
    
    

* * *

## MCP 协议是什么？为什么重要？

MCP（Model Context Protocol）是 Anthropic 在 2024 年底推出的开放协议，定义了 AI 模型和外部工具之间的通信标准。 

**之前** ：每个 AI 平台都要自己写插件（OpenAI 写 Function Calling，Claude 写 Tool Use，各自一套）。 **有了 MCP** ：工具开发者只需要写一个 MCP 服务器，所有支持 MCP 的 AI 平台都能用。 

这意味着： 

  * **playwright-mcp** 不只能用在 OpenClaw，也能用在 Claude Desktop、Cursor 等任何支持 MCP 的平台
  * 以后会有越来越多 MCP 工具（数据库、文件系统、API 调用），即插即用



* * *

## 最后

playwright-mcp 只是一个开始。MCP 生态正在快速爆发，未来 AI agent 的能力边界会越来越模糊。 

给 AI 装上浏览器的那一刻，它就不再是一个"只能聊天的工具"了——它变成了一个**能看、能点、能操作的数字助手** 。 

如果你也在用 OpenClaw 或类似的 AI agent 工具，强烈建议装一下。体验完全不一样。 

* * *

_本文基于实际安装操作整理，所有步骤均经过验证。如果你在安装过程中遇到问题，欢迎在评论区交流。_

### 💬 评论区
