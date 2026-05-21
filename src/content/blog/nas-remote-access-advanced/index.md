---
title: "NAS远程访问进阶：Tailscale零配置VPN、WireGuard自建、frp内网穿透方案对比"
publishDate: '2024-01-01'
description: "NAS远程访问进阶：Tailscale零配置VPN、WireGuard自建、frp内网穿透方案对比 - halo的技术博客"
tags:
  - 硬件数码
language: Chinese
---

NAS远程访问TailscaleWireGuardfrpVPN

# NAS远程访问进阶：Tailscale零配置VPN、WireGuard自建、frp内网穿透方案对比 我之前写过一篇讲IPv6和DDNS的文章，有人问：你说的这些我都试过，IPv6不稳定，DDNS要折腾域名，有没有更简单的办法？ 有。而且不止一种。 今天讲三个"进阶方案"：Tailscale、WireGuard自建、frp内网穿透。三个都能绕过"没有公网IP"这个坑，但思路完全不同。 

* * *

\--- 

## Tailscale：装上就能用

Tailscale这两年在NAS圈子里火得很快。我自己也用，主要原因是**真的不需要任何配置** 。 原理很简单：基于WireGuard加密协议，但加了一层"中控节点"帮你搞定NAT穿透。你NAS上装个客户端，手机上装个客户端，登录同一个账号，就能互相访问。全程不需要动路由器、不需要开端口、不需要懂任何网络知识。 核心功能： \- **MagicDNS** ：给每台设备分配一个域名，比如`nas.tailscale.local`，不用记IP \- **DERP中继** ：如果两端都处在严格NAT后面，Tailscale有自己的中继服务器帮你兜底 \- **Subnet Relay** ：开启后，不只是NAS，整个家庭网络都能通过Tailscale访问 \- **Taildrop** ：直接往其他设备传文件，类比AirDrop但跨平台 免费额度：100台设备，个人用绑绑够。 **缺点要说清楚** ：你的流量经过Tailscale的服务器，虽然是加密的，但隐私敏感的人会有顾虑。另外它依赖Tailscale公司活着，长期稳定性打个问号——虽然目前看起来很稳。 

[图片位：Tailscale工作原理图]

[图片：Tailscale网络拓扑示意]

* * *

\--- 

## WireGuard自建：性能最强，但需要折腾

如果你有一定技术基础，WireGuard是更好的长期选择。 WireGuard是最新的VPN协议，内核级实现，速度比OpenVPN快一大截。稳定性也强，我在软路由上跑了两年没掉过线。 自建的关键是需要一台有公网IP的VPS（中转服务器）。国内云服务器一个月大约30-50块，不算贵。 配置流程大概是这样： ``` 1. VPS上安装WireGuard，生成服务器密钥对 2. NAS上安装WireGuard客户端，生成客户端密钥对 3. 交换公钥，设置AllowedIPs 4. VPS防火墙开放51820/UDP端口 5. NAS启动WireGuard，连上VPS ``` 具体命令各家NAS不一样，群晖用Docker装，威联通和飞牛OS也都有官方或社区安装包。恩山无线论坛上有大量教程可以参考。 

[图片位：WireGuard架构图]

[图片：WireGuard VPN架构]

**自建的好处** ： \- 完全自主，不依赖任何第三方服务 \- 可以搭全家人的VPN，不限设备数 \- 流量完全走你自己的VPS，隐私有保障 \- 理论上可以访问任意端口，不只是NAS的Web界面 **自建的代价** ： \- 需要一台VPS，有成本 \- NAT穿透需要手动配置（如果NAS在内网深处） \- 出问题了要自己排查 

* * *

\--- 

## frp内网穿透：按需开通，不浪费

frp和前面两个方案思路完全不同。 Tailscale和WireGuard是"把设备连成一张虚拟网络"，frp是"把本地端口暴露到公网"。你可以理解为frp是一个反向代理：它运行在VPS上，接收外部请求，转发给家里NAS上的具体服务。 适合场景： \- 只想访问NAS的Web界面或某几个端口，不想搭VPN \- 家里NAS在内网深处，WireGuard的NAT穿透配置搞不定 \- 想用自定义域名访问，不同服务用不同端口 配置比WireGuard稍复杂，但灵活性更高。比如我可以把NAS的5000端口映射到VPS的5000端口，把另一个服务的8006映射到8006，一目了然。 frp也有Web界面管理工具，比如frpc-desktop，可以可视化配置，对不想写INI文件的人比较友好。 

[图片位：frp工作原理图]

[图片：frp内网穿透原理]

* * *

\--- 

## 三个方案横向对比

| Tailscale| 零难度| 免费（100台）| 快| 一般（经第三方）| 非技术用户，追求省事  
---|---|---|---|---|---|---  
| WireGuard自建| 中等| 30-50元/VPS| 最快| 好（完全自主）| 技术用户，长期稳定  
| frp| 中等偏高| 30元起/VPS| 快| 好| 只想开特定端口  
  
* * *

\--- 

## 我的选择

我自己是WireGuard+Tailscale混着用。 Tailscale用于手机快速访问NAS，看看照片、下载个文件，随开随用。回家或者临时连一下，不需要任何仪式感。 WireGuard用于电脑远程办公，需要访问家里整网资源的时候用它。NAS、打印机、Home Assistant全部走WireGuard，跑满带宽，跑满我家下行500M上行50M的宽带。 frp我没有用，因为我有公网VPS，直接WireGuard就覆盖了需求。但我知道有些朋友的NAS在内网深处，WireGuard的NAT穿透配置复杂，frp反而是最简单的解法。 **没有哪个方案绝对好** ，关键看你的网络环境和动手能力。 

* * *

\--- 你用哪种方案访问NAS？评论区说说你的网络情况和遇到过的坑。
