---
title: "NAS日志监控与告警：半夜硬盘快满了你都不知道？"
publishDate: '2026-04-08'
description: "NAS日志监控与告警：半夜硬盘快满了你都不知道？ - halo的技术博客"
tags:
  - 硬件数码
language: Chinese
---

NAS日志监控SMART硬盘健康告警系统

去年有一天，我的群晖 NAS 突然发不出通知了。登录一看，硬盘空间已经用完，所有服务都挂了。日志显示，某个 Docker 容器把日志写满了整个磁盘，而这台 NAS 没有设置任何容量告警。

从那之后我开始认真研究 NAS 监控。现在我的 NAS 有完整的告警体系：硬盘健康、空间容量、Docker 日志、异常登录，任何异常都会在 5 分钟内推送到我手机。这篇文章说说我怎么搭的这套系统。

![监控面板](https://blog.halo26812.eu.org/images/nas-log-monitoring/dashboard.jpg)

## NAS 自带日志系统

### 群晖日志中心（Log Center）

群晖 DSM 7 自带 Log Center，可以集中管理系统日志、SMB 访问日志、SSH 登录日志、防火墙日志等。在"控制面板 → 日志中心"里可以查看和配置日志规则。

设置日志推送：点击"存档设置"→ 启用"系统事件通过邮件发送"，填入你的邮箱地址。群晖支持 SMTP 直接发邮件，不需要额外配置。

### 威联通 Log Center

威联通 QTS 的 Log Center 功能类似，支持日志集中收集和告警规则设置。可以把 Syslog 转发到其他服务器做集中分析。

### 飞牛OS / 绿联

这类系统目前没有专门的日志中心，但可以通过 SSH 登录查看 /var/log 目录下的系统日志，或者用 Docker 部署专门的日志收集工具（如 rsyslog）。

## SMART 监控：硬盘健康数据解读

硬盘是 NAS 最脆弱的环节，提前知道硬盘要坏比数据丢了再恢复重要得多。SMART（Self-Monitoring, Analysis and Reporting Technology）是硬盘自带的健康监控系统。

群晖 NAS 在"存储管理器"→ "HDD/SSD"里能看到每块硬盘的 SMART 信息。如果看到警告图标，说明硬盘有问题。

![服务器监控](https://blog.halo26812.eu.org/images/nas-log-monitoring/server.jpg)

### 关键 SMART 参数解读

参数| 含义| 正常值| 警告值  
---|---|---|---  
Reallocated Sectors Count (05)| 已重映射的坏扇区数| 0| >0 → 硬盘即将故障  
Current Pending Sector Count (C7)| 等待重映射的不稳定扇区| 0| >0 → 扇区正在恶化  
Power-On Hours (09)| 硬盘累计工作时间| —| >30000小时需关注  
Temperature (194)| 当前温度| 30-45°C| >55°C 需降温  
Wear Leveling Count (177)| SSD 磨损程度| —| 接近 100% → 寿命将尽  
  
我自己用 HD Tune Pro 在电脑上跑过全盘扫描，检查是否有坏道。群晖自带的 SMART 检测只是表面扫描，要做全盘扫描需要用专门的工具。

### 推荐工具：smartctl 命令

通过 SSH 连接 NAS，直接运行 smartctl 查看详细数据：
    
    
    smartctl -a /dev/sda   # 查看 sda 硬盘完整 SMART 数据
    smartctl -H /dev/sda   # 健康状态快速检查（Passed/Failed）
    smartctl -t short /dev/sda  # 快速自检（几分钟）
    smartctl -t long /dev/sda   # 完整自检（可能需要数小时）
    

建议每周跑一次 short 检测，每月跑一次 long 检测，结果记录到日志里。

## 告警方式：邮件 / PushPlus / 钉钉机器人

告警的精髓是：消息要能及时看到，否则告警没有意义。

### 邮件告警

最基础的方案，群晖内置的邮件告警配置最简单：控制面板 → 通知中心 → 电子邮件，设置 SMTP 服务器（QQ 邮箱、163 邮箱都行，注意开启 SMTP 服务并生成专用密码）。

缺点：邮件可能进垃圾箱，或者半夜看到手机亮了一下点开又睡了。

### PushPlus 微信推送

PushPlus（pushplus.plus）是一个免费的微信消息推送服务，注册后得到一个 Token，通过 HTTP 请求就能往微信发消息。

在群晖里用 Python 脚本或者 Docker 的 watchtower 配合 PushPlus，可以把告警消息直接推送到微信，比邮件更及时。

### 钉钉机器人告警

钉钉群可以创建自定义机器人，获取 Webhook 地址后，用 curl 就能发告警消息：
    
    
    curl -X POST 'https://oapi.dingtalk.com/robot/send?access_token=你的Token' \
      -H 'Content-Type: application/json' \
      -d '{"msgtype": "text", "text": {"content": "NAS告警：硬盘温度超过60°C"}}'
    

钉钉的好处是可以附带手机震动，半夜也能叫醒你。

![告警通知](https://blog.halo26812.eu.org/images/nas-log-monitoring/alert.jpg)

## 存储容量预警：设定阈值自动告警

这是最容易被忽视的告警。我的群晖设置了三个容量告警阈值：

  * **80% 告警** ：发邮件提醒整理空间
  * **90% 告警** ：微信推送，提醒立即处理
  * **95% 告警** ：钉钉推送 + 短信，硬盘几乎满了



群晖自带的"存储空间通知"在控制面板 → 通知中心 → 存储空间里设置，填入阈值百分比和通知邮箱即可。

另外我设置了定时任务（每天早上 9 点），用 df 命令检查各存储池的占用情况，输出到一个日志文件，这样能看出容量的历史变化趋势。

## Docker 日志管理：容器日志自动清理

开头说的那个磁盘被 Docker 日志写满的事，教会了我必须管好容器日志。

Docker 容器运行时会持续写入日志到 /var/lib/docker/containers/ 目录，默认不清理，大容器几天就能写满整个磁盘。

解决方法：在 docker-compose.yml 里加上日志限制：
    
    
    services:
      your-app:
        image: xxx
        logging:
          driver: "json-file"
          options:
            max-size: "10m"
            max-file: "3"
    

这会把每个容器的日志文件限制在 10MB，最多保留 3 个文件，超过就自动轮转。

如果是已经跑着的容器，可以执行：
    
    
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
      sane/docker-log-options \
      your-container-name --log-opt max-size=10m --log-opt max-file=3
    

## 异常访问检测：SSH 暴力破解告警

公网 IP 的 NAS，每天都会被扫描和暴力破解。SSH 22 端口如果暴露在公网，一定会有大量登录尝试。

我的处理方式：

  1. **改变 SSH 端口** ：把 22 改成高位端口（如 2222），减少 90% 的自动扫描
  2. **禁用密码登录** ：只允许 SSH Key 登录，完全杜绝暴力破解  
`PasswordAuthentication no`
  3. **安装 Fail2Ban** ：自动封禁连续登录失败的 IP  
群晖可以通过 SSH 安装 fail2ban，或者用 Docker 运行
  4. **登录成功 / 失败通知** ：在 /etc/pam.d/sshd 或者 .bashrc 里加一句，当有人 SSH 登录成功时给手机发消息



设置 Fail2Ban 后，系统会自动封禁 10 分钟内登录失败超过 3 次的 IP。我查日志看到被封禁的 IP 列表，每次都有几十个，说明这套机制确实在工作。

## 推荐工具：WatchYourLAN（局域网设备监控）

WatchYourLAN 是一个轻量级的局域网设备监控工具，扫描局域网内的设备，发现新设备接入或者设备离线时发送通知。

用途：监控家里有没有陌生设备接入 WiFi，及时发现蹭网或者被入侵的情况。

部署方法（Docker）：
    
    
    services:
      watchyourlan:
        image: aceberg/watchyourlan
        container_name: watchyourlan
        network_mode: host
        ports:
          - 8080:8080
        volumes:
          - ./data:/data
    

界面很简洁，可以看到所有在线设备的 IP、MAC 地址、hostname。如果某个设备突然出现（比如陌生 MAC），就发告警。

## 我的监控体系总结

目前我的 NAS 监控包括：

  * 硬盘 SMART 健康状态（每周自动检测）
  * 存储空间使用率（阈值告警）
  * Docker 容器日志（自动轮转清理）
  * SSH 登录（Fail2Ban + 登录通知）
  * 局域网设备变化（WatchYourLAN）



整套体系搭起来花了半天时间，现在能睡安稳觉了。NAS 上跑的所有服务我都能远程掌控，有什么问题 5 分钟内知道。

最关键的一点：不要相信 NAS 永远不会出问题，要假设它随时可能出问题。监控和告警不是为了让它不出问题，而是为了出了问题能第一时间发现、最小化损失。
