---
title: "用NAS搭建私有Git服务器：Gitea/群晖Git Server对比"
publishDate: '2026-04-08'
description: "用NAS搭建私有Git服务器：Gitea/群晖Git Server对比 - halo的技术博客"
tags:
  - 硬件数码
language: Chinese
---

NASGitGitea自托管私有仓库

去年我把个人项目从 GitHub 私有仓库迁到了自己 NAS 上。不是为了装，是真的算了一笔账：免费私有仓库有 500MB 限制，想扩容就得掏钱；而我自己有两块 4TB 硬盘，跑个 Git 服务绰绰有余。用了快一年，说说真实体验。

![NAS服务器](https://blog.halo26812.eu.org/images/nas-git-server/server.jpg)

## 为什么用 NAS 跑 Git？三个理由

第一个是隐私。我有几个不想公开的项目，放在 GitHub 总觉得不太安心。放到自己 NAS 上，数据在自己手里，心理上踏实很多。

第二个是速度。家里是千兆局域网，push 和 pull 都在局域网跑，速度比 GitHub 快多了。我测过，一个 500MB 的仓库，GitHub 拉取大概要 2 分钟，局域网只要 15 秒。

第三个是成本。一台入门级 NAS 2000 元左右，能跑 Git、Docker、相册、视频，Git 服务只是附带的；GitHub 私有仓库年费 48 刀（约 350 元），NAS 摊下来还更划算。

![终端操作](https://blog.halo26812.eu.org/images/nas-git-server/terminal.jpg)

## 方案一：Gitea —— 轻量级 GitHub 替代

Gitea 是一个开源的自托管 Git 服务，功能和 GitHub 差不多，但资源占用少很多。一台性能一般的 NAS 就能跑得动。

### 部署方法（Docker Compose）

在 NAS 上新建一个目录，扔进 docker-compose.yml：
    
    
    version: '3'
    services:
      gitea:
        image: gitea/gitea:latest
        container_name: gitea
        restart: always
        ports:
          - "3000:3000"
          - "2222:22"
        volumes:
          - ./data:/data
          - /etc/timezone:/etc/timezone:ro
        environment:
          - DOMAIN=你的NAS地址
          - HTTP_PORT=3000
          - SSH_PORT=2222
          - ROOT_URL=http://你的NAS地址:3000/
    

跑起来之后，浏览器打开 http://NAS地址:3000，按引导配置管理员账号就行了。第一次配置会要求你设置数据库，推荐用 SQLite，简单省事。

### SSH Key 配置

本地机器上生成 SSH Key：
    
    
    ssh-keygen -t ed25519 -C "你的邮箱"
    cat ~/.ssh/id_ed25519.pub
    

在 Gitea 网页右上角点头像 → 设置 → SSH 密钥，把公钥粘贴进去。以后 clone 和 push 就不用输入密码了。

Clone 示例：
    
    
    git clone ssh://git@192.168.1.100:2222/你的用户名/仓库名.git
    

### 我的使用感受

Gitea 界面很干净，支持 Issues、PR、Wiki，还能装主题。团队用也够，我给两个朋友的开发小组搭了一套，反馈不错。它的缺点是需要手动维护更新，不像 GitHub 那样省心；另外如果 NAS 配置低，网页响应会慢。

![Git操作](https://blog.halo26812.eu.org/images/nas-git-server/git.jpg)

## 方案二：群晖 Git Server —— 官方套件，适合小团队

如果你是群晖 NAS 用户，不想折腾 Docker，可以用群晖自带的 Git Server 套件。DSM 7 在套件中心直接装就行。

装完之后，在控制面板里找到"版本管理"→"Git"，创建一个仓库，给用户分配权限，就能用了。

群晖 Git Server 的优点是集成度高，不用额外配置，用户管理直接用群晖自己的账号体系。缺点是功能比较基础，没有 Issue 和 PR，也不支持代码审查。

### 群晖 DSM 7 搭建步骤

  1. 打开套件中心，安装"Git Server"
  2. 进入控制面板 → 终端机/SNMP → 启动 SSH
  3. 通过 SSH 连接 NAS，创建仓库：`git init --bare /volume1/git/项目名.git`
  4. 在群晖文件管理器里给对应用户设置读写权限
  5. 本地 Clone：`git clone ssh://用户名@NAS地址/volume1/git/项目名.git`



## Gitea vs 群晖 Git Server 怎么选？

对比项| Gitea| 群晖Git Server  
---|---|---  
功能丰富度| ★★★★★ 接近GitHub| ★★☆☆☆ 基础功能  
部署难度| ★★★☆☆ 需要Docker| ★★★★★ 套件直接装  
资源占用| 中等（500MB RAM+）| 极低  
Issue/PR| 支持| 不支持  
适合场景| 个人/小团队开发| 简单代码存储  
  
## vs GitHub/GitLab：私有性 vs 便利性

这里有个取舍要说明白。

GitHub 的优点是全球访问、CI/CD 完善、开发者生态好。缺点是私有仓库要付费，而且数据在别人服务器上。

GitLab 功能最全，但资源占用太大，入门级 NAS 带不动 Community Edition。

我的判断是这样的：

  * 公开项目 → 直接用 GitHub
  * 私有项目，团队规模≤3人 → NAS + Gitea 是性价比最高的选择
  * 私有项目，团队规模>5人，有 CI/CD 需求 → 考虑 GitLab 或者直接买 GitHub Team



## 内网 vs 公网访问：安全问题

跑在内网的 Git 服务，外人访问不到，相对安全。但如果你想在公司或者外面访问 NAS 上的 Git，就要把端口暴露出来，这就涉及安全问题了。

我的建议：

  1. SSH 端口不要用默认的 22，改成一个高位端口（比如 2222）
  2. 启用 Fail2Ban，防止暴力破解
  3. 公网访问用 Tailscale 或者 WireGuard VPN，不要直接暴露 22 和 3000 端口
  4. Gitea 的注册功能最好关闭，只邀请需要的人



如果你不想折腾这些，可以用群晖自带的 QuickConnect，虽然速度慢一点，但安全性有保障。

## 我的结论

用了一年 NAS 上的 Git 服务，真实感受是：比想象中好用，但前提是你得愿意花 1-2 小时配置好，之后就不用管了。

对于有代码洁癖的程序员来说，把自己的代码仓库放在自己硬盘上，是一件很有安全感的事情。如果你的 NAS 配置还过得去，强烈推荐试试 Gitea。
