---
title: "飞牛NAS进阶玩法：Docker容器推荐与部署教程"
publishDate: '2024-01-01'
description: "飞牛NAS进阶玩法：Docker容器推荐与部署教程 - halo的技术博客"
tags:
  - 硬件数码
language: Chinese
---

飞牛NASDocker容器部署进阶教程

# 飞牛NAS进阶玩法：Docker容器推荐与部署教程

2026-04-07 · 硬件数码/NAS存储 · 🔖 飞牛NAS, Docker, 容器部署, 进阶教程

飞牛NAS（fnOS）最大的优势之一就是Docker支持完善。通过Docker，可以扩展NAS的功能，把它变成一台多功能服务器。

这篇文章推荐几个实用的Docker容器，并提供详细的部署教程。

## 一、Docker基础操作

### 1\. 开启SSH

进入【设置】→【SSH】，开启SSH服务。默认端口22，密码就是飞牛NAS的登录密码。

### 2\. Docker镜像源配置

国内用户建议配置镜像源加速：
    
    
    # 进入Docker设置
    # 镜像源地址：
    https://docker.m.daocloud.io
    https://dockerhub.icu

### 3\. 两种部署方式

  * **图形界面** ：适合新手，点鼠标完成
  * **Docker Compose** ：适合进阶，配置文件管理，方便迁移



## 二、实用容器推荐

### 1\. qBittorrent - 下载神器

最常用的BT下载工具，支持RSS订阅、远程访问。

**docker-compose.yml** ：
    
    
    version: '3.7'
    services:
      qbittorrent:
        image: linuxserver/qbittorrent:latest
        container_name: qbittorrent
        environment:
          - PUID=1000
          - PGID=1001
          - TZ=Asia/Shanghai
          - WEBUI_PORT=8080
        volumes:
          - ./config:/config
          - /vol2/1000/downloads:/downloads
        network_mode: host
        restart: unless-stopped

**访问** ：http://NAS的IP:8080

**默认账号** ：admin / adminadmin

### 2\. Home Assistant - 智能家居中枢

统一管理小米、涂鸦、飞利浦等智能家居设备。
    
    
    version: '3'
    services:
      homeassistant:
        image: homeassistant/home-assistant:stable
        container_name: homeassistant
        environment:
          - TZ=Asia/Shanghai
        volumes:
          - ./config:/config
        network_mode: host
        restart: unless-stopped

**访问** ：http://NAS的IP:8123

### 3\. Jellyfin - 影音服务器

开源免费的影音服务器，自动刮削海报、支持多设备播放。
    
    
    version: '3'
    services:
      jellyfin:
        image: jellyfin/jellyfin:latest
        container_name: jellyfin
        environment:
          - TZ=Asia/Shanghai
        volumes:
          - ./config:/config
          - ./cache:/cache
          - /vol2/1000/media:/media
        network_mode: host
        restart: unless-stopped

**访问** ：http://NAS的IP:8096

### 4\. Heimdall - 个人导航页

把所有NAS服务整合到一个导航页面，一目了然。
    
    
    version: '3'
    services:
      heimdall:
        image: lscr.io/linuxserver/heimdall:latest
        container_name: heimdall
        environment:
          - PUID=1000
          - PGID=1001
          - TZ=Asia/Shanghai
        volumes:
          - ./config:/config
        ports:
          - "8710:80"
        restart: unless-stopped

**访问** ：http://NAS的IP:8710

### 5\. Homebox - 内网测速

测试局域网到NAS的网速，排查网络问题。
    
    
    version: '3'
    services:
      homebox:
        image: ghcr.io/homarr/homebox:latest
        container_name: homebox
        environment:
          - TZ=Asia/Shanghai
        ports:
          - "3300:3300"
        volumes:
          - ./data:/data
        restart: unless-stopped

**访问** ：http://NAS的IP:3300

### 6\. MrDoc - 个人知识库

在线文档系统，可以当个人Wiki使用。
    
    
    version: '3'
    services:
      mrdoc:
        image: registry.cn-hangzhou.aliyuncs.com/zmister/mrdoc:v9.1
        container_name: mrdoc
        volumes:
          - ./MrDoc:/app/MrDoc
        ports:
          - "10086:10086"
        restart: always

**访问** ：http://NAS的IP:10086

### 7\. LibreTV - 在线观影

聚合多个视频源，免费无广告观影。
    
    
    version: '3'
    services:
      libretv:
        image: bestzwei/libretv:latest
        container_name: libretv
        ports:
          - "8899:8080"
        environment:
          - PASSWORD=your_password
        restart: unless-stopped

**访问** ：http://NAS的IP:8899

## 三、部署步骤（以qBittorrent为例）

### 方式一：图形界面部署

  1. 打开飞牛NAS的Docker应用
  2. 点击【镜像仓库】，搜索 qbittorrent
  3. 点击下载镜像
  4. 下载完成后，点击【容器】→【添加容器】
  5. 填写容器名称，勾选"开机自动开启"
  6. 配置端口映射：主机端口8080 → 容器端口8080
  7. 配置存储位置：映射配置目录和下载目录
  8. 点击创建，等待启动



### 方式二：Docker Compose部署

  1. 打开Docker应用，点击【Compose】→【新建项目】
  2. 输入项目名称（如 qbittorrent）
  3. 选择存储路径
  4. 点击【创建docker-compose.yml】
  5. 粘贴上面的配置内容
  6. 勾选【创建项目后立即启动】
  7. 点击确定，等待部署完成



**💡 建议** ：使用Docker Compose部署，配置文件可以备份，迁移到其他NAS时直接复制粘贴就行。 

## 四、常见问题

### 1\. 容器启动失败？

  * 检查端口是否被占用
  * 检查存储目录权限（右键文件夹→属性→确保可读写）
  * 查看容器日志，根据错误信息排查



### 2\. 访问不了Web界面？

  * 检查端口映射是否正确
  * 检查防火墙是否放行端口
  * 确认容器状态为"运行中"



### 3\. 数据存储在哪里？

容器内的数据需要映射到NAS存储空间，否则容器删除后数据会丢失。

### 4\. 如何更新容器？
    
    
    # 1. 停止并删除旧容器
    # 2. 拉取新镜像
    # 3. 重新创建容器

## 五、进阶技巧

### 1\. 网络模式选择

模式 | 特点 | 适用场景  
---|---|---  
Bridge（桥接） | 需要手动映射端口 | 大多数应用  
Host（主机） | 容器直接使用主机网络 | 需要IPv6、多端口应用  
  
### 2\. 资源限制

可以在创建容器时限制CPU和内存使用，避免某个容器占用过多资源影响NAS整体性能。

### 3\. 自动重启策略
    
    
    restart: unless-stopped  # 除非手动停止，否则自动重启
    restart: always          # 总是自动重启
    restart: on-failure      # 失败时重启

## 六、容器推荐清单

容器 | 用途 | 推荐指数  
---|---|---  
qBittorrent | BT下载 | ⭐⭐⭐⭐⭐  
Jellyfin | 影音服务器 | ⭐⭐⭐⭐⭐  
Home Assistant | 智能家居 | ⭐⭐⭐⭐  
Heimdall | 导航页 | ⭐⭐⭐⭐  
Homebox | 内网测速 | ⭐⭐⭐  
MrDoc | 知识库 | ⭐⭐⭐  
LibreTV | 在线观影 | ⭐⭐⭐  
  
**⚠️ 注意** ：不要装太多容器！每个容器都会占用系统资源。够用就行，不是越多越好。 

* * *

_数据来源：飞牛NAS论坛、CSDN博客、今日头条、腾讯云_
