---
title: "给NAS装了个私有磁力搜索引擎，1500万种子随时搜"
publishDate: '2024-01-01'
description: "给NAS装了个私有磁力搜索引擎，1500万种子随时搜 - halo的技术博客"
tags:
  - 硬件数码
language: Chinese
---

NASBitmagnetDHT爬虫磁力搜索

找资源最烦什么？广告弹窗、链接失效、还要担心隐私泄露。直到我给NAS装了Bitmagnet，才发现——原来可以自己建一个磁力搜索引擎。 

## 一、Bitmagnet是什么？

简单说：**你自己私有的"磁力链接搜索引擎"。**

传统种子网站是中心化的——网站挂了，资源就没了。Bitmagnet不一样，它通过**DHT网络** 主动爬取全球BitTorrent网络中的种子信息。

这意味着：

  * ✅ 完全隐私：所有数据都在你的NAS上
  * ✅ 持续更新：24/7不间断爬取新资源
  * ✅ 去中心化：不受任何第三方服务影响
  * ✅ 自主可控：你可以完全控制索引的内容



## 二、核心功能

### 1\. DHT网络爬虫（杀手级功能）

这是Bitmagnet最独特的地方。

传统的BitTorrent客户端只是被动地从DHT网络获取peer信息，而Bitmagnet**主动爬取DHT网络中的所有info hash** ，并请求对应的种子元数据。

结果就是：大量"隐藏"资源被挖掘出来，这些资源在传统种子网站上根本找不到。

### 2\. 智能内容分类

Bitmagnet内置了强大的内容分类器，能自动识别：

  * 资源类型（电影、电视剧、音乐、软件）
  * 语言版本
  * 分辨率（720p、1080p、4K）
  * 来源（BluRay、WEBRip、HDTV）



还能与**The Movie Database** 等外部数据库集成，自动补充影片信息。

### 3\. 多种访问方式

方式| 用途  
---|---  
Web UI| 响应式网页界面，支持多语言  
GraphQL API| 强大查询能力，内置Playground  
Torznab端点| 与Sonarr、Radarr等套件集成  
导入功能| 批量导入种子（如RARBG备份）  
  
## 三、实际效果

我的NAS运行了几个月后，达到了以下规模：

指标| 数据  
---|---  
数据库记录| 1500万+ 种子资源  
存储占用| 约90GB  
资源覆盖| 电影、电视剧、纪录片、音乐、软件  
  
**这样的数据规模意味着：几乎任何你能想到的资源，在Bitmagnet中都能找到。**

而且由于它是实时爬取的，新发布的资源往往能在几分钟内就被索引到。

## 四、部署要求

### 系统要求

项目| 要求  
---|---  
数据库| PostgreSQL  
存储空间| 至少100GB，预留50GB增长  
网络| 稳定的网络连接（DHT爬虫需要）  
容器| Docker / Docker Compose  
  
### 最小化部署示例
    
    
    version: '3'
    
    services:
      bitmagnet:
        image: ghcr.io/bitmagnet-io/bitmagnet:latest
        container_name: bitmagnet
        ports:
          - "3333:3333"
          - "3334:3334/tcp"
          - "3334:3334/udp"
        restart: unless-stopped
        environment:
          - POSTGRES_HOST=postgres
          - POSTGRES_PASSWORD=postgres
        command:
          - worker
          - run
          - --keys=http_server
          - --keys=queue_server
          - --keys=dht_crawler
        depends_on:
          postgres:
            condition: service_healthy
    
      postgres:
        image: postgres:16-alpine
        container_name: bitmagnet-postgres
        volumes:
          - ./data:/var/lib/postgresql/data
        restart: unless-stopped
        environment:
          - POSTGRES_PASSWORD=postgres
          - POSTGRES_DB=bitmagnet
          - PGUSER=postgres
        shm_size: 1g
        healthcheck:
          test: ["CMD-SHELL", "pg_isready"]
          start_period: 20s
          interval: 10s
    

## 五、与媒体库集成

Bitmagnet的**Torznab端点** 可以与Servarr套件无缝集成：

  * **Prowlarr** ：作为索引器添加 http://NAS_IP:3333/torznab
  * **Sonarr** ：自动抓取电视剧
  * **Radarr** ：自动抓取电影



配合Jellyfin/Emby，可以实现：**搜索 → 抓取 → 整理 → 播放** 全流程自动化。

## 六、注意事项

### 隐私和安全

由于Bitmagnet会主动参与DHT网络，建议：

  * 在路由器上设置适当的防火墙规则
  * 考虑使用VPN增加匿名性
  * 定期备份PostgreSQL数据库



### 性能优化

  * 调整DHT爬虫并发数平衡性能和资源消耗
  * 使用SSD存储数据库获得更好查询性能
  * 定期清理无用的数据库记录



### 版权提醒

Bitmagnet只是一个学习项目，请注意版权问题，合理使用。

* * *

## 一句话总结

> **Bitmagnet = 私有磁力搜索引擎 + 24/7爬虫 + 与媒体库无缝集成。**

对于追求数据自主权和隐私保护的NAS用户来说，这绝对是一个值得部署的项目。

**一次部署，长期受益。**

* * *

_数据来源：知乎、GitHub、CSDN博客_
