---
title: "NAS性能优化指南：榨干硬件的全部潜力"
publishDate: '2026-04-08'
description: "NAS性能优化指南：榨干硬件的全部潜力 - halo的技术博客"
tags:
  - 硬件数码
language: Chinese
---

NAS性能优化SMB多通道SSD缓存网络

买回来 NAS 发现跑不满带宽？文件传输慢得让人抓狂？其实大多数情况下，不是硬件不行，是设置没优化。我把我折腾过的优化方法整理了一下，按投入产出比排序，从零成本到需要投入的方案都有。

## 一、先诊断：瓶颈在哪里？

![速度测试](/images/nas-performance-optimization/speed-test.jpg)

优化之前先要搞清楚瓶颈在哪。常见瓶颈有三个：

  * **网络瓶颈** ：NAS 是千兆（1Gbps），实际跑到 110MB/s 左右是正常的。如果跑到 200MB/s，可能是测速工具的算法问题，不是真实速度。
  * **硬盘瓶颈** ：机械硬盘顺序读大概 150MB/s，随机读写会掉到 10-50MB/s。HDD 做 docker 镜像存储时特别容易遇到随机读写瓶颈。
  * **CPU/内存瓶颈** ：NAS CPU 通常较弱，跑加密、压缩、转码时容易成为瓶颈。



先用 NAS 自带的测试工具跑一下，看看实际速度。我用 iperf3 测内网带宽，用 dd 测硬盘读写：
    
    
    # 测硬盘写速度
    dd if=/dev/zero of=/volume1/testfile bs=1M count=1024 oflag=direct
    
    # 测硬盘读速度
    dd if=/volume1/testfile of=/dev/null bs=1M iflag=direct

## 二、零成本优化：SMB多通道（效果最明显）

![服务器机房](/images/nas-performance-optimization/server-room.jpg)

如果你的 NAS 有两个网口，这是最值得做的优化，而且完全免费。

SMB 多通道允许同时用多个网络连接传输数据。比如 NAS 有两个 2.5Gbps 网口，电脑用一个 2.5Gbps 网口，理论上就能跑到 2.5Gbps 带宽。

设置方法：

  1. NAS 端：将两个网口都接入同一子网，启用 SMB 多通道（群晖在控制面板→文件服务→SMB→启用"启用SMB 3.0多通道"）
  2. 电脑端：插两块网卡，或者用支持链路聚合的路由器
  3. 验证：用 `Get-SmbMultichannelConnection -ServerName NAS_IP` 看是否有多通道连接



实测效果：双 2.5Gbps 下，传单个大文件速度从 280MB/s 提升到 450MB/s，接近单个 SSD 的速度了。

## 三、SSD缓存：要不要加？

SSD 缓存是 NAS 性能优化里被神话最多的功能。实际效果取决于你的使用场景。

**读缓存** （最常用）：把频繁访问的数据放在 SSD 上，加速随机读取。适合跑数据库、照片库索引、Docker 容器目录。

**写缓存** ：把写入请求先放到 SSD，再异步刷到 HDD。可以显著提升小文件写入速度，但有数据丢失风险（突然断电会导致未刷盘的数据丢失）。

**不适合的场景** ：大文件顺序传输（NAS 带宽本身就受限于网络，小文件缓存没用）。监控录像写入（连续写入，不需要缓存）。

我的建议：**不要买 NAS 专用的 SSD 缓存加速卡** ，又贵又鸡肋。如果你有多盘位，直接拿一块 SATA SSD 做成读缓存，专门给 Docker 和照片库加速即可。

## 四、网络协议：SMB vs NFS vs AFP

不同协议速度差距明显：

协议| 实测速度（千兆）| 适用场景  
---|---|---  
SMB 3.0（多通道）| 450MB/s（双2.5G）| Windows/全平台，通用  
SMB 3.0（单通道）| 110-120MB/s| 日常文件共享  
NFS| 120-130MB/s| Linux/Mac，高性能  
AFP| 110MB/s| 旧版macOS，逐步淘汰  
  
结论：**macOS 用户用 NFS** ，速度比 SMB 快一点。Windows 用户老老实实用 SMB 3.0。Linux 服务器访问 NAS，用 NFS 最省心。

## 五、Docker资源限制

NAS 上的 Docker 容器默认不限资源，跑多了会互相抢。我现在的配置：
    
    
    services:
      jellyfin:
        mem_limit: 2g
        mem_reservation: 512m
        cpus: 2
    
      nextcloud:
        mem_limit: 1g
        mem_reservation: 256m
        cpus: 1
    
      homeassistant:
        mem_limit: 1g
        mem_reservation: 256m
        cpus: 0.5

原则：给每个服务设置内存上限，避免某个容器内存泄漏把整个 NAS 卡死。CPU 限制相对宽松，但 Jellyfin 视频转码时要设高一点。

## 六、文件系统选择

群晖支持 Btrfs 和 ext4。Btrfs 支持快照（数据保护强），ext4 性能稍好一点。

我的建议：**数据盘用 Btrfs，Docker 盘用 ext4** 。Docker 容器不存在快照需求，ext4 性能更好。数据盘需要快照保护，用 Btrfs。

## 七、优化前后对比

我用 J4125 + 4×8TB HDD + 双 2.5Gbps 网口的配置实测：

  * 默认设置（千兆，SMB 单通道）：**112MB/s**
  * 开启 SMB 多通道（2.5Gbps×2）：**430MB/s**
  * \+ SSD 读缓存（Docker 目录）：**450MB/s**
  * \+ NFS 协议替代 SMB：**460MB/s**



最后卡在 460MB/s，是因为电脑端 SSD 写入速度就只有这么快了。

* * *

_数据来源：群晖官方性能白皮书，个人实测（J4125，4×8TB WD Red，2×2.5Gbps，2026年4月）_
