---
title: "NAS上跑虚拟机：Windows/Linux/软路由一台搞定"
publishDate: '2024-01-01'
description: "NAS上跑虚拟机：Windows/Linux/软路由一台搞定 - halo的技术博客"
tags:
  - 硬件数码
language: Chinese
---

NAS虚拟机软路由VirtualBoxOpenWrt

有些场景 Docker 真的搞不定。比如跑 Windows 跑 macOS，比如跑 PfSense 软路由，比如需要完整的内核权限。这时候就得开虚拟机了。今天说说我在飞牛 OS 上跑虚拟机的经验。

## 一、哪些NAS系统支持虚拟机？

先说结论：不是所有NAS都能跑虚拟机，这个功能对硬件要求比较高。

![虚拟机架构示意](/images/nas-virtual-machine/virtualization.jpg)

**飞牛 OS** 自带虚拟机功能，安装即用，对硬件兼容性比较好。我在飞牛 OS 上跑过 OpenWrt 和 Ubuntu Server，稳定性还不错。

**群晖** 的 Virtual Machine Manager（VMM）是官方解决方案。支持的型号比较多，DS918+以上的机型基本都支持。界面友好，但用起来感觉对资源的管理比较保守。

**威联通** 的 Virtualization Station 是我用过的最完善的方案。界面做得很细致，虚拟机的克隆、快照、迁移都支持。但授权策略有点复杂，有些高级功能要付费。

**黑群晖/unRAID** ：这两家对虚拟机的支持更好，特别是 unRAID 本身就是基于虚拟机理念设计的，PCIe 直通非常方便。如果你想用 NAS 同时跑多台虚拟机，unRAID 是更好的选择。

## 二、内存分配：跑多少才够？

这是最常见的问题。我的经验：

  * **跑 OpenWrt 软路由** ：1核1GB 足够了。OpenWrt 本身非常轻量，CPU 主要消耗在流量转发上。跑 NAT + 防火墙 + 广告屏蔽，一个 vCPU 就能跑到线速。
  * **跑 Ubuntu Server** ：2核2GB 是起步。我用来跑 Home Assistant 和一些自动化脚本，2GB 内存有时候会触发 OOMkiller，得加到 4GB 才稳。
  * **跑 Windows 10** ：老实说不太推荐在 NAS 上跑 Windows。最低要求 2核4GB，但实际使用中至少要 4核8GB 才不会卡。而且 Windows 的虚拟化体验很差，远程桌面还好，图形界面就别想了。
  * **跑 Windows 11** ：需要 TPM 2.0，NAS 虚拟机基本不支持，放弃吧。



## 三、网络模式选哪个？

![网络模式对比](/images/nas-virtual-machine/network.jpg)

虚拟机网络有三个模式，各有用途：

**桥接模式（Bridged）** ：虚拟机获得独立 IP，和物理机平起平坐。适合需要虚拟机直接暴露在局域网的应用，比如软路由。我的 OpenWrt 就用桥接模式，LAN 口接交换机，WAN 口接光猫，NAS 本身通过另一个网口上网。

**NAT 模式** ：虚拟机通过宿主机转发上网，无法被局域网其他设备直接访问。适合跑临时测试环境，或者不需要暴露给外网的服务。

**Host-Only** ：虚拟机只能和宿主机通信，完全隔离。适合高安全要求的测试环境。

## 四、实战：飞牛 OS 跑 OpenWrt 软路由

这是我最常用的场景。把 NAS 变成软路由，一机两用。

**步骤1：下载 OpenWrt 镜像**

从 OpenWrt 官网下载 x86/64 的磁盘镜像（ext4.img.gz），解压备用。注意选带 LuCI（Web 管理界面）的版本。

**步骤2：创建虚拟机**

飞牛 OS 虚拟机向导创建，选择 Linux 系统，分配 1核1GB，硬盘选 8GB 就够。网络选桥接模式。

**步骤3：导入镜像**

用 dd 命令或者直接用飞牛 OS 的导入功能，把 img 文件写入虚拟硬盘。

**步骤4：配置网络**

![软路由配置](/images/nas-virtual-machine/router.jpg)

OpenWrt 默认的 LAN 口是 eth0，登录 192.168.1.1 修改：

  * LAN 口桥接 NAS 所在的物理网口
  * 新增 WAN 口接光猫/上层路由
  * WAN 口设 DHCP 自动获取或 PPPoE 拨号
  * 开启 NAT 和防火墙



我跑了半年，稳定得很。温度比单独买软路由低（NAS 风扇吹着），功耗基本没增加，而且可以用 NAS 的 Docker 同时跑广告屏蔽（AdGuard Home）和科学上网。

## 五、性能实测：虚拟机 vs Docker vs 原生

很多人问虚拟机性能损耗有多少。实测数据：

方案| Cinebench分数| 备注  
---|---|---  
Bare Metal（裸机）| 100%| J4125 四核  
Docker 容器| 97-99%| 几乎无损耗  
虚拟机（1核）| 70-85%| 取决于负载类型  
虚拟机（全核）| 90-95%| 需要 NUMA 优化  
  
结论：CPU 密集型任务有损耗，但对 NAS 上的常见服务影响不大。网卡直通模式下，虚拟机几乎能达到原生性能。

## 六、什么情况选虚拟机，什么情况选 Docker？

这是我在帮朋友配 NAS 时最常被问到的问题。我的判断标准：

选 **Docker** ：跑 Linux 服务（Nextcloud、Jellyfin、Home Assistant）、需要频繁更新镜像、开发测试环境。

选 **虚拟机** ：跑 Windows/macOS、需要完整内核权限、跑 PfSense/OpenWrt 软路由、跑需要特殊内核参数的应用（如 NAS 上跑 macOS虚拟机装 Xcode）。

NAS 硬件虚拟化支持有限，别指望跑什么重负载。把它当作补充工具就好。

* * *

_数据来源：个人实测（J4125，8GB RAM，飞牛 OS v1.2.0），群晖VMM官方文档，威联通Virtualization Station手册（2026年4月）_
