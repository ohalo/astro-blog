---
title: "NAS与智能家居联动：Home Assistant接入米家全攻略"
publishDate: '2026-04-08'
description: "NAS与智能家居联动：Home Assistant接入米家全攻略 - halo的技术博客"
tags:
  - 硬件数码
language: Chinese
---

NAS智能家居HomeAssistant米家物联网自动化

智能家居品牌太多，各家 App 互不兼容，买了一堆设备却要用一堆 App 控制？Home Assistant 就是来解决这个问题的。把它跑在 NAS 上，24 小时运行，统一管理所有品牌的设备，一个界面管全屋。聊聊我的搭建经验。

## 一、Home Assistant 是什么？

![智能家居](/images/nas-smart-home/smart-home.jpg)

Home Assistant（简称 HA）是一个开源的智能家居中枢，支持接入市面上几乎所有主流智能设备品牌：米家、天猫精灵、Yeelight、绿米、飞利浦 Hue、宜家 TRÅDFRI、SONOS……基本上你能想到的它都支持。

它有两个核心能力：

  * **统一控制** ：一个界面管所有设备，不需要切换 N 个 App。
  * **自动化** ：设备之间可以联动，比如"门锁打开"→"走廊灯亮"→"空调调到 24 度"。



## 二、为什么用NAS跑Home Assistant？

Home Assistant 官方推荐用树莓派，但用 NAS 跑有几个明显优势：

![Home Assistant界面](/images/nas-smart-home/home-assistant.jpg)

  * **24 小时运行** ：NAS 本来就是 24 小时开着的，不需要额外设备。
  * **低功耗** ：NAS 待机功耗本来就低，跑 HA 几乎不增加电费。
  * **数据存储** ：HA 的历史数据、日志、录像都可以直接存在 NAS 上，不需要外接 U 盘或者 SD 卡。
  * **稳定可靠** ：NAS 的硬件比树莓派稳定多了，不会因为 SD 卡损坏而丢数据。



## 三、Docker安装Home Assistant

最简单的方式是用 Docker 安装：
    
    
    docker run -d \
      --name homeassistant \
      --restart unless-stopped \
      --network host \
      -v /path/to/ha_config:/config \
      -v /run/dbus:/run/dbus:ro \
      --privileged \
      homeassistant/home-assistant:latest

飞牛 OS 和群晖都可以在 Docker 管理界面里图形化安装，不需要记命令。安装包大概 1GB，第一次启动需要等几分钟初始化。

安装完成后，浏览器打开 http://NAS_IP:8123，按引导设置即可。

## 四、接入米家设备

![智能家居面板](/images/nas-smart-home/smarthome-panel.jpg)

这是大多数人最需要的场景。米家设备接入 HA 有几种方式：

**方式1：Xiaomi Gateway 3（推荐）**

小米多模网关支持的设备，可以通过本地网络直接接入 HA，不需要云端。配置方法：

  1. 在 HA 的"集成"页面搜索"Xiaomi"，添加 Xiaomi Miio 集成
  2. 填入网关的 IP 地址和 token（token 需要在米家 App 里获取）
  3. HA 会自动发现网关下的所有子设备



这种方式最稳定，响应速度快，而且断网时也能用。

**方式2：HACS + Xiaomi Miot Auto**

如果你的设备不在 Xiaomi Miio 支持列表里，可以安装 HACS（Home Assistant Community Store），然后通过 Xiaomi Miot Auto 插件接入更多设备。

token 获取方法：在米家 App 里长按设备→设备信息→点击"micolinki" 5次→出现 token。或者用 MiHome Tools App 直接导出。

## 五、实用自动化场景

接入只是第一步，真正有用的是自动化。我自己常用的几个场景：

**场景1：回家模式**

门锁检测到指定指纹开门（我的指纹）→ 打开走廊灯 → 打开客厅灯 → 空调调到 24 度 → 播放欢迎语音。

**场景2：离家模式**

门锁上提反锁 → 关闭所有灯 → 关闭所有空调 → 开启安防模式（摄像头开始录像）→ 发送通知到手机。

**场景3：影院模式**

按下投影仪遥控器（通过红外传感器检测）→ 关闭客厅主灯 → 打开氛围灯 → 放下投影幕布 → 打开功放。一步到位。

**场景4：安防告警**

门窗传感器检测到异常开启（安防模式激活时）→ 摄像头截图 → 发送告警到微信 → 响铃威慑。

## 六、Home Assistant vs Apple HomeKit vs 米家

三个方案各有优劣：

方案| 优点| 缺点  
---|---|---  
Home Assistant| 品牌全覆盖，自动化最强，完全本地| 配置复杂，需要一定学习成本  
Apple HomeKit| 隐私好，界面美观，Siri控制| 设备兼容少，价格贵  
米家| 设备丰富，价格便宜，生态完整| 云端依赖，重度隐私担忧  
  
我的方案：NAS 跑 HA，HA 统一接入米家设备，苹果 HomeKit 通过 HA 桥接插件接入，iPhone 上的 Home App 可以直接控制米家设备。两边生态的好处都占了。

## 七、长期运行稳定性

HA 在 NAS 上跑了一年半，稳定性还不错。需要注意几个问题：

  * **内存占用** ：HA 默认占用 500MB-1GB 内存，加载设备多了会增加到 2GB。注意 NAS 总内存够不够。
  * **数据库大小** ：HA 的历史数据默认用 SQLite，时间久了会很大。定期清理或者换用 MariaDB。
  * **HA Core 升级** ：升级前最好做快照备份，升级失败回退比较麻烦。



目前 HA 的 Home Assistant OS 版本对 NAS 支持最好，建议用虚拟机安装（HA OS），比 Docker 版更稳定。如果你的 NAS 内存 8GB 以上，跑 HA OS 虚拟机完全没有压力。

* * *

_数据来源：Home Assistant 官方文档，Xiaomi Miot Auto GitHub，个人实测（飞牛 OS，J4125，8GB RAM，2026年4月）_
