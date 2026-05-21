---
title: "NAS电子书管理：Calibre-Web搭建个人数字图书馆"
publishDate: '2024-01-01'
description: "NAS电子书管理：Calibre-Web搭建个人数字图书馆 - halo的技术博客"
tags:
  - 硬件数码
language: Chinese
---

NASCalibre电子书数字图书馆阅读

我的电子书架上有 3000 多本书，小说、技术书籍、有声书、漫画，什么都有。以前存在电脑里，每次找书都要靠记忆文件名；后来全搬到了 NAS 上，用 Calibre-Web 管理，体验完全不一样了。这篇文章说说我怎么搭的这套系统，以及踩过的坑。

![书架上的书籍](https://blog.halo26812.eu.org/images/nas-ebook-library/books.jpg)

## Calibre 是什么？书库管理的基础

Calibre 是一款开源的电子书管理软件，支持 epub、mobi、azw3、PDF 等几乎所有格式。Google Books、豆瓣、ISBN 都能自动抓取元数据和封面。

它的逻辑是：所有电子书集中存在一个"书库"文件夹里，Calibre 在书库旁边创建数据库文件，记录每本书的元数据（书名、作者、出版社、出版年份、简介、封面等）。

这个设计的好处是：书库文件夹可以直接挂载到 NAS 上，用任何设备通过网络访问，不需要依赖 Calibre 软件本身。

### 书库文件夹结构
    
    
    Calibre 书库/
      a/          # 按作者首字母分目录
        author_A/
          book1.epub
          book2.mobi
        author_B/
          book3.azw3
      metadata.db  # Calibre 数据库文件（不要动）
      .calibre/    # 配置文件夹
    

## Calibre-Web：浏览器里的私人图书馆

Calibre 本身是桌面软件，想在手机和网页上访问？Calibre-Web 就是这个需求的答案。

Calibre-Web 是一个基于 Web 的 Calibre 书库浏览器，界面清爽，搜索方便，支持中文。在 NAS 上用 Docker 跑一个，随时随地访问你的整个书架。

### 部署方法（Docker Compose）
    
    
    version: '3'
    services:
      calibre-web:
        image: linuxserver/calibre-web
        container_name: calibre-web
        restart: unless-stopped
        ports:
          - "8083:8083"
        volumes:
          - ./config:/config
          - /你的/书库路径:/books
        environment:
          - TZ=Asia/Shanghai
    

启动后在浏览器打开 http://NAS地址:8083，默认账号 admin，密码 admin123，登录后第一件事是改密码。

第一次设置时，要把书库路径指向你的 Calibre 书库文件夹（在 NAS 的共享目录里）。Calibre-Web 会读取 metadata.db，直接加载你已有的书库，不需要重新录入。

![图书馆](https://blog.halo26812.eu.org/images/nas-ebook-library/library.jpg)

## 格式转换：epub/mobi/azw3 互转

电子书格式是个坑，Kindle 用 azw3/mobi，微信读书用 epub，平板看 PDF，各有各的格式。

Calibre 自带格式转换工具，只要你的书库里有 Calibre，它就能自动转换。转换时选好目标格式，调整一下排版参数（边距、字号、图片质量），几秒钟就能出结果。

支持转换：epub ↔ mobi ↔ azw3 ↔ PDF，以及 txt、rtf、html 等。

具体转换操作：Calibre 主界面选中书籍 → 右键 → 转换书籍 → 选择输出格式 → 开始转换。转换后的文件会保存在书库同一个文件夹里。

## 微信读书 / Kindle 对接：Send to Kindle 服务

Calibre-Web 有个"Kindle 推送"功能，填入你的 Kindle 邮箱地址（kindle@free.kindle.com 或者你的私人 Kindle 邮箱），选定书籍，一键推送到 Kindle。

但微信读书不支持直接推送，需要借助第三方工具。推荐 Calibre 的"微信读书助手"插件，可以把 Calibre 书库里的书同步到微信读书。插件安装在 Calibre 桌面版里，配置好账号之后就能用了。

另外还有个叫 Koreader 的开源阅读器，支持 Calibre 同步功能，在 KOReader（Kindle 越狱固件）上体验很好。

## 有声书支持：Calibre 的有声书整理

Calibre 原生支持把有声书（mp3、m4a、m4b 格式）纳入书库管理。元数据录入方式和电子书一样，封面对应有声书封面，描述填好就行。

不过 Calibre-Web 的有声书播放功能比较基础，依赖浏览器本身。如果对有声书播放要求高，可以搭配 Emby 或者 Jellyfin（视频/音频服务器）来实现更好的播放体验。

我的做法是：电子书用 Calibre-Web，有声书单独用 AudioStation（群晖）或 Plex 处理，两套系统各司其职。

![阅读](https://blog.halo26812.eu.org/images/nas-ebook-library/reader.jpg)

## 手机阅读 App 推荐

Calibre-Web 本身有网页版，但手机上看 PDF 和长篇电子书还是需要一个专门的阅读 App。以下是我用过的几个：

### 静读天下（Moon+ Reader）

Android 上最好的阅读 App 之一，支持 epub、txt、pdf、mobi、azw3 所有格式。内置书城（可以绑定 NAS 的 Calibre-Web 作为个人书库），自定义字体、背景、翻页动画，功能极其丰富。

缺点是 pro 版收费（约 30 元），但值得。

### 多看阅读（Duokan）

国产老牌阅读器，对中文排版优化得很好，适合看网文和小说。支持 Calibre 书库同步（需要先在 Calibre 里设置"多看阅读"插件）。

iOS 和 Android 都有，功能不如静读天下丰富，但排版体验很舒服。

### Kindle App

如果你有 Kindle，直接用 Kindle App，通过 Calibre 推送的 azw3/mobi 格式，体验最稳定。Amazon 的生态系统封闭但成熟，Calibre 的 Send to Kindle 功能用好了其实很顺手。

## 书籍整理技巧：作者系列、年份、体裁分类

书多了必须靠分类，不然根本找不到。我整理书库有几个原则：

  1. **按作者分目录** ：Calibre 默认按作者首字母分，一目了然
  2. **用标签而非文件夹** ：一本《三体》既是科幻又是刘慈欣又是中国文学，放在文件夹只能选一个，放标签能同时标记三个，方便搜索
  3. **系列书用自定义列** ：Calibre 支持自定义元数据列，我给每个系列设置了"系列名"和"系列序号"，搜索系列时直接筛选
  4. **出版年份做辅助筛选** ：同一年出版的多本书，容易和当年的大事回忆挂钩，加个年份标签方便回溯



## 我的藏书量和查找体验

我的书库目前 3000+ 本，刚建的时候觉得分类会很复杂，用了两个月之后发现 Calibre 的搜索比我想象的好用。

最常用的搜索方式：书名关键词（模糊匹配）、作者名、标签。Calibre-Web 的搜索支持正则表达式，用熟了之后查找效率很高。

踩过的坑：

  * 不要手动改 metadata.db，一旦损坏书库就废了
  * 导入新书时养成习惯顺手补全元数据，拖得越久越不想补
  * 封面图片太大（>500KB）的可以压缩，不然书库加载会慢



## 总结

如果你藏书量超过 500 本，电脑里文件夹越整越乱，Calibre + Calibre-Web 是目前最优的解决方案。一次性搭好，之后就是往书库里扔书、自动整理、随时随地阅读。

投入成本：一台能跑 Docker 的 NAS（2000 元左右）+ 一个晚上配置时间。长期收益：告别各平台分散的书库，拥有一个完全属于自己、没有任何广告的私人图书馆。
