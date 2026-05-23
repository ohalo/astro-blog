---
title: "Terminal生产力：程序员都应该掌握的Shell工作流"
publishDate: '2026-05-23'
description: "Terminal生产力：程序员都应该掌握的Shell工作流 - halo的技术博客"
tags:
  - 其他
language: Chinese
---

Terminal生产力：程序员都应该掌握的Shell工作流

2016年我刚工作的时候，Terminal对我而言就是"cd + ls + git"三件套。IDE能搞定的事，我绝不进Terminal。后来项目需要处理日志文件，几百兆的文本，用IDE打开直接卡死，"用grep吧"——同事丢给我一个命令，我照着敲，跑出来了，但完全不理解背后的逻辑。

那是我重新认识命令行的起点。到现在，我的日常开发里Terminal的使用频率已经超过了任何GUI工具。这篇文章分享的是我从"会用ls"到"用命令行解决实际问题"这条路上，积累下来最实用的工作流。

![Shell的真正威力在于组合：管道让简单命令完成复杂任务](/images/terminal-productivity/terminal-workflow.jpg)

## 管道和重定向：Unix设计哲学的精髓

如果只让我教一个人一个命令行概念，我会选管道。管道（|）的作用是把前一个命令的输出变成后一个命令的输入。这种"组合"的思想是Unix设计的精髓，也是命令行效率远超GUI的根本原因。

举几个我每天都在用的例子：

`git log --oneline | head -20` — 看最近20条commit，输出简洁到一行一条

`ps aux | grep node | grep -v grep` — 查找运行中的Node进程，剔除grep自身

`cat access.log | awk '{print $7}' | sort | uniq -c | sort -rn | head -20` — 找出访问量最高的20个URL

第三个例子展示了管道的威力：这个命令从日志文件中提取访问路径、排序、统计、去重、再按访问量排序，输出访问量最高的前20个URL。同样的功能用GUI工具需要几步？打开文件、导出Excel、用透视表——Terminal里一个管道就搞定了。

![管道让每个命令专注做一件事，通过组合完成复杂任务](/images/terminal-productivity/pipe-diagram.jpg)

## find：精准定位文件的艺术

find是我从"会用ls"到"真正用好命令行"的转折点。find的能力远不止"找文件"这么简单——它可以按时间、按大小、按权限、按文件名模式查找，找到后还能接续执行其他命令。

几个我高频使用的场景：

找过去24小时内修改过的文件：`find . -type f -mtime -1`

找大于100MB的文件：`find . -type f -size +100M`

找所有node_modules目录并删除：`find . -name 'node_modules' -type d -exec rm -rf {} +`

找所有TS文件但排除test文件：`find . -name '*.ts' -not -name '*.test.ts'`

最后这个例子有一个细节：`-not -name`用来排除匹配。这在清理测试文件、备份文件的时候特别有用。

## grep：文本搜索的瑞士军刀

grep的用途很简单：在文件里搜索包含特定文本的行。但它的参数让搜索变得极其灵活。

`grep -r "TODO" --include="*.ts" .` — 在所有TS文件里递归搜索TODO注释

`grep -n "function" server.js` — 加上行号，方便定位

`grep -E "error|warning|critical" app.log` — 用正则同时匹配多个关键词

`grep -v "^#" config.toml | grep -v "^$"` — 去掉注释行和空行，只看有效配置

最后一个用法是我处理配置文件时的常用技巧。很多配置文件用#做注释，用空行做分隔，直接cat出来有很多干扰信息，加两个grep就能过滤干净。

## awk：结构化文本处理

awk在入门阶段容易被忽视，因为它的语法看起来有点奇怪。但一旦掌握了基本用法，它就是处理表格数据的神器。

AWK的工作原理是：逐行处理文本，每行按分隔符切分成字段，然后对每个字段执行指定的操作。默认的分隔符是空格，字段用$1、$2、$3引用，$0代表整行。

我用一个实际场景说明：有一次需要从一段日志里提取用户名和请求路径，原始数据是这样的：

`[2026-05-23 10:15:32] User:zhangsan Path:/api/users Endpoint:GET`

用awk提取用户名和路径：`awk -F'User:| Path:' '{print $2, $3}' access.log`

这个命令用-F指定了两个分隔符（"User:"和" Path:"），然后打印第2和第3个字段。

更复杂的例子：统计每个用户的请求次数。`cat access.log | grep "User:" | awk -F'User:' '{print $2}' | awk '{print $1}' | sort | uniq -c | sort -rn`

![awk的字段处理模型：每行按分隔符切分后，可以独立操作每个字段](/images/terminal-productivity/awk-processing.jpg)

## xargs：让命令消费列表

xargs的作用是把管道传来的列表转换成命令的参数。这在需要对一批文件执行同一操作的时候特别有用。

最常见的场景：find找到一堆文件，然后删除它们。

错误做法：`find . -name "*.tmp" | rm` — rm不会从管道读取参数

正确做法：`find . -name "*.tmp" | xargs rm` — xargs把find的输出变成rm的参数

危险的场景：文件名有空格的时候，xargs可能会把一个文件名切成多个参数。加一个-0参数可以处理这种情况：`find . -name "*.tmp" -print0 | xargs -0 rm`

另一个高频用法：并行执行。`cat urls.txt | xargs -I{} -P 10 curl -O {}` — -P 10表示最多10个并发，-I{}设置占位符代表每行输入。

## sed：流式文本编辑

sed是stream editor的缩写，用来对文本进行流式转换。最常用的功能是查找替换。

`sed 's/old/new/g' file.txt` — 把文件中所有old替换成new

`sed -i 's/old/new/g' *.js` — -i表示 in-place，直接修改文件

`sed -n '10,20p' file.txt` — 提取第10到20行

sed的正则替换非常强大。比如要把所有形如 `component: 'xxx'` 的配置改成 `component: "xxx"`（单引号变双引号）：`sed "s/component: '\([^']*\)'/component: \"\1\"/g"`

但坦白说，sed的正则表达式语法比较反人类，复杂的替换我会用awk或者Python。但简单的查找替换，sed是最快的。

## tmux：终端多窗口管理

tmux不是文本处理工具，但它彻底改变了我的终端使用方式。它允许在同一个终端里创建多个"面板"，每个面板可以运行不同的命令，面板可以分屏、可以独立滚动、可以detach（后台运行）。

我的tmux工作流是这样的：打开tmux，创建多个窗口——一个写代码、一个跑服务、一个查日志、一个看文档。需要切换的时候用快捷键，不需要开多个Terminal窗口。

更重要的是tmux的session管理。跑一个长时间任务的时候，我习惯开一个tmux session，把任务跑在里面，然后detach。这样即使ssh断开，任务也会继续跑，下次ssh进来可以重新attach回来。

![tmux让终端变成真正的多任务工作空间，session可以detach/reattach](/images/terminal-productivity/tmux-session.jpg)

## 把它们组合起来：真实工作场景

单个命令用处有限，把它们组合起来才能发挥最大威力。分享几个我的真实工作场景：

**场景1：分析API接口性能。**我需要从nginx日志里找出响应时间最慢的20个请求。

`cat access.log | awk '{print $7, $NF}' | grep "api" | sort -k2 -rn | head -20`

假设日志格式是：请求路径 响应时间（毫秒），这个命令过滤出api请求，按响应时间倒序排列，取前20。

**场景2：清理Git仓库里不需要的大文件。**有时候不小心把一个大文件commit了，删掉之后历史记录里还有，每次clone都要下载。

`git filter-branch --tree-filter 'rm -f large-file.zip' HEAD`

这个命令会重写历史，删除所有commit里的大文件。之后可以安全地推送了。

**场景3：批量重命名。**把所有 `IMG_*.jpg` 改成 `photo-*.jpg`。

`for f in IMG_*.jpg; do mv "$f" "${f/IMG_/photo-}"; done`

这是bash的字符串替换语法，${f/old/new}会把第一个匹配的old换成new。

## 快速参考：最常用的20个命令参数组合

总结一下我使用频率最高的命令参数，供参考：

- `ls -lh` — 可读性友好的文件大小

- `ls -lt` — 按修改时间排序

- `find . -name "*.log" -mtime +7` — 7天前的日志文件

- `grep -rn "TODO" --include="*.ts"` — 递归搜索TS文件

- `grep -E "error|warn" log.txt` — 同时匹配多个词

- `awk -F',' '{print $1, $3}' data.csv` — CSV取特定列

- `sort -u` — 排序并去重

- `uniq -c` — 统计重复行次数

- `cat file | wc -l` — 统计行数

- `xargs -I{} cmd {}` — 占位符模式

- `sed -i 's/old/new/g' *.txt` — 批量替换

- `tar -czf archive.tar.gz dir/` — 压缩目录

- `tar -xzf archive.tar.gz` — 解压

- `ssh -L 8080:localhost:3000` — 本地端口转发

- `curl -I url` — 只看HTTP头

- `du -sh *` — 各目录大小

- `ps aux | grep node` — 查找进程

- `kill -9 pid` — 强制结束进程

- `chmod +x script.sh` — 加执行权限

- `tmux new -s work` — 创建命名session

## 写在最后

命令行不是一个"炫技"的工具，它的价值在于让你用最少的操作完成最多的工作。我现在的原则是：如果一个任务要点击超过3次GUI，那就应该用命令行来解决。

学习曲线确实存在。我记得刚开始用grep的时候，正则表达式记不住，参数记不住，管道也经常写错。但这个投入是值得的——一旦熟练了，效率的提升是数量级的。

最后一点建议：不要试图一次性学完所有命令。用到的时候查，用多了自然就记住了。我现在还有一些命令的细节记不住，但我知道它们存在，需要的时候查一下就好。关键是建立"这个问题可以用命令行解决"的意识，然后你会慢慢找到适合你的命令组合。