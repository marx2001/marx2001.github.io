---
layout: post
title: "wannierberrier的安装"
subtitle: "Experience Sharing"
background: '/img/bg-sci-note.jpg'
categories: sci-note
permalink: /sci-note_posts/20251203-wann
---


# <center>Recording​</center>


### 1. 步骤
创建虚拟环境
```shell
conda create -n wannier-sym python=3.12.12 -y

conda activate wannier-berrier
```
使用清华源安装
```shell
pip install wannierberri -i https://pypi.tuna.tsinghua.edu.cn/simple

http://mirrors.aliyun.com/pypi/simple/       # 阿里云 
https://pypi.mirrors.ustc.edu.cn/simple/     # 中国科技大学  
http://pypi.douban.com/simple                # 豆瓣  
https://pypi.python.org/simple/              # Python官方
http://pypi.v2ex.com/simple/                 # v2ex 
http://pypi.mirrors.opencas.cn/simple/       # 中国科学院  
https://pypi.tuna.tsinghua.edu.cn/simple/    # 清华大学 
```

查看所需依赖的版本：

```shell

python3 --version Python 3.14.0

```