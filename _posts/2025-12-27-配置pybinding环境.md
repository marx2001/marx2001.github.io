---
layout: post
title: "角态验证1-配置pybinding环境-Ubuntu系统配置"
subtitle: "Experience Sharing"
background: '/img/bg-sci-note.jpg'
categories: sci-note
permalink: /sci-note_posts/20251227-pybinding
---

## <center>步骤</center>

　　windows环境下，需要下载cmake，和Visual Studio Build Tools：
```shell
https://cmake.org/download/#latest
https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/
```
勾选：

```shell

☑ Add CMake to the system PATH for all users

☑ C++ build tools

☑ MSVC v143

☑ Windows 10/11 SDK

```

创建环境

```shell

conda create -n pybinding_env python=3.9 -y
conda activate pybinding_env

```

安装miniconda后，运行

```shell

conda install numpy scipy matplotlib

```

然后

```shell

pip install pybinding

```

安装了Cmake和visual Studio installer仍然不能成功安装pybinding，尝试使用Linux环境，安装Ubantu

## <center>安装Ubantu</center>

使用DiskGenius V4.6.4.2 x64专业版 ,选择有空余空间的磁盘，拖动棕色进度条，留出合适的空间。

使用Rufus 4.7.2231，写入Ubuntu镜像文件到安装u盘中，U盘要求16G以上。