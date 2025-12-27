---
layout: post
title: "角态验证1-配置pybinding环境"
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