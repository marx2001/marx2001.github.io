---
layout: post
title: "(原创脚本)角态验证1-配置pybinding环境-Ubuntu系统配置"
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

12.28 用ubuntu系统成功安装了pybinding，用conda安装，先创建环境，然后用conda安装前置库，然后再安装pybinding。

在ubuntu中也不能直接用pip安装pybinding，只能用conda 安装0.9.5稳定版的pybinding。

安装完pybinding之后，由于是ipynb格式，需要jupyter notebook使用的kernel与安装的pybinding环境一致，这样代码才能在该环境中找到所需要的包并运行。

先激活环境，然后安装ipynernel，运行命令安装ipykernel并添加到jupyter可用的knernel列表中：

```shell

conda activate pybinding_env
conda install -n pybinding_env ipykernel --update-deps --force-reinstall
python -m ipykernel install --user --name pybinding_env --display-name "Python (pybinding_env)"

```
这个命令会将pybinding_env环境注册为一个jupyter kernel，名称为Python(pybinding_env)，可根据实况修改名称。

重新启动jupter notebook，已经打开的话需要重新关闭，在创建或打开notebook时，在kernel菜单中选择刚刚添加的python(pybinding_env) kernel，这样就可以分模块运行代码。


上述是用conda安装0.9.5稳定版的pybinding，目前已经有包名为pybinding-dev的预发行版本，是1.0.6版本的pybinding，安装流程如下:

创建Python3.10版本以上的环境

```shell
conda create -n pybinding python=3.10
conda activate pybinding
conda install numpy matplotlib scipy
python -m pip install -U "pip<25.3"
pip install pybinding-dev==1.0.6 --no-use-pep517 --no-cache-dir
pip show pybinding-dev
conda install -n pybinding ipykernel --update-deps --force-reinstall
```
如果是新配置的vscode，可能没有安装python扩展，根据右下角vscode的提示，安装扩展即可。