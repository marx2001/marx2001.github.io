# 申博研究工作手册（LaTeX 工程）

本目录把博客 `_posts` 中 front matter 分类为 `sci-note` 的研究文章整理为一册独立 PDF，不修改个人博客页面。文件名标记为“论文阅读”的文章不纳入手册。正文采用“模块（Part）→专题（Chapter）→文章（Section）→文内标题”的层级结构，共分为五个模块：

1. 第一性原理计算教程
2. 软件学习
3. 论文使用代码
4. 绘图流程与代码
5. 心得笔记

长代码从正文抽离到代码附录；正文保留原文说明、公式、表格和图片，并在原位置给出代码清单编号。每篇文章的信息栏只显示日期和在线原文。`classification.json` 会列出收录文章的分类依据、代码块数和图片数，便于人工复核。

## 模板

采用 [ElegantBook](https://github.com/ElegantLaTeX/ElegantBook) 作为书籍模板。该项目面向 LaTeX 书籍写作、原生支持中文；本工程附带 `elegantbook.cls`，其许可为 LPPL 1.3c 或更高版本。模板文件保留原版权与许可声明。

## 编译

在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\phd-application-handbook-tex\build.ps1
```

构建流程会调用 Jekyll 读取文章元数据并用站点同一 Markdown 转换器生成正文，然后用 XeLaTeX 编译两遍。中间文件位于 `tmp/pdfs/phd-application-handbook/`；最终文件位于：

`output/pdf/phd-application-research-handbook.pdf`

## 主要文件

- `collect_sci_note.rb`：严格依据 front matter 收集 `sci-note` 文章。
- `build_handbook.py`：分类、HTML→LaTeX、图片归档与代码附录生成器。
- `classification.json`：收录文章的分类清单（每次构建自动更新）。
- `research_handbook.tex`：自动生成的完整 TeX 源文件。
- `elegantbook.cls`：ElegantBook 模板类。
