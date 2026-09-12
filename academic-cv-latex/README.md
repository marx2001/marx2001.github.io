# 马睿骁学术简历 LaTeX 工程

本工程以 `D:\3_申博\2.docx` 为基础内容，使用 GitHub 上的 [Awesome-CV](https://github.com/posquit0/Awesome-CV) 模板整理为个人学术简历。

## 文件结构

- `cv.tex`：主文件和个人信息
- `sections/`：研究方向、教育背景、研究经历、技能、实践经历、奖项和校园服务
- `awesome-cv.cls`：Awesome-CV 文档类
- `build.ps1`：Windows 编译脚本
- `马睿骁-学术简历.pdf`：编译结果

## 编译方法

在 PowerShell 中运行：

```powershell
.\build.ps1
```

模板使用 XeLaTeX，不支持直接改用 pdfLaTeX。编译脚本会先生成目录与版面信息，再输出最终 PDF。

## 模板来源

- 仓库：https://github.com/posquit0/Awesome-CV
- 本次采用版本：`8b850b477803a929a6dd74a74e3d5ab6b735d869`
- 许可证：LPPL 1.3c，原许可证文件保存在 `LICENCE`
