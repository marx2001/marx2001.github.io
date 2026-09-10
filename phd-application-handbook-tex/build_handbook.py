#!/usr/bin/env python3
"""Build a PhD-application research handbook from every sci-note post."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from lxml import etree
from lxml import html as lxml_html
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
TEX_DIR = Path(__file__).resolve().parent
COLLECTOR = TEX_DIR / "collect_sci_note.rb"
TEX_FILE = TEX_DIR / "research_handbook.tex"
MANIFEST_FILE = TEX_DIR / "classification.json"
ASSET_DIR = TEX_DIR / "assets"
CODE_DIR = TEX_DIR / "code"
COVER_SOURCE = ROOT / "img" / "bg-sci-note.jpg"

PART_META = {
    "first-principles": {
        "title": "第一性原理计算教程",
        "intro": "集中呈现 VASP 结构优化、能带、态密度、磁交换参数与 DMI 等基础计算流程，形成从建模到结果验证的第一性原理方法主线。",
        "subgroups": ["VASP 基础计算", "磁性参数与 DMI", "结构建模与计算方法"],
    },
    "software-learning": {
        "title": "软件学习",
        "intro": "独立整理科研软件与工具链的安装、配置、参数逻辑和测试记录，覆盖 Wannier90、TB2J、pyw90、pythTB、WannierTools、VASP2KP、IRVSP、pybinding、Materials Studio、LaTeX 与网站维护。",
        "subgroups": ["Wannier90 与电子结构工具", "紧束缚、拓扑与磁性软件", "科研写作、建模与网站工具"],
    },
    "paper-code": {
        "title": "论文使用代码",
        "intro": "汇集论文最终代码、原创研究程序、高通量自动化、紧束缚模型、超超交换验证与机器学习相界搜索，突出成果复现和研究创新。",
        "subgroups": ["论文最终代码与结果复现", "原创研究与机器学习", "高通量、自动化与模型构建"],
    },
    "plotting": {
        "title": "绘图流程与代码",
        "intro": "覆盖能带、轨道投影、自旋密度、Berry 曲率、3D MAE、斯格明子与原子结构渲染，突出从原始数据到论文级图件的完整流程。",
        "subgroups": ["能带、轨道与电子结构可视化", "拓扑、磁性与三维场可视化", "结构建模与三维渲染"],
    },
    "research-notes": {
        "title": "心得笔记",
        "intro": "集中收录复现体会、计算调试、方法反思与阶段性研究总结，展示研究问题如何被发现、分析和修正。",
        "subgroups": ["计算调试与方法反思", "阶段心得与研究总结"],
    },
}
PART_ORDER = list(PART_META)

PLOTTING_KEYWORDS = (
    "绘制3dmae", "绘制自旋密度", "快速绘制轨道投影能带", "pyprocar绘制能带",
    "mlwfs绘制轨道", "绘制精美的投影能带", "计算贝里曲率",
    "blender绘图", "blender绘制", "论文图集",
)
EXCLUDED_ARTICLE_KEYWORDS = ("论文阅读",)
CORE_RESEARCH_KEYWORDS = (
    "2026-09-09", "论文最终代码", "机器学习", "optuna", "tb模型", "超超交换",
    "量子自旋霍尔", "自旋陈数", "边缘态与霍尔响应", "拓扑相界", "计算磁参数j",
)
SOFTWARE_KEYWORDS = (
    "wannier", "pyw90", "tb2j", "wannier tools", "wanniertools", "wannierberrier",
    "wannierberri", "wannsym", "wansymm", "pythtb", "vasp2kp", "ir2tb", "irvsp",
    "pybinding", "material studio", "materials studio", "申博latex", "操作手册",
    "扩展导航栏", "本地预览网页", "更新翻页", "重大更新", "外观更新", "网站维护",
)
NOTE_KEYWORDS = (
    "心得", "笔记", "方法论", "数值调试", "前置知识", "归一化问题", "复现文献", "系数推导",
)
PAPER_CODE_KEYWORDS = (
    "脚本", "代码", "python解", "高通量", "转角材料构建", "判断团簇",
    "快速计算投影能带", "kp构造",
)
LANG_EXT = {
    "python": "py", "py": "py", "bash": "sh", "shell": "sh", "sh": "sh",
    "powershell": "ps1", "javascript": "js", "js": "js", "ruby": "rb",
    "json": "json", "yaml": "yml", "yml": "yml", "html": "html", "css": "css",
    "latex": "tex", "tex": "tex", "fortran": "f90", "c": "c", "cpp": "cpp", "matlab": "m",
}
ESCAPE_MAP = {
    "\\": r"\textbackslash{}\allowbreak{}", "/": r"/\allowbreak{}",
    "{": r"\{", "}": r"\}", "#": r"\#", "$": r"\$", "%": r"\%", "&": r"\&",
    "_": r"\_\allowbreak{}", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
}
SPECIAL_TEX_MAP = {
    "∣": r"\ensuremath{\mid}", "∬": r"\ensuremath{\iint}", "∝": r"\ensuremath{\propto}",
    "∼": r"\ensuremath{\sim}", "∈": r"\ensuremath{\in}", "↔": r"\ensuremath{\leftrightarrow}",
    "⟨": r"\ensuremath{\langle}", "⟩": r"\ensuremath{\rangle}", "△": r"\ensuremath{\triangle}",
    "∫": r"\ensuremath{\int}", "′": r"\ensuremath{\prime}", "ʜ": "H",
}
MATH_PATTERN = re.compile(
    r"(\\\[(?:.|\n)*?\\\]|\\\((?:.|\n)*?\\\)|\$\$(?:.|\n)*?\$\$|\$(?!\s)(?:[^$\n]|\n(?!\n))*?(?<!\s)\$)",
    re.MULTILINE,
)


@dataclass
class CodeBlock:
    article_no: int
    sequence: int
    language: str
    relative_path: str
    line_count: int


@dataclass
class Article:
    source_path: str
    title: str
    subtitle: str
    date: str
    url: str
    html: str
    part: str
    subgroup: str
    reason: str
    article_no: int = 0
    tex_body: str = ""
    code_blocks: list[CodeBlock] = field(default_factory=list)
    image_count: int = 0
    missing_images: list[str] = field(default_factory=list)


def run_collector() -> list[dict]:
    command = f"bundle exec ruby {COLLECTOR}"
    proc = subprocess.run(
        ["cmd.exe", "/d", "/c", command], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode:
        raise RuntimeError(f"Jekyll 文章收集失败：\n{stderr[-4000:]}")
    try:
        posts = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Jekyll 收集器未返回有效 JSON。\n{stderr[-2000:]}\n{stdout[:1000]}") from exc
    if not posts:
        raise RuntimeError("未找到 front matter 分类为 sci-note 的文章。")
    paths = [post["source_path"] for post in posts]
    if len(paths) != len(set(paths)):
        raise RuntimeError("sci-note 文章路径存在重复，已停止生成。")
    return posts


def normalized_key(text: str) -> str:
    return re.sub(r"[\s（）()【】\[\]：:·—_\-]+", "", text).lower()


def classify(post: dict) -> tuple[str, str, str]:
    title = post["title"]
    haystack = normalized_key(title + " " + post["source_path"])
    matched = next((w for w in PLOTTING_KEYWORDS if normalized_key(w) in haystack), None)
    if matched:
        if any(w in haystack for w in ("blender", "原子结构")):
            subgroup = "结构建模与三维渲染"
        elif any(w in haystack for w in ("3dmae", "贝里曲率", "斯格明子", "自旋密度")):
            subgroup = "拓扑、磁性与三维场可视化"
        else:
            subgroup = "能带、轨道与电子结构可视化"
        return "plotting", subgroup, f"绘图关键词：{matched}"

    matched = next((w for w in CORE_RESEARCH_KEYWORDS if normalized_key(w) in haystack), None)
    if matched:
        if "最终代码" in haystack or "20260909" in haystack:
            subgroup = "论文最终代码与结果复现"
        else:
            subgroup = "原创研究与机器学习"
        return "paper-code", subgroup, f"论文研究关键词：{matched}"

    matched = next((w for w in SOFTWARE_KEYWORDS if normalized_key(w) in haystack), None)
    if matched:
        if any(w in haystack for w in ("wannier", "pyw90", "wanniertools", "wannierberri", "wannsym")):
            subgroup = "Wannier90 与电子结构工具"
        elif any(w in haystack for w in ("tb2j", "pythtb", "vasp2kp", "ir2tb", "irvsp", "pybinding")):
            subgroup = "紧束缚、拓扑与磁性软件"
        else:
            subgroup = "科研写作、建模与网站工具"
        return "software-learning", subgroup, f"软件学习关键词：{matched}"

    matched = next((w for w in NOTE_KEYWORDS if normalized_key(w) in haystack), None)
    if matched:
        if any(w in haystack for w in ("调试", "方法论", "归一化", "复现", "前置知识", "系数推导")):
            subgroup = "计算调试与方法反思"
        else:
            subgroup = "阶段心得与研究总结"
        return "research-notes", subgroup, f"心得/方法关键词：{matched}"

    matched = next((w for w in PAPER_CODE_KEYWORDS if normalized_key(w) in haystack), None)
    if matched:
        if any(w in haystack for w in ("高通量", "脚本", "构建", "判断团簇", "投影能带")):
            subgroup = "高通量、自动化与模型构建"
        else:
            subgroup = "原创研究与机器学习"
        return "paper-code", subgroup, f"研究代码关键词：{matched}"

    if any(w in haystack for w in ("dmi", "磁交换", "磁参数", "磁各向异性")):
        subgroup = "磁性参数与 DMI"
    elif any(w in haystack for w in ("结构", "建模", "晶格", "异质结", "转角")):
        subgroup = "结构建模与计算方法"
    else:
        subgroup = "VASP 基础计算"
    return "first-principles", subgroup, "教程/理论文章：纳入第一性原理方法链"


def tex_escape_plain(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for char in ("\u00a0", "\u200b", "\u200d", "\u20e3", "\ufeff", "\ufe0f"):
        text = text.replace(char, " " if char == "\u00a0" else "")
    return "".join(SPECIAL_TEX_MAP.get(char, ESCAPE_MAP.get(char, char)) for char in text)


def text_to_tex(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"[\t\r\n ]+", " ", text)
    pieces, cursor = [], 0
    for match in MATH_PATTERN.finditer(text):
        pieces.append(tex_escape_plain(text[cursor:match.start()]))
        math = match.group(0)
        if math.startswith("$$"):
            pieces.append(r"\[" + math[2:-2].strip() + r"\]")
        elif math.startswith("$"):
            pieces.append(r"\(" + math[1:-1].strip() + r"\)")
        else:
            pieces.append(math)
        cursor = match.end()
    pieces.append(tex_escape_plain(text[cursor:]))
    return "".join(pieces)


def url_for_tex(url: str) -> str:
    return (url.replace("\\", "/").replace("%", r"\%").replace("#", r"\#")
            .replace("_", r"\_").replace("{", r"\{").replace("}", r"\}"))


def safe_text_content(node: etree._Element) -> str:
    return re.sub(r"\s+", " ", node.text_content()).strip()


class ImageManager:
    def __init__(self) -> None:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        self.cache: dict[Path, str] = {}
        self.missing: Counter[str] = Counter()
        Image.MAX_IMAGE_PIXELS = None
        if COVER_SOURCE.exists():
            shutil.copy2(COVER_SOURCE, ASSET_DIR / "cover.jpg")

    def resolve(self, src: str, source_path: str) -> Path | None:
        if not src or src.startswith(("data:", "http://", "https://", "//")):
            return None
        clean = unquote(urlparse(src).path).replace("\\", "/")
        candidates = (
            [ROOT / clean.lstrip("/")] if clean.startswith("/")
            else [ROOT / clean, (ROOT / source_path).parent / clean, ROOT / "img" / clean]
        )
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(ROOT.resolve())
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                return resolved
        return None

    def stage(self, src: str, source_path: str) -> str | None:
        source = self.resolve(src, source_path)
        if source is None:
            self.missing[src] += 1
            return None
        if source in self.cache:
            return self.cache[source]
        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
        suffix = source.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".pdf"}:
            target_suffix = ".jpg" if suffix == ".jpeg" else suffix
        elif suffix in {".gif", ".webp", ".bmp", ".tif", ".tiff"}:
            target_suffix = ".png"
        else:
            self.missing[src] += 1
            return None
        target = ASSET_DIR / f"figure-{digest}{target_suffix}"
        try:
            if target_suffix == suffix or (suffix == ".jpeg" and target_suffix == ".jpg"):
                shutil.copy2(source, target)
            else:
                with Image.open(source) as image:
                    image.seek(0)
                    image = ImageOps.exif_transpose(image)
                    if image.mode not in {"RGB", "RGBA"}:
                        image = image.convert("RGBA")
                    image.save(target, "PNG", optimize=True)
        except Exception:
            self.missing[src] += 1
            return None
        rel = target.relative_to(TEX_DIR).as_posix()
        self.cache[source] = rel
        return rel


class HtmlToLatex:
    def __init__(self, article: Article, images: ImageManager) -> None:
        self.article = article
        self.images = images
        self.code_sequence = 0

    def convert(self) -> str:
        root = lxml_html.fragment_fromstring(self.article.html, create_parent="div")
        chunks, first_content = [], True
        for child in root:
            if not isinstance(child.tag, str):
                continue
            tag = child.tag.lower()
            if first_content and tag in {"h1", "h2"}:
                heading, title = normalized_key(safe_text_content(child)), normalized_key(self.article.title)
                if heading and (heading in title or title in heading):
                    first_content = False
                    if child.tail:
                        chunks.append(text_to_tex(child.tail))
                    continue
            chunks.append(self.render_node(child))
            if child.tail:
                chunks.append(text_to_tex(child.tail))
            if safe_text_content(child):
                first_content = False
        return "\n".join(piece for piece in chunks if piece and piece.strip())

    def render_children(self, node: etree._Element) -> str:
        chunks = [text_to_tex(node.text)]
        for child in node:
            chunks.append(self.render_node(child))
            chunks.append(text_to_tex(child.tail))
        return "".join(chunks)

    def code_block(self, node: etree._Element) -> str:
        self.code_sequence += 1
        code_node = node.find(".//code")
        code = (code_node if code_node is not None else node).text_content()
        code = (code.replace("\r\n", "\n").replace("\r", "\n")
                .replace("\ufe0f", "").replace("\u200d", "").replace("\u20e3", "")
                .rstrip() + "\n")
        for symbol, replacement in {"✅": "[OK]", "❌": "[X]", "❗": "[!]"}.items():
            code = code.replace(symbol, replacement)
        language = ""
        for candidate in [node] + list(node.iterancestors()):
            match = re.search(r"(?:language|lang)-([A-Za-z0-9_+#.-]+)", candidate.get("class", ""))
            if match:
                language = match.group(1).lower()
                break
        language = language or "text"
        ext = LANG_EXT.get(language, "txt")
        filename = f"{self.article.article_no:03d}-{self.code_sequence:03d}.{ext}"
        path = CODE_DIR / filename
        path.write_text(code, encoding="utf-8")
        block = CodeBlock(
            self.article.article_no, self.code_sequence, language,
            path.relative_to(TEX_DIR).as_posix(), max(1, code.count("\n")),
        )
        self.article.code_blocks.append(block)
        return (
            "\n" + r"\begin{codeindexbox}{"
            + f"代码清单 A.{self.article.article_no}.{self.code_sequence}" + "}\n"
            + rf"语言：{tex_escape_plain(language)}；完整代码见附录第 \pageref{{code:{self.article.article_no}:{self.code_sequence}}} 页，"
            + f"共 {block.line_count} 行。\n"
            + r"\end{codeindexbox}" + "\n"
        )

    def render_image(self, node: etree._Element) -> str:
        src, alt = node.get("src", "").strip(), node.get("alt", "").strip()
        staged = self.images.stage(src, self.article.source_path)
        if not staged:
            self.article.missing_images.append(src)
            return (
                "\n" + r"\begin{missingfigure}" + "\n原图未随仓库提供："
                + tex_escape_plain(src or "未标注路径") + "\n" + r"\end{missingfigure}" + "\n"
            )
        self.article.image_count += 1
        caption = ""
        if alt and alt.lower() not in {"image", "图片", "figure"}:
            caption = "\n" + r"\caption{" + text_to_tex(alt) + "}"
        return (
            "\n" + r"\par\begin{figure}[H]\centering" + "\n"
            + r"\includegraphics[max width=0.88\linewidth,max height=0.62\textheight,keepaspectratio]{"
            + staged + "}" + caption + "\n" + r"\end{figure}\par" + "\n"
        )

    def render_table(self, node: etree._Element) -> str:
        rows = node.xpath(".//tr")
        parsed, columns = [], 1
        for row in rows:
            values = [text_to_tex(safe_text_content(cell)) for cell in row.xpath("./th|./td")]
            if values:
                parsed.append(values)
                columns = max(columns, len(values))
        if not parsed:
            return ""
        width = max(0.08, 0.80 / columns)
        spec = " ".join([f"p{{{width:.3f}\\textwidth}}"] * columns)
        lines = [r"\begin{center}\small", rf"\begin{{longtable}}{{@{{}}{spec}@{{}}}}", r"\toprule"]
        for index, row in enumerate(parsed):
            row += [""] * (columns - len(row))
            lines.append(" & ".join(row) + r" \\")
            if index == 0:
                lines.append(r"\midrule")
        lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{center}"])
        return "\n".join(lines)

    def render_list(self, node: etree._Element, ordered: bool) -> str:
        environment = "enumerate" if ordered else "itemize"
        items = [
            r"\item " + self.render_children(child).strip()
            for child in node
            if isinstance(child.tag, str) and child.tag.lower() == "li"
        ]
        if not items:
            return ""
        return f"\n\\begin{{{environment}}}\n" + "\n".join(items) + f"\n\\end{{{environment}}}\n"

    def render_node(self, node: etree._Element) -> str:
        if not isinstance(node.tag, str):
            return ""
        tag = node.tag.lower()
        if tag == "script" and "math/tex" in node.get("type", ""):
            math = (node.text or "").strip()
            return ("\n" + r"\[" + math + r"\]" + "\n") if "mode=display" in node.get("type", "") else r"\(" + math + r"\)"
        if tag in {"style", "script", "noscript"}:
            return ""
        if tag == "pre":
            return self.code_block(node)
        if tag in {"div", "figure"} and node.xpath(".//pre"):
            return self.code_block(node.xpath(".//pre")[0])
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            command = {1: "subsection", 2: "subsubsection", 3: "subsubsection", 4: "paragraph"}.get(int(tag[1]), "paragraph")
            title = text_to_tex(safe_text_content(node))
            return f"\n\\{command}{{{title}}}\n" if title else ""
        if tag == "p":
            body = self.render_children(node).strip()
            return f"\n{body}\\par\n" if body else ""
        if tag == "img":
            return self.render_image(node)
        if tag == "a":
            body, href = self.render_children(node).strip(), node.get("href", "").strip()
            if node.xpath(".//img"):
                return body
            if href.startswith(("http://", "https://")) and body:
                return rf"\href{{{url_for_tex(href)}}}{{{body}}}"
            return body or text_to_tex(href)
        if tag in {"strong", "b"}:
            return r"\textbf{" + self.render_children(node) + "}"
        if tag in {"em", "i"}:
            return r"\emph{" + self.render_children(node) + "}"
        if tag == "code":
            return r"\texttt{" + tex_escape_plain(node.text_content()) + "}"
        if tag == "kbd":
            return r"\tcbox[kbdstyle]{" + tex_escape_plain(node.text_content()) + "}"
        if tag == "blockquote":
            return "\n" + r"\begin{quotebox}" + "\n" + self.render_children(node).strip() + "\n" + r"\end{quotebox}" + "\n"
        if tag == "ul":
            return self.render_list(node, False)
        if tag == "ol":
            return self.render_list(node, True)
        if tag == "table":
            return "\n" + self.render_table(node) + "\n"
        if tag == "br":
            return r"\\"
        if tag == "hr":
            return "\n" + r"\medskip\hrule\medskip" + "\n"
        if tag == "sup":
            return r"\textsuperscript{" + self.render_children(node) + "}"
        if tag == "sub":
            return r"\textsubscript{" + self.render_children(node) + "}"
        if tag == "center":
            return "\n" + r"\begin{center}" + self.render_children(node) + r"\end{center}" + "\n"
        if tag == "details":
            return "\n" + r"\begin{notebox}{补充说明}" + "\n" + self.render_children(node) + "\n" + r"\end{notebox}" + "\n"
        if tag == "summary":
            return r"\textbf{" + self.render_children(node) + "}"
        return self.render_children(node)


def preamble() -> str:
    return r"""\documentclass[lang=cn,scheme=chinese,color=blue,device=normal,toc=onecol,math=cm]{elegantbook}

\usepackage{fvextra}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{float}
\usepackage{caption}
\usepackage[export]{adjustbox}
\usepackage{ragged2e}
\tcbuselibrary{most}
\usepackage{bookmark}

\IfFontExistsTF{DejaVu Sans Mono}{\setmonofont{DejaVu Sans Mono}}{}
\IfFontExistsTF{Microsoft YaHei}{\setCJKmonofont{Microsoft YaHei}}{}
\definecolor{ResearchNavy}{HTML}{17365D}
\definecolor{ResearchBlue}{HTML}{2F75B5}
\definecolor{ResearchCyan}{HTML}{DDEBF7}
\definecolor{ResearchGold}{HTML}{C9952E}
\definecolor{ResearchGray}{HTML}{F3F6F9}
\hypersetup{
  colorlinks=true, linkcolor=ResearchNavy, urlcolor=ResearchBlue, citecolor=ResearchNavy,
  pdftitle={计算材料与拓扑磁性研究工作手册}, pdfauthor={Mrx},
  pdfsubject={博士研究生申请科研手册}
}
\setcounter{tocdepth}{1}
\setcounter{secnumdepth}{1}
\setlength{\emergencystretch}{8em}
\setlength{\parskip}{0.35em}
\renewcommand{\arraystretch}{1.25}
\captionsetup{font=small,labelfont=bf}
\raggedbottom
\sloppy
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000
\makeatletter
\let\cleardoublepage\clearpage
\makeatother

% 五级阅读层次：部分（模块）> 章（专题）> 节（文章）> 小节（文内标题）。
\titleformat{\chapter}[display]
  {\normalfont\bfseries\color{ResearchNavy}}
  {\Large\color{ResearchGold}\IfAppendix{\appendixname\ \thechapter}{专题 \thechapter}}
  {0.45em}{\titlerule[1.1pt]\vspace{0.75ex}\Huge\filright}
\titlespacing*{\chapter}{0pt}{-12pt}{1.5\baselineskip}
\titleformat{\section}[hang]
  {\Large\bfseries\color{ResearchBlue}}
  {\color{ResearchGold}\thesection}{0.75em}{}
\titlespacing*{\section}{0pt}{2.2\baselineskip}{0.9\baselineskip}
\titleformat{\subsection}[hang]
  {\Large\bfseries\color{ResearchNavy}}
  {}{0pt}{\textcolor{ResearchGold}{\ensuremath{\blacktriangleright}}\hspace{0.55em}\raggedright}
\titlespacing*{\subsection}{0pt}{1.7\baselineskip}{0.65\baselineskip}
\titleformat{\subsubsection}[hang]
  {\large\bfseries\color{ResearchNavy}}
  {}{0pt}{\textcolor{ResearchGold}{\ensuremath{\blacktriangleright}}\hspace{0.5em}\raggedright}
\renewcommand{\cftpartfont}{\Large\bfseries\color{ResearchNavy}}
\renewcommand{\cftpartpagefont}{\Large\bfseries\color{ResearchNavy}}
\renewcommand{\cftchapfont}{\large\bfseries\color{ResearchBlue}}
\renewcommand{\cftchappagefont}{\large\bfseries\color{ResearchBlue}}
\renewcommand{\cftsecfont}{\normalsize\color{black!82}}
\renewcommand{\cftsecpagefont}{\normalsize\color{ResearchNavy}}

\newtcolorbox{profilebox}[1]{enhanced,breakable,colback=ResearchGray,colframe=ResearchBlue,boxrule=0.7pt,arc=2mm,left=3mm,right=3mm,top=2mm,bottom=2mm,title=\textbf{#1},fonttitle=\bfseries}
\newtcolorbox{partbox}{enhanced,breakable,colback=ResearchCyan,colframe=ResearchNavy,boxrule=0.8pt,arc=2mm,left=4mm,right=4mm,top=3mm,bottom=3mm}
\newtcolorbox{articlemeta}{enhanced,breakable,colback=white,colframe=ResearchGold,boxrule=0.7pt,arc=1.5mm,left=3mm,right=3mm,top=2mm,bottom=2mm}
\newtcolorbox{codeindexbox}[1]{enhanced,breakable,colback=ResearchGray,colframe=ResearchBlue,boxrule=0.55pt,arc=1.5mm,left=3mm,right=3mm,top=1.5mm,bottom=1.5mm,title=\textbf{#1},fonttitle=\small}
\newtcolorbox{quotebox}{enhanced,breakable,colback=ResearchGray,colframe=ResearchBlue,leftrule=2.5pt,rightrule=0pt,toprule=0pt,bottomrule=0pt,sharp corners,left=4mm,right=2mm,top=1.5mm,bottom=1.5mm}
\newtcolorbox{missingfigure}{enhanced,breakable,colback=red!2,colframe=red!45!black,boxrule=0.5pt,arc=1mm,left=3mm,right=3mm,top=2mm,bottom=2mm}
\newtcolorbox{notebox}[1]{enhanced,breakable,colback=ResearchGray,colframe=ResearchBlue,title=\textbf{#1},boxrule=0.5pt}
\tcbset{kbdstyle/.style={on line,colback=ResearchGray,colframe=ResearchBlue,boxrule=0.4pt,arc=1mm}}

\title{计算材料与拓扑磁性\\研究工作手册}
\subtitle{博士研究生申请科研手册 · 方法、软件、代码、绘图与心得}
\author{Mrx}
\institute{个人科研成果与方法体系整理}
\date{2026 年 9 月}
\cover{assets/cover.jpg}

\begin{document}
\maketitle
"""


def handbook_intro(_articles: list[Article]) -> str:
    return r"""
\frontmatter
\tableofcontents
\mainmatter
"""


def article_tex(article: Article) -> str:
    online = "https://marx2001.github.io" + article.url
    return (
        "\n" + r"\section{" + text_to_tex(article.title) + "}\n"
        + r"\begin{articlemeta}" + "\n"
        + r"\textbf{日期：}" + tex_escape_plain(article.date)
        + r"\hfill\textbf{在线原文：}\href{" + url_for_tex(online) + r"}{marx2001.github.io}" + "\n"
        + r"\end{articlemeta}" + "\n" + article.tex_body + "\n"
    )


def code_appendix(articles: list[Article]) -> str:
    chunks = [
        r"\appendix", r"\chapter{完整代码清单}",
        "本附录按正文文章号排序，完整保留原始代码。"
        r"每个代码块同时保存为工程 \texttt{code/} 目录中的独立 UTF-8 文件，"
        "便于复制、检索与复现。",
    ]
    for article in (item for item in articles if item.code_blocks):
        chunks.append(r"\section*{" + text_to_tex(article.title) + "}")
        for block in article.code_blocks:
            chunks.extend([
                r"\phantomsection\label{code:" + f"{block.article_no}:{block.sequence}" + "}",
                r"\subsection*{代码清单 A." + f"{block.article_no}.{block.sequence}" + "（"
                + tex_escape_plain(block.language) + f"，{block.line_count} 行）" + "}",
                r"\VerbatimInput[fontsize=\scriptsize,breaklines=true,breakanywhere=true,tabsize=2]{"
                + block.relative_path + "}",
                r"\medskip",
            ])
    return "\n".join(chunks)


def colophon(articles: list[Article], images: ImageManager) -> str:
    return rf"""
\backmatter
\chapter*{{编制与版本说明}}
\addcontentsline{{toc}}{{chapter}}{{编制与版本说明}}
\begin{{profilebox}}{{可复现性}}
本手册由 \texttt{{collect\_sci\_note.rb}} 与 \texttt{{build\_handbook.py}}
自动生成。文章范围由 Jekyll front matter 决定；正文分类清单、代码文件、
图片副本与完整 \LaTeX{{}} 源码均保存在独立工程目录中。
本次构建收录 {len(articles)} 篇文章、{sum(len(a.code_blocks) for a in articles)} 个代码块，
成功归档 {len(images.cache)} 个唯一图片文件。
\end{{profilebox}}

\begin{{profilebox}}{{模板声明}}
版式基于 ElegantLaTeX 团队的 ElegantBook（LPPL 1.3c 或更高版本）；
项目地址：\url{{https://github.com/ElegantLaTeX/ElegantBook}}。
本工程保留模板类中的原版权与许可声明。
\end{{profilebox}}

\vfill
\begin{{center}}
{{\color{{ResearchNavy}}\Large\bfseries 以可复现的方法组织研究，以清晰的证据呈现能力。}}\\[1em]
{{\small 生成日期：2026 年 9 月 10 日}}
\end{{center}}
\end{{document}}
"""


def main() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    all_posts = run_collector()
    raw_posts = [
        post for post in all_posts
        if not any(normalized_key(keyword) in normalized_key(post["title"] + " " + post["source_path"])
                   for keyword in EXCLUDED_ARTICLE_KEYWORDS)
    ]
    articles = []
    for post in raw_posts:
        part, subgroup, reason = classify(post)
        articles.append(Article(
            source_path=post["source_path"], title=post["title"], subtitle=post["subtitle"],
            date=post["date"], url=post["url"], html=post["html"],
            part=part, subgroup=subgroup, reason=reason,
        ))

    subgroup_rank = {
        part: {name: index for index, name in enumerate(meta["subgroups"])}
        for part, meta in PART_META.items()
    }
    articles.sort(key=lambda article: (
        PART_ORDER.index(article.part), subgroup_rank[article.part][article.subgroup],
        article.date, article.title,
    ))
    for index, article in enumerate(articles, start=1):
        article.article_no = index

    images = ImageManager()
    for article in articles:
        article.tex_body = HtmlToLatex(article, images).convert()

    manifest = {
        "generated_on": "2026-09-11", "source_category": "sci-note",
        "article_count": len(articles),
        "part_counts": dict(Counter(article.part for article in articles)),
        "articles": [{
            "article_no": article.article_no, "source_path": article.source_path,
            "title": article.title, "date": article.date, "part": article.part,
            "part_title": PART_META[article.part]["title"], "subgroup": article.subgroup,
            "classification_reason": article.reason, "code_blocks": len(article.code_blocks),
            "images": article.image_count, "missing_images": article.missing_images, "url": article.url,
        } for article in articles],
    }
    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    chunks = [preamble(), handbook_intro(articles)]
    for part in PART_ORDER:
        part_articles, meta = [a for a in articles if a.part == part], PART_META[part]
        chunks.append(r"\part{" + text_to_tex(meta["title"]) + "}")
        first_subgroup = True
        for subgroup in meta["subgroups"]:
            subgroup_articles = [a for a in part_articles if a.subgroup == subgroup]
            if not subgroup_articles:
                continue
            topic_intro = f"本专题收录 {len(subgroup_articles)} 篇文章。"
            if first_subgroup:
                topic_intro = text_to_tex(meta["intro"]) + r"\par\medskip " + topic_intro
                first_subgroup = False
            chunks.extend([
                r"\chapter{" + text_to_tex(subgroup) + "}",
                r"\begin{partbox}" + topic_intro + r"\end{partbox}",
            ])
            chunks.extend(article_tex(article) for article in subgroup_articles)

    chunks.extend([code_appendix(articles), r"\end{document}"])
    TEX_FILE.write_text("\n".join(chunks), encoding="utf-8", newline="\n")

    part_counts = Counter(article.part for article in articles)
    print(f"文章：{len(articles)}")
    for part in PART_ORDER:
        print(f"  {PART_META[part]['title']}：{part_counts[part]}")
    print(f"代码块：{sum(len(article.code_blocks) for article in articles)}")
    print(f"图片：{sum(article.image_count for article in articles)}")
    print(f"缺失图片引用：{sum(len(article.missing_images) for article in articles)}")
    print(TEX_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
