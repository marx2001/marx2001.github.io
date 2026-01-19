from pathlib import Path
import re

# ============================================================
# 0) 当前脚本所在目录
# ============================================================
workdir = Path(__file__).resolve().parent

# ============================================================
# 1) 查找所有 wannier90*.dat 文件
# ============================================================
dat_files = sorted(workdir.glob("wannier90*.dat"))

if not txt_files:
    print("No wannier90*.dat files found.")
    raise SystemExit

print(f"Found {len(dat_files)} file(s):")
for f in dat_files:
    print("  -", f.name)

# ============================================================
# 2) 逐个文件清洗并另存为 .dat
# ============================================================
for infile in txt_files:
    outfile = infile.with_suffix(".dat")
    print(f"\nProcessing: {infile.name} -> {outfile.name}")

    text = infile.read_text(encoding="utf-8")

    # 1) 去掉 Mathematica 的“续行符”：反斜杠 + 换行
    text = re.sub(r"\\\s*\n", " ", text)

    # 2) 把字符串里的 \n 还原成真正换行
    text = text.replace("\\n", "\n")
    text = text.replace('"', '')
    text = text.replace('"', '')

    # 3) 压缩多余空格（>=2 个压成 1 个）
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 4) 去掉空行 + 行尾空格
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]

    # 写出 .dat 文件
    outfile.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"  [OK] Written: {outfile.name}")

print("\nAll files processed.")
