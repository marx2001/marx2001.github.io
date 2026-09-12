#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_level_diagram.py

通用能级排布图绘制脚本：
- 从 orb_centers.csv 读每个轨道能量中心 E_center
- 从 auto_groups.csv 读简并分组（哪些轨道一组）

两种使用方式：
A) 显式指定文件（推荐、最稳）
   python plot_level_diagram.py --orb XXX_orb_centers.csv --groups XXX_auto_groups.csv --out level.png

B) 只给 prefix 自动匹配（省事）
   python plot_level_diagram.py --prefix TcAuto --dir . --out Tc_level.png
   会自动找：
     <dir>/<prefix>_orb_centers.csv
     <dir>/<prefix>_auto_groups.csv

可选：
  --eref -1.46     # 平移能量参考（如把费米能级平移到0）
  --title "..."    # 标题
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def guess_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_orb_list(x):
    if pd.isna(x):
        return []
    s = str(x).strip()

    # 提取常见 d 轨道 token（可按需扩展，比如 p/s/f）
    tokens = re.findall(r"(dxy|dxz|dyz|dz2|dx2y2)", s, flags=re.IGNORECASE)
    if tokens:
        return [t.lower() for t in tokens]

    # fallback: 逗号/空格分割
    s = s.strip("[](){}")
    parts = re.split(r"[,\s]+", s)
    parts = [p.strip().strip("'\"").lower() for p in parts if p.strip()]
    return parts


def resolve_inputs(args):
    """
    解析输入：
    - 若用户显式给 --orb 和 --groups：直接用
    - 否则要求给 --prefix，并在 --dir 中匹配 <prefix>_orb_centers.csv / <prefix>_auto_groups.csv
    """
    if args.orb and args.groups:
        orb_path = Path(args.orb)
        grp_path = Path(args.groups)
    else:
        if not args.prefix:
            raise ValueError("请使用：要么同时提供 --orb 与 --groups；要么提供 --prefix（可配合 --dir）。")
        base = Path(args.dir).expanduser().resolve()
        orb_path = base / f"{args.prefix}_orb_centers.csv"
        grp_path = base / f"{args.prefix}_auto_groups.csv"

    if not orb_path.exists():
        raise FileNotFoundError(f"找不到 orb_centers.csv: {orb_path}")
    if not grp_path.exists():
        raise FileNotFoundError(f"找不到 auto_groups.csv: {grp_path}")

    return orb_path, grp_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orb", default=None, help="显式指定 orb_centers.csv 路径")
    ap.add_argument("--groups", default=None, help="显式指定 auto_groups.csv 路径")
    ap.add_argument("--prefix", default=None, help="自动匹配前缀：<prefix>_orb_centers.csv / <prefix>_auto_groups.csv")
    ap.add_argument("--dir", default=".", help="自动匹配时搜索目录（默认当前目录）")

    ap.add_argument("--out", default="level_diagram.png", help="输出图片名")
    ap.add_argument("--eref", type=float, default=0.0, help="能量参考平移：E_plot = E_center - eref")
    ap.add_argument("--title", default=None, help="图标题")
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--fontsize", type=int, default=12)
    ap.add_argument("--linewidth", type=float, default=3.0)
    ap.add_argument("--xspan", type=float, default=0.55, help="每条能级线横向半宽度")
    args = ap.parse_args()

    orb_path, grp_path = resolve_inputs(args)

    df_orb = pd.read_csv(orb_path)
    df_grp = pd.read_csv(grp_path)

    # ---- 1) orb_centers: 识别轨道列 + 能量列
    col_orb = guess_col(df_orb, ["orb", "orbital", "orb_name", "name", "orbital_name"])
    col_E = guess_col(df_orb, ["E_center", "Ecentre", "Ecent", "E", "center_eV", "energy_center"])
    if col_orb is None or col_E is None:
        raise ValueError(
            f"[orb_centers] 列名识别失败。现有列：{list(df_orb.columns)}\n"
            f"需要类似：orb/orbital + E_center"
        )

    orb2E = {}
    for _, r in df_orb.iterrows():
        o = str(r[col_orb]).strip().lower()
        orb2E[o] = float(r[col_E]) - args.eref

    # ---- 2) auto_groups: 识别轨道列表列 + 组能量列（可选）+ 简并度列（可选）
    col_list = guess_col(df_grp, ["orbs", "orbitals", "orb_list", "members", "group_orbs"])
    col_grpE = guess_col(df_grp, ["E_center", "E_group", "group_E", "E", "center_eV"])
    col_size = guess_col(df_grp, ["size", "deg", "degeneracy", "n_orb", "N"])

    groups = []
    for _, r in df_grp.iterrows():
        orbs = parse_orb_list(r[col_list]) if col_list else []
        orbs = [o.lower() for o in orbs if o]
        if not orbs:
            # 兜底：从整行提取
            orbs = parse_orb_list(" ".join(map(str, r.values)))

        # 组能量：优先 groups 表里的 E_center，否则用成员平均
        if col_grpE and (not pd.isna(r[col_grpE])):
            Eg = float(r[col_grpE]) - args.eref
        else:
            Es = [orb2E[o] for o in orbs if o in orb2E]
            Eg = float(np.mean(Es)) if Es else np.nan

        # 简并度：优先 size 列，否则用 orbs 数量
        if col_size and (not pd.isna(r[col_size])):
            gsize = int(r[col_size])
        else:
            gsize = len(orbs)

        # label 排序显示更直观（可按需扩展）
        pretty_order = ["dxy", "dx2y2", "dz2", "dxz", "dyz"]
        orbs_sorted = sorted(orbs, key=lambda x: pretty_order.index(x) if x in pretty_order else 999)
        label = " + ".join(orbs_sorted) if orbs_sorted else "Group"

        groups.append({"Eg": Eg, "orbs": orbs_sorted, "size": gsize, "label": label})

    groups = [g for g in groups if np.isfinite(g["Eg"])]
    if not groups:
        raise RuntimeError("没有解析到任何有效的 group 能量 Eg，请检查 groups.csv 格式。")

    # 按能量排序（低->高）
    groups.sort(key=lambda g: g["Eg"])

    # ---- 3) 绘图
    n = len(groups)
    fig_h = max(4.0, 1.2 + 0.8 * n)
    fig, ax = plt.subplots(figsize=(6.4, fig_h))

    x0 = 0.0
    xspan = args.xspan
    x_offsets = np.linspace(-0.15, 0.15, n) if n > 1 else np.array([0.0])

    for i, g in enumerate(groups):
        y = g["Eg"]
        xo = x_offsets[i]

        ax.hlines(y, x0 - xspan + xo, x0 + xspan + xo, linewidth=args.linewidth)

        deg_txt = f"({g['size']})" if g["size"] else ""
        txt = f"{g['label']} {deg_txt}".strip()

        ax.text(x0 + xspan + 0.10, y, txt, va="center", fontsize=args.fontsize)
        ax.text(x0 - xspan - 0.10, y, f"{y:.3f} eV", va="center", ha="right", fontsize=args.fontsize - 1)

    ax.set_xlim(-1.2, 1.9)
    ax.set_xticks([])
    ax.set_ylabel(f"Energy (eV)  [E_center - {args.eref:g}]", fontsize=args.fontsize)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    if args.title:
        ax.set_title(args.title, fontsize=args.fontsize + 2)
    else:
        # 默认标题：用文件名提示来源
        ax.set_title(f"Level diagram from {orb_path.name}", fontsize=args.fontsize + 1)

    ys = [g["Eg"] for g in groups]
    ypad = 0.15 * (max(ys) - min(ys) + 1e-9)
    ymin = (min(ys) - ypad) if args.ymin is None else args.ymin
    ymax = (max(ys) + ypad) if args.ymax is None else args.ymax
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)

    print(f"[OK] orb_centers: {orb_path}")
    print(f"[OK] auto_groups: {grp_path}")
    print(f"[OK] Saved: {args.out}")
    print("[INFO] Levels (low -> high):")
    for g in groups:
        print(f"  Eg={g['Eg']:.6f} eV   size={g['size']}   {g['label']}")


if __name__ == "__main__":
    main()
