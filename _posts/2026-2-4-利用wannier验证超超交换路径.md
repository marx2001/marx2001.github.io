---
layout: post
title: "(原创自研)利用wannier验证超超交换路径"
subtitle: "Experience Sharing"
background: '/img/bg-sci-note.jpg'
categories: sci-note
permalink: /sci-note_posts/20260204-super1
---

## <center>说明</center>
为验证MNX2Y6体系铁磁基态的产生机制，用wannier提取跃迁矩阵元，判断哪个元素主导超超交换路径。

1step.py 是提取在位能，2step.py是对各个原子进行超超交换路径组合，然后除以对应的在位能，形成一个值，根据这个值的大小判断超超交换路径贡献，找到路径最大的那个。

## <center>代码的基本用法</center>  

(1)第一步，把 Wannier 轨道“编号”跟“物理含义”一一对应起来，也就是建立一张 WF index →（原子、轨道类型、位置） 的映射表。没有这张表，你从 wannier90_hr.dat 抽出来的矩阵元只是数字，没法说清它对应 Tc–Se、Se–Ir 还是 Tc–Tc 的哪条路径。

准备文件

```python
wannier90_centres.xyz
wannier90_hr.dat
wannier90.wout
```

即，第一步要确认wannier90拟合的正确性，并输出上述文件。

(2) Step2 的本质是：把 wannier90_hr.dat 里的 Hmn(R)变成“带几何距离的 hopping 列表”，再按你关心的子空间对（Tc–Tc、Tc–Se、Se–Ir…）和壳层做统计。

workflow:

准备：
```python
wannier90_hr.dat
wannier90_centres.xyz
wannier90.win
```

输出：
```python
1step.csv
2step.csv
```

名字随意，可以通过代码修改

代码：
```python

import re
import csv
import math
import argparse
from collections import defaultdict, namedtuple

import numpy as np

Edge = namedtuple("Edge", "pair shell dist absH re im Rx Ry Rz m n atom_m atom_n elem_m elem_n")


# =========================
# Parse wannier90.win
# =========================
def _extract_block(txt, block_name):
    m = re.search(rf"begin\s+{block_name}(.*?)end\s+{block_name}", txt, re.S | re.I)
    return None if not m else m.group(1).strip()

def parse_win_lattice_and_atoms(win_path):
    """
    Read unit_cell_cart and atoms_cart from wannier90.win.
    Assumes Angstrom if unit not specified (your file matches this).
    Returns:
      A (3x3) with columns = a1,a2,a3 in Angstrom, so cart = A @ frac
      atoms: list of dict {id, elem, r_cart(np.array shape(3))}
    """
    with open(win_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    cell_block = _extract_block(txt, "unit_cell_cart")
    if cell_block is None:
        raise RuntimeError("Cannot find unit_cell_cart block in wannier90.win")

    lines = [ln.strip() for ln in cell_block.splitlines() if ln.strip()]
    # Some files might include "Ang" or "Bohr" as first line; yours doesn't.
    unit = None
    if re.match(r"^(ang|angstrom|bohr)\b", lines[0], re.I):
        unit = lines[0].lower()
        vec_lines = lines[1:4]
    else:
        unit = "ang"
        vec_lines = lines[0:3]

    if len(vec_lines) < 3:
        raise RuntimeError("unit_cell_cart has < 3 lattice vectors")

    a1 = np.array([float(x) for x in vec_lines[0].split()[:3]], dtype=float)
    a2 = np.array([float(x) for x in vec_lines[1].split()[:3]], dtype=float)
    a3 = np.array([float(x) for x in vec_lines[2].split()[:3]], dtype=float)

    if "bohr" in unit:
        bohr_to_ang = 0.52917721092
        a1 *= bohr_to_ang
        a2 *= bohr_to_ang
        a3 *= bohr_to_ang

    # A columns are lattice vectors
    A = np.stack([a1, a2, a3], axis=1)  # shape (3,3)

    atoms_block = _extract_block(txt, "atoms_cart")
    if atoms_block is None:
        raise RuntimeError("Cannot find atoms_cart block in wannier90.win")

    atoms = []
    for idx, ln in enumerate([x for x in atoms_block.splitlines() if x.strip()], start=1):
        parts = ln.split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        r = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        atoms.append({"id": idx, "elem": elem, "r": r})

    if not atoms:
        raise RuntimeError("atoms_cart parsed but got 0 atoms")

    return A, atoms


# =========================
# Parse wannier90_centres.xyz
# =========================
def parse_centres_xyz(xyz_path):
    """
    Parse wannier90_centres.xyz:
    line1 = N
    line2 = comment
    then N lines: <label> x y z
    Return centers list indexed 1..N: centers[wf] = np.array([x,y,z])
    """
    with open(xyz_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    n = int(lines[0])
    if len(lines) < 2 + n:
        raise RuntimeError(f"centres.xyz incomplete: need {2+n} lines, got {len(lines)}")

    centers = [None] * (n + 1)
    for i in range(n):
        parts = lines[2 + i].split()
        if len(parts) < 4:
            raise RuntimeError(f"Bad centres.xyz line: {lines[2+i]}")
        centers[i + 1] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
    return centers


# =========================
# Parse wannier90_hr.dat
# =========================
def parse_hr_dat(hr_path):
    """
    Yields (Rx, Ry, Rz, m, n, reH, imH)
    """
    with open(hr_path, "r", encoding="utf-8", errors="ignore") as f:
        _ = f.readline()  # comment
        num_wann = int(f.readline().strip())
        nrpts = int(f.readline().strip())

        # read degeneracy list
        degen = []
        while len(degen) < nrpts:
            ln = f.readline()
            if not ln:
                raise RuntimeError("Unexpected EOF while reading degeneracy list")
            ln = ln.strip()
            if not ln:
                continue
            degen += [int(x) for x in ln.split()]

        # data
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 7:
                continue
            Rx, Ry, Rz = int(parts[0]), int(parts[1]), int(parts[2])
            m, n = int(parts[3]), int(parts[4])
            reH, imH = float(parts[5]), float(parts[6])
            yield Rx, Ry, Rz, m, n, reH, imH


# =========================
# WF grouping (edit ranges here)
# =========================
def wf_group(wf):
    # Your current mapping:
    if 1 <= wf <= 10:
        return "Tc_d"
    if 11 <= wf <= 20:
        return "Ir_d"
    if 21 <= wf <= 56:
        return "Se_p"
    if 57 <= wf <= 68:
        return "Ge_p"
    return "OTHER"


def pair_name(g1, g2):
    # Use a fixed ordering to avoid the earlier "Se_p<->Tc_d" name surprise.
    order = {"Tc_d": 1, "Ir_d": 2, "Se_p": 3, "Ge_p": 4, "OTHER": 99}
    if order[g1] <= order[g2]:
        return f"{g1}<->{g2}"
    return f"{g2}<->{g1}"


# =========================
# Minimum-image mapping: WF -> nearest atom
# =========================
def cart_to_frac(A, r_cart):
    # cart = A @ frac
    return np.linalg.solve(A, r_cart)

def frac_to_cart(A, f):
    return A @ f

def wrap_delta_frac(df):
    # to [-0.5, 0.5)
    return df - np.round(df)

def map_wf_to_atoms(A, centers, atoms):
    """
    Returns:
      wf_atom_id[wf] = atom index in atoms list (0-based)
      wf_atom_elem[wf] = element string
    Uses minimum-image distance in fractional space.
    """
    inv_needed = False  # we use solve each time; stable enough for 68 WF

    atom_frac = []
    for a in atoms:
        atom_frac.append(cart_to_frac(A, a["r"]))
    atom_frac = np.array(atom_frac)  # shape (Nat,3)

    wf_atom_id = [None] * len(centers)
    wf_atom_elem = [None] * len(centers)

    for wf in range(1, len(centers)):
        r = centers[wf]
        f_w = cart_to_frac(A, r)

        # compute minimum image distance to each atom
        best_i = None
        best_d = 1e30
        for i, f_a in enumerate(atom_frac):
            df = wrap_delta_frac(f_w - f_a)
            dr = frac_to_cart(A, df)
            d = float(np.linalg.norm(dr))
            if d < best_d:
                best_d = d
                best_i = i

        wf_atom_id[wf] = best_i
        wf_atom_elem[wf] = atoms[best_i]["elem"]

    return wf_atom_id, wf_atom_elem


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Step2 (interatomic): summarize hoppings by subspace pair + distance shell")
    ap.add_argument("--win", required=True, help="wannier90.win")
    ap.add_argument("--centres", required=True, help="wannier90_centres.xyz")
    ap.add_argument("--hr", required=True, help="wannier90_hr.dat")
    ap.add_argument("--tol", type=float, default=0.10, help="distance bin size (Ang), e.g. 0.05~0.15")
    ap.add_argument("--min_absH", type=float, default=1e-4, help="min |H| kept (eV)")
    ap.add_argument("--topk", type=int, default=30, help="topK edges per (pair,shell)")
    ap.add_argument("--skip_same_atom_R0", action="store_true",
                    help="skip terms where (atom_m==atom_n) AND (R==0). Recommended to remove onsite/local terms.")
    ap.add_argument("--skip_diag_R0", action="store_true",
                    help="skip diagonal terms (m==n) at R==0 (onsite). Usually redundant if skip_same_atom_R0 is on.")
    ap.add_argument("--pairs", default="ALL", help="comma-separated allowed pairs like Tc_d<->Se_p, or ALL")
    ap.add_argument("--out_summary", default="hopping_summary_by_shell.csv")
    ap.add_argument("--out_edges", default="", help="optional: write filtered edges csv (empty=off)")
    args = ap.parse_args()

    A, atoms = parse_win_lattice_and_atoms(args.win)
    centers = parse_centres_xyz(args.centres)

    wf_atom_id, wf_atom_elem = map_wf_to_atoms(A, centers, atoms)

    # Allowed pairs
    allowed_pairs = None
    if args.pairs.strip().upper() != "ALL":
        allowed_pairs = set([p.strip() for p in args.pairs.split(",") if p.strip()])

    # stats
    cnt = defaultdict(int)
    sumsq = defaultdict(float)
    maxv = defaultdict(float)
    top_edges = defaultdict(list)
    filtered_edges = []

    # Precompute lattice vectors for translation
    a1 = A[:, 0]
    a2 = A[:, 1]
    a3 = A[:, 2]

    def R_to_T(Rx, Ry, Rz):
        return Rx * a1 + Ry * a2 + Rz * a3

    for Rx, Ry, Rz, m, n, reH, imH in parse_hr_dat(args.hr):
        absH = math.hypot(reH, imH)
        if absH < args.min_absH:
            continue

        gm = wf_group(m)
        gn = wf_group(n)
        if gm == "OTHER" or gn == "OTHER":
            continue

        pair = pair_name(gm, gn)
        if allowed_pairs is not None and pair not in allowed_pairs:
            continue

        atom_m = wf_atom_id[m]
        atom_n = wf_atom_id[n]
        elem_m = wf_atom_elem[m]
        elem_n = wf_atom_elem[n]

        # Skip onsite/local terms if requested
        if (Rx, Ry, Rz) == (0, 0, 0):
            if args.skip_diag_R0 and (m == n):
                continue
            if args.skip_same_atom_R0 and (atom_m == atom_n):
                continue

        # distance between WF centers considering translation R
        T = R_to_T(Rx, Ry, Rz)
        dr = (centers[n] + T) - centers[m]
        dist = float(np.linalg.norm(dr))

        # bin to shell
        shell = round(dist / args.tol) * args.tol
        key = (pair, shell)

        cnt[key] += 1
        sumsq[key] += absH * absH
        if absH > maxv[key]:
            maxv[key] = absH

        e = Edge(pair, shell, dist, absH, reH, imH, Rx, Ry, Rz, m, n, atom_m, atom_n, elem_m, elem_n)

        lst = top_edges[key]
        lst.append(e)
        lst.sort(key=lambda x: x.absH, reverse=True)
        if len(lst) > args.topk:
            lst[:] = lst[:args.topk]

        if args.out_edges:
            filtered_edges.append(e)

    # write summary
    with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pair", "shell_A", "count", "max_absH_eV", "rms_absH_eV",
            f"top{args.topk}_edges(m,n,atom_m,atom_n,elem_m,elem_n,Rx,Ry,Rz,absH,dist)"
        ])
        for (pair, shell) in sorted(cnt.keys(), key=lambda x: (x[0], x[1])):
            c = cnt[(pair, shell)]
            rms = math.sqrt(sumsq[(pair, shell)] / c) if c else 0.0
            tops = top_edges[(pair, shell)]
            tops_str = "; ".join([
                f"{e.m}-{e.n}|a{e.atom_m+1}-a{e.atom_n+1}|{e.elem_m}-{e.elem_n}"
                f"@({e.Rx},{e.Ry},{e.Rz})|{e.absH:.6g}|d={e.dist:.3f}"
                for e in tops
            ])
            w.writerow([pair, f"{shell:.3f}", c, f"{maxv[(pair, shell)]:.6g}", f"{rms:.6g}", tops_str])

    # optional edges
    if args.out_edges:
        with open(args.out_edges, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "pair", "shell_A", "dist_A", "absH_eV", "re", "im",
                "Rx", "Ry", "Rz", "m", "n",
                "atom_m", "atom_n", "elem_m", "elem_n"
            ])
            for e in filtered_edges:
                w.writerow([
                    e.pair, f"{e.shell:.3f}", f"{e.dist:.6f}", f"{e.absH:.8g}",
                    f"{e.re:.8g}", f"{e.im:.8g}",
                    e.Rx, e.Ry, e.Rz, e.m, e.n,
                    e.atom_m+1, e.atom_n+1, e.elem_m, e.elem_n
                ])

    print("Done.")
    print("Summary:", args.out_summary)
    if args.out_edges:
        print("Edges:", args.out_edges)


if __name__ == "__main__":
    main()


```

命令行输入：
```shell

python 1step.py --win wannier90.win --centres wannier90_centres.xyz --hr wannier90_hr.dat --tol 0.10 --min_absH 1e-4 --topk 30 --out_summary 1step.csv --pairs ALL --skip_same_atom_R0 --out_edges 2step.csv

```
可以调整命令来规定输出文件名

(3)读取你已经生成的 2step.csv（跨原子 edges），自动筛选 Tc–Se 与 Se–Ir 的近邻 hopping，构
造 Tc → Se → Ir → Se → Tc 的候选“超超交换路径”，并按路径权重排序输出一个csv文件。

对于三跳跃体系，类似超交换，可用下方脚本：
```python

import csv
import math
import argparse
from dataclasses import dataclass
from collections import defaultdict
from typing import Tuple, List, Dict, Iterable


@dataclass(frozen=True)
class DirEdge:
    # directed edge: u(cell_shift) -> v(cell_shift + R)
    u_atom: int          # 1-based
    v_atom: int          # 1-based
    u_elem: str
    v_elem: str
    u_wf: int
    v_wf: int
    Rx: int
    Ry: int
    Rz: int
    absH: float
    dist: float


def parse_edges_csv(path: str) -> List[DirEdge]:
    """
    Read out_edges csv produced by step2_interatomic.py (your 2step.csv).
    Expected columns include:
      pair,shell_A,dist_A,absH_eV,re,im,Rx,Ry,Rz,m,n,atom_m,atom_n,elem_m,elem_n
    atom_* are 1-based in the file.
    """
    edges = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        required = ["dist_A", "absH_eV", "Rx", "Ry", "Rz", "m", "n", "atom_m", "atom_n", "elem_m", "elem_n"]
        for k in required:
            if k not in r.fieldnames:
                raise RuntimeError(f"Missing column '{k}' in {path}. Found: {r.fieldnames}")

        for row in r:
            try:
                dist = float(row["dist_A"])
                absH = float(row["absH_eV"])
                Rx, Ry, Rz = int(row["Rx"]), int(row["Ry"]), int(row["Rz"])
                m, n = int(row["m"]), int(row["n"])
                atom_m, atom_n = int(row["atom_m"]), int(row["atom_n"])
                elem_m, elem_n = row["elem_m"].strip(), row["elem_n"].strip()
            except Exception:
                continue

            # Create BOTH directions:
            # m(home) -> n(shift R)
            edges.append(DirEdge(atom_m, atom_n, elem_m, elem_n, m, n, Rx, Ry, Rz, absH, dist))
            # n(home) -> m(shift -R)
            edges.append(DirEdge(atom_n, atom_m, elem_n, elem_m, n, m, -Rx, -Ry, -Rz, absH, dist))
    return edges


def is_pair(e: DirEdge, a: str, b: str) -> bool:
    return (e.u_elem == a and e.v_elem == b)


def add_shift(s: Tuple[int, int, int], R: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (s[0] + R[0], s[1] + R[1], s[2] + R[2])


def abs_R(R: Tuple[int, int, int]) -> int:
    return abs(R[0]) + abs(R[1]) + abs(R[2])


def main():
    ap = argparse.ArgumentParser(description="Step4: build Tc–Se–Ir–Se–Tc paths from out_edges csv")
    ap.add_argument("--edges", required=True, help="Input edges csv, e.g. 2step.csv")
    ap.add_argument("--out", default="top_paths.csv", help="Output paths csv")

    ap.add_argument("--d_tcse", type=float, default=3.0, help="Max distance for Tc–Se edges (Ang)")
    ap.add_argument("--d_seir", type=float, default=3.0, help="Max distance for Se–Ir edges (Ang)")

    ap.add_argument("--min_absH", type=float, default=1e-3,
                    help="Min |t| for edges used in path building (eV). "
                         "Use 1e-4 if you want everything, 1e-3 speeds up a lot.")
    ap.add_argument("--top_per_se", type=int, default=60,
                    help="Keep top N Tc–Se edges per Se atom (after filtering)")
    ap.add_argument("--top_per_ir_se", type=int, default=80,
                    help="Keep top N Se–Ir edges per Se atom for each direction (Se->Ir and Ir->Se)")
    ap.add_argument("--top_paths", type=int, default=500,
                    help="Number of top paths to write")

    ap.add_argument("--require_same_tc_atom", action="store_true",
                    help="Require end Tc atom id == start Tc atom id (useful when there is only one Tc in primitive cell)")
    ap.add_argument("--max_netR_L1", type=int, default=6,
                    help="Filter paths by L1 norm of net cell shift |dRx|+|dRy|+|dRz| <= this. "
                         "Set large (e.g. 999) to disable.")
    args = ap.parse_args()

    # ---------- load directed edges ----------
    all_dir = parse_edges_csv(args.edges)

    # Identify Tc/Ir/Se atom ids from elements (in directed edges)
    tc_atoms = sorted({e.u_atom for e in all_dir if e.u_elem == "Tc"} | {e.v_atom for e in all_dir if e.v_elem == "Tc"})
    ir_atoms = sorted({e.u_atom for e in all_dir if e.u_elem == "Ir"} | {e.v_atom for e in all_dir if e.v_elem == "Ir"})
    se_atoms = sorted({e.u_atom for e in all_dir if e.u_elem == "Se"} | {e.v_atom for e in all_dir if e.v_elem == "Se"})

    if not tc_atoms:
        raise RuntimeError("No Tc atoms found in edges file (elem == 'Tc').")
    if not ir_atoms:
        raise RuntimeError("No Ir atoms found in edges file (elem == 'Ir').")
    if not se_atoms:
        raise RuntimeError("No Se atoms found in edges file (elem == 'Se').")

    # For this material there is typically 1 Ir in cell; if multiple, we allow any.
    # We'll build paths via any Ir atom (still correct).
    print(f"[INFO] Tc atoms: {tc_atoms}")
    print(f"[INFO] Ir atoms: {ir_atoms}")
    print(f"[INFO] Se atoms: {se_atoms}")

    # ---------- filter edges for path building ----------
    # We need:
    # Tc -> Se (directed)   AND Se -> Tc (directed) for last hop
    # Se -> Ir (directed)   AND Ir -> Se (directed) for middle hops
    tc_to_se = defaultdict(list)   # key: (Tc_atom) -> list[DirEdge] where Tc->Se
    se_to_tc = defaultdict(list)   # key: (Se_atom) -> list[DirEdge] where Se->Tc
    se_to_ir = defaultdict(list)   # key: (Se_atom) -> list[DirEdge] where Se->Ir
    ir_to_se = defaultdict(list)   # key: (Ir_atom) -> list[DirEdge] where Ir->Se

    for e in all_dir:
        if e.absH < args.min_absH:
            continue

        if is_pair(e, "Tc", "Se") and e.dist <= args.d_tcse:
            tc_to_se[e.u_atom].append(e)
        elif is_pair(e, "Se", "Tc") and e.dist <= args.d_tcse:
            se_to_tc[e.u_atom].append(e)
        elif is_pair(e, "Se", "Ir") and e.dist <= args.d_seir:
            se_to_ir[e.u_atom].append(e)
        elif is_pair(e, "Ir", "Se") and e.dist <= args.d_seir:
            ir_to_se[e.u_atom].append(e)

    # ---------- per-atom pruning (keep strongest edges) ----------
    def keep_top(d: Dict[int, List[DirEdge]], topn: int):
        for k in list(d.keys()):
            lst = d[k]
            lst.sort(key=lambda x: x.absH, reverse=True)
            if len(lst) > topn:
                d[k] = lst[:topn]

    # for Tc->Se: prune per Tc atom? We want per Se too, but easiest is per Tc then per Se later in path
    keep_top(tc_to_se, topn=max(args.top_per_se, 50))
    keep_top(se_to_tc, topn=max(args.top_per_se, 50))
    keep_top(se_to_ir, topn=max(args.top_per_ir_se, 80))
    keep_top(ir_to_se, topn=max(args.top_per_ir_se, 80))

    # Build helper indices by Se atom for faster chaining
    # Tc->Se edges grouped by Se target
    tc_to_se_by_se = defaultdict(list)  # Se_atom -> edges
    for tc, lst in tc_to_se.items():
        for e in lst:
            tc_to_se_by_se[e.v_atom].append(e)
    for se, lst in tc_to_se_by_se.items():
        lst.sort(key=lambda x: x.absH, reverse=True)
        if len(lst) > args.top_per_se:
            tc_to_se_by_se[se] = lst[:args.top_per_se]

    # Se->Tc edges grouped by Se source already: se_to_tc[Se]
    for se, lst in se_to_tc.items():
        lst.sort(key=lambda x: x.absH, reverse=True)
        if len(lst) > args.top_per_se:
            se_to_tc[se] = lst[:args.top_per_se]

    # Se->Ir per Se
    for se, lst in se_to_ir.items():
        lst.sort(key=lambda x: x.absH, reverse=True)
        if len(lst) > args.top_per_ir_se:
            se_to_ir[se] = lst[:args.top_per_ir_se]

    # Ir->Se per Ir
    for ir, lst in ir_to_se.items():
        lst.sort(key=lambda x: x.absH, reverse=True)
        if len(lst) > args.top_per_ir_se:
            ir_to_se[ir] = lst[:args.top_per_ir_se]

    # ---------- enumerate paths ----------
    # State uses (atom_id, cell_shift). Start Tc in home cell (0,0,0).
    # Directed edge always moves to (v_atom, shift + (Rx,Ry,Rz)).
    paths = []

    def push_path(W, e1, e2, e3, e4, netR, tc0, tc1, se1, ir, se2):
        paths.append((
            W,
            tc0, se1, ir, se2, tc1,
            netR,
            e1, e2, e3, e4
        ))

    # For each start Tc atom
    for tc0 in tc_atoms:
        shift0 = (0, 0, 0)

        # Step 1: Tc -> Se1
        for e1 in tc_to_se.get(tc0, []):
            shift_se1 = add_shift(shift0, (e1.Rx, e1.Ry, e1.Rz))
            se1 = e1.v_atom

            # Step 2: Se1 -> Ir
            for e2 in se_to_ir.get(se1, []):
                shift_ir = add_shift(shift_se1, (e2.Rx, e2.Ry, e2.Rz))
                ir = e2.v_atom

                # Step 3: Ir -> Se2
                for e3 in ir_to_se.get(ir, []):
                    shift_se2 = add_shift(shift_ir, (e3.Rx, e3.Ry, e3.Rz))
                    se2 = e3.v_atom

                    # Step 4: Se2 -> Tc1
                    for e4 in se_to_tc.get(se2, []):
                        shift_tc1 = add_shift(shift_se2, (e4.Rx, e4.Ry, e4.Rz))
                        tc1 = e4.v_atom

                        if args.require_same_tc_atom and (tc1 != tc0):
                            continue

                        netR = shift_tc1  # Tc1 cell relative to Tc0(home)
                        if args.max_netR_L1 < 999 and abs_R(netR) > args.max_netR_L1:
                            continue

                        # Path weight (simple, reproducible)
                        W = e1.absH * e2.absH * e3.absH * e4.absH
                        push_path(W, e1, e2, e3, e4, netR, tc0, tc1, se1, ir, se2)

    if not paths:
        raise RuntimeError("No paths found. Try lowering --min_absH, increasing distance windows, or relaxing --max_netR_L1.")

    # Sort and keep top
    paths.sort(key=lambda x: x[0], reverse=True)
    paths = paths[:args.top_paths]

    # ---------- write output ----------
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "W_path",
            "Tc0_atom", "Se1_atom", "Ir_atom", "Se2_atom", "Tc1_atom",
            "net_dRx", "net_dRy", "net_dRz",
            # step1
            "t1_absH", "d1_A", "wf1_u", "wf1_v", "R1x", "R1y", "R1z",
            # step2
            "t2_absH", "d2_A", "wf2_u", "wf2_v", "R2x", "R2y", "R2z",
            # step3
            "t3_absH", "d3_A", "wf3_u", "wf3_v", "R3x", "R3y", "R3z",
            # step4
            "t4_absH", "d4_A", "wf4_u", "wf4_v", "R4x", "R4y", "R4z",
        ])

        for W, tc0, se1, ir, se2, tc1, netR, e1, e2, e3, e4 in paths:
            w.writerow([
                f"{W:.8g}",
                tc0, se1, ir, se2, tc1,
                netR[0], netR[1], netR[2],

                f"{e1.absH:.8g}", f"{e1.dist:.6f}", e1.u_wf, e1.v_wf, e1.Rx, e1.Ry, e1.Rz,
                f"{e2.absH:.8g}", f"{e2.dist:.6f}", e2.u_wf, e2.v_wf, e2.Rx, e2.Ry, e2.Rz,
                f"{e3.absH:.8g}", f"{e3.dist:.6f}", e3.u_wf, e3.v_wf, e3.Rx, e3.Ry, e3.Rz,
                f"{e4.absH:.8g}", f"{e4.dist:.6f}", e4.u_wf, e4.v_wf, e4.Rx, e4.Ry, e4.Rz,
            ])

    print(f"[DONE] wrote {len(paths)} paths to: {args.out}")
    print("[TIP] If you get too many paths or it runs slow: increase --min_absH (e.g. 2e-3 or 5e-3), or lower --top_per_se/--top_per_ir_se.")


if __name__ == "__main__":
    main()


```

运行方法（针对你当前文件名：2step.csv）：

```shell
python step4_build_paths.py --edges 2step.csv --out top_paths.csv ^
  --d_tcse 3.0 --d_seir 3.0 ^
  --min_absH 1e-3 ^
  --top_per_se 60 --top_per_ir_se 80 ^
  --top_paths 500 ^
  --require_same_tc_atom ^
  --max_netR_L1 6
```

如果尽可能不漏(慢)：

```shell

python step4_build_paths.py --edges 2step.csv --out top_paths.csv ^
  --d_tcse 3.0 --d_seir 3.0 ^
  --min_absH 1e-4 ^
  --top_per_se 80 --top_per_ir_se 120 ^
  --top_paths 1000 ^
  --require_same_tc_atom ^
  --max_netR_L1 8


```
上面这个脚本没有测试过，使用需谨慎。


四跃迁桥连结构

```python

import csv
import argparse
from dataclasses import dataclass
from collections import defaultdict
from typing import Tuple, List, Dict


@dataclass(frozen=True)
class DirEdge:
    # directed edge: u(cell_shift) -> v(cell_shift + R)
    u_atom: int          # 1-based
    v_atom: int          # 1-based
    u_elem: str
    v_elem: str
    u_wf: int
    v_wf: int
    Rx: int
    Ry: int
    Rz: int
    absH: float
    dist: float


def parse_edges_csv(path: str) -> List[DirEdge]:
    """
    Read out_edges csv produced by your step2 script (2step.csv).
    Columns expected:
      dist_A, absH_eV, Rx,Ry,Rz, m,n, atom_m,atom_n, elem_m,elem_n
    """
    edges = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        required = ["dist_A", "absH_eV", "Rx", "Ry", "Rz",
                    "m", "n", "atom_m", "atom_n", "elem_m", "elem_n"]
        for k in required:
            if k not in r.fieldnames:
                raise RuntimeError(f"Missing column '{k}' in {path}. Found: {r.fieldnames}")

        for row in r:
            try:
                dist = float(row["dist_A"])
                absH = float(row["absH_eV"])
                Rx, Ry, Rz = int(row["Rx"]), int(row["Ry"]), int(row["Rz"])
                m, n = int(row["m"]), int(row["n"])
                atom_m, atom_n = int(row["atom_m"]), int(row["atom_n"])
                elem_m, elem_n = row["elem_m"].strip(), row["elem_n"].strip()
            except Exception:
                continue

            # add BOTH directions (m@0->n@R) and (n@0->m@-R)
            edges.append(DirEdge(atom_m, atom_n, elem_m, elem_n, m, n, Rx, Ry, Rz, absH, dist))
            edges.append(DirEdge(atom_n, atom_m, elem_n, elem_m, n, m, -Rx, -Ry, -Rz, absH, dist))
    return edges


def add_shift(s: Tuple[int, int, int], R: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (s[0] + R[0], s[1] + R[1], s[2] + R[2])


def abs_R_L1(R: Tuple[int, int, int]) -> int:
    return abs(R[0]) + abs(R[1]) + abs(R[2])


def keep_top_per_key(d: Dict[int, List[DirEdge]], topn: int):
    for k in list(d.keys()):
        lst = d[k]
        lst.sort(key=lambda x: x.absH, reverse=True)
        if len(lst) > topn:
            d[k] = lst[:topn]


def main():
    ap = argparse.ArgumentParser(
        description="Build Tc–Se–X–Se–Tc (X in {Ir,Ge}) paths from out_edges csv and rank by hopping-product weight."
    )
    ap.add_argument("--edges", required=True, help="Input edges csv (e.g., 2step.csv)")
    ap.add_argument("--out", default="top_paths.csv", help="Output paths csv")

    # distance windows (Ang)
    ap.add_argument("--d_tcse", type=float, default=3.0, help="Max distance for Tc–Se edges (Ang)")
    ap.add_argument("--d_sex", type=float, default=3.0, help="Max distance for Se–X edges (Ang), applies to all mediators")

    # which mediators to consider
    ap.add_argument("--mediators", default="Ir,Ge",
                    help="Comma-separated mediator elements, e.g. Ir,Ge or Ir only or Ge only")

    # pruning
    ap.add_argument("--min_absH", type=float, default=1e-3,
                    help="Min |t| for edges used in path building (eV). 1e-3 is fast; 1e-4 is exhaustive but slower.")
    ap.add_argument("--top_per_se_tc", type=int, default=60,
                    help="Keep top N Tc–Se edges per Se atom")
    ap.add_argument("--top_per_se_x", type=int, default=120,
                    help="Keep top N Se–X edges per Se atom (for each mediator)")
    ap.add_argument("--top_per_x", type=int, default=120,
                    help="Keep top N X–Se edges per X atom (for each mediator)")

    ap.add_argument("--top_paths", type=int, default=800,
                    help="Number of top paths (overall) to write")

    ap.add_argument("--require_same_tc_atom", action="store_true",
                    help="Require end Tc atom id == start Tc atom id (useful when there is only one Tc in primitive cell)")
    ap.add_argument("--max_netR_L1", type=int, default=6,
                    help="Filter by |dRx|+|dRy|+|dRz| <= this (set 999 to disable)")

    args = ap.parse_args()

    mediators = [x.strip() for x in args.mediators.split(",") if x.strip()]
    if not mediators:
        raise RuntimeError("No mediators specified. Use --mediators Ir,Ge or similar.")

    all_dir = parse_edges_csv(args.edges)

    # detect atom sets
    tc_atoms = sorted({e.u_atom for e in all_dir if e.u_elem == "Tc"} | {e.v_atom for e in all_dir if e.v_elem == "Tc"})
    se_atoms = sorted({e.u_atom for e in all_dir if e.u_elem == "Se"} | {e.v_atom for e in all_dir if e.v_elem == "Se"})
    if not tc_atoms:
        raise RuntimeError("No Tc atoms found (elem == 'Tc').")
    if not se_atoms:
        raise RuntimeError("No Se atoms found (elem == 'Se').")

    med_atoms = {M: sorted({e.u_atom for e in all_dir if e.u_elem == M} | {e.v_atom for e in all_dir if e.v_elem == M})
                 for M in mediators}

    print(f"[INFO] Tc atoms: {tc_atoms}")
    print(f"[INFO] Se atoms: {se_atoms}")
    for M in mediators:
        print(f"[INFO] {M} atoms: {med_atoms.get(M, [])}")

    # ---------- build filtered edge pools ----------
    # Tc->Se and Se->Tc
    tc_to_se = defaultdict(list)   # key Tc atom -> edges Tc->Se
    se_to_tc = defaultdict(list)   # key Se atom -> edges Se->Tc

    # Se->X and X->Se for each mediator X
    se_to_x = {M: defaultdict(list) for M in mediators}  # key Se atom -> edges Se->M
    x_to_se = {M: defaultdict(list) for M in mediators}  # key M atom  -> edges M->Se

    for e in all_dir:
        if e.absH < args.min_absH:
            continue

        # Tc<->Se
        if e.u_elem == "Tc" and e.v_elem == "Se" and e.dist <= args.d_tcse:
            tc_to_se[e.u_atom].append(e)
            continue
        if e.u_elem == "Se" and e.v_elem == "Tc" and e.dist <= args.d_tcse:
            se_to_tc[e.u_atom].append(e)
            continue

        # Se<->X
        if e.dist <= args.d_sex:
            for M in mediators:
                if e.u_elem == "Se" and e.v_elem == M:
                    se_to_x[M][e.u_atom].append(e)
                elif e.u_elem == M and e.v_elem == "Se":
                    x_to_se[M][e.u_atom].append(e)

    # ---------- prune edges ----------
    keep_top_per_key(tc_to_se, topn=max(args.top_per_se_tc, 30))
    keep_top_per_key(se_to_tc, topn=max(args.top_per_se_tc, 30))

    for M in mediators:
        keep_top_per_key(se_to_x[M], topn=max(args.top_per_se_x, 50))
        keep_top_per_key(x_to_se[M], topn=max(args.top_per_x, 50))

    # Build by-Se index for Tc->Se (so we can start from Se if needed)
    tc_to_se_by_se = defaultdict(list)  # key Se atom -> edges Tc->Se (from any Tc)
    for tc, lst in tc_to_se.items():
        for e in lst:
            tc_to_se_by_se[e.v_atom].append(e)
    for se, lst in tc_to_se_by_se.items():
        lst.sort(key=lambda x: x.absH, reverse=True)
        if len(lst) > args.top_per_se_tc:
            tc_to_se_by_se[se] = lst[:args.top_per_se_tc]

    # ---------- enumerate paths ----------
    # Path: Tc0@0 -> Se1 -> X -> Se2 -> Tc1
    paths = []

    def push_path(M, W, tc0, se1, x, se2, tc1, netR, e1, e2, e3, e4):
        paths.append((
            M, W,
            tc0, se1, x, se2, tc1,
            netR,
            e1, e2, e3, e4
        ))

    for tc0 in tc_atoms:
        shift0 = (0, 0, 0)

        for e1 in tc_to_se.get(tc0, []):  # Tc->Se1
            se1 = e1.v_atom
            shift_se1 = add_shift(shift0, (e1.Rx, e1.Ry, e1.Rz))

            for M in mediators:
                # Se1 -> X
                for e2 in se_to_x[M].get(se1, []):
                    x = e2.v_atom
                    shift_x = add_shift(shift_se1, (e2.Rx, e2.Ry, e2.Rz))

                    # X -> Se2
                    for e3 in x_to_se[M].get(x, []):
                        se2 = e3.v_atom
                        shift_se2 = add_shift(shift_x, (e3.Rx, e3.Ry, e3.Rz))

                        # Se2 -> Tc1
                        for e4 in se_to_tc.get(se2, []):
                            tc1 = e4.v_atom
                            shift_tc1 = add_shift(shift_se2, (e4.Rx, e4.Ry, e4.Rz))

                            if args.require_same_tc_atom and (tc1 != tc0):
                                continue

                            netR = shift_tc1
                            if args.max_netR_L1 < 999 and abs_R_L1(netR) > args.max_netR_L1:
                                continue

                            W = e1.absH * e2.absH * e3.absH * e4.absH
                            push_path(M, W, tc0, se1, x, se2, tc1, netR, e1, e2, e3, e4)

    if not paths:
        raise RuntimeError(
            "No paths found. Try: lower --min_absH, increase --d_tcse/--d_sex, "
            "or relax --max_netR_L1 / remove --require_same_tc_atom."
        )

    # sort by weight
    paths.sort(key=lambda x: x[1], reverse=True)
    paths = paths[:args.top_paths]

    # ---------- write output ----------
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "mediator", "W_path",
            "Tc0_atom", "Se1_atom", "X_atom", "Se2_atom", "Tc1_atom",
            "net_dRx", "net_dRy", "net_dRz",
            # step1 Tc->Se1
            "t1_absH", "d1_A", "wf1_u", "wf1_v", "R1x", "R1y", "R1z",
            # step2 Se1->X
            "t2_absH", "d2_A", "wf2_u", "wf2_v", "R2x", "R2y", "R2z",
            # step3 X->Se2
            "t3_absH", "d3_A", "wf3_u", "wf3_v", "R3x", "R3y", "R3z",
            # step4 Se2->Tc1
            "t4_absH", "d4_A", "wf4_u", "wf4_v", "R4x", "R4y", "R4z",
        ])

        for M, W, tc0, se1, x, se2, tc1, netR, e1, e2, e3, e4 in paths:
            w.writerow([
                M, f"{W:.8g}",
                tc0, se1, x, se2, tc1,
                netR[0], netR[1], netR[2],

                f"{e1.absH:.8g}", f"{e1.dist:.6f}", e1.u_wf, e1.v_wf, e1.Rx, e1.Ry, e1.Rz,
                f"{e2.absH:.8g}", f"{e2.dist:.6f}", e2.u_wf, e2.v_wf, e2.Rx, e2.Ry, e2.Rz,
                f"{e3.absH:.8g}", f"{e3.dist:.6f}", e3.u_wf, e3.v_wf, e3.Rx, e3.Ry, e3.Rz,
                f"{e4.absH:.8g}", f"{e4.dist:.6f}", e4.u_wf, e4.v_wf, e4.Rx, e4.Ry, e4.Rz,
            ])

    print(f"[DONE] wrote {len(paths)} paths to: {args.out}")
    print("[TIP] If too slow: raise --min_absH (e.g. 2e-3~5e-3) or lower --top_per_*.")


if __name__ == "__main__":
    main()


```

使用方法：

```shell
python 2step.py --edges 2step.csv --out Ir.csv --mediators Ir --d_tcse 3.0 --d_sex 3.0 --min_absH 1e-4 --top_paths 2000 --require_same_tc_atom --max_netR_L1 6 --exclude_netR0 --delta_mode sequential --delta_floor 1e-3 --hr wannier90_hr.dat
```

```shell
python 2step.py --edges 2step.csv --out Ir_0.1_delta_floor.csv --mediators Ir --d_tcse 3.0 --d_sex 3.0 --min_absH 1e-4 --top_paths 2000 --require_same_tc_atom --max_netR_L1 6 --exclude_netR0 --delta_mode sequential --delta_floor 1e-1 --hr wannier90_hr.dat
```
分别输出Ge和Ir


源文件目录：/public/home/cssong/song/1mrx/9_single_layer/19_ReIrGe2S6/TcIrGeSe/
3_static_band/4_wannier/ncl
(4)根据你想要探究的目的，超超交换路径是通过提供铁电性的迁移原子 Ge 还是非磁性原子 Ir 来主导，
我们可以修改脚本，让它在输出路径时能够分别计算 Ge 和 Ir 路径的贡献，并将这些路径的 权重贡献 
和参与原子频率明确标识出来。

这一步与(3)不矛盾，但代码有所不同，选择(4)也许更准一些。

```python

import csv
import argparse
from dataclasses import dataclass
from collections import defaultdict
from typing import Tuple, List, Dict


@dataclass(frozen=True)
class DirEdge:
    u_atom: int          # 1-based atom number
    v_atom: int
    u_elem: str
    v_elem: str
    u_wf: int
    v_wf: int
    Rx: int
    Ry: int
    Rz: int
    absH: float          # hopping magnitude
    dist: float          # distance


def parse_edges_csv(path: str) -> List[DirEdge]:
    edges = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        required = ["dist_A", "absH_eV", "Rx", "Ry", "Rz",
                    "m", "n", "atom_m", "atom_n", "elem_m", "elem_n"]
        for k in required:
            if k not in r.fieldnames:
                raise RuntimeError(f"Missing column '{k}' in {path}. Found: {r.fieldnames}")

        for row in r:
            try:
                dist = float(row["dist_A"])
                absH = float(row["absH_eV"])
                Rx, Ry, Rz = int(row["Rx"]), int(row["Ry"]), int(row["Rz"])
                m, n = int(row["m"]), int(row["n"])
                atom_m, atom_n = int(row["atom_m"]), int(row["atom_n"])
                elem_m, elem_n = row["elem_m"].strip(), row["elem_n"].strip()
            except Exception:
                continue

            # Add both directions of the edge
            edges.append(DirEdge(atom_m, atom_n, elem_m, elem_n, m, n, Rx, Ry, Rz, absH, dist))
            edges.append(DirEdge(atom_n, atom_m, elem_n, elem_m, n, m, -Rx, -Ry, -Rz, absH, dist))
    return edges


def add_shift(s: Tuple[int, int, int], R: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (s[0] + R[0], s[1] + R[1], s[2] + R[2])


def abs_R_L1(R: Tuple[int, int, int]) -> int:
    return abs(R[0]) + abs(R[1]) + abs(R[2])


def keep_top_per_key(d: Dict[int, List[DirEdge]], topn: int):
    for k in list(d.keys()):
        lst = d[k]
        lst.sort(key=lambda x: x.absH, reverse=True)
        if len(lst) > topn:
            d[k] = lst[:topn]


def main():
    ap = argparse.ArgumentParser(description="Build Tc–Se–X–Se–Tc paths from out_edges csv and rank by hopping-product weight.")
    ap.add_argument("--edges", required=True, help="Input edges csv (e.g., 2step.csv)")
    ap.add_argument("--out", default="top_paths.csv", help="Output paths csv")
    ap.add_argument("--d_tcse", type=float, default=3.0, help="Max distance for Tc–Se edges (Ang)")
    ap.add_argument("--d_sex", type=float, default=3.0, help="Max distance for Se–X edges (Ang), applies to all mediators")
    ap.add_argument("--mediators", default="Ir,Ge", help="Comma-separated mediator elements, e.g. Ir,Ge or Ir only or Ge only")
    ap.add_argument("--min_absH", type=float, default=1e-3, help="Min |t| for edges used in path building (eV). 1e-3 is fast; 1e-4 is exhaustive but slower.")
    ap.add_argument("--top_per_se_tc", type=int, default=60, help="Keep top N Tc–Se edges per Se atom")
    ap.add_argument("--top_per_se_x", type=int, default=120, help="Keep top N Se–X edges per Se atom (for each mediator)")
    ap.add_argument("--top_per_x", type=int, default=120, help="Keep top N X–Se edges per X atom (for each mediator)")
    ap.add_argument("--top_paths", type=int, default=800, help="Number of top paths (overall) to write")
    ap.add_argument("--require_same_tc_atom", action="store_true", help="Require end Tc atom id == start Tc atom id (useful when there is only one Tc in primitive cell)")
    ap.add_argument("--max_netR_L1", type=int, default=6, help="Filter by |dRx|+|dRy|+|dRz| <= this (set 999 to disable)")

    args = ap.parse_args()

    mediators = [x.strip() for x in args.mediators.split(",") if x.strip()]
    if not mediators:
        raise RuntimeError("No mediators specified. Use --mediators Ir,Ge or similar.")

    all_dir = parse_edges_csv(args.edges)

    # detect atom sets
    tc_atoms = sorted({e.u_atom for e in all_dir if e.u_elem == "Tc"} | {e.v_atom for e in all_dir if e.v_elem == "Tc"})
    se_atoms = sorted({e.u_atom for e in all_dir if e.u_elem == "Se"} | {e.v_atom for e in all_dir if e.v_elem == "Se"})
    if not tc_atoms:
        raise RuntimeError("No Tc atoms found (elem == 'Tc').")
    if not se_atoms:
        raise RuntimeError("No Se atoms found (elem == 'Se').")
    med_atoms = {M: sorted({e.u_atom for e in all_dir if e.u_elem == M} | {e.v_atom for e in all_dir if e.v_elem == M}) for M in mediators}

    print(f"[INFO] Tc atoms: {tc_atoms}")
    print(f"[INFO] Se atoms: {se_atoms}")
    for M in mediators:
        print(f"[INFO] {M} atoms: {med_atoms.get(M, [])}")

    # ---------- build filtered edge pools ----------
    tc_to_se = defaultdict(list)  # Tc->Se edges
    se_to_tc = defaultdict(list)  # Se->Tc edges
    se_to_x = {M: defaultdict(list) for M in mediators}  # Se->X edges for each mediator
    x_to_se = {M: defaultdict(list) for M in mediators}  # X->Se edges for each mediator

    for e in all_dir:
        if e.absH < args.min_absH:
            continue

        # Tc<->Se
        if e.u_elem == "Tc" and e.v_elem == "Se" and e.dist <= args.d_tcse:
            tc_to_se[e.u_atom].append(e)
            continue
        if e.u_elem == "Se" and e.v_elem == "Tc" and e.dist <= args.d_tcse:
            se_to_tc[e.u_atom].append(e)
            continue

        # Se<->X
        if e.dist <= args.d_sex:
            for M in mediators:
                if e.u_elem == "Se" and e.v_elem == M:
                    se_to_x[M][e.u_atom].append(e)
                elif e.u_elem == M and e.v_elem == "Se":
                    x_to_se[M][e.u_atom].append(e)

    # ---------- prune edges ----------
    keep_top_per_key(tc_to_se, topn=max(args.top_per_se_tc, 30))
    keep_top_per_key(se_to_tc, topn=max(args.top_per_se_tc, 30))
    for M in mediators:
        keep_top_per_key(se_to_x[M], topn=max(args.top_per_se_x, 50))
        keep_top_per_key(x_to_se[M], topn=max(args.top_per_x, 50))

    # ---------- build paths ----------
    paths = []

    def push_path(M, W, tc0, se1, x, se2, tc1, netR, e1, e2, e3, e4):
        paths.append((
            M, W,
            tc0, se1, x, se2, tc1,
            netR,
            e1, e2, e3, e4
        ))

    for tc0 in tc_atoms:
        shift0 = (0, 0, 0)

        for e1 in tc_to_se.get(tc0, []):  # Tc->Se1
            se1 = e1.v_atom
            shift_se1 = add_shift(shift0, (e1.Rx, e1.Ry, e1.Rz))

            for M in mediators:
                # Se1 -> X
                for e2 in se_to_x[M].get(se1, []):
                    x = e2.v_atom
                    shift_x = add_shift(shift_se1, (e2.Rx, e2.Ry, e2.Rz))

                    # X -> Se2
                    for e3 in x_to_se[M].get(x, []):
                        se2 = e3.v_atom
                        shift_se2 = add_shift(shift_x, (e3.Rx, e3.Ry, e3.Rz))

                        # Se2 -> Tc1
                        for e4 in se_to_tc.get(se2, []):
                            tc1 = e4.v_atom
                            shift_tc1 = add_shift(shift_se2, (e4.Rx, e4.Ry, e4.Rz))

                            if args.require_same_tc_atom and (tc1 != tc0):
                                continue

                            netR = shift_tc1
                            if args.max_netR_L1 < 999 and abs_R_L1(netR) > args.max_netR_L1:
                                continue

                            W = e1.absH * e2.absH * e3.absH * e4.absH
                            push_path(M, W, tc0, se1, x, se2, tc1, netR, e1, e2, e3, e4)

    if not paths:
        raise RuntimeError(
            "No paths found. Try: lower --min_absH, increase --d_tcse/--d_sex, "
            "or relax --max_netR_L1 / remove --require_same_tc_atom."
        )

    # sort by weight
    paths.sort(key=lambda x: x[1], reverse=True)
    paths = paths[:args.top_paths]

    # ---------- write output ----------
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "mediator", "W_path",
            "Tc0_atom", "Se1_atom", "X_atom", "Se2_atom", "Tc1_atom",
            "net_dRx", "net_dRy", "net_dRz",
            # step1 Tc->Se1
            "t1_absH", "d1_A", "wf1_u", "wf1_v", "R1x", "R1y", "R1z",
            # step2 Se1->X
            "t2_absH", "d2_A", "wf2_u", "wf2_v", "R2x", "R2y", "R2z",
            # step3 X->Se2
            "t3_absH", "d3_A", "wf3_u", "wf3_v", "R3x", "R3y", "R3z",
            # step4 Se2->Tc1
            "t4_absH", "d4_A", "wf4_u", "wf4_v", "R4x", "R4y", "R4z",
        ])

        for M, W, tc0, se1, x, se2, tc1, netR, e1, e2, e3, e4 in paths:
            w.writerow([
                M, f"{W:.8g}",
                tc0, se1, x, se2, tc1,
                netR[0], netR[1], netR[2],

                f"{e1.absH:.8g}", f"{e1.dist:.6f}", e1.u_wf, e1.v_wf, e1.Rx, e1.Ry, e1.Rz,
                f"{e2.absH:.8g}", f"{e2.dist:.6f}", e2.u_wf, e2.v_wf, e2.Rx, e2.Ry, e2.Rz,
                f"{e3.absH:.8g}", f"{e3.dist:.6f}", e3.u_wf, e3.v_wf, e3.Rx, e3.Ry, e3.Rz,
                f"{e4.absH:.8g}", f"{e4.dist:.6f}", e4.u_wf, e4.v_wf, e4.Rx, e4.Ry, e4.Rz,
            ])

    print(f"[DONE] wrote {len(paths)} paths to: {args.out}")
    print("[TIP] If too slow: raise --min_absH (e.g. 2e-3~5e-3) or lower --top_per_*.")


if __name__ == "__main__":
    main()


```
要依次修改delta_floor的值来确保全局不会因为分子过小而导致错误。分别测试0.1，0.5，1 。

(5) 与DFT结果的对比

上述代码都是从全局角度分析Ir和Ge究竟谁占超超交换的主导，而J1是DFT计算得到的J的主导，通过在
TB模型中计算J1元素分辨，来判断谁的贡献更大，代码如下：

```python

import pandas as pd
import argparse

J1_NETR = {
    ( 1, 0, 0),
    (-1, 0, 0),
    ( 0, 1, 0),
    ( 0,-1, 0),
    ( 1, 1, 0),
    (-1,-1, 0),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path file (Ir_xxx.csv or Ge_xxx.csv)")
    ap.add_argument("--topk", type=int, default=100, help="sum topK scores after filtering to J1 shell")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # 兼容列名：你文件里通常是 net_dRx/net_dRy/net_dRz 与 score_num_over_denom
    for c in ["net_dRx", "net_dRy", "net_dRz", "score_num_over_denom", "mediator"]:
        if c not in df.columns:
            raise RuntimeError(f"missing column: {c}, have: {list(df.columns)}")

    df["netR"] = list(zip(df["net_dRx"], df["net_dRy"], df["net_dRz"]))
    sub = df[df["netR"].isin(J1_NETR)].copy()

    if sub.empty:
        raise RuntimeError("No rows in J1 shell after filtering. Check your net_dR convention.")

    sub = sub.sort_values("score_num_over_denom", ascending=False)

    topk = min(args.topk, len(sub))
    ssum = sub["score_num_over_denom"].head(topk).sum()
    smax = sub["score_num_over_denom"].iloc[0]
    med = sub["mediator"].iloc[0]

    print(f"[INFO] file: {args.csv}")
    print(f"[INFO] mediator (top row): {med}")
    print(f"[INFO] J1-shell rows: {len(sub)}")
    print(f"[INFO] top1 score: {smax:.6g}")
    print(f"[INFO] sum top{topk} score: {ssum:.6g}")

if __name__ == "__main__":
    main()


```

运行方法：

```shell
python j1_filter_sum.py --csv Ir_1.0_delta_floor.csv --topk 100
python j1_filter_sum.py --csv Ge_1.0_delta_floor.csv --topk 100
```


上述依据：

<img src="/img/超超交换路径/bg-t1.png" style="width:100%; margin-bottom:5px;">
<img src="/img/超超交换路径/bg-t2.png" style="width:100%; margin-bottom:5px;">
<img src="/img/超超交换路径/bg-t3.png" style="width:100%; margin-bottom:5px;">


## <center>实际应用</center>

比较Ir和Ge哪个贡献更大（上面已经完成），下方是在不考虑soc的情况下，分别对spin up和spin down进行贡献比较，确保spin up能代表性的指出交换路径是d-p杂化组成的。


(1) 产生1step文件
```shell
    python 1step.py --win wannier90.1.win --centres wannier90.1_centres.xyz --hr wannier90.1_hr.dat --tol 0.10 --min_absH 1e-3 --topk 30 --out_summary 1step_up.csv --pairs ALL --skip_same_atom_R0 --out_edges 2step_up.csv

    python 1step.py --win wannier90.2.win --centres wannier90.2_centres.xyz --hr wannier90.2_hr.dat --tol 0.10 --min_absH 1e-3 --topk 30 --out_summary 1step_dw.csv --pairs ALL --skip_same_atom_R0 --out_edges 2step_dw.csv
```  

(2) 产生2step文件

```shell
    python 2step.py --edges 2step_dw.csv --out dw.csv --mediators Ir --d_tcse 3.0 --d_sex 3.0 --min_absH 1e-3 --top_paths 2000 --require_same_tc_atom --max_netR_L1 6 --exclude_netR0 --delta_mode sequential --delta_floor 1e-3 --hr wannier90.2_hr.dat

    python 2step.py --edges 2step_up.csv --out up.csv --mediators Ir --d_tcse 3.0 --d_sex 3.0 --min_absH 1e-3 --top_paths 2000 --require_same_tc_atom --max_netR_L1 6 --exclude_netR0 --delta_mode sequential --delta_floor 1e-3 --hr wannier90.1_hr.dat

    python 2step.py --edges 2step_dw.csv --out dw.csv --mediators Ge --d_tcse 3.0 --d_sex 3.0 --min_absH 1e-3 --top_paths 2000 --require_same_tc_atom --max_netR_L1 6 --exclude_netR0 --delta_mode sequential --delta_floor 1e-3 --hr wannier90.2_hr.dat

    python 2step.py --edges 2step_up.csv --out up.csv --mediators Ge --d_tcse 3.0 --d_sex 3.0 --min_absH 1e-3 --top_paths 2000 --require_same_tc_atom --max_netR_L1 6 --exclude_netR0 --delta_mode sequential --delta_floor 1e-3 --hr wannier90.1_hr.dat
```
(3)与DFT进行对比
```shell
    python compare_DFT_J1.py --csv dw.csv --topk 100
    python compare_DFT_J1.py --csv up.csv --topk 100
```
这一步的输出类似于
```shell
    [INFO] file: up.csv
    [INFO] mediator (top row): Ge
    [INFO] J1-shell rows: 2000
    [INFO] top1 score: 70.4132
    [INFO] sum top100 score: 2565.74
```

结论是，Ir比Ge贡献大的多，经过改变dleta_floor，也是一样的结论，spin up和spin down并不能在数量上严格一致，但只要都支持d-p-d-p-d结论即可，用前面的评分标准定量分析，用MLWFS
的spin up通道定性演示(因为spin down的图极为混乱，看不清化学键)