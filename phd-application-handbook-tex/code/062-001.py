#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1step.py (Index-based robust version)  [SOC-ready + output re/im]

输出 edges CSV 现在包含：
  absH_eV, reH_eV, imH_eV, phase_rad, phase_deg
便于后续做：
  - 轨道杂化 (absH / absH^2)
  - 2×2 SOC spinor block Frobenius norm / singular values (更接近文献“hopping幅度”口径)
"""

import re
import csv
import math
import argparse
from collections import defaultdict, namedtuple
import numpy as np

# -------------------------
# Edge record
# -------------------------
Edge = namedtuple(
    "Edge",
    "pair shell dist absH reH imH phase_rad Rx Ry Rz m n atom_m atom_n elem_m elem_n group_m group_n orb_m orb_n"
)

# =========================
# Parse wannier90.win
# =========================
def _extract_block(txt, block_name):
    m = re.search(rf"begin\s+{block_name}(.*?)end\s+{block_name}", txt, re.S | re.I)
    return None if not m else m.group(1).strip()

def parse_win_lattice_and_atoms(win_path):
    with open(win_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    cell_block = _extract_block(txt, "unit_cell_cart")
    if cell_block is None:
        raise RuntimeError("Cannot find unit_cell_cart block in wannier90.win")

    lines = [ln.strip() for ln in cell_block.splitlines() if ln.strip()]
    if re.match(r"^(ang|angstrom|bohr)\b", lines[0], re.I):
        unit = lines[0].lower()
        vec_lines = lines[1:4]
    else:
        unit = "ang"
        vec_lines = lines[0:3]

    a1 = np.array([float(x) for x in vec_lines[0].split()[:3]], dtype=float)
    a2 = np.array([float(x) for x in vec_lines[1].split()[:3]], dtype=float)
    a3 = np.array([float(x) for x in vec_lines[2].split()[:3]], dtype=float)

    if "bohr" in unit:
        bohr_to_ang = 0.52917721092
        a1 *= bohr_to_ang
        a2 *= bohr_to_ang
        a3 *= bohr_to_ang

    A = np.stack([a1, a2, a3], axis=1)  # columns are lattice vectors

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
# Parse centres.xyz (WF-only)
# =========================
def parse_wf_centres_from_xyz(xyz_path, natoms, nwann):
    with open(xyz_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    N = int(lines[0])
    if N < natoms + nwann:
        raise RuntimeError(f"centres.xyz N={N} < natoms+nwann={natoms+nwann}")

    start = 2 + natoms
    end = start + nwann
    if len(lines) < end:
        raise RuntimeError(f"centres.xyz insufficient lines: need >= {end}, got {len(lines)}")

    wf_centers = [None] * (nwann + 1)
    for i in range(nwann):
        parts = lines[start + i].split()
        if len(parts) < 4:
            raise RuntimeError(f"Bad WF centre line: {lines[start+i]}")
        wf_centers[i + 1] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)

    return wf_centers

# =========================
# Parse hr.dat
# =========================
def read_hr_header_num_wann(hr_path):
    with open(hr_path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        return int(f.readline().strip())

def parse_hr_dat(hr_path):
    """
    Yield (Rx,Ry,Rz,m,n,reH,imH) from wannier90_hr.dat. m,n are 1-based.
    """
    with open(hr_path, "r", encoding="utf-8", errors="ignore") as f:
        _ = f.readline()
        _num_wann = int(f.readline().strip())
        nrpts = int(f.readline().strip())

        deg = []
        while len(deg) < nrpts:
            ln = f.readline()
            if not ln:
                raise RuntimeError("Unexpected EOF while reading degeneracy list")
            ln = ln.strip()
            if not ln:
                continue
            deg += [int(x) for x in ln.split()]

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
# Group/order helpers
# =========================
def pair_name(g1, g2, order_map):
    o1 = order_map.get(g1, 999)
    o2 = order_map.get(g2, 999)
    if o1 < o2:
        return f"{g1}<->{g2}"
    if o2 < o1:
        return f"{g2}<->{g1}"
    return f"{min(g1, g2)}<->{max(g1, g2)}"

# =========================
# Index-based WF mapping
# =========================
D_ORBS = ["dxy", "dyz", "dxz", "dz2", "dx2-y2"]
P_ORBS = ["px", "py", "pz"]

def build_index_mapping(atoms, num_wann):
    if num_wann != 68:
        raise RuntimeError(
            f"Expected num_wann=68 for your system, but got {num_wann}. "
            f"If your projections changed, update index blocks accordingly."
        )

    idx_Tc = [i for i,a in enumerate(atoms) if a["elem"] == "Tc"]
    idx_Ir = [i for i,a in enumerate(atoms) if a["elem"] == "Ir"]
    idx_Se = [i for i,a in enumerate(atoms) if a["elem"] == "Se"]
    idx_Ge = [i for i,a in enumerate(atoms) if a["elem"] == "Ge"]

    if len(idx_Tc) != 1 or len(idx_Ir) != 1 or len(idx_Se) != 6 or len(idx_Ge) != 2:
        raise RuntimeError(
            f"Atom counts mismatch: Tc={len(idx_Tc)}, Ir={len(idx_Ir)}, "
            f"Se={len(idx_Se)}, Ge={len(idx_Ge)}. Check atoms_cart."
        )

    wf_atom = [None] * (num_wann + 1)
    wf_elem = [None] * (num_wann + 1)
    wf_group = [None] * (num_wann + 1)
    wf_orb = [None] * (num_wann + 1)

    for wf in range(1, 11):
        wf_atom[wf] = idx_Tc[0]
        wf_elem[wf] = "Tc"
        wf_group[wf] = "Tc_d"
        i = (wf - 1) // 2
        wf_orb[wf] = D_ORBS[i]

    for wf in range(11, 21):
        wf_atom[wf] = idx_Ir[0]
        wf_elem[wf] = "Ir"
        wf_group[wf] = "Ir_d"
        i = (wf - 11) // 2
        wf_orb[wf] = D_ORBS[i]

    start = 21
    for a_i in range(6):
        atom_idx = idx_Se[a_i]
        for local in range(6):
            wf = start + a_i * 6 + local
            wf_atom[wf] = atom_idx
            wf_elem[wf] = "Se"
            wf_group[wf] = "Se_p"
            j = local // 2
            wf_orb[wf] = P_ORBS[j]

    start = 57
    for a_i in range(2):
        atom_idx = idx_Ge[a_i]
        for local in range(6):
            wf = start + a_i * 6 + local
            wf_atom[wf] = atom_idx
            wf_elem[wf] = "Ge"
            wf_group[wf] = "Ge_p"
            j = local // 2
            wf_orb[wf] = P_ORBS[j]

    return wf_atom, wf_elem, wf_group, wf_orb

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", required=True, help="wannier90.win")
    ap.add_argument("--centres", required=True, help="wannier90_centres.xyz")
    ap.add_argument("--hr", required=True, help="wannier90_hr.dat")
    ap.add_argument("--tol", type=float, default=0.10, help="distance bin size (Ang)")
    ap.add_argument("--min_absH", type=float, default=1e-4, help="min |H| kept (eV)")
    ap.add_argument("--topk", type=int, default=30, help="topK edges per (pair,shell)")
    ap.add_argument("--skip_same_atom_R0", action="store_true")
    ap.add_argument("--skip_diag_R0", action="store_true")
    ap.add_argument("--pairs", default="ALL", help="comma-separated allowed pairs like Tc_d<->Se_p, or ALL")
    ap.add_argument("--out_summary", default="hopping_summary_by_shell.csv")
    ap.add_argument("--out_edges", default="", help="optional edges csv output")
    ap.add_argument("--group_order", default="Tc_d,Ir_d,Se_p,Ge_p,OTHER")

    args = ap.parse_args()

    order_list = [x.strip() for x in args.group_order.split(",") if x.strip()]
    order_map = {g: i for i, g in enumerate(order_list, start=1)}

    A, atoms = parse_win_lattice_and_atoms(args.win)
    natoms = len(atoms)
    num_wann = read_hr_header_num_wann(args.hr)
    wf_centers = parse_wf_centres_from_xyz(args.centres, natoms=natoms, nwann=num_wann)

    wf_atom_id, wf_elem, wf_group, wf_orb = build_index_mapping(atoms, num_wann)

    counts = defaultdict(int)
    for wf in range(1, num_wann + 1):
        counts[wf_elem[wf]] += 1
    print("[INFO] num_wann:", num_wann, " natoms:", natoms)
    print("[INFO] WF counts by elem (index-based):", dict(sorted(counts.items())))

    allowed_pairs = None
    if args.pairs.strip().upper() != "ALL":
        allowed_pairs = set([p.strip() for p in args.pairs.split(",") if p.strip()])

    cnt = defaultdict(int)
    sumsq = defaultdict(float)
    maxv = defaultdict(float)
    top_edges = defaultdict(list)
    filtered_edges = []

    a1 = A[:, 0]; a2 = A[:, 1]; a3 = A[:, 2]
    def R_to_T(Rx, Ry, Rz):
        return Rx * a1 + Ry * a2 + Rz * a3

    for Rx, Ry, Rz, m, n, reH, imH in parse_hr_dat(args.hr):
        absH = math.hypot(reH, imH)
        if absH < args.min_absH:
            continue

        atom_m = wf_atom_id[m]
        atom_n = wf_atom_id[n]
        elem_m = wf_elem[m]
        elem_n = wf_elem[n]
        gm = wf_group[m]
        gn = wf_group[n]
        if gm == "OTHER" or gn == "OTHER":
            continue

        pair = pair_name(gm, gn, order_map=order_map)
        if allowed_pairs is not None and pair not in allowed_pairs:
            continue

        if (Rx, Ry, Rz) == (0, 0, 0):
            if args.skip_diag_R0 and (m == n):
                continue
            if args.skip_same_atom_R0 and (atom_m == atom_n):
                continue

        T = R_to_T(Rx, Ry, Rz)
        dr = (wf_centers[n] + T) - wf_centers[m]
        dist = float(np.linalg.norm(dr))
        shell = round(dist / args.tol) * args.tol

        # complex phase
        phase_rad = math.atan2(imH, reH) if absH > 0 else 0.0

        key = (pair, shell)
        cnt[key] += 1
        sumsq[key] += absH * absH
        if absH > maxv[key]:
            maxv[key] = absH

        e = Edge(
            pair, shell, dist, absH, reH, imH, phase_rad,
            Rx, Ry, Rz, m, n,
            atom_m, atom_n, elem_m, elem_n,
            gm, gn, wf_orb[m], wf_orb[n]
        )

        lst = top_edges[key]
        lst.append(e)
        lst.sort(key=lambda x: x.absH, reverse=True)
        if len(lst) > args.topk:
            lst[:] = lst[:args.topk]

        if args.out_edges:
            filtered_edges.append(e)

    # ---- summary ----
    with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pair", "shell_A", "count", "max_absH_eV", "rms_absH_eV",
            f"top{args.topk}_edges(m,n,atom_m,atom_n,group_m,group_n,orb_m,orb_n,R,absH,dist,phase)"
        ])
        for (pair, shell) in sorted(cnt.keys(), key=lambda x: (x[0], x[1])):
            c = cnt[(pair, shell)]
            rms = math.sqrt(sumsq[(pair, shell)] / c) if c else 0.0
            tops = top_edges[(pair, shell)]
            tops_str = "; ".join([
                f"{e.m}-{e.n}|a{e.atom_m+1}-a{e.atom_n+1}"
                f"|{e.group_m}-{e.group_n}|{e.orb_m}-{e.orb_n}"
                f"@({e.Rx},{e.Ry},{e.Rz})|{e.absH:.6g}|d={e.dist:.3f}|ph={e.phase_rad:.3f}"
                for e in tops
            ])
            w.writerow([pair, f"{shell:.3f}", c, f"{maxv[(pair, shell)]:.6g}", f"{rms:.6g}", tops_str])

    # ---- edges ----
    if args.out_edges:
        with open(args.out_edges, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "pair", "shell_A", "dist_A",
                "absH_eV", "reH_eV", "imH_eV", "phase_rad", "phase_deg",
                "Rx", "Ry", "Rz", "m", "n",
                "atom_m", "atom_n", "elem_m", "elem_n",
                "group_m", "group_n", "orb_m", "orb_n"
            ])
            for e in filtered_edges:
                w.writerow([
                    e.pair, f"{e.shell:.3f}", f"{e.dist:.6f}",
                    f"{e.absH:.10g}", f"{e.reH:.10g}", f"{e.imH:.10g}",
                    f"{e.phase_rad:.10g}", f"{(e.phase_rad * 180.0 / math.pi):.10g}",
                    e.Rx, e.Ry, e.Rz, e.m, e.n,
                    e.atom_m + 1, e.atom_n + 1, e.elem_m, e.elem_n,
                    e.group_m, e.group_n, e.orb_m, e.orb_n
                ])

    print("Done.")
    print("Summary:", args.out_summary)
    if args.out_edges:
        print("Edges :", args.out_edges)

if __name__ == "__main__":
    main()
