import re
import csv
import math
import argparse
from collections import defaultdict, namedtuple

import numpy as np

Edge = namedtuple(
    "Edge",
    "pair shell dist absH re im Rx Ry Rz m n atom_m atom_n elem_m elem_n group_m group_n"
)

# =========================
# Parse wannier90.win
# =========================
def _extract_block(txt, block_name):
    m = re.search(rf"begin\s+{block_name}(.*?)end\s+{block_name}", txt, re.S | re.I)
    return None if not m else m.group(1).strip()

def parse_win_lattice_and_atoms(win_path):
    """
    Read unit_cell_cart and atoms_cart from wannier90.win.
    Assumes Angstrom if unit not specified.
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

    A = np.stack([a1, a2, a3], axis=1)  # (3,3), columns are lattice vectors

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
        _num_wann = int(f.readline().strip())
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
# Group mapping: element -> group label
# =========================
def parse_group_map(s):
    """
    "Tc:Tc_d,Ir:Ir_d,Se:Se_p,Ge:Ge_p" -> dict
    """
    mp = {}
    s = (s or "").strip()
    if not s:
        return mp
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad group_map item '{item}', expected Elem:Group")
        k, v = item.split(":", 1)
        mp[k.strip()] = v.strip()
    return mp

def elem_to_group(elem, group_map):
    """
    Default rules (can be overridden by --group_map):
      Tc -> Tc_d, Ir -> Ir_d, Se -> Se_p, Ge -> Ge_p
    """
    if elem in group_map:
        return group_map[elem]
    if elem == "Tc":
        return "Tc_d"
    if elem == "Ir":
        return "Ir_d"
    if elem == "Se":
        return "Se_p"
    if elem == "Ge":
        return "Ge_p"
    return "OTHER"

def pair_name(g1, g2, order_map):
    """
    Deterministic ordering, avoid Se_p<->Tc_d flip.
    """
    o1 = order_map.get(g1, 999)
    o2 = order_map.get(g2, 999)
    if o1 < o2:
        return f"{g1}<->{g2}"
    if o2 < o1:
        return f"{g2}<->{g1}"
    return f"{min(g1,g2)}<->{max(g1,g2)}"


# =========================
# Minimum-image mapping: WF -> nearest atom
# =========================
def cart_to_frac(A, r_cart):
    return np.linalg.solve(A, r_cart)

def frac_to_cart(A, f):
    return A @ f

def wrap_delta_frac(df):
    return df - np.round(df)

def map_wf_to_atoms(A, centers, atoms):
    """
    Returns:
      wf_atom_id[wf] = atom index in atoms list (0-based)
      wf_atom_elem[wf] = element string
    Uses minimum-image distance in fractional space.
    """
    atom_frac = []
    for a in atoms:
        atom_frac.append(cart_to_frac(A, a["r"]))
    atom_frac = np.array(atom_frac)  # (Nat,3)

    wf_atom_id = [None] * len(centers)
    wf_atom_elem = [None] * len(centers)

    for wf in range(1, len(centers)):
        f_w = cart_to_frac(A, centers[wf])

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
    ap = argparse.ArgumentParser(
        description="Summarize hoppings by (group-pair, distance shell) using WF centres + lattice translations."
    )
    ap.add_argument("--win", required=True, help="wannier90.win")
    ap.add_argument("--centres", required=True, help="wannier90_centres.xyz")
    ap.add_argument("--hr", required=True, help="wannier90_hr.dat")
    ap.add_argument("--tol", type=float, default=0.10, help="distance bin size (Ang), e.g. 0.05~0.15")
    ap.add_argument("--min_absH", type=float, default=1e-4, help="min |H| kept (eV)")
    ap.add_argument("--topk", type=int, default=30, help="topK edges per (pair,shell)")
    ap.add_argument("--skip_same_atom_R0", action="store_true",
                    help="skip terms where (atom_m==atom_n) AND (R==0). Recommended to remove onsite/local terms.")
    ap.add_argument("--skip_diag_R0", action="store_true",
                    help="skip diagonal terms (m==n) at R==0 (onsite).")
    ap.add_argument("--pairs", default="ALL", help="comma-separated allowed pairs like Tc_d<->Se_p, or ALL")
    ap.add_argument("--out_summary", default="hopping_summary_by_shell.csv")
    ap.add_argument("--out_edges", default="", help="optional: write filtered edges csv (empty=off)")

    # NEW: element -> group mapping (to replace hard-coded wf index ranges)
    ap.add_argument("--group_map", default="Tc:Tc_d,Ir:Ir_d,Se:Se_p,Ge:Ge_p",
                    help="Override element->group mapping, e.g. 'Tc:Tc_d,Ir:Ir_d,Se:Se_p'.")
    ap.add_argument("--group_order", default="Tc_d,Ir_d,Se_p,Ge_p,OTHER",
                    help="Group ordering used to format A<->B deterministically.")

    args = ap.parse_args()

    group_map = parse_group_map(args.group_map)
    order_list = [x.strip() for x in args.group_order.split(",") if x.strip()]
    order_map = {g: i for i, g in enumerate(order_list, start=1)}

    A, atoms = parse_win_lattice_and_atoms(args.win)
    centers = parse_centres_xyz(args.centres)

    wf_atom_id, wf_atom_elem = map_wf_to_atoms(A, centers, atoms)

    # quick sanity print: how many WFs mapped to each element
    elem_counts = defaultdict(int)
    for wf in range(1, len(centers)):
        elem_counts[wf_atom_elem[wf]] += 1
    print("[INFO] WF->nearest-atom element counts:", dict(sorted(elem_counts.items(), key=lambda x: x[0])))

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

        atom_m = wf_atom_id[m]
        atom_n = wf_atom_id[n]
        elem_m = wf_atom_elem[m]
        elem_n = wf_atom_elem[n]

        gm = elem_to_group(elem_m, group_map)
        gn = elem_to_group(elem_n, group_map)

        # drop OTHER by default (consistent with your original behavior)
        if gm == "OTHER" or gn == "OTHER":
            continue

        pair = pair_name(gm, gn, order_map=order_map)
        if allowed_pairs is not None and pair not in allowed_pairs:
            continue

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

        e = Edge(pair, shell, dist, absH, reH, imH, Rx, Ry, Rz, m, n,
                 atom_m, atom_n, elem_m, elem_n, gm, gn)

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
            f"top{args.topk}_edges(m,n,atom_m,atom_n,elem_m,elem_n,group_m,group_n,Rx,Ry,Rz,absH,dist)"
        ])
        for (pair, shell) in sorted(cnt.keys(), key=lambda x: (x[0], x[1])):
            c = cnt[(pair, shell)]
            rms = math.sqrt(sumsq[(pair, shell)] / c) if c else 0.0
            tops = top_edges[(pair, shell)]
            tops_str = "; ".join([
                f"{e.m}-{e.n}|a{e.atom_m+1}-a{e.atom_n+1}|{e.elem_m}-{e.elem_n}"
                f"|{e.group_m}-{e.group_n}"
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
                "atom_m", "atom_n", "elem_m", "elem_n",
                "group_m", "group_n"
            ])
            for e in filtered_edges:
                w.writerow([
                    e.pair, f"{e.shell:.3f}", f"{e.dist:.6f}", f"{e.absH:.8g}",
                    f"{e.re:.8g}", f"{e.im:.8g}",
                    e.Rx, e.Ry, e.Rz, e.m, e.n,
                    e.atom_m + 1, e.atom_n + 1, e.elem_m, e.elem_n,
                    e.group_m, e.group_n
                ])

    print("Done.")
    print("Summary:", args.out_summary)
    if args.out_edges:
        print("Edges:", args.out_edges)


if __name__ == "__main__":
    main()
