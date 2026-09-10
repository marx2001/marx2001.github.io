
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1step.py  (Python 3.8/3.9 compatible)

NEW: export edges csv (2step.csv) from wannier90_hr.dat, like your old 1step.py did,
but now robust for num_wann=88 with projections:
  Tc: d
  Se: p
  Ge: d;p
  Ir: d

Core features
-------------
(1) Parse wannier90.win: unit_cell_cart, atoms_cart, projections
(2) Parse wannier90.wout: Final State WF centres & spreads (authoritative)
(3) Parse wannier90_hr.dat: all hopping matrix elements H_{mn}(R)
(4) Assign each WF to an element using quota-balanced assignment so element counts
    match expected from projections & atom counts (handles bond-centred WFs).
(5) Split Ge WFs into Ge_d / Ge_p using spread / onsite / hybrid.
(6) Export directed edges to CSV with columns compatible with your downstream 2step.py:
    shell, dist, absH, re, im, Rx,Ry,Rz, u_wf,v_wf, u_atom,v_atom, u_elem,v_elem, pair

Optional:
- Export a summary CSV per shell.

Usage (export edges)
--------------------
python 1step.py --win wannier90.win --wout wannier90.wout --hr wannier90_hr.dat \
  --out_edges 2step.csv --tol 0.10 --min_absH 1e-6

If you want fewer edges:
  --topk_per_shell 400

Ge d/p splitting mode:
  --ge_split spread   (default)
  --ge_split onsite
  --ge_split hybrid
"""

import argparse
import csv
import json
import re
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np


# =========================
# WIN parsing
# =========================
def _extract_block(txt, name):
    m = re.search(r"begin\s+%s(.*?)end\s+%s" % (re.escape(name), re.escape(name)), txt, re.S | re.I)
    return None if not m else m.group(1).strip()


def parse_win(win_path):
    txt = Path(win_path).read_text(encoding="utf-8", errors="ignore")

    blk = _extract_block(txt, "unit_cell_cart")
    if blk is None:
        raise RuntimeError("Cannot find begin/end unit_cell_cart in win.")
    lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("ang"):
        lines = lines[1:]
    if len(lines) < 3:
        raise RuntimeError("unit_cell_cart must contain 3 lattice vectors.")
    A = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(3)], dtype=float)

    blk = _extract_block(txt, "atoms_cart")
    if blk is None:
        raise RuntimeError("Cannot find begin/end atoms_cart in win.")
    atoms = []
    for ln in blk.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 4:
            continue
        elem = parts[0].strip()
        x, y, z = map(float, parts[1:4])
        atoms.append((elem, np.array([x, y, z], float)))
    if not atoms:
        raise RuntimeError("No atoms parsed from atoms_cart.")

    blk = _extract_block(txt, "projections")
    if blk is None:
        raise RuntimeError("Cannot find begin/end projections in win.")
    proj = defaultdict(set)
    for ln in blk.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith("!"):
            continue
        if ":" not in ln:
            continue
        lhs, rhs = ln.split(":", 1)
        elem = lhs.strip()
        rhs = rhs.strip().replace(" ", "")
        for orb in rhs.split(";"):
            if orb:
                proj[elem].add(orb.lower())
    if not proj:
        raise RuntimeError("Parsed empty projections. Check win projections block.")

    return A, atoms, proj


# =========================
# WOUT parsing (Final State)
# =========================
def parse_wout_final_centres_spreads(wout_path):
    txt = Path(wout_path).read_text(encoding="utf-8", errors="ignore")
    idx = txt.lower().rfind("final state")
    if idx < 0:
        raise RuntimeError("Cannot find 'Final State' in wannier90.wout")

    centres = {}
    spreads = {}

    pat = re.compile(
        r"^\s*WF centre and spread\s+(\d+)\s+\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)\s+([-\d\.]+)\s*$"
    )

    for ln in txt[idx:].splitlines():
        m = pat.match(ln)
        if not m:
            continue
        i = int(m.group(1))
        x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
        s = float(m.group(5))
        centres[i] = np.array([x, y, z], float)
        spreads[i] = s

    if not centres:
        raise RuntimeError("No WF centres parsed from Final State section.")
    num_wann = max(centres.keys())
    return num_wann, centres, spreads


# =========================
# HR parsing
# =========================
def read_hr_all(hr_path):
    """
    Return:
      num_wann, onsite(dict), hoppings(list of tuples)
    where hoppings are (Rx,Ry,Rz,m,n,re,im)
    """
    raw = Path(hr_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(raw) < 4:
        raise RuntimeError("hr.dat too short")
    num_wann = int(raw[1].split()[0])
    nrpts = int(raw[2].split()[0])

    deg = []
    i = 3
    while len(deg) < nrpts:
        deg.extend([int(x) for x in raw[i].split()])
        i += 1
        if i >= len(raw):
            raise RuntimeError("Unexpected EOF while reading degeneracies.")

    onsite = {}
    hops = []
    for ln in raw[i:]:
        parts = ln.split()
        if len(parts) < 7:
            continue
        Rx, Ry, Rz = map(int, parts[0:3])
        m = int(parts[3])
        n = int(parts[4])
        re_ = float(parts[5])
        im_ = float(parts[6])
        hops.append((Rx, Ry, Rz, m, n, re_, im_))
        if Rx == 0 and Ry == 0 and Rz == 0 and m == n:
            onsite[m] = re_
    return num_wann, onsite, hops


# =========================
# PBC helpers
# =========================
def pbc_dist(A, r1, r2):
    invA = np.linalg.inv(A)
    f1 = r1 @ invA
    f2 = r2 @ invA
    df = f2 - f1
    df -= np.round(df)
    dr = df @ A
    return float(np.linalg.norm(dr))


def shift_by_R(A, vec, Rx, Ry, Rz):
    return vec + Rx * A[0] + Ry * A[1] + Rz * A[2]


# =========================
# Expected WF counts from projections + atoms_cart
# =========================
def expected_counts(atoms, proj, num_wann):
    atom_counts = Counter([e for e, _ in atoms])
    orb_dim = {"s": 1, "p": 3, "d": 5, "f": 7}

    base_total = 0
    elem_orb = defaultdict(dict)
    elem_total = {}

    for elem, orbs in proj.items():
        n = atom_counts.get(elem, 0)
        tot = 0
        for o in orbs:
            if o not in orb_dim:
                raise RuntimeError("Unknown orbital '%s' in projections for %s" % (o, elem))
            tot += orb_dim[o] * n
        elem_total[elem] = tot
        base_total += tot
        for o in orbs:
            elem_orb[elem][o] = orb_dim[o] * n

    factor = 2 if num_wann == 2 * base_total else 1

    elem_total = {k: v * factor for k, v in elem_total.items()}
    for elem in list(elem_orb.keys()):
        for o in list(elem_orb[elem].keys()):
            elem_orb[elem][o] *= factor

    s = sum(elem_total.values())
    if s != num_wann:
        raise RuntimeError(
            "Expected counts sum %d != num_wann %d. Check atoms_cart/projections or SOC/spinor." % (s, num_wann)
        )
    return factor, elem_total, elem_orb


# =========================
# Quota-balanced WF->element assignment
# =========================
def assign_wfs_to_elements_quota(A, atoms, centres, elem_quota):
    elem_atoms = defaultdict(list)
    for ai, (elem, pos) in enumerate(atoms, start=1):
        elem_atoms[elem].append((ai, pos))

    wf_choices = {}
    for wf, c in centres.items():
        opts = []
        for elem, lst in elem_atoms.items():
            bestd = 1e18
            best_ai = None
            for ai, pos in lst:
                d = pbc_dist(A, c, pos)
                if d < bestd:
                    bestd = d
                    best_ai = ai
            opts.append((bestd, elem, best_ai))
        opts.sort(key=lambda x: x[0])
        wf_choices[wf] = opts

    order = sorted(centres.keys(), key=lambda wf: wf_choices[wf][0][0])

    quota = dict(elem_quota)
    wf_elem = {}
    wf_atom = {}

    for wf in order:
        assigned = False
        for d, elem, ai in wf_choices[wf]:
            if quota.get(elem, 0) > 0:
                wf_elem[wf] = elem
                wf_atom[wf] = ai
                quota[elem] -= 1
                assigned = True
                break
        if not assigned:
            wf_elem[wf] = wf_choices[wf][0][1]
            wf_atom[wf] = wf_choices[wf][0][2]

    remain = {k: v for k, v in quota.items() if v != 0}
    if remain:
        raise RuntimeError("Quota assignment failed, remaining quota: %s" % remain)
    return wf_elem, wf_atom


# =========================
# Split Ge into d/p
# =========================
def split_ge_dp(wfs_ge, spreads, onsite, n_ge_d, n_ge_p, mode):
    wfs_ge = list(wfs_ge)
    if len(wfs_ge) != n_ge_d + n_ge_p:
        raise RuntimeError("Ge WF count %d != expected %d" % (len(wfs_ge), n_ge_d + n_ge_p))

    if mode == "spread":
        order = sorted(wfs_ge, key=lambda i: spreads[i])
    elif mode == "onsite":
        order = sorted(wfs_ge, key=lambda i: onsite.get(i, 0.0))
    elif mode == "hybrid":
        sp = np.array([spreads[i] for i in wfs_ge], float)
        ep = np.array([onsite.get(i, 0.0) for i in wfs_ge], float)
        spz = (sp - sp.mean()) / (sp.std() + 1e-12)
        epz = (ep - ep.mean()) / (ep.std() + 1e-12)
        score = spz + epz
        order = [w for _, w in sorted(zip(score, wfs_ge), key=lambda x: x[0])]
    else:
        raise ValueError("Unknown ge_split mode: %s" % mode)

    ge_d = set(order[:n_ge_d])
    ge_p = set(order[n_ge_d:n_ge_d + n_ge_p])
    return ge_d, ge_p


# =========================
# Shell clustering by distance
# =========================
def assign_shells_by_tol(dist_list, tol):
    """
    Given distances, cluster into shells by tolerance tol.
    Return:
      shell_id per dist index, and shell_centers list
    """
    if not dist_list:
        return [], []
    order = sorted(range(len(dist_list)), key=lambda i: dist_list[i])
    centers = []
    shell_id = [None] * len(dist_list)

    for idx in order:
        d = dist_list[idx]
        placed = False
        for s, c in enumerate(centers):
            if abs(d - c) <= tol:
                shell_id[idx] = s + 1
                # update center (simple running average)
                centers[s] = (c + d) * 0.5
                placed = True
                break
        if not placed:
            centers.append(d)
            shell_id[idx] = len(centers)
    return shell_id, centers


# =========================
# Main: export edges
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", required=True)
    ap.add_argument("--wout", required=True)
    ap.add_argument("--hr", required=True)

    ap.add_argument("--out_edges", default="", help="Output edges CSV (e.g. 2step.csv). If empty, no edges exported.")
    ap.add_argument("--out_summary", default="", help="Optional per-shell summary CSV.")

    ap.add_argument("--min_absH", type=float, default=1e-6)
    ap.add_argument("--tol", type=float, default=0.10, help="Distance tolerance for shell clustering (Ang).")
    ap.add_argument("--topk_per_shell", type=int, default=0,
                    help="If >0, keep only topK edges (by absH) per shell.")

    ap.add_argument("--ge_split", default="spread", choices=["spread", "onsite", "hybrid"])
    ap.add_argument("--dump_groups", default="", help="Optional JSON to dump WF index sets.")

    args = ap.parse_args()

    A, atoms, proj = parse_win(args.win)
    num_wann_wout, centres, spreads = parse_wout_final_centres_spreads(args.wout)
    num_wann_hr, onsite, hops = read_hr_all(args.hr)

    if num_wann_wout != num_wann_hr:
        raise RuntimeError("num_wann mismatch: wout=%d, hr=%d. Use matching files." % (num_wann_wout, num_wann_hr))
    num_wann = num_wann_wout

    factor, elem_total, elem_orb = expected_counts(atoms, proj, num_wann)
    print("[INFO] num_wann=%d  factor=%d" % (num_wann, factor))
    print("[INFO] expected per-element totals: %s" % dict(elem_total))
    print("[INFO] expected per-element-orbital: %s" % {k: dict(v) for k, v in elem_orb.items()})

    wf_elem, wf_atom = assign_wfs_to_elements_quota(A, atoms, centres, elem_total)

    elem_wfs = defaultdict(list)
    for wf in range(1, num_wann + 1):
        elem_wfs[wf_elem[wf]].append(wf)

    Tc_set = set(elem_wfs.get("Tc", []))
    Se_set = set(elem_wfs.get("Se", []))
    Ir_set = set(elem_wfs.get("Ir", []))
    Ge_set = set(elem_wfs.get("Ge", []))

    # split Ge into d/p (only if Ge has both in projections)
    n_ge_d = elem_orb.get("Ge", {}).get("d", 0)
    n_ge_p = elem_orb.get("Ge", {}).get("p", 0)
    if n_ge_d > 0 and n_ge_p > 0:
        Ge_d, Ge_p = split_ge_dp(list(Ge_set), spreads, onsite, n_ge_d, n_ge_p, mode=args.ge_split)
    else:
        Ge_d, Ge_p = set(Ge_set), set()

    print("[INFO] derived sizes: Tc=%d Se=%d Ir=%d Ge=%d (Ge_d=%d Ge_p=%d)" %
          (len(Tc_set), len(Se_set), len(Ir_set), len(Ge_set), len(Ge_d), len(Ge_p)))

    if args.dump_groups:
        out = {
            "Tc": sorted(Tc_set),
            "Se": sorted(Se_set),
            "Ir": sorted(Ir_set),
            "Ge": sorted(Ge_set),
            "Ge_d": sorted(Ge_d),
            "Ge_p": sorted(Ge_p),
        }
        Path(args.dump_groups).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("[INFO] dumped group WF indices -> %s" % args.dump_groups)

    # Build edges from hr.dat
    rows = []
    dist_list = []

    for (Rx, Ry, Rz, m, n, re_, im_) in hops:
        if m == n and Rx == 0 and Ry == 0 and Rz == 0:
            continue
        if m < 1 or m > num_wann or n < 1 or n > num_wann:
            continue
        absH = float(np.hypot(re_, im_))
        if absH < args.min_absH:
            continue

        # distance between WF centers (n shifted by lattice R)
        cm = centres[m]
        cn = shift_by_R(A, centres[n], Rx, Ry, Rz)
        dist = float(np.linalg.norm(cn - cm))

        u_elem = wf_elem[m]
        v_elem = wf_elem[n]
        u_atom = wf_atom[m]
        v_atom = wf_atom[n]

        # pair label (for downstream filtering/analysis)
        # Additionally label Ge as Ge_d / Ge_p when possible.
        def label_elem_wf(elem, wf):
            if elem == "Ge":
                if wf in Ge_d:
                    return "Ge_d"
                if wf in Ge_p:
                    return "Ge_p"
                return "Ge"
            if elem == "Tc":
                return "Tc_d"   # your projections are Tc:d
            if elem == "Se":
                return "Se_p"   # Se:p
            if elem == "Ir":
                return "Ir_d"   # Ir:d
            return elem

        u_tag = label_elem_wf(u_elem, m)
        v_tag = label_elem_wf(v_elem, n)
        pair = "%s-%s" % (u_tag, v_tag)

        rows.append({
            "shell": 0,  # fill later
            "dist": dist,
            "absH": absH,
            "re": re_,
            "im": im_,
            "Rx": Rx, "Ry": Ry, "Rz": Rz,
            "u_wf": m, "v_wf": n,
            "u_atom": u_atom, "v_atom": v_atom,
            "u_elem": u_elem, "v_elem": v_elem,
            "pair": pair
        })
        dist_list.append(dist)

    if not rows:
        raise RuntimeError("No edges passed filters. Lower --min_absH or check hr.dat.")

    # Assign shells
    shell_id, centers = assign_shells_by_tol(dist_list, tol=args.tol)
    for i, sid in enumerate(shell_id):
        rows[i]["shell"] = sid

    # Optionally keep topK per shell
    if args.topk_per_shell and args.topk_per_shell > 0:
        by_shell = defaultdict(list)
        for r in rows:
            by_shell[r["shell"]].append(r)
        new_rows = []
        for sh, lst in by_shell.items():
            lst.sort(key=lambda x: x["absH"], reverse=True)
            new_rows.extend(lst[:args.topk_per_shell])
        rows = new_rows
        print("[INFO] after topk_per_shell=%d: edges=%d" % (args.topk_per_shell, len(rows)))

    # Write edges
    if args.out_edges:
        outp = Path(args.out_edges)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["pair", "shell", "dist", "absH", "re", "im",
                      "Rx", "Ry", "Rz", "u_wf", "v_wf", "u_atom", "v_atom", "u_elem", "v_elem"]
        with outp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fieldnames})
        print("[INFO] wrote edges -> %s  (rows=%d)" % (str(outp), len(rows)))

    # Optional summary per shell
    if args.out_summary:
        by_shell = defaultdict(list)
        for r in rows:
            by_shell[r["shell"]].append(r)

        summ = []
        for sh in sorted(by_shell.keys()):
            lst = by_shell[sh]
            dmean = float(np.mean([x["dist"] for x in lst]))
            amax = float(np.max([x["absH"] for x in lst]))
            asum = float(np.sum([x["absH"] for x in lst]))
            n = len(lst)
            summ.append({
                "shell": sh,
                "N_edges": n,
                "dist_mean": dmean,
                "absH_max": amax,
                "absH_sum": asum
            })

        outp = Path(args.out_summary)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
            w.writeheader()
            for r in summ:
                w.writerow(r)
        print("[INFO] wrote summary -> %s" % str(outp))


if __name__ == "__main__":
    main()
