
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# helpers: parse hr.dat
# =========================
def parse_hr_dat(hr_path: Path):
    lines = hr_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        raise RuntimeError("hr.dat too short")
    num_wann = int(lines[1].strip())
    nrpts = int(lines[2].strip())

    degen = []
    idx = 3
    while len(degen) < nrpts:
        degen.extend([int(t) for t in lines[idx].split()])
        idx += 1
    rec_lines = lines[idx:]
    return num_wann, rec_lines


def build_H0(num_wann, rec_lines):
    H0 = np.zeros((num_wann, num_wann), dtype=np.complex128)
    for ln in rec_lines:
        if not ln.strip():
            continue
        toks = ln.split()
        if len(toks) < 7:
            continue
        Rx, Ry, Rz = int(toks[0]), int(toks[1]), int(toks[2])
        if (Rx, Ry, Rz) != (0, 0, 0):
            continue
        m = int(toks[3]) - 1
        n = int(toks[4]) - 1
        re0 = float(toks[5])
        im0 = float(toks[6])
        H0[m, n] = re0 + 1j * im0
    return 0.5 * (H0 + H0.conj().T)


# =========================
# helpers: parse edges.csv -> WF -> (atom_id, elem)
# =========================
def read_edges_wf_map(edges_path: Path):
    df = pd.read_csv(edges_path)
    needed = ["m", "n", "atom_m", "atom_n", "elem_m", "elem_n"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"edges.csv missing required column: {c}")

    wf_to_pairs = {}
    for _, r in df.iterrows():
        m = int(r["m"]); n = int(r["n"])
        am = int(r["atom_m"]); an = int(r["atom_n"])
        em = str(r["elem_m"]); en = str(r["elem_n"])
        wf_to_pairs.setdefault(m, set()).add((am, em))
        wf_to_pairs.setdefault(n, set()).add((an, en))

    bad = []
    wf_map = {}
    for wf, s in wf_to_pairs.items():
        if len(s) != 1:
            bad.append((wf, sorted(list(s))))
        else:
            wf_map[wf] = next(iter(s))

    if bad:
        msg = ["[ERROR] Some WFs map to multiple (atom_id, elem) labels. Fix edges.csv labeling first."]
        for wf, pairs in bad[:50]:
            msg.append(f"  wf={wf}: {pairs}")
        raise RuntimeError("\n".join(msg))

    return wf_map


def atoms_by_element(wf_map, num_wann):
    elem_to_atoms = {}
    for wf in range(1, num_wann + 1):
        aid, elem = wf_map[wf]
        elem_to_atoms.setdefault(elem, set()).add(aid)
    return {e: sorted(list(s)) for e, s in elem_to_atoms.items()}


# =========================
# helpers: parse win projections (explicit order)
# =========================
def norm_orb(tok: str) -> str:
    t = tok.strip().lower()
    t = t.replace("dx2y2", "dx2-y2").replace("dx2_y2", "dx2-y2")
    t = t.replace("d(z2)", "dz2").replace("d(z^2)", "dz2").replace("d(z**2)", "dz2")
    t = t.replace("d(x2-y2)", "dx2-y2")
    return t


def expand_token(tok: str):
    t = norm_orb(tok)
    if t == "d":
        return ["dxy", "dyz", "dxz", "dz2", "dx2-y2"]
    if t == "p":
        return ["px", "py", "pz"]
    if t == "s":
        return ["s"]
    return [t]


def parse_win_projections(win_path: Path):
    lines = win_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_proj = False
    proj_lines = []
    for ln in lines:
        s = ln.strip()
        low = s.lower()
        if low.startswith("begin projections"):
            in_proj = True
            continue
        if low.startswith("end projections"):
            in_proj = False
            continue
        if not in_proj:
            continue
        if (not s) or s.startswith("#") or s.startswith("!"):
            continue
        proj_lines.append(s)

    out = []
    for raw in proj_lines:
        if ":" not in raw:
            continue
        elem, rhs = raw.split(":", 1)
        elem = elem.strip()
        rhs = rhs.replace(",", " ").replace(";", " ")
        toks = [t for t in rhs.split() if t.strip()]
        orbs = []
        for t in toks:
            orbs += expand_token(t)
        out.append((elem, orbs))

    if not out:
        raise RuntimeError("[ERROR] No projections found in win. Check begin/end projections block.")
    return out


def build_wf_labels_from_win(proj_spec, elem_to_atoms, num_wann):
    """
    Construct expected WF label sequence in the SAME ordering as Wannier90 projections expansion.
    We assume SOC spinor duplication if num_wann == 2 * spatial_count.
    Return: list labels (1..num_wann): (atom_id, elem, orb)
    """
    spatial = []
    for elem, orbs in proj_spec:
        if elem not in elem_to_atoms:
            raise RuntimeError(f"[ERROR] win projections element '{elem}' not in edges elements {list(elem_to_atoms.keys())}")
        for aid in elem_to_atoms[elem]:
            for orb in orbs:
                spatial.append((aid, elem, orb))

    if len(spatial) == num_wann:
        return spatial, False
    if 2 * len(spatial) == num_wann:
        spinor = []
        for item in spatial:
            spinor.append(item)
            spinor.append(item)
        return spinor, True

    raise RuntimeError(
        "[ERROR] projection expansion count mismatch.\n"
        f"  spatial_count={len(spatial)}, num_wann={num_wann}\n"
        "Expected num_wann == spatial_count (no spinor) OR num_wann == 2*spatial_count (SOC spinor).\n"
    )


# =========================
# local diag + orbital labeling from basis
# =========================
def diag_local(H0, wf_list_1based):
    idx = [w - 1 for w in wf_list_1based]
    sub = H0[np.ix_(idx, idx)]
    evals, evecs = np.linalg.eigh(sub)
    order = np.argsort(np.real(evals))
    return np.real(evals[order]), evecs[:, order]


def try_pairs(evals_rel):
    if len(evals_rel) % 2 != 0:
        return []
    out = []
    for i in range(0, len(evals_rel), 2):
        e1 = float(evals_rel[i]); e2 = float(evals_rel[i + 1])
        out.append((0.5 * (e1 + e2), abs(e2 - e1), i, i + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", default="wannier90_hr.dat")
    ap.add_argument("--edges", default="edges.csv")
    ap.add_argument("--win", default="wannier90.win")
    ap.add_argument("--Ef", type=float, default=0.0, help="output energies as E_rel = E - Ef (eV)")
    ap.add_argument("--out_prefix", default="cfFINAL")
    ap.add_argument("--pair_tol", type=float, default=0.02, help="warn if pair split > tol (eV)")
    args = ap.parse_args()

    hr_path = Path(args.hr)
    edges_path = Path(args.edges)
    win_path = Path(args.win)
    for p in [hr_path, edges_path, win_path]:
        if not p.exists():
            raise FileNotFoundError(p.resolve())

    num_wann, rec_lines = parse_hr_dat(hr_path)
    H0 = build_H0(num_wann, rec_lines)

    wf_map = read_edges_wf_map(edges_path)
    elem_to_atoms = atoms_by_element(wf_map, num_wann)

    proj_spec = parse_win_projections(win_path)
    expected_labels, spinor_dup = build_wf_labels_from_win(proj_spec, elem_to_atoms, num_wann)

    # Build WF groups per atom from edges (ground truth for which WF belongs to which atom)
    atom_to_wfs = {}
    atom_to_elem = {}
    for wf in range(1, num_wann + 1):
        aid, elem = wf_map[wf]
        atom_to_wfs.setdefault(aid, []).append(wf)
        atom_to_elem[aid] = elem
    for aid in atom_to_wfs:
        atom_to_wfs[aid] = sorted(atom_to_wfs[aid])

    # Now: assign each WF an orbital label by matching its position in the expected sequence.
    # This requires that the WF ordering produced by Wannier90 follows the projection expansion ordering.
    # We'll build wf_orb[wf] = orb_label.
    wf_orb = {}
    wf_expected_atom = {}
    for wf in range(1, num_wann + 1):
        aid_e, elem_e, orb_e = expected_labels[wf - 1]
        wf_expected_atom[wf] = (aid_e, elem_e)
        wf_orb[wf] = orb_e

    # Consistency check: WF->atom from edges should match expected atom from win ordering
    mism = []
    for wf in range(1, num_wann + 1):
        aid_true, elem_true = wf_map[wf]
        aid_e, elem_e = wf_expected_atom[wf]
        if (aid_true != aid_e) or (elem_true != elem_e):
            mism.append((wf, (aid_true, elem_true), (aid_e, elem_e)))
    if mism:
        # We do not abort; but we warn loudly because orbital labeling would be unreliable.
        print("[WARN] WF ordering does NOT match win projection expansion ordering for these WFs (first 20 shown):")
        for it in mism[:20]:
            print(f"  wf={it[0]} edges={it[1]} expected_from_win={it[2]}")
        print("[WARN] In this case, you must reorder WFs by parsing wout spread table or use a stricter mapping method.")
        # If this happens, stop now to avoid producing wrong orbital order.
        raise RuntimeError("WF ordering mismatch: cannot safely label orbitals from win ordering.")

    # Outputs
    out_levels = f"{args.out_prefix}_levels.csv"
    out_order = f"{args.out_prefix}_orbital_order.csv"
    out_pairs = f"{args.out_prefix}_pair_centers.csv"

    rows_levels = []
    rows_pairs = []
    lowest = {}  # (aid, elem, orb) -> lowest E_rel

    for aid in sorted(atom_to_wfs.keys()):
        elem = atom_to_elem[aid]
        wfs = atom_to_wfs[aid]
        evals, evecs = diag_local(H0, wfs)
        evals_rel = evals - args.Ef

        # basis orbital label list for this atom (same length as wfs)
        basis_orbs = [wf_orb[wf] for wf in wfs]

        for j in range(len(evals_rel)):
            c2 = np.abs(evecs[:, j]) ** 2
            # orbital weight = sum |c_i|^2 over basis functions with same orb label
            orb_w = {}
            for i, orb in enumerate(basis_orbs):
                orb_w[orb] = orb_w.get(orb, 0.0) + float(c2[i])
            dom_orb = max(orb_w.keys(), key=lambda k: orb_w[k])
            dom_w = orb_w[dom_orb]

            rows_levels.append({
                "atom_id": aid,
                "elem": elem,
                "n_wf": len(wfs),
                "level_index": j + 1,
                "E_rel": float(evals_rel[j]),
                "dominant_orb": dom_orb,
                "dominant_w": float(dom_w)
            })

            key = (aid, elem, dom_orb)
            lowest[key] = min(lowest.get(key, 1e30), float(evals_rel[j]))

        # pair diagnostics
        pairs = try_pairs(evals_rel)
        for pi, (center, split, i1, i2) in enumerate(pairs, start=1):
            warn = int(split > args.pair_tol)
            # label by combined weights
            c2 = (np.abs(evecs[:, i1]) ** 2 + np.abs(evecs[:, i2]) ** 2)
            orb_w = {}
            for i, orb in enumerate(basis_orbs):
                orb_w[orb] = orb_w.get(orb, 0.0) + float(c2[i])
            dom_orb = max(orb_w.keys(), key=lambda k: orb_w[k])
            dom_w = orb_w[dom_orb]

            rows_pairs.append({
                "atom_id": aid,
                "elem": elem,
                "n_wf": len(wfs),
                "pair_index": pi,
                "center_rel": float(center),
                "split": float(split),
                "dominant_orb_pair": dom_orb,
                "dominant_w_pair": float(dom_w),
                "warn_split_gt_tol": warn
            })

    pd.DataFrame(rows_levels).to_csv(out_levels, index=False)
    pd.DataFrame(rows_pairs).to_csv(out_pairs, index=False)

    out_rows = []
    for (aid, elem, orb), E0 in sorted(lowest.items(), key=lambda x: (x[0][1], x[0][0], x[1], x[0][2])):
        out_rows.append({
            "atom_id": aid, "elem": elem, "orbital": orb,
            "lowest_E_rel": float(E0)
        })
    pd.DataFrame(out_rows).to_csv(out_order, index=False)

    print("[OK] Done.")
    print(f"  spinor_dup = {spinor_dup}")
    print(f"  wrote: {out_levels}")
    print(f"  wrote: {out_pairs}")
    print(f"  wrote: {out_order}")
    print("Notes:")
    print("  - This route labels orbitals from win projection expansion order; it is stable and does NOT depend on chk/amn.")
    print("  - If you are FM+SOC (TR broken), pair_centers are diagnostic only; do not treat them as Kramers by default.")


if __name__ == "__main__":
    main()
