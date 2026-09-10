
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_hr_dat(hr_path: Path):
    lines = hr_path.read_text(encoding="utf-8", errors="ignore").splitlines()
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
        H0[m, n] = float(toks[5]) + 1j * float(toks[6])
    return 0.5 * (H0 + H0.conj().T)


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
        msg = ["[ERROR] WF->(atom,elem) inconsistent in edges.csv"]
        for wf, pairs in bad[:30]:
            msg.append(f"  wf={wf}: {pairs}")
        raise RuntimeError("\n".join(msg))
    return wf_map


def atoms_by_element(wf_map, num_wann):
    elem_to_atoms = {}
    for wf in range(1, num_wann + 1):
        aid, elem = wf_map[wf]
        elem_to_atoms.setdefault(elem, set()).add(aid)
    return {e: sorted(list(s)) for e, s in elem_to_atoms.items()}


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
        raise RuntimeError("[ERROR] No projections in win.")
    return out


def build_expected_labels(proj_spec, elem_to_atoms, num_wann):
    spatial = []
    for elem, orbs in proj_spec:
        if elem not in elem_to_atoms:
            raise RuntimeError(f"[ERROR] elem {elem} in win not found in edges-derived elements {list(elem_to_atoms.keys())}")
        for aid in elem_to_atoms[elem]:
            for orb in orbs:
                spatial.append((aid, elem, orb))
    if len(spatial) == num_wann:
        return spatial, False
    if 2 * len(spatial) == num_wann:
        spinor = []
        for x in spatial:
            spinor.append(x); spinor.append(x)
        return spinor, True
    raise RuntimeError(f"[ERROR] projection count mismatch: spatial={len(spatial)} num_wann={num_wann}")


def diag_local(H0, wf_list_1based):
    idx = [w - 1 for w in wf_list_1based]
    sub = H0[np.ix_(idx, idx)]
    evals, evecs = np.linalg.eigh(sub)
    order = np.argsort(np.real(evals))
    return np.real(evals[order]), evecs[:, order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", default="wannier90_hr.dat")
    ap.add_argument("--edges", default="edges.csv")
    ap.add_argument("--win", default="wannier90.win")
    ap.add_argument("--Ef", type=float, default=0.0)
    ap.add_argument("--out_prefix", default="cfFW")
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
    expected_labels, spinor_dup = build_expected_labels(proj_spec, elem_to_atoms, num_wann)

    # verify WF ordering matches expected atom labels
    mism = []
    for wf in range(1, num_wann + 1):
        aid_true, elem_true = wf_map[wf]
        aid_e, elem_e, _ = expected_labels[wf - 1]
        if (aid_true != aid_e) or (elem_true != elem_e):
            mism.append((wf, (aid_true, elem_true), (aid_e, elem_e)))
    if mism:
        print("[WARN] WF ordering mismatch with win expansion ordering; cannot label safely.")
        for it in mism[:20]:
            print("  wf", it[0], "edges", it[1], "expected", it[2])
        raise RuntimeError("WF ordering mismatch; stop.")

    # WF -> orbital label (basis orbital)
    wf_orb = {wf: expected_labels[wf - 1][2] for wf in range(1, num_wann + 1)}

    # group WFs by atom
    atom_to_wfs = {}
    atom_to_elem = {}
    for wf in range(1, num_wann + 1):
        aid, elem = wf_map[wf]
        atom_to_wfs.setdefault(aid, []).append(wf)
        atom_to_elem[aid] = elem
    for aid in atom_to_wfs:
        atom_to_wfs[aid] = sorted(atom_to_wfs[aid])

    # outputs
    out_levels = f"{args.out_prefix}_levels.csv"
    out_full = f"{args.out_prefix}_levels_fullweights.csv"
    out_order = f"{args.out_prefix}_orbital_order.csv"

    rows_levels = []
    rows_full = []
    lowest = {}

    for aid in sorted(atom_to_wfs.keys()):
        elem = atom_to_elem[aid]
        wfs = atom_to_wfs[aid]
        evals, evecs = diag_local(H0, wfs)
        evals_rel = evals - args.Ef

        basis_orbs = [wf_orb[wf] for wf in wfs]
        unique_orbs = sorted(list(set(basis_orbs)))

        for j in range(len(evals_rel)):
            c2 = np.abs(evecs[:, j])**2
            orb_w = {o: 0.0 for o in unique_orbs}
            for i, o in enumerate(basis_orbs):
                orb_w[o] += float(c2[i])

            dom_orb = max(orb_w.keys(), key=lambda k: orb_w[k])
            dom_w = orb_w[dom_orb]

            rows_levels.append({
                "atom_id": aid, "elem": elem, "n_wf": len(wfs),
                "level_index": j+1, "E_rel": float(evals_rel[j]),
                "dominant_orb": dom_orb, "dominant_w": float(dom_w)
            })

            r = {"atom_id": aid, "elem": elem, "n_wf": len(wfs),
                 "level_index": j+1, "E_rel": float(evals_rel[j])}
            # add all orbital weights as columns
            for o in unique_orbs:
                r[f"w_{o}"] = float(orb_w[o])
            rows_full.append(r)

            key = (aid, elem, dom_orb)
            lowest[key] = min(lowest.get(key, 1e30), float(evals_rel[j]))

    pd.DataFrame(rows_levels).to_csv(out_levels, index=False)
    pd.DataFrame(rows_full).to_csv(out_full, index=False)

    out_rows = []
    for (aid, elem, orb), E0 in sorted(lowest.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        out_rows.append({"atom_id": aid, "elem": elem, "orbital": orb, "lowest_E_rel": float(E0)})
    pd.DataFrame(out_rows).to_csv(out_order, index=False)

    print("[OK] wrote:")
    print(" ", out_levels)
    print(" ", out_full)
    print(" ", out_order)
    print("[INFO] spinor_dup =", spinor_dup)


if __name__ == "__main__":
    main()
