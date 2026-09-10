
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
