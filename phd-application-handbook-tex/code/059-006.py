
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
