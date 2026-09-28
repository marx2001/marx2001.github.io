import csv
import argparse
from dataclasses import dataclass
from collections import defaultdict
from typing import Tuple, List, Dict, Optional


@dataclass(frozen=True)
class DirEdge:
    # directed edge: u(cell_shift) -> v(cell_shift + R)
    u_atom: int
    v_atom: int
    u_elem: str
    v_elem: str
    u_wf: int   # 1-based wf index
    v_wf: int   # 1-based wf index
    Rx: int
    Ry: int
    Rz: int
    absH: float
    dist: float
    pair: str = ""   # optional: "Se_p-Ge_d", "Se_p-Ir_d", ...
    shell: int = 0   # optional


def _pick(row: dict, *keys: str, default=None):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def parse_edges_csv(path: str, add_reverse: bool = False) -> List[DirEdge]:
    """
    Read edges csv produced by:
      - old 1step: dist_A, absH_eV, Rx,Ry,Rz, m,n, atom_m,atom_n, elem_m,elem_n
      - new 1step: dist, absH, Rx,Ry,Rz, u_wf,v_wf, u_atom,v_atom, u_elem,v_elem, pair(optional), shell(optional)

    For new 1step outputs, edges are already directed and include both directions in hr.dat,
    so add_reverse should usually be False to avoid doubling.
    """
    edges: List[DirEdge] = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        fns = r.fieldnames or []
        fset = set(fns)

        is_new = ("u_wf" in fset and "v_wf" in fset and "u_atom" in fset and "v_atom" in fset)
        is_old = ("m" in fset and "n" in fset and "atom_m" in fset and "atom_n" in fset)

        if not (is_new or is_old):
            raise RuntimeError(f"Unrecognized edges schema in {path}. Found columns: {fns}")

        for row in r:
            try:
                dist = float(_pick(row, "dist", "dist_A"))
                absH = float(_pick(row, "absH", "absH_eV"))
                Rx, Ry, Rz = int(_pick(row, "Rx")), int(_pick(row, "Ry")), int(_pick(row, "Rz"))

                u_wf = int(_pick(row, "u_wf", "m"))
                v_wf = int(_pick(row, "v_wf", "n"))

                u_atom = int(_pick(row, "u_atom", "atom_m"))
                v_atom = int(_pick(row, "v_atom", "atom_n"))

                u_elem = str(_pick(row, "u_elem", "elem_m")).strip()
                v_elem = str(_pick(row, "v_elem", "elem_n")).strip()

                pair = str(_pick(row, "pair", default="") or "")
                shell = int(_pick(row, "shell", default="0") or 0)
            except Exception:
                continue

            e = DirEdge(u_atom, v_atom, u_elem, v_elem, u_wf, v_wf, Rx, Ry, Rz, absH, dist, pair=pair, shell=shell)
            edges.append(e)

            if add_reverse:
                edges.append(DirEdge(
                    v_atom, u_atom, v_elem, u_elem,
                    v_wf, u_wf, -Rx, -Ry, -Rz,
                    absH, dist, pair=pair, shell=shell
                ))

    return edges


def load_onsite_from_hr(hr_path: str, num_wann: Optional[int] = None) -> Dict[int, float]:
    """Extract onsite energies eps[m] = Re(H_mm(R=0)) from wannier90_hr.dat."""
    with open(hr_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if len(lines) < 4:
        raise RuntimeError(f"hr.dat too short: {hr_path}")

    try:
        nw = int(lines[1].split()[0])
    except Exception:
        raise RuntimeError("Failed to parse num_wann from hr.dat line2.")
    if num_wann is not None and nw != num_wann:
        print(f"[WARN] num_wann mismatch: hr={nw} vs --num_wann={num_wann}. Using hr value.")
    num_wann = nw

    try:
        nrpts = int(lines[2].split()[0])
    except Exception:
        raise RuntimeError("Failed to parse nrpts from hr.dat line3.")

    deg = []
    idx = 3
    while idx < len(lines) and len(deg) < nrpts:
        for p in lines[idx].split():
            try:
                deg.append(int(p))
            except Exception:
                pass
        idx += 1
    if len(deg) < nrpts:
        raise RuntimeError("Failed to read full degeneracy list from hr.dat.")

    eps: Dict[int, float] = {}
    for j in range(idx, len(lines)):
        parts = lines[j].split()
        if len(parts) < 7:
            continue
        try:
            Rx, Ry, Rz = int(parts[0]), int(parts[1]), int(parts[2])
            m, n = int(parts[3]), int(parts[4])
            re = float(parts[5])
        except Exception:
            continue
        if Rx == 0 and Ry == 0 and Rz == 0 and m == n:
            eps[m] = re

    if len(eps) < num_wann:
        print(f"[WARN] onsite found {len(eps)}/{num_wann}. Some WFs missing onsite.")
    else:
        print(f"[INFO] onsite found {len(eps)}/{num_wann}.")
    return eps


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


def delta_pair(eps: Dict[int, float], wf_a: int, wf_b: int, floor: float) -> float:
    ea = eps.get(wf_a, None)
    eb = eps.get(wf_b, None)
    if ea is None or eb is None:
        return float("nan")
    d = abs(ea - eb)
    return max(d, floor)


def print_panel_summary(tag: str, uniq_sorted: List[tuple], panel_topk: int):
    """
    uniq_sorted elements: (score, N, D, d1, d2, d3, sig, ...)
    """
    if not uniq_sorted:
        print(f"[PANEL] {tag}: no paths")
        return

    k = min(panel_topk, len(uniq_sorted))
    top1 = uniq_sorted[0][0]
    sumk = sum(x[0] for x in uniq_sorted[:k])

    # also show N and D behavior (helps diagnose delta blow-up)
    N1, D1 = uniq_sorted[0][1], uniq_sorted[0][2]
    Nk = [x[1] for x in uniq_sorted[:k]]
    Dk = [x[2] for x in uniq_sorted[:k]]

    def _safe_mean(arr):
        return sum(arr) / max(1, len(arr))

    print(f"[PANEL] {tag}")
    print(f"  unique_paths : {len(uniq_sorted)}")
    print(f"  top1 score   : {top1:.6g}    (N={N1:.6g}, D={D1:.6g})")
    print(f"  sum top{k}   : {sumk:.6g}")
    print(f"  top{k} <N>   : {_safe_mean(Nk):.6g}")
    print(f"  top{k} <D>   : {_safe_mean(Dk):.6g}")


def main():
    ap = argparse.ArgumentParser(
        description="Rank Tc–Se–X–Se–Tc paths by (product |t|) / (product Δ) using onsite energies from hr.dat."
    )
    ap.add_argument("--edges", required=True, help="Input edges csv (from 1step.py)")
    ap.add_argument("--hr", required=True, help="wannier90_hr.dat path for onsite extraction")
    ap.add_argument("--out", default="top_paths_ratio.csv", help="Output ranked paths csv")

    ap.add_argument("--mediators", default="Ir,Ge", help="Comma-separated mediators, e.g. Ir,Ge or Ir only")
    ap.add_argument("--d_tcse", type=float, default=3.0, help="Max distance for Tc–Se edges (Ang)")
    ap.add_argument("--d_sex", type=float, default=3.0, help="Max distance for Se–X edges (Ang)")
    ap.add_argument("--min_absH", type=float, default=1e-6, help="Min |t| (eV) used for edges in path building")

    # channel filter using 'pair' column
    ap.add_argument("--X_orb", choices=["auto", "Ir_d", "Ge_d", "Ge_p", "Ge_all"],
                    default="auto",
                    help=("Filter mediator orbital channel using edge 'pair' labels from 1step.py. "
                          "auto: Ir->Ir_d, Ge->Ge_all."))

    ap.add_argument("--top_per_se_tc", type=int, default=60)
    ap.add_argument("--top_per_se_x", type=int, default=120)
    ap.add_argument("--top_per_x", type=int, default=120)
    ap.add_argument("--top_paths", type=int, default=2000, help="How many top paths to output after ranking")

    ap.add_argument("--require_same_tc_atom", action="store_true")
    ap.add_argument("--max_netR_L1", type=int, default=6, help="Filter by |dRx|+|dRy|+|dRz| <= this (999 disables)")
    ap.add_argument("--exclude_netR0", action="store_true", help="Drop net_dR=(0,0,0) loops")

    ap.add_argument("--delta_mode", choices=["sequential", "pairwise"], default="sequential")
    ap.add_argument("--delta_floor", type=float, default=1e-3)

    ap.add_argument("--add_reverse_edges", action="store_true",
                    help="Also add reversed edges. Usually OFF for new 1step edges (already directed).")

    # NEW: panel printing
    ap.add_argument("--panel_topk", type=int, default=100,
                    help="In-shell panel: report sum of topK scores (default 100).")

    args = ap.parse_args()

    mediators = [x.strip() for x in args.mediators.split(",") if x.strip()]
    if not mediators:
        raise RuntimeError("No mediators specified.")

    eps = load_onsite_from_hr(args.hr)
    all_dir = parse_edges_csv(args.edges, add_reverse=args.add_reverse_edges)

    tc_atoms = sorted({e.u_atom for e in all_dir if e.u_elem == "Tc"} | {e.v_atom for e in all_dir if e.v_elem == "Tc"})
    se_atoms = sorted({e.u_atom for e in all_dir if e.u_elem == "Se"} | {e.v_atom for e in all_dir if e.v_elem == "Se"})
    if not tc_atoms:
        raise RuntimeError("No Tc atoms found.")
    if not se_atoms:
        raise RuntimeError("No Se atoms found.")

    print(f"[INFO] Tc atoms: {tc_atoms}")
    print(f"[INFO] Se atoms: {se_atoms}")
    for M in mediators:
        med_atoms = sorted({e.u_atom for e in all_dir if e.u_elem == M} | {e.v_atom for e in all_dir if e.v_elem == M})
        print(f"[INFO] {M} atoms: {med_atoms}")

    def edge_pass_channel(e: DirEdge, M: str) -> bool:
        if not e.pair:
            return True
        p = e.pair
        if M == "Ir":
            # Ir is d in your setting
            return ("Ir_d" in p)
        if M == "Ge":
            if args.X_orb == "Ge_d":
                return ("Ge_d" in p)
            if args.X_orb == "Ge_p":
                return ("Ge_p" in p)
            if args.X_orb in ("Ge_all", "auto"):
                return ("Ge_d" in p) or ("Ge_p" in p) or ("Ge" in p)
            return True
        return True

    tc_to_se = defaultdict(list)
    se_to_tc = defaultdict(list)
    se_to_x = {M: defaultdict(list) for M in mediators}
    x_to_se = {M: defaultdict(list) for M in mediators}

    for e in all_dir:
        if e.absH < args.min_absH:
            continue

        if e.u_elem == "Tc" and e.v_elem == "Se" and e.dist <= args.d_tcse:
            tc_to_se[e.u_atom].append(e)
            continue
        if e.u_elem == "Se" and e.v_elem == "Tc" and e.dist <= args.d_tcse:
            se_to_tc[e.u_atom].append(e)
            continue

        if e.dist <= args.d_sex:
            for M in mediators:
                if e.u_elem == "Se" and e.v_elem == M and edge_pass_channel(e, M):
                    se_to_x[M][e.u_atom].append(e)
                elif e.u_elem == M and e.v_elem == "Se" and edge_pass_channel(e, M):
                    x_to_se[M][e.u_atom].append(e)

    keep_top_per_key(tc_to_se, topn=max(args.top_per_se_tc, 30))
    keep_top_per_key(se_to_tc, topn=max(args.top_per_se_tc, 30))
    for M in mediators:
        keep_top_per_key(se_to_x[M], topn=max(args.top_per_se_x, 50))
        keep_top_per_key(x_to_se[M], topn=max(args.top_per_x, 50))

    def compute_denoms(tc_wf, se1_wf, x_wf, se2_wf):
        floor = args.delta_floor
        if args.delta_mode == "sequential":
            d1 = delta_pair(eps, se1_wf, tc_wf, floor)
            d2 = delta_pair(eps, x_wf,  se1_wf, floor)
            d3 = delta_pair(eps, se2_wf, x_wf,  floor)
        else:
            d1 = delta_pair(eps, se1_wf, tc_wf, floor)
            d2 = delta_pair(eps, x_wf,   tc_wf, floor)
            d3 = delta_pair(eps, se2_wf, tc_wf, floor)

        if any([d != d for d in (d1, d2, d3)]):
            return float("nan"), d1, d2, d3
        return d1 * d2 * d3, d1, d2, d3

    # ---------- enumerate ----------
    paths_by_M = {M: [] for M in mediators}

    for tc0 in tc_atoms:
        shift0 = (0, 0, 0)

        for e1 in tc_to_se.get(tc0, []):
            se1 = e1.v_atom
            shift_se1 = add_shift(shift0, (e1.Rx, e1.Ry, e1.Rz))

            for M in mediators:
                for e2 in se_to_x[M].get(se1, []):
                    x = e2.v_atom
                    shift_x = add_shift(shift_se1, (e2.Rx, e2.Ry, e2.Rz))

                    for e3 in x_to_se[M].get(x, []):
                        se2 = e3.v_atom
                        shift_se2 = add_shift(shift_x, (e3.Rx, e3.Ry, e3.Rz))

                        for e4 in se_to_tc.get(se2, []):
                            tc1 = e4.v_atom
                            shift_tc1 = add_shift(shift_se2, (e4.Rx, e4.Ry, e4.Rz))

                            if args.require_same_tc_atom and (tc1 != tc0):
                                continue

                            netR = shift_tc1
                            if args.exclude_netR0 and netR == (0, 0, 0):
                                continue
                            if args.max_netR_L1 < 999 and abs_R_L1(netR) > args.max_netR_L1:
                                continue

                            N = e1.absH * e2.absH * e3.absH * e4.absH
                            D, d1, d2, d3 = compute_denoms(e1.u_wf, e1.v_wf, e2.v_wf, e3.v_wf)
                            if D != D:
                                continue
                            score = N / D

                            sig = (
                                M,
                                tc0, se1, x, se2, tc1,
                                netR[0], netR[1], netR[2],
                                (e1.u_wf, e1.v_wf, e1.Rx, e1.Ry, e1.Rz),
                                (e2.u_wf, e2.v_wf, e2.Rx, e2.Ry, e2.Rz),
                                (e3.u_wf, e3.v_wf, e3.Rx, e3.Ry, e3.Rz),
                                (e4.u_wf, e4.v_wf, e4.Rx, e4.Ry, e4.Rz),
                            )

                            paths_by_M[M].append((score, N, D, d1, d2, d3, sig,
                                                  M, tc0, se1, x, se2, tc1, netR, e1, e2, e3, e4))

    # ---------- de-dup within each mediator ----------
    uniq_by_M = {}
    for M, paths in paths_by_M.items():
        if not paths:
            uniq_by_M[M] = []
            continue
        best = {}
        for item in paths:
            sc, sig = item[0], item[6]
            if sig not in best or sc > best[sig][0]:
                best[sig] = item
        uniq = list(best.values())
        uniq.sort(key=lambda x: x[0], reverse=True)
        uniq_by_M[M] = uniq

    # ---------- panel print ----------
    print("\n===== SHELL PANEL SUMMARY =====")
    for M in mediators:
        tag = f"{M} (X_orb={args.X_orb})"
        print_panel_summary(tag, uniq_by_M[M], panel_topk=args.panel_topk)

    # ---------- merge for output CSV ----------
    merged = []
    for M in mediators:
        merged.extend(uniq_by_M[M])

    if not merged:
        raise RuntimeError("No paths found after applying constraints. Try loosening windows or lowering --min_absH.")

    merged.sort(key=lambda x: x[0], reverse=True)
    merged = merged[:args.top_paths]

    # ---------- write output ----------
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "mediator", "X_orb",
            "score_num_over_denom", "N_prod_absH", "D_prod_delta",
            "delta1", "delta2", "delta3",
            "Tc0_atom", "Se1_atom", "X_atom", "Se2_atom", "Tc1_atom",
            "net_dRx", "net_dRy", "net_dRz",
            "t1_absH", "d1_A", "wf1_u", "wf1_v", "R1x", "R1y", "R1z",
            "t2_absH", "d2_A", "wf2_u", "wf2_v", "R2x", "R2y", "R2z",
            "t3_absH", "d3_A", "wf3_u", "wf3_v", "R3x", "R3y", "R3z",
            "t4_absH", "d4_A", "wf4_u", "wf4_v", "R4x", "R4y", "R4z",
        ])

        for score, N, D, d1, d2, d3, sig, M, tc0, se1, x, se2, tc1, netR, e1, e2, e3, e4 in merged:
            w.writerow([
                M, args.X_orb,
                f"{score:.10g}", f"{N:.10g}", f"{D:.10g}",
                f"{d1:.10g}", f"{d2:.10g}", f"{d3:.10g}",
                tc0, se1, x, se2, tc1,
                netR[0], netR[1], netR[2],
                f"{e1.absH:.10g}", f"{e1.dist:.6f}", e1.u_wf, e1.v_wf, e1.Rx, e1.Ry, e1.Rz,
                f"{e2.absH:.10g}", f"{e2.dist:.6f}", e2.u_wf, e2.v_wf, e2.Rx, e2.Ry, e2.Rz,
                f"{e3.absH:.10g}", f"{e3.dist:.6f}", e3.u_wf, e3.v_wf, e3.Rx, e3.Ry, e3.Rz,
                f"{e4.absH:.10g}", f"{e4.dist:.6f}", e4.u_wf, e4.v_wf, e4.Rx, e4.Ry, e4.Rz,
            ])

    print(f"\n[DONE] wrote ranked paths to: {args.out}")
    print(f"[INFO] delta_mode={args.delta_mode}, delta_floor={args.delta_floor} eV")


if __name__ == "__main__":
    main()
