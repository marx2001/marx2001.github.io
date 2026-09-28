
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
    u_group: str
    v_group: str
    u_wf: int   # 1-based wf index
    v_wf: int   # 1-based wf index
    Rx: int
    Ry: int
    Rz: int
    absH: float
    dist: float


def parse_edges_csv(path: str, prefer_group: bool = True) -> List[DirEdge]:
    """
    Read out_edges csv produced by NEW 1step (recommended) or old format.
    Required columns (minimum):
      dist_A, absH_eV, Rx,Ry,Rz, m,n, atom_m,atom_n, elem_m,elem_n
    Optional (NEW, preferred):
      group_m, group_n

    If group_m/group_n exists and prefer_group=True, use them; otherwise fallback to elem.
    """
    edges: List[DirEdge] = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        required = ["dist_A", "absH_eV", "Rx", "Ry", "Rz",
                    "m", "n", "atom_m", "atom_n", "elem_m", "elem_n"]
        for k in required:
            if k not in r.fieldnames:
                raise RuntimeError(f"Missing column '{k}' in {path}. Found: {r.fieldnames}")

        has_group = ("group_m" in r.fieldnames) and ("group_n" in r.fieldnames)
        if prefer_group and (not has_group):
            print("[WARN] group_m/group_n not found. Falling back to elem_m/elem_n filtering.")

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

            if has_group:
                gm = row["group_m"].strip()
                gn = row["group_n"].strip()
            else:
                gm = elem_m
                gn = elem_n

            # Add both directions to make path enumeration easier.
            edges.append(DirEdge(atom_m, atom_n, elem_m, elem_n, gm, gn, m, n, Rx, Ry, Rz, absH, dist))
            edges.append(DirEdge(atom_n, atom_m, elem_n, elem_m, gn, gm, n, m, -Rx, -Ry, -Rz, absH, dist))

    return edges


def load_onsite_from_hr(hr_path: str, num_wann: Optional[int] = None) -> Dict[int, float]:
    """
    Parse wannier90_hr.dat and extract onsite energies eps[m] = Re(H_mm(R=0)).
    Returns dict: wf_index(1-based) -> eps_eV
    """
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
        parts = lines[idx].split()
        for p in parts:
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
        print(f"[WARN] onsite found {len(eps)}/{num_wann}. Some WFs missing onsite (unusual).")
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


def main():
    ap = argparse.ArgumentParser(
        description="Rank Tc–Se–X–Se–Tc paths by (product |t|) / (product Δ) using onsite energies from hr.dat."
    )
    ap.add_argument("--edges", required=True, help="Input edges csv (NEW 1step out_edges is recommended)")
    ap.add_argument("--hr", required=True, help="wannier90_hr.dat path for onsite extraction")
    ap.add_argument("--out", default="top_paths_ratio.csv", help="Output ranked paths csv")

    # NEW: use group labels (Tc_d / Se_p / Ir_d ...) if present
    ap.add_argument("--prefer_group", action="store_true", help="Prefer group_m/group_n if present (recommended).")
    ap.add_argument("--tc_group", default="Tc_d", help="Group label for Tc-centered d-like WFs (default: Tc_d)")
    ap.add_argument("--se_group", default="Se_p", help="Group label for Se-centered p-like WFs (default: Se_p)")
    ap.add_argument("--mediators", default="Ir_d",
                    help="Comma-separated mediator group labels (e.g. Ir_d,Ge_p). If no group cols, use element symbols (Ir,Ge).")

    ap.add_argument("--d_tcse", type=float, default=3.0, help="Max distance for Tc–Se edges (Ang)")
    ap.add_argument("--d_sex", type=float, default=3.0, help="Max distance for Se–X edges (Ang)")
    ap.add_argument("--min_absH", type=float, default=1e-3, help="Min |t| (eV) used for edges in path building")

    ap.add_argument("--top_per_se_tc", type=int, default=60)
    ap.add_argument("--top_per_se_x", type=int, default=120)
    ap.add_argument("--top_per_x", type=int, default=120)
    ap.add_argument("--top_paths", type=int, default=2000, help="How many top paths to output after ranking")

    ap.add_argument("--require_same_tc_atom", action="store_true")
    ap.add_argument("--max_netR_L1", type=int, default=6, help="Filter by |dRx|+|dRy|+|dRz| <= this (999 disables)")
    ap.add_argument("--exclude_netR0", action="store_true", help="Drop net_dR=(0,0,0) loops (recommended for Tc–Tc exchange)")

    ap.add_argument("--delta_mode", choices=["sequential", "pairwise"], default="sequential",
                    help=("How to define Δ product:\n"
                          "  sequential: Δ1=|ε(Se1)-ε(Tc)|, Δ2=|ε(X)-ε(Se1)|, Δ3=|ε(Se2)-ε(X)|\n"
                          "  pairwise:   Δ1=|ε(Se1)-ε(Tc)|, Δ2=|ε(X)-ε(Tc)|,  Δ3=|ε(Se2)-ε(Tc)|\n"
                          "Both are proxies; sequential is closer to step-by-step virtual hopping."))
    ap.add_argument("--delta_floor", type=float, default=1e-3,
                    help="Lower bound for each Δ (eV) to avoid division blow-up when ε nearly equal")

    args = ap.parse_args()

    mediators = [x.strip() for x in args.mediators.split(",") if x.strip()]
    if not mediators:
        raise RuntimeError("No mediators specified.")

    eps = load_onsite_from_hr(args.hr)

    # IMPORTANT: prefer_group=True only makes sense if group_m/group_n exist.
    all_dir = parse_edges_csv(args.edges, prefer_group=args.prefer_group)

    # Decide whether we are filtering by group or elem:
    # If group columns existed, DirEdge.u_group != DirEdge.u_elem in general; else group==elem.
    use_group = True if args.prefer_group else False

    # Helper accessors
    def Utag(e: DirEdge) -> str:
        return e.u_group if use_group else e.u_elem

    def Vtag(e: DirEdge) -> str:
        return e.v_group if use_group else e.v_elem

    tc_tag = args.tc_group if use_group else "Tc"
    se_tag = args.se_group if use_group else "Se"

    # Build node lists
    tc_atoms = sorted({e.u_atom for e in all_dir if Utag(e) == tc_tag} |
                      {e.v_atom for e in all_dir if Vtag(e) == tc_tag})
    se_atoms = sorted({e.u_atom for e in all_dir if Utag(e) == se_tag} |
                      {e.v_atom for e in all_dir if Vtag(e) == se_tag})

    if not tc_atoms:
        raise RuntimeError(f"No Tc nodes found under tag '{tc_tag}'. Check --prefer_group/--tc_group.")
    if not se_atoms:
        raise RuntimeError(f"No Se nodes found under tag '{se_tag}'. Check --prefer_group/--se_group.")

    print(f"[INFO] use_group={use_group}")
    print(f"[INFO] Tc tag: {tc_tag}, Tc atoms: {tc_atoms}")
    print(f"[INFO] Se tag: {se_tag}, Se atoms: {se_atoms}")
    for M in mediators:
        med_atoms = sorted({e.u_atom for e in all_dir if Utag(e) == M} |
                           {e.v_atom for e in all_dir if Vtag(e) == M})
        print(f"[INFO] mediator '{M}' atoms: {med_atoms}")

    # Build filtered edge pools
    tc_to_se = defaultdict(list)  # Tc atom -> edges Tc->Se
    se_to_tc = defaultdict(list)  # Se atom -> edges Se->Tc
    se_to_x = {M: defaultdict(list) for M in mediators}  # Se atom -> edges Se->M
    x_to_se = {M: defaultdict(list) for M in mediators}  # M atom  -> edges M->Se

    for e in all_dir:
        if e.absH < args.min_absH:
            continue

        u = Utag(e)
        v = Vtag(e)

        if u == tc_tag and v == se_tag and e.dist <= args.d_tcse:
            tc_to_se[e.u_atom].append(e)
            continue
        if u == se_tag and v == tc_tag and e.dist <= args.d_tcse:
            se_to_tc[e.u_atom].append(e)
            continue

        if e.dist <= args.d_sex:
            for M in mediators:
                if u == se_tag and v == M:
                    se_to_x[M][e.u_atom].append(e)
                elif u == M and v == se_tag:
                    x_to_se[M][e.u_atom].append(e)

    keep_top_per_key(tc_to_se, topn=max(args.top_per_se_tc, 30))
    keep_top_per_key(se_to_tc, topn=max(args.top_per_se_tc, 30))
    for M in mediators:
        keep_top_per_key(se_to_x[M], topn=max(args.top_per_se_x, 50))
        keep_top_per_key(x_to_se[M], topn=max(args.top_per_x, 50))

    paths = []

    def compute_denoms(tc_wf, se1_wf, x_wf, se2_wf) -> Tuple[float, float, float, float]:
        floor = args.delta_floor
        if args.delta_mode == "sequential":
            d1 = delta_pair(eps, se1_wf, tc_wf, floor)
            d2 = delta_pair(eps, x_wf,  se1_wf, floor)
            d3 = delta_pair(eps, se2_wf, x_wf,  floor)
        else:
            d1 = delta_pair(eps, se1_wf, tc_wf, floor)
            d2 = delta_pair(eps, x_wf,   tc_wf, floor)
            d3 = delta_pair(eps, se2_wf, tc_wf, floor)

        if any([d != d for d in (d1, d2, d3)]):  # NaN
            return float("nan"), d1, d2, d3
        return d1 * d2 * d3, d1, d2, d3

    for tc0 in tc_atoms:
        shift0 = (0, 0, 0)

        for e1 in tc_to_se.get(tc0, []):  # Tc->Se1
            se1 = e1.v_atom
            shift_se1 = add_shift(shift0, (e1.Rx, e1.Ry, e1.Rz))

            for M in mediators:
                for e2 in se_to_x[M].get(se1, []):  # Se1->X
                    x = e2.v_atom
                    shift_x = add_shift(shift_se1, (e2.Rx, e2.Ry, e2.Rz))

                    for e3 in x_to_se[M].get(x, []):  # X->Se2
                        se2 = e3.v_atom
                        shift_se2 = add_shift(shift_x, (e3.Rx, e3.Ry, e3.Rz))

                        for e4 in se_to_tc.get(se2, []):  # Se2->Tc1
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

                            paths.append((score, N, D, d1, d2, d3, sig, M,
                                          tc0, se1, x, se2, tc1, netR, e1, e2, e3, e4))

    if not paths:
        raise RuntimeError("No paths found. Try lowering --min_absH or loosening distance windows "
                           "or check tc/se/mediator tags.")

    best = {}
    for item in paths:
        score = item[0]
        sig = item[6]
        if sig not in best or score > best[sig][0]:
            best[sig] = item
    uniq = list(best.values())
    print(f"[INFO] paths raw={len(paths)}, unique={len(uniq)} (by signature)")

    uniq.sort(key=lambda x: x[0], reverse=True)
    uniq = uniq[:args.top_paths]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "mediator",
            "score_num_over_denom", "N_prod_absH", "D_prod_delta",
            "delta1", "delta2", "delta3",
            "Tc0_atom", "Se1_atom", "X_atom", "Se2_atom", "Tc1_atom",
            "net_dRx", "net_dRy", "net_dRz",
            "t1_absH", "d1_A", "wf1_u", "wf1_v", "R1x", "R1y", "R1z",
            "t2_absH", "d2_A", "wf2_u", "wf2_v", "R2x", "R2y", "R2z",
            "t3_absH", "d3_A", "wf3_u", "wf3_v", "R3x", "R3y", "R3z",
            "t4_absH", "d4_A", "wf4_u", "wf4_v", "R4x", "R4y", "R4z",
        ])

        for score, N, D, d1, d2, d3, sig, M, tc0, se1, x, se2, tc1, netR, e1, e2, e3, e4 in uniq:
            w.writerow([
                M,
                f"{score:.10g}", f"{N:.10g}", f"{D:.10g}",
                f"{d1:.10g}", f"{d2:.10g}", f"{d3:.10g}",
                tc0, se1, x, se2, tc1,
                netR[0], netR[1], netR[2],
                f"{e1.absH:.10g}", f"{e1.dist:.6f}", e1.u_wf, e1.v_wf, e1.Rx, e1.Ry, e1.Rz,
                f"{e2.absH:.10g}", f"{e2.dist:.6f}", e2.u_wf, e2.v_wf, e2.Rx, e2.Ry, e2.Rz,
                f"{e3.absH:.10g}", f"{e3.dist:.6f}", e3.u_wf, e3.v_wf, e3.Rx, e3.Ry, e3.Rz,
                f"{e4.absH:.10g}", f"{e4.dist:.6f}", e4.u_wf, e4.v_wf, e4.Rx, e4.Ry, e4.Rz,
            ])

    print(f"[DONE] wrote ranked unique paths to: {args.out}")
    print(f"[INFO] delta_mode={args.delta_mode}, delta_floor={args.delta_floor} eV")
    print(f"[INFO] prefer_group={args.prefer_group}, tc_group={args.tc_group}, se_group={args.se_group}, mediators={mediators}")


if __name__ == "__main__":
    main()
