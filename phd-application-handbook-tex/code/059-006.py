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


def parse_edges_csv(path: str, prefer_group: bool = True) -> Tuple[List[DirEdge], Dict[int, str]]:
    """
    Read edges.csv produced by 1step.py.

    Required columns:
      dist_A, absH_eV, Rx,Ry,Rz, m,n, atom_m,atom_n, elem_m,elem_n
    Optional (recommended):
      group_m, group_n

    Returns:
      dir_edges: directed edges (both directions expanded)
      wf_tag: mapping wf_index -> tag (group if present else element)
    """
    edges: List[DirEdge] = []
    wf_tag: Dict[int, str] = {}

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        required = ["dist_A", "absH_eV", "Rx", "Ry", "Rz",
                    "m", "n", "atom_m", "atom_n", "elem_m", "elem_n"]
        for k in required:
            if k not in (r.fieldnames or []):
                raise RuntimeError(f"Missing column '{k}' in {path}. Found: {r.fieldnames}")

        has_group = ("group_m" in (r.fieldnames or [])) and ("group_n" in (r.fieldnames or []))
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

            # wf -> tag (group or element)
            if m not in wf_tag:
                wf_tag[m] = gm
            if n not in wf_tag:
                wf_tag[n] = gn

            # Add both directions
            edges.append(DirEdge(atom_m, atom_n, elem_m, elem_n, gm, gn, m, n, Rx, Ry, Rz, absH, dist))
            edges.append(DirEdge(atom_n, atom_m, elem_n, elem_m, gn, gm, n, m, -Rx, -Ry, -Rz, absH, dist))

    return edges, wf_tag


def load_onsite_from_hr(hr_path: str) -> Dict[int, float]:
    """
    Extract onsite energies eps[m] = Re(H_mm(R=0)) from wannier90_hr.dat.
    """
    with open(hr_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise RuntimeError(f"hr.dat too short: {hr_path}")

    num_wann = int(lines[1].split()[0])
    nrpts = int(lines[2].split()[0])

    # degeneracy list
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


def delta_pair_with_UJ(
    eps: Dict[int, float],
    wf_tag: Dict[int, str],
    wf_a: int,
    wf_b: int,
    floor: float,
    enable_UJ: bool,
    U_Tc: float,
    U_Ir: float,
    JH_Tc: float,
    JH_Ir: float,
    hund_coef: float,
    tc_tag_for_UJ: str,
    ir_tag_for_UJ: str,
) -> float:
    """
    Δ(a,b) = max(|εa-εb| + UJ(a,b), floor)

    UJ(a,b) rule (minimal & robust):
      if either endpoint tag matches tc_tag_for_UJ -> add (U_Tc + hund_coef*JH_Tc)
      if either endpoint tag matches ir_tag_for_UJ -> add (U_Ir + hund_coef*JH_Ir)

    NOTE:
      - This is a controlled phenomenological correction.
      - hund_coef default is set to 2.0 (more suitable when d-manifold is not 5-fold degenerate).
    """
    ea = eps.get(wf_a, None)
    eb = eps.get(wf_b, None)
    if ea is None or eb is None:
        return float("nan")

    base = abs(ea - eb)
    if not enable_UJ:
        return max(base, floor)

    ta = wf_tag.get(wf_a, "")
    tb = wf_tag.get(wf_b, "")

    uj = 0.0
    if (ta == tc_tag_for_UJ) or (tb == tc_tag_for_UJ):
        uj += (U_Tc + hund_coef * JH_Tc)
    if (ta == ir_tag_for_UJ) or (tb == ir_tag_for_UJ):
        uj += (U_Ir + hund_coef * JH_Ir)

    return max(base + uj, floor)


def main():
    ap = argparse.ArgumentParser(
        description="Rank Tc–Se–X–Se–Tc paths by (product |t|) / (product Δ). "
                    "Δ can include U + hund_coef*JH when Tc/Ir d tags are involved."
    )
    ap.add_argument("--edges", required=True, help="edges.csv from 1step.py")
    ap.add_argument("--hr", required=True, help="wannier90_hr.dat for onsite extraction")
    ap.add_argument("--out", default="top_paths_ratio.csv", help="Output ranked paths csv")

    # Filtering mode
    ap.add_argument("--prefer_group", action="store_true",
                    help="Prefer group_m/group_n if present. If not, fall back to elem.")
    ap.add_argument("--tc_group", default="Tc_d", help="Tc tag when prefer_group is ON (default Tc_d)")
    ap.add_argument("--se_group", default="Se_p", help="Se tag when prefer_group is ON (default Se_p)")
    ap.add_argument("--mediators", default="Ir_d",
                    help="Comma-separated mediator tags. If prefer_group OFF, use elements Ir/Ge...")

    # Geometry and pruning
    ap.add_argument("--d_tcse", type=float, default=3.0, help="Max distance for Tc–Se edges (Ang)")
    ap.add_argument("--d_sex", type=float, default=3.0, help="Max distance for Se–X edges (Ang)")
    ap.add_argument("--min_absH", type=float, default=1e-4, help="Min |t| (eV) used for edges in path building")
    ap.add_argument("--top_per_se_tc", type=int, default=60)
    ap.add_argument("--top_per_se_x", type=int, default=120)
    ap.add_argument("--top_per_x", type=int, default=120)
    ap.add_argument("--top_paths", type=int, default=2000)

    ap.add_argument("--require_same_tc_atom", action="store_true")
    ap.add_argument("--max_netR_L1", type=int, default=6)
    ap.add_argument("--exclude_netR0", action="store_true")

    # Denominator form
    ap.add_argument("--delta_mode", choices=["sequential", "pairwise"], default="sequential")
    ap.add_argument("--delta_floor", type=float, default=0.1, help="Lower bound for each Δ (eV)")

    # U/JH controls
    ap.add_argument("--enable_UJ", action="store_true",
                    help="Enable adding U + hund_coef*JH to Δ when Tc/Ir tags are involved.")
    ap.add_argument("--U_Tc", type=float, default=3.0, help="Tc Hubbard U (eV). You set: 3.0")
    ap.add_argument("--U_Ir", type=float, default=1.0, help="Ir Hubbard U (eV). You set: 1.0")
    ap.add_argument("--JH_Tc", type=float, default=0.5, help="Tc Hund J_H (eV). Default 0.5")
    ap.add_argument("--JH_Ir", type=float, default=0.4, help="Ir Hund J_H (eV). Default 0.4")

    # IMPORTANT: default 2.0 (recommended for your split d-manifold)
    ap.add_argument("--hund_coef", type=float, default=2.0,
                    help="Coefficient in U + hund_coef*JH. Default 2.0 (recommended). "
                         "Try 1/2/4 for sensitivity test.")

    args = ap.parse_args()

    mediators = [x.strip() for x in args.mediators.split(",") if x.strip()]
    if not mediators:
        raise RuntimeError("No mediators specified.")

    eps = load_onsite_from_hr(args.hr)
    all_dir, wf_tag = parse_edges_csv(args.edges, prefer_group=args.prefer_group)

    # Decide tag space
    use_group = True if args.prefer_group else False

    def Utag(e: DirEdge) -> str:
        return e.u_group if use_group else e.u_elem

    def Vtag(e: DirEdge) -> str:
        return e.v_group if use_group else e.v_elem

    tc_tag = args.tc_group if use_group else "Tc"
    se_tag = args.se_group if use_group else "Se"

    # These are the tags that trigger UJ penalty:
    tc_tag_for_UJ = tc_tag
    ir_tag_for_UJ = ("Ir_d" if use_group else "Ir")

    tc_atoms = sorted({e.u_atom for e in all_dir if Utag(e) == tc_tag} |
                      {e.v_atom for e in all_dir if Vtag(e) == tc_tag})
    se_atoms = sorted({e.u_atom for e in all_dir if Utag(e) == se_tag} |
                      {e.v_atom for e in all_dir if Vtag(e) == se_tag})

    if not tc_atoms:
        raise RuntimeError(f"No Tc nodes found under tag '{tc_tag}'. Check --prefer_group/--tc_group.")
    if not se_atoms:
        raise RuntimeError(f"No Se nodes found under tag '{se_tag}'. Check --prefer_group/--se_group.")

    print(f"[INFO] use_group={use_group}, enable_UJ={args.enable_UJ}, hund_coef={args.hund_coef}")
    print(f"[INFO] Tc tag: {tc_tag}, Tc atoms: {tc_atoms}")
    print(f"[INFO] Se tag: {se_tag}, Se atoms: {se_atoms}")
    print(f"[INFO] mediators: {mediators}")
    if args.enable_UJ:
        print(f"[INFO] U_Tc={args.U_Tc} eV, U_Ir={args.U_Ir} eV, JH_Tc={args.JH_Tc} eV, JH_Ir={args.JH_Ir} eV")

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

    def compute_denoms(tc_wf, se1_wf, x_wf, se2_wf) -> Tuple[float, float, float, float]:
        floor = args.delta_floor
        if args.delta_mode == "sequential":
            d1 = delta_pair_with_UJ(
                eps, wf_tag, se1_wf, tc_wf, floor,
                enable_UJ=args.enable_UJ,
                U_Tc=args.U_Tc, U_Ir=args.U_Ir, JH_Tc=args.JH_Tc, JH_Ir=args.JH_Ir,
                hund_coef=args.hund_coef,
                tc_tag_for_UJ=tc_tag_for_UJ, ir_tag_for_UJ=ir_tag_for_UJ
            )
            d2 = delta_pair_with_UJ(
                eps, wf_tag, x_wf, se1_wf, floor,
                enable_UJ=args.enable_UJ,
                U_Tc=args.U_Tc, U_Ir=args.U_Ir, JH_Tc=args.JH_Tc, JH_Ir=args.JH_Ir,
                hund_coef=args.hund_coef,
                tc_tag_for_UJ=tc_tag_for_UJ, ir_tag_for_UJ=ir_tag_for_UJ
            )
            d3 = delta_pair_with_UJ(
                eps, wf_tag, se2_wf, x_wf, floor,
                enable_UJ=args.enable_UJ,
                U_Tc=args.U_Tc, U_Ir=args.U_Ir, JH_Tc=args.JH_Tc, JH_Ir=args.JH_Ir,
                hund_coef=args.hund_coef,
                tc_tag_for_UJ=tc_tag_for_UJ, ir_tag_for_UJ=ir_tag_for_UJ
            )
        else:
            d1 = delta_pair_with_UJ(
                eps, wf_tag, se1_wf, tc_wf, floor,
                enable_UJ=args.enable_UJ,
                U_Tc=args.U_Tc, U_Ir=args.U_Ir, JH_Tc=args.JH_Tc, JH_Ir=args.JH_Ir,
                hund_coef=args.hund_coef,
                tc_tag_for_UJ=tc_tag_for_UJ, ir_tag_for_UJ=ir_tag_for_UJ
            )
            d2 = delta_pair_with_UJ(
                eps, wf_tag, x_wf, tc_wf, floor,
                enable_UJ=args.enable_UJ,
                U_Tc=args.U_Tc, U_Ir=args.U_Ir, JH_Tc=args.JH_Tc, JH_Ir=args.JH_Ir,
                hund_coef=args.hund_coef,
                tc_tag_for_UJ=tc_tag_for_UJ, ir_tag_for_UJ=ir_tag_for_UJ
            )
            d3 = delta_pair_with_UJ(
                eps, wf_tag, se2_wf, tc_wf, floor,
                enable_UJ=args.enable_UJ,
                U_Tc=args.U_Tc, U_Ir=args.U_Ir, JH_Tc=args.JH_Tc, JH_Ir=args.JH_Ir,
                hund_coef=args.hund_coef,
                tc_tag_for_UJ=tc_tag_for_UJ, ir_tag_for_UJ=ir_tag_for_UJ
            )

        if any([d != d for d in (d1, d2, d3)]):  # NaN
            return float("nan"), d1, d2, d3
        return d1 * d2 * d3, d1, d2, d3

    paths = []

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
                            paths.append((score, N, D, d1, d2, d3,
                                          M, tc0, se1, x, se2, tc1, netR,
                                          e1, e2, e3, e4))

    if not paths:
        raise RuntimeError("No paths found. Try loosening distances, lowering --min_absH, "
                           "or check tags (prefer_group/tc_group/se_group/mediators).")

    # de-dup by signature
    best = {}
    for item in paths:
        score = item[0]
        key = (item[6], item[7], item[8], item[9], item[10], item[11], item[12],
               (item[13].u_wf, item[13].v_wf, item[13].Rx, item[13].Ry, item[13].Rz),
               (item[14].u_wf, item[14].v_wf, item[14].Rx, item[14].Ry, item[14].Rz),
               (item[15].u_wf, item[15].v_wf, item[15].Rx, item[15].Ry, item[15].Rz),
               (item[16].u_wf, item[16].v_wf, item[16].Rx, item[16].Ry, item[16].Rz))
        if key not in best or score > best[key][0]:
            best[key] = item

    uniq = list(best.values())
    uniq.sort(key=lambda x: x[0], reverse=True)
    uniq = uniq[:args.top_paths]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            # run meta
            "use_group", "enable_UJ", "U_Tc", "U_Ir", "JH_Tc", "JH_Ir", "hund_coef", "delta_mode", "delta_floor",
            # path score
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

        for (score, N, D, d1, d2, d3,
             M, tc0, se1, x, se2, tc1, netR,
             e1, e2, e3, e4) in uniq:
            w.writerow([
                str(use_group), str(args.enable_UJ),
                f"{args.U_Tc:.6g}", f"{args.U_Ir:.6g}", f"{args.JH_Tc:.6g}", f"{args.JH_Ir:.6g}",
                f"{args.hund_coef:.6g}", args.delta_mode, f"{args.delta_floor:.6g}",
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
    print(f"[INFO] enable_UJ={args.enable_UJ}, U_Tc={args.U_Tc}, U_Ir={args.U_Ir}, "
          f"JH_Tc={args.JH_Tc}, JH_Ir={args.JH_Ir}, hund_coef={args.hund_coef}")


if __name__ == "__main__":
    main()
