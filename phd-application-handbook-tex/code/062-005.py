#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paper_fig5_like_soc.py  (final, with --unique_bonds)

One-shot, maximum literature-like reproduction of Fig.5(b)/(d):
- bond-resolved NN selection by atom distance window (POSCAR + Rx,Ry,Rz)
- SOC spinor 2×2 block -> Frobenius norm t_eff
- export: matrix heatmap + long-table + raw edges for top bond-images
- NEW: --unique_bonds to avoid double counting when A==B (e.g., Tc-Tc)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

D_ORBS = ["dxy", "dyz", "dxz", "dz2", "dx2-y2"]
P_ORBS = ["px", "py", "pz"]

# ---------------- POSCAR ----------------
def read_poscar(path: str):
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]

    scale = float(lines[1].split()[0])
    a1 = np.array(list(map(float, lines[2].split()[:3]))) * scale
    a2 = np.array(list(map(float, lines[3].split()[:3]))) * scale
    a3 = np.array(list(map(float, lines[4].split()[:3]))) * scale
    A = np.stack([a1, a2, a3], axis=1)  # columns

    species = lines[5].split()
    counts  = list(map(int, lines[6].split()))
    if len(species) != len(counts):
        raise RuntimeError("POSCAR species/counts mismatch")

    idx = 7
    if lines[idx].lower().startswith("selective"):
        idx += 1

    mode_line = lines[idx].lower()
    if mode_line.startswith("d"):
        mode = "Direct"
    elif mode_line.startswith("c") or mode_line.startswith("k"):
        mode = "Cartesian"
    else:
        raise RuntimeError(f"Unknown coordinate mode: {lines[idx]}")
    idx += 1

    nat = sum(counts)
    coords = np.array([list(map(float, lines[idx+i].split()[:3])) for i in range(nat)], dtype=float)
    if mode == "Cartesian":
        frac = np.linalg.solve(A, coords.T).T
    else:
        frac = coords.copy()

    elems_by_atom = []
    for sp, c in zip(species, counts):
        elems_by_atom += [sp] * c

    return A, np.asarray(frac), elems_by_atom

def bond_distance_with_R(A, frac, i0, j0, R):
    df = (frac[j0] + np.array(R, dtype=float)) - frac[i0]
    dr = A @ df
    return float(np.linalg.norm(dr))

# ---------------- plotting ----------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def heatmap_with_values(mat, xlabels, ylabels, title, out_png: Path, cbar_label="t_eff"):
    fig = plt.figure(figsize=(6.6, 5.0), dpi=180)
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    finite = np.isfinite(mat)
    vmax = float(np.max(mat[finite])) if np.any(finite) else 1.0
    thresh = 0.55 * vmax

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            s = "0" if abs(v) < 1e-15 else f"{v:.3g}"
            color = "white" if v >= thresh else "black"
            ax.text(j, i, s, ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(np.arange(-.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ---------------- orientation / filter ----------------
def orient_to_AB(df, A, B):
    """
    Keep (A,B) or (B,A). Swap BA->AB so elem_m=A, elem_n=B.
    Note: If A==B, this will keep both directions as-is; use --unique_bonds to avoid double counting.
    """
    df = df.copy()
    mAB = (df["elem_m"] == A) & (df["elem_n"] == B)
    mBA = (df["elem_m"] == B) & (df["elem_n"] == A)
    df_AB = df.loc[mAB].copy()
    df_BA = df.loc[mBA].copy()

    if len(df_BA) > 0 and A != B:
        for c1, c2 in [("m","n"),("atom_m","atom_n"),("elem_m","elem_n"),("orb_m","orb_n")]:
            tmp = df_BA[c1].copy()
            df_BA[c1] = df_BA[c2].values
            df_BA[c2] = tmp.values

    return pd.concat([df_AB, df_BA], ignore_index=True)

def filter_kind(df, kind):
    D = set(D_ORBS); P = set(P_ORBS)
    if kind == "dd":
        return df[df["orb_m"].isin(D) & df["orb_n"].isin(D)].copy(), D_ORBS, D_ORBS
    if kind == "pp":
        return df[df["orb_m"].isin(P) & df["orb_n"].isin(P)].copy(), P_ORBS, P_ORBS
    # dp after AB: m is d, n is p
    return df[df["orb_m"].isin(D) & df["orb_n"].isin(P)].copy(), D_ORBS, P_ORBS

# ---------------- SOC pairing & t_eff ----------------
def build_spin_of_wf(dfAB, strict=False):
    """
    Determine SOC Kramers pair (2 WFs) for each (atom_id, spatial_orb).
    Assign spin index 0/1 by sorting WF ids.
    """
    groups = {}
    for side in ["m", "n"]:
        wf = dfAB[side].astype(int).to_numpy()
        atom = dfAB[f"atom_{side}"].astype(int).to_numpy()
        orb  = dfAB[f"orb_{side}"].astype(str).to_numpy()
        for w, a, o in zip(wf, atom, orb):
            key = (int(a), str(o))
            groups.setdefault(key, set()).add(int(w))

    spin_of_wf = {}
    bad = []
    for (a, o), s in groups.items():
        wfs = sorted(list(s))
        if len(wfs) != 2:
            bad.append(((a, o), wfs))
            continue
        spin_of_wf[wfs[0]] = 0
        spin_of_wf[wfs[1]] = 1

    if bad:
        msg = ["[WARN] Some (atom,orb) do not have exactly 2 WFs (SOC)."]
        for (a, o), wfs in bad[:30]:
            msg.append(f"  atom={a}, orb={o}, wfs={wfs}")
        msg.append("  (showing up to 30)")
        print("\n".join(msg))
        if strict:
            raise RuntimeError("strict_spinor enabled: found (atom,orb) with !=2 WFs.")

    return spin_of_wf

def teff_block(df_block, spin_of_wf):
    """
    df_block: fixed bond image + fixed (orb_m, orb_n).
    Construct 2×2 complex H and compute Frobenius norm.
    """
    H = np.zeros((2, 2), dtype=np.complex128)
    for _, r in df_block.iterrows():
        m = int(r["m"]); n = int(r["n"])
        sm = spin_of_wf.get(m, None)
        sn = spin_of_wf.get(n, None)
        if sm is None or sn is None:
            continue
        H[sm, sn] = complex(float(r["reH_eV"]), float(r["imH_eV"]))
    return float(np.sqrt(np.sum(np.abs(H) ** 2)))

def build_teff_matrix_for_bond(df_bond, A_orbs, B_orbs, spin_of_wf):
    mat = np.zeros((len(A_orbs), len(B_orbs)), dtype=float)
    ia = {o: i for i, o in enumerate(A_orbs)}
    ib = {o: i for i, o in enumerate(B_orbs)}
    for (om, on), g in df_bond.groupby(["orb_m", "orb_n"]):
        if om not in ia or on not in ib:
            continue
        mat[ia[om], ib[on]] = teff_block(g, spin_of_wf)
    return mat

def long_table_orb_pairs(df_bond, spin_of_wf):
    """
    Long table: each row is (orb_m, orb_n, t_eff, and the 2x2 complex matrix entries).
    """
    rows = []
    for (om, on), g in df_bond.groupby(["orb_m", "orb_n"]):
        H = np.zeros((2, 2), dtype=np.complex128)
        for _, r in g.iterrows():
            m = int(r["m"]); n = int(r["n"])
            sm = spin_of_wf.get(m, None)
            sn = spin_of_wf.get(n, None)
            if sm is None or sn is None:
                continue
            H[sm, sn] = complex(float(r["reH_eV"]), float(r["imH_eV"]))
        teff = float(np.sqrt(np.sum(np.abs(H) ** 2)))
        rows.append({
            "orb_m": om, "orb_n": on, "t_eff": teff,
            "H00_re": H[0, 0].real, "H00_im": H[0, 0].imag,
            "H01_re": H[0, 1].real, "H01_im": H[0, 1].imag,
            "H10_re": H[1, 0].real, "H10_im": H[1, 0].imag,
            "H11_re": H[1, 1].real, "H11_im": H[1, 1].imag,
        })
    dfL = pd.DataFrame(rows)
    if not dfL.empty:
        dfL.sort_values("t_eff", ascending=False, inplace=True)
    return dfL

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Fig.5-like (bond-resolved) orbital hybridization with SOC t_eff (Frobenius norm of 2x2 spinor block)."
    )
    ap.add_argument("--poscar", required=True)
    ap.add_argument("--edges", required=True, help="edges_with_orb_reim.csv (must include reH_eV/imH_eV)")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--A", required=True, help="e.g. Tc")
    ap.add_argument("--B", required=True, help="e.g. Se")
    ap.add_argument("--kind", default="dp", choices=["dd", "dp", "pp"])

    ap.add_argument("--dmin", type=float, required=True, help="NN distance window min (Å)")
    ap.add_argument("--dmax", type=float, required=True, help="NN distance window max (Å)")

    ap.add_argument("--min_absH", type=float, default=0.0, help="pre-filter edges by absH_eV")
    ap.add_argument("--top_bonds", type=int, default=10, help="export top N NN bond-images")
    ap.add_argument("--strict_spinor", action="store_true", help="require exactly 2 WFs per (atom,orb)")
    # ---- NEW ----
    ap.add_argument("--unique_bonds", action="store_true",
                    help="When A==B, keep only atom_m < atom_n to avoid double counting.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    A_lat, frac, elems = read_poscar(args.poscar)
    nat = len(elems)

    df = pd.read_csv(args.edges)
    need = ["elem_m", "elem_n", "atom_m", "atom_n", "orb_m", "orb_n", "absH_eV",
            "reH_eV", "imH_eV", "Rx", "Ry", "Rz", "m", "n"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"edges missing columns: {miss}")

    if args.min_absH > 0:
        df = df[df["absH_eV"] >= args.min_absH].copy()

    # orient + kind filter
    dfAB = orient_to_AB(df, args.A, args.B)
    dfAB, A_orbs, B_orbs = filter_kind(dfAB, args.kind)
    if dfAB.empty:
        raise RuntimeError("No edges after A/B/kind filtering.")

    # SOC pairing
    spin_of_wf = build_spin_of_wf(dfAB, strict=args.strict_spinor)

    # compute atom distance per edge (atom_m -> atom_n + R)
    am = dfAB["atom_m"].astype(int).to_numpy()
    an = dfAB["atom_n"].astype(int).to_numpy()
    Rx = dfAB["Rx"].astype(int).to_numpy()
    Ry = dfAB["Ry"].astype(int).to_numpy()
    Rz = dfAB["Rz"].astype(int).to_numpy()

    if am.min() < 1 or an.min() < 1 or am.max() > nat or an.max() > nat:
        raise RuntimeError("atom indices out of POSCAR range")

    d_atom = np.empty(len(dfAB), dtype=float)
    for i in range(len(dfAB)):
        d_atom[i] = bond_distance_with_R(A_lat, frac, am[i]-1, an[i]-1, (Rx[i], Ry[i], Rz[i]))

    dfAB = dfAB.copy()
    dfAB["d_atom_R_A"] = d_atom

    # NN window selection (paper-like)
    dfNN = dfAB[(dfAB["d_atom_R_A"] >= args.dmin) & (dfAB["d_atom_R_A"] <= args.dmax)].copy()
    if dfNN.empty:
        raise RuntimeError("No edges in NN window. Widen dmin/dmax or check POSCAR/edges.")

    # ---- NEW: avoid double counting for A==B ----
    if args.unique_bonds and args.A == args.B:
        dfNN = dfNN[dfNN["atom_m"].astype(int) < dfNN["atom_n"].astype(int)].copy()
        if dfNN.empty:
            raise RuntimeError("After --unique_bonds filtering, no edges left. Check distance window.")

    # bond-image identity
    bond_cols = ["atom_m", "atom_n", "Rx", "Ry", "Rz"]
    bond_rows = []
    mats_cache = {}

    for key, g in dfNN.groupby(bond_cols):
        mat = build_teff_matrix_for_bond(g, A_orbs, B_orbs, spin_of_wf)
        score = float(mat.sum())
        dval = float(g["d_atom_R_A"].mean())
        bond_rows.append((*key, dval, len(g), score))
        mats_cache[key] = (mat, g)

    bond_sum = pd.DataFrame(
        bond_rows,
        columns=["atom_m", "atom_n", "Rx", "Ry", "Rz", "d_atom_A", "n_edges", "sum_teff"]
    ).sort_values("sum_teff", ascending=False)

    # export global ranking
    bond_rank = outdir / f"BONDS_{args.A}-{args.B}_{args.kind}_TEFF_d{args.dmin}-{args.dmax}.csv"
    bond_sum.to_csv(bond_rank, index=False)

    # export top bonds
    topN = min(args.top_bonds, len(bond_sum))
    summary = []

    for bi in range(topN):
        r = bond_sum.iloc[bi]
        key = (int(r["atom_m"]), int(r["atom_n"]), int(r["Rx"]), int(r["Ry"]), int(r["Rz"]))
        mat, g = mats_cache[key]
        a_m, a_n, Rx0, Ry0, Rz0 = key
        dval = float(r["d_atom_A"])

        tag = f"{args.A}{a_m}-{args.B}{a_n}_R{Rx0}{Ry0}{Rz0}_{args.kind}_TEFF"
        bdir = outdir / tag
        safe_mkdir(bdir)

        # (1) matrix + heatmap
        pd.DataFrame(mat, index=A_orbs, columns=B_orbs).to_csv(bdir / f"{tag}_matrix.csv")
        title = f"{tag} | d={dval:.3f}Å | Σ t_eff={mat.sum():.4g}"
        heatmap_with_values(
            mat, B_orbs, A_orbs, title, bdir / f"{tag}.png",
            cbar_label=r"$t^{\mathrm{eff}}=\|H^{2\times2}\|_F$ (eV)"
        )

        # (2) long-table orbital pairs
        dfL = long_table_orb_pairs(g, spin_of_wf)
        dfL.to_csv(bdir / f"{tag}_orbital_pairs_long.csv", index=False)

        # (3) flat list from matrix (all entries)
        flat = []
        for i, ao in enumerate(A_orbs):
            for j, bo in enumerate(B_orbs):
                flat.append((ao, bo, mat[i, j]))
        flat.sort(key=lambda x: x[2], reverse=True)
        pd.DataFrame(flat, columns=["A_orb", "B_orb", "t_eff"]).to_csv(
            bdir / f"{tag}_orbital_pairs_matrix_entries.csv", index=False
        )

        # (4) raw edges used
        g.to_csv(bdir / f"{tag}_edges.csv", index=False)

        summary.append({
            "rank": bi + 1,
            "bond_tag": tag,
            "atom_m": a_m, "atom_n": a_n,
            "Rx": Rx0, "Ry": Ry0, "Rz": Rz0,
            "d_atom_A": dval,
            "sum_teff": float(mat.sum()),
            "max_teff": float(mat.max()),
            "argmax_pair": f"{flat[0][0]}-{flat[0][1]}" if flat else "NA",
        })

    pd.DataFrame(summary).to_csv(outdir / f"SUMMARY_top{topN}_bonds_TEFF.csv", index=False)

    print(f"[OK] NN edges: {len(dfNN)} in window [{args.dmin},{args.dmax}] Å")
    print(f"[OK] unique NN bond-images: {len(bond_sum)}")
    print(f"[OK] bond ranking: {bond_rank}")
    print(f"[OK] exported top bond packages: {topN}")
    print(f"[OK] summary: {outdir / f'SUMMARY_top{topN}_bonds_TEFF.csv'}")

if __name__ == "__main__":
    main()
