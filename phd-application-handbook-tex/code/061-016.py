#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COLS = ["E","s","py","pz","px","dxy","dyz","dz2","dxz","dx2y2","tot"]
D_ORBS = ["dxy","dyz","dz2","dxz","dx2y2"]

def load_pdos(path: Path):
    return pd.read_csv(path, sep=r"\s+", comment="#", names=COLS, skiprows=1)

def trapz(E, Y):
    return np.trapz(Y, E)

def band_center(E, D, Emin, Emax):
    m = (E >= Emin) & (E <= Emax)
    Ew = E[m]; Dw = D[m]
    denom = trapz(Ew, Dw)
    if abs(denom) < 1e-14:
        return np.nan, 0.0
    num = trapz(Ew, Ew * Dw)
    return num/denom, denom

def cosine_sim(E, A, B, Emin, Emax):
    m = (E >= Emin) & (E <= Emax)
    a = A[m]; b = B[m]
    # L2-normalize by integral of square (continuous analog)
    na = np.sqrt(trapz(E[m], a*a))
    nb = np.sqrt(trapz(E[m], b*b))
    if na < 1e-14 or nb < 1e-14:
        return np.nan
    return trapz(E[m], a*b) / (na*nb)

def auto_groups(sim_mat, labels, thr=0.95):
    # 简单阈值连通分量：sim>=thr 即连边
    n = len(labels)
    visited = [False]*n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if not visited[v] and sim_mat[u,v] >= thr:
                    visited[v] = True
                    stack.append(v)
        groups.append([labels[k] for k in sorted(comp)])
    return groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--up", required=True)
    ap.add_argument("--dw", required=True)
    ap.add_argument("--Emin", type=float, default=-1.0)
    ap.add_argument("--Emax", type=float, default=0.0)
    ap.add_argument("--thr", type=float, default=0.95, help="cosine similarity threshold for grouping")
    ap.add_argument("--out_prefix", default="Auto")
    args = ap.parse_args()

    up = load_pdos(Path(args.up))
    dw = load_pdos(Path(args.dw))

    if not np.allclose(up["E"].values, dw["E"].values, atol=1e-10):
        raise RuntimeError("Energy grids differ between UP/DW.")

    E = up["E"].to_numpy(float)
    Dup = up[D_ORBS].to_numpy(float)
    Ddw = np.abs(dw[D_ORBS].to_numpy(float))
    D = Dup + Ddw  # spin-summed DOS

    # --- similarity matrix ---
    n = len(D_ORBS)
    sim = np.zeros((n,n), float)
    for i in range(n):
        for j in range(n):
            if i == j:
                sim[i,j] = 1.0
            elif j < i:
                sim[i,j] = sim[j,i]
            else:
                sim[i,j] = cosine_sim(E, D[:,i], D[:,j], args.Emin, args.Emax)

    df_sim = pd.DataFrame(sim, index=D_ORBS, columns=D_ORBS)
    df_sim.to_csv(f"{args.out_prefix}_sim_matrix.csv")

    groups = auto_groups(sim, D_ORBS, thr=args.thr)

    # --- per-orbital band centers ---
    orb_rows = []
    for i, orb in enumerate(D_ORBS):
        Ec, Nw = band_center(E, D[:,i], args.Emin, args.Emax)
        orb_rows.append({"orb": orb, "E_center": Ec, "N_window": Nw})
    df_orb = pd.DataFrame(orb_rows).sort_values("E_center")
    df_orb.to_csv(f"{args.out_prefix}_orb_centers.csv", index=False)

    # --- per-group band centers (sum DOS in group) ---
    grp_rows = []
    for g in groups:
        idx = [D_ORBS.index(o) for o in g]
        Dg = D[:,idx].sum(axis=1)
        Ec, Nw = band_center(E, Dg, args.Emin, args.Emax)
        grp_rows.append({"group": "(" + ",".join(g) + ")", "members": ",".join(g), "E_center": Ec, "N_window": Nw})
    df_grp = pd.DataFrame(grp_rows).sort_values("E_center")
    df_grp.to_csv(f"{args.out_prefix}_auto_groups.csv", index=False)

    print("[OK] similarity matrix:", f"{args.out_prefix}_sim_matrix.csv")
    print("[OK] orbital centers:", f"{args.out_prefix}_orb_centers.csv")
    print("[OK] auto groups:", f"{args.out_prefix}_auto_groups.csv")
    print("\n[Auto groups @ thr =", args.thr, "]")
    print(df_grp.to_string(index=False))

if __name__ == "__main__":
    main()
