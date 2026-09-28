#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def orbital_sort_key(orb):
    p_order = ["px", "py", "pz"]
    d_order = ["dxy", "dyz", "dxz", "dz2", "dx2-y2"]
    if orb in p_order:
        return (0, p_order.index(orb))
    if orb in d_order:
        return (1, d_order.index(orb))
    return (2, orb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="elemCFdist_elem_orb_hist.csv")
    ap.add_argument("--elem", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--normalize", action="store_true",
                    help="Normalize each orbital curve by its total area (sum of weighted_count).")
    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    p = Path(args.hist)
    if not p.exists():
        raise FileNotFoundError(p.resolve())

    df = pd.read_csv(p)
    need = {"elem", "orbital", "E_center", "weighted_count"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns in hist csv: {sorted(list(miss))}")

    sub = df[df["elem"].astype(str).str.strip() == args.elem].copy()
    if len(sub) == 0:
        raise RuntimeError(f"No rows for elem={args.elem} in {p.name}")

    sub["orbital"] = sub["orbital"].astype(str).str.strip()

    orbs = sorted(sub["orbital"].unique(), key=orbital_sort_key)

    plt.figure(figsize=(5.6, 4.8))

    for orb in orbs:
        s = sub[sub["orbital"] == orb].sort_values("E_center")
        x = s["E_center"].to_numpy(float)
        y = s["weighted_count"].to_numpy(float)

        if args.normalize:
            area = float(np.sum(y))
            if area > 0:
                y = y / area

        plt.plot(x, y, linewidth=1.8, label=orb)

    plt.xlabel("Energy (E_rel) [eV]")
    plt.ylabel("Weighted intensity" + (" (normalized)" if args.normalize else ""))

    ttl = args.title if args.title else f"{args.elem}: orbital-weighted energy distributions"
    plt.title(ttl)

    if args.xmin is not None or args.xmax is not None:
        plt.xlim(args.xmin, args.xmax)

    plt.legend()
    plt.tight_layout()

    out = args.out if args.out else f"{args.elem}_orbital_distributions.png"
    plt.savefig(out, dpi=args.dpi)
    print("[OK] saved:", out)


if __name__ == "__main__":
    main()
