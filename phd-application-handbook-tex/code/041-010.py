
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def orbital_sort_key(orb):
    # order p then d if mixed; within sets use common order
    p_order = ["px", "py", "pz"]
    d_order = ["dxy", "dyz", "dxz", "dz2", "dx2-y2"]
    if orb in p_order:
        return (0, p_order.index(orb))
    if orb in d_order:
        return (1, d_order.index(orb))
    return (2, orb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="elemCFdist_elem_orb_weighted_stats.csv")
    ap.add_argument("--elem", required=True, help="Element symbol, e.g. Se / Tc / Ir / Ge")
    ap.add_argument("--out", default=None, help="Output image path (png/pdf). If omitted, auto name.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--band", action="store_true",
                    help="Draw q25-q75 interval as a vertical band around the median energy.")
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    p = Path(args.stats)
    if not p.exists():
        raise FileNotFoundError(p.resolve())

    df = pd.read_csv(p)
    need = {"elem", "orbital", "wq25_E", "wq50_E", "wq75_E", "weight_sum"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns in stats csv: {sorted(list(miss))}")

    sub = df[df["elem"].astype(str).str.strip() == args.elem].copy()
    if len(sub) == 0:
        raise RuntimeError(f"No rows for elem={args.elem} in {p.name}")

    sub["orbital"] = sub["orbital"].astype(str).str.strip()
    sub = sub.sort_values("orbital", key=lambda s: s.map(orbital_sort_key))

    orbs = sub["orbital"].tolist()
    y50 = sub["wq50_E"].to_numpy(float)
    y25 = sub["wq25_E"].to_numpy(float)
    y75 = sub["wq75_E"].to_numpy(float)

    # x positions: 1..N
    xs = np.arange(1, len(orbs) + 1)

    plt.figure(figsize=(4.8, 5.5))

    # draw median level as horizontal line segments
    for x, e in zip(xs, y50):
        plt.hlines(e, x - 0.35, x + 0.35, linewidth=2)

    # optional: draw q25-q75 as vertical band
    if args.band:
        for x, lo, hi in zip(xs, y25, y75):
            plt.vlines(x, lo, hi, linewidth=2, alpha=0.7)

    plt.xticks(xs, orbs, rotation=0)
    plt.ylabel("Energy (E_rel) [eV]")

    ttl = args.title if args.title else f"{args.elem}: orbital level positions (weighted q50)"
    plt.title(ttl)

    if args.ymin is not None or args.ymax is not None:
        plt.ylim(args.ymin, args.ymax)

    plt.tight_layout()

    out = args.out if args.out else f"{args.elem}_level_diagram.png"
    plt.savefig(out, dpi=args.dpi)
    print("[OK] saved:", out)


if __name__ == "__main__":
    main()
