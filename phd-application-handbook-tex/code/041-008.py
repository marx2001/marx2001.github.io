
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def weighted_quantile(values, weights, qs):
    """values, weights: 1D arrays; qs in [0,1]"""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = (weights > 0) & np.isfinite(values)
    values = values[m]; weights = weights[m]
    if len(values) == 0:
        return [np.nan for _ in qs]
    s = np.argsort(values)
    v = values[s]; w = weights[s]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return [float(np.interp(q, cw, v)) for q in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="cfFW_levels_fullweights.csv",
                    help="*_levels_fullweights.csv")
    ap.add_argument("--out_prefix", default="elemDist")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--emin", type=float, default=None)
    ap.add_argument("--emax", type=float, default=None)
    ap.add_argument("--min_weight", type=float, default=1e-6,
                    help="discard contributions with orbital weight < this")
    args = ap.parse_args()

    p = Path(args.full)
    if not p.exists():
        raise FileNotFoundError(p.resolve())

    df = pd.read_csv(p)

    # find weight columns
    wcols = [c for c in df.columns if c.startswith("w_")]
    if not wcols:
        raise RuntimeError("No w_* columns found. Use cfFINAL_export_fullweights.py first.")

    # energy range
    E = df["E_rel"].to_numpy(dtype=float)
    emin = np.nanmin(E) if args.emin is None else args.emin
    emax = np.nanmax(E) if args.emax is None else args.emax
    edges = np.linspace(emin, emax, args.bins + 1)

    stats_rows = []
    hist_rows = []

    # for each element + orbital: treat each level contributes weight w_orb at energy E
    for elem in sorted(df["elem"].unique()):
        sub = df[df["elem"] == elem].copy()
        Es = sub["E_rel"].to_numpy(dtype=float)

        for wc in wcols:
            orb = wc[2:]  # remove "w_"
            ws = sub[wc].to_numpy(dtype=float)

            m = ws >= args.min_weight
            if m.sum() == 0:
                continue

            v = Es[m]
            w = ws[m]

            wsum = float(w.sum())
            mean = float(np.sum(w * v) / wsum)
            var = float(np.sum(w * (v - mean)**2) / wsum)
            std = float(np.sqrt(var))

            q25, q50, q75 = weighted_quantile(v, w, [0.25, 0.50, 0.75])
            vmin = float(np.min(v))
            vmax = float(np.max(v))

            stats_rows.append({
                "elem": elem, "orbital": orb,
                "weight_sum": wsum,
                "wmean_E": mean,
                "wstd_E": std,
                "wmin_E": vmin,
                "wmax_E": vmax,
                "wq25_E": q25,
                "wq50_E": q50,
                "wq75_E": q75
            })

            # weighted histogram
            hist, _ = np.histogram(v, bins=edges, weights=w)
            centers = 0.5 * (edges[:-1] + edges[1:])
            for x, h in zip(centers, hist):
                hist_rows.append({
                    "elem": elem, "orbital": orb,
                    "E_center": float(x),
                    "weighted_count": float(h)
                })

    stats_df = pd.DataFrame(stats_rows).sort_values(["elem", "orbital"])
    hist_df = pd.DataFrame(hist_rows).sort_values(["elem", "orbital", "E_center"])

    out_stats = f"{args.out_prefix}_elem_orb_weighted_stats.csv"
    out_hist = f"{args.out_prefix}_elem_orb_hist.csv"
    stats_df.to_csv(out_stats, index=False)
    hist_df.to_csv(out_hist, index=False)

    print("[OK] wrote:")
    print(" ", out_stats)
    print(" ", out_hist)
    print(f"[INFO] energy range: [{emin:.6f}, {emax:.6f}] eV, bins={args.bins}")


if __name__ == "__main__":
    main()
