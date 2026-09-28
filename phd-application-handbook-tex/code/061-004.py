#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="cfA2_levels.csv", help="*_levels.csv from the CF script")
    ap.add_argument("--out_prefix", default="cf_elem", help="output prefix")
    ap.add_argument("--min_dom_w", type=float, default=0.20,
                    help="discard levels whose dominant_w < this threshold (label too mixed)")
    args = ap.parse_args()

    levels_path = Path(args.levels)
    if not levels_path.exists():
        raise FileNotFoundError(levels_path.resolve())

    df = pd.read_csv(levels_path)

    need = {"atom_id", "elem", "n_wf", "level_index", "E_rel", "dominant_orb", "dominant_w"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"levels file missing columns: {sorted(list(miss))}")

    # clean
    df["dominant_orb"] = df["dominant_orb"].astype(str).str.strip()
    df = df[df["dominant_orb"] != ""].copy()

    # filter mixed labels
    df_f = df[df["dominant_w"] >= args.min_dom_w].copy()

    # long table output
    long_out = f"{args.out_prefix}_elem_orb_levels_long.csv"
    df_f.to_csv(long_out, index=False)

    # base stats
    stats = (
        df_f.groupby(["elem", "dominant_orb"])["E_rel"]
        .agg(count="count", mean="mean", std="std", min="min", max="max")
        .reset_index()
    )

    # quantiles (robust way)
    qs = (
        df_f.groupby(["elem", "dominant_orb"])["E_rel"]
        .quantile([0.25, 0.50, 0.75])
        .unstack(level=-1)
        .reset_index()
        .rename(columns={0.25: "q25", 0.50: "q50", 0.75: "q75"})
    )

    stats = stats.merge(qs, on=["elem", "dominant_orb"], how="left")

    # per-element splitting summary (range of orbital medians)
    med = stats.pivot_table(index="elem", columns="dominant_orb", values="q50", aggfunc="first")

    elem_split = []
    for elem in med.index:
        vals = med.loc[elem].dropna().to_numpy()
        if len(vals) == 0:
            continue
        elem_split.append({
            "elem": elem,
            "orbital_median_range": float(vals.max() - vals.min()),
            "orbital_median_min": float(vals.min()),
            "orbital_median_max": float(vals.max()),
            "n_orbitals": int(len(vals))
        })
    elem_split_df = pd.DataFrame(elem_split).sort_values(["elem"])

    # write outputs
    stats_out = f"{args.out_prefix}_elem_orb_stats.csv"
    split_out = f"{args.out_prefix}_elem_splitting_summary.csv"

    stats.sort_values(["elem", "dominant_orb"]).to_csv(stats_out, index=False)
    elem_split_df.to_csv(split_out, index=False)

    print("[OK] wrote:")
    print(" ", long_out)
    print(" ", stats_out)
    print(" ", split_out)
    print(f"[INFO] kept {len(df_f)} / {len(df)} levels (min_dom_w={args.min_dom_w})")

    if len(elem_split_df) > 0:
        print("\n[Summary] per-element orbital median range (q50 max-min):")
        for _, r in elem_split_df.iterrows():
            print(f"  {r['elem']}: range={r['orbital_median_range']:.4f} eV over {r['n_orbitals']} orbitals")


if __name__ == "__main__":
    main()
