
#!/usr/bin/env python3
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Extract Tc–Tc direct-exchange hoppings from edges.csv (Tc_d <-> Tc_d), ranked by |H|."
    )
    ap.add_argument("--edges", default="edges.csv", help="Input edges.csv")
    ap.add_argument("--out", default="TcTc_direct_hoppings_top.csv", help="Output csv")
    ap.add_argument("--topn", type=int, default=50, help="Top-N hoppings to output (default 50)")

    ap.add_argument("--prefer_group", action="store_true",
                    help="Use group_m/group_n tags if present (recommended).")
    ap.add_argument("--tc_tag", default="Tc_d",
                    help="Tc tag when using group columns (default Tc_d). "
                         "If not using group, this becomes element symbol (default Tc).")
    ap.add_argument("--dist_max", type=float, default=0.0,
                    help="Optional distance cutoff in Å (0 = no cutoff).")
    ap.add_argument("--min_absH", type=float, default=0.0,
                    help="Optional minimum |H| cutoff in eV (default 0).")

    ap.add_argument("--undirected_dedup", action="store_true",
                    help="Deduplicate A->B and B->A (keep the larger |H|). Recommended for clean tables.")
    args = ap.parse_args()

    df = pd.read_csv(args.edges)

    # Basic required columns
    required = ["absH_eV", "dist_A", "atom_m", "atom_n", "Rx", "Ry", "Rz", "m", "n", "elem_m", "elem_n"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {args.edges}: {missing}\nAvailable: {list(df.columns)}")

    has_group = ("group_m" in df.columns) and ("group_n" in df.columns)
    use_group = args.prefer_group and has_group

    # Normalize tag in element-mode
    if use_group:
        tc_tag = args.tc_tag
        tag_m = df["group_m"].astype(str).str.strip()
        tag_n = df["group_n"].astype(str).str.strip()
    else:
        tc_tag = args.tc_tag.split("_")[0]  # Tc_d -> Tc
        tag_m = df["elem_m"].astype(str).str.strip()
        tag_n = df["elem_n"].astype(str).str.strip()

    # Filter Tc-Tc (direct exchange channel)
    sub = df[(tag_m == tc_tag) & (tag_n == tc_tag)].copy()

    # Cutoffs
    sub["absH_eV"] = pd.to_numeric(sub["absH_eV"], errors="coerce")
    sub["dist_A"] = pd.to_numeric(sub["dist_A"], errors="coerce")
    sub = sub.dropna(subset=["absH_eV", "dist_A"])

    if args.min_absH > 0:
        sub = sub[sub["absH_eV"] >= args.min_absH]
    if args.dist_max and args.dist_max > 0:
        sub = sub[sub["dist_A"] <= args.dist_max]

    if sub.empty:
        raise RuntimeError(
            f"No Tc–Tc edges found. Check --prefer_group and --tc_tag.\n"
            f"use_group={use_group}, tc_tag={tc_tag}, "
            f"rows={len(df)}, columns={list(df.columns)}"
        )

    # Optional: keep only one of A->B and B->A
    if args.undirected_dedup:
        # Create undirected key: (min(atom), max(atom), Rx,Ry,Rz, min(wf), max(wf)) is too strict.
        # For a clean "direct exchange" table, usually dedup by (min(atom), max(atom), Rx,Ry,Rz) is enough.
        a = sub["atom_m"].astype(int)
        b = sub["atom_n"].astype(int)
        sub["_a"] = a.where(a <= b, b)
        sub["_b"] = b.where(a <= b, a)
        sub["_key"] = (
            sub["_a"].astype(str) + "-" + sub["_b"].astype(str) + "|R=" +
            sub["Rx"].astype(int).astype(str) + "," +
            sub["Ry"].astype(int).astype(str) + "," +
            sub["Rz"].astype(int).astype(str)
        )
        # Keep max |H| per key
        sub = sub.sort_values("absH_eV", ascending=False).groupby("_key", as_index=False).head(1)

    # Sort by |H| desc and take topn
    sub = sub.sort_values("absH_eV", ascending=False).head(args.topn)

    # Build a nice edge label
    sub["edge_str"] = "Tc" + sub["atom_m"].astype(int).astype(str) + "-Tc" + sub["atom_n"].astype(int).astype(str)

    # Include re/im if present
    cols = [
        "edge_str",
        "absH_eV",
        "dist_A",
        "atom_m", "atom_n",
        "m", "n",
        "Rx", "Ry", "Rz",
    ]
    if "reH_eV" in sub.columns:
        cols.insert(2, "reH_eV")
    if "imH_eV" in sub.columns:
        cols.insert(3 if "reH_eV" in sub.columns else 2, "imH_eV")
    # Also keep shell/group/pair if present
    for c in ["shell", "pair", "group_m", "group_n", "elem_m", "elem_n"]:
        if c in sub.columns and c not in cols:
            cols.append(c)

    out = sub[cols].copy()
    out.to_csv(args.out, index=False)

    # Print preview
    print(f"\n[INFO] use_group={use_group}, tc_tag={tc_tag}")
    print(f"[INFO] Tc–Tc edges found: {len(df[(tag_m==tc_tag) & (tag_n==tc_tag)])}")
    print(f"[INFO] output topn={len(out)} -> {args.out}\n")
    print(out.head(min(20, len(out))).to_string(index=False))
    print()

if __name__ == "__main__":
    main()
