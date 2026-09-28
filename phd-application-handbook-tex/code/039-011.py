
#!/usr/bin/env python3
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Extract Tc–Se, Se–X, X–Se, Se–Tc hoppings from chains_top3.csv and print numeric values."
    )
    ap.add_argument("--in_csv", default="chains_top3.csv", help="Input chains table from 2step grouping")
    ap.add_argument("--out_csv", default="hopping_edges_from_chains.csv", help="Output expanded edge list")
    ap.add_argument("--top_edge_each", type=int, default=10, help="Top-N edges per edge-type by |t| (default 10)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # required columns
    req = ["path_str", "t1_absH", "t2_absH", "t3_absH", "t4_absH"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing columns in {args.in_csv}: {miss}. Available: {list(df.columns)}")

    # atom indices if present
    atom_cols = ["Tc0", "Se1", "X", "Se2", "Tc1"]
    has_atoms = all(c in df.columns for c in atom_cols)

    # mediator tag if present
    med_col = "mediator" if "mediator" in df.columns else None

    rows = []
    for _, r in df.iterrows():
        path = r["path_str"]
        source = r["source"] if "source" in df.columns else ""
        chain_id = r["chain_id"] if "chain_id" in df.columns else None
        rank_in_chain = r["rank_in_chain"] if "rank_in_chain" in df.columns else None

        if has_atoms:
            Tc0, Se1, X, Se2, Tc1 = int(r["Tc0"]), int(r["Se1"]), int(r["X"]), int(r["Se2"]), int(r["Tc1"])
        else:
            Tc0 = Se1 = X = Se2 = Tc1 = None

        med = str(r[med_col]) if med_col else ""
        # Expand 4 steps
        rows.append({
            "chain_id": chain_id, "rank_in_chain": rank_in_chain, "source": source, "mediator": med,
            "path_str": path, "edge_type": "Tc-Se", "edge_str": f"Tc{Tc0}-Se{Se1}" if has_atoms else "",
            "absH_eV": float(r["t1_absH"])
        })
        rows.append({
            "chain_id": chain_id, "rank_in_chain": rank_in_chain, "source": source, "mediator": med,
            "path_str": path, "edge_type": "Se-X", "edge_str": f"Se{Se1}-X{X}" if has_atoms else "",
            "absH_eV": float(r["t2_absH"])
        })
        rows.append({
            "chain_id": chain_id, "rank_in_chain": rank_in_chain, "source": source, "mediator": med,
            "path_str": path, "edge_type": "X-Se", "edge_str": f"X{X}-Se{Se2}" if has_atoms else "",
            "absH_eV": float(r["t3_absH"])
        })
        rows.append({
            "chain_id": chain_id, "rank_in_chain": rank_in_chain, "source": source, "mediator": med,
            "path_str": path, "edge_type": "Se-Tc", "edge_str": f"Se{Se2}-Tc{Tc1}" if has_atoms else "",
            "absH_eV": float(r["t4_absH"])
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    # Print per-path detailed values
    print("\n=== Per-path hoppings (|t| in eV) ===")
    show_cols = ["chain_id", "rank_in_chain", "source", "path_str", "edge_type", "edge_str", "absH_eV"]
    if "chain_id" not in out.columns:
        show_cols = ["path_str", "edge_type", "edge_str", "absH_eV"]
    print(out[show_cols].to_string(index=False))

    # Top edges per type
    print(f"\n=== Top {args.top_edge_each} edges per edge_type (by |t|) ===")
    for et in ["Tc-Se", "Se-X", "X-Se", "Se-Tc"]:
        sub = out[out["edge_type"] == et].sort_values("absH_eV", ascending=False).head(args.top_edge_each)
        print(f"\n[{et}]")
        print(sub[["path_str", "edge_str", "absH_eV"]].to_string(index=False))

    print(f"\n[DONE] wrote: {args.out_csv}\n")


if __name__ == "__main__":
    main()
