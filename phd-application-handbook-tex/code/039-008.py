#!/usr/bin/env python3
import argparse
import pandas as pd


def find_col(df, candidates, required=True, name=""):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(f"Cannot find column for {name}. Tried {candidates}. "
                           f"Available columns: {list(df.columns)}")
    return None


def pretty_mediator(tag: str) -> str:
    # Ir_d -> Ir, Ge_p -> Ge
    if isinstance(tag, str) and "_" in tag:
        return tag.split("_")[0]
    return str(tag)


def load_2step(csv_path: str, source_label: str, topk: int):
    df = pd.read_csv(csv_path)

    score_col = find_col(df, ["score_num_over_denom", "score", "S"], True, "score")
    med_col   = find_col(df, ["mediator", "mediator_tag", "M"], True, "mediator")

    Tc0 = find_col(df, ["Tc0", "Tc0_atom"], True, "Tc0")
    Se1 = find_col(df, ["Se1", "Se1_atom"], True, "Se1")
    X   = find_col(df, ["X", "X_atom"], True, "X")
    Se2 = find_col(df, ["Se2", "Se2_atom"], True, "Se2")
    Tc1 = find_col(df, ["Tc1", "Tc1_atom"], True, "Tc1")

    netRx = find_col(df, ["net_dRx", "netRx", "net_Rx"], required=False, name="netRx")
    netRy = find_col(df, ["net_dRy", "netRy", "net_Ry"], required=False, name="netRy")
    netRz = find_col(df, ["net_dRz", "netRz", "net_Rz"], required=False, name="netRz")
    has_netR = (netRx is not None) and (netRy is not None) and (netRz is not None)

    df = df.copy()
    df["score"] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=["score"])
    df = df.sort_values("score", ascending=False)

    if topk and topk > 0:
        df = df.head(topk)

    df["source"] = source_label
    df["mediator"] = df[med_col].astype(str)

    df["Tc0"] = df[Tc0].astype(int)
    df["Se1"] = df[Se1].astype(int)
    df["X"]   = df[X].astype(int)
    df["Se2"] = df[Se2].astype(int)
    df["Tc1"] = df[Tc1].astype(int)

    if has_netR:
        df["netRx"] = df[netRx].astype(int)
        df["netRy"] = df[netRy].astype(int)
        df["netRz"] = df[netRz].astype(int)
    else:
        df["netRx"] = 0
        df["netRy"] = 0
        df["netRz"] = 0

    # path string: Tc2-Se9-Ir1-Se8-Tc2
    def mk(row):
        M = pretty_mediator(row["mediator"])
        return f"Tc{row['Tc0']}-Se{row['Se1']}-{M}{row['X']}-Se{row['Se2']}-Tc{row['Tc1']}"
    df["path_str"] = df.apply(mk, axis=1)

    return df


def main():
    ap = argparse.ArgumentParser(
        description="Merge Ir/Ge 2step outputs, group by atom-chain, and keep top-3 rows per chain by score."
    )
    ap.add_argument("--ir_csv", default="Ir_d_UJ.csv")
    ap.add_argument("--ge_csv", default="Ge_1e-3_0.1.csv")
    ap.add_argument("--top_each", type=int, default=2000,
                    help="Read top K rows from each file before grouping (0 = read all). Default 2000.")
    ap.add_argument("--top_per_chain", type=int, default=3,
                    help="Keep top N rows per atom-chain (default 3).")
    ap.add_argument("--key_with_netR", action="store_true",
                    help="Include netR (net_dR*) in chain key (recommended if netR exists).")
    ap.add_argument("--out", default="chains_top3.csv")
    args = ap.parse_args()

    topk = None if args.top_each == 0 else args.top_each

    df_ir = load_2step(args.ir_csv, "Ir", 0 if topk is None else topk)
    df_ge = load_2step(args.ge_csv, "Ge", 0 if topk is None else topk)

    df = pd.concat([df_ir, df_ge], ignore_index=True)

    # group key: atom chain (+ optional netR)
    key_cols = ["Tc0", "Se1", "X", "Se2", "Tc1"]
    if args.key_with_netR:
        key_cols += ["netRx", "netRy", "netRz"]

    # rank chains by their best score (for stable chain_id)
    best = df.groupby(key_cols, dropna=False)["score"].max().reset_index()
    best = best.sort_values("score", ascending=False).reset_index(drop=True)
    best["chain_id"] = range(1, len(best) + 1)

    df = df.merge(best[key_cols + ["chain_id"]], on=key_cols, how="left")

    # within each chain keep top N
    df = df.sort_values("score", ascending=False).copy()
    df["rank_in_chain"] = df.groupby("chain_id")["score"].rank(method="first", ascending=False).astype(int)
    out = df[df["rank_in_chain"] <= args.top_per_chain].copy()
    out = out.sort_values(["chain_id", "rank_in_chain"], ascending=[True, True])

    # choose output columns (keep more if present)
    cols = ["chain_id", "rank_in_chain", "source", "path_str", "score",
            "Tc0", "Se1", "X", "Se2", "Tc1"]
    if args.key_with_netR:
        cols += ["netRx", "netRy", "netRz"]

    # optional physics columns if present
    for c in ["t1_absH", "t2_absH", "t3_absH", "t4_absH", "delta1", "delta2", "delta3"]:
        if c in out.columns:
            cols.append(c)

    # also include raw mediator tag (Ir_d / Ge_p) if you want
    if "mediator" in out.columns:
        cols.insert(cols.index("path_str"), "mediator")

    out[cols].to_csv(args.out, index=False)

    # brief preview
    print("\n=== Per-chain top results (top per chain) ===")
    print(f"[INFO] merged_rows={len(df)}  chains={best.shape[0]}  output_rows={len(out)}")
    print(out[cols].head(30).to_string(index=False))
    print(f"\n[DONE] wrote: {args.out}\n")


if __name__ == "__main__":
    main()
