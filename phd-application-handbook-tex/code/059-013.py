
import pandas as pd
import argparse

J1_NETR = {
    ( 1, 0, 0),
    (-1, 0, 0),
    ( 0, 1, 0),
    ( 0,-1, 0),
    ( 1, 1, 0),
    (-1,-1, 0),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path file (Ir_xxx.csv or Ge_xxx.csv)")
    ap.add_argument("--topk", type=int, default=100, help="sum topK scores after filtering to J1 shell")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # 兼容列名：你文件里通常是 net_dRx/net_dRy/net_dRz 与 score_num_over_denom
    for c in ["net_dRx", "net_dRy", "net_dRz", "score_num_over_denom", "mediator"]:
        if c not in df.columns:
            raise RuntimeError(f"missing column: {c}, have: {list(df.columns)}")

    df["netR"] = list(zip(df["net_dRx"], df["net_dRy"], df["net_dRz"]))
    sub = df[df["netR"].isin(J1_NETR)].copy()

    if sub.empty:
        raise RuntimeError("No rows in J1 shell after filtering. Check your net_dR convention.")

    sub = sub.sort_values("score_num_over_denom", ascending=False)

    topk = min(args.topk, len(sub))
    ssum = sub["score_num_over_denom"].head(topk).sum()
    smax = sub["score_num_over_denom"].iloc[0]
    med = sub["mediator"].iloc[0]

    print(f"[INFO] file: {args.csv}")
    print(f"[INFO] mediator (top row): {med}")
    print(f"[INFO] J1-shell rows: {len(sub)}")
    print(f"[INFO] top1 score: {smax:.6g}")
    print(f"[INFO] sum top{topk} score: {ssum:.6g}")

if __name__ == "__main__":
    main()
