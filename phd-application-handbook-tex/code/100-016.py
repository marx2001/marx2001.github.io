
# =========================
# Output high-symmetry x-coordinates (VASPKIT-like klabels)
# x-axis coordinate = cumulative |dk| along the path (in 1/Å)
# =========================

# 1) compute cumulative distance axis x_kdist (same curve, but physical distance)
kcart_path = np.array([red_to_cart2(k[:2]) for k in k_path])  # (N,2) in 1/Å
dk = np.linalg.norm(np.diff(kcart_path, axis=0), axis=1)
x_kdist = np.concatenate([[0.0], np.cumsum(dk)])              # (N,)

# 2) high-symmetry points positions on x-axis: use your existing tick indices
# tick already stores indices on the discrete path
klabels_lines = []
klabels_lines.append("# K-Label    Coordinate of high-symmetry k-point in band-structure plots")
for name, idx in zip(lab, tick):
    # optional: map "G" -> "GAMMA" to match VASPKIT style
    out_name = "GAMMA" if name in ["G", "Γ", "Gamma", "GAMMA"] else name
    klabels_lines.append(f"{out_name:>10s}   {x_kdist[idx]:.3f}")

# 3) print and save
print("\n".join(klabels_lines))

with open("klabels_berry.dat", "w", encoding="utf-8") as f:
    f.write("\n".join(klabels_lines) + "\n")

print("[INFO] saved klabels_berry.dat")
print("[INFO] x_kdist (1/Å) stored in variable: x_kdist")
