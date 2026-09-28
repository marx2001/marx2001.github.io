# ===== Next cell: plot 1D Berry-curvature curve with high-symmetry x-axis + write final.dat =====
import numpy as np
import matplotlib.pyplot as plt

# ---- Safety checks: pick which Omega you already computed ----
# Prefer Omega_VBM if exists, else Omega_VB
if "Omega_VBM" in globals():
    Omega_y = np.asarray(Omega_VBM, float)
    y_label = r"$\Omega_{xy}$ (VBM single band)"
elif "Omega_VB" in globals():
    Omega_y = np.asarray(Omega_VB, float)
    y_label = r"$\Omega_{xy}$ (occupied sum)"
else:
    raise NameError("Neither Omega_VBM nor Omega_VB found. Please run the Berry-curvature calculation cell first.")

# ---- Build k-distance x-axis (Å^-1) ----
kcart_path = np.array([red_to_cart2(k[:2]) for k in k_path])  # (N,2)
dk = np.linalg.norm(np.diff(kcart_path, axis=0), axis=1)      # (N-1,)
x_kdist = np.concatenate([[0.0], np.cumsum(dk)])              # (N,)

# ---- High-symmetry ticks (must match your k_path construction) ----
tick_idx = [0]
pos = 0
for i in range(len(nodes)-1):
    seg_len = N_PER_SEG if i == 0 else (N_PER_SEG - 1)  # because later segments drop first point
    pos += seg_len
    tick_idx.append(pos - 1)

tick_pos = x_kdist[tick_idx]
tick_lab = [name for name, _ in nodes]

# ---- Save final.dat ----
# Columns:
# 1) x_kdist (Å^-1)
# 2) Omega_y
# 3) k1 (reduced)
# 4) k2 (reduced)
# 5) idx (0-based point index)
final_dat = "final.dat"
header = "x_kdist(1/A) Omega_xy k1_red k2_red idx"
out = np.column_stack([x_kdist, Omega_y, k_path[:,0], k_path[:,1], np.arange(len(k_path))])
np.savetxt(final_dat, out, fmt="%.10f", header=header)
print("[INFO] saved", final_dat)

# ---- Plot ----
plt.figure(figsize=(8.0, 3.6), dpi=220)
plt.plot(x_kdist, Omega_y, lw=2.0)
plt.axhline(0, lw=0.8)

for tp in tick_pos:
    plt.axvline(tp, lw=0.6)

plt.xticks(tick_pos, tick_lab)
plt.xlabel("k-path: " + "–".join(tick_lab) + r"  (cumulative |dk|, 1/Å)")
plt.ylabel(y_label)
plt.tight_layout()
plt.show()

print("\n[For your plotting]")
print("x-axis = column 1 : x_kdist(1/A)")
print("y-axis = column 2 : Omega_xy")
