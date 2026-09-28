
# ===== Cell: output conduction-band Berry curvature along the same k-path =====
import numpy as np
import matplotlib.pyplot as plt

# 1) sanity: require CB result already computed in the main cell
if "Omega_CB3" not in globals():
    raise NameError("Omega_CB3 not found. Please run the main calculation cell (the loop over k_path) first.")
if "k_path" not in globals():
    raise NameError("k_path not found. Please run the k-path construction cell first.")
if "x" not in globals():
    x = np.arange(len(Omega_CB3), dtype=float)

# 2) build a clean output file name
try:
    ncb = int(N_CB)
except Exception:
    ncb = None

cb_prefix = f"{OUT_PREFIX}_CB_first{ncb}" if ncb is not None else f"{OUT_PREFIX}_CB"
cb_dat = cb_prefix + ".dat"
cb_png = cb_prefix + ".png"

# 3) save CB curve data
# columns:
# 1) idx (x-axis index)
# 2) k1_red
# 3) k2_red
# 4) Omega_CB (sum over first N_CB conduction bands)
header = "idx k1_red k2_red Omega_CB(sum_first_NCB)"
np.savetxt(
    cb_dat,
    np.column_stack([x, k_path[:, 0], k_path[:, 1], Omega_CB3]),
    fmt="%.10f",
    header=header
)
print("[INFO] saved", cb_dat)

# 4) plot CB curve (same x-axis ticks as your VB plot)
plt.figure(figsize=(8.0, 3.6), dpi=220)
plt.plot(x, Omega_CB3, lw=2.0)
plt.axhline(0, lw=0.8)
for tp in tick:
    plt.axvline(tp, lw=0.6)
plt.xticks(tick, lab)
plt.xlabel("k-path: G–K'–M–K–G")
plt.ylabel(r"$\Omega_{xy}$ (first $N_\mathrm{CB}$ conduction bands sum)")
plt.tight_layout()
plt.savefig(cb_png, bbox_inches="tight")
plt.show()
print("[INFO] saved", cb_png)

print("\n[For plotting from file]")
print("x-axis = column 1 (idx)")
print("y-axis = column 4 (Omega_CB)")
