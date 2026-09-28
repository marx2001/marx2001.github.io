# ===== Cell: build k-distance x-axis + print high-symmetry k coords =====
import numpy as np
import pandas as pd

# 1) Print high-symmetry points (reduced + Cartesian)
rows = []
for name, kred in nodes:
    kred = np.asarray(kred, float)
    k_cart = red_to_cart2(kred[:2])  # 2D cart (Å^-1)
    rows.append({
        "label": name,
        "k_red_1": kred[0], "k_red_2": kred[1], "k_red_3": kred[2] if kred.size > 2 else 0.0,
        "k_cart_x(1/Å)": k_cart[0], "k_cart_y(1/Å)": k_cart[1],
        "|k_cart|(1/Å)": float(np.linalg.norm(k_cart)),
    })

df_kpts = pd.DataFrame(rows)
display(df_kpts)

# 2) Build cumulative k-path distance as x-axis (Å^-1)
kcart_path = np.array([red_to_cart2(k[:2]) for k in k_path])  # (N,2)
dk = np.linalg.norm(np.diff(kcart_path, axis=0), axis=1)      # (N-1,)
x_kdist = np.concatenate([[0.0], np.cumsum(dk)])              # (N,)

# 3) Get tick positions (x-axis locations) for each high-symmetry node
#    This must match how you built k_path: first point of first seg kept;
#    later segments drop the first point (seg[1:]).
tick_idx = [0]
pos = 0
for i in range(len(nodes)-1):
    seg_len = N_PER_SEG if i == 0 else (N_PER_SEG - 1)
    pos += seg_len
    tick_idx.append(pos - 1)

tick_pos = x_kdist[tick_idx]
tick_lab = [name for name, _ in nodes]

print("tick_idx =", tick_idx)
print("tick_pos (1/Å) =", tick_pos)
print("tick_lab =", tick_lab)

# Now, in your plot cell, use:
# plt.plot(x_kdist, Omega_VB)  or plt.plot(x_kdist, Omega_VBM)
# plt.xticks(tick_pos, tick_lab)
