
# ===== Cell: keep top/right spines (box), but NO top/right ticks =====
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# ---------------------------
# 0) Pick y-data
# ---------------------------
if "Omega_VBM" in globals():
    y0 = np.asarray(Omega_VBM, float)
    ysrc = "Omega_VBM"
elif "Omega_VB" in globals():
    y0 = np.asarray(Omega_VB, float)
    ysrc = "Omega_VB"
else:
    raise NameError("Omega_VBM or Omega_VB not found. Please run the Berry-curvature calculation cell first.")

x0 = np.asarray(x, float) if "x" in globals() else np.arange(len(y0), dtype=float)

if "tick" not in globals() or "lab" not in globals():
    raise NameError("tick/lab not found. Please run the k-path construction cell first.")

tick_idx = list(tick)
tick_lab = list(lab)

def _rename(lbl):
    if lbl in ["G", "Γ", "Gamma", "GAMMA"]:
        return "Γ"
    if lbl in ["Kp", "K'", "K-"]:
        return "K−"
    if lbl in ["K", "K+"]:
        return "K+"
    return lbl

tick_lab2 = [_rename(s) for s in tick_lab]

# ---------------------------
# 1) Units: Å^2 -> Bohr^2
# ---------------------------
CONVERT_A2_TO_BOHR2 = True
ANG_PER_BOHR = 0.529177210903
A2_TO_BOHR2 = (1.0 / ANG_PER_BOHR) ** 2

if CONVERT_A2_TO_BOHR2:
    y = y0 * A2_TO_BOHR2
    y_label = r"$\Omega_z\;(\mathrm{Bohr}^2)$"
    y_unit = "Bohr^2"
else:
    y = y0.copy()
    y_label = r"$\Omega_z\;(\mathrm{\AA}^2)$"
    y_unit = "Å^2"

# ---------------------------
# 2) Gentle smoothing (Hermite) with extrema at K− and K+
# ---------------------------
Nsub = 40
m = np.zeros_like(y)
m[1:-1] = (y[2:] - y[:-2]) / (x0[2:] - x0[:-2])

idx_Km = None
idx_Kp = None
for lbl, idx in zip(tick_lab2, tick_idx):
    if lbl == "K−":
        idx_Km = idx
    if lbl == "K+":
        idx_Kp = idx

if idx_Km is not None:
    m[idx_Km] = 0.0
if idx_Kp is not None:
    m[idx_Kp] = 0.0
m[0] = 0.0
m[-1] = 0.0

def hermite_segment(xa, xb, ya, yb, ma, mb, t):
    h = xb - xa
    t2 = t * t
    t3 = t2 * t
    h00 =  2*t3 - 3*t2 + 1
    h10 =      t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 =      t3 -   t2
    return h00*ya + h10*h*ma + h01*yb + h11*h*mb

xd_list = [x0[0]]
yd_list = [y[0]]
for i in range(len(x0) - 1):
    xa, xb = x0[i], x0[i+1]
    ya, yb = y[i], y[i+1]
    ma, mb = m[i], m[i+1]
    tt = np.linspace(0.0, 1.0, Nsub+1)[1:]
    xd = xa + (xb - xa) * tt
    yd = hermite_segment(xa, xb, ya, yb, ma, mb, tt)
    xd_list.append(xd)
    yd_list.append(yd)

x_smooth = np.concatenate([np.atleast_1d(v) for v in xd_list])
y_smooth = np.concatenate([np.atleast_1d(v) for v in yd_list])

# ---------------------------
# 3) Plot styling
# ---------------------------
W_cm, H_cm = 12.9, 4.66
fig = plt.figure(figsize=(W_cm/2.54, H_cm/2.54), dpi=600)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
})

ax = plt.gca()

# curve: light blue
ax.plot(
    x_smooth, y_smooth,
    lw=1.8,
    color="#8ecae6",
    solid_capstyle="round",
    solid_joinstyle="round"
)

# dashed guides: high-symmetry + y=0, thin 25%, light red
guide_lw = 0.8 * 0.75
guide_color = "#f4a3a3"
guide_ls = (0, (4, 3))

for tp in tick_idx:
    ax.axvline(tp, lw=guide_lw, color=guide_color, linestyle=guide_ls, zorder=0)
ax.axhline(0.0, lw=guide_lw, color=guide_color, linestyle=guide_ls, zorder=0)

# x ticks at high-symmetry points
ax.set_xticks(tick_idx)
ax.set_xticklabels(tick_lab2, fontsize=10)

# y range + 4 y-ticks
ax.set_ylim(-14, 10)
ax.yaxis.set_major_locator(FixedLocator([-14, -6, 2, 10]))

ax.set_ylabel(y_label, fontsize=12)
ax.set_xlabel(r"$k$-path", fontsize=11)

# Keep top/right spines (box), but disable top/right ticks
ax.tick_params(direction="in", length=4.0, width=1.0, labelsize=10,
               top=False, right=False, bottom=True, left=True)

# Ensure spines are visible (do NOT hide them)
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)

plt.tight_layout(pad=0.5)

out_png = "berrycurve_pub_v4.png"
out_pdf = "berrycurve_pub_v4.pdf"
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()

# ---------------------------
# 4) Print K− / K+ values
# ---------------------------
def _val_at(idx):
    return float(y[idx])

print("[INFO] saved:", out_png, "and", out_pdf)
print("[INFO] Omega source:", ysrc, "| units:", y_unit)

if idx_Km is not None:
    print(f"[K−] idx={idx_Km}, Omega={_val_at(idx_Km): .10f} {y_unit}")
else:
    print("[K−] not found; available:", tick_lab2)

if idx_Kp is not None:
    print(f"[K+] idx={idx_Kp}, Omega={_val_at(idx_Kp): .10f} {y_unit}")
else:
    print("[K+] not found; available:", tick_lab2)
