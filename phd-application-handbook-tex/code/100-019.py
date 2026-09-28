
#上个代码用的是hermite插值，然后单位转换，转换后数值太大了，现在不进行单位转换
#改用未经单位转换的数据

# ===== Cell: CB Berry curvature (Hermite + publication style, NO unit conversion) =====
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# ---------------------------
# 0) Sanity check
# ---------------------------
if "Omega_CB3" not in globals():
    raise NameError("Omega_CB3 not found. Run main calculation cell first.")
if "k_path" not in globals():
    raise NameError("k_path not found.")
if "tick" not in globals() or "lab" not in globals():
    raise NameError("tick/lab not found.")
if "x" not in globals():
    x = np.arange(len(Omega_CB3), dtype=float)

y = np.asarray(Omega_CB3, float)
x0 = np.asarray(x, float)

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

print("max raw =", np.max(y))
print("min raw =", np.min(y))

# ---------------------------
# 1) Hermite smoothing
# ---------------------------
Nsub = 40
m = np.zeros_like(y)
m[1:-1] = (y[2:] - y[:-2]) / (x0[2:] - x0[:-2])

# enforce extrema at K− and K+
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
    t2 = t*t
    t3 = t2*t
    h00 =  2*t3 - 3*t2 + 1
    h10 =      t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 =      t3 -   t2
    return h00*ya + h10*h*ma + h01*yb + h11*h*mb

xd_list = [x0[0]]
yd_list = [y[0]]

for i in range(len(x0)-1):
    xa, xb = x0[i], x0[i+1]
    ya, yb = y[i], y[i+1]
    ma, mb = m[i], m[i+1]
    tt = np.linspace(0, 1, Nsub+1)[1:]
    xd = xa + (xb - xa) * tt
    yd = hermite_segment(xa, xb, ya, yb, ma, mb, tt)
    xd_list.append(xd)
    yd_list.append(yd)

x_smooth = np.concatenate([np.atleast_1d(v) for v in xd_list])
y_smooth = np.concatenate([np.atleast_1d(v) for v in yd_list])

print("max smooth =", np.max(y_smooth))

# ---------------------------
# 2) Save data
# ---------------------------
header = "idx k1_red k2_red Omega_CB_A2"
np.savetxt(
    "CB_berrycurve.dat",
    np.column_stack([x0, k_path[:,0], k_path[:,1], y]),
    fmt="%.10f",
    header=header
)
print("[INFO] saved CB_berrycurve.dat")

# ---------------------------
# 3) Publication-style plot
# ---------------------------
W_cm, H_cm = 12.9, 4.66
fig = plt.figure(figsize=(W_cm/2.54, H_cm/2.54), dpi=600)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
})

ax = plt.gca()

# curve
ax.plot(
    x_smooth, y_smooth,
    lw=1.8,
    color="#8ecae6",
    solid_capstyle="round",
    solid_joinstyle="round"
)

# dashed guides
guide_lw = 0.8 * 0.75
guide_color = "#f4a3a3"
guide_ls = (0, (4, 3))

for tp in tick_idx:
    ax.axvline(tp, lw=guide_lw, color=guide_color, linestyle=guide_ls, zorder=0)

ax.axhline(0.0, lw=guide_lw, color=guide_color, linestyle=guide_ls, zorder=0)

# x ticks
ax.set_xticks(tick_idx)
ax.set_xticklabels(tick_lab2, fontsize=10)

# y range 自动根据数据留冗余
ymax = np.max(y_smooth)
ymin = np.min(y_smooth)
margin = 0.1 * (ymax - ymin)

ax.set_ylim(ymin - margin, ymax + margin)

ax.set_ylabel(r"$\Omega_z\;(\mathrm{\AA}^2)$", fontsize=12)
ax.set_xlabel(r"$k$-path", fontsize=11)

# box but no top/right ticks
ax.tick_params(direction="in", length=4.0, width=1.0,
               top=False, right=False, bottom=True, left=True)

ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)

plt.tight_layout(pad=0.5)

plt.savefig("CB_berrycurve_pub.png", bbox_inches="tight")
plt.savefig("CB_berrycurve_pub.pdf", bbox_inches="tight")
plt.show()

print("[INFO] saved CB_berrycurve_pub.png/pdf")

# ---------------------------
# 4) Print K− / K+
# ---------------------------
def _val_at(idx):
    return float(y[idx])

if idx_Km is not None:
    print(f"[K−] Omega = {_val_at(idx_Km): .10f} Å²")
if idx_Kp is not None:
    print(f"[K+] Omega = {_val_at(idx_Kp): .10f} Å²")
