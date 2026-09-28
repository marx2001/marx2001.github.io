
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =========================
# Parameters
# =========================
nkx, nky = 21, 21
nkx, nky = nkx * 3, nky * 3

# =========================
# Load data
# =========================
data = np.loadtxt('BERRYCURV.dat', skiprows=20)

# colormap: blue-white-red
colors = ['blue', 'white', 'red']
cmap = LinearSegmentedColormap.from_list('custom_red_white_blue', colors, N=256)

# Extract reciprocal coordinates + Berry curvature
kx = data[:, 4]  # 倒数第五列
ky = data[:, 5]  # 倒数第四列
berry = data[:, 3]  # 贝里曲率

# Read B1, B2 from header (line 9 and 10, 0-based index 8,9)
with open('BERRYCURV.dat', 'r') as f:
    lines = f.readlines()
B1 = np.array([float(x) for x in lines[8].split()[-3:-1]])
B2 = np.array([float(x) for x in lines[9].split()[-3:-1]])

# Convert to Cartesian-like coords used in your original script
kx_n = B1[0] * kx + B2[0] * ky
ky_n = B1[1] * kx + B2[1] * ky

# Reshape to grid
kx_grid = kx_n.reshape(nkx, nky)
ky_grid = ky_n.reshape(nkx, nky)
berry_grid = berry.reshape(nkx, nky)

# =========================
# Plot: ONLY contourf + colorbar
# =========================
fig, ax = plt.subplots(figsize=(10, 8))

contour = ax.contourf(kx_grid, ky_grid, berry_grid, levels=100, cmap=cmap)

# Remove everything from main axes: frame/spines/ticks/labels
ax.set_axis_off()
ax.set_aspect('equal', adjustable='box')

# Keep colorbar (with label; if you also want to remove label, set label="" below)
cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'Berry Curvature $(\AA^2)$', fontsize=12)

# Save: tight to avoid extra margins, but keep colorbar included
fig.savefig("Berry.png", dpi=600, bbox_inches='tight', pad_inches=0)
plt.close(fig)
