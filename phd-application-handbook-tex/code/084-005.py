import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
nkx, nky = 21, 21
nkx, nky = nkx * 3, nky * 3

# =========================
# Load data
# =========================
data = np.loadtxt("BERRYCURV.dat", skiprows=20)

# Extract reciprocal coordinates + Berry curvature
kx = data[:, 4]      # 倒数第五列
ky = data[:, 5]      # 倒数第四列
berry = data[:, 3]   # 贝里曲率

# Read B1, B2 from header (line 9 and 10, 0-based index 8,9)
with open("BERRYCURV.dat", "r") as f:
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

# 颜色：反转红蓝渐变（你要 RdBu 的反转就用 RdBu_r）
level = 100
cf = ax.contourf(kx_grid, ky_grid, berry_grid, level, cmap="RdBu_r")

# 去掉主图所有边框/坐标轴/刻度/标题
ax.set_axis_off()
ax.set_aspect("equal", adjustable="box")

# 只保留 colour bar（需要的话可删掉 label 这行）
cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"Berry Curvature $(\AA^2)$", fontsize=12)

# 保存图片（tight 以尽量减少白边，同时包含 colorbar）
fig.savefig("Berry.png", dpi=600, bbox_inches="tight", pad_inches=0)
plt.close(fig)
