
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import Normalize, LinearSegmentedColormap

# ===== 数据加载 =====
data = np.loadtxt('MAE.dat', skiprows=1)
phi = data[:, 0] * np.pi / 180
theta = data[:, 1] * np.pi / 180
R = data[:, 2]

# ===== 网格生成 =====
nphi, ntheta = 500, 500
phinew = np.linspace(0, np.pi, nphi)
thetanew = np.linspace(0, 2*np.pi, ntheta)
xx, yy = np.meshgrid(phinew, thetanew)

# ===== 镜像对称插值 =====
phi_mirror = np.pi - xx
theta_mirror = (yy + np.pi) % (2*np.pi)
R_upper = griddata((phi, theta), R, (xx, yy), method='cubic')
R_lower = griddata((phi, theta), R, (phi_mirror, theta_mirror), method='cubic')
Rnew = np.where(xx > np.pi/2, R_lower, R_upper)

# ===== 坐标转换 (放大 4 倍) =====
scale_factor = 4.0
x = scale_factor * Rnew * np.sin(xx) * np.cos(yy)
y = scale_factor * Rnew * np.sin(xx) * np.sin(yy)
z = scale_factor * Rnew * np.cos(xx)

# ===== 柔和渐变色 (橙-黄-绿) =====
cmap_soft = LinearSegmentedColormap.from_list(
    "OrangeYellowGreen", ["#FFA500", "#FFFF66", "#008000"]
)

# ===== 可视化 =====
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 统一颜色映射
norm = Normalize(vmin=np.min(Rnew), vmax=np.max(Rnew))
colors = cmap_soft(norm(Rnew))

surf = ax.plot_surface(
    x, y, z,
    facecolors=colors,
    rcount=300, ccount=300,
    antialiased=True
)

# ===== 视角和比例 =====
ax.view_init(elev=30, azim=-50)

# 自动保持 x,y,z 方向比例一致
ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])

# ===== 设置坐标范围（比图形大 1.5 倍） =====
margin = 1.5
ax.set_xlim([np.min(x) * margin, np.max(x) * margin])
ax.set_ylim([np.min(y) * margin, np.max(y) * margin])
ax.set_zlim([np.min(z) * margin, np.max(z) * margin])

# ===== 坐标轴样式：只保留网格，去掉刻度数字和标签 =====
ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.grid(True)

# 网格线颜色调浅
ax.xaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 1)
ax.yaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 1)
ax.zaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 1)

# ===== 颜色条（柔和渐变） =====
mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_soft)
mappable.set_array([])
cbar = fig.colorbar(mappable, shrink=0.5, aspect=10)
cbar.set_label("")  # 去掉标签

# ===== 保存 & 显示 =====
plt.tight_layout()
plt.savefig("MAE_3D_orange_yellow_green_scaled4_margin15.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()
