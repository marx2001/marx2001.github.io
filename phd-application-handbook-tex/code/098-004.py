
import numpy as np
from scipy.interpolate import griddata

# ========================= 参数设置 =========================
input_file = "MAE.dat"     # 输入文件名（确保文件在当前目录）
output_prefix = "MAE_3D"   # 输出文件前缀
nphi = 500                 # 极角网格点数 (0到π)
ntheta = 500               # 方位角网格点数 (0到2π)
# ===========================================================

# ------------------------- 1. 读取数据 -------------------------
print("正在读取数据...")
data = np.loadtxt(input_file, skiprows=1)  # 跳过标题行
phi_deg = data[:, 0]       # 第一列：极角φ（单位：度）
theta_deg = data[:, 1]     # 第二列：方位角θ（单位：度）
R = data[:, 2]             # 第三列：R值

# ------------------------- 2. 角度转换 -------------------------
phi = np.deg2rad(phi_deg)  # 转换为弧度 [0, π]
theta = np.deg2rad(theta_deg)  # 转换为弧度 [0, 2π]

# ------------------------- 3. 生成新网格 -----------------------
print("生成插值网格...")
phinew = np.linspace(0, np.pi, nphi)          # 极角范围 [0, π]
thetanew = np.linspace(0, 2*np.pi, ntheta)    # 方位角范围 [0, 2π]
xx, yy = np.meshgrid(phinew, thetanew, indexing='ij')  # 创建网格

# ------------------------- 4. 插值计算 -------------------------
print("正在进行立方插值...")
points = np.column_stack((phi, theta))        # 原始数据点坐标
xi = np.column_stack((xx.ravel(), yy.ravel())) # 新网格点坐标

# 使用立方插值计算Rnew，并处理可能的NaN值
Rnew = griddata(points, R, xi, method='cubic')
Rnew = Rnew.reshape(xx.shape)                  # 转换为矩阵形状
Rnew = np.nan_to_num(Rnew, nan=np.nanmean(Rnew))  # 用均值填充缺失值

# ------------------------- 5. 转换为笛卡尔坐标 ------------------
print("坐标转换中...")
x = Rnew * np.sin(xx) * np.cos(yy)
y = Rnew * np.sin(xx) * np.sin(yy)
z = Rnew * np.cos(xx)

# 生成镜像数据（关于xy平面的对称）
x_mirror = x
y_mirror = y
z_mirror = -z

# 合并原数据和镜像数据（沿极角φ方向堆叠）
x_combined = np.vstack((x, x_mirror))
y_combined = np.vstack((y, y_mirror))
z_combined = np.vstack((z, z_mirror))
R_combined = np.vstack((Rnew, Rnew))

# ------------------------- 6. 保存为Origin矩阵文件 --------------
print("保存数据文件...")
np.savetxt(f"{output_prefix}_X.txt", x_combined, delimiter='\t', fmt='%.6f')
np.savetxt(f"{output_prefix}_Y.txt", y_combined, delimiter='\t', fmt='%.6f')
np.savetxt(f"{output_prefix}_Z.txt", z_combined, delimiter='\t', fmt='%.6f')
np.savetxt(f"{output_prefix}_R.txt", R_combined, delimiter='\t', fmt='%.6f')

print("处理完成！请在当前目录查看生成的文件。")
