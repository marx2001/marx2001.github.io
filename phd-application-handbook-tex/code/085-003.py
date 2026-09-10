import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from tqdm import tqdm
from scipy.interpolate import CloughTocher2DInterpolator

# 参数设置
#discrete_num = 500               # 控制色块数  (对于清晰度更重要，但是也大幅影响出图速度，并且由于数据本身的连续性， 太大不一定会有肉眼提升)
#save_dpi = discrete_num/5        # 导出图像分辨率，由于figsize = (5,5) 因此取 discrete_num 的 1/5 可以保证精度同时让图像最小


input_dir = "./"                    # ovf 文件所在目录
output_dir = "."                # 输出图像目录
#is_plot_inplain = 0                # 0：只根据Sz设置颜色, 1：Sx, Sy决定颜色 Sz决定明暗
#cmap = 'bwr'                       # 如果只根据Sz, 控制Sz映射颜色  （默认 bwr：红色 +z 方向）

#is_colorbar = 0                    # 是否加上colorbar

# ============ 晶格与原子信息（手动设置） ===================
repeat_x = 300         # 扩胞尺寸
repeat_y = 300
repeat_z = 1

mag_atom_posi = [
    [0.0000027711196385, 0.0000078820326969, 0.3339943794051239],
    [0.5000027711196385, 0.0000078820326969, 0.3339943794051239],
    [0.2500027711196385, 0.5000078820326969, 0.3339943794051239],
    [0.7500027711196385, 0.5000078820326969, 0.3339943794051239]
]

base_vec = [
    [13.1502104227330392, -0.0001278456871606, 0.0013066726293966],  # a 向量
    [-0.0001009508214391, 11.3881951329254409, -0.0008005358804461],  # b 向量
    [0.0016905845507726, -0.0009144982993895, 17.5548063557554279]   # c 向量
]

# repeat_x = 50         # 扩胞尺寸
# repeat_y = 50
# repeat_z = 1

# mag_atom_posi = [
#     [0.0, 0.0, 0.0]  # 原胞中原子的相对坐标
# ]

# base_vec = [
#     [1.0, 0.0, 0.0],  # a 向量
#     [0.0, 1.0, 0.0],  # b 向量
#     [0.0, 0.0, 1.0]   # c 向量
# ]
# ============================================================

# 自动扩胞生成所有原子绝对坐标
def generate_supercell_positions(mag_atom_posi, base_vec, repeat_x, repeat_y, repeat_z):
    base_vec = np.array(base_vec)  # shape: (3, 3)
    pos_list = []

    # Step 1: 将分数坐标转换为绝对坐标
    mag_atom_abs = []
    for atom in mag_atom_posi:
        frac = np.array(atom)  # shape: (3,)
        abs_pos = frac @ base_vec  # 点乘得到绝对坐标
        mag_atom_abs.append(abs_pos)

    # Step 2: 对每个晶胞进行平移复制
    for i in range(repeat_z):
        for j in range(repeat_y):
            for k in range(repeat_x):
                shift = k * base_vec[0] + j * base_vec[1] + i * base_vec[2]
                for abs_pos in mag_atom_abs:
                    new_pos = abs_pos + shift
                    pos_list.append(new_pos)

    pos_array = np.array(pos_list)
    return pos_array[:, 0], pos_array[:, 1], pos_array[:, 2]  # 返回 x, y, z 坐标

# 绘图主函数
def plot_inplain(x, y, spin_x, spin_y, spin_z, fig_path, xi, yi, dpi=100):
    XI, YI = np.meshgrid(xi, yi)

    interpolator_x = CloughTocher2DInterpolator(np.column_stack([x, y]), spin_x)
    interpolator_y = CloughTocher2DInterpolator(np.column_stack([x, y]), spin_y)
    interpolator_z = CloughTocher2DInterpolator(np.column_stack([x, y]), spin_z)

    sx = interpolator_x(XI, YI)
    sx = np.clip(sx, -1.0, 1.0)
    sy = interpolator_y(XI, YI)
    sy = np.clip(sy, -1.0, 1.0)
    sz = interpolator_z(XI, YI)
    sz = np.clip(sz, -1.0, 1.0)

    hue = (np.arctan2(sy, sx) + np.pi) / (2 * np.pi)
    value = Normalize(vmin=-1, vmax=1)(sz)
    value = np.clip(value, 0, 1)

    hsv = np.zeros((sz.shape[0], sz.shape[1], 3))
    hsv[..., 0] = hue
    hsv[..., 1] = 1 - np.abs(1 - 2 * value)
    hsv[..., 2] = value
    rgb = colors.hsv_to_rgb(hsv)

    plt.imshow(rgb, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', aspect='auto')
    #plt.colorbar(format='%5.2f')
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent = True)
    plt.close()

def plot(x, y, spin_x, spin_y, spin_z, fig_path, xi, yi, cmap, dpi=100):
    XI, YI = np.meshgrid(xi, yi)
    interpolator_z = CloughTocher2DInterpolator(np.column_stack([x, y]), spin_z)
    sz = interpolator_z(XI, YI)
    sz = np.clip(sz, -1.0, 1.0)

    plt.figure(figsize=(5, 5))
    plt.imshow(sz, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', aspect='auto', cmap = cmap, vmin=-1, vmax=1)
    if is_colorbar:
        plt.colorbar(format='%5.2f')
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent = True)
    plt.close()

# 主执行函数
def main():
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件列表
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".ovf"))
    if not files:
        print("[!]未找到 ovf 文件")
        return

    # 生成原子坐标与插值网格
    x, y, z = generate_supercell_positions(mag_atom_posi, base_vec, repeat_x, repeat_y, repeat_z)
    #xi = np.linspace(x.min(), x.max(), discrete_num)
    #yi = np.linspace(y.min(), y.max(), discrete_num)

    # 批量绘图
    for file in tqdm(files, desc="处理文件"):
        file_path = os.path.join(input_dir, file)
        if os.path.isdir(file_path):
            continue
        data = np.loadtxt(file_path)

        spin_x, spin_y, spin_z = data[:, 0], data[:, 1], data[:, 2]

        spin_data = np.column_stack((x,y,z,spin_x, spin_y, spin_z))

        fig_name = os.path.splitext(file)[0] + ".csv"
        fig_path = os.path.join(output_dir, fig_name)
        np.savetxt(fig_path, spin_data, delimiter=',')
        

    print("[OK] 自旋数据已生成，保存在：", output_dir)

# 执行
if __name__ == "__main__":
    main()
