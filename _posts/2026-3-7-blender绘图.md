---
layout: post
title: "(原创脚本)blender绘制斯格明子"
subtitle: "Experience Sharing"
background: '/img/bg-sci-note.jpg'
categories: sci-note
permalink: /sci-note_posts/20260307-blender
---

## <center>说明</center>

均为原创，禁止抄袭，转载注明出处，否则必会追究。

## <center>方法</center>  

(1) 原子自旋模拟，提交spirit任务得到ovf文件。

(2) 收集所有的ovf文件，并转换成图片，脚本如下：

sh脚本负责收集，python负责绘制，配色为原创：

```shell

#!/bin/bash -x
mkdir result
for i in {0,5,10,15,20,25}
	do
for j in {0,100,200,300,400,500}
	do 
	cp B$i/T$j/*600000.ovf ./result/B$i-T$j-final.ovf
done
done

```

```python

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pylab import *
import zhshen as zs

# 设置文件路径
path_top = "/public/home/cssong/song/1mrx/9_single_layer/19_ReIrGe2S6/TcIrGeSe/5_spirit/2_spirit/2_spirit/result/"
fig_path = path_top + "0-figs-Mz"

FILE1 = path_top + "POSCAR"
repeat_x = 300
repeat_y = 300
repeat_z = 1

# 读取POSCAR文件
mag_atom_posi = zs.read_poscar(FILE1)[4]
base_vec = zs.read_poscar(FILE1)[2]

# 生成超胞原子坐标
super_posi = []
count = 0
for i in range(int(repeat_z)):
    for j in range(int(repeat_y)):
        for k in range(int(repeat_x)):
            for m in range(len(mag_atom_posi)):
                super_posi.append([])
                super_posi[count].append(float(mag_atom_posi[m][0]) + float(k)*float(base_vec[0][0]) + float(j)*float(base_vec[1][0]) + float(i)*float(base_vec[2][0]))
                super_posi[count].append(float(mag_atom_posi[m][1]) + float(k)*float(base_vec[0][1]) + float(j)*float(base_vec[1][1]) + float(i)*float(base_vec[2][1]))
                super_posi[count].append(float(mag_atom_posi[m][2]) + float(k)*float(base_vec[0][2]) + float(j)*float(base_vec[1][2]) + float(i)*float(base_vec[2][2]))
                count += 1

super_posi = np.array(super_posi)
x1 = super_posi[:,0]
y1 = super_posi[:,1]
z1 = super_posi[:,2]

# 确保图片保存文件夹存在
try:
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print("   Directory created：" + fig_path)
    else:
        print("   " + fig_path + " exists!!!")
except BaseException as msg:
    print("   新建目录失败：" + str(msg))

# 自旋文件参数
j1 = ['0','5','10','15','20']

#j1 = ['25']
j2 = ['0','100','200','300','400','500']
spirit_or_vampire = 1  # 1 for spirit

# 遍历所有文件
for i in range(len(j1)):
    for j in range(len(j2)):
        FILE2 = path_top + "B" + str(j1[i]) + "-" + "T" + str(j2[j]) + "-final.ovf"
        fig_name = fig_path + '/' + "B" + str(j1[i]) + "-" + "T" + str(j2[j]) + "-final.png"
        
        # 检查文件是否存在
        if not os.path.exists(FILE2):
            print(f"   文件不存在: {FILE2}")
            continue
            
        data2 = np.loadtxt(FILE2)
        
        if spirit_or_vampire == 1:  # spirit格式
            spin_x1 = data2[:,0]
            spin_y1 = data2[:,1]
            spin_z1 = data2[:,2]
        
        # 提取顶层的自旋信息
        up_z = np.max(z1)
        x = []
        y = []
        z = []
        spin_x = []
        spin_y = []
        spin_z = []
        
        for ii in range(len(z1)):
            if abs(z1[ii] - up_z) < 3.0:
                x.append(x1[ii])
                y.append(y1[ii])
                spin_x.append(spin_x1[ii])
                spin_y.append(spin_y1[ii])
                spin_z.append(spin_z1[ii])
        
        x = np.array(x)
        y = np.array(y)
        spin_z = np.array(spin_z)
        
        # 网格化
        nx, ny = 300, 300
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)
        zi = griddata((x, y), spin_z, (xi[None,:], yi[:,None]))
        
        # 创建图形，设置分辨率和大小
        fig = plt.figure(figsize=(4, 4), dpi=300, facecolor='white', edgecolor='white')
        ax = fig.add_subplot(111)
        
        # 设置白色背景
        fig.patch.set_facecolor('white')
        ax.patch.set_facecolor('white')
        
        # 绘制填充等高线图
        levels = np.linspace(-1.0, 1.0, 150)  # 自旋分量范围
        cf = ax.contourf(xi, yi, zi, levels, cmap='RdBu', extend='both')
        
        # 移除边框、坐标轴、刻度
        ax.set_frame_on(False)  # 移除边框
        ax.axis('off')  # 移除坐标轴
        
        # 移除所有空白边距
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        
        # 保存图片，不包含空白边距
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0, facecolor='white', edgecolor='white')
        plt.close(fig)  # 关闭图形，释放内存
        
        print(f"   已生成: {fig_name}")

print("   所有图片生成完成!")

```

(3) 筛选需要绘制的ovf文件和对应的图片。复制到同一个文件夹中

(4) 先将ovf转换为csv文件，记录自旋数据,这是一个ipynb文件，需要opencv库。

```python

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
        print("❗未找到 ovf 文件")
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
        

    print("✅ 自旋数据已生成，保存在：", output_dir)

# 执行
if __name__ == "__main__":
    main()


```

(5) 框选想要绘制的区域。

```python

import cv2
import os

# =========================
# 路径设置
# =========================
path = './'   # 原图文件夹
savepath = './resize/'

os.makedirs(savepath, exist_ok=True)

# =========================
# 找到所有 png
# =========================
png_files = sorted([f for f in os.listdir(path) if f.endswith('.png')])

if len(png_files) == 0:
    raise FileNotFoundError("当前文件夹下没有找到 .png 文件")

# =========================
# 用第一张图手动框选 ROI
# =========================
first_file = png_files[0]
first_path = os.path.join(path, first_file)

img0 = cv2.imread(first_path)
if img0 is None:
    raise ValueError(f"无法读取图片: {first_path}")

print(f"请在弹出的窗口中框选需要保留的区域：{first_file}")
print("操作说明：")
print("1. 鼠标左键拖动框选")
print("2. 按 Enter 或 Space 确认")
print("3. 按 c 取消重选")

x, y, w, h = cv2.selectROI("Select ROI", img0, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise ValueError("你没有选中有效区域，程序终止。")

print(f"选中的 ROI: x={x}, y={y}, w={w}, h={h}")
print(f"对应裁剪写法: img[{y}:{y+h}, {x}:{x+w}]")

# =========================
# 批量裁剪
# =========================
for file in png_files:
    png_name = os.path.join(path, file)
    img = cv2.imread(png_name)

    if img is None:
        print(f"跳过无法读取的文件: {file}")
        continue

    # 防止不同图片尺寸略有不同导致越界
    H, W = img.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)

    cropped = img[y0:y1, x0:x1]

    savepng_name = os.path.join(savepath, file)
    cv2.imwrite(savepng_name, cropped)
    print(f"已保存: {savepng_name}")

print('结束')

```

(6) 记住上一步输出的ROI这一行内容，计算想要绘制的区域的坐标，同时记得修改你的png和csv文件名。

```shell

ROI: x=912, y=602, w=79, h=68

```

将坐标填入下方脚本
```python

import cv2
import numpy as np

png_file = "B15-T500-final.png"
csv_file = "2026-03-05_08-36-18_Image-00_Spins_600000.csv"

# 你的 ROI
x, y, w, h = 912, 602, 79, 68

# 读图片尺寸
img = cv2.imread(png_file)
if img is None:
    raise ValueError(f"无法读取图片: {png_file}")
H, W = img.shape[:2]

# 读 csv
data = np.loadtxt(csv_file, delimiter=',', skiprows=1)

xmin, xmax = data[:, 0].min(), data[:, 0].max()
ymin, ymax = data[:, 1].min(), data[:, 1].max()
zmin, zmax = data[:, 2].min(), data[:, 2].max()

# 像素 -> 实际坐标
x_range = (
    float(xmin + (x / W) * (xmax - xmin)),
    float(xmin + ((x + w) / W) * (xmax - xmin))
)

y_range = (
    float(ymin + ((H - (y + h)) / H) * (ymax - ymin)),
    float(ymin + ((H - y) / H) * (ymax - ymin))
)

z_range = (
    float(zmin - 1.0),
    float(zmax + 1.0)
)

print("可直接填写到 filter.py 中：")
print(f"x_range={x_range},")
print(f"y_range={y_range},")
print(f"z_range={z_range},")

```

(7)转换xywh坐标为xyz坐标，即将图片中的位置转换为csv文件的位置。

```python

import cv2
import numpy as np

png_file = "B15-T500-final.png"
csv_file = "2026-03-05_08-36-18_Image-00_Spins_600000.csv"

# 你的 ROI
x, y, w, h = 912, 602, 79, 68

# 读图片尺寸
img = cv2.imread(png_file)
if img is None:
    raise ValueError(f"无法读取图片: {png_file}")
H, W = img.shape[:2]

# 读 csv
data = np.loadtxt(csv_file, delimiter=',', skiprows=1)

xmin, xmax = data[:, 0].min(), data[:, 0].max()
ymin, ymax = data[:, 1].min(), data[:, 1].max()
zmin, zmax = data[:, 2].min(), data[:, 2].max()

# 像素 -> 实际坐标
x_range = (
    float(xmin + (x / W) * (xmax - xmin)),
    float(xmin + ((x + w) / W) * (xmax - xmin))
)

y_range = (
    float(ymin + ((H - (y + h)) / H) * (ymax - ymin)),
    float(ymin + ((H - y) / H) * (ymax - ymin))
)

z_range = (
    float(zmin - 1.0),
    float(zmax + 1.0)
)

print("可直接填写到 下方cell 中：")
print(f"x_range={x_range},")
print(f"y_range={y_range},")
print(f"z_range={z_range},")

```

(8)上一个脚本的输出如下所示，填入到下一个脚本结尾的相应的xyz坐标处：

```shell

可直接填写到 filter.py 中：
x_range=(2995.7427548963037, 3255.24496400857),
y_range=(1506.399353766607, 1699.6781734408135),
z_range=(4.623772823642497, 7.254555104183036),

```
```python

import numpy as np

def filter_csv_by_xyz(file_path, x_range, y_range, z_range, output_path=None):
    """
    根据给定的 x, y, z 范围筛选 CSV 文件中的点，并保存更新后的数据，
    同时打印筛选前后 xyz 的范围。
    """
    data = np.loadtxt(file_path, delimiter=",")

    # 筛选前范围
    print("筛选前范围：")
    print(f"x: [{data[:,0].min():.6e}, {data[:,0].max():.6e}]")
    print(f"y: [{data[:,1].min():.6e}, {data[:,1].max():.6e}]")
    print(f"z: [{data[:,2].min():.6e}, {data[:,2].max():.6e}]")

    # 筛选
    mask = (
        (data[:, 0] >= x_range[0]) & (data[:, 0] <= x_range[1]) &
        (data[:, 1] >= y_range[0]) & (data[:, 1] <= y_range[1]) &
        (data[:, 2] >= z_range[0]) & (data[:, 2] <= z_range[1])
    )
    filtered_data = data[mask]

    # 筛选后范围
    if filtered_data.size == 0:
        print("警告：筛选后没有点满足条件！")
    else:
        print("筛选后范围：")
        print(f"x: [{filtered_data[:,0].min():.6e}, {filtered_data[:,0].max():.6e}]")
        print(f"y: [{filtered_data[:,1].min():.6e}, {filtered_data[:,1].max():.6e}]")
        print(f"z: [{filtered_data[:,2].min():.6e}, {filtered_data[:,2].max():.6e}]")

    # 默认覆盖原文件
    if output_path is None:
        output_path = file_path

    np.savetxt(output_path, filtered_data, delimiter=",", fmt="%.18e")
    print(f"已筛选并保存至 {output_path}，剩余 {filtered_data.shape[0]} 个点。")

    return filtered_data.shape[0]

filter_csv_by_xyz(
    "2026-03-05_08-36-18_Image-00_Spins_600000.csv",   ## 源文件名
    x_range=(2995.7427548963037, 3255.24496400857),
    y_range=(1506.399353766607, 1699.6781734408135),
    z_range=(4.623772823642497, 7.254555104183036),
    output_path="data_filtered.csv"  ## 新文件名
)



```

(9)计算切割后的斯格明子的自旋数据并形成spin texture图，观察是否是自己想要的那一个区域。

```python

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator

input_csv = "data_filtered.csv"
output_dir = "./result_fig"
output_name = "data_filtered_sz.png"

discrete_num = 500
save_dpi = 300
cmap = "bwr"

os.makedirs(output_dir, exist_ok=True)

data = np.loadtxt(input_csv, delimiter=",")
x = data[:, 0]
y = data[:, 1]
sz = data[:, 5]

xi = np.linspace(x.min(), x.max(), discrete_num)
yi = np.linspace(y.min(), y.max(), discrete_num)
XI, YI = np.meshgrid(xi, yi)

interp_sz = CloughTocher2DInterpolator(np.column_stack([x, y]), sz)
SZ = interp_sz(XI, YI)
SZ = np.clip(SZ, -1, 1)

plt.figure(figsize=(6, 6))
plt.imshow(
    SZ,
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    aspect="equal"
)
plt.axis("off")
plt.savefig(os.path.join(output_dir, output_name), dpi=save_dpi,
            bbox_inches="tight", pad_inches=0, transparent=True)
plt.close()

print("完成")

```
(10)下方是blender代码，要把blender工程文件和你的上面的输出文件放到同一个文件夹。

这个是抠图版的代码，只绘制斯格明子本身

```python

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator

input_csv = "data_filtered.csv"
output_dir = "./result_fig"
output_name = "data_filtered_sz.png"

discrete_num = 500
save_dpi = 300
cmap = "bwr"

os.makedirs(output_dir, exist_ok=True)

data = np.loadtxt(input_csv, delimiter=",")
x = data[:, 0]
y = data[:, 1]
sz = data[:, 5]

xi = np.linspace(x.min(), x.max(), discrete_num)
yi = np.linspace(y.min(), y.max(), discrete_num)
XI, YI = np.meshgrid(xi, yi)

interp_sz = CloughTocher2DInterpolator(np.column_stack([x, y]), sz)
SZ = interp_sz(XI, YI)
SZ = np.clip(SZ, -1, 1)

plt.figure(figsize=(6, 6))
plt.imshow(
    SZ,
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    aspect="equal"
)
plt.axis("off")
plt.savefig(os.path.join(output_dir, output_name), dpi=save_dpi,
            bbox_inches="tight", pad_inches=0, transparent=True)
plt.close()

print("完成")

```

这个是方形的背景

```python

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator

input_csv = "data_filtered.csv"
output_dir = "./result_fig"
output_name = "data_filtered_sz.png"

discrete_num = 500
save_dpi = 300
cmap = "bwr"

os.makedirs(output_dir, exist_ok=True)

data = np.loadtxt(input_csv, delimiter=",")
x = data[:, 0]
y = data[:, 1]
sz = data[:, 5]

xi = np.linspace(x.min(), x.max(), discrete_num)
yi = np.linspace(y.min(), y.max(), discrete_num)
XI, YI = np.meshgrid(xi, yi)

interp_sz = CloughTocher2DInterpolator(np.column_stack([x, y]), sz)
SZ = interp_sz(XI, YI)
SZ = np.clip(SZ, -1, 1)

plt.figure(figsize=(6, 6))
plt.imshow(
    SZ,
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    aspect="equal"
)
plt.axis("off")
plt.savefig(os.path.join(output_dir, output_name), dpi=save_dpi,
            bbox_inches="tight", pad_inches=0, transparent=True)
plt.close()

print("完成")

```

上面是绘制ovf到blender的全部流程，后面会记录如何自己创建一个自旋数据csv文件，并用blender绘制出来。