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
