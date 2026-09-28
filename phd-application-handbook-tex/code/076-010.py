# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc 
rc('text',usetex=False)
plt.rc('font', family='DejaVu Sans', size=20)

################
min=-2 
max=1
step = (max - min)/10
fonts = 15
###############
orbs=['px','py','pz','z2','x2','xy','yz','xz']
bndfile_up = 'bnd-up.dat'
bndfile_dn = 'bnd-dn.dat'
figname = "Project_both_spins.pdf"

# 设置颜色
col='cyan'
kp = ['Γ','M','K','Γ','K']

# 读取k点信息
m=open("note-bnd",'r+')
ml=m.readlines()
ml=[ml[i].strip().split() for i in range(len(ml))]
knum=int(ml[0][0])
bnum=int(ml[0][1])

base=[ml[1],ml[2],ml[3]]
base=np.array(base)
dkvec=[]
for k in range(4,len(ml)-1):
 ml[k]=[float(ml[k][j]) for j in range(3)]
for k in range(5,len(ml)-1): 
 dkvec.append((np.array(ml[k])-np.array(ml[k-1])).tolist())

def r(m,n):
 global rfinal
 global r1
 global r2
 global r3
 r1=0
 r2=0
 r3=0
 r1=[float(m[0])*float(n[0][i]) for i in range(3)]
 r2=[float(m[1])*float(n[1][i]) for i in range(3)]
 r3=[float(m[2])*float(n[2][i]) for i in range(3)]
 x=r1[0]+r2[0]+r3[0]
 y=r1[1]+r2[1]+r3[1]
 z=r1[2]+r2[2]+r3[2]
 rfinal=np.sqrt(x**2+y**2+z**2)

sum=0
k=[0 for i in range(len(ml)-6)]
for i in range(len(ml)-6):
 r(dkvec[i],base)
 k[i]=rfinal
 sum=k[i]+sum

mesh=int(ml[len(ml)-1][0])
x=[0]
a0=0
for i in range(len(ml)-6):
 for j in range(mesh):
  a0=a0+k[i]/(mesh+0.0)
  x.append(a0)

plt.figure(figsize=(20, 20))

def plt_pro(row_offset, i, orb, col, lab, spin):
    """绘制单个轨道的投影能带
    
    Parameters:
    -----------
    row_offset : int
        行偏移量，0表示spin up，4表示spin down
    i : int
        轨道索引 (1-8)
    orb : str
        轨道数据文件名
    col : str
        颜色
    lab : str
        轨道标签
    spin : str
        自旋类型 'up' 或 'dn'
    """
    # 计算子图位置：2行4列，加上行偏移
    plt.subplot(8, 4, row_offset * 4 + i)
    
    # 选择对应的能带文件
    if spin == 'up':
        bndfile = bndfile_up
        spin_color = 'red'  # spin up用红色
        spin_label = '(↑)'
    else:
        bndfile = bndfile_dn
        spin_color = 'blue'  # spin down用蓝色
        spin_label = '(↓)'
    
    # 读取能带数据
    a = np.loadtxt(bndfile)
    a.resize(knum, bnum)
    b = np.transpose(a)
    
    # 设置坐标范围
    plt.ylim(min, max)
    plt.xlim(0, sum)
    
    # 绘制费米能级线
    plt.plot([0, sum], [0, 0], 'r-.', linewidth=1)
    
    # 绘制高对称点竖线
    hline = 0
    xlab_val = [0]
    for j in range(len(ml)-6):
        hline += k[j]
        xlab_val.append(hline)
    
    hline = 0
    for j in range(len(ml)-7):
        hline += k[j]
        plt.plot([hline, hline], [min, max+step], 'b-.', linewidth=1)
    
    # 绘制能带
    for j in range(bnum):
        plt.plot(x, b[j], 'k--', linewidth=1)
    
    # 读取轨道投影数据
    a2 = np.loadtxt(orb)
    lens = len(a2)//2
    
    if spin == 'up':
        a2 = a2[0:lens]  # 前半部分是spin up
    else:
        a2 = a2[lens:]   # 后半部分是spin down
    
    a2.resize(knum, bnum)
    b2 = np.transpose(a2)
    
    # 绘制轨道投影散点
    for j in range(bnum):
        # 能带线
        plt.plot(x, b[j], 'k--', linewidth=0.5, alpha=0.3)
        # 轨道投影散点
        ax = plt.scatter(x, b[j], c=col, s=b2[j]*300, lw=0, alpha=0.7)
    
    # 设置标签
    if row_offset == 0:  # spin up行
        if i == 1:  # 第一列添加y轴标签
            plt.ylabel('Energy (eV)\nSpin Up')
        plt.title(f'{lab} {spin_label}', fontsize=16)
    else:  # spin down行
        if i == 1:  # 第一列添加y轴标签
            plt.ylabel('Energy (eV)\nSpin Down')
        plt.title(f'{lab} {spin_label}', fontsize=16)
    
    # 设置x轴刻度
    if row_offset == 7:  # 最后一行显示x轴标签
        plt.xticks(xlab_val, kp)
        plt.xlabel('k-path')
    else:
        plt.xticks(xlab_val, [])
    
    # 设置y轴刻度
    if i == 1:  # 第一列显示y轴刻度
        plt.yticks(np.arange(min, max+step, step))
    else:
        plt.yticks([])
    
    plt.tight_layout()
    ax = plt.gca()

# 绘制spin up的8个轨道
spin_up_orbs = orbs
for i, (orb, col_orb, lab) in enumerate(zip(orbs, 
                                            ["gray", "gray", "gray", "cyan", "red", "red", "blue", "blue"],
                                            [r'$p_x$', r'$p_y$', r'$p_z$', r'$d_{z^2}$', 
                                             r'$d_{x^2-y^2}$', r'$d_{xy}$', r'$d_{yz}$', r'$d_{xz}$']), 1):
    plt_pro(0, i, orb, col_orb, lab, 'up')

# 绘制spin down的8个轨道
spin_dn_orbs = orbs
for i, (orb, col_orb, lab) in enumerate(zip(orbs,
                                            ["gray", "gray", "gray", "cyan", "red", "red", "blue", "blue"],
                                            [r'$p_x$', r'$p_y$', r'$p_z$', r'$d_{z^2}$',
                                             r'$d_{x^2-y^2}$', r'$d_{xy}$', r'$d_{yz}$', r'$d_{xz}$']), 1):
    plt_pro(4, i, orb, col_orb, lab, 'dn')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1, wspace=0.1)
plt.savefig(figname, format='pdf', bbox_inches='tight', dpi=300)
plt.show()
