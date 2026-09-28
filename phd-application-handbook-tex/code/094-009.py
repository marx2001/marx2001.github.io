
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc 
rc('text',usetex=False)
plt.rc('font', family='DejaVu Sans', size=20)

################
min_val = -2 
max_val = 1
step = (max_val - min_val)/10
fonts = 15
spin = 'both'  # 修改为 'both' 表示同时绘制两个自旋通道
###############
orbs=['px','py','pz','z2','x2','xy','yz','xz']
kp = ['Γ','M','K','Γ','K']

# 根据spin参数设置文件和输出名
if spin == 'up':
    bndfiles = ['bnd-up.dat']
    orb_prefixes = ['']  # 不加后缀
    figname = "Project_upbnd.pdf"
    colors = ['blue']  # spin up用蓝色
    
elif spin == 'dn':
    bndfiles = ['bnd-dn.dat']
    orb_prefixes = ['']  # 不加后缀
    figname = "Project_dnbnd.pdf"
    colors = ['red']  # spin down用红色
    
elif spin == 'both':  # 同时绘制两个自旋通道
    bndfiles = ['bnd-up.dat', 'bnd-dn.dat']
    orb_prefixes = ['', '']  # 轨道文件名相同
    figname = "Project_bothbnd.pdf"
    colors = ['blue', 'red']  # up蓝色，down红色
    
elif spin == 'no':
    bndfiles = ['bnd-up.dat']
    orb_prefixes = ['']
    figname = "Project_nospinbnd.pdf"
    colors = ['black']

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
sum_k=0
k=[0 for i in range(len(ml)-6)]
for i in range(len(ml)-6):
 r(dkvec[i],base)
 k[i]=rfinal
 sum_k=k[i]+sum_k
mesh=int(ml[len(ml)-1][0])
x=[0]
a0=0
for i in range(len(ml)-6):
 for j in range(mesh):
  a0=a0+k[i]/(mesh+0.0)
  x.append(a0)

plt.figure(figsize=(20,16))

def plt_pro(i, orb_name, col, lab):
    plt.subplot(2,4,i)
    
    # 设置坐标轴范围
    plt.ylim(min_val, max_val)
    plt.xlim(0, sum_k)
    
    # 绘制费米面和对称线
    plt.plot([0,sum_k],[0,0],'r-.',linewidth=1)
    hline=0
    xlab_val = [0]
    for i in range(len(ml)-6):
        hline=hline+k[i]
        xlab_val.append(hline)
    hline=0
    for i in range(len(ml)-7):
        hline=hline+k[i]
        plt.plot([hline,hline],[min_val,max_val+step],'b-.', linewidth=1)
    
    # 处理每个自旋通道
    for spin_idx, (bndfile, color) in enumerate(zip(bndfiles, colors)):
        # 读取能带数据
        a = np.loadtxt(bndfile)
        a.resize(knum, bnum)
        b = np.transpose(a)
        
        # 绘制能带线
        for j in range(bnum):
            if spin == 'both':
                # 两个自旋通道用不同的线型和颜色
                if spin_idx == 0:  # spin up
                    plt.plot(x, b[j], '--', color=color, linewidth=1, alpha=0.6)
                else:  # spin down
                    plt.plot(x, b[j], '-', color=color, linewidth=1, alpha=0.6)
            else:
                plt.plot(x, b[j], 'k--', linewidth=1)
        
        # 读取轨道投影数据
        a2 = np.loadtxt(orb_name)
        lens = len(a2) // 2
        
        if spin == 'both':
            # 对于两个自旋通道，需要选择正确的部分
            if spin_idx == 0:  # spin up
                a2_up = a2[0:lens]
                a2_up.resize(knum, bnum)
                b2_up = np.transpose(a2_up)
                for j in range(bnum):
                    plt.scatter(x, b[j], c=color, s=b2_up[j]*300, lw=0, alpha=0.5)
            else:  # spin down
                a2_dn = a2[lens:]
                a2_dn.resize(knum, bnum)
                b2_dn = np.transpose(a2_dn)
                for j in range(bnum):
                    plt.scatter(x, b[j], c=color, s=b2_dn[j]*300, lw=0, alpha=0.5, marker='^')
        else:
            # 单个自旋通道
            if spin == 'up':
                a2 = a2[0:lens]
            elif spin == 'dn':
                a2 = a2[lens:]
            
            a2.resize(knum, bnum)
            b2 = np.transpose(a2)
            for j in range(bnum):
                plt.scatter(x, b[j], c=col, s=b2[j]*300, lw=0, alpha=0.7)
    
    # 设置图例
    if spin == 'both':
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='blue', linestyle='--', lw=2, label='Spin Up'),
                          Line2D([0], [0], color='red', linestyle='-', lw=2, label='Spin Down'),
                          plt.scatter([], [], c=col, s=100, label=lab)]
        plt.legend(handles=legend_elements, frameon=False, fontsize=15, loc='upper right')
    else:
        ax = plt.scatter([], [], c=col, s=100, label=lab)
        plt.legend([ax], [lab], frameon=False, fontsize=20, loc='upper right')
    
    # 设置坐标轴
    plt.ylabel('Energy (eV)')
    plt.xticks(xlab_val, kp)
    plt.tight_layout()
    plt.yticks(np.arange(min_val, max_val+step, step))
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)

# 绘制所有轨道的投影
plt_pro(1, orbs[0], "gray", r'$p_x$')
plt_pro(2, orbs[1], "gray", r'$p_y$')
plt_pro(3, orbs[2], "gray", r'$p_z$')
plt_pro(4, orbs[3], "cyan", r'$d_{z^2}$')
plt_pro(5, orbs[4], "red", r'$d_{x^2-y^2}$')
plt_pro(6, orbs[5], "red", r'$d_{xy}$')
plt_pro(7, orbs[6], "blue", r'$d_{yz}$')
plt_pro(8, orbs[7], "blue", r'$d_{xz}$')

plt.subplots_adjust(wspace=0.25, hspace=0.1)
plt.savefig(figname, format='pdf', bbox_inches='tight', dpi=300)
plt.show()
