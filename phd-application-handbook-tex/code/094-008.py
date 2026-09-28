
import numpy as np
from numpy import *
import matplotlib as mpl
import os
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import cm
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
rc('text', usetex=False)
plt.rc('font', family='DejaVu Sans', size=20)

max_val = 2   
min_val = -3  
step = (max_val - min_val)/10
mode = 2
spin = 'True'

fn = 'bnd.pdf'
m = open("note-bnd", 'r+')
ml = m.readlines()
ml = [ml[i].strip().split() for i in range(len(ml))]
knum = int(ml[0][0])
bnum = int(ml[0][1])

base = [ml[1], ml[2], ml[3]]
base = np.array(base)
dkvec = []
for k in range(4, len(ml)-1):
    ml[k] = [float(ml[k][j]) for j in range(3)]  # kvec to float
for k in range(5, len(ml)-1): 
    dkvec.append((np.array(ml[k]) - np.array(ml[k-1])).tolist())  # distance of kvec

def r(m, n):
    global rfinal
    global r1
    global r2
    global r3
    r1 = 0
    r2 = 0
    r3 = 0
    r1 = [float(m[0]) * float(n[0][i]) for i in range(3)]
    r2 = [float(m[1]) * float(n[1][i]) for i in range(3)]
    r3 = [float(m[2]) * float(n[2][i]) for i in range(3)]
    x = r1[0] + r2[0] + r3[0]
    y = r1[1] + r2[1] + r3[1]
    z = r1[2] + r2[2] + r3[2]
    rfinal = np.sqrt(x**2 + y**2 + z**2)

sum_k = 0
k = [0 for i in range(len(ml)-6)]
for i in range(len(ml)-6):
    r(dkvec[i], base)
    k[i] = rfinal
    sum_k = k[i] + sum_k
mesh = int(ml[len(ml)-1][0])

if mode == 1:
    x = []
else:
    x = [0]
a0 = 0
for i in range(len(ml)-6):
    for j in range(mesh):
        a0 = a0 + k[i]/(mesh + 0.0)
        x.append(a0)

# 定义高对称点标签 - 这是你原始代码中的定义
klabel = ['Γ','M','K','Γ','K']

plt.figure(figsize=(6, 10))

####### spin up ###########
with open("bnd-up.dat", 'r') as file:
    lines = file.readlines()
    data_up = np.array([float(line.strip()) for line in lines])
    data_up.resize(knum, bnum)
    bands_up = np.transpose(data_up)
    
plt.ylim(min_val, max_val)
plt.xlim(0, sum_k)

# 绘制 spin up 能带（蓝色）
for i in range(bnum):
    plt.plot(x, bands_up[i], 'b', linewidth=1.5)

####### spin down ###########
if spin == 'True':
    data_dn = np.loadtxt('bnd-dn.dat')
    data_dn.resize(knum, bnum)
    bands_dn = np.transpose(data_dn)
    
    # 绘制 spin down 能带（红色）
    for i in range(bnum):
        plt.plot(x, bands_dn[i], 'r', linewidth=1.5)

    plt.plot([0, sum_k], [0, 0], 'r-.')

hline = 0
xlab_val = [0]
for i in range(len(ml)-6):
    hline = hline + k[i]
    xlab_val.append(hline)
    
hline = 0
for i in range(len(ml)-7):
    hline = hline + k[i]
    plt.plot([hline, hline], [min_val, max_val + step], 'b-.', linewidth=1)

plt.ylabel('Energy (eV)', fontsize=25)
plt.xticks(xlab_val, klabel)  
plt.tight_layout()
plt.yticks(np.arange(min_val, max_val + step, step))
ax = plt.gca()
ax.tick_params(labelsize=20)
plt.savefig(fn, dpi=300, bbox_inches='tight')
plt.show()
