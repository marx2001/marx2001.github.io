---
layout: post
title: "wannierberrier的安装"
subtitle: "Experience Sharing"
background: '/img/bg-sci-note.jpg'
categories: sci-note
permalink: /sci-note_posts/20251203-wann
---


# <center>Recording​</center>


### 1. 步骤
创建虚拟环境
```shell
conda create -n wannier-sym python=3.12.12 -y

conda activate wannier-sym
```
使用清华源安装
```shell
pip install wannierberri -i https://pypi.tuna.tsinghua.edu.cn/simple

http://mirrors.aliyun.com/pypi/simple/       # 阿里云 
https://pypi.mirrors.ustc.edu.cn/simple/     # 中国科技大学  
http://pypi.douban.com/simple                # 豆瓣  
https://pypi.python.org/simple/              # Python官方
http://pypi.v2ex.com/simple/                 # v2ex 
http://pypi.mirrors.opencas.cn/simple/       # 中国科学院  
https://pypi.tuna.tsinghua.edu.cn/simple/    # 清华大学 
```

查看所需依赖的版本：

```shell

python3 --version 

```

装完环境重启一下shell面板，会更新python的版本显示。

修改并更新知乎中的代码，还需要安装FFT等库，安装很简单，就不在此一一列出了。

目前的代码是大概正确的，报错是因为晶格常数不完全对称，但是没有找到代码中调整tolerance的api，所以我用高对称性结构重新加入wannier接口拟合一遍。
第一版代码，存在报错，这是为了记录，可以及时进行版本回退

```shell

import wannierberri as wberri
import numpy as np
import matplotlib.pyplot as plt

# 读取哈密顿数据
system_Te = wberri.System_tb(tb_file='wannier90_tb.dat', berry=True)
system_Te_sym = wberri.System_tb(tb_file='wannier90_tb.dat', berry=True)


#哈密顿对称化：
system_Te_sym.symmetrize(
    # 每一个原子的位置
    positions = np.array([[0.5000000000000000, 0.0000000000000000, 0.5036182718074184],   
                    [ 0.0000000000000000, 0.5000000000000000, 0.5036182718074184],
                   [0.5000000000000000, 0.5000000000000000, 0.6102267399141962],
                   [0.5000000000000000, 0.5000000000000000, 0.3960128566905964],
                   [0.0000000000000000, 0.0000000000000000, 0.4865206618880932]]),
    # 原子的名称
    atom_name = ['Nb','Nb','Se','Se','O'],
    # 投影轨道和wannier90.win中保持一致
    proj = ['Nb:d','Se:p','O:p'],
    # 没有自旋极化或开启SOC
    soc=True,
    )

#生成k点路径
path=wberri.Path(system_Te,
    #设置k点路径的起点和终点；如果两条k点路径不连续，请在中间加None来分隔
    k_nodes=[[0,0,0],[0.5,0.0,0.0],None,
            [0.5,0.5,0.0],[0.0,0.5,0.0],None,
            [0.0,0.0,0.0]],
    labels=["G","X","M","Y","G"],
    #与k点数目成比例
    length=1000) 

#计算对称和非对称体系的能带
quantities = {"Energy":wberri.calculators.tabulate.Energy()}

calculators={}
calculators ["tabulate"] = wberri.calculators.TabulatorAll(quantities,ibands=[18,19],mode="path")

path_result_Te = wberri.run(
        system=system_Te,
        grid=path,
        calculators=calculators,
        print_Kpoints=False)

path_result_Te_sym = wberri.run(
        system=system_Te_sym,
        grid=path,
        calculators=calculators,
        print_Kpoints=False)

#读取第22,23能带的本征值
band_Te = path_result_Te.results['tabulate'].get_data(quantity='Energy',iband=(21,22))
band_Te_sym = path_result_Te_sym.results['tabulate'].get_data(quantity='Energy',iband=(21,22))

#读取k点路径数据
band_k=path.getKline()

#绘制非对称体系的能带
segments = path.get_segments()
band_k = path.getKline()

plt.figure(figsize=(6,5))
for seg in segments:
    k_seg = band_k[seg]
    E_seg = band_Te[seg]
    plt.plot(k_seg, E_seg, 'r', linewidth=2)

plt.xticks(path.getKtick(), path.getKlabel(), fontsize=11)
plt.ylabel('Energy (eV)', fontsize=12)
plt.xlabel('k-path', fontsize=12)
plt.title('Unsymmetrized bands')
plt.grid(True, alpha=0.3)
plt.savefig("unsymmetrized.png", dpi=300)
plt.show()


#绘制对称体系的能带
plt.figure(figsize=(6,5))
for seg in segments:
    k_seg = band_k[seg]
    E_seg = band_Te_sym[seg]
    plt.plot(k_seg, E_seg, 'b', linewidth=2)

plt.xticks(path.getKtick(), path.getKlabel(), fontsize=11)
plt.ylabel('Energy (eV)', fontsize=12)
plt.xlabel('k-path', fontsize=12)
plt.title('Symmetrized bands')
plt.grid(True, alpha=0.3)
plt.savefig("symmetrized.png", dpi=300)
plt.show()


```