---
layout: post
title: "利用Wannier Tools的对称化hr.dat"
subtitle: "Experience Sharing"
background: '/img/bg-sci-note.jpg'
categories: sci-note
permalink: /sci-note_posts/20251215-wt-sym
---


# <center>Tutorials-irvsp补充和指令​</center>

### 简介

确保hr.dat仍保留DFT计算中的对称性，这样能保证wannier90的计算结果是正确的

# <center>流程​</center>

1. 准备POSCAR

2. 判断空间群： 

```shell

conda activate mrx_phonopy 

```

然后 

```shell
phonopy --symmetry --tolerance 0.005
```

判断空间群

3. 复制新POSCAR，再做计算静态自洽、能带计算

# <center>Wannier Tools对称化​</center>

1. 准备输入文件locaxis.in

```python

# some times for sake of symmetry we can set local axis for each atom in wannier. if not
# set line 20 is 0. if set, line 20 is number of atoms with local axis followed with the 
# serial number of atom in wann/projection and local z-axis, x-axisthe. As for the examples
# in comments of wann.in, one may only set local axises to a subset of atoms in wannier
##################################################################
# 4
# 1 3 d f    --> local axis setted
# 2 6 s d f  --> local axis setted
# 3 2 s p    
# 4 5 d      --> local axis setted
# In this example we only set local axis to No.1, No.2 and No.4 atoms in wanneir. so we write
#-------------------------------
# 3
# 1  zx1 zy1 zz1  xx1 xy1 xz1
# 2  zx2 zy2 zz2  xx2 xy2 xz2
# 4  zx3 zy3 zz3  xx3 xy3 xz3
#--------------------------------
##################################################################
# line 20 is number of atoms with local axis followed with No. and local axis
0 
1 0.0 0.0 1.0   1.0 0.0 0.0
2 0.0 0.0 1.0   1.0 0.0 0.0
3 0.0 0.0 1.0   1.0 0.0 0.0
4 0.0 0.0 1.0   1.0 0.0 0.0


```

采用晶格对称性，设置line 20为0，即不读取原子坐标系。

准备poscar.in，内容：

```python

# crystal cell vector3: line3 a1; line 4 a2; line 5 a3. sometimes small 
# numerical errors of {a1,a2,a3} will give rise to worse symmed hamiltonian
4.1630311801153548   0.0000000000000000   0.0000000000000000
0.0000000000000000   4.1630215413893197   0.0000000000000000
0.0000000000000000   0.0000000000000000   16.6440706251267692   
# fractional axis of each exactly as in POSCAR. line 8, number of atoms. line 9
# to line 9+natom-1 is each atoms fractional axis. Mind you the errors in it.
 5 
     0.0000008564285920    0.5000001817884865    0.5036182718074187 Nb
     0.4999997757884529    0.9999988531661401    0.5036214396997352 Nb
     0.9999996457745581    0.9999999313471903    0.6102267399141965 Se
     0.9999993935379692    0.0000008878344957    0.3960128566905965 Se
     0.5000003284704491    0.5000001458636731    0.4865206618880933 O

```
复制poscar进去。

对应的轨道，准备文件wann.in


```python

# oribitals sequences are adopted as in wannier90/manuals: p = [pz,px,pz], t2g = [dxz,dyz,dxy] and 
# d = [dz2,dxz,dyz,dx2_y2,dxy], f = [fz3,fxz2,fyz2,fzx2_zy2,fxyz,fx3_3xy2,f3yx2_y3]
# more often than not atoms in wannier is just a sub set of crystal cell atoms. you should point out the
# POSCAR              wannier90.win/projections
##################################################################
# atom1               (not projected)
# atom2               wann_atom3=poscar_atom2  (orbs: s p)
# atom3               wann_atom1=poscar_atom3  (orbs: d f) 
# atom4               (not_projected)
# atom5               wann_atom4=poscar_atom5  (orbs: s) 
# atom6               wann_atom2=poscar_atom6  (orbs: s d f)
# so we have the mapping between the atoms sequence in wannier_hr.dat(or wout/spread) and the atoms in 
# poscar as wann1=poscar3, wann2=poscar6, wann3=poscar2, wann4=poscar5
# so we have to write 
#---------------------------
# 3
# 1 3 d f
# 2 6 s d f
# 3 2 s p
# 4 5 s
#---------------------------
##################################################################
# line 24 number of atoms projected in wannier and followed with the mapping and orbs
5  
T
1 1 d   
2 2 d   
3 3 p 
4 4 p 
5 5 p


```

准备wannier90.in文件，内容：

```python


kmesh_tol=0.00001
!exclude_bands:1-2
dis_win_min =-9.61
dis_win_max =6.06
dis_froz_min =-6.39
dis_froz_max =1.21
begin projections
Nb  :   d
Se : p
O : p
end projections
write_hr =.true.   #step2
write_xyz=true #step2
guiding_centres = true    #step2
bands_plot =.true.
begin kpoint_path
  G  0.000000  0.000000  0.000000   X  0.500000  0.000000  0.000000
  X  0.500000  0.000000  0.000000   M 0.500000  0.500000  0.000000
  M 0.500000  0.500000  0.000000   Y  0.000000  0.500000  0.000000
  Y  0.000000  0.500000  0.000000   G  0.000000  0.000000  0.000000
end kpoint_path

num_iter          = 1000
num_print_cycles  =   40
dis_num_iter      = 5000
dis_mix_ratio     =  0.3
# This part was generated automatically by VASP
num_bands = 80
num_wann = 38
spinors = .true.
begin unit_cell_cart
     4.1630312     0.0000000     0.0000000
     0.0000000     4.1630215     0.0000000
     0.0000000     0.0000000    16.6440706
end unit_cell_cart
begin atoms_cart
Nb       0.0000036     2.0815115     8.3822581
Nb       2.0815147     4.1630168     8.3823108
Se       4.1630297     4.1630213    10.1566570
Se       4.1630287     0.0000037     6.5912660
O        2.0815170     2.0815114     8.0976843
end atoms_cart
mp_grid =     5     5     1
begin kpoints
      0.000000000000      0.000000000000      0.000000000000
      0.200000000000      0.000000000000      0.000000000000
     -0.200000000000      0.000000000000      0.000000000000
      0.000000000000      0.200000000000      0.000000000000
      0.000000000000     -0.200000000000      0.000000000000
      0.400000000000      0.000000000000      0.000000000000
     -0.400000000000      0.000000000000      0.000000000000
      0.000000000000      0.400000000000      0.000000000000
      0.000000000000     -0.400000000000      0.000000000000
      0.200000000000      0.200000000000      0.000000000000
     -0.200000000000     -0.200000000000      0.000000000000
     -0.200000000000      0.200000000000      0.000000000000
      0.200000000000     -0.200000000000      0.000000000000
      0.400000000000      0.200000000000      0.000000000000
     -0.400000000000     -0.200000000000      0.000000000000
     -0.200000000000      0.400000000000      0.000000000000
      0.200000000000     -0.400000000000      0.000000000000
     -0.400000000000      0.200000000000      0.000000000000
      0.400000000000     -0.200000000000      0.000000000000
      0.200000000000      0.400000000000      0.000000000000
     -0.200000000000     -0.400000000000      0.000000000000
      0.400000000000      0.400000000000      0.000000000000
     -0.400000000000     -0.400000000000      0.000000000000
     -0.400000000000      0.400000000000      0.000000000000
      0.400000000000     -0.400000000000      0.000000000000
end kpoints


```

准备hr.dat文件，未对称化的。

2. 激活环境,运行程序

```shell

conda activate sym_env

```
运行程序

```shell

python /public/home/cssong/install_package/cssong/pack/wannier_tools-master/wannhr_symm/symmhr_addrptblock/symmhr_addrptblock.py

```

记得把python库和上面的symmhr_addrptblock.py放到一个文件夹内，要不然运行不了

看日志，没有报错的话，就得到了最终的hr.dat

要用wannier1.2版本。