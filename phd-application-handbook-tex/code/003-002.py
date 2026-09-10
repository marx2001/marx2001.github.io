SYSTEM =  BrSeCr
LREAL= Auto
ALGO= Normal

ISTART = 1
ICHARG = 11 #=11用于电子能带和态密度计算，此过程中电荷密度不变

ENCUT =520
NCORE= 4

ISMEAR = 0 
SIGMA = 0.04
GGA=PE

ISPIN=2
MAGMOM = 0 4 8*0

#LSORBIT=.TRUE.
#LORBMOM=.TRUE.

LWAVE=.F; 
LCHARG=.F
EDIFF = 1E-6
EDIFFG = -0.01
ISYM = 0
VOSKOWN=1
GGA_COMPAT =.F

#NBANDS=50
#NEDOS=301
LORBIT=11         #常用=11，输出包含IM投影的PROCAR和DOSCAR
LDAU=.T
LDAUTYPE=2
LDAUL=2 2 -1 -1                  #库伦排斥的轨道，对应元素
LDAUU=0.5 2.0 0.0 0.0        #几个元素几个数
LDAUJ=0.0 0.0 0.0           #stoner交换参数大小
LMAXMIX=4
