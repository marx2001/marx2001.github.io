
&TB_FILE
Hrfile = 'wannier90_hr.dat'  ! 紧束缚模型文件
Package = 'VASP'             ! 使用VASP软件包
/

&CONTROL
BulkBand_calc         = F    ! 不计算体材料能带
SlabSS_calc           = T    ! 计算表面谱函数
Z2_3D_calc            = F    ! 不计算3D Z2拓扑不变量
/

&SYSTEM
SOC = 1                 ! 自旋轨道耦合
E_FERMI = -0.6465       ! 费米能级
Numoccupied = 6         ! 占据态数目
/

&PARAMETERS
Eta_Arc = 0.001     ! 无穷小量，用于展宽
E_arc =-0.05        ! 计算费米弧的能量
OmegaMin = -1.3     ! 能量范围最小值，用于能带绘图
OmegaMax =  1.3     ! 能量范围最大值，用于能带绘图
OmegaNum = 401      ! 能量点数，能带绘图步长
Nk1 = 101           ! k点数量（奇数更佳）
Nk2 = 101           ! k点数量（奇数更佳）
Np  = 2             ! k点数量（奇数更佳）
/

SURFACE                  ! 表面定义（锯齿形方向）
 1  0  0                 ! 表面法向量
 0  0  1                 ! 第二个基矢
 0  1  0                 ! 第三个基矢

KPATH_BULK               ! 体材料k点路径
3                        ! k路径段数（仅用于体材料能带）
G 0.00000  0.00000 0.0000 M 0.50000  0.00000 0.0000  ! Γ到M点
M 0.50000  0.00000 0.0000 K 0.33333  0.33333 0.0000  ! M到K点
K 0.33333  0.33333 0.0000 G 0.00000  0.00000 0.0000  ! K到Γ点

KPATH_SLAB               ! 薄膜k点路径
2                        ! k路径段数（用于二维情况）
X  0.5  0.0 G  0.0  0.0  ! X到Γ点（二维情况k路径）
G  0.0  0.0 X  0.5  0.0  ! Γ到X点

LATTICE                  ! 晶格参数
Angstrom                 ! 单位：埃
     2.1017795    -3.6403930     0.0000000  ! 晶格矢量a
     2.1017795     3.6403930     0.0000000  ! 晶格矢量b
     0.0000000     0.0000000    15.0000000  ! 晶格矢量c

ATOM_POSITIONS           ! 原子位置
2                        ! 投影原子数目
Cartisen                 ! 笛卡尔坐标
Bi       2.1017795     1.2134643    10.7681236  ! 铋原子位置1
Bi       2.1017795    -1.2134643     9.0367958   ! 铋原子位置2

PROJECTORS               ! 投影子设置
3 3                      ! 每个原子的投影轨道数
Bi pz px py              ! 铋原子1的轨道：pz, px, py
Bi pz px py              ! 铋原子2的轨道：pz, px, py

!WANNIER_CENTRES         ! 从wannier90.wout复制
!Cartesian               ! 笛卡尔坐标
