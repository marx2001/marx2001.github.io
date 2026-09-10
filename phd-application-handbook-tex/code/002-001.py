你的工作目录/
├── bands/                    # 能带计算目录 (当前运行脚本的目录)
│   ├── KPOINTS              # 能带路径文件 (包含高对称点)
│   ├── POSCAR -> ../POSCAR  # [推荐] 链接到结构文件
│   ├── POTCAR -> ../POTCAR  # [推荐] 链接到赝势文件
│   ├── INCAR                # 能带计算INCAR (ICHARG=11)
│   ├── vasprun.xml          # 能带计算输出
│   ├── PROCAR               # 能带计算投影输出
│   └── plot_fatbands.py     # 脚本文件 (在这里运行)
│
├── dos/                     # DOS计算目录
│   ├── KPOINTS             # DOS K点文件 (通常更密集)
│   ├── POSCAR -> ../POSCAR
│   ├── POTCAR -> ../POTCAR
│   ├── INCAR               # DOS计算INCAR (LORBIT=11)
│   └── vasprun.xml         # DOS计算输出
│
├── POSCAR                  # 晶体结构文件
└── POTCAR                  # 赝势文件 (由所有原子的POTCAR拼接)
