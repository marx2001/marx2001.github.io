---
layout: post
title: "(建模流程)Materials Studio 2023 手动搭建 Cs₃V₉Te₁₃ 晶体结构"
subtitle: "Experience Sharing"
background: '/img/bg-sci-note.jpg'
categories: sci-note
permalink: /sci-note_posts/20260708-build
---

## <center>说明</center>

本文记录如何在 **Materials Studio 2023** 中，根据文献给出的空间群、晶格参数和非等价原子分数坐标，手动搭建块体材料 **Cs₃V₉Te₁₃** 的晶体结构。

核心思想是：

```text
先设置晶胞参数和空间群 → 输入非等价原子位点 → 让 Materials Studio 根据空间群自动展开完整晶胞
```

不要根据论文中的结构示意图手动画原子，也不要直接把非等价位点当成完整 POSCAR。对于 Cs₃V₉Te₁₃，文献给出的 6 个原子坐标是 **asymmetric unit / 非等价原子位点**，需要通过空间群 **P -6 2 m, No. 189** 自动展开，最终得到完整的 25 原子原胞。

---

## <center>一、需要准备的结构信息</center>

### 1. 晶格和空间群

```text
Material: Cs3V9Te13
Crystal system: Hexagonal
Space group: P -6 2 m
Space group number: 189

a = 10.1327 Å
b = 10.1327 Å
c = 8.2181 Å
alpha = 90°
beta  = 90°
gamma = 120°
```

注意：

```text
必须选择 No. 189, P -6 2 m
不要选成 P -6 m 2
```

在 Materials Studio 中，如果通过符号不好找空间群，建议直接输入空间群编号：

```text
189
```

只要软件显示的空间群编号是 **189**，就是正确的。

---

### 2. 非等价原子分数坐标

在 Materials Studio 里输入的是 **fractional coordinates / 分数坐标**，不是 Cartesian coordinates / 笛卡尔坐标。

| 位点 | Element | Name | a/x | b/y | c/z | 展开后数量 |
|---|---|---|---:|---:|---:|---:|
| Cs | Cs | Cs1 | 0.0000 | 0.3632 | 0.0000 | 3 |
| V1 | V | V1 | 0.8220 | 0.5334 | 0.5000 | 6 |
| V2 | V | V2 | 0.8469 | 0.8469 | 0.5000 | 3 |
| Te1 | Te | Te1 | 0.0000 | 0.77119 | 0.71975 | 6 |
| Te2 | Te | Te2 | 0.666667 | 0.333333 | 0.25213 | 4 |
| Te3 | Te | Te3 | 0.58986 | 0.58986 | 0.5000 | 3 |

最终完整原胞应该为：

```text
Cs: 3
V : 9
Te: 13
Total: 25 atoms
```

也就是化学式：

```text
Cs3V9Te13
```

---

## <center>二、Materials Studio 2023 详细建模步骤</center>

## Step 1：新建一个 3D Atomistic Document

打开 **Materials Studio 2023 Visualizer**。

点击顶部菜单：

```text
File → New
```

选择：

```text
3D Atomistic Document
```

新建一个空白结构窗口。

---

## Step 2：进入 Build Crystal 建晶体窗口

在顶部菜单栏点击：

```text
Build → Crystals → Build Crystal...
```

不同安装版本中，菜单名称可能略有差异，也可能显示为：

```text
Build → Build Crystal...
```

或者：

```text
Build → Crystals → Crystal Builder...
```

你要进入的是可以设置 **space group / lattice parameters / asymmetric unit atoms** 的晶体构建窗口。

---

## Step 3：设置空间群

在 **Build Crystal** 窗口中找到：

```text
Symmetry / Space Group
```

设置：

```text
Space group number = 189
```

确认软件显示类似：

```text
P -6 2 m
```

或者：

```text
P-62m
```

这里一定要注意：

```text
正确：No. 189, P -6 2 m
错误：P -6 m 2
```

如果空间群选错，后面展开出来的原子位置会完全不同。

---

## Step 4：设置晶格参数

在 **Lattice Parameters / Cell Parameters** 区域输入：

```text
a = 10.1327
b = 10.1327
c = 8.2181

alpha = 90
beta  = 90
gamma = 120
```

单位默认为 Å。

特别注意：

```text
gamma = 120°
```

不要填成 60°。虽然 60° 和 120° 有时可以表示等价的六角晶格，但为了和文献坐标以及后续 VASP POSCAR 保持一致，这里应该使用 **γ = 120°**。

---

## Step 5：打开 Add Atoms 窗口

在 **Build Crystal** 主窗口中找到添加原子的按钮，一般为：

```text
Add Atoms
```

或者在菜单中选择：

```text
Atoms → Add
```

打开后会出现 **Add Atoms** 小窗口。

这个窗口里右侧的：

```text
a:
b:
c:
```

就是原子的分数坐标，对应常见写法：

```text
x, y, z
```

也就是说：

```text
a = fractional x
b = fractional y
c = fractional z
```

不要把这里的 a、b、c 理解为晶格常数。晶格常数已经在前面的 Lattice Parameters 中设置过了。

---

## Step 6：逐个输入 6 个非等价原子位点

每输入一个位点，就点击一次窗口右下方的：

```text
Add
```

然后再修改元素和坐标，继续添加下一个位点。

---

### 1. 添加 Cs 位点

设置：

```text
Element: Cs
Name: Cs1
Oxidation state: 0
Occupancy: 1.0

a = 0.0000
b = 0.3632
c = 0.0000
```

点击：

```text
Add
```

---

### 2. 添加 V1 位点

设置：

```text
Element: V
Name: V1
Oxidation state: 0
Occupancy: 1.0

a = 0.8220
b = 0.5334
c = 0.5000
```

点击：

```text
Add
```

注意：

```text
Element 只能选 V
Name 可以写 V1
```

不要把元素名称改成 V1，因为 V1 不是化学元素，只是不等价 V 位点标签。

---

### 3. 添加 V2 位点

设置：

```text
Element: V
Name: V2
Oxidation state: 0
Occupancy: 1.0

a = 0.8469
b = 0.8469
c = 0.5000
```

点击：

```text
Add
```

---

### 4. 添加 Te1 位点

设置：

```text
Element: Te
Name: Te1
Oxidation state: 0
Occupancy: 1.0

a = 0.0000
b = 0.77119
c = 0.71975
```

点击：

```text
Add
```

---

### 5. 添加 Te2 位点

设置：

```text
Element: Te
Name: Te2
Oxidation state: 0
Occupancy: 1.0

a = 0.666667
b = 0.333333
c = 0.25213
```

点击：

```text
Add
```

---

### 6. 添加 Te3 位点

设置：

```text
Element: Te
Name: Te3
Oxidation state: 0
Occupancy: 1.0

a = 0.58986
b = 0.58986
c = 0.5000
```

点击：

```text
Add
```

---

## Step 7：应用空间群并生成完整晶胞

6 个非等价位点全部 Add 完成后，关闭 **Add Atoms** 小窗口，回到 **Build Crystal** 主窗口。

在主窗口中点击：

```text
Build
```

或者：

```text
OK
```

或者：

```text
Apply
```

不同版本按钮名称略有不同。你要找的是让软件根据当前空间群和非等价位点生成完整晶体结构的按钮。

如果主窗口中有类似选项：

```text
Apply symmetry
Generate symmetry equivalents
```

需要勾选或点击它。

因为你输入的是非等价原子位点，必须通过空间群 **P -6 2 m, No. 189** 自动展开，才能得到完整晶胞。

---

## <center>三、建完后如何检查是否正确</center>

## 1. 检查总原子数

生成结构后，查看 composition / atom count。

正确结果应该是：

```text
Cs3 V9 Te13
```

总原子数应该是：

```text
25 atoms
```

如果最终只有 6 个原子，说明没有应用空间群展开。

如果最终是 50 个、75 个或更多原子，说明可能重复应用了 symmetry，或者把完整坐标当成非等价位点再次输入了。

---

## 2. 检查 V 原子层

正确结构中，所有 V 原子的分数坐标 z 应该都是：

```text
z = 0.5000
```

这说明 V 原子集中在同一个二维层内。

如果 V 原子分布在很多不同 z 层上，需要检查空间群或坐标是否输入错误。

---

## 3. 沿 c 轴观察 V 子晶格

切换视角，沿 c 轴俯视结构。

可以尝试：

```text
View → Along Axis → c
```

如果菜单中没有该选项，也可以手动旋转到 c 轴俯视。

正确的 V 子晶格应该是一个复杂二维网络，由三角形、四边形和五边形交织构成，不是普通 kagome 晶格。

---

## 4. 侧面观察层状结构

沿 a 轴或 b 轴观察结构。

正确结构应该表现为明显层状堆叠，大致可以理解为：

```text
Cs — Te — V/Te — Te — Cs
```

其中 V 原子主要位于中间的 V-Te 层内。

---

## <center>四、常见错误与排查</center>

## 错误 1：空间群选错

正确空间群是：

```text
No. 189, P -6 2 m
```

错误情况包括：

```text
P -6 m 2
其他 P-6xx 空间群
```

解决方法：

```text
不要按符号猜，直接输入空间群编号 189。
```

---

## 错误 2：把 Add Atoms 里的 a、b、c 当成晶格常数

Add Atoms 窗口中的：

```text
a, b, c
```

是原子的分数坐标，不是晶格参数。

例如：

```text
a = 0.8220
b = 0.5334
c = 0.5000
```

表示这个 V 原子位于分数坐标：

```text
(0.8220, 0.5334, 0.5000)
```

---

## 错误 3：坐标类型用成 Cartesian

本流程中的 6 个坐标都是：

```text
Fractional coordinates
```

不是：

```text
Cartesian coordinates
```

如果软件提供坐标类型选择，一定选择 **Fractional**。

---

## 错误 4：把 V1、V2 当成元素

正确输入方式：

```text
Element: V
Name: V1
```

错误输入方式：

```text
Element: V1
```

Te1、Te2、Te3 同理。

正确方式：

```text
Element: Te
Name: Te1 / Te2 / Te3
```

---

## 错误 5：只输入 6 个原子后直接导出

6 个原子只是非等价位点，不是完整晶胞。

正确流程是：

```text
输入 6 个非等价位点 → Apply symmetry / Build → 得到 25 原子完整晶胞
```

如果没有展开，后续 VASP 计算结构一定是错的。

---

## 错误 6：gamma 填成 60°

建议使用：

```text
gamma = 120°
```

不要填成：

```text
gamma = 60°
```

这样可以保持和文献坐标以及 VASP 常用六角晶胞写法一致。

---

## <center>五、导出 POSCAR</center>

结构确认无误后，先保存 Materials Studio 文件：

```text
File → Save As
```

保存为：

```text
.xsd
```

然后导出 VASP 文件：

```text
File → Export...
```

文件类型选择：

```text
VASP POSCAR
```

或者：

```text
VASP 5
```

导出后用文本编辑器打开 POSCAR，检查元素和数量。

推荐最终 POSCAR 中的元素顺序为：

```text
Cs V Te
3 9 13
Direct
```

如果 Materials Studio 导出的元素顺序不是这个，比如：

```text
Cs Te V
```

本身不是错误，但后续写 `MAGMOM` 时必须严格按照 POSCAR 中的真实原子顺序设置。

---

## <center>六、用于磁性计算的额外记录</center>

对于后续 DFT 磁性、J 参数、DMI 计算，需要特别记录 V 原子的分组。

这个结构中有两类不等价 V 位点：

```text
V1: 展开后 6 个 V 原子
V2: 展开后 3 个 V 原子
```

文献中也区分 V1 和 V2。后续设置磁构型时，不建议简单把所有 V 完全等同处理。

如果导出 POSCAR 后原子顺序不清楚，可以在 Materials Studio 中通过坐标检查：

```text
V1 初始非等价位点: (0.8220, 0.5334, 0.5000)
V2 初始非等价位点: (0.8469, 0.8469, 0.5000)
```

所有与 V1 对称等价的 V 原子属于 V1 组，所有与 V2 对称等价的 V 原子属于 V2 组。

---

## <center>七、最简操作清单</center>

```text
1. 打开 Materials Studio 2023 Visualizer
2. File → New → 3D Atomistic Document
3. Build → Crystals → Build Crystal...
4. Space group 输入 189
5. 确认显示 P -6 2 m
6. 输入晶格参数：
   a = 10.1327
   b = 10.1327
   c = 8.2181
   alpha = 90
   beta  = 90
   gamma = 120
7. 点击 Add Atoms
8. 依次输入 6 个非等价位点：
   Cs  0.0000    0.3632    0.0000
   V   0.8220    0.5334    0.5000
   V   0.8469    0.8469    0.5000
   Te  0.0000    0.77119   0.71975
   Te  0.666667  0.333333  0.25213
   Te  0.58986   0.58986   0.5000
9. 每输入一行点击一次 Add
10. 回到 Build Crystal 主窗口
11. 点击 Apply symmetry / Build / OK
12. 检查总原子数是否为 25
13. 检查化学式是否为 Cs3V9Te13
14. 检查 9 个 V 是否都位于 z = 0.5
15. File → Export → VASP POSCAR
```

---

## <center>八、正确模型的判断标准</center>

最终模型必须同时满足：

```text
1. 空间群：P -6 2 m, No. 189
2. 晶格：a = b = 10.1327 Å, c = 8.2181 Å, gamma = 120°
3. 组成：Cs3V9Te13
4. 总原子数：25
5. V 原子数：9
6. 9 个 V 原子都在 z = 0.5 层
7. V 子晶格沿 c 轴俯视为复杂二维网络，而不是普通 kagome
```

只要这些条件都满足，这个 Materials Studio 模型就基本正确，可以用于后续导出 POSCAR 并进行 VASP 结构优化、磁性计算、J 参数或 DMI 计算。
