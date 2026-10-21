---
background: /img/bg-sci-log.jpg
categories: learning
layout: post
permalink: /learning/20261020-quantum-framework
subtitle: Experience Sharing
title: "20261020+量子力学的初步框架：薛定谔方程、态矢量、表象与算符"
math: true
last_modified_at: 2026-10-20
---

# 量子力学的初步框架：从薛定谔方程到态矢量、波函数与算符

学习量子力学时，可以先建立一条主线：**确定物理体系与哈密顿算符，通过薛定谔方程描述量子态的演化，选择表象写出波函数，再用物理量算符研究该量子态的性质。**

这里重点理解各概念的**物理角色及其相互关系**，而不是具体求解某一个势函数下的微分方程。

---

## 一、整体认识：量子力学主要在做什么？

对于一个给定的量子体系，可以按照下面的思路研究：

1. **建立模型**：明确粒子的质量、所处势场及其他相互作用，构造哈密顿算符 <span class="math-inline" markdown="0">&#92;(&#92;hat H&#92;)</span>。
2. **确定量子态**：结合初始条件和适当的边界条件，求解薛定谔方程，得到随时间演化的量子态。
3. **选择表象**：用位置、动量或能量本征态等不同基底，表示同一个量子态。
4. **研究物理性质**：利用位置、动量、能量等算符，计算测量概率、期望值等物理信息。

可以概括为：

<div class="math-display" markdown="0">
\[
\boxed{
\text{物理体系}\ \longrightarrow\ \hat H\ \longrightarrow\
|\psi(t)\rangle\ \longrightarrow\
\text{选取表象}\ \longrightarrow\
\text{计算物理量}
}
\]
</div>

**注意：** 这是便于学习的一条思路，不是严格的先后顺序。算符不仅在最后使用，哈密顿算符从建立方程时就已经出现。

---

## 二、薛定谔方程：量子态如何随时间演化？

量子力学的时间依赖薛定谔方程可以写成抽象态矢量形式：

<div class="math-display" markdown="0">
\[
\boxed{i\hbar\frac{d}{dt}|\psi(t)\rangle=\hat H|\psi(t)\rangle}
\]
</div>

其中：

- <span class="math-inline" markdown="0">&#92;(&#92;lvert &#92;psi(t)&#92;rangle&#92;)</span>：时刻 <span class="math-inline" markdown="0">&#92;(t&#92;)</span> 的量子态（态矢量）。
- <span class="math-inline" markdown="0">&#92;(&#92;hat H&#92;)</span>：哈密顿算符，描述体系的能量并决定量子态的时间演化。
- <span class="math-inline" markdown="0">&#92;(&#92;hbar&#92;)</span>：约化普朗克常数。

例如，对一维非相对论单粒子，在位置表象中，若势函数为 <span class="math-inline" markdown="0">&#92;(V(x,t)&#92;)</span>，则：

<div class="math-display" markdown="0">
\[
\hat H=-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2}+V(x,t)
\]
</div>

因此，薛定谔方程具体写为：

<div class="math-display" markdown="0">
\[
\boxed{
i\hbar\frac{\partial\psi(x,t)}{\partial t}
=-\frac{\hbar^2}{2m}\frac{\partial^2\psi(x,t)}{\partial x^2}
+V(x,t)\psi(x,t)
}
\]
</div>

**理解：** 选定势函数相当于明确方程中的一个关键物理条件；再结合初始条件及适当的边界条件，就可以求出量子态的演化规律。这里描述的是**波函数如何演化**，而不是经典力学意义上的确定运动轨迹。

---

## 三、态矢量与波函数：同一个量子态的两种表达层次

**态矢量** <span class="math-inline" markdown="0">&#92;(&#92;lvert &#92;psi&#92;rangle&#92;)</span> 是量子态的抽象数学描述，本身不依赖于选择哪一组基底。

**波函数** <span class="math-inline" markdown="0">&#92;(&#92;psi(x)&#92;)</span> 是该态矢量在**位置表象**下的表示：

<div class="math-display" markdown="0">
\[
\boxed{\psi(x)=\langle x|\psi\rangle}
\]
</div>

因此，**态矢量与波函数描述的是同一个量子态，但严格来说，波函数是态矢量在特定表象下的分量函数，并不等同于抽象态矢量本身。**

### 1. 为什么函数可以类比为列向量？

将波函数在若干位置 <span class="math-inline" markdown="0">&#92;(x_1,x_2,&#92;ldots,x_N&#92;)</span> 处取值，可得到：

<div class="math-display" markdown="0">
\[
\psi(x)\quad\longrightarrow\quad
\begin{pmatrix}
\psi(x_1)\\
\psi(x_2)\\
\vdots\\
\psi(x_N)
\end{pmatrix}
\]
</div>

- 整个 <span class="math-inline" markdown="0">&#92;(&#92;psi(x)&#92;)</span> 是位置表象下的**波函数**。
- 每个 <span class="math-inline" markdown="0">&#92;(&#92;psi(x_i)&#92;)</span> 是在位置 <span class="math-inline" markdown="0">&#92;(x_i&#92;)</span> 处的**概率幅**，而不是不同的“方程解”。
- 这个列向量是连续波函数的一种**离散取样示意**。在严格的离散数值表示中，还需要考虑基底的归一化、网格间距等因素。

对于归一化的一维位置波函数，<span class="math-inline" markdown="0">&#92;(&#92;lvert&#92;psi(x)&#92;rvert^2&#92;)</span> 是位置的概率密度，而非粒子在 <span class="math-inline" markdown="0">&#92;(x&#92;)</span> 处的一条确定轨迹。

### 2. 态矢量为什么能够线性叠加？

在一组完整的离散正交归一基底 <span class="math-inline" markdown="0">&#92;(&#92;{&#92;lvert n&#92;rangle&#92;}&#92;)</span> 下：

<div class="math-display" markdown="0">
\[
\boxed{|\psi\rangle=\sum_n c_n|n\rangle,
\qquad c_n=\langle n|\psi\rangle}
\]
</div>

这里的 <span class="math-inline" markdown="0">&#92;(&#92;lvert n&#92;rangle&#92;)</span> 是基矢，<span class="math-inline" markdown="0">&#92;(c_n&#92;)</span> 是对应的展开系数。把这些系数排列起来，就是态矢量在这组基底下的列向量表示。

**关键区别：** 态矢量能够展开为多个基矢的线性组合，不代表列向量中的每个元素都是一个完整波函数。

---

## 四、表象理论：同一个量子态，可以怎样表示？

不同表象的核心区别在于：**选择的基底不同，而不是量子态本身不同。**

| 表象 | 所选基底 | 态矢量对应的分量 |
| --- | --- | --- |
| 位置表象 | 位置本征态 <span class="math-inline" markdown="0">&#92;(&#92;lvert x&#92;rangle&#92;)</span> | <span class="math-inline" markdown="0">&#92;(&#92;psi(x)=&#92;langle x&#92;vert &#92;psi&#92;rangle&#92;)</span> |
| 动量表象 | 动量本征态 <span class="math-inline" markdown="0">&#92;(&#92;lvert p&#92;rangle&#92;)</span> | <span class="math-inline" markdown="0">&#92;(&#92;phi(p)=&#92;langle p&#92;vert &#92;psi&#92;rangle&#92;)</span> |
| 能量表象 | 哈密顿算符的本征态 <span class="math-inline" markdown="0">&#92;(&#92;lvert n&#92;rangle&#92;)</span> | <span class="math-inline" markdown="0">&#92;(c_n=&#92;langle n&#92;vert &#92;psi&#92;rangle&#92;)</span> |

位置表象体现量子态在位置基底上的概率幅分布；能量表象体现它在各个能量本征态上的展开系数。

### 位置表象如何转换为能量表象？

设 <span class="math-inline" markdown="0">&#92;(&#92;hat H&#92;lvert n&#92;rangle=E_n&#92;lvert n&#92;rangle&#92;)</span>，并令 <span class="math-inline" markdown="0">&#92;(u_n(x)=&#92;langle x&#92;vert n&#92;rangle&#92;)</span>，则在离散、完备、正交归一的能量本征态基底下：

<div class="math-display" markdown="0">
\[
\boxed{
\begin{aligned}
c_n&=\int u_n^*(x)\psi(x)\,dx,\\[4pt]
\psi(x)&=\sum_n c_n u_n(x).
\end{aligned}
}
\]
</div>

这两种表示能够互相转换。若体系还包含连续能谱，应将相应的求和推广为积分。

**结论：** 表象变换的本质是**更换基底、改变展开系数**；它不会因为换了一种数学表示就改变物理量子态。

---

## 五、算符：在线性代数与量子力学之间建立联系

在线性代数中，矩阵对向量进行线性变换：

<div class="math-display" markdown="0">
\[
A\mathbf v=\mathbf w
\]
</div>

量子力学中的线性算符具有类似的数学结构：

<div class="math-display" markdown="0">
\[
\boxed{\hat A|\psi\rangle=|\phi\rangle}
\]
</div>

**抽象算符在选定基底后，可以用矩阵表示。** 但普通的算符作用与表象变换不能混为一谈：

- **算符作用**：产生新的数学向量，例如 <span class="math-inline" markdown="0">&#92;(&#92;hat A&#92;lvert &#92;psi&#92;rangle&#92;)</span>。这不一定意味着实际发生了一次物理过程。
- **表象变换**：对同一个抽象量子态，换用不同基底表示；向量的坐标改变，但物理态不变。

### 1. 在求解量子态时使用算符

哈密顿算符的本征方程为：

<div class="math-display" markdown="0">
\[
\boxed{\hat H|n\rangle=E_n|n\rangle}
\]
</div>

它对应线性代数中的 <span class="math-inline" markdown="0">&#92;(A&#92;mathbf v=&#92;lambda&#92;mathbf v&#92;)</span>。求解这一方程，可以找到允许的能量本征值及对应本征态。它们也可用作进一步求解薛定谔方程的基底。

### 2. 在已知量子态后使用算符

若已知归一化的波函数 <span class="math-inline" markdown="0">&#92;(&#92;psi(x)&#92;)</span>，可以通过物理量算符计算期望值。例如，在一维位置表象下：

**位置期望值：**

<div class="math-display" markdown="0">
\[
\boxed{\langle x\rangle=\int\psi^*(x)\,x\,\psi(x)\,dx}
\]
</div>

**动量期望值：**

<div class="math-display" markdown="0">
\[
\boxed{
\langle p\rangle
=\int\psi^*(x)\left(-i\hbar\frac{d}{dx}\right)\psi(x)\,dx
}
\]
</div>

这些期望值是在相同制备条件下重复测量的统计平均值，**不一定等于某一次测量的结果**。

需要注意：算符直接作用于波函数，通常得到的仍是一个函数；要获得期望值这样的数值，还需要与态进行内积等运算。

### 3. 算符是在解方程前使用，还是解方程后使用？

**两个阶段都使用。** 哈密顿算符参与建立和求解薛定谔方程；位置、动量、能量等算符又能用来研究已知态的物理性质。算符还用于描述时间演化与对称变换，因此不存在“只用于解方程”或“只用于计算物理量”的划分。

---

## 六、把各部分连接起来：从能量本征态到波函数

对于不显含时间的哈密顿算符，假设其能量本征态构成一组完整的离散基底：

<div class="math-display" markdown="0">
\[
\hat H|n\rangle=E_n|n\rangle.
\]
</div>

确定初态后，其随时间演化的形式为：

<div class="math-display" markdown="0">
\[
\boxed{
|\psi(t)\rangle=\sum_n c_n(0)e^{-iE_nt/\hbar}|n\rangle
}
\]
</div>

选择位置表象，同一个量子态可以写成：

<div class="math-display" markdown="0">
\[
\boxed{
\psi(x,t)=\sum_n c_n(0)e^{-iE_nt/\hbar}u_n(x)
}
\]
</div>

其中 <span class="math-inline" markdown="0">&#92;(u_n(x)=&#92;langle x&#92;vert n&#92;rangle&#92;)</span>。这两式不是两个不同的物理解，而是**同一个随时间演化的量子态的抽象表示与位置表象表示**。

---

## 七、容易混淆的几个概念

| 容易产生的理解 | 更准确的理解 |
| --- | --- |
| 波函数就是态矢量本身 | 波函数是抽象态矢量在位置表象下的表示 |
| 列向量每一项都是一个波函数 | 每一项是所选基底下的一个分量；位置取样时是 <span class="math-inline" markdown="0">&#92;(&#92;psi(x_i)&#92;)</span> |
| 波函数给出粒子的确定运动轨迹 | 波函数提供测量结果的概率幅信息，其演化由薛定谔方程决定 |
| 表象变换就是算符作用产生新态 | 表象变换保持抽象量子态不变，仅改变其表示 |
| 算符作用后就直接得到物理量的数值 | 一般仍得到向量；期望值需要进行内积等运算 |
| 算符只用于求解波函数 | 算符也用于研究可观测量、时间演化与对称性 |

---

## 八、总结：四个核心概念的分工

- **薛定谔方程：** 描述量子态如何随时间演化。
- **态矢量：** 抽象地描述量子态。
- **表象理论：** 在不同基底下表示同一个量子态；波函数是位置表象下的表示。
- **算符：** 描述物理可观测量及其数学作用，其中哈密顿算符还决定时间演化。

用一句话总结：**建立哈密顿算符，利用薛定谔方程研究量子态；选择合适的表象写出波函数，再通过算符提取可测量的物理信息。**

这就是当前阶段理解量子力学的基本框架。后续学习的许多内容，是在此基础上研究具体势场、可观测量、近似方法以及更一般的量子体系。

<!-- 与线性代数笔记一致的数学公式渲染配置 -->
<style>
.math-display { overflow-x: auto; overflow-y: hidden; margin: 1.1em 0; }
</style>
<script>
(function () {
  if (window.MathJax) return;
  window.MathJax = {
    tex: {
      inlineMath: [['\\(', '\\)']],
      displayMath: [['\\[', '\\]']],
      processEscapes: true
    }
  };
  var script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
  script.async = true;
  document.head.appendChild(script);
})();
</script>
