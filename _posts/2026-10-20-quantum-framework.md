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

# 量子力学的初步框架：薛定谔方程、态矢量、表象与算符

我对量子力学的初步理解是：**确定具体的物理体系和势函数，通过薛定谔方程研究量子态的演化；用态矢量描述量子态，用不同表象表示同一个量子态，再利用算符研究它的物理性质。**

---

## 一、薛定谔方程：求解量子态的演化

薛定谔方程已经给出来了，确定具体势函数后，再结合初始条件和必要的边界条件，就可以求解量子态随时间的演化规律。

<div class="math-display" markdown="0">
\[
i\hbar\frac{d}{dt}|\psi(t)\rangle=\hat H|\psi(t)\rangle
\]
</div>

这里求解的是**量子态或波函数随时间如何变化**，而不是经典力学中粒子的确定运动轨迹。哈密顿算符决定量子态的时间演化。

## 二、态矢量与波函数：同一个量子态的不同表达

态矢量可以类比为包含全部状态信息的抽象向量。**波函数不是态矢量的另一种解，而是态矢量在位置表象下的具体表示。** 二者描述的是同一个量子态。

<div class="math-display" markdown="0">
\[
\psi(x)=\langle x|\psi\rangle
\]
</div>

从函数的角度理解，把一段连续波函数在若干位置取值，这些函数值可以组成列向量：

<div class="math-display" markdown="0">
\[
\psi(x)\ \longrightarrow\
\begin{pmatrix}
\psi(x_0)\\
\psi(x_1)\\
\vdots\\
\psi(x_n)
\end{pmatrix}
\]
</div>

**列向量的每一项是波函数在对应位置上的分量（概率幅），不是一个独立的波函数或一个不同的解。** 这只是连续波函数的离散取样示意。

## 三、表象理论：同一个量子态，更换描述角度

位置表象，就是从**位置基底**的角度描述量子态；能量表象，就是从**能量本征态基底**的角度描述同一个量子态。

<div class="math-display" markdown="0">
\[
\underbrace{\psi(x)=\langle x|\psi\rangle}_{\text{位置表象}}
\qquad
\underbrace{c_n=\langle E_n|\psi\rangle}_{\text{能量表象}}
\]
</div>

**不同表象描述的是同一个量子态，只不过选择的基矢不同，数学表示也不同。** 从位置表象转换到能量表象，本质是更换基底，并不改变原来的量子态。

表象变换与一般的算符作用需要区分：**表象变换改变坐标表示；算符作用于态矢量，通常得到另一个向量。**

## 四、算符：既用于求解，也用于研究物理性质

可以把算符类比为线性代数中的矩阵变换。更准确地说，**算符是作用于态矢量的数学对象，在选定基底后可以表示为矩阵。**

<div class="math-display" markdown="0">
\[
\hat A|\psi\rangle=|\phi\rangle
\]
</div>

算符既可以在得到态矢量之前使用，也可以在得到态矢量之后使用：

- **求解量子态时**：哈密顿算符用于建立薛定谔方程，求解能量本征值、本征态及其演化。
- **已知量子态后**：物理量算符用于计算位置、动量、能量等可观测量的测量概率、期望值等信息。

算符直接作用于态矢量，通常得到的仍是向量；若要计算归一化量子态下某个物理量的**期望值**，则需要：

<div class="math-display" markdown="0">
\[
\langle A\rangle=\langle\psi|\hat A|\psi\rangle
\]
</div>

因此，**算符不只是求解薛定谔方程的方法，它更根本的作用是描述物理量及其与量子态之间的数学联系。**

## 五、我的整体认识

**确定物理体系与势函数 → 构造哈密顿算符、求解薛定谔方程 → 得到量子态 → 选择基底写出波函数或其他表象的展开系数 → 利用算符研究物理性质。**

这是便于理解量子力学的逻辑框架，而非严格的先后操作顺序。**态矢量描述量子态，表象决定如何表示它，算符描述物理量及其作用，薛定谔方程决定量子态随时间的演化。**

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
