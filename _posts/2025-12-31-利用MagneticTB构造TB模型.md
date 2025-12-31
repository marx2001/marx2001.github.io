---
layout: post
title: "利用MagneticTB构造TB模型"
subtitle: "Experience Sharing"
background: '/img/bg-sci-note.jpg'
categories: sci-note
permalink: /sci-note_posts/20251230-Mag
---

## <center>代码如下</center>

　

```shell

Needs["MagneticTB`"]
msgop[bnsdict[{99, 165}]]
sgop = msgop[bnsdict[{99, 165}]]
init[lattice -> {{a, 0, 0}, {0, a, 0}, {0, 0, c}}, 
  lattpar -> {a -> 4.1630263607523368, c -> 16.6440706251267727}, 
  wyckoffposition -> {{{0.0, 0.5, 0.503}, {0, 0, 
      1}}, {{0.4999997757884529, 0.999, 0.5036214396997352}, {0, 
      0, -1}}, {{0.999, 0.999, 0.610}, {0, 0, 0}}, {{0.999, 0.0, 
      0.396}, {0, 0, 0}}, {{0.5, 0.5, 0.486}, {0, 0, 0}}}, 
  symminformation -> sgop, 
  basisFunctions -> {{"dxydn", "dyzdn", "dx2-y2dn", "dz2dn", "dz2up", 
     "dx2-y2up", "dxzup", "dxyup", "pxup", "pyup", "pxdn", "pydn"}}];
hamsoc = Sum[symham[i], {i, 3}];
MatrixForm[hamsoc]
```