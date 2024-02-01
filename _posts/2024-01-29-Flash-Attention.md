---
title: "Flash Attention"
classes: # wide
mathjax: true
categories:
  - Machine Learning
---

Flash Attention是Attention的一种优化算法，在不失准确性的同时，可以大幅度减少wall-clock time，减少内存占用。核心在于两点：

- 对输入分块进行softmax计算（tiling）
- 不保留中间变量attention matrix，而是在反向的时候冲计算（recomputation）

如下图所示，GPU的中间变量可以存储在3个地方：

1. SRAM (Static Random Access Memory)，GPU的片上缓存，容量最小，成本最高，速度最快
2. HBM （high bandwidth memory），我们通常所说的GPU显存，速度介于SRAM和CPU DRAM之间
3. CPU DRAM，CPU的内存，速度最慢

![FA-fig1](/assets/images/FA-fig1.png){: .align-center}

核心的思路是避免反复在HBM进行读写，而尽可能使用SRAM来加速Attention的处理速度。

## 原始的标准Attention

![FA-algo0](/assets/images/FA-algo0.png){: .align-center}

## Flash Attention
![FA-algo1](/assets/images/FA-algo1.png){: .align-center}
