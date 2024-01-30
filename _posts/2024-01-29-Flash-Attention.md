---
title: "Flash Attention"
classes: # wide
mathjax: true
categories:
  - Machine Learning
---

Flash Attention是Attention的一种优化算法，在不失正确性的同时，可以大幅度减少wall-clock time，减少内存占用。核心在于两点：

- 对输入分块进行softmax计算（tiling）
- 不保留中间变量attention matrix，而是在反向的时候冲计算（recomputation）

## 原始的标准Attention
![FA-algo0](/assets/images/FA-algo0.png){: .align-center}

## Flash Attention
![FA-algo1](/assets/images/FA-algo1.png){: .align-center}
