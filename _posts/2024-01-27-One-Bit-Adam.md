---
title: "One bit Adam"
classes: # wide
mathjax: true
categories:
  - Machine Learning
---

One bit adam是一种分布式Adam优化器，可以大幅度压缩分布式通信数据量，加速训练。这个方法在微软的DeepSpeed加速库中已有应用，原始论文参考[1]

## Background

### 动机

现今的模型规模越来越大，同时也意味着优化器存储的weights和gradients数量越来越大。在分布式训练时我们会把优化器中的weights和gradients切分后存储在不同rank来节约内存，但是这会导致通信量的增加。通信量增加同时也会导致训练的收敛速度更慢，因为需要更多的时间来进行通信。由此，1-bit adam作为一种通信压缩的adam变体被提出了。

### 相关工作

1-bit adam并不是第一个通信压缩算法，SGD和Momentum这类梯度线性算法已经实现了1-bit压缩（而不损失loss收敛速度和精度）

1-bit adam将1-bit SGD和Momentum中的压缩技术用于Adam

## Algorithm



## Reference

[1] [1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed](https://arxiv.org/abs/2102.02888)
