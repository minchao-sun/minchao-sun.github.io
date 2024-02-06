---
title: "Network Compression"
classes: #wide
mathjax: true
categories:
  - Machine Learning
---

如今模型参数量越来越多，但是在edge device或者推理的时候没有那么多的算力或是内存怎么办呢，这里介绍模型压缩的技术。

模型压缩大体上有这么些方式：

* Network Pruning
* Knowledge Distillation
* Parameter Quantization

## Network Pruning

Networks are typically over-parameterized. 

网络中有很多的冗余的weights或neurons，因此可以对network进行剪枝。

## Knowledge Distillation

Knowledge Distillation或者知识蒸馏是先训练一个大网络作为teacher model，然后基于这个teacher model去训练student model。

Student model可能是把teacher model中去掉一部分layer得到的，因此也有文章把这种方法叫做layer reduction。

## Parameter Quantization

参数量化指的是可以将参数由浮点数(FP32/FP16等等)转换为整形(INT8等)，这是因为推理等场景下不需要非常高的精度也可以达成不错的效果，同时可以加速计算的速度。
