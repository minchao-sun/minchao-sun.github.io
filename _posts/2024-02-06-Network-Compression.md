---
title: "Network Compression"
classes: #wide
mathjax: true
categories:
  - Machine Learning
---

如今模型参数量越来越多，但是在edge device或者推理的时候没有那么多的算力或是内存怎么办呢，这里介绍模型压缩的技术来帮你用更小的模型达到相近的模型表现。

模型压缩大体上有这么些方式：

* Network Pruning
* Knowledge Distillation
* Parameter Quantization

这些方法通常都需要你首先有一个训练好的大模型teacher model，然后基于这个比较大的teacher model重新训练一个相对小的student model。

看到这里你可能会有这么几个疑问：

Q1：那么为什么不直接用小模型训练呢？

> A：因为从实际经验来说，从大模型裁剪或压缩成的小模型表现要更好，直接用小模型训练，可能会得不到可用的结果。目前没有严格的数学证明，但是有一些相关的假说讨论这一点：比如lottery ticket hypothesis

Q：那即便用模型压缩我也要训练一个大模型，为什么我不直接用大模型呢，压缩有什么用？

> A：第一，在手机或者某些算力没有那么大的设备上，我们不得不用小的模型，不然跑不起来；第二，通过Knowledge distillation等方法得到的小模型有时候还能获得更好的表现。压缩之后的模型通常是用于推理场景。

## Network Pruning

Networks are typically over-parameterized. 

网络中有很多的冗余的weights或neurons，因此可以对network进行剪枝。

## Knowledge Distillation

Student model把teacher model中去掉一部分layer得到的，因此也有文章把这种方法叫做layer reduction。

## Parameter Quantization

参数量化指的是可以将参数由浮点数(FP32/FP16等等)转换为整形(INT8等)，这是因为推理等场景下不需要非常高的精度也可以达成不错的效果，同时可以加速计算的速度。
