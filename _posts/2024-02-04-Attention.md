---
title: "Self Attention和Multi-head Attention"
classes: #wide
mathjax: true
tag:
  - LLM
categories:
  - Machine Learning
---

随着Transformer架构的横空出世，Attention注意力机制开始被广为人知，下面我们来仔细探究Attention的机制和实现，同时分析一下Transformer中的多头注意力和标准的注意力有何区别。

![transformer-arch](/assets/images/transformer-arch.png){: .align-center}

## Self attention 自注意力

### attention matrix的计算及其含义

我们先来看计算公式
$$
\text{Attention}(Q,K,V)=\text{softmax}(QK^T)V
$$

公式中的Q代表query，K代表key，V代表Value，QKV是三个大小相同的矩阵$n\times d_{model}$，其中n是输入序列的长度，$d_{model}$是隐藏层大小，是一个超参数，原始论文中为512。得到的输出也是一个$n\times d_{model}$​​的矩阵，我们称之为Attention matrix



### 什么是“Self” Attention

Self attention中的self代表什么呢？这意味着这是一个自注意力，也就是Q、K、V都等于输入本身。这样一来也就意味着：Q、K、V都是来自输入的复制，Self attention中没有可学习的参数。可学习参数将会在多头注意力（Multi-head attention）中引入。

### Scaled dot-product attention

Transformer中使用的Attention做了一点修改，对QK进行了缩放，所以叫Scaled dot-product attention。
$$
\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

1. 为什么用dot-product注意力呢？

   因为这个在实现上是最容易的，点积也非常适合大矩阵运算。与此同时，论文作者做了实验，发现这种最简单的方式达到的效果和其他方式没有显著区别。

2. 为什么要scale呢？

   在$d_k$比较小的情况下，做scale意义不大，但是在$d_k$比较大的时候，$QK^T$的结果相差可能会比较大，导致softmax之后的输出比较两极化（要么接近1要么接近于0），这会导致梯度变得非常小。引入scale可以让输出的分布更平滑一点。

## Multi-head attention 多头注意力

下图是Multi-head attention的示意图

![multi-head-atten](/assets/images/multi-head-atten.png){: .align-center}
