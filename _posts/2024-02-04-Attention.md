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

$$
\text{Attention}(Q,K,V)=\text{softmax}(QK^T)V
$$

Transformer中使用的Attention做了一点修改，对QK进行了缩放，所以叫Scaled dot-product attention

$$
\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

## Multi-head attention 多头注意力

下图是Multi-head attention的示意图

![multi-head-atten](/assets/images/multi-head-atten.png){: .align-center}
