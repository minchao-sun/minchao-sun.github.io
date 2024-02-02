---
title: "如何估计Transformer模型的参数量级"
classes: wide
mathjax: true
tag:
  - LLM
categories:
  - Machine Learning
---

这篇文章旨在帮助你快速估算一个使用了Transformer架构的大模型的参数数量。（不论是使用了完整的Transformer还是只使用了一部分）

我在这里仅提供一种最为简略的估算方法，更为严谨的计算方法可以参考Dmytro Nikolaiev (Dimid)的[博客](https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0)。

## 太长不看

### 代码方法

如果你可以把模型跑起来，当然可以通过代码来直接计算，你将会得到一个相当准确的答案。

```python
import torch

def count_parameters(model: torch.nn.Module) -> int:
    """ Returns the number of learnable parameters for a PyTorch model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### 数学方法

| 参数                 | 精确公式                                           | 估计公式                        | 更粗略的估计                       |
| -------------------- | -------------------------------------------------- | ------------------------------- | ---------------------------------- |
| Multi-head attention | $4(d_{model}^2+d_{model})$                         | $4d_{model}^2$                  | /                                  |
| Feed-forward         | $2d_{model}d_{ff}+d_{model}+d_{ff}$                | $2d_{model}d_{ff}$              | /                                  |
| Layer norm           | $2 d_{model}$                                      | $0$                             | /                                  |
| Encoder              | $4d_{model}^2+2d_{model}d_{ff}+9d_{model}+d_{ff}$  | $4d_{model}^2+2d_{model}d_{ff}$ | $12d_{model}\approx10 d_{model}^2$ |
| Decoder              | $8d_{model}^2+2d_{model}d_{ff}+15d_{model}+d_{ff}$ | $8d_{model}^2+2d_{model}d_{ff}$ | $16d_{model}\approx10 d_{model}^2$ |

  参考上表可以大致估计一个**单层**Encoder或者Decoder的参数量，需要注意的是：通常情况下我们默认$d_{ff}=4d_{model}$，事实上$d_{ff}$仍是一个独立的超参数，在某些模型中如有特别设置，则需要考虑这一点。此外，这里计算的是单层的Encoder或者Decoder，通常网络中会有数十甚至数百层，需要再乘上这个参数，才能得到总的参数量。

## A closer look

我们可以将原始的Transformer结构拆开，分成Encoder和Decoder模块。Decoder和Encoder的主要区别在于它多了一个带掩码的注意力层。我们可以把他们进一步拆分：

Encoder = Multi-head attention + Feed-forward + Layer norm

Decoder = Masked multi-head attention + Multi-head attention + Feed-forward + Layer norm

![transformer-arch](/assets/images/transformer-arch.png){: .align-center}

我们只是近似地估计参数量，因此可以认为 Decoder = Multi-head attention + Encoder

### Multi-head attention

下图是Multi-head attention的示意图，Q、K、V分别经过3次线性投影，然后经过Scaled dot-product attention（这一步骤内部没有可学习参数），最后再通过一次线性层得到输出。由此可知，可学习的参数均在线性投影中。

![multi-head-atten](/assets/images/multi-head-atten.png){: .align-center}

最后的linear层输入和输出的维度都是$d_{model}$，参数量的大小也就是$d_{model}d_{model}+d_{model}$，其中的一次项是bias，在估算中，我们忽略一次项，只统计二次项。

同时注意力机制中的Q/K/V各自拥有一个$d_{model}d_{model}$大小的权重（实际上内部根据head的数量做了切分，但是我们只考虑整体的参数数量），因此这里有$3d_{model}^2$的参数（忽略一次项的bias）。

综上，一个注意力层拥有的参数数量约为 $4d_{model}^2$

### Feed-forwad

前向反馈层还包含一个中间的隐藏层，维度是$d_{ff}$，注意这仍然是一个可以自行调节的超参数。不过，在原始论文以及BERT等后续论文中，$d_{ff}$的值都被默认设定为$4d_{model}$。因此默认情况下，除非有特别说明$d_{ff}$的取值，我们一般默认$d_{ff}=4d_{model}$。

前向反馈是由两个全连接构成的。参数量大小为：$d_{model}d_{ff}+d_{ff}d_{model}=2d_{model}d_{ff}$，同样的，我们忽略了bias。在默认情况下，$2d_{model}d_{ff}=8d_{model}^2$。

### Layer norm

layer norm只会贡献一次项$2 d_{model}$，我们在计算上忽略这一项。

### 总结

可以看到我们实际上只需要考虑attention和feed-forward中的参数，对于encoder，只有一个attention，则是$12d_{model}^2$，而decoder需要再增加一个attention，那么就是$16d_{model}^2$。



