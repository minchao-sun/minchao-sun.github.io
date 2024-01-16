---
title: "如何估计Transformer模型的参数量级"
classes: wide
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



