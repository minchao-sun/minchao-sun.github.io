---
title: "如何估计Transformer模型的参数量级"
classes: wide
tag:
  - LLM
categories:
  - Machine Learning
---

这篇文章旨在帮助你快速估算一个使用了Transformer架构的大模型的参数数量。（不论是使用了完整的Transformer还是只使用了一部分）

计算方法来自于Dmytro Nikolaiev (Dimid)的[博客](https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0)。

## 太长不看🙈

Too long don't read

| 参数                 | 精确公式                                           | 估计公式                        | 更粗略的估计                       |
| -------------------- | -------------------------------------------------- | ------------------------------- | ---------------------------------- |
| Multi-head attention | $4(d_{model}^2+d_{model})$                         | $4d_{model}^2$                  | /                                  |
| Feed-forward         | $2d_{model}d_{ff}+d_{model}+d_{ff}$                | $2d_{model}d_{ff}$              | /                                  |
| Layer norm           | $2 d_{model}$                                      | 0                               | /                                  |
| Encoder              | $4d_{model}^2+2d_{model}d_{ff}+9d_{model}+d_{ff}$  | $4d_{model}^2+2d_{model}d_{ff}$ | $12d_{model}\approx10 d_{model}^2$ |
| Decoder              | $8d_{model}^2+2d_{model}d_{ff}+15d_{model}+d_{ff}$ | $8d_{model}^2+2d_{model}d_{ff}$ | $16d_{model}\approx10 d_{model}^2$ |

  



