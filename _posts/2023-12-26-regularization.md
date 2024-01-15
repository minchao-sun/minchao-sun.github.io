---
title:  "Regularization"
classes: wide
toc: true
toc_sticky: true
tags:
  - Math
categories:
  - Machine Learning
---

正则化是一个试图将“答案”变简单的过程。在机器学习中，正则化通常是为了增强模型的泛化性能和防止过拟合。

显式的正则化是在损失函数中增加一个正则化项，用于衡量模型的复杂度。换言之，模型不应只是以最小化损失（Empirical risk）为目标：

$$
\min{(\text{Loss}(\text{Data|Model}))}
$$

而是以最小化损失和复杂度为目标，这被称为**结构风险最小化**:

$$
\min{(\text{Loss}(\text{Data|Model})+\text{complexity(Model)})}
$$

现在，我们的训练优化算法由两项构成：一个是**损失项**，用于衡量模型与数据的拟合度；另一个是**正则化项**，用于衡量模型的复杂度。在实践中，这个正则化项通常是一个L1或L2范数。

范数(norm)是具有“长度”概念的函数，其为向量空间内所有向量赋予非零的正**长度**或**大小**。

$p\text{-norm}$:

$$
\forall p\ge 1 \in \mathbb{R}, \text{ the } p\text{-norm of vector } \boldsymbol{x}=(x_1,...,x_n) \text{ is} \\
||x||_p:=\left(\sum_{i=1}^n |x_i|^p\right)^{1/p}
$$

## L1 Regularization

L1范数可以理解为空间中的曼哈顿距离，因此也称为曼哈顿范数。

L1范数可以为模型带来稀疏性，它对权重的绝对值之和进行惩罚，并鼓励权重值为0。



$$
||\boldsymbol{x}||=|x_1|+...+|x_n|
$$

## L2 Regularization

L2范数可以理解为空间中的欧几里得距离，因此也称为欧几里得范数。

$$
||\boldsymbol{x}||_2=\sqrt{x_1^2+...+x_n^2}
$$
