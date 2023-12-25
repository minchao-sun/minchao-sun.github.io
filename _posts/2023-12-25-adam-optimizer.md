---
title:  "Adam Optimizer"
---

Adam优化器是随机梯度下降的扩展版本[1]

The name is derived from adaptive moment estimation.

通常情况下Adam拥有比较好的泛化性能(generalization)

Adam优化器是两种优化器的结合：Momentum和Root Mean Square Propagation(RMSP)

## Momentum

Momentum的权重更新如下

$$
w_{t+1}=w_t-\alpha m_t \\
\text{where}\hspace{5 mm} m_t=\beta m_{t-1} + (1-\beta) g_t
$$

其中

$m_t=t$ 时刻的梯度累积

$\alpha=$ learning rate

$g_t = t$ 时刻的梯度

$\beta$ 是一个超参数，是一个0到1之间的常量

## Root Mean Square Propagation(RMSP/RMSprop)

RMSprop是一种试图改进 AdaGrad 的自适应学习算法。它不像 AdaGrad 那样采用梯度平方的累计和，而是采用“指数移动平均值”。RMSprop权重更新

$$
w_{t+1}=w_t-\frac{\alpha_t}{\sqrt{v_t}+\varepsilon}g_t \\
\text{where}\hspace{5 mm} v_t=\beta v_{t-1}+(1-\beta)g_t^2
$$

$\alpha_t=t$ 时刻的学习率

$g_t = t$ 时刻的梯度

$\beta$ 是一个超参数，是一个0到1之间的常量

$\varepsilon$ 是一个非常小的常量，防止DividedByZero Error

$v_t=$ 过去梯度的平方

## Adam
adam结合了上述两种优化器。把 $m_t$ 和 $v_t$ 的公式拿下来，同时把两个公式里的 $\beta$ 用下标区分

$$
m_t=\beta_1 m_{t-1} + (1-\beta)g_t \\
v_t=\beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

初始情况下，$m_t$ 和 $v_t$ 都为0

Adam权重更新过程如下

$$
\begin{align*}
    m_t &:= \beta_1+(1-\beta_1)g_t \\
    v_t &:= \beta_2 + (1-\beta_2)g_t^2 \\
    \hat{m_t} &:= \frac{m_t}{1-\beta_1} \\
    \hat{v_t} &:= \frac{v_t}{1-\beta_2} \\
    w_{t+1} &:= w_t - \alpha\frac{\hat{m_t}}{\sqrt{\hat{v_t}}+\varepsilon}
\end{align*}
$$

$\hat{m_t}$ 和 $\hat{v_t}$ 是对m和v的无偏估计

因此，Adam需要的超参数包括：

- $\alpha$ 学习率
- $\beta_1$ 和 $\beta_2$ 是梯度的一阶矩估计和二阶矩估计的衰减率
- $\varepsilon$ 是一个很小的常量

Adam 需要保持的值包括：

- $m_t$ 梯度的移动平均，或者叫动量 momentum
- $v_t$ 梯度方差的移动平均，或者叫 variance
- $w_t$ 在混合精度训练中，由于优化器需要使用FP32（单精度浮点数）的weights来更新，因此需要单独保存一份FP32的权重weights

## Reference

[1] Kingma, D.P., & BA, J. (2014). Adam: A Method for Stochastic Optimization. https://arxiv.org/abs/1412.6980