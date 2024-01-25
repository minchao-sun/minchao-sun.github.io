---
title:  "Adam Optimizer"
toc: true
toc_sticky: true
mathjax: true
categories:
  - Machine Learning
---

Adam优化器是随机梯度下降的扩展版本[1]

The name is derived from adaptive moment estimation.

通常情况下Adam拥有比较好的泛化性能(generalization)

Adam优化器是两种优化器的结合：Momentum和Root Mean Square Propagation(RMSP)

## Momentum

Momentum的权重更新如下

$$
\boldsymbol{w}_{t+1}\leftarrow \boldsymbol{w}_t-\alpha \boldsymbol{m}_t \\
\text{where}\hspace{5 mm} \boldsymbol{m}_t\leftarrow \beta \boldsymbol{m}_{t-1} + (1-\beta) \boldsymbol{g}_t
$$

其中

$\boldsymbol{m}_t=t$ 时刻的梯度累积

$\alpha=$ learning rate

$\boldsymbol{g}_t = t$ 时刻的梯度

$\beta$ 是一个超参数，是一个0到1之间的常量

## Root Mean Square Propagation(RMSP/RMSprop)

RMSprop是一种试图改进 AdaGrad 的自适应学习算法。它不像 AdaGrad 那样采用梯度平方的累计和，而是采用“指数移动平均值”。RMSprop权重更新

$$
\boldsymbol{w}_t\leftarrow \boldsymbol{w}_{t-1}-\frac{\alpha_t}{\sqrt{\boldsymbol{v}_t}+\varepsilon}\boldsymbol{g}_t \\
\text{where}\hspace{5 mm} \boldsymbol{v}_t\leftarrow \beta \boldsymbol{v}_{t-1}+(1-\beta)\boldsymbol{g}_t^2
$$

$\alpha_t=t$ 时刻的学习率

$\boldsymbol{g}_t = t$ 时刻的梯度

$\beta$ 是一个超参数，是一个0到1之间的常量

$\varepsilon$ 是一个非常小的常量，防止DividedByZero Error

$\boldsymbol{v}_t=$ 过去梯度的平方

## Adam
adam结合了上述两种优化器。把 $\boldsymbol{m}_t$ 和 $\boldsymbol{v}_t$ 的公式拿下来，同时把两个公式里的 $\beta$ 用下标区分

$$
\boldsymbol{m}_t\leftarrow\beta_1 \boldsymbol{m}_{t-1} + (1-\beta)\boldsymbol{g}_t \\
\boldsymbol{v}_t\leftarrow\beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2)\boldsymbol{g}_t^2
$$

初始情况下，$\boldsymbol{m}_t$ 和 $\boldsymbol{v}_t$ 都为0

Adam权重更新过程如下

$$
\begin{align*}
    & t\leftarrow t+1 \\
    & \boldsymbol{g}_t \leftarrow \nabla f_t(\boldsymbol{w}_{t-1}) \\
    &\boldsymbol{m}_t \leftarrow \beta_1\boldsymbol{m}_{t-1}+(1-\beta_1)\boldsymbol{g}_t \\
    &\boldsymbol{v}_t \leftarrow \beta_2\boldsymbol{v}_{t-1} + (1-\beta_2)\boldsymbol{g}_t^2 \\
    &\hat{\boldsymbol{m}_t} \leftarrow \frac{\boldsymbol{m}_t}{1-\beta_1^t} \\
    &\hat{\boldsymbol{v}_t} \leftarrow \frac{\boldsymbol{v}_t}{1-\beta_2^t} \\
    &\boldsymbol{w}_t \leftarrow \boldsymbol{w}_{t-1} - \alpha\frac{\hat{\boldsymbol{m}_t}}{\sqrt{\hat{\boldsymbol{v}_t}}+\varepsilon}
\end{align*}
$$

$\hat{\boldsymbol{m}_t}$ 和 $\hat{\boldsymbol{v}_t}$ 是对m和v的无偏估计

因此，Adam需要的超参数包括：

- $\alpha$ 学习率
- $\beta_1$ 和 $\beta_2$ 是梯度的一阶矩估计和二阶矩估计的衰减率
- $\varepsilon$ 是一个很小的常量

Adam 需要保存的值包括：

- $\boldsymbol{m}_t$: 梯度的移动平均，或者叫动量 momentum
- $\boldsymbol{v}_t$: 梯度方差的移动平均，或者叫 variance
- $\boldsymbol{w}_t$: 在混合精度训练中，由于优化器需要使用FP32（单精度浮点数）的weights来更新，因此需要单独保存一份FP32的权重weights

## AdamW

AdamW在原有的Adam基础上加上了权重衰减（weight decay）[2]

区别在于权重更新的时候：

$$
\boldsymbol{w}_t \leftarrow (1-\alpha\lambda)\boldsymbol{w}_{t-1} - \alpha\frac{\hat{\boldsymbol{m}_t}}{\sqrt{\hat{\boldsymbol{v}_t}}+\varepsilon}
$$

参考pytorch的实现：

$$
\begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}
$$

## Reference

[1] Kingma, D.P., & BA, J. (2014). [Adam: A Method for Stochastic Optimization.](https://arxiv.org/abs/1412.6980)

[2] [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)