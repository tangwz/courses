---
course: "optimization-techniques-ml"
chapter: "adaptive-learning-rate-algorithms"
lesson: "rmsprop-algorithm"
sourceId: 1263
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-3-adaptive-learning-rate-algorithms/rmsprop-algorithm"
title: "RMSprop：处理AdaGrad学习率递减的问题"
description: "RMSprop如何通过使用平方梯度的移动平均值来修改AdaGrad，以避免学习率过快衰减。"
order: 3
plots: []
sourceHash: "fe0f20b3a0f5ff273be08d59e5806df0cddd94dabc981ce35adc2ee09cced752"
sourceCorrections: []
---

AdaGrad通过累积过去梯度的平方来调整每个参数 (parameter)的学习率。虽然这有助于应对不同的曲率，但它带来了一个潜在问题：分母中累积的和$\sum g_i^2$单调增长。经过多次迭代，特别是在具有非凸目标函数的深度学习 (deep learning)情况中，这个分母会变得非常大，导致有效学习率过早地趋近于零。这会有效地停止学习过程，远未收敛时便如此。

RMSprop（均方根传播），一个由Geoff Hinton提出的未发表的自适应学习率方法，直接处理了这个问题。RMSprop没有让平方梯度的和无限累积，而是使用平方梯度的*指数衰减平均*。这意味着较旧的梯度对平均值的贡献逐渐减少，阻止分母无限增大并停止学习。

我们来看其机制。在每个时间步$t$，RMSprop首先为当前参数$\theta_t$计算梯度$g_t = \nabla_{\theta} J(\theta_t)$。然后，它维护平方梯度的移动平均值$E[g^2]_t$，更新方式如下：


$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g_t^2
$$


具体来说：

- $g_t^2$表示梯度向量 (vector)$g_t$的逐元素平方。
- $E[g^2]_t$是平方梯度在时间步$t$的移动平均估计值。
- $\gamma$是*衰减率*（或动量项），一个超参数 (hyperparameter)，通常设置为类似0.9、0.99或0.999的值。它控制过去平方梯度相对于当前梯度的权重 (weight)分配。$\gamma$值越高，意味着平均值包含更长的梯度历史信息。

与此相比，AdaGrad的分母实际上具有$\gamma = 1$（无衰减），并持续添加当前平方梯度。通过使用$\gamma < 1$，RMSprop确保非常旧梯度的影响随时间减小。

RMSprop的参数更新规则然后使用这个移动平均值：


$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$


在此：

- $\eta$是全局学习率，另一个超参数。
- $\epsilon$是一个小的常数（例如$10^{-8}$），为数值稳定性而添加，以防止除以零，特别是在$E[g^2]_t$可能非常小的初始步骤中。
- 除法和平方根运算是逐元素执行的。

核心思想很直观：如果某个参数的近期梯度很大，$E[g^2]_t$将很大，并且该参数的有效学习率（$\eta / \sqrt{E[g^2]_t + \epsilon}$）将减小。反之，如果近期梯度很小，有效学习率将增加。重要的一点是，由于$E[g^2]_t$是一个移动平均值，如果近期梯度再次变小，它也可以减小，从而允许学习率恢复，这与AdaGrad不同。

### 超参数 (parameter) (hyperparameter)

RMSprop引入了衰减率$\gamma$以及全局学习率$\eta$和稳定性常数$\epsilon$。

- **学习率（$\eta$）：** 仍需仔细选择。典型的起始值可能是$0.001$。
- **衰减率（$\gamma$）：** 控制平方梯度平均值的记忆长度。常见值为$0.9$或$0.99$。接近1的值意味着更慢的衰减，并考虑更长的历史信息。
- **Epsilon ($\epsilon$)：** 通常设置为类似$10^{-8}$或$10^{-6}$这样的小值，并且通常不需要大量调整。

### 优势和注意事项

- **处理AdaGrad的学习率过快衰减问题：** 通过使用移动平均值，RMSprop避免了AdaGrad学习率单调递减的问题，使其更适合长时间训练和非凸问题。
- **良好的实际表现：** RMSprop在实践中通常表现良好，特别是对于训练循环神经网络 (neural network) (RNN)，尽管它作为一种通用默认算法在某种程度上已被Adam（我们接下来会讨论）所取代。
- **超参数 (parameter) (hyperparameter)调整：** 需要调整$\eta$和$\gamma$，尽管默认值通常提供一个合理的起始点。

本质上，RMSprop修改了AdaGrad的分母，使用指数加权平均值而非累积和。这一简单的改变避免了可能导致AdaGrad停滞不前的激进学习率衰减，从而得到一个更有效且常用的自适应优化算法。它为Adam算法奠定了基础，Adam通过引入动量进一步完善了这一思想。

## 参考资料

- [Neural Networks for Machine Learning, Lecture 6e: RmsProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) — Geoffrey Hinton (2012)
  Pages: Slide 29
  介绍了RMSprop作为一种自适应学习率算法，阐述了其核心机制以缓解AdaGrad学习率下降过快的问题。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  这篇基础性论文介绍了Adam优化器，并提供了RMSprop作为其前身算法的背景，讨论了其动机并与其他自适应方法进行了比较。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8
  一本关于自适应学习率方法的综合教科书解释，涵盖了RMSprop、其运行机制以及在其他优化器中的位置。
