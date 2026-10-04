---
course: "introduction-to-deep-learning"
chapter: "backpropagation-advanced-optimization"
lesson: "gradient-descent-momentum"
sourceId: 5047
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-4-backpropagation-advanced-optimization/gradient-descent-momentum"
title: "带有动量的梯度下降"
description: "学习动量如何加速梯度下降并帮助跳出局部最小值。"
order: 5
plots: ["plots/5047-0.json"]
sourceHash: "9c40dd6c6115f9ee6b3b4b5baf2bd846754b74923717dcb8e814fea990282b9c"
sourceCorrections: []
---

正如我们所见，标准梯度下降 (gradient descent)仅根据当前位置计算的梯度来更新权重 (weight)。尽管这种方法可行，但它可能导致训练效率低下，特别是在某些类型的损失函数 (loss function)形状中。想象损失曲面中一个狭窄的沟壑或山谷。梯度下降倾向于在沟壑的陡峭壁上反复振荡，沿着底部向最小值缓慢移动。同样，它可能停滞在局部最小值，或者在鞍点处显著减速。

### 思想：加入惯性

为了解决这些问题，我们可以引入**动量**的理念，借鉴物理学的思想。想象一个球从山上滚下来。球不是仅仅计算当前位置最陡峭的坡度并稍微沿该方向移动（像标准梯度下降 (gradient descent)那样），而是具有*动量*。它当前的速率会影响它在下一步的移动。如果它已经以某个方向快速移动，即使当前的坡度略有改变，它也倾向于保持该方向。这种动量有助于它：

1. 在一致的坡度上**加速**。
2. 通过随时间平均抵消的梯度方向来**抑制振荡**。
3. 由于其积累的速度，**越过**小的局部最小值或平台。

带有动量的梯度下降将物理类比应用于优化过程。它引入了一个“速率”向量 (vector) $v$，该向量累积了过去梯度的指数衰减移动平均值。权重 (weight)更新随后会同时考虑当前梯度和这个速率。

### 动量如何运作

带有动量的梯度下降 (gradient descent)的更新规则在每个迭代 $t$ 中包含两个步骤：

1. **更新速率向量 (vector) $v_t$：**

   
   $$
   v_t = \beta v_{t-1} + \eta \nabla L(w_{t-1})
   $$
   

   这里：

   - $v_t$ 是时间步 $t$ 的速率向量。
   - $v_{t-1}$ 是前一时间步的速率向量（初始化为零）。
   - $\beta$ 是**动量系数**，这是一个超参数 (parameter) (hyperparameter)，通常在0.5到0.99之间（常用0.9）。它决定了保留多少过去的速率。更高的 $\beta$ 值意味着过去的梯度影响更大。
   - $\eta$ 是学习率。
   - $\nabla L(w_{t-1})$ 是损失函数 (loss function) $L$ 对前一时间步 $t-1$ 的权重 (weight) $w$ 的梯度。

   这一步主要通过取旧速率的一部分（$\beta$）并加上缩放的当前梯度（$\eta \nabla L(w_{t-1})$）来计算新的速率。如果当前梯度方向与之前的速率方向一致，则速率的大小会增加。如果梯度方向振荡，速率倾向于被抑制，因为相反的梯度项会随时间部分抵消。
2. **更新权重 $w_t$：**

   
   $$
   w_t = w_{t-1} - v_t
   $$
   

   权重通过沿着新计算的速率向量 $v_t$ 的方向移动来更新。

通过加入速率 $v_t$（其中包含近期梯度历史的信息），更新变得更平滑、更快，尤其是在梯度方向一致的路径上。

### 动量效果的可视化

考虑优化一个带有狭窄山谷的函数。标准梯度下降 (gradient descent)可能会在两侧壁之间来回跳动，而动量则会走一条更直接的路径。



![梯度下降与动量路径对比](plots/5047-0.json)



> 样本损失曲面上的优化路径。标准梯度下降（粉色）在狭窄山谷中振荡，向最小值（接近原点）缓慢移动。动量（蓝色）抑制了这些振荡，并沿着谷底加速，走了一条更直接的路径。

### 动量系数 $\beta$

动量系数 $\beta$ 控制着过去梯度的影响。

- 如果 $\beta = 0$，更新规则就会简化为标准梯度下降 (gradient descent)（$v_t = \eta \nabla L(w_{t-1})$，$w_t = w_{t-1} - \eta \nabla L(w_{t-1})$）。
- 当 $\beta$ 接近1时，过去的梯度被赋予更大的权重 (weight)，导致更新更平滑，但如果学习率也较高，优化器可能会越过最小值。

$\beta$ 的常用值是0.9或更高，例如0.99。通常从0.9开始使用效果不错，并在训练期间可能增加它。这个参数 (parameter)，就像学习率一样，可能需要针对特定问题进行调整以达到最佳效果。

### 在PyTorch中的实现

在PyTorch等框架中使用动量很简单。在定义优化器时，只需指定 `momentum` 参数 (parameter)即可。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 假设 'model' 是你定义的神经网络 (nn.Module)
# 假设 'learning_rate' 是你选择的学习率（例如 0.01）

# 使用带有动量的SGD定义优化器
momentum_coefficient = 0.9
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_coefficient)

# --- 在你的训练循环中 ---
# loss.backward() # 计算梯度

# optimizer.step() # 使用带有动量的SGD更新权重
# optimizer.zero_grad() # 重置梯度以进行下一次迭代
```

通过简单地将 `momentum` 参数添加到 `optim.SGD` 构造函数中，优化器会自动实现前面描述的速率计算和权重 (weight)更新步骤。

带有动量的梯度下降 (gradient descent)是标准梯度下降的显著改进，通常能带来更快的收敛速度和对复杂损失曲面的更好应对。尽管它有助于应对许多问题，但进一步的改进促成了RMSprop和Adam等自适应方法的出现，我们将在下一节中进行考察。

## 参考资料

- [On the importance of initialization and momentum in deep learning](http://proceedings.mlr.press/v28/sutskever13.pdf) — Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning (ICML); Volume: 28; Pages: 1139-1147
  一篇重要论文，强调了在训练深度神经网络时，动量与适当初始化相结合的显著优势。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8: Optimization for Training Deep Models
  深度学习领域的标准教材，详细解释了优化算法，包括动量的机制和优势。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Optimization](https://cs231n.github.io/optimization-2/) — Stanford University CS231n Course Staff (2023)
  备受推崇的斯坦福CS231n课程笔记的一部分，提供了易于理解且图文并茂的动量梯度下降解释。
- [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) — PyTorch Developers (2024)
  PyTorch 官方的随机梯度下降优化器文档，清晰地说明了 `momentum` 参数及其在实现中的直接使用。
