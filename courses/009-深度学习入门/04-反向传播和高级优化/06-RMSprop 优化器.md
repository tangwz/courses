# RMSprop 优化器

来源：[原文](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-4-backpropagation-advanced-optimization/rmsprop-optimizer)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准梯度下降 (gradient descent) (SGD) 及其动量变体在更新所有参数 (parameter)时使用相同的学习率（可能受动量影响而缩放），但这可能不是最理想的。设想一个损失函数 (loss function)，它在一个方向上非常陡峭，但在另一个方向上却相当平坦。单一的学习率可能会在陡峭方向上引起震荡，或者在平坦方向上移动过慢。我们需要一种方法来为每个参数单独调整学习率。

RMSprop（均方根传播）是一种优化算法，旨在通过保持每个参数各自的学习率来解决这个问题。它通过跟踪每个参数梯度*平方*的移动平均值来实现这一点。核心思想是将特定权重 (weight)的学习率除以该权重近期梯度幅度的运行平均值。

### RMSprop 机制

RMSprop 修改了梯度下降 (gradient descent)的更新规则。对于每个参数 (parameter)（以权重 (weight) $w$ 为例），它计算梯度平方的指数衰减平均值。设 $S_{dw}$ 为权重 $w$ 在给定迭代中的此移动平均值。$S_{dw}$ 的更新规则是：


$$
S_{dw} = \beta S_{dw} + (1 - \beta) \left( \frac{\partial L}{\partial w} \right)^2
$$


这里：

- $\frac{\partial L}{\partial w}$ 是损失函数 (loss function) $L$ 对权重 $w$ 的梯度。
- $(\frac{\partial L}{\partial w})^2$ 是梯度的逐元素平方。
- $\beta$ 是一个超参数 (hyperparameter)，即衰减率，通常设置为 0.9、0.99 或类似的值。它控制过去梯度平方与当前梯度的权重分配。较高的 $\beta$ 意味着平均值包含更长的历史信息。
- $S_{dw}$ 累积梯度平方信息。如果 $w$ 的近期梯度很大，$S_{dw}$ 也会很大，反之亦然。

参数更新规则随后使用此移动平均值来缩放学习率 $\alpha$：


$$
w = w - \alpha \frac{\frac{\partial L}{\partial w}}{\sqrt{S_{dw} + \epsilon}}
$$


类似地，对于偏置 (bias)参数 $b$：


$$
S_{db} = \beta S_{db} + (1 - \beta) \left( \frac{\partial L}{\partial b} \right)^2
$$


$$
b = b - \alpha \frac{\frac{\partial L}{\partial b}}{\sqrt{S_{db} + \epsilon}}
$$


项 $\sqrt{S_{dw} + \epsilon}$（或 $\sqrt{S_{db} + \epsilon}$）是近期梯度的均方根 (RMS)，算法因此得名。添加小值 $\epsilon$（epsilon，例如 $10^{-8}$）是为了数值稳定性，以防止在 $S_{dw}$ 可能变得非常接近零的情况下发生除以零的错误。

### 直观解释

这如何提供帮助？

- **调整学习率：** 如果特定权重 (weight) $w$ 的梯度持续较大，$S_{dw}$ 将会变大。除以 $\sqrt{S_{dw} + \epsilon}$ 有效地降低了该特定权重的学习率，从而避免大步长并抑制损失函数 (loss function)陡峭区域的震荡。
- **加速平坦方向：** 反之，如果 $w$ 的梯度较小或正在减小，$S_{dw}$ 将会较小。除以一个小的 $\sqrt{S_{dw} + \epsilon}$ 提高了该权重的有效学习率，从而允许更大的步长并在标准 SGD 可能缓慢前进的平坦区域中取得更快进展。

本质上，RMSprop 根据每个参数 (parameter)梯度的历史幅度自动调整其步长。

### 优点与考量

**优点：**

- **自适应学习率：** 自动基于每个参数 (parameter)调整学习率，通常比 SGD 或 Momentum 在某些问题上收敛更快。
- **隐式学习率调整：** 相比 SGD，降低了对全局学习率 $\alpha$ 选择的敏感性，尽管 $\alpha$ 仍需设置。

**考量：**

- **超参数 (hyperparameter)：** 引入了衰减率 $\beta$ 和稳定性项 $\epsilon$ 作为超参数，尽管它们的默认值（如 $\beta = 0.99$, $\epsilon = 10^{-8}$）通常效果良好。
- **并非万能：** 虽然有效，但 RMSprop 在某些复杂的优化场景中可能仍然表现不佳。

RMSprop 是优化算法中的一个重要进展。它通过引入基于近期梯度大小的、针对每个参数的调整，解决了单一全局学习率的局限性。

### 在 PyTorch 中使用 RMSprop

在 PyTorch 这样的框架中实现 RMSprop 简单明了。配置优化器时，只需从 `torch.optim` 模块中选择 `RMSprop` 即可。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 假设您已定义模型，例如：
# model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
# 定义输入数据和目标标签
# inputs = torch.randn(64, 10)
# labels = torch.randn(64, 1)
# loss_fn = nn.MSELoss()

# 示例：定义一个简单的线性模型用于演示
model = nn.Linear(10, 1) 

# 定义超参数
learning_rate = 0.001
# 注意：PyTorch 使用 'alpha' 作为平滑常数（即我们符号中的 beta）
beta_rms = 0.99 
epsilon = 1e-8 

# 实例化 RMSprop 优化器
optimizer = optim.RMSprop(model.parameters(), 
                          lr=learning_rate, 
                          alpha=beta_rms, # 这是衰减率 beta
                          eps=epsilon, 
                          momentum=0) # 标准 RMSprop 在此处没有动量项

# --- 训练步骤示例 ---
# 假设输入和标签可用
# optimizer.zero_grad()          # 清除之前的梯度
# outputs = model(inputs)        # 前向传播
# loss = loss_fn(outputs, labels) # 计算损失
# loss.backward()                # 反向传播
# optimizer.step()               # 使用 RMSprop 更新权重
# -----------------------------

print(f"已创建优化器: {optimizer}")
```

> 一个简单的例子，展示了如何在 PyTorch 中实例化 RMSprop 优化器。请注意，`optim.RMSprop` 中的参数 (parameter) `alpha` 对应于算法描述中讨论的衰减率 $\beta$。标准 RMSprop 本身不包含动量项，尽管 PyTorch 的实现允许添加（在此处设置为 0 以获得基本版本）。

RMSprop 提供了一种调整训练期间学习率的有效方法。下一节将介绍 Adam 优化器，它将 RMSprop 的自适应学习率方法与我们之前看到的动量原理结合起来，创建了当今深度学习 (deep learning)中最广泛使用的优化器之一。

## 参考资料

- [Lecture 6e: RMSProp. Neural Networks for Machine Learning.](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) — Geoffrey Hinton (2012)
  Journal: University of Toronto (via Coursera)
  介绍RMSprop算法，该算法通过基于梯度平方的移动平均值来调整每个参数的学习率。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8, section 8.5.3
  全面解释包括RMSprop在内的优化算法的理论基础、公式和直观理解。
- [torch.optim.RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html) — PyTorch Contributors (2024)
  PyTorch中RMSprop优化器的官方文档，详细说明其参数和典型用法。
- [Adam: A Method for Stochastic Optimization](https://doi.org/10.48550/arXiv.1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化器，它在RMSprop自适应学习率的基础上融入动量，提供有价值的比较。

---

[上一节](05-%E5%B8%A6%E6%9C%89%E5%8A%A8%E9%87%8F%E7%9A%84%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md) · [下一节](07-Adam%20%E4%BC%98%E5%8C%96%E5%99%A8.md)
