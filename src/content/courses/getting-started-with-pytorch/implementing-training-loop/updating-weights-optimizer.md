---
course: "getting-started-with-pytorch"
chapter: "implementing-training-loop"
lesson: "updating-weights-optimizer"
sourceId: 2197
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/updating-weights-optimizer"
title: "使用优化器更新权重"
description: "使用优化器根据计算出的梯度更新模型权重。"
order: 7
plots: []
sourceHash: "97908f9a02f0350c3accf2c6fdb337d8346285323e1c16a1e41170ff6d218c29"
sourceCorrections: []
---

模型的参数 (parameter)（具有`requires_grad=True`的权重 (weight)和偏差）在其`.grad`属性中包含已计算的梯度。这些梯度是在损失计算并使用`loss.backward()`执行反向传播 (backpropagation)时生成的。它们，例如$\nabla_{\theta} L$，表示在参数空间中会使损失增长最快的方向。为了使损失最小化，我们需要朝着*相反*的方向调整参数。这正是优化器的作用。

在第4章中，您学习了如何实例化优化器，例如`torch.optim.SGD`或`torch.optim.Adam`，并传入模型的参数（`model.parameters()`）以及学习率等配置信息。现在，在训练循环中，您使用优化器的`step()`方法来执行参数更新。

```python
# 假设模型、损失函数和优化器已定义
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# ... 在一个批次的训练循环内部 ...

# 前向传播
outputs = model(inputs)

# 计算损失
loss = criterion(outputs, labels)

# 反向传播 - 计算梯度
loss.backward()

# 使用优化器更新权重
optimizer.step()
```

调用`optimizer.step()`会遍历在优化器初始化时注册的所有参数。对于每个参数`p`，它会使用存储在`p.grad`中的梯度来更新参数值`p.data`。

随机梯度下降 (gradient descent)（SGD）使用的最基本的更新规则是：


$$
\text{参数} = \text{参数} - \text{学习率} \times \text{梯度}
$$


或者更正式地，对于参数$\theta$：


$$
\theta_{new} = \theta_{old} - \eta \nabla_{\theta} L
$$


其中$\eta$是学习率（类似于创建优化器时传入的`lr`参数），而$\nabla_{\theta} L$是通过`loss.backward()`计算的梯度，优化器通过`parameter.grad`在内部访问它。

不同的优化器实现了更复杂的更新规则。例如，像Adam这样的优化器对每个参数使用自适应学习率并融入动量思想，但核心思想保持不变：使用计算出的梯度调整参数以最小化损失。`optimizer.step()`调用将所选优化算法定义的具体更新逻辑进行了封装。

在调用`optimizer.step()`*之前*，必须先调用`loss.backward()`。`backward()`调用计算梯度，而`step()`调用则使用这些梯度来更新权重。如果不先调用`backward()`，`.grad`属性将不会被填充（或者会包含来自上一次迭代的旧值），优化器也就无法知道如何有效地调整参数。

更新权重后的下一个重要步骤是在处理下一个批次之前清除梯度。我们将在下一节“梯度清零”中介绍这一点。

## 参考资料

- [torch.optim](https://pytorch.org/docs/stable/optim.html) — PyTorch Contributors (2025)
  Publisher: PyTorch Foundation
  PyTorch 优化包的官方文档，包含 SGD 和 Adam 等优化器，以及用于参数更新的 `optimizer.step()` 方法。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  提出 Adam 优化算法的开创性论文，该算法是一种广泛使用的采用自适应学习率和动量的优化方法。
