---
course: "deep-learning-regularization-optimization"
chapter: "foundational-optimizers"
lesson: "nesterov-accelerated-gradient"
sourceId: 4983
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-5-foundational-optimizers/nesterov-accelerated-gradient"
title: "涅斯捷罗夫加速梯度 (NAG)"
description: "阐述涅斯捷罗夫加速梯度 (NAG) 的更新规则及其优点。"
order: 7
plots: []
sourceHash: "b19bcd8dcab4c8e0faa43d6ffd0a1bb047780a20648235844ebd2d4ed573d418"
sourceCorrections: []
---

动量通过累积过去的梯度来加速梯度下降 (gradient descent)，使优化器能在一致的方向上更快地移动并抑制震荡。涅斯捷罗夫加速梯度（NAG），有时也称为涅斯捷罗夫动量，通过引入一个预测性的元素来进一步优化。设想一个球从山上滚下来；动量赋予了球惯性。但是，如果球能在完全基于当前动量和梯度来确定下一步行动之前，智能地向前看一点呢？这种智能的超前预测机制就是涅斯捷罗夫加速梯度背后的主要思想。

### “向前看”步骤

标准动量在当前位置 $\theta_{t-1}$ 计算梯度，然后沿着更新后的累积梯度（动量 $v_t$）的方向迈出一大步。

NAG 采取略微不同的方法。它首先沿着之前的动量方向迈出“向前看”的一步。它计算如果只应用前一步的累积速度，参数 (parameter)将会到达的位置。接着，它在这个“向前看”位置而非原始位置计算梯度。最后，它利用这个向前看的梯度来调整最终的更新步骤。

这为什么有效？如果动量步将我们带到接近最小值的位置，或爬上不应爬的斜坡，向前看位置的梯度会指向最小值或向下，比标准动量更有效地修正轨迹。它充当一个修正因子，防止优化器越过最小值，并在许多情况下实现更快的收敛。

### 数学公式

我们来比较更新规则。回顾标准动量的更新方式：

1. 计算速度更新：
   $v_t = \mu v_{t-1} + \eta \nabla L(\theta_{t-1})$
2. 更新参数 (parameter)：
   $\theta_t = \theta_{t-1} - v_t$

这里，$\mu$ 是动量系数（通常约为 0.9），$\eta$ 是学习率，$v_t$ 是步骤 $t$ 的速度向量 (vector)，而 $\nabla L(\theta_{t-1})$ 是在*当前*位置 $\theta_{t-1}$ 处评估的损失函数 (loss function) $L$ 相对于参数 $\theta$ 的梯度。

涅斯捷罗夫加速梯度修改了这个过程：

1. 计算近似未来位置（向前看）：
   $\theta_{lookahead} = \theta_{t-1} - \mu v_{t-1}$
   这一步仅根据之前的速度更新来估计参数将到达的位置。
2. 在向前看位置计算梯度：
   $g_t = \nabla L(\theta_{lookahead})$
   注意梯度是在 $\theta_{lookahead}$ 处计算的，而非 $\theta_{t-1}$。
3. 使用向前看梯度更新速度：
   $v_t = \mu v_{t-1} + \eta g_t$
4. 使用*新*速度更新参数：
   $\theta_t = \theta_{t-1} - v_t$

简单来说，NAG 采用沿动量方向稍前位置计算的梯度，从而提供一个更明智的更新方向。

### 可视化差异

我们可以可视化动量与 NAG 之间的差异。

> 更新步骤比较。标准动量在当前位置（黑点）计算梯度，并将其添加到动量步（紫色虚线）中。NAG 首先将动量步（紫色虚线）移动到向前看位置（红点），在那里计算梯度（红色虚线），然后将此梯度添加到动量步中，得到最终更新（蓝色实线）。

### NAG 的优点

向前看计算使 NAG 相较于标准动量具有多项优势：

1. **响应能力提升：** 如果梯度在动量步后方向发生显著变化，NAG 能更快地响应。它本质上在进行最终更新之前“修正”了动量方向。
2. **减少越过最小值：** 通过向前看，NAG 更不容易越过最小值，尤其是在狭窄山谷或曲率快速变化的区域。如果动量步过于激进，向前看梯度会充当一个制动器。
3. **通常收敛更快：** 经验结果常表明，在各种深度学习 (deep learning)任务上，NAG 比标准动量收敛更快。

尽管稍显复杂，但实际实现通常只需在优化器中设置一个标志。

### NAG 的实际使用

大多数深度学习 (deep learning)框架在其标准 SGD 优化器中将 NAG 作为可选功能实现。例如，在 PyTorch 中，你在创建 `torch.optim.SGD` 优化器时启用 NAG：

```python
import torch
import torch.optim as optim

# 假设 'model_parameters' 从你的神经网络中获取
# model_parameters = model.parameters() 
# 实际参数的占位符：
model_parameters = [torch.randn(10, 5, requires_grad=True)] 

learning_rate = 0.01
momentum_coeff = 0.9

# 实例化启用涅斯捷罗夫动量的 SGD 优化器
optimizer = optim.SGD(
    model_parameters,
    lr=learning_rate,
    momentum=momentum_coeff,
    nesterov=True  # 启用涅斯捷罗夫加速梯度
)

# --- 训练循环中的使用示例 ---
# optimizer.zero_grad() # 重置梯度
# loss = calculate_loss(...) # 计算损失
# loss.backward() # 计算梯度
# optimizer.step() # 使用 NAG 更新参数

print(f"优化器已创建: {optimizer}")
```

使用 SGD 时，启用涅斯捷罗夫动量通常是与标准动量一同尝试的良好默认选项，因为它经常提供更好的收敛表现，且额外计算成本极小。它代表了一种简单而有效的改进，用于在复杂损失曲面中进行优化。

## 参考资料

- [Introductory Lectures on Convex Optimization: A Basic Course](https://doi.org/10.1007/978-1-4419-8853-9) — Yurii Nesterov (2004)
  Publisher: Springer; DOI: [10.1007/978-1-4419-8853-9](https://doi.org/10.1007/978-1-4419-8853-9)
  提供了加速梯度法的基本解释，包括Nesterov的原始算法。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 287-322 (Chapter 8: Optimization for Training Deep Models)
  一本全面涵盖深度学习优化算法的教材，详细讨论了Nesterov动量。
- [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch中随机梯度下降优化器的官方文档，详细说明了用于启用NAG的`nesterov`参数。
- [Lecture 6: Optimization Part 1](http://cs231n.stanford.edu/2023/slides/lecture_3.pdf) — Fei-Fei Li, Yunzhu Li, Ruohan Gao (2023)
  Publisher: Stanford University
  来自知名深度学习课程的讲义，提供了包括Nesterov加速梯度在内的优化算法的清晰解释和可视化。
