---
course: "getting-started-with-pytorch"
chapter: "implementing-training-loop"
lesson: "zeroing-gradients"
sourceId: 2198
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/zeroing-gradients"
title: "梯度清零"
description: "在下一次迭代前清除先前梯度的必要性。"
order: 8
plots: []
sourceHash: "7c06e4bd3bd632299ab60ddd0bc6ad5a66468d70bcdf1c1947fdac850cc78795"
sourceCorrections: []
---

调用 `$loss.backward()` 计算损失相对于模型参数 (parameter)（那些 `requires_grad=True` 的张量）的梯度。这些梯度表示为减少损失所需的改变方向和大小，它们存储在每个参数张量的 `.grad` 属性中。随后，`$optimizer.step()` 会根据所选的优化算法（如 SGD 或 Adam）使用这些存储的梯度来更新参数值。

然而，PyTorch 在反向传播 (backpropagation)过程中处理梯度的方式有一个不明显但非常重要的细节：**PyTorch 会累积梯度**。当您调用 `$loss.backward()$` 时，为每个参数新计算的梯度会 *添加* 到该参数 `.grad` 属性中已有的值上。

如果您不处理这个问题，请考虑训练循环在多次迭代中会发生什么：

1. **第一次迭代：** 计算损失 $L_1$，调用 `$loss_1.backward()`。梯度 $\nabla_{\theta} L_1$ 被计算并存储在 `param.grad` 中。调用 `$optimizer.step()$`。
2. **第二次迭代：** 计算损失 $L_2$，调用 `$loss_2.backward()`。新的梯度 $\nabla_{\theta} L_2$ 被计算。PyTorch 将这些梯度 *添加* 到现有梯度中，因此 `param.grad` 现在持有 $\nabla_{\theta} L_1 + \nabla_{\theta} L_2$。调用 `$optimizer.step()$`。

第二次迭代中的优化器步骤使用了不正确的梯度信息。它使用了当前批次和上一批次的混合梯度。这会阻止模型良好地学习，因为权重 (weight)更新是基于来自不同数据点的陈旧且混合的梯度信号。

### `optimizer.zero_grad()` 的用途

为了防止这种累积并确保优化器 *仅* 基于当前批次的梯度来更新权重 (weight)，您必须在为下一次迭代计算梯度之前手动重置梯度。这正是 `optimizer.zero_grad()` 方法所做的。

调用 `optimizer.zero_grad()` 会遍历优化器被配置来管理的所有参数 (parameter) ($\theta$)，并将其 `.grad` 属性重置为零（或 `None`）。

### 何时调用 `optimizer.zero_grad()`

您需要在每次训练迭代中调用 `optimizer.zero_grad()` 一次。最常见且推荐的做法是在循环的 *开始* 处调用它，即在处理下一个批次之前：

```python
# 训练循环示例代码片段
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # 1. 清零上一迭代的梯度
        optimizer.zero_grad()

        # 2. 前向传播：计算预测输出
        outputs = model(inputs)

        # 3. 计算损失
        loss = criterion(outputs, labels)

        # 4. 反向传播：计算损失相对于参数的梯度
        loss.backward()

        # 5. 执行一次优化步骤（参数更新）
        optimizer.step()

    # ... (评估、日志记录等) ...
```

或者，您可以在 `optimizer.step()` *之后* 立即调用 `optimizer.zero_grad()`。在标准循环中，其功能结果是相同的。将其放在开始处可以清楚地划分新批次处理的开始。核心是，它必须在下一次 `loss.backward()` 调用 *之前* 被调用，以避免梯度在迭代之间累积。

忘记清零梯度是 PyTorch 训练循环中常见的错误源，这通常会导致模型无法收敛或表现出异常的学习情况。请务必确保 `optimizer.zero_grad()` 在您的训练迭代中位置正确。

虽然梯度累积通常是不希望发生的，但当 GPU 内存受限时，它可以被有意地用作一种方法来模拟更大的批次大小。在这种情况下，您可以对多个小批次执行前向和反向传播 (backpropagation)，通过在每次 `loss.backward()` 后 *不* 调用 `optimizer.zero_grad()` 来累积梯度，然后只在处理完所需数量的小批次后才调用 `optimizer.step()` 和 `optimizer.zero_grad()`。但是，对于常规训练，每次迭代清零梯度是常规步骤。

## 参考资料

- [torch.optim](https://pytorch.org/docs/stable/optim.html) — PyTorch Authors (2025)
  Publisher: PyTorch Foundation
  PyTorch优化器的官方文档，详细说明了`zero_grad()`方法及其在每次训练迭代中重置梯度的作用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本基础性教科书，解释了反向传播、梯度下降和神经网络优化过程的理论基础。
- [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — PyTorch Authors (2024)
  Publisher: PyTorch
  PyTorch官方教程，演示了神经网络的完整训练循环，包括`optimizer.zero_grad()`的正确位置和使用。
