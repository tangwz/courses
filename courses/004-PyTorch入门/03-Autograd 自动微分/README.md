# 第 3 章：Autograd 自动微分

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-3-automatic-differentiation-autograd)

[返回课程目录](../README.md)

有效地训练神经网络需要调整模型参数以最小化损失函数，这通常使用梯度下降或其变体。这个过程的**主要部分**是计算损失函数相对于每个参数的梯度，数学上表示为损失 $L$ 和参数 $w$ 的 $\frac{\partial L}{\partial w}$。对于复杂模型，手动推导和实现这些梯度计算是不切实际的。

本章介绍 PyTorch 的自动微分引擎 Autograd，它旨在自动计算这些梯度。我们将**了解** PyTorch 如何在对张量执行操作时动态构建计算图。你将学习如何使用 `requires_grad=True` 标记张量以进行梯度计算，使用 `.backward()` 触发反向传播以计算梯度，以及查看存储在 `.grad` 属性中的结果梯度。我们还将涉及梯度累积，使用 `optimizer.zero_grad()` 清零梯度的**必要性**，以及如何使用 `torch.no_grad()` 等上下文临时禁用梯度计算，以提高推理或评估期间的效率。

## 小节

- 1. [自动微分的原理](01-%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E7%9A%84%E5%8E%9F%E7%90%86.md)
- 2. [PyTorch 计算图](02-PyTorch%20%E8%AE%A1%E7%AE%97%E5%9B%BE.md)
- 3. [张量与梯度计算 (\`requires_grad\`)](03-%E5%BC%A0%E9%87%8F%E4%B8%8E%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97%20%28%60requires_grad%60%29.md)
- 4. [执行反向传播 (\`backward()\`)](04-%E6%89%A7%E8%A1%8C%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%20%28%60backward%28%29%60%29.md)
- 5. [访问梯度（\`.grad\`）](05-%E8%AE%BF%E9%97%AE%E6%A2%AF%E5%BA%A6%EF%BC%88%60.grad%60%EF%BC%89.md)
- 6. [禁用梯度追踪](06-%E7%A6%81%E7%94%A8%E6%A2%AF%E5%BA%A6%E8%BF%BD%E8%B8%AA.md)
- 7. [梯度累积](07-%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF.md)
- 8. [动手实践：Autograd 运用](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9AAutograd%20%E8%BF%90%E7%94%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-3-automatic-differentiation-autograd/quiz)
