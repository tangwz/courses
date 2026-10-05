# 第 4 章：训练与评估：Keras 方法到 PyTorch 循环的对应

来源：[原章节](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs)

[返回课程目录](../README.md)

在学习了如何通过与 Keras 对比来定义 PyTorch 中的模型架构之后，我们现在转向这些模型的训练和评估过程。如果您习惯使用 TensorFlow Keras，您可能已经使用过 `model.compile()` 来指定优化器和损失函数，然后使用 `model.fit()` 来管理训练迭代。PyTorch 采用一种更清晰的方法，您需要从基础组成部分构建训练循环。这种方式提供了更大的灵活性，并让您对整个训练过程有更清晰的认识。

本章将指导您在 PyTorch 中构建这些自定义的训练和评估循环。我们将考察损失函数（来自 `torch.nn` 或 `torch.nn.functional`）和优化算法（在 `torch.optim` 包中）是如何实现和使用的，并将它们与 TensorFlow Keras 的对应部分进行比较。您将学习标准的 PyTorch 训练流程：执行前向传播以获取预测，计算损失，通过 `loss.backward()` 计算梯度，以及使用 `optimizer.step()` 更新模型参数。此外，我们还将讨论如何组织模型评估，有效地追踪性能指标，并引入类似于 Keras Callbacks 的训练控制机制。

## 小节

- 1. [训练模式：TensorFlow 的 fit 方法与 PyTorch 训练循环](01-%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%BC%8F%EF%BC%9ATensorFlow%20%E7%9A%84%20fit%20%E6%96%B9%E6%B3%95%E4%B8%8E%20PyTorch%20%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)
- 2. [TensorFlow 和 PyTorch 中的损失函数](02-TensorFlow%20%E5%92%8C%20PyTorch%20%E4%B8%AD%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 3. [优化算法：TensorFlow 和 PyTorch 优化器](03-%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%EF%BC%9ATensorFlow%20%E5%92%8C%20PyTorch%20%E4%BC%98%E5%8C%96%E5%99%A8.md)
- 4. [PyTorch 中的梯度计算与权重更新](04-PyTorch%20%E4%B8%AD%E7%9A%84%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97%E4%B8%8E%E6%9D%83%E9%87%8D%E6%9B%B4%E6%96%B0.md)
- 5. [性能指标：Keras 指标与 PyTorch 对应的实现](05-%E6%80%A7%E8%83%BD%E6%8C%87%E6%A0%87%EF%BC%9AKeras%20%E6%8C%87%E6%A0%87%E4%B8%8E%20PyTorch%20%E5%AF%B9%E5%BA%94%E7%9A%84%E5%AE%9E%E7%8E%B0.md)
- 6. [PyTorch 中的模型评估循环](06-PyTorch%20%E4%B8%AD%E7%9A%84%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E5%BE%AA%E7%8E%AF.md)
- 7. [训练控制：Keras 回调与 PyTorch 自定义逻辑](07-%E8%AE%AD%E7%BB%83%E6%8E%A7%E5%88%B6%EF%BC%9AKeras%20%E5%9B%9E%E8%B0%83%E4%B8%8E%20PyTorch%20%E8%87%AA%E5%AE%9A%E4%B9%89%E9%80%BB%E8%BE%91.md)
- 8. [动手实践：实现一个完整的训练和评估循环](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AA%E5%AE%8C%E6%95%B4%E7%9A%84%E8%AE%AD%E7%BB%83%E5%92%8C%E8%AF%84%E4%BC%B0%E5%BE%AA%E7%8E%AF.md)

章节测验：[在线测验](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs/quiz)
