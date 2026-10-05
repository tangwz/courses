# 第 1 章：PyTorch 内部机制与自动求导

来源：[原章节](https://apxml.com/zh/courses/advanced-pytorch/chapter-1-pytorch-internals-autograd)

[返回课程目录](../README.md)

为了高效处理复杂任务，理解 PyTorch 的内部运作机制会有帮助。本章着重讲解基本构成要素：PyTorch 张量、自动求导（autograd）机制以及将它们联系起来的计算图。

我们将研究张量的结构和内存管理方式。你会了解到操作执行时 PyTorch 如何动态构建计算图，以及 autograd 引擎如何遍历这些图来计算梯度，例如损失 $L$ 对权重 $w$ 的偏导数 $\frac{\partial L}{\partial w}$。

主要内容包括：
*   `torch.Tensor` 的内部结构。
*   动态计算图的创建和使用方式。
*   autograd 引擎在反向传播过程中的逐步操作。
*   通过 `torch.autograd.Function` 定义 `forward` 和 `backward` 方法来实现自定义操作。
*   计算高阶梯度。
*   检查梯度和可视化计算图的方法。
*   PyTorch 中高效内存使用的注意事项。

熟悉这些核心组成部分对于调试复杂模型、优化性能以及实现标准库之外的自定义功能都十分有益。最后，我们将通过一个实践练习来构建自己的 autograd 函数。

## 小节

- 1. [张量实现细节](01-%E5%BC%A0%E9%87%8F%E5%AE%9E%E7%8E%B0%E7%BB%86%E8%8A%82.md)
- 2. [理解计算图](02-%E7%90%86%E8%A7%A3%E8%AE%A1%E7%AE%97%E5%9B%BE.md)
- 3. [Autograd 引擎机制](03-Autograd%20%E5%BC%95%E6%93%8E%E6%9C%BA%E5%88%B6.md)
- 4. [自定义 Autograd 函数：前向与反向](04-%E8%87%AA%E5%AE%9A%E4%B9%89%20Autograd%20%E5%87%BD%E6%95%B0%EF%BC%9A%E5%89%8D%E5%90%91%E4%B8%8E%E5%8F%8D%E5%90%91.md)
- 5. [高阶梯度计算](05-%E9%AB%98%E9%98%B6%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97.md)
- 6. [梯度查看与图可视化](06-%E6%A2%AF%E5%BA%A6%E6%9F%A5%E7%9C%8B%E4%B8%8E%E5%9B%BE%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- 7. [内存管理考量](07-%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%E8%80%83%E9%87%8F.md)
- 8. [实践操作：构建自定义自动求导函数](08-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E6%9E%84%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC%E5%87%BD%E6%95%B0.md)
