# 第 6 章：实现训练循环

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop)

[返回课程目录](../README.md)

你已经学习了如何使用 `torch.nn` 定义模型，以及如何高效地使用 `Dataset` 和 `DataLoader` 准备数据。本章将侧重于整合这些组成部分，以实际训练你的神经网络。我们将构建被称为训练循环的标准迭代过程。

你将学习涉及的重要步骤：

*   设置模型、损失函数和优化器。
*   迭代 `DataLoader` 提供的数据批次。
*   执行**前向传播**：将输入数据送入模型以获得预测结果。
*   计算**损失**：使用所选标准来衡量模型预测与真实标签之间的差异，通常表示为 $L$。
*   执行**反向传播**：通过调用 `$loss.backward()` 计算损失相对于模型参数 ($\nabla_{\theta} L$) 的梯度。
*   更新**模型权重**：使用优化器根据计算出的梯度调整模型参数 ($\theta$)，通常通过 `$optimizer.step()`。
*   在下一次迭代前，使用 `$optimizer.zero_grad()` 将梯度归零。

此外，我们将介绍如何实现一个独立的循环来评估模型在验证集或测试集上的表现，以及在训练期间或之后保存和加载模型状态（检查点）的方法。到本章结束时，你将能够为你的 PyTorch 模型实现一个完整的训练和评估流程。

## 小节

- 1. [训练循环的构成](01-%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF%E7%9A%84%E6%9E%84%E6%88%90.md)
- 2. [设置模型、损失函数和优化器](02-%E8%AE%BE%E7%BD%AE%E6%A8%A1%E5%9E%8B%E3%80%81%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%92%8C%E4%BC%98%E5%8C%96%E5%99%A8.md)
- 3. [使用 DataLoader 遍历数据](03-%E4%BD%BF%E7%94%A8%20DataLoader%20%E9%81%8D%E5%8E%86%E6%95%B0%E6%8D%AE.md)
- 4. [前向传播：获取预测结果](04-%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%EF%BC%9A%E8%8E%B7%E5%8F%96%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.md)
- 5. [计算损失](05-%E8%AE%A1%E7%AE%97%E6%8D%9F%E5%A4%B1.md)
- 6. [反向传播：计算梯度](06-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%EF%BC%9A%E8%AE%A1%E7%AE%97%E6%A2%AF%E5%BA%A6.md)
- 7. [使用优化器更新权重](07-%E4%BD%BF%E7%94%A8%E4%BC%98%E5%8C%96%E5%99%A8%E6%9B%B4%E6%96%B0%E6%9D%83%E9%87%8D.md)
- 8. [梯度清零](08-%E6%A2%AF%E5%BA%A6%E6%B8%85%E9%9B%B6.md)
- 9. [实现评估循环](09-%E5%AE%9E%E7%8E%B0%E8%AF%84%E4%BC%B0%E5%BE%AA%E7%8E%AF.md)
- 10. [保存和加载模型检查点](10-%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B%E6%A3%80%E6%9F%A5%E7%82%B9.md)
- 11. [动手实践：完整训练流程](11-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%8C%E6%95%B4%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/quiz)
