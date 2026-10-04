# 第 1 章：衔接 TensorFlow 与 PyTorch：核心要点

来源：[原章节](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-1-pytorch-tensorflow-core-concepts)

[返回课程目录](../README.md)

本章对 TensorFlow 和 PyTorch 进行基础对比，侧重它们的核心运作方式。如果您已熟悉 TensorFlow，本章将帮助您把现有知识对应到 PyTorch 环境中。

您将学会：
*   区分 TensorFlow 的静态计算图与 PyTorch 动态的“定义即运行”方式。
*   比较 `tf.Tensor` 和 `torch.Tensor`，包括它们的创建、属性和常用操作。
*   了解 PyTorch 的自动微分机制 `autograd` 与 TensorFlow 的 `tf.GradientTape` 之间的关系。
*   查看两个框架如何与 NumPy 集成以实现高效数据处理。
*   在 PyTorch 中管理跨不同设备（例如 CPU 和 GPU）的计算。

我们将涵盖主要异同点，为 PyTorch 的使用打好底子。本章最后将提供实践练习，以应用这些核心要点。

## 小节

- 1. [从 TensorFlow 到 PyTorch：开发者指南](01-%E4%BB%8E%20TensorFlow%20%E5%88%B0%20PyTorch%EF%BC%9A%E5%BC%80%E5%8F%91%E8%80%85%E6%8C%87%E5%8D%97.md)
- 2. [TensorFlow 计算图与 PyTorch 动态计算的对比](02-TensorFlow%20%E8%AE%A1%E7%AE%97%E5%9B%BE%E4%B8%8E%20PyTorch%20%E5%8A%A8%E6%80%81%E8%AE%A1%E7%AE%97%E7%9A%84%E5%AF%B9%E6%AF%94.md)
- 3. [张量比较：tf.Tensor 与 torch.Tensor](03-%E5%BC%A0%E9%87%8F%E6%AF%94%E8%BE%83%EF%BC%9Atf.Tensor%20%E4%B8%8E%20torch.Tensor.md)
- 4. [基础张量运算：对比视角](04-%E5%9F%BA%E7%A1%80%E5%BC%A0%E9%87%8F%E8%BF%90%E7%AE%97%EF%BC%9A%E5%AF%B9%E6%AF%94%E8%A7%86%E8%A7%92.md)
- 5. [自动微分：GradientTape 与 Autograd 对比](05-%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%EF%BC%9AGradientTape%20%E4%B8%8E%20Autograd%20%E5%AF%B9%E6%AF%94.md)
- 6. [NumPy 在 PyTorch 和 TensorFlow 中的结合](06-NumPy%20%E5%9C%A8%20PyTorch%20%E5%92%8C%20TensorFlow%20%E4%B8%AD%E7%9A%84%E7%BB%93%E5%90%88.md)
- 7. [设备管理：CPU和GPU控制](07-%E8%AE%BE%E5%A4%87%E7%AE%A1%E7%90%86%EF%BC%9ACPU%E5%92%8CGPU%E6%8E%A7%E5%88%B6.md)
- 8. [动手实践：张量操作与自动求导](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C%E4%B8%8E%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC.md)

章节测验：[在线测验](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-1-pytorch-tensorflow-core-concepts/quiz)
