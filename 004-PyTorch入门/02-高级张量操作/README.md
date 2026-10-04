# 第 2 章：高级张量操作

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-2-advanced-tensor-manipulations)

[返回课程目录](../README.md)

在前面介绍的基础张量操作之上，本章将讲解更高级的张量操作方法。对张量结构、数据类型和设备放置进行精细控制，对于有效准备数据和实现复杂的深度学习模型来说非常重要。

在本章中，你将学会：

*   使用索引和切片方法，精确选择和修改张量元素。
*   使用 `view()`、`reshape()` 和 `permute()` 重构张量，而不改变其数据。
*   使用 `cat()`、`stack()`、`split()` 和 `chunk()` 组合多个张量或将单个张量分割成不同部分。
*   使用 PyTorch 的广播机制，对形状兼容但不同的张量之间执行操作。
*   处理不同的数值数据类型（例如 `float`、`int`），并相应地转换张量类型。
*   使用 `.to(device)` 在 CPU 内存和 GPU 加速器之间传输张量。

掌握这些操作对于处理深度学习工作流中遇到的多样化数据格式和计算要求是必要的。

## 小节

- 1. [张量索引与切片](01-%E5%BC%A0%E9%87%8F%E7%B4%A2%E5%BC%95%E4%B8%8E%E5%88%87%E7%89%87.md)
- 2. [张量的重塑与维度调整](02-%E5%BC%A0%E9%87%8F%E7%9A%84%E9%87%8D%E5%A1%91%E4%B8%8E%E7%BB%B4%E5%BA%A6%E8%B0%83%E6%95%B4.md)
- 3. [张量的合并与分割](03-%E5%BC%A0%E9%87%8F%E7%9A%84%E5%90%88%E5%B9%B6%E4%B8%8E%E5%88%86%E5%89%B2.md)
- 4. [理解广播机制](04-%E7%90%86%E8%A7%A3%E5%B9%BF%E6%92%AD%E6%9C%BA%E5%88%B6.md)
- 5. [张量数据类型](05-%E5%BC%A0%E9%87%8F%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B.md)
- 6. [CPU 与 GPU 张量](06-CPU%20%E4%B8%8E%20GPU%20%E5%BC%A0%E9%87%8F.md)
- 7. [练习：张量操作技巧](07-%E7%BB%83%E4%B9%A0%EF%BC%9A%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C%E6%8A%80%E5%B7%A7.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-2-advanced-tensor-manipulations/quiz)
