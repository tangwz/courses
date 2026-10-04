# 第 4 章：量化感知训练 (QAT)

来源：[原章节](https://apxml.com/zh/courses/practical-llm-quantization/chapter-4-quantization-aware-training-qat)

[返回课程目录](../README.md)

尽管训练后量化（PTQ）提供了一种在模型训练完成后进行量化的直接方法，但有时会导致精度明显下降，尤其是在量化到4比特整数等极低精度时。当保持精度是优先事项，且PTQ的结果不足以满足要求时，量化感知训练（QAT）提供了一种不同的方法。

QAT将量化过程直接整合到模型训练或微调阶段。通过在训练期间模拟低精度算术的影响，模型学习调整其权重，以最大限度地减少量化引起的精度损失。这通常能使最终量化模型获得比对同一原始模型应用PTQ更高的精度，特别是对于激进的量化目标。

在本章中，您将学习QAT的基础知识：

*   如何在训练循环中通过“假量化”操作模拟量化。
*   直通估计器（STE）技术，用于计算通过不可微分的量化函数（$q(x)$）的梯度。
*   使用常用的深度学习库实现QAT流程。
*   使用QAT对预训练模型进行微调的步骤。
*   比较QAT相对于PTQ的优缺点。
*   设置和运行QAT实验的实际考量。

完成本章后，您将掌握判断何时适合使用QAT以及如何实现它以生成更准确的低精度模型的知识。

## 小节

- 1. [量化感知训练的必要性](01-%E9%87%8F%E5%8C%96%E6%84%9F%E7%9F%A5%E8%AE%AD%E7%BB%83%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7.md)
- 2. [训练期间模拟量化影响](02-%E8%AE%AD%E7%BB%83%E6%9C%9F%E9%97%B4%E6%A8%A1%E6%8B%9F%E9%87%8F%E5%8C%96%E5%BD%B1%E5%93%8D.md)
- 3. [直通估计器 (STE)](03-%E7%9B%B4%E9%80%9A%E4%BC%B0%E8%AE%A1%E5%99%A8%20%28STE%29.md)
- 4. [使用深度学习框架实现量化感知训练](04-%E4%BD%BF%E7%94%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E5%AE%9E%E7%8E%B0%E9%87%8F%E5%8C%96%E6%84%9F%E7%9F%A5%E8%AE%AD%E7%BB%83.md)
- 5. [使用量化节点微调模型](05-%E4%BD%BF%E7%94%A8%E9%87%8F%E5%8C%96%E8%8A%82%E7%82%B9%E5%BE%AE%E8%B0%83%E6%A8%A1%E5%9E%8B.md)
- 6. [QAT 与 PTQ 的优缺点对比](06-QAT%20%E4%B8%8E%20PTQ%20%E7%9A%84%E4%BC%98%E7%BC%BA%E7%82%B9%E5%AF%B9%E6%AF%94.md)
- 7. [QAT实施中的实际考量](07-QAT%E5%AE%9E%E6%96%BD%E4%B8%AD%E7%9A%84%E5%AE%9E%E9%99%85%E8%80%83%E9%87%8F.md)
- 8. [动手实践：搭建简单量化感知训练运行环境](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%90%AD%E5%BB%BA%E7%AE%80%E5%8D%95%E9%87%8F%E5%8C%96%E6%84%9F%E7%9F%A5%E8%AE%AD%E7%BB%83%E8%BF%90%E8%A1%8C%E7%8E%AF%E5%A2%83.md)

章节测验：[在线测验](https://apxml.com/zh/courses/practical-llm-quantization/chapter-4-quantization-aware-training-qat/quiz)
