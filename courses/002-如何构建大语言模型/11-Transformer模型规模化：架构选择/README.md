# 第 11 章：Transformer模型规模化：架构选择

来源：[原章节](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-11-scaling-transformers-architectural-choices)

[返回课程目录](../README.md)

在构建了一个Transformer模型之后，下一步是要弄明白如何有效地增大其规模。单纯地增大模型规模并非总是最佳途径；随着模型增长，特定的架构选择对性能、训练稳定性和计算需求有很大影响。

本章侧重讨论Transformer模型规模化过程中的设计考量。我们将讨论：

*   将模型大小、数据集大小和计算量与性能关联起来的经验性缩放法则，常表示为 $Performance \propto Compute^{\alpha} Data^{\beta} Size^{\gamma}$。
*   增加模型深度（层数）与宽度（隐藏维度大小）所带来的影响。
*   GeLU或SwiGLU等不同激活函数在前馈网络中的表现。
*   将层归一化置于残差连接之前或之后（Pre-LN对比Post-LN）所带来的影响。
*   对稀疏注意力机制的介绍，这些机制旨在处理自注意力在极长序列中的二次方复杂度问题。

在本章结束时，你将对设计更大、能力更强的Transformer模型时可用的架构调整手段，以及每种选择所关联的权衡取舍，有更清楚的认识。

## 小节

- 1. [神经网络语言模型的缩放定律](01-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E7%BC%A9%E6%94%BE%E5%AE%9A%E5%BE%8B.md)
- 2. [深度与宽度取舍](02-%E6%B7%B1%E5%BA%A6%E4%B8%8E%E5%AE%BD%E5%BA%A6%E5%8F%96%E8%88%8D.md)
- 3. [激活函数选择 (ReLU, GeLU, SwiGLU)](03-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E9%80%89%E6%8B%A9%20%28ReLU%2C%20GeLU%2C%20SwiGLU%29.md)
- 4. [规范化层放置位置（前置LN vs. 后置LN）](04-%E8%A7%84%E8%8C%83%E5%8C%96%E5%B1%82%E6%94%BE%E7%BD%AE%E4%BD%8D%E7%BD%AE%EF%BC%88%E5%89%8D%E7%BD%AELN%20vs.%20%E5%90%8E%E7%BD%AELN%EF%BC%89.md)
- 5. [稀疏注意力机制简介](05-%E7%A8%80%E7%96%8F%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E7%AE%80%E4%BB%8B.md)
