# 第 1 章：AI 计算基础设施入门

来源：[原章节](https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure)

[返回课程目录](../README.md)

AI 模型的性能与其运行的硬件直接相关。要构建高效系统，您必须首先了解机器学习工作负载的计算需求以及满足这些需求的硬件组件。本章提供初步分析。

您将学会区分训练和推理工作负载及其不同的硬件要求。我们将分析 CPU 在顺序任务中的作用以及 GPU 在深度学习中常见的并行计算中的作用。我们将比较它们的架构，并了解为何 GPU 擅长同时执行数千次操作，例如神经网络核心的矩阵乘法 ($C = A \cdot B$)。本讨论还将涉及 TPU 等专用加速器，以及内存、存储和网络的辅助作用。

本章最后是一个实践练习，您将在 CPU 和 GPU 上对一项任务进行基准测试，以便亲身观察这些性能差异。

## 小节

- 1. [人工智能工作负载概览](01-%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD%E6%A6%82%E8%A7%88.md)
- 2. [CPU在AI系统中的作用](02-CPU%E5%9C%A8AI%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 3. [GPU 在加速人工智能中的作用](03-GPU%20%E5%9C%A8%E5%8A%A0%E9%80%9F%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E4%B8%AD%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 4. [CPU与GPU架构在机器学习中的对比](04-CPU%E4%B8%8EGPU%E6%9E%B6%E6%9E%84%E5%9C%A8%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E5%AF%B9%E6%AF%94.md)
- 5. [TPU及其他ASIC简介](05-TPU%E5%8F%8A%E5%85%B6%E4%BB%96ASIC%E7%AE%80%E4%BB%8B.md)
- 6. [内存及其对大型模型的重要性](06-%E5%86%85%E5%AD%98%E5%8F%8A%E5%85%B6%E5%AF%B9%E5%A4%A7%E5%9E%8B%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%87%8D%E8%A6%81%E6%80%A7.md)
- 7. [AI数据集存储方案](07-AI%E6%95%B0%E6%8D%AE%E9%9B%86%E5%AD%98%E5%82%A8%E6%96%B9%E6%A1%88.md)
- 8. [分布式系统的网络考量](08-%E5%88%86%E5%B8%83%E5%BC%8F%E7%B3%BB%E7%BB%9F%E7%9A%84%E7%BD%91%E7%BB%9C%E8%80%83%E9%87%8F.md)
- 9. [动手实践：CPU 与 GPU 性能对比](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9ACPU%20%E4%B8%8E%20GPU%20%E6%80%A7%E8%83%BD%E5%AF%B9%E6%AF%94.md)
