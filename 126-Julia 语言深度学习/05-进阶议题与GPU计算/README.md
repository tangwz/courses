# 第 5 章：进阶议题与GPU计算

来源：[原章节](https://apxml.com/zh/courses/julia-deep-learning/chapter-5-advanced-topics-gpu-computing)

[返回课程目录](../README.md)

在已经掌握 Julia 和 Flux.jl 中创建及训练神经网络的基础知识后，本章将讨论一些进阶技术和操作点。我们将着重介绍如何大幅加速模型训练和运行的方法，特别是通过使用图形处理器（GPU）。

您将学会：
*   配置并使用 CUDA.jl 以在 NVIDIA GPU 上运行 Flux.jl 模型，并有效管理 CPU 与 GPU 之间的数据。
*   对模型进行性能分析，以找出性能瓶颈并应用优化策略。
*   使用已有的模型，这是基于成熟架构进行开发的常见做法。
*   理解 Flux.jl 中基础的生成模型方法。
*   了解 Julia 生态系统中可用的其他深度学习库，并理解如何将 Python 库整合到您的 Julia 工作流程中。
*   最后，我们将讨论如何为部署您的 Julia 深度学习应用做准备。

完成本章学习后，您将更有能力去处理更具挑战性的深度学习任务，优化模型以提高速度，并考虑在更广泛的应用中使用模型的实际操作方面。

## 小节

- 1. [GPU 加速：使用 CUDA.jl 和 Flux](01-GPU%20%E5%8A%A0%E9%80%9F%EF%BC%9A%E4%BD%BF%E7%94%A8%20CUDA.jl%20%E5%92%8C%20Flux.md)
- 2. [在GPU上管理数据](02-%E5%9C%A8GPU%E4%B8%8A%E7%AE%A1%E7%90%86%E6%95%B0%E6%8D%AE.md)
- 3. [Flux模型性能分析与优化](03-Flux%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E4%B8%8E%E4%BC%98%E5%8C%96.md)
- 4. [在 Julia 中使用预训练模型](04-%E5%9C%A8%20Julia%20%E4%B8%AD%E4%BD%BF%E7%94%A8%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B.md)
- 5. [使用 Flux 介绍生成模型](05-%E4%BD%BF%E7%94%A8%20Flux%20%E4%BB%8B%E7%BB%8D%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B.md)
- 6. [Julia其他深度学习库简要概览](06-Julia%E5%85%B6%E4%BB%96%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%BA%93%E7%AE%80%E8%A6%81%E6%A6%82%E8%A7%88.md)
- 7. [互操作性：从 Julia 调用 Python 库用于深度学习](07-%E4%BA%92%E6%93%8D%E4%BD%9C%E6%80%A7%EF%BC%9A%E4%BB%8E%20Julia%20%E8%B0%83%E7%94%A8%20Python%20%E5%BA%93%E7%94%A8%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0.md)
- 8. [Julia 深度学习应用程序的部署方式](08-Julia%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%BA%94%E7%94%A8%E7%A8%8B%E5%BA%8F%E7%9A%84%E9%83%A8%E7%BD%B2%E6%96%B9%E5%BC%8F.md)
- 9. [实践：使用GPU加速训练](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8GPU%E5%8A%A0%E9%80%9F%E8%AE%AD%E7%BB%83.md)
