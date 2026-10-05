# 第 2 章：高性能 TensorFlow

来源：[原章节](https://apxml.com/zh/courses/advanced-tensorflow/chapter-2-high-performance-tensorflow)

[返回课程目录](../README.md)

训练复杂模型或大规模部署时，常会达到计算性能的极限。缓慢的训练周期会增加开发时间和成本，而高推理延迟则会损害用户体验。基于对 TensorFlow 执行模型的理解，本章着重讲解如何使您的 TensorFlow 代码运行得更快、更高效。

您将学会如何使用 TensorBoard Profiler 系统地找出性能瓶颈。本章将介绍提升硬件利用率的方法，重点讲解 GPU，并引入 Google 的张量处理器 (TPU)。主要的优化策略将详细介绍，包括：

*   **混合精度训练：** 使用 $float16$ 等低精度数值格式，以加快计算速度并减少兼容硬件上的内存占用。
*   **XLA（加速线性代数）编译：** 使 TensorFlow 的编译器能够合并运算，并为特定加速器生成优化代码。
*   **高效输入管道：** 设计 `tf.data` 管道，以有效地预取和准备数据，防止 CPU 在训练期间成为瓶颈。

在本章结束时，您将掌握分析 TensorFlow 模型和数据管道性能特点的工具和知识，能够应用特定优化措施，在各种硬件平台上获得明显的提速。

## 小节

- 1. [使用 TensorBoard Profiler 分析 TensorFlow 代码性能](01-%E4%BD%BF%E7%94%A8%20TensorBoard%20Profiler%20%E5%88%86%E6%9E%90%20TensorFlow%20%E4%BB%A3%E7%A0%81%E6%80%A7%E8%83%BD.md)
- 2. [优化 GPU 利用率](02-%E4%BC%98%E5%8C%96%20GPU%20%E5%88%A9%E7%94%A8%E7%8E%87.md)
- 3. [混合精度训练技术](03-%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83%E6%8A%80%E6%9C%AF.md)
- 4. [张量处理单元（TPU）介绍](04-%E5%BC%A0%E9%87%8F%E5%A4%84%E7%90%86%E5%8D%95%E5%85%83%EF%BC%88TPU%EF%BC%89%E4%BB%8B%E7%BB%8D.md)
- 5. [XLA（加速线性代数）编译](05-XLA%EF%BC%88%E5%8A%A0%E9%80%9F%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0%EF%BC%89%E7%BC%96%E8%AF%91.md)
- 6. [tf.data 管道的性能考量](06-tf.data%20%E7%AE%A1%E9%81%93%E7%9A%84%E6%80%A7%E8%83%BD%E8%80%83%E9%87%8F.md)
- 7. [模型性能分析与加速实践](07-%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E4%B8%8E%E5%8A%A0%E9%80%9F%E5%AE%9E%E8%B7%B5.md)
