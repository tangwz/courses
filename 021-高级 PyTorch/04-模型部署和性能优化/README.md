# 第 4 章：模型部署和性能优化

来源：[原章节](https://apxml.com/zh/courses/advanced-pytorch/chapter-4-deployment-performance-optimization)

[返回课程目录](../README.md)

在开发和训练复杂的 PyTorch 模型之后，重点转向如何将它们投入实际应用。本章将介绍模型部署和性能优化，提供使模型在推理时更快、更小、更节省资源的方法。

我们将介绍使用 TorchScript 进行模型序列化，学习跟踪和脚本两种方法。您将学习模型压缩技术，包括量化（静态、动态和量化感知训练）和剪枝策略，以减小模型大小和计算需求。我们将使用 PyTorch Profiler 来识别 CPU 和 GPU 执行中的性能瓶颈。此外，您还将学习将模型导出为 ONNX 格式以获得更广泛的兼容性，并学习使用 TorchServe 高效地提供模型服务。

在本章结束时，您将掌握分析模型性能的实用技能，并运用多种优化技术，这对于将 PyTorch 模型从开发环境部署到生产环境是必不可少的。

## 小节

- 1. [TorchScript 基础: 追踪与脚本化](01-TorchScript%20%E5%9F%BA%E7%A1%80-%20%E8%BF%BD%E8%B8%AA%E4%B8%8E%E8%84%9A%E6%9C%AC%E5%8C%96.md)
- 2. [模型量化技术](02-%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96%E6%8A%80%E6%9C%AF.md)
- 3. [模型剪枝策略](03-%E6%A8%A1%E5%9E%8B%E5%89%AA%E6%9E%9D%E7%AD%96%E7%95%A5.md)
- 4. [PyTorch Profiler 性能分析](04-PyTorch%20Profiler%20%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90.md)
- 5. [通过外部库优化算子](05-%E9%80%9A%E8%BF%87%E5%A4%96%E9%83%A8%E5%BA%93%E4%BC%98%E5%8C%96%E7%AE%97%E5%AD%90.md)
- 6. [模型导出为 ONNX 格式](06-%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA%E4%B8%BA%20ONNX%20%E6%A0%BC%E5%BC%8F.md)
- 7. [使用 TorchServe 提供模型服务](07-%E4%BD%BF%E7%94%A8%20TorchServe%20%E6%8F%90%E4%BE%9B%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1.md)
- 8. [实践：模型性能分析与量化](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E4%B8%8E%E9%87%8F%E5%8C%96.md)
