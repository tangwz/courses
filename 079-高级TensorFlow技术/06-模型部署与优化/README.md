# 第 6 章：模型部署与优化

来源：[原章节](https://apxml.com/zh/courses/advanced-tensorflow/chapter-6-model-deployment-optimization)

[返回课程目录](../README.md)

训练 TensorFlow 模型是一个重要进展，但工作并未就此结束。为了创造价值，模型必须高效可靠地投入运行。本章讲解了将模型从开发环境迁移到生产系统或边缘设备的重要阶段。

您将学习使用 SavedModel 格式打包 TensorFlow 模型进行部署的标准方法。我们将研究 TensorFlow Serving，它是一个专为高性能模型服务而设计的方案，包括如何使用常用协议与已部署的模型进行交互。此外，我们还将介绍优化策略，例如量化，这些策略旨在减小模型大小并提高推理速度。最后，我们介绍了 TensorFlow Lite，用于转换和准备模型，以便在移动、嵌入式以及其他资源受限的环境中部署。重点在于使您训练的模型能够被实际使用并高效运行所需的具体步骤。

## 小节

- 1. [保存和加载高级模型格式](01-%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E9%AB%98%E7%BA%A7%E6%A8%A1%E5%9E%8B%E6%A0%BC%E5%BC%8F.md)
- 2. [TensorFlow Serving 简介](02-TensorFlow%20Serving%20%E7%AE%80%E4%BB%8B.md)
- 3. [使用 REST 和 gRPC 部署 TensorFlow Serving 模型](03-%E4%BD%BF%E7%94%A8%20REST%20%E5%92%8C%20gRPC%20%E9%83%A8%E7%BD%B2%20TensorFlow%20Serving%20%E6%A8%A1%E5%9E%8B.md)
- 4. [模型优化方法](04-%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95.md)
- 5. [TensorFlow Lite (TF Lite) 简介](05-TensorFlow%20Lite%20%28TF%20Lite%29%20%E7%AE%80%E4%BB%8B.md)
- 6. [为 TF Lite 转换模型](06-%E4%B8%BA%20TF%20Lite%20%E8%BD%AC%E6%8D%A2%E6%A8%A1%E5%9E%8B.md)
- 7. [优化设备端推理](07-%E4%BC%98%E5%8C%96%E8%AE%BE%E5%A4%87%E7%AB%AF%E6%8E%A8%E7%90%86.md)
- 8. [动手实践：使用 TF Serving 部署模型](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20TF%20Serving%20%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B.md)
