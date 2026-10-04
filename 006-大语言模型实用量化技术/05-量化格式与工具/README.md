# 第 5 章：量化格式与工具

来源：[原章节](https://apxml.com/zh/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling)

[返回课程目录](../README.md)

当模型通过训练后量化（Post-Training Quantization）或量化感知训练（Quantization-Aware Training）等方法进行量化后，一些实际问题随之而来：这些低精度模型如何高效地保存、加载和运行？标准模型序列化方法可能无法最佳处理$INT4$或$INT8$等格式所需的特有结构和元数据（如缩放因子或零点）。

本章将针对这些实际考量进行讲解，介绍量化LLM生态系统中常用的格式和软件工具。我们将介绍：

*   **专用文件格式：** 了解GGUF等格式的结构和用途，用于GPTQ量化模型的约定，以及AWQ格式的细节。
*   **核心库：** 使用Hugging Face Optimum等库进行量化和模型管理，以及`bitsandbytes`在推理时执行高效低比特操作。

您将熟悉将模型转换为这些格式，并使用相关工具高效地加载和运行它们。

## 小节

- 1. [常见量化模型格式概述](01-%E5%B8%B8%E8%A7%81%E9%87%8F%E5%8C%96%E6%A8%A1%E5%9E%8B%E6%A0%BC%E5%BC%8F%E6%A6%82%E8%BF%B0.md)
- 2. [GGUF：结构与使用](02-GGUF%EF%BC%9A%E7%BB%93%E6%9E%84%E4%B8%8E%E4%BD%BF%E7%94%A8.md)
- 3. [GPTQ格式：库支持与应用](03-GPTQ%E6%A0%BC%E5%BC%8F%EF%BC%9A%E5%BA%93%E6%94%AF%E6%8C%81%E4%B8%8E%E5%BA%94%E7%94%A8.md)
- 4. [AWQ 格式说明](04-AWQ%20%E6%A0%BC%E5%BC%8F%E8%AF%B4%E6%98%8E.md)
- 5. [使用 Hugging Face Transformers 和 Optimum](05-%E4%BD%BF%E7%94%A8%20Hugging%20Face%20Transformers%20%E5%92%8C%20Optimum.md)
- 6. [使用bitsandbytes进行量化](06-%E4%BD%BF%E7%94%A8bitsandbytes%E8%BF%9B%E8%A1%8C%E9%87%8F%E5%8C%96.md)
- 7. [模型转换与加载工具](07-%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2%E4%B8%8E%E5%8A%A0%E8%BD%BD%E5%B7%A5%E5%85%B7.md)
- 8. [实践：转换和加载量化格式](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%BD%AC%E6%8D%A2%E5%92%8C%E5%8A%A0%E8%BD%BD%E9%87%8F%E5%8C%96%E6%A0%BC%E5%BC%8F.md)

章节测验：[在线测验](https://apxml.com/zh/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/quiz)
