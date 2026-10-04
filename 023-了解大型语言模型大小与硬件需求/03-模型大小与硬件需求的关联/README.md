# 第 3 章：模型大小与硬件需求的关联

来源：[原章节](https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-3-model-size-hardware-connection)

[返回课程目录](../README.md)

在介绍了大型语言模型、它们的参数计量方式以及GPU和显存等必要的硬件组成部分之后，我们现在将这些方面联系起来。本章将讨论LLM中的参数数量如何直接转化为硬件资源要求。

您将了解：

*   **内存占用：** 模型参数如何加载到内存中，主要是显存。
*   **数据类型与精度：** 不同数值格式（如$FP16$、$INT8$）对内存使用的影响。
*   **量化：** 一种减少模型内存占用的方法。
*   **计算要求：** 理解与模型大小相关的计算负载（$FLOPS$）。
*   **内存带宽：** 为什么内存访问速度对LLM性能很重要。

## 小节

- 1. [模型参数与内存占用](01-%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0%E4%B8%8E%E5%86%85%E5%AD%98%E5%8D%A0%E7%94%A8.md)
- 2. [数据类型与精度 (FP16, INT8)](02-%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B%E4%B8%8E%E7%B2%BE%E5%BA%A6%20%28FP16%2C%20INT8%29.md)
- 3. [量化简介](03-%E9%87%8F%E5%8C%96%E7%AE%80%E4%BB%8B.md)
- 4. [计算需求 (FLOPS)](04-%E8%AE%A1%E7%AE%97%E9%9C%80%E6%B1%82%20%28FLOPS%29.md)
- 5. [内存带宽的重要性](05-%E5%86%85%E5%AD%98%E5%B8%A6%E5%AE%BD%E7%9A%84%E9%87%8D%E8%A6%81%E6%80%A7.md)

章节测验：[在线测验](https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-3-model-size-hardware-connection/quiz)
