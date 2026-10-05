# 第 5 章：估算硬件需求

来源：[原章节](https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-5-estimating-hardware-needs)

[返回课程目录](../README.md)

在明确了LLM大小与所需硬件组成部分之间的联系后，本章将介绍估算这些需求的方法。您将学习如何估算运行特定大型语言模型所需的内存量，特别是显存（VRAM）。

我们将介绍一个常用的经验法则，它将模型参数与显存使用量关联起来，同时考虑$FP16$等数据类型。通常采用一个简化的计算公式：
$$Required \, VRAM \approx Parameter \, Count \times Bytes \, Per \, Parameter$$
除了这个初步估算，我们还将讨论影响内存需求的其他因素，例如处理过程中的激活内存占用、上下文长度以及批处理大小的影响。您还将学习如何检查自己系统的硬件规格，并通过实际例子应用这些估算技术。本章将为您提供实用的工具，以便在运行大型语言模型之前评估硬件需求。

## 小节

- 1. [显存需求估算：参数量经验法则](01-%E6%98%BE%E5%AD%98%E9%9C%80%E6%B1%82%E4%BC%B0%E7%AE%97%EF%BC%9A%E5%8F%82%E6%95%B0%E9%87%8F%E7%BB%8F%E9%AA%8C%E6%B3%95%E5%88%99.md)
- 2. [考虑激活内存](02-%E8%80%83%E8%99%91%E6%BF%80%E6%B4%BB%E5%86%85%E5%AD%98.md)
- 3. [影响实际使用量的因素](03-%E5%BD%B1%E5%93%8D%E5%AE%9E%E9%99%85%E4%BD%BF%E7%94%A8%E9%87%8F%E7%9A%84%E5%9B%A0%E7%B4%A0.md)
- 4. [检查硬件配置](04-%E6%A3%80%E6%9F%A5%E7%A1%AC%E4%BB%B6%E9%85%8D%E7%BD%AE.md)
- 5. [实践：简单的显存估算](05-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%AE%80%E5%8D%95%E7%9A%84%E6%98%BE%E5%AD%98%E4%BC%B0%E7%AE%97.md)

章节测验：[在线测验](https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-5-estimating-hardware-needs/quiz)
