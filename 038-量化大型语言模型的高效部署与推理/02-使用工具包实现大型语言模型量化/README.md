# 第 2 章：使用工具包实现大型语言模型量化

来源：[原章节](https://apxml.com/zh/courses/quantized-llm-deployment/chapter-2-implementing-llm-quantization-toolkits)

[返回课程目录](../README.md)

在上一章学习了高级量化技术的理论后，本节将侧重于使用常用软件库进行这些方法的实际操作。目标是让您从理解低比特量化（例如$INT4$）等原理以及GPTQ和AWQ等算法，转向在大型语言模型上实际执行这些操作。

您将使用专为LLM量化设计的广泛使用的工具包：

*   学习`bitsandbytes`如何支持高效的低比特操作。
*   了解如何使用`Transformers`和`Accelerate`直接在Hugging Face生态系统内集成量化。
*   分别使用`AutoGPTQ`和`AutoAWQ`库来应用GPTQ和AWQ。

在本章中，我们将介绍使用这些工具量化模型的必要步骤，考察如何比较不同库获得的结果和性能特点，并处理模型和工具包之间可能出现的兼容性挑战。学习结束时，您将拥有使用这些库准备大型语言模型以进行高效部署的实践经验。

## 小节

- 1. [LLM量化库概览](01-LLM%E9%87%8F%E5%8C%96%E5%BA%93%E6%A6%82%E8%A7%88.md)
- 2. [使用 bitsandbytes 进行低位操作](02-%E4%BD%BF%E7%94%A8%20bitsandbytes%20%E8%BF%9B%E8%A1%8C%E4%BD%8E%E4%BD%8D%E6%93%8D%E4%BD%9C.md)
- 3. [使用 Hugging Face Transformers 和 Accelerate 实现量化](03-%E4%BD%BF%E7%94%A8%20Hugging%20Face%20Transformers%20%E5%92%8C%20Accelerate%20%E5%AE%9E%E7%8E%B0%E9%87%8F%E5%8C%96.md)
- 4. [使用 AutoGPTQ 应用 GPTQ](04-%E4%BD%BF%E7%94%A8%20AutoGPTQ%20%E5%BA%94%E7%94%A8%20GPTQ.md)
- 5. [使用 AutoAWQ 应用 AWQ](05-%E4%BD%BF%E7%94%A8%20AutoAWQ%20%E5%BA%94%E7%94%A8%20AWQ.md)
- 6. [比较工具包的输出和性能](06-%E6%AF%94%E8%BE%83%E5%B7%A5%E5%85%B7%E5%8C%85%E7%9A%84%E8%BE%93%E5%87%BA%E5%92%8C%E6%80%A7%E8%83%BD.md)
- 7. [处理模型兼容性问题](07-%E5%A4%84%E7%90%86%E6%A8%A1%E5%9E%8B%E5%85%BC%E5%AE%B9%E6%80%A7%E9%97%AE%E9%A2%98.md)
- 8. [实践：使用多种工具包量化模型](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%E5%A4%9A%E7%A7%8D%E5%B7%A5%E5%85%B7%E5%8C%85%E9%87%8F%E5%8C%96%E6%A8%A1%E5%9E%8B.md)
