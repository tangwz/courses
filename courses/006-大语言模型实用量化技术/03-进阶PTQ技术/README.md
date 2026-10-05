# 第 3 章：进阶PTQ技术

来源：[原章节](https://apxml.com/zh/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques)

[返回课程目录](../README.md)

尽管训练后量化（PTQ）提供了一种简单直接的方法来减小模型大小并加速推理，但之前讨论的基础方法有时可能会导致明显的精度下降，尤其是在追求极低精度（如4比特整数$INT4$）时。简单的校准可能无法捕获足够的信息来有效保持模型性能。

本章侧重于专门为实现更高精度而开发的进阶PTQ技术，这些技术通常能在不重新训练的情况下，使量化模型表现接近原始全精度模型。

您将学习到：

*   **GPTQ（通用PTQ）：** 了解该方法如何使用近似的二阶信息（Hessian）逐层更准确地量化权重。
*   **AWQ（激活感知权重量化）：** 学习AWQ如何根据激活值的大小识别并保留重要权重，通过缩放权重使量化更简单。
*   **SmoothQuant：** 研究这项技术，它通过平滑权重和激活之间的分布，来解决量化具有大异常值的激活所面临的难题。
*   **比较与实现：** 分析这些方法之间的权衡，并讨论应用它们时的实际考量。
*   **动手实践：** 应用GPTQ算法量化大型语言模型。

在本章结束时，您将理解这些进阶技术背后的原理，并能够应用它们以实现更高效的LLM量化。

## 小节

- 1. [GPTQ介绍](01-GPTQ%E4%BB%8B%E7%BB%8D.md)
- 2. [理解 GPTQ 算法机制](02-%E7%90%86%E8%A7%A3%20GPTQ%20%E7%AE%97%E6%B3%95%E6%9C%BA%E5%88%B6.md)
- 3. [AWQ：激活感知权重量化](03-AWQ%EF%BC%9A%E6%BF%80%E6%B4%BB%E6%84%9F%E7%9F%A5%E6%9D%83%E9%87%8D%E9%87%8F%E5%8C%96.md)
- 4. [SmoothQuant：减轻激活离群值](04-SmoothQuant%EF%BC%9A%E5%87%8F%E8%BD%BB%E6%BF%80%E6%B4%BB%E7%A6%BB%E7%BE%A4%E5%80%BC.md)
- 5. [高级PTQ方法比较](05-%E9%AB%98%E7%BA%A7PTQ%E6%96%B9%E6%B3%95%E6%AF%94%E8%BE%83.md)
- 6. [高级PTQ的实施考量](06-%E9%AB%98%E7%BA%A7PTQ%E7%9A%84%E5%AE%9E%E6%96%BD%E8%80%83%E9%87%8F.md)
- 7. [动手实践：使用 GPTQ 进行量化](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20GPTQ%20%E8%BF%9B%E8%A1%8C%E9%87%8F%E5%8C%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques/quiz)
