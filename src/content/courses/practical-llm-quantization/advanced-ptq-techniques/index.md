---
course: "practical-llm-quantization"
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques"
sourceId: 890
chapter: "advanced-ptq-techniques"
title: "进阶PTQ技术"
order: 3
description: "了解进阶训练后量化方法，如GPTQ、AWQ和SmoothQuant，以提升LLM精度。"
hasQuiz: true
---

尽管训练后量化（PTQ）提供了一种简单直接的方法来减小模型大小并加速推理，但之前讨论的基础方法有时可能会导致明显的精度下降，尤其是在追求极低精度（如4比特整数$INT4$）时。简单的校准可能无法捕获足够的信息来有效保持模型性能。

本章侧重于专门为实现更高精度而开发的进阶PTQ技术，这些技术通常能在不重新训练的情况下，使量化模型表现接近原始全精度模型。

您将学习到：

*   **GPTQ（通用PTQ）：** 了解该方法如何使用近似的二阶信息（Hessian）逐层更准确地量化权重。
*   **AWQ（激活感知权重量化）：** 学习AWQ如何根据激活值的大小识别并保留重要权重，通过缩放权重使量化更简单。
*   **SmoothQuant：** 研究这项技术，它通过平滑权重和激活之间的分布，来解决量化具有大异常值的激活所面临的难题。
*   **比较与实现：** 分析这些方法之间的权衡，并讨论应用它们时的实际考量。
*   **动手实践：** 应用GPTQ算法量化大型语言模型。

在本章结束时，您将理解这些进阶技术背后的原理，并能够应用它们以实现更高效的LLM量化。
