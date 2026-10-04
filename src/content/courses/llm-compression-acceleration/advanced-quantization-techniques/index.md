---
course: "llm-compression-acceleration"
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-2-advanced-quantization-techniques"
sourceId: 847
chapter: "advanced-quantization-techniques"
title: "进阶量化技术"
order: 2
description: "为大语言模型实施和分析进阶量化方法，例如PTQ、QAT和低于4比特精度。"
hasQuiz: false
---

在明确了高效部署大型语言模型的主要挑战之后，本章重点关注量化，这是一种减小模型大小和加速推理的主要方法。尽管基本的量化带来了好处，但要在不牺牲模型准确性的前提下取得显著成效，则需要更精细的方法。

本章将超越入门知识，介绍专门应用于大型语言模型的进阶量化方法。您将学习如何：

*   实施并分析专为大型Transformer模型设计的进阶训练后量化（PTQ）校准方法和量化感知训练（QAT）。
*   研究极限量化技术，将精度推至4比特以下（例如NF4、FP4），并考察二值/三值表示。
*   设计混合精度策略，将不同级别的量化（$int8$、$fp4$、$int4$等）应用于模型的不同组成部分。
*   理解特定硬件特性如何加速量化计算。
*   建立严格的规程，用于评估压缩、推理速度和模型准确性之间的权衡。

我们将考察这些技术的实际应用，包括异常值处理、准确性保持以及量化与硬件加速器之间的配合。本章最后安排了一个实施PTQ和QAT的动手练习。
