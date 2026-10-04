---
course: "practical-llm-quantization"
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-2-post-training-quantization-ptq"
sourceId: 888
chapter: "post-training-quantization-ptq"
title: "训练后量化 (PTQ)"
order: 2
description: "学习将 PTQ 应用于大型语言模型的方法，包括校准、静态与动态量化对比以及常用算法。"
hasQuiz: true
---

训练后量化 (PTQ) 是直接对已训练好的模型进行量化。这种做法避免了计算成本高昂的再训练，因此成为减小模型大小和提升推理速度的有效选择。PTQ 的工作方式是将模型的权重（有时也包括激活值）从像 $FP32$ 这样的高精度格式转换为 $INT8$ 或 $INT4$ 等低精度整数类型。

本章将介绍 PTQ 的实际应用方面：

*   基本原理和工作流程。
*   校准数据在确定量化范围中的作用。
*   静态和动态量化方法的区别。
*   计算量化参数的常用算法。
*   量化过程中处理异常值的技术。
*   了解基本 PTQ 方法固有的精度局限。
*   通过一个实际操作的例子实现静态 PTQ。
