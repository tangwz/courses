# 第 2 章：训练后量化 (PTQ)

来源：[原章节](https://apxml.com/zh/courses/practical-llm-quantization/chapter-2-post-training-quantization-ptq)

[返回课程目录](../README.md)

训练后量化 (PTQ) 是直接对已训练好的模型进行量化。这种做法避免了计算成本高昂的再训练，因此成为减小模型大小和提升推理速度的有效选择。PTQ 的工作方式是将模型的权重（有时也包括激活值）从像 $FP32$ 这样的高精度格式转换为 $INT8$ 或 $INT4$ 等低精度整数类型。

本章将介绍 PTQ 的实际应用方面：

*   基本原理和工作流程。
*   校准数据在确定量化范围中的作用。
*   静态和动态量化方法的区别。
*   计算量化参数的常用算法。
*   量化过程中处理异常值的技术。
*   了解基本 PTQ 方法固有的精度局限。
*   通过一个实际操作的例子实现静态 PTQ。

## 小节

- 1. [训练后量化原理](01-%E8%AE%AD%E7%BB%83%E5%90%8E%E9%87%8F%E5%8C%96%E5%8E%9F%E7%90%86.md)
- 2. [校准：选择有代表性的数据](02-%E6%A0%A1%E5%87%86%EF%BC%9A%E9%80%89%E6%8B%A9%E6%9C%89%E4%BB%A3%E8%A1%A8%E6%80%A7%E7%9A%84%E6%95%B0%E6%8D%AE.md)
- 3. [静态量化与动态量化](03-%E9%9D%99%E6%80%81%E9%87%8F%E5%8C%96%E4%B8%8E%E5%8A%A8%E6%80%81%E9%87%8F%E5%8C%96.md)
- 4. [常见的 PTQ 算法](04-%E5%B8%B8%E8%A7%81%E7%9A%84%20PTQ%20%E7%AE%97%E6%B3%95.md)
- 5. [处理 PTQ 中的异常值](05-%E5%A4%84%E7%90%86%20PTQ%20%E4%B8%AD%E7%9A%84%E5%BC%82%E5%B8%B8%E5%80%BC.md)
- 6. [将PTQ应用于LLM层](06-%E5%B0%86PTQ%E5%BA%94%E7%94%A8%E4%BA%8ELLM%E5%B1%82.md)
- 7. [基础PTQ的局限性](07-%E5%9F%BA%E7%A1%80PTQ%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 8. [动手实践：应用静态PTQ](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BA%94%E7%94%A8%E9%9D%99%E6%80%81PTQ.md)

章节测验：[在线测验](https://apxml.com/zh/courses/practical-llm-quantization/chapter-2-post-training-quantization-ptq/quiz)
