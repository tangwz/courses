# 第 2 章：低秩适配 (LoRA) 详解

来源：[原章节](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-2-lora-in-depth)

[返回课程目录](../README.md)

第1章概述了微调大型语言模型所涉及的计算和内存挑战。本章将通过着重介绍低秩适配 (LoRA) 来解决这些问题，LoRA是一种特定且广泛使用的参数高效微调 (PEFT) 技术。

我们从 LoRA 背后的核心假设开始：模型权重在适配过程中产生的变化 $\Delta W$ 具有较低的“内在秩”。这意味着 $\Delta W$ 可以通过两个小得多的矩阵 $B$ 和 $A$ 的乘积来有效近似。

$$ \Delta W \approx BA $$

这里 $W \in \mathbb{R}^{d \times k}$，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，并且秩 $r \ll \min(d, k)$。

在本章中，您将学到：

*   LoRA 的理论依据和数学表述。
*   LoRA 如何将权重更新分解为低秩矩阵（$A$ 和 $B$）。
*   选择秩 $r$ 和配置缩放参数 $\alpha$ 的实际考量。
*   如何在神经网络层内，特别是线性层中实现 LoRA 修改。
*   将 LoRA 集成到标准 Transformer 模块中的方法，针对注意力机制或前馈网络等特定组件。

通过本章的学习，您将理解 LoRA 的运行机制，并能够实现其基本形式，以进行高效的 LLM 微调。

## 小节

- 1. [LoRA 假说：适配的低本征秩](01-LoRA%20%E5%81%87%E8%AF%B4%EF%BC%9A%E9%80%82%E9%85%8D%E7%9A%84%E4%BD%8E%E6%9C%AC%E5%BE%81%E7%A7%A9.md)
- 2. [LoRA的数学表述](02-LoRA%E7%9A%84%E6%95%B0%E5%AD%A6%E8%A1%A8%E8%BF%B0.md)
- 3. [权重更新矩阵的分解](03-%E6%9D%83%E9%87%8D%E6%9B%B4%E6%96%B0%E7%9F%A9%E9%98%B5%E7%9A%84%E5%88%86%E8%A7%A3.md)
- 4. [秩选择策略](04-%E7%A7%A9%E9%80%89%E6%8B%A9%E7%AD%96%E7%95%A5.md)
- 5. [缩放参数 Alpha](05-%E7%BC%A9%E6%94%BE%E5%8F%82%E6%95%B0%20Alpha.md)
- 6. [LoRA 层的实施](06-LoRA%20%E5%B1%82%E7%9A%84%E5%AE%9E%E6%96%BD.md)
- 7. [将 LoRA 融入 Transformer 架构](07-%E5%B0%86%20LoRA%20%E8%9E%8D%E5%85%A5%20Transformer%20%E6%9E%B6%E6%9E%84.md)
- 8. [实际操作：应用基础LoRA](08-%E5%AE%9E%E9%99%85%E6%93%8D%E4%BD%9C%EF%BC%9A%E5%BA%94%E7%94%A8%E5%9F%BA%E7%A1%80LoRA.md)
