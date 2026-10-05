# 第 5 章：MoE在现代架构中的应用

来源：[原章节](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-5-integrating-moe-into-architectures)

[返回课程目录](../README.md)

前几章阐明了单个专家混合层的工作原理。现在，我们将讨论如何将其融入现代神经网络架构，实现其实际应用。MoE最常见的用途是增加模型容量，同时不按比例增加计算开销，这通常通过用MoE层替换标准前馈网络（FFN）来实现。

本章将为这一融入过程提供技术指南。本章将涉及：

*   在Transformer模型中，用稀疏MoE层替换密集型FFN块的具体方法。
*   架构方面的考虑，包括为实现最佳性能而放置MoE层的频率和深度。
*   MoE理念在视觉Transformer (ViT)和多模态系统中的适配。
*   模型参数与计算量（FLOPs）之间平衡关系的定量分析。

MoE的一个主要优势在于将总参数与单次前向传播所需的计算分离。一个MoE模型可能包含 $N$ 个专家，但对于任何给定的词元，门控网络会将其路由到少数 $k$ 个专家子集，其中 $k \ll N$。因此，总计算开销是 $k$ 的函数，而模型的总参数量是 $N$ 的函数。这种关系可以表示为：

$$
\text{Total Parameters} \propto N \times \text{Parameters per Expert}
$$

$$
\text{Computational FLOPs} \propto k \times \text{FLOPs per Expert}
$$

本章最后会有一个动手练习，你将修改一个标准Transformer实现以使用稀疏MoE层，从而将这些架构理念付诸实践。

## 小节

- 1. [将FFN替换为Transformer中的MoE层](01-%E5%B0%86FFN%E6%9B%BF%E6%8D%A2%E4%B8%BATransformer%E4%B8%AD%E7%9A%84MoE%E5%B1%82.md)
- 2. [MoE 层的位置：频率与部位](02-MoE%20%E5%B1%82%E7%9A%84%E4%BD%8D%E7%BD%AE%EF%BC%9A%E9%A2%91%E7%8E%87%E4%B8%8E%E9%83%A8%E4%BD%8D.md)
- 3. [视觉Transformer (ViT) 中的MoE](03-%E8%A7%86%E8%A7%89Transformer%20%28ViT%29%20%E4%B8%AD%E7%9A%84MoE.md)
- 4. [多模态模型中的MoE](04-%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B%E4%B8%AD%E7%9A%84MoE.md)
- 5. [架构变体及其特性](05-%E6%9E%B6%E6%9E%84%E5%8F%98%E4%BD%93%E5%8F%8A%E5%85%B6%E7%89%B9%E6%80%A7.md)
- 6. [分析参数与FLOPs的权衡](06-%E5%88%86%E6%9E%90%E5%8F%82%E6%95%B0%E4%B8%8EFLOPs%E7%9A%84%E6%9D%83%E8%A1%A1.md)
- 7. [实践：修改Transformer模型以使用MoE](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BF%AE%E6%94%B9Transformer%E6%A8%A1%E5%9E%8B%E4%BB%A5%E4%BD%BF%E7%94%A8MoE.md)
