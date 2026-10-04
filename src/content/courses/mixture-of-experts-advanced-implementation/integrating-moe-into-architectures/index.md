---
course: "mixture-of-experts-advanced-implementation"
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-5-integrating-moe-into-architectures"
sourceId: 1236
chapter: "integrating-moe-into-architectures"
title: "MoE在现代架构中的应用"
order: 5
description: "了解在何处以及如何将MoE层融入Transformer、视觉Transformer和多模态模型以提升性能。"
hasQuiz: false
---

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
