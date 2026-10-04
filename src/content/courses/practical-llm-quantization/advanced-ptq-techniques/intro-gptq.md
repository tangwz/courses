---
course: "practical-llm-quantization"
chapter: "advanced-ptq-techniques"
lesson: "intro-gptq"
sourceId: 4624
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques/intro-gptq"
title: "GPTQ介绍"
description: "GPTQ算法概述，用于准确的训练后权重量化。"
order: 1
plots: []
sourceHash: "e10a1d0398d79ff91ea49cac76a309eb8e9ccf3099bd12c2d67d56a426252424"
sourceCorrections: []
---

基本的训练后量化 (quantization)（PTQ）方法，例如MinMax缩放，提供了一种计算开销较低的模型量化方式。然而，当目标是激进的量化级别，如4比特（INT4）甚至3比特整数时，这些简单的方法通常会导致模型精度出现明显的下降。原因是它们通常只根据层内权重 (weight)或激活值的范围或分布来确定量化参数 (parameter)（缩放因子和零点），而不直接考虑量化误差对模型后续计算和最终输出的作用。

GPTQ，代表**通用训练后量化**，旨在解决这一局限。它是一种更精密的PTQ方法，专门用于保持更高精度，尤其是在极低比特宽度下。GPTQ的核心思路是以一种方式进行权重量化，从而使引入到*层输出*中的误差最小化，而不仅仅是最小化原始权重与量化权重之间的误差。

GPTQ不是简单地根据统计数据独立量化层中的所有权重，而是逐层操作。对于每一层，它迭代地选择要量化的权重，并确定它们的量化值（$W_q$），使得原始层输出（$WX$）与量化层输出（$W_qX$）之间的平方误差最小化，使用一小批校准数据（$X$）。从数学上讲，目标是找到使以下表达式最小化的$W_q$:


$$
||W X - W_q X||_2^2
$$


为了高效地实现这一最小化，GPTQ采用近似二阶方法。它使用从层重建误差相对于权重的Hessian矩阵（二阶偏导数矩阵）中获取的信息。这使得GPTQ能够就如何在量化过程中舍入每个权重值做出更明智的决定，考虑层输出对该特定权重变化的敏感性。这个过程比基本校准计算量更大，但比量化感知训练（QAT）的要求低得多，因为它每层只需要处理一次校准数据。

使用GPTQ的主要好处是，与简单的PTQ方法相比，它能够大幅度地保持精度，特别是对于INT4和INT3量化。它通常允许模型量化到这些低比特宽度，同时下游任务的性能退化最小，使其成为在内存和计算资源受限情况下部署大型语言模型的热门选择。GPTQ仍然是一种训练后技术，这意味着它不需要访问原始训练数据集，也不涉及任何模型重训练或微调 (fine-tuning)，只需要少量有代表性的校准数据。

在下一节中，我们将更详细地研究GPTQ算法的机制，阐述它是如何使用Hessian信息并执行逐层量化过程的。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍GPTQ算法的基础论文，详细阐述了其层级重建误差最小化以及使用近似二阶方法进行低位宽量化。
- [Quantization with bitsandbytes and Transformers](https://huggingface.co/docs/transformers/main/en/quantization) — Hugging Face (2024)
  使用Hugging Face Transformers库将包括GPTQ在内的量化技术应用于大型语言模型的实践指南，展示了其在实际中的应用。
