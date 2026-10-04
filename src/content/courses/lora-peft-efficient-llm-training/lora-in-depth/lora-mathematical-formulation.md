---
course: "lora-peft-efficient-llm-training"
chapter: "lora-in-depth"
lesson: "lora-mathematical-formulation"
sourceId: 4118
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-2-lora-in-depth/lora-mathematical-formulation"
title: "LoRA的数学表述"
description: "阐述定义LoRA更新矩阵的数学方程以及它们在模型权重上的使用。"
order: 2
plots: []
sourceHash: "edd7555b607192abe30daeb9e13e5125bc14e6024a51794bb6b37a3e3728d160"
sourceCorrections: []
---

低秩适应（LoRA）引入了一种特定的数学结构，其设计理念是基于适应过程中权重 (weight)更新具有低内在秩的假设，旨在提升参数 (parameter)效率。在标准微调 (fine-tuning)中，通过添加一个增量矩阵$\Delta W \in \mathbb{R}^{d \times k}$来更新预训练 (pre-training)权重矩阵$W_0 \in \mathbb{R}^{d \times k}$，从而得到适应后的权重$W = W_0 + \Delta W$。这种传统训练方法需要学习$\Delta W$中的所有$d \times k$个参数。

LoRA提出了一种不同的方法。我们不直接学习可能很大的$\Delta W$，而是使用低秩分解对其进行近似。具体来说，$\Delta W$由两个较小矩阵$B \in \mathbb{R}^{d \times r}$和$A \in \mathbb{R}^{r \times k}$的乘积表示：


$$
\Delta W \approx BA
$$


这里，$r$是分解的秩，LoRA的主要思想是$r \ll \min(d, k)$。这个约束大大减少了我们需要学习的参数数量。原始权重$W_0$保持冻结（训练期间不更新），而矩阵$A$和$B$包含表示任务特定适应的可训练参数。

考虑通过LoRA修改的层的正向传播。对于输入$x$，原始输出是$h = W_0 x$。加入LoRA更新后的修改输出变为：


$$
h = W_0 x + \Delta W x = W_0 x + B A x
$$


在使用LoRA进行微调时，只有矩阵$A$和$B$中的参数通过梯度下降 (gradient descent)进行更新。原始权重$W_0$保持不变。

为了更好地控制适应过程，LoRA引入了一个常数缩放因子$\alpha$。这个标量调节由$BA$施加的更新的幅度。通常的做法是将更新按$\frac{\alpha}{r}$进行缩放。这种归一化 (normalization)有助于稳定训练，尤其是在改变秩$r$时。LoRA修改层的最终正向传播方程是：


$$
h = W_0 x + \frac{\alpha}{r} B A x
$$


我们来分析参数效率的提升。完整微调需要为$\Delta W$矩阵学习$d \times k$个参数。使用LoRA，我们只需要学习$A$和$B$中的参数。LoRA中可训练参数的总数是$A$ ($r \times k$)和$B$ ($d \times r$)中参数的总和，即$r(d + k)$。由于$r$通常远小于$d$和$k$，可训练参数的数量显著减少。例如，如果$d=4096$、$k=4096$且$r=8$，完整微调大约需要$16.7$百万个参数，而LoRA对于该特定层仅需要$8 \times (4096 + 4096) = 65,536$个参数。这表示参数减少了99%以上。

该结构可以被视为原始权重矩阵添加了一个并行路径：

> 经过LoRA修改层的正向传播。输入$x$通过冻结权重$W_0$，同时并行地通过可训练的低秩矩阵$A$和$B$。低秩路径在添加到原始输出之前，会按$\alpha/r$进行缩放。

关于初始化，一种常见策略是使用随机高斯值初始化$A$，并用零初始化$B$。这确保了$\Delta W = BA$在训练开始时为零，这意味着适应后的模型与预训练模型$W_0$的起始状态完全一致。缩放因子$\alpha$通常设置为与$r$的初始值匹配，尽管它也可以作为一个超参数 (hyperparameter)来处理。我们将在后续章节（秩选择策略、缩放参数Alpha、LoRA初始化策略）中更详细地分析初始化和超参数选择，例如$r$和$\alpha$。

总而言之，LoRA的数学表述提供了一种具体机制，用于通过低秩结构$BA$来近似权重更新$\Delta W$。通过冻结原始权重$W_0$并仅训练小型矩阵$A$和$B$，LoRA实现了显著的参数效率，大幅减少了微调大型模型所需的计算和内存开销，同时力求保持适应能力。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=qV-UnyfA_xQ) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2022)
  Journal: International Conference on Learning Representations (ICLR); Publisher: International Conference on Learning Representations (ICLR)
  介绍 LoRA 的原始论文，详细阐述其数学公式、参数效率和实验验证。
- [What is LoRA?](https://huggingface.co/docs/peft/en/conceptual_guides/lora) — Hugging Face (2024)
  Publisher: Hugging Face
  官方文档，提供了 LoRA 机制在流行的 PEFT 库中的实践指南和解释。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，涵盖深度学习的数学先决条件，包括与 LoRA 低秩近似相关的线性代数概念（如矩阵分解）。
