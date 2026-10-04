# LoRA 假说：适配的低本征秩

来源：[原文](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-2-lora-in-depth/lora-hypothesis-intrinsic-rank)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型进行全量微调 (fine-tuning)需要更新所有参数 (parameter)，这导致巨大的计算和内存开销。这使得适配大型模型通常不切实际，特别是在面向多个下游任务或资源受限的情况下。LoRA 提供了一个有吸引力的选择，它基于一个关于模型适配性质的特定假说来运行。

LoRA 背后的核心思想，由 Hu 等人（2021）提出，是**低本征秩假说**。该假说认为，在将大型预训练 (pre-training)模型适配到特定下游任务时，权重 (weight)矩阵（$\Delta W$）中的*变化*不需要涵盖原始权重矩阵（$W$）的全部维度。相反，适配主要发生在一个低得多的维度子空间内。数学上，尽管更新矩阵 $\Delta W = W' - W$ 与原始权重矩阵 $W \in \mathbb{R}^{d \times k}$ 具有相同的维度，但其有效“秩”可能显著小得多。

可以这样理解：一个大型预训练模型已经在其权重中包含了大量的通用知识。当针对特定任务进行微调时，例如情感分析或代码生成，我们并没有从根本上改变其对语言的核心理解。相反，我们是在引导其现有能力适应目标任务的特点和特定模式。低本征秩假说表明，这种引导，这种适配增量，可以通过在**高维权重空间中**沿着相对较少的方向或维度修改原始权重来有效表示。

这一理念与线性代数和矩阵分析中的发现有相似之处。一个矩阵通常可以由另一个低秩矩阵很好地近似。你在第 1 章中遇到的奇异值分解（SVD）为寻找任意给定矩阵的最佳低秩近似提供了理论基础。尽管 LoRA 在训练期间不会显式计算 $\Delta W$ 的 SVD（因为 $\Delta W$ 本身就是我们旨在高效学习的内容），但其潜在原则是相似的：我们假定 $\Delta W$ *可以*以低秩形式表示。

如果变化量 $\Delta W$ 确实具有低本征秩 $r$，其中 $r \ll \min(d, k)$，那么它可以由两个较小矩阵的乘积有效近似：

$\Delta W \approx BA$

在此，$B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$。LoRA 提出不学习整个 $\Delta W$ 的 $d \times k$ 个参数，而是仅学习 $B$ 和 $A$ 中的参数。$B$ 和 $A$ 中参数的总数为 $d \times r + r \times k = r(d+k)$。由于 $r$ 的选择远小于 $d$ 和 $k$，可训练参数的数量 $r(d+k)$ 会远小于 $d \times k$。

例如，考虑一个线性层，其中 $d=4096$，$k=4096$。全量更新的 $\Delta W$ 有超过 1600 万个参数。如果我们假定适配可以通过秩 $r=8$ 来表示，LoRA 矩阵 $B$ 和 $A$ 将共有 $8 \times (4096 + 4096) = 65,536$ 个参数。这表示针对该特定层的适配，需要训练和存储的参数数量减少了 99% 以上。

这种低秩假说是 LoRA 高效性的根本。通过假定任务特定的适配存在于一个低秩子空间中，LoRA 用其低秩因子 $B$ 和 $A$ 的优化来替代 $\Delta W$ 的直接优化，同时保持原始模型权重 $W$ 不变。随后的章节将详细介绍基于这一基本假定的精确数学公式和实现策略。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  这篇原创论文介绍了低秩适应 (LoRA) 方法及其潜在的低固有秩假设。

---

[上一节](../01-%E5%9B%9E%E9%A1%BE%E5%BE%AE%E8%B0%83%E4%B8%8E%E6%95%88%E7%8E%87%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7/04-%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%96%B9%E6%B3%95%E7%9A%84%E5%88%86%E7%B1%BB.md) · [下一节](02-LoRA%E7%9A%84%E6%95%B0%E5%AD%A6%E8%A1%A8%E8%BF%B0.md)
