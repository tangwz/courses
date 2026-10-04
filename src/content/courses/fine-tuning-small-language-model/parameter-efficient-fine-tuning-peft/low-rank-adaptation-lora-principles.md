---
course: "fine-tuning-small-language-model"
chapter: "parameter-efficient-fine-tuning-peft"
lesson: "low-rank-adaptation-lora-principles"
sourceId: 8513
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-4-parameter-efficient-fine-tuning-peft/low-rank-adaptation-lora-principles"
title: "低秩自适应 (LoRA) 原理"
description: "学习低秩自适应的数学原理以及它如何减少可训练参数的数量。"
order: 2
plots: ["plots/8513-0.json"]
sourceHash: "ff29ff611c069c7c968e7552cfa10d5c6b76471edc5520ad338a276b0985b527"
sourceCorrections: []
---

分析语言模型学习的数学原理对于理解参数 (parameter)缩减很有必要。在标准训练中，更新稠密层需要学习一个完整的权重 (weight)变化矩阵，记为 $\Delta W$。如果预训练 (pre-training)权重矩阵 $W$ 的维度是 $d \times k$，那么更新矩阵 $\Delta W$ 的维度也必须是 $d \times k$。对于隐藏层维度达数千的现代语言模型，单个权重矩阵就可能包含数千万个参数。

研究人员发现，过度参数化的神经网络 (neural network)具有较低的本征维度。这意味着，虽然模型在预训练期间需要数十亿个参数来学习通用的语言表征，但在适配特定的下游任务时，并不需要整个参数空间。所需的权重更新可以在更低维的空间中得到准确表示。低秩自适应应用这一原理来减轻训练的计算负担。

LoRA 并非计算和存储庞大的 $\Delta W$ 矩阵，而是冻结原始矩阵 $W$，并使用矩阵分解来近似更新。更新矩阵 $\Delta W$ 被分解为两个较小的矩阵 $A$ 和 $B$。

$W_{new} = W + \Delta W = W + BA$

在此等式中，$W \in \mathbb{R}^{d \times k}$。我们定义一个秩参数 $r$，其中 $r \ll \min(d, k)$。分解过程创建了矩阵 $A \in \mathbb{R}^{r \times k}$ 和矩阵 $B \in \mathbb{R}^{d \times r}$。当 $B$ 与 $A$ 相乘时，结果矩阵的维度与原始的 $d \times k$ 一致，使其能够与原始权重矩阵 $W$ 进行逐元素相加。

当计算可训练参数量时，这种方法的数学效率显而易见。假设 Transformer 中的一个线性层具有输入维度 $k = 4096$ 和输出维度 $d = 4096$。

标准微调 (fine-tuning)需要更新整个 $4096 \times 4096$ 矩阵。

$4096 \times 4096 = 16,777,216 \text{ 个参数}$

如果应用秩 $r = 8$ 的 LoRA，则只需训练矩阵 $A$ 和 $B$。矩阵 $A$ 的维度为 $8 \times 4096$，矩阵 $B$ 的维度为 $4096 \times 8$。

$(4096 \times 8) + (8 \times 4096) = 32,768 + 32,768 = 65,536 \text{ 个参数}$



![参数量：全量更新对比 LoRA](plots/8513-0.json)



> 标准权重更新与低秩适配器配置之间的可训练参数量对比。

相比近 1700 万个参数，仅更新 65,536 个参数，使该层的可训练参数减少了 99.6% 以上。由于 Adam 等现代优化器会为每个可训练参数存储梯度的运行平均值，这种参数量的大幅减少直接降低了对 GPU 显存 (VRAM)的需求。

在训练的前向传播过程中，模型同时通过冻结权重和可训练的适配器矩阵处理输入向量 (vector) $x$。该操作表示为：

$h = Wx + BAx$

> 使用低秩自适应矩阵的 Transformer 层中的数据流。

这些矩阵的初始化经过专门设计以确保训练稳定性。矩阵 $A$ 使用随机高斯分布初始化。矩阵 $B$ 初始化为零。由于矩阵 $B$ 初始全为零，乘积 $BA$ 在训练开始时恰好为零。这保证了在第一个训练步中 $\Delta W = 0$，意味着在发生第一次权重更新之前，网络的行为与未经修改的基础模型完全一致。

LoRA 还引入了一个缩放因子 $\alpha$ (alpha) 来管理权重更新的幅度。在加到基础权重之前，$B$ 和 $A$ 的乘积会乘以 $\alpha$ 与 $r$ 的比值。

$\Delta W = \frac{\alpha}{r} BA$

这种缩放机制确保了在超参数 (hyperparameter)调优期间更改秩 $r$ 时，不需要大幅调整学习率。如果你为了捕捉自定义数据集中更复杂的模式而增加秩，$\alpha / r$ 的比值会归一化 (normalization)初始梯度，使不同配置下的学习过程保持数学上的稳定。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=nZeVKeeFYf9) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2022)
  Journal: The Tenth International Conference on Learning Representations (ICLR); Publisher: OpenReview.net; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍 LoRA 架构的原始研究论文，提供了数学推导和实证结果。
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://aclanthology.org/2021.acl-long.568/) — Armen Aghajanyan, Sonal Gupta, Luke Zettlemoyer (2021)
  Journal: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Pages: 7319-7328; DOI: [10.18653/v1/2021.acl-long.568](https://doi.org/10.18653/v1/2021.acl-long.568)
  研究并确立了语言模型中的低本征维度概念，为 LoRA 提供了理论依据。
- [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods](https://github.com/huggingface/peft) — Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, Benjamin Bossan (2022)
  Publisher: Hugging Face
  在 PyTorch 环境中实现 LoRA 最广泛使用的官方库文档和代码仓库。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: Advances in Neural Information Processing Systems; Volume: 36; Pages: 10088-10115; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  LoRA 原理的扩展，引入了量化技术以进一步降低微调时的显存需求。
