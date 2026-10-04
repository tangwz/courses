---
course: "meta-learning-foundation-models"
chapter: "few-shot-adaptation-foundation-models"
lesson: "peft-vs-meta-learning"
sourceId: 4364
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-5-few-shot-adaptation-foundation-models/peft-vs-meta-learning"
title: "比较 PEFT 与 元学习 方法"
description: "Analyzing the trade-offs between PEFT methods and traditional meta-learning for few-shot adaptation."
order: 5
plots: []
sourceHash: "83e8586cd2c03201a93d0eb9fce55904e9f691faa4280b97268b5899aa2da1af"
sourceCorrections: []
---

参数 (parameter)高效微调 (fine-tuning) (PEFT) 方法，例如适配器 (Adapters)、LoRA 和 提示微调 (Prompt Tuning)，是用于高效适应大型基础模型的途径。元学习策略同样应对着有效少样本适应的挑战。PEFT 方法与元学习策略的比较揭示了它们的独特之处。PEFT 和元学习都旨在为大型基础模型实现有效的少样本适应，但它们在不同假设下运作，并采用不同的机制。了解它们之间的权衡对于为特定适应问题选择合适的策略是必需的。

### 核心机制与目标

元学习从根本上来说，是关于*学习如何学习*。诸如模型无关元学习 (MAML) 及其变体（第 2 章）等技术，旨在找到一个模型初始参数 (parameter) $\theta'$，它能通过少量梯度步长（内循环）快速适应新任务。在元训练（外循环）期间，在任务分布上进行优化的元目标，通常是支持集适应*之后*查询集上的性能。像原型网络（第 3 章）这样的基于度量的方法，学习一个嵌入 (embedding)空间，在这个空间中，可以根据从支持集得出的类别原型距离直接进行分类。这里的目标是学习一个有利于少样本泛化的度量空间。基于优化的观点（第 4 章）通常将其构建为一个双层优化问题，明确地优化适应后的性能。

相反，PEFT 方法不明确包含一个专注于学习适应*过程*的元训练阶段。相反，它们假设存在一个强大的预训练 (pre-training)基础模型 $\theta$，其核心参数保持不变。适应过程只涉及使用其有限数据为目标任务训练一小部分*新*或*修改过*的参数 $\Delta \theta$（例如，适配器层、低秩矩阵、提示嵌入）。目标是在少样本示例上进行标准的监督学习 (supervised learning)（例如，最小化交叉熵），但受限于小的参数集 $\Delta \theta$。PEFT 根本上依赖于这样的假设：基础模型的表示已经非常有效，只需对特定任务进行微小、局部的调整即可。

> 元学习与 PEFT 在少样本适应工作流上的对比。元学习包含一个跨多个任务的独特元训练阶段，而 PEFT 则通过在目标任务上调整一小部分参数来直接适应预训练模型。

### 计算与内存考量

一个主要区别在于计算需求，尤其是在准备阶段。

- **元学习：** 基于梯度的元学习，尤其是涉及标准 MAML 等二阶导数的方法，在元训练期间会产生大量的计算和内存开销。计算元梯度需要通过内循环优化过程进行反向传播 (backpropagation)，通常需要存储中间激活或重新计算它们，这与模型大小的扩展性不佳。一阶近似（FOMAML、Reptile）和隐式微分（iMAML）可以缓解这种情况，但可能会牺牲性能。如果基础编码器是预训练 (pre-training)且冻结的，基于度量的方法在元训练期间通常要求较低，计算主要集中在度量学习方面。对于这两种类型，适应（元测试）的计算成本通常较低。扩展元学习通常需要第 6 章中讨论的复杂技术，例如分布式训练和内存优化策略。
- **PEFT：** PEFT 方法完全避开了昂贵的元训练阶段。主要的计算成本是在目标任务上对小型参数 (parameter)集 $\Delta \theta$ 进行微调 (fine-tuning)。由于 $|\Delta \theta| \ll |\theta|$，此过程比完全微调或基于梯度的元训练快很多，并且需要的内存也少得多。此外，存储适应后的模型效率很高，因为只需要与基础模型一起保存少量的 PEFT 参数 $\Delta \theta$，从而能够高效部署许多特定任务的模型。

### 数据要求与泛化

数据假设也不同：

- **元学习：** 在元训练期间，需要访问相关任务的分布。所学适应策略的有效性在很大程度上取决于这些元训练任务与元测试期间遇到的目标任务的多样性和关联性。元训练任务分布与目标任务之间发生大的偏移会导致适应性能不佳（负迁移）。
- **PEFT：** 不需要元训练任务分布。其有效性取决于预训练 (pre-training)基础模型的质量和通用性。它使用冻结参数 (parameter) $\theta$ 中编码的广泛知识，并仅使用目标少样本任务中的示例进行适应。这使得当缺少多样化训练任务集时，PEFT 更易于应用，前提是存在一个合适的基础模型。

在性能方面，如果元训练任务与目标任务结构和分布高度一致，元学习可能获得更好的结果，因为它明确地针对该范围内的快速适应进行优化。PEFT 提供了一个强大且通常更易于实现的基线，它依赖于基础模型的原始能力。其性能受限于基础模型的预训练特征与目标任务要求的一致程度。

### 灵活性与整合性

元学习提供了一个更通用的框架，即使没有大型预训练 (pre-training)模型也适用，尽管它通常能从这些模型中受益（例如，在预训练嵌入 (embedding)之上进行元学习）。PEFT 本身与大型预训练基础模型的可用性紧密关联。

还值得留意的是，这些方法并非完全互相排斥。本章稍后讨论的混合策略可能会结合两者的元素。例如，可以使用元学习来寻找 PEFT 参数 (parameter)的最佳初始参数，或者在元学习算法的内循环中应用 PEFT 技术，以提高适应效率。

### 选择合适的方法

PEFT 和元学习之间的决定涉及考量几个因素：

1. **元训练数据的可用性：** 如果存在大量且多样化的相关元训练任务，元学习可能会通过明确优化适应过程而提供更高的性能潜力。否则，PEFT 更具实用性。
2. **计算预算：** PEFT 对适应的计算要求通常较低，特别是与基于梯度的元学习的元训练阶段相比。如果用于大量元训练的资源有限，PEFT 更受青睐。
3. **存在强大的基础模型：** PEFT 很大程度上依赖于一个有能力的预训练 (pre-training)模型。如果该方面没有合适的基础模型，元学习（可能从零开始训练或基于较小的基础模型）可能是唯一的选择。
4. **适应速度与准备时间：** 元学习需要在元训练中进行前期投入，但在测试时可能提供非常快的适应速度。PEFT 没有元训练成本，但每个新任务都需要进行特定任务的调整。
5. **存储成本：** PEFT 在高效存储许多特定任务模型方面表现出色，因为只有少量 PEFT 参数 (parameter)不同。通过元学习完全适应的多个模型（如果元学习器输出特定任务参数）的存储成本可能更高，除非元学习器本身（例如，共享初始参数）是主要成果。

实践中，考虑到大型基础模型的普遍性和能力，PEFT 方法因其简单性、高效性和强大的经验性能而变得非常受欢迎，通常在资源受限的情况下作为少样本适应的首选策略。然而，元学习仍然是一种有力的方法，特别是在旨在优化适应机制本身，或在处理学习共同初始参数或度量空间明显有利的任务分布时。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Pages: 1126-1135; DOI: [10.48550/arXiv.1703.03400](https://doi.org/10.48550/arXiv.1703.03400)
  介绍了模型无关元学习 (MAML)，这是一种用于少样本适应的基础优化型元学习算法。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  提出了低秩适应 (LoRA)，这是一种被广泛采用的用于大型基础模型的参数高效微调方法。
