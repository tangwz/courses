---
course: "lora-peft-efficient-llm-training"
chapter: "peft-methodologies-survey"
lesson: "prompt-tuning-variations"
sourceId: 4137
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-3-peft-methodologies-survey/prompt-tuning-variations"
title: "提示词微调与P-Tuning变体"
description: "讨论提示词微调及其变体（如P-Tuning），侧重于可学习的提示词嵌入。"
order: 4
plots: []
sourceHash: "10db96dac5b1fe14770f2a82f32f8cb6471b718e54ff53ab81758d0e60e7fc5a"
sourceCorrections: []
---

提示词 (prompt)微调 (fine-tuning)提供了一种独特的模型行为引导策略。与那些通过修改或增加Transformer层中的组件来进行模型微调的方法不同，提示词微调通过操作模型的输入来实现这一点，而完全不改变基础模型的内部权重 (weight)。此方法简单但对许多任务来说非常有效，是现有参数 (parameter)效率最高的策略之一。

### 主要原理：可学习的软提示词 (prompt)

传统提示词涉及在实际输入前添加离散的文本指令（例如：“将英语翻译成法语：”）。提示词微调 (fine-tuning)用*可学习的连续向量 (vector)*取代了这些手动编写的文本提示词，这些向量通常被称为“软提示词”或“提示词嵌入 (embedding)”。

想象一下，你想对LLM进行摘要任务的微调。你不是仅仅输入要摘要的文本，而是在其前面添加一个包含$k$个特殊提示词嵌入的序列：

$[P_1, P_2, ..., P_k, E(x_1), E(x_2), ..., E(x_n)]$

此处：

- $E(x_i)$ 表示第$i$个输入词元 (token)$x_i$的标准嵌入。
- $P_j$ 是第$j$个可训练的提示词嵌入向量。每个$P_j$与词嵌入具有相同的维度（例如，$d_{model}$）。
- $k$ 是一个指示软提示词长度的超参数 (parameter) (hyperparameter)。

在微调过程中，*唯一*被更新的参数是这$k$个提示词向量$P_1, ..., P_k$。所有原始LLM参数（注意力层、前馈网络中的权重 (weight)，嵌入查找表$E$等）都保持冻结。优化过程通过反向传播 (backpropagation)调整提示词嵌入$P_j$，以便将它们添加到输入序列前，从而引导冻结的LLM生成所需的输出（例如，一个正确的摘要）。

这种方法极大地减少了可训练参数的数量，即使对于数十亿参数的模型，通常也只减少到几千或几万个（远低于总参数的0.1%）。这使得训练的内存效率非常高。

### 初始化的影响

提示词 (prompt)嵌入 (embedding)$P_j$的初始化方式可以明显影响训练稳定性和最终性能。常用策略有：

1. **随机初始化：** 使用小的随机值初始化向量 (vector)，类似于其他网络权重 (weight)的初始化方式。这种方法简单，但可能需要仔细调整学习率，并可能导致更长的收敛时间。
2. **词汇初始化：** 使用模型词汇表 (vocabulary)中与目标任务相关的特定词汇的平均嵌入来初始化提示词嵌入。例如，对于摘要任务，你可以使用“Summarize”、“TLDR”、“Abstract”、“Condense”等词汇的嵌入进行初始化。这可以为优化过程提供一个更好的起点。

### P-Tuning：提升提示词 (prompt)的稳定性和能力

尽管基础提示词微调 (fine-tuning)效率很高，但其性能有时会落后于完全微调或LoRA等方法，特别是在GLUE或SuperGLUE等基准测试中发现的复杂自然语言理解（NLU）任务上。开发P-Tuning等变体就是为了解决这些局限性。

**P-Tuning (v1)：引入提示词编码器**

P-Tuning (v1) 观察到手动选择最优提示词的离散性很难实现，并且独立的提示词嵌入 (embedding)（如基础提示词微调中所示）可能缺乏表现力。它提出了两个主要方面：

1. **提示词编码器：** P-Tuning没有直接学习静态提示词嵌入$P_j$，而是使用一个小型神经网络 (neural network)（通常是BiLSTM后接MLP），称为“提示词编码器”。该编码器将虚拟提示词标记 (token)序列作为输入，并动态生成实际的连续提示词嵌入。这使得提示词嵌入之间可以存在依赖关系，从而可能增加它们的表现力。
2. **锚定标记：** 有时会在输入中散布任务特定的锚定标记，以进一步引导模型。

然而，P-Tuning v1仍然主要集中在输入嵌入层周围进行修改，有时会出现优化不稳定的情况。

**P-Tuning v2（深度提示词微调）：层级特定提示词**

P-Tuning v2（常被称为深度提示词微调）通过将可训练的提示词嵌入应用于Transformer的*每个层*，而不仅仅是输入层，从而大幅增强了该方法。

在这种设置中，对于每个层$l$，都会维护一组可训练的提示词嵌入$P_j^{(l)}$。这些嵌入会添加到进入该层的隐藏状态序列之前。这类似于Prefix Tuning，后者也在每个层注入可调参数 (parameter)。然而，P-Tuning v2通常只添加前缀*向量 (vector)*，而Prefix Tuning通常涉及专门为注意力机制 (attention mechanism)学习前缀键值对。

通过允许对模型内部计算在每个层进行直接影响，P-Tuning v2克服了浅层提示词微调的局限性，即初始提示词的影响可能在更深层消散。它在具有挑战性的NLU基准测试上表现出接近完全微调的性能，同时保持了高参数效率（尽管略低于基础提示词微调的效率，但仍远高于LoRA或完全微调）。它有效地为冻结的LLM提供了一种层级“引导机制”。

### 优点与缺点

**提示词 (prompt)微调 (fine-tuning)与P-Tuning：**

- **优点：**

  - **参数 (parameter)效率最高：** 在常见的PEFT方法中，它需要的微调参数最少，从而为适配任务带来最小的存储开销（只需存储少量提示词向量 (vector)）。
  - **内存占用极小：** 训练所需的GPU内存远少于完全微调甚至LoRA，因为梯度只为少量提示词参数计算。
  - **基础模型完整性：** 原始LLM权重 (weight)保持不变，简化了部署并确保了基础模型的能力得到保留。切换任务可以像加载不同提示词向量一样简单。
- **缺点：**

  - **性能敏感性：** 结果可能对提示词长度（$k$）、学习率和初始化等超参数 (hyperparameter)敏感。基础提示词微调在某些复杂任务上的表现可能不如更具侵入性的方法。
  - **有限的控制（基础提示词微调）：** 仅修改输入嵌入 (embedding)可能不足以大幅改变模型的核心行为，使其适应与预训练 (pre-training)目标非常不同的任务。P-Tuning v2在很大程度上缓解了这一点。
  - **可解释性挑战：** 软提示词是学习到的向量，缺乏文本提示词的直接人类可读含义。理解特定软提示词集为何有效并不简单。

提示词微调及其P-Tuning变体构成了一组有吸引力的PEFT技术，当计算资源高度受限或需要单个基础模型实例同时支持多项任务时，它们尤其有价值。特别是P-Tuning v2，在参数效率和性能之间取得了良好的平衡，在许多NLU任务上与更复杂的方法相媲美。

## 参考资料

- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243/) — Brian Lester, Rami Al-Rfou, and Noah Constant (2021)
  Journal: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 3045-3059; DOI: [10.18653/v1/2021.emnlp-main.243](https://doi.org/10.18653/v1/2021.emnlp-main.243)
  这篇基础论文介绍了Prompt Tuning，它使用可学习的连续提示向量来引导大型语言模型。
- [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Even with 0.1% Parameters](https://arxiv.org/abs/2110.07602) — Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, Jie Tang (2022)
  Journal: Proceedings of the 60th Annual Meeting of the Association of Computational Linguistics; Volume: abs/2110.07602; DOI: [10.48550/arXiv.2110.07602](https://doi.org/10.48550/arXiv.2110.07602)
  介绍了P-Tuning v2，它将提示嵌入应用于Transformer的每一层，显著提升了复杂NLU任务的性能。
