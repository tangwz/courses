---
course: "lora-peft-efficient-llm-training"
chapter: "peft-methodologies-survey"
lesson: "prefix-tuning-theory"
sourceId: 4135
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-3-peft-methodologies-survey/prefix-tuning-theory"
title: "前缀微调：通过连续前缀进行调节"
description: "了解前缀微调如何将可调节的连续向量添加到输入序列嵌入中。"
order: 3
plots: []
sourceHash: "2d96dc3377062d44ff9dbb256ca8250ee311b22e4a9f339d6497dab02da9299a"
sourceCorrections: []
---

不同于在模型现有层中嵌入 (embedding)可训练模块的方法，前缀微调 (fine-tuning)提供了一种不同的参数 (parameter)高效调整方式。它不修改内部权重 (weight)或添加适配器模块，而是通过在输入或隐藏状态前附加一系列连续的、特定于任务的向量 (vector)（即*前缀*），来调节冻结的预训练 (pre-training)模型的行为。

核心思路是学习一小组参数，这些参数能够有效地引导大型固定模型的激活，使其朝向所需的下游任务行为。可以设想，在模型处理实际输入之前，我们给它一个特殊的、学习到的“指令序列”。这个指令序列并非由离散的词元 (token)组成，而是由通过梯度下降 (gradient descent)直接优化的连续向量构成。

### 前缀微调 (fine-tuning)的工作方式

前缀微调在Transformer架构中通常涉及将可学习的前缀向量 (vector)添加到自注意力 (self-attention)机制 (attention mechanism)中*每一层*使用的键 ($K$) 和值 ($V$)。原始模型参数 (parameter)保持不变。

让我们来看一下标准的自注意力计算：


$$
Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


这里，$Q$、$K$ 和 $V$ 通常是输入隐藏状态 $h$ 的线性投影：$Q = hW_Q$，$K = hW_K$，$V = hW_V$。

前缀微调通过将前缀向量 $P_K$ 和 $P_V$ 连接到投影的键和值上来修改此过程：


$$
K' = [P_K; K] = [P_K; hW_K]
$$


$$
V' = [P_V; V] = [P_V; hW_V]
$$


查询 $Q$ 保持不变（$Q = hW_Q$）。注意力计算变为：


$$
Attention(Q, K', V') = \text{softmax}\left(\frac{Q(K')^T}{\sqrt{d_k}}\right)V'
$$


前缀向量 $P_K$ 和 $P_V$ 各自包含 $L_p$ 个向量，而 $L_p$ 则是选择的前缀长度（一个超参数 (hyperparameter)）。每个向量的维度与原始键/值向量相同（通常是 $d_k$ 或 $d_{model}$）。这些前缀参数共同形成一个矩阵 $P$，是微调过程中*唯一*更新的参数。

> Transformer层中前缀微调的图示。原始权重 (weight) (WQ, WK, WV) 被冻结。可训练的前缀参数 (P) 被映射到 $P_K$ 和 $P_V$，并在注意力计算之前附加到原始的 K 和 V 矩阵前。

### 参数 (parameter)数量与初始化

前缀微调 (fine-tuning)中的可训练参数数量取决于前缀长度 $L_p$、模型的隐藏维度 $d_{model}$（为简化起见，假设 $d_k = d_v = d_{model}$，尽管实际上前缀可能通过较小的MLP进行映射），以及层数 $N$：


$$
\text{可训练参数} \approx L_p \times d_{model} \times N \times 2
$$


（因子2是因为键和值有独立的前缀）。通常，会使用一个小型多层感知器（MLP）来将一个更小的初始前缀矩阵投影到 $P_K$ 和 $P_V$ 所需的完整维度，从而进一步减少参数。

与LoRA或适配器微调相比，前缀微调可以非常参数高效，尤其是当 $L_p$ 较小（例如10-100）时。前缀参数 $P$ 通常是随机初始化的。

### 训练与重参数 (parameter)化

在训练过程中，梯度仅相对于前缀参数 $P$ 计算，而大型预训练 (pre-training)模型保持冻结。使用AdamW等标准优化器。

经常采用的一个技术细节是*重参数化*。不是直接优化每层的前缀参数 $P \in \mathbb{R}^{L_p \times d_{model}}$，而是学习一个更小的矩阵 $P' \in \mathbb{R}^{L_p \times d_{emb}}$，同时学习两个投影矩阵 $W_{proj, K}, W_{proj, V} \in \mathbb{R}^{d_{emb} \times d_{model}}$。这样，$P_K = P'W_{proj, K}$ 和 $P_V = P'W_{proj, V}$。如果 $d_{emb} < d_{model}$，这会减少可训练参数的数量。

### 优点与考量

- **参数 (parameter)效率：** 通常比完全微调 (fine-tuning)甚至LoRA等方法（取决于秩 $r$ 与前缀长度 $L_p$）所需调整的参数少得多。
- **不修改基础模型：** 原始的LLM权重 (weight)未被触动，这简化了部署，因为同一个基础模型可以通过更换前缀来服务多个任务。
- **在生成方面的有效性：** 已显示出良好的性能，特别是在自然语言生成任务中，其中调节模型的输出格式或风格很重要。

然而，请考虑以下几点：

- **性能变动性：** 在复杂的NLU任务上，可能不总是能达到完全微调或配置良好的LoRA的性能上限。
- **优化挑战：** 调整连续向量 (vector)有时可能不如调整离散提示或权重矩阵那样稳定或直观。
- **超参数 (hyperparameter)敏感性：** 前缀长度 $L_p$ 是一个重要的超参数，需要仔细调整。

### 前缀微调 (fine-tuning)对比提示微调

前缀微调与**提示微调**密切相关。主要区别在于调节发生的位置。提示微调通常只在输入层序列前附加可学习的嵌入 (embedding)，使其参数 (parameter)效率更高。前缀微调则通过将学习到的向量 (vector)注入*每一层*的注意力机制 (attention mechanism)，可能提供更大的表达能力来影响模型在整个生成过程中的内部表示，尽管代价是参数量略多于提示微调。

前缀微调提供了一种优雅的方式来调整LLMs，通过将计算工作集中在学习一个小的、特定于任务的“控制序列”上，而不是改变模型的核心知识。它在PEFT工具包中是一种有价值的替代方案，特别是在严格的参数效率和非侵入式模型调整是主要目标时。

## 参考资料

- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/) — Xiang Lisa Li, Percy Liang (2021)
  Journal: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Pages: 4582–4597; DOI: [10.18653/v1/2021.acl-long.353](https://doi.org/10.18653/v1/2021.acl-long.353)
  介绍前缀微调的原始论文，这是一种参数高效的方法，通过在注意力层的输入前添加任务特定的连续向量来调节冻结的语言模型。
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243/) — Brian Lester, Rami Al-Rfou, Noah Constant (2021)
  Journal: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 3045–3059; DOI: [10.18653/v1/2021.emnlp-main.243](https://doi.org/10.18653/v1/2021.emnlp-main.243)
  介绍提示微调，一种更参数高效的方法，通过仅在输入序列前添加可学习嵌入来调节大型语言模型，与前缀微调形成对比。
