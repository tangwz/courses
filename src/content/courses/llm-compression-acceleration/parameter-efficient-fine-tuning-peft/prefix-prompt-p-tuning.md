---
course: "llm-compression-acceleration"
chapter: "parameter-efficient-fine-tuning-peft"
lesson: "prefix-prompt-p-tuning"
sourceId: 4386
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-5-parameter-efficient-fine-tuning-peft/prefix-prompt-p-tuning"
title: "前缀微调、提示微调与P-Tuning"
description: "实现并比较基于提示的参数高效微调（PEFT）方法，例如前缀微调、提示微调和P-Tuning v2。"
order: 3
plots: ["plots/4386-0.json"]
sourceHash: "c014162d391b905e4861e7721c1aceea58aa13557e6ee73a38f1bbc7f39e2294"
sourceCorrections: []
---

在参数 (parameter)高效微调 (fine-tuning)（PEFT）技术中，有一类方法着重于使用学习到的连续*提示*或*前缀*来调整输入或中间激活。这些方法是修改现有模型权重 (weight)或在Transformer块内添加适配器层的替代方案。它们将原始大型语言模型（LLM）完全冻结，只训练少量与提示相关的参数，这些参数引导模型针对特定任务的行为。这种方式因其极小的参数开销和简洁性而特别受欢迎。我们将介绍此类别的三种重要技术：前缀微调、提示微调和P-Tuning。

### 前缀微调 (fine-tuning)

前缀微调将任务特定的*前缀*（即连续向量 (vector)序列）引入预训练 (pre-training)Transformer模型的注意力机制 (attention mechanism)中。与用户提供的离散文本提示不同，这些前缀是在微调期间学习的自由参数 (parameter)。主要思路是将这些学习到的前缀向量添加到每个Transformer层内多头自注意力 (self-attention)计算中使用的键($K$)和值($V$)前面。

设头部原始的注意力计算为：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


其中 $Q, K, V$ 是查询、键和值矩阵，而 $d_k$ 是键的维度。

前缀微调通过将学习到的前缀向量 $P_k$ 和 $P_v$（长度为 $L_p$）添加到从输入序列（长度为 $L_{seq}$）派生的原始键和值前面来修改此计算：


$$
K' = [P_k; K] \quad (\text{维度 } (L_p + L_{seq}) \times d_k)
$$


$$
V' = [P_v; V] \quad (\text{维度 } (L_p + L_{seq}) \times d_k)
$$


然后注意力计算变为：


$$
\text{注意力}(Q, K', V') = \text{softmax}\left(\frac{Q(K')^T}{\sqrt{d_k}}\right)V'
$$


重要的一点是，原始LLM参数保持冻结。只有每个层的前缀参数$P_k$和$P_v$被训练。这使得前缀能够像“方向盘”一样，引导注意力机制的侧重，并影响模型针对目标任务的输出分布，而无需改变基础模型的权重 (weight)。

> 前缀微调概览。学习到的前缀（$P_k, P_v$）修改每个冻结注意力层内的键和值矩阵。

为了提升训练稳定性和性能，前缀参数（$P_k, P_v$）通常不直接训练。相反，它们通过一个较小的重参数化网络（通常是一个小型多层感知机MLP）生成，该网络作用于一组更小的核心前缀参数。这有助于管理维度并提供更平滑的优化曲面。

前缀微调仅为每层的前缀（$P_k, P_v$）以及小型重参数化网络添加参数。这通常只占原始LLM参数的极小一部分（例如，0.1% - 1%）。它表现出良好的性能，特别是在文本摘要和表格到文本生成等生成任务上，这些任务中控制模型的生成风格和内容很重要。

### 提示微调 (fine-tuning)

提示微调是前缀微调的一种简化形式。提示微调不向每个层的注意力机制 (attention mechanism)中添加前缀，而是将一系列学习到的连续向量 (vector)（通常称为*软提示*或*提示嵌入 (embedding)*）直接添加到输入词元 (token)嵌入的前面。

想象输入词元嵌入序列 $E = [e_1, e_2, ..., e_n]$。提示微调将 $k$ 个可训练提示嵌入 $P = [p_1, p_2, ..., p_k]$ 添加到此序列前面：


$$
E' = [p_1, ..., p_k; e_1, ..., e_n]
$$


这个增强序列 $E'$ 随后被送入冻结LLM的第一层。模型的其余部分照常处理此序列，所有原始权重 (weight)保持冻结。

> 提示微调概览。学习到的提示嵌入（$P$）在进入冻结LLM之前被添加到输入序列嵌入的前面。

提示微调中可训练参数 (parameter)的数量非常少，通常只有几千个，具体取决于所选的提示长度（$k$）和嵌入维度。它只需为每个任务存储这$k$个向量。初始化很重要；使用预训练 (pre-training)模型中相关词汇的嵌入来初始化这些提示嵌入通常比随机初始化获得更好的结果。

尽管参数效率很高，但提示微调的影响主要限于初始表示。与前缀微调或完全微调等方法相比，它在较难的序列生成任务或需要对模型跨层内部表示有更多控制的基准测试（例如SuperGLUE）上可能表现不佳。然而，对于许多自然语言理解（NLU）任务，它可以用极少的可训练参数实现有竞争力的性能。

### P-Tuning和P-Tuning v2

P-Tuning (v1) 基于可学习的连续提示的思路构建，但引入了一个*提示编码器*。P-Tuning不学习像提示微调 (fine-tuning)那样的静态提示嵌入 (embedding)，而是使用一个小型可训练神经网络 (neural network)（例如MLP或LSTM）作为提示编码器来动态生成连续提示向量 (vector)。这些生成的向量随后被插入到输入序列中，类似于提示微调。使用编码器可以对连续提示词 (prompt)元之间的依赖关系进行建模，可能生成表达力更强的提示。P-Tuning v1还引入了*锚点词元 (token)*以表明任务特定预测的位置。虽然显示出潜力，但它有时在复杂NLU任务上存在不稳定性和局限性。

P-Tuning v2 的开发是为了解决这些局限性并弥合与完全微调的性能差距，特别是在具有挑战性的NLU基准测试上。它结合了提示/前缀微调的思路：

1. **深度提示微调：** 与前缀微调类似，P-Tuning v2 不仅在输入层，还在Transformer的*更深层*应用可训练的提示向量。这使得学习到的提示可以在多个阶段影响模型的处理。
2. **去除重参数 (parameter)化：** 与前缀微调不同，P-Tuning v2 通常取消了用于更深层提示向量的重参数化网络（MLP），从而简化了架构。
3. **任务特定适应：** 它被证明在各种模型规模和任务（包括序列标注）上都特别有效，展现出比早期基于提示的方法更广泛的适用性。

实质上，P-Tuning v2 作用类似于应用于键和值的多层提示微调（效果上与前缀微调的逐层干预类似），但在参数生成方面可能实现更简洁的实现。它通常比基本提示微调需要更多参数，但比前缀微调少（因为缺少重参数化网络），并且远少于完全微调。与提示微调或P-Tuning v1相比，它在复杂NLU基准测试上通常能实现非常接近完全微调的性能。

### 比较与考量

这些基于提示的PEFT方法提供了不同的权衡：

- **参数 (parameter)效率：** 提示微调 (fine-tuning)的参数效率最高，其次是P-Tuning v2，然后是前缀微调（主要是由于其重参数化网络）。所有这些方法都比完全微调效率高得多。
- **性能：**
  - 对于简单的NLU任务或以极致参数效率为目标时，**提示微调**可能就足够了。
  - 对于复杂的NLU基准测试（如SuperGLUE）和更好的稳定性，**P-Tuning v2**通常提供更接近完全微调的性能。
  - 对于需要对输出进行精细控制的生成任务，**前缀微调**已展现出强大的能力。
- **实现复杂度：** 提示微调最简单。前缀微调涉及重参数化，而P-Tuning v2则需要管理跨多个层的提示。
- **训练稳定性：** 初始化策略（例如，使用词汇嵌入 (embedding)）对于提示微调很重要。前缀微调中的重参数化有助于稳定性。P-Tuning v2旨在比v1提高稳定性。



![基于提示的PEFT方法比较](plots/4386-0.json)



> 参数效率（训练的原始模型参数百分比）与不同基于提示的PEFT方法相对于完全微调的相对任务性能之间的权衡。实际性能会根据模型大小、任务和实现细节有很大差异。

选择前缀微调、提示微调和P-Tuning取决于具体的任务需求、可接受的性能权衡以及可用的计算资源。它们是完全微调的强大替代方案，通过学习小型、任务特定的连续提示，有效地引导冻结LLM的行为，从而实现大型模型的有效适应。像Hugging Face的PEFT这样的库提供了应用这些技术的便捷实现。

## 参考资料

- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/) — Xiang Lisa Li, Percy Liang (2021)
  Journal: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Pages: 4582-4597; DOI: [10.18653/v1/2021.acl-long.353](https://doi.org/10.18653/v1/2021.acl-long.353)
  介绍了Prefix Tuning的创始论文，这是一种学习注意力键和值连续前缀的参数高效微调方法。
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243/) — Brian Lester, Rami Al-Rfou, Noah Constant (2021)
  Journal: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 3045–3059; DOI: [10.18653/v1/2021.emnlp-main.243](https://doi.org/10.18653/v1/2021.emnlp-main.243)
  本文介绍了Prompt Tuning，一种通过在冻结LLM输入前添加可学习软提示嵌入来实现高效适应的方法。
- [GPT Understands, Too](https://arxiv.org/abs/2103.10385) — Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, Jie Tang (2021)
  Journal: arXiv; DOI: [10.48550/arXiv.2103.10385](https://doi.org/10.48550/arXiv.2103.10385)
  介绍了P-Tuning（v1），它使用可训练的提示编码器生成文本的连续提示向量。这是P-Tuning v2的前身。
