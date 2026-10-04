---
course: "lora-peft-efficient-llm-training"
chapter: "advanced-lora-variants"
lesson: "combining-lora-other-peft"
sourceId: 4152
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-4-advanced-lora-variants/combining-lora-other-peft"
title: "将LoRA与其他PEFT方法结合"
description: "探讨LoRA与适配器或提示调优等方法结合的策略，以期获得潜在的性能提升。"
order: 6
plots: []
sourceHash: "d38ef4d9ad064200e7feb8ad68ab3cc5d94f635972d9e6d848f8773452cd42e5"
sourceCorrections: []
---

虽然低秩适应（LoRA）提供了一种强大且高效的微调 (fine-tuning)大型语言模型（LLMs）的方式，但它主要通过低秩更新来修改现有权重 (weight)矩阵。其他参数 (parameter)高效微调（PEFT）方法，例如适配器调优或前缀调优，以不同的方式干预模型的架构或激活路径。这引发了一个有趣的问题：将LoRA与其他PEFT技术结合使用，能否产生更好的结果，或与单独使用某一种方法相比，提供不同的权衡？

结合这些方法的动机源于一个假设：不同的PEFT技术可能捕获任务适应的互补方面。LoRA侧重于调整现有层内的内在表示，而其他方法可能擅长注入新的计算路径（适配器）或引导模型的注意力机制 (attention mechanism)（前缀/提示调优）。

### LoRA与适配器模块结合

适配器调优包括在预训练 (pre-training)Transformer模型的层内插入小型、可训练的神经网络 (neural network)模块（适配器），通常是在注意力或前馈子层之后。原始模型权重 (weight)保持不变，只训练适配器参数 (parameter)。

一种可能的结合方式是，将LoRA应用于标准权重矩阵（例如，注意力中的查询、键、值、输出投影以及前馈层），同时*插入并训练*适配器模块。

**架构：**

设想一个标准Transformer块。

1. 注意力机制 (attention mechanism)的权重矩阵（$W_q, W_k, W_v, W_o$）可以使用LoRA修改：$W_q \rightarrow W_q + \Delta W_q$，其中 $\Delta W_q = \alpha \frac{B_q A_q}{r}$。
2. 适配器模块可以插入到注意力块（及其层归一化 (normalization)）*之后*。
3. 同样，前馈网络的权重矩阵（$W_{ffn1}, W_{ffn2}$）也可以接收LoRA更新。
4. 另一个适配器模块可以插入到前馈块（及其层归一化）*之后*。

> 图示说明了LoRA（修改现有MHA和FFN层）和适配器模块在Transformer块内的集成。

**潜在优势：**

- **互补适应：** LoRA可以为现有权重提供广泛的适应，而适配器则添加局部、可能非线性的计算，这些计算专门针对任务定制。
- **灵活性：** 允许调整LoRA的秩 $r$ 和缩放因子 $\alpha$，以及适配器的架构和参数，提供更多自由度。

**挑战：**

- **复杂性增加：** 管理两组可调参数（LoRA矩阵 $A, B$ 和适配器权重）增加了实现和超参数 (hyperparameter)调整的复杂性。
- **参数预算：** 尽管与完全微调 (fine-tuning)相比仍然参数高效，但与仅使用LoRA或仅使用适配器相比，可训练参数的总数会增加。
- **优化：** 寻找同时适用于LoRA更新和适配器训练的最佳学习率和调度可能需要仔细的实验。训练期间两种更新类型之间可能存在干扰。

### LoRA与前缀调优或提示调优结合

前缀调优和提示调优引入可训练参数 (parameter)，这些参数间接影响模型，通常通过在注意力层中向键和值状态添加连续向量 (vector)（前缀），或者通过在输入序列前附加可调嵌入 (embedding)。

将LoRA与这些方法结合意味着同时训练修改内部权重 (weight)的LoRA矩阵（$A, B$）和前缀/提示向量。

**联动：**

1. **LoRA：** 修改内部权重矩阵 $W_q, W_k, W_v, W_o, W_{ffn1}, W_{ffn2}$，如前所述。
2. **前缀调优：** 在计算注意力分数*之前*，向注意力机制 (attention mechanism)内计算出的键和值添加可训练的前缀向量 $P_k, P_v$。经LoRA修改的 $W_k, W_v$ 将处理原始输入以生成键/值，然后这些键/值与学到的前缀拼接。
3. **提示调优：** 在它们进入第一个Transformer层*之前*，向输入序列嵌入添加可训练的提示嵌入 $E_{prompt}$。这些修改后的嵌入随后由LoRA适应的层处理。

**潜在优势：**

- **正交控制：** LoRA调整模型内部处理信息的方式，而前缀/提示调优调整模型操作的*有效输入*或上下文 (context)。这种分离可能允许对适应进行更精细的控制。
- **定向干预：** 可以假设使用前缀/提示调优来引导模型的关注点或高级行为，而LoRA则微调 (fine-tuning)较低级别的表示。

**挑战：**

- **训练动态：** 同时优化低秩矩阵因子（LoRA）和连续向量嵌入（前缀/提示）可能具有挑战性。它们可能需要不同的学习率或优化策略。
- **可解释性：** 精确理解这两种方法如何共同作用以产生最终输出变得更加困难。
- **回报递减：** 存在一种可能性，即一种方法的优势可能盖过另一种，或者将它们结合并不能带来显著优于经过良好调优的单一PEFT方法的性能，同时还增加了复杂性。

### LoRA与量化 (quantization)结合（QLoRA）

虽然QLoRA常被视为一种独特技术（并且之前已详细介绍），但从根本上，它可以看作是一种结合方式：

1. **量化：** 基础模型的权重 (weight)被大量量化（例如，到4位NormalFloat, $NF4$），以大幅减少内存占用。这是一种在微调 (fine-tuning)*之前*应用的模型压缩技术。
2. **LoRA：** 标准LoRA适配器（通常以BFloat16等更高精度进行训练）被添加到量化后的基础模型中。只训练LoRA参数 (parameter)。

这种特殊的结合直接解决了微调大型模型的内存限制，使其非常实用。QLoRA的成功表明，即使底层基础权重的精度已显著降低，LoRA也能有效适应模型。这或许是涉及LoRA的最广泛采用和验证的结合方式。

### 结合方法的一般考虑

在考虑将LoRA与其他PEFT技术结合时，请记住以下几点：

- **任务依赖性：** 任何结合方式的有效性都可能取决于具体任务。某些任务可能通过增加非线性适配器能力获得更多好处，而其他任务可能对前缀调优与LoRA结合带来的上下文 (context)引导效果反应更好。
- **超参数 (parameter) (hyperparameter)调优：** 超参数的搜索空间显著扩大。需要*协调地*调整LoRA参数（$r$, $\alpha$, 目标模块）、适配器参数（维度、激活函数 (activation function)）或前缀/提示参数（长度、初始化）。这需要仔细的方法，可能使用如序列优化或在结合超参数空间中搜索等技术。
- **计算开销：** 尽管参数数量保持较低，但训练期间的*计算图*可能变得更复杂，可能影响训练速度，具体取决于结合方式和实现。
- **实证验证：** 理论上的好处需要通过实证来验证。必须将结合方法与经过良好调优的基准PEFT方法（包括仅使用可能更高秩的LoRA）在相关评估指标上进行比较。

将LoRA等PEFT方法与适配器或前缀调优结合是一个活跃的研究领域。这些结合方式虽然可能提供增强的灵活性和性能，但在实现、调优和分析方面增加了复杂性。QLoRA作为一种非常成功的结合方式脱颖而出，主要侧重于内存效率，证明了在修改（量化 (quantization)）后的基础模型上训练LoRA适配器的可行性。与许多先进技术一样，需要仔细的实验和评估来确定特定任务和计算预算的最佳方法。

## 参考资料

- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) — Xiang Lisa Li, Percy Liang (2021)
  Journal: Annual Meeting of the Association for Computational Linguistics (ACL); Pages: 1706-1718; DOI: [10.48550/arXiv.2101.00190](https://doi.org/10.48550/arXiv.2101.00190)
  介绍了Prefix Tuning，通过优化连续前缀向量来适应模型，为与LoRA组合提供了一种方法。
- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  详细介绍了QLoRA，这是LoRA与4位量化的一种成功组合，展示了结合PEFT技术的实用性。
- [UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning](https://arxiv.org/abs/2110.07577) — Yuning Mao, Lambert Mathias, Rui Hou, Amjad Almahairi, Hao Ma, Jiawei Han, Wen-tau Yih, Madian Khabsa (2022)
  Journal: Annual Meeting of the Association for Computational Linguistics (ACL); Pages: 5500-5517; DOI: [10.48550/arXiv.2110.07577](https://doi.org/10.48550/arXiv.2110.07577)
  提出了一个统一框架，用于结合各种PEFT方法，包括LoRA、适配器和Prefix-Tuning，支持多PEFT策略。
