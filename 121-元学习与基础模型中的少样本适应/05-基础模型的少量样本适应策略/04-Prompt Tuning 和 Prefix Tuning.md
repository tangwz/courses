# Prompt Tuning 和 Prefix Tuning

来源：[原文](https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-5-few-shot-adaptation-foundation-models/prompt-prefix-tuning)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Prompt Tuning 和 Prefix Tuning 是旨在通过学习到的输入信号来调整基础模型行为的策略，它们使核心模型参数 (parameter)保持完全不变。这些技术提供了一种替代方法，与修改现有模型权重 (weight)（例如 LoRA 等低秩修改方法）不同，并且属于参数高效微调 (fine-tuning) (PEFT) 的范畴。这些方法不改变模型计算的函数 ($f_{	heta}(x)$)，而是学习一个特定任务的前缀或提示 ($p$) 来修改输入或内部状态，有效地计算 $f_{	heta}(p, x)$。这种方法提供了极致的参数效率，因为在调整过程中只优化提示或前缀的参数。

### Prompt Tuning: 通过学习到的输入嵌入 (embedding)引导

Prompt Tuning 引入一组可学习的连续向量 (vector)，通常称为“软提示”，直接到冻结的基础模型的输入嵌入序列中。想象将一小段特定任务的“指令”预置到前面，它们不是离散文本，而是通过梯度下降 (gradient descent)优化的连续嵌入向量。

**机制:**
设原始输入序列嵌入为 $X = [e_1, e_2, ..., e_n]$，其中 $e_i \in \mathbb{R}^d$ 且 $d$ 是嵌入维度。Prompt Tuning 会在前面添加一个包含 $k$ 个可学习提示嵌入的序列 $P = [p_1, p_2, ..., p_k]$，其中每个 $p_j \in \mathbb{R}^d$ 都是一个可训练参数 (parameter)。然后，馈入 Transformer 第一层的修改后的输入序列是 $[p_1, ..., p_k, e_1, ..., e_n]$。

> Prompt Tuning 的流程。可训练的软提示嵌入被预置到输入序列嵌入之前，然后由冻结的基础模型处理。

**参数效率:**
可训练参数的数量极其少：$k \times d$。对于一个典型的基础模型，如果 $d=4096$ 且提示长度 $k=20$，这只相当于大约 82,000 个参数，比基础模型中数十亿的参数少几个数量级，并且显著少于 LoRA 或 Adapters 在典型配置下的参数量。

**训练:**
软提示 $P$ 的参数是使用标准的梯度下降技术来优化的，基于特定任务的损失函数 (loss function)（例如，分类任务的交叉熵，生成任务的语言模型损失）。基础模型的参数 ($\theta$) 在整个过程中保持固定。这些提示嵌入的初始化会显著影响性能；常见策略包括从模型的词汇嵌入中采样或使用特定的初始化方案。

**优点:**

- **极致参数效率：** 对训练和部署多个已调整模型来说，存储和计算开销极小。
- **不修改模型：** 基础模型保持不变，简化了部署和推理 (inference)基础设施。不同任务只需加载不同的微小提示向量。
- **对 NLU 有效：** 已在 GLUE 和 SuperGLUE 等基准测试的自然语言理解任务中展现出竞争性性能，有时以显著更少的参数量达到与完全微调 (fine-tuning)相当的性能。

**局限:**

- **表达力：** 对于需要模型行为显著改变的复杂任务，仅修改输入嵌入可能不如改变模型内部计算（如 LoRA 或 Adapters）那么强大。
- **优化挑战：** 训练可能对学习率、提示长度 ($k$) 和初始化等超参数 (hyperparameter)敏感。更长的提示会增加参数，但并非总能带来更好的结果。
- **性能可变性：** 性能会根据基础模型规模和具体任务而有显著差异。

### Prefix Tuning: 注意力层的条件化

Prefix Tuning 将学习到的连续提示的理念更进一步，通过将可训练参数 (parameter)直接插入到 Transformer 层的激活状态中，特别是针对多头注意力 (multi-head attention)机制 (attention mechanism)。Prefix Tuning 不仅仅是预置到输入中，它还会将可学习的“前缀”向量 (vector)添加到每个层（或一部分层）中用于注意力计算的键 ($K$) 和值 ($V$) 中。

**机制:**
对于 Transformer 层，注意力机制根据查询 ($Q$)、键 ($K$) 和值 ($V$) 计算注意力分数。Prefix Tuning 引入一个可训练前缀矩阵 $P_{prefix} \in \mathbb{R}^{k \times d}$，其中 $k$ 是前缀长度， $d$ 是隐藏维度。这个前缀通常通过小型可训练的前馈网络（重参数化）来投影，以生成层特定的键和值前缀 $P_K$ 和 $P_V$，它们的形状都为 $\mathbb{R}^{k \times d_{attn}}$，其中 $d_{attn}$ 是每个头的键/值维度。

这些前缀随后与该层原始的键和值连接起来，然后再进行注意力计算：


$$
K_{新} = \text{连接}(P_K, K)
$$


$$
V_{新} = \text{连接}(P_V, V)
$$


查询 $Q$ 随后会关注这个增强的键和值集合。关键的是，原始模型参数，包括 $Q, K, V$ 的投影矩阵，保持冻结。只有初始前缀矩阵 $P_{prefix}$ 的参数以及可能的小型重参数化网络会进行训练。

> Prefix Tuning 在单个 Transformer 层内的流程。可训练的前缀参数经过处理后，被注入到注意力机制的键 (K) 和值 (V) 矩阵中，影响其行为而不改变冻结的模型权重 (weight)。

**参数效率:**
可训练参数的数量取决于前缀长度 $k$、应用前缀的层数 $L$ 以及隐藏维度 $d$（或 $d_{attn}$，取决于实现细节，可能包含小型重参数化网络）。通常，与基础模型相比，参数数量仍然极低，常常与 Prompt Tuning 相当或略多，但显著少于完全微调 (fine-tuning)或 Adapters/LoRA。

**表达力和优点:**

- **直接注意力控制：** 通过直接影响注意力层中的键和值状态，Prefix Tuning 可以在模型内部表示和序列生成过程中施加比 Prompt Tuning 更精细的控制。
- **强大的生成性能：** 与 Prompt Tuning 相比，通常在复杂序列生成任务（例如，摘要、数据到文本）上表现出更优越的性能。
- **参数效率：** 保持了 PEFT 的核心优势，可训练参数极少。

**局限:**

- **实现复杂性：** 由于需要修改注意力层内部结构，实现起来比 Prompt Tuning 略微复杂。
- **超参数 (hyperparameter)敏感性：** 性能可能对前缀长度的选择、应用前缀的层以及重参数化网络的结构（如果使用）敏感。

### Prompt Tuning 与 Prefix Tuning 对比

Prompt Tuning 和 Prefix Tuning 都通过学习连续向量 (vector)来提供极致的参数 (parameter)效率，同时保持基础模型冻结。

- **机制：** Prompt Tuning 将可学习向量添加到输入序列嵌入 (embedding)中。Prefix Tuning 将可学习向量添加到注意力层内部的键和值矩阵中。
- **控制：** Prefix Tuning 通过操纵注意力计算，可能对模型的生成过程提供更直接的控制。
- **性能：** Prefix Tuning 在复杂生成任务上通常优于 Prompt Tuning，而 Prompt Tuning 对 NLU 任务可能非常有效。
- **参数：** 两种方法使用的参数都远少于完全微调 (fine-tuning)，甚至少于其他 PEFT 方法（如 LoRA/Adapters）。Prefix Tuning 根据配置，参数可能比 Prompt Tuning 略多。

将这些方法与**硬提示**（手动编写的文本指令）和**上下文 (context)学习**（在输入中直接提供任务示例而无需梯度更新）区分开来很重要。软提示和前缀是为特定任务优化的学习到的连续表示，它们像是嵌入在模型连续向量空间中的高度专业化、经过梯度调整的指令。总而言之，Prompt Tuning 和 Prefix Tuning 为将大型基础模型调整到下游任务提供了强大、轻量级的机制，在训练和推理 (inference)过程中计算成本极低。它们代表了与传统微调的显著不同，在保持基础模型完整性的同时，通过学习到的条件信号实现了有效的少样本调整。它们的适用性取决于具体任务、性能要求和可接受的实现复杂性。

## 参考资料

- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243/) — Brian Lester, Rami Al-Rfou, Noah Constant (2021)
  Journal: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 3045–3059; DOI: [10.18653/v1/2021.emnlp-main.243](https://doi.org/10.18653/v1/2021.emnlp-main.243)
  介绍了提示调优（Prompt Tuning），展示了其在以极少的可训练参数适应大型语言模型方面的有效性。
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/) — Xiang Lisa Li, Percy Liang (2021)
  Journal: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Volume: 1; Pages: 4582–4597; DOI: [10.18653/v1/2021.acl-long.353](https://doi.org/10.18653/v1/2021.acl-long.353)
  介绍了前缀调优（Prefix Tuning），这是一种通过向注意力层添加可训练前缀来调节语言模型的方法，在生成任务中尤其有效。
- [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://ieeexplore.ieee.org/document/10103770) — Vladimir Lialin, Vladislav Korablinov, Anna Kurochkin (2023)
  Journal: IEEE Access; Publisher: IEEE; Volume: 11; Pages: 37426-37452; DOI: [10.1109/ACCESS.2023.3270428](https://doi.org/10.1109/ACCESS.2023.3270428)
  一篇关于参数高效微调（PEFT）技术的综述，涵盖了提示调优和前缀调优等方法及其应用。

---

[上一节](03-%E4%BD%8E%E7%A7%A9%E9%80%82%E9%85%8D%20%28LoRA%29.md) · [下一节](05-%E6%AF%94%E8%BE%83%20PEFT%20%E4%B8%8E%20%E5%85%83%E5%AD%A6%E4%B9%A0%20%E6%96%B9%E6%B3%95.md)
