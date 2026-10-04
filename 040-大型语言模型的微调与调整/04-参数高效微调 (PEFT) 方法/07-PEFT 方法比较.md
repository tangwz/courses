# PEFT 方法比较

来源：[原文](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning/comparison-peft-techniques)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为大型语言模型选择合适的参数 (parameter)高效微调 (fine-tuning) (PEFT) 方法，在特定资源限制下非常重要。尽管所有 PEFT 技术都旨在与完全微调相比降低计算负担，但它们在机制、性能特点和资源需求方面存在很大差异。主要 PEFT 方法，包括 LoRA、QLoRA、适配器模块、提示调优和前缀调优，将被比较以突出它们的差异。

了解这些差异将帮助您为自己的具体任务、模型大小和可用硬件选择最适合的方法。我们将根据以下几个方面进行评估：

- **可训练参数数量：** 引入并训练了多少新参数？
- **训练资源使用：** 主要指微调过程中的 GPU 内存消耗。
- **推理 (inference)性能：** 调优后对延迟和吞吐量 (throughput)的影响。
- **存储需求：** 表示微调适配结果所需的存储空间大小。
- **任务性能：** 与完全微调相比，在下游任务上的总体效果。
- **实现复杂性：** 使用常用库进行设置和使用的便捷性。

### 低秩适配 (LoRA)

LoRA 通过向预训练 (pre-training)模型的特定层（通常是注意力机制 (attention mechanism)的权重 (weight)矩阵）注入可训练的低秩矩阵 ($W_A$ 和 $W_B$) 来运行。原始权重 ($W_0$) 保持不变。更新表示为 $\Delta W = W_A W_B$，其中分解的秩 $r$（$W_A \in \mathbb{R}^{d \times r}$，$W_B \in \mathbb{R}^{r \times k}$）远小于原始维度 ($d, k$)。

- **参数 (parameter)：** 仅调整低秩矩阵 $W_A$ 和 $W_B$。参数数量很少，通常小于模型总参数的 1%，并且取决于选择的秩 $r$ 和适配层的数量。
- **训练：** 相比完全微调 (fine-tuning)，需要的 GPU 内存大大减少，因为梯度仅针对小型适配器矩阵计算。冻结的基础模型仍需加载，这会占用一定的内存。
- **推理 (inference)：** LoRA 适配器在部署前通常可以合并回原始权重 ($W = W_0 + W_A W_B$) 中。与基础模型相比，这会导致**没有额外推理延迟**，这是一个主要优点。如果不合并，每适配一个层会增加两次矩阵乘法。
- **存储：** 每个任务只需保存小型 $W_A$ 和 $W_B$ 矩阵，这样可以高效存储并在不同的微调适配之间切换。
- **性能：** 在许多任务上通常能达到与完全微调非常接近的性能，尤其是在秩 $r$ 选择得当时。
- **实现：** 得到 Hugging Face 的 PEFT 等库的广泛支持，使其实现起来相对简单。

### 量化 (quantization)低秩适配 (QLoRA)

QLoRA 直接基于 LoRA，并增加了重要的优化：即基础模型的量化。训练时，大型、冻结的基础模型权重 (weight)以量化格式（例如 4 位 NormalFloat）加载，大幅减少其内存占用。只有 LoRA 适配器进行训练，通常使用更高的精度，如 16 位 BrainFloat (BFloat16)。双重量化和分页优化器等技术可以进一步减少内存使用。

- **参数 (parameter)：** *可训练*参数的数量与 LoRA 相同（仅 $W_A, W_B$）。
- **训练：** 在这些方法中，QLoRA 提供的训练内存需求减少最显著。它使得在单个高端消费级 GPU 上微调 (fine-tuning)超大型模型（例如 650 亿参数以上）成为可能，这在标准 LoRA 或完全微调中是不可能的。
- **推理 (inference)：** 需要谨慎处理。模型必须进行反量化（可能会损失部分内存优势），或者必须使用支持特定量化格式的核进行推理（例如 4 位推理），这可能会根据硬件支持情况影响延迟。合并适配器是可行的，但会得到一个带有合并适配器的量化基础模型。
- **存储：** 与 LoRA 类似，仅需存储小型适配器权重。基础模型保持量化状态。
- **性能：** 尽管基础模型经过量化，但它旨在与 LoRA 和完全微调的性能相媲美，研究表明它通常表现非常好。
- **实现：** 由于需要特定的量化库（如 `bitsandbytes`）以及管理量化基础模型与高精度适配器之间的配合，它比 LoRA 更复杂。

### 适配器模块

适配器涉及在现有 Transformer 架构中插入小型、可训练的“瓶颈”层。这些模块通常由一个下投射前馈层、一个非线性层和一个上投射层组成。它们通常添加到每个 Transformer 块中的多头注意力 (multi-head attention)子层和前馈子层之后。原始模型参数 (parameter)保持不变。

- **参数：** 仅训练新插入适配器层内部的参数。总数量通常很少，与 LoRA 相似，取决于瓶颈维度和插入的适配器数量。
- **训练：** 内存需求远低于完全微调 (fine-tuning)，与 LoRA 的内存需求相似（不包括 QLoRA 基础模型量化 (quantization)带来的优势）。
- **推理 (inference)：** 适配器层在前向传播过程中增加了额外的计算。与 LoRA 不同，这些模块无法轻易合并到原始模型权重 (weight)中。这会导致与基础模型或已合并的 LoRA 相比，推理延迟持续增加。
- **存储：** 每个任务只需存储适配器模块的权重。
- **性能：** 可以达到很好的性能，尽管有时略低于 LoRA 或完全微调。性能取决于瓶颈维度和放置策略等因素。
- **实现：** 简单且受到库的支持。

### 提示调优和前缀调优

这些方法采取不同的方法：它们将整个预训练 (pre-training) LLM 完全保持不变。它们不修改权重 (weight)，而是学习连续的向量 (vector)嵌入 (embedding)，作为任务特定的指令或上下文 (context)。

- **提示调优：** 学习一系列连续嵌入（一个“软提示”），附加到输入序列嵌入之前。
- **前缀调优：** 学习连续嵌入，附加到每个注意力层中的键和值向量之前。
- **参数 (parameter)：** 仅调整提示或前缀嵌入。这导致可训练参数数量极少（通常只有几千个），*独立于*基础模型的大小。
- **训练：** 在内存和计算方面效率很高，因为梯度仅针对小型嵌入向量计算。
- **推理 (inference)：** 开销极小。提示调优将软提示的长度添加到输入序列长度中。前缀调优为注意力层中的前缀向量增加了计算。由于基础模型未受影响，不适用合并。
- **存储：** 仅需存储微小的学习到的提示/前缀向量，使其成为存储效率最高的方法。
- **性能：** 效果因任务而异。尤其在生成任务上表现良好，但在需要更复杂推理或修改模型内部表示的任务上，可能不如 LoRA 或适配器等方法。对初始化和超参数 (hyperparameter)敏感。
- **实现：** 受到库的支持，但与权重空间修改方法不同。

### 比较概览

下表对不同方法的可训练参数 (parameter)相对规模和典型训练内存需求进行了定性比较。



[交互图表：PEFT 方法：参数与内存相对比较](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning/comparison-peft-techniques#plot-1tg8yf5)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "name": "可训练参数（占完全微调的百分比）",
      "x": [
        "Full FT",
        "LoRA",
        "QLoRA",
        "Adapters",
        "Prompt Tuning"
      ],
      "y": [
        100,
        0.1,
        0.1,
        0.2,
        0.01
      ],
      "marker": {
        "color": "#4263eb"
      }
    },
    {
      "type": "bar",
      "name": "训练内存（占完全微调的百分比）",
      "x": [
        "Full FT",
        "LoRA",
        "QLoRA",
        "Adapters",
        "Prompt Tuning"
      ],
      "y": [
        100,
        30,
        15,
        35,
        20
      ],
      "marker": {
        "color": "#12b886"
      },
      "yaxis": "y2"
    }
  ],
  "layout": {
    "title": {
      "text": "PEFT 方法：参数与内存相对比较"
    },
    "barmode": "group",
    "xaxis": {
      "title": {
        "text": "微调方法"
      }
    },
    "yaxis": {
      "title": {
        "text": "可训练参数 (%)",
        "color": "#4263eb"
      },
      "range": [
        0,
        105
      ]
    },
    "yaxis2": {
      "title": {
        "text": "约训练内存 (%)",
        "color": "#12b886"
      },
      "overlaying": "y",
      "side": "right",
      "range": [
        0,
        105
      ],
      "showgrid": false
    },
    "legend": {
      "x": 0.01,
      "y": 0.99
    },
    "margin": {
      "l": 60,
      "r": 60,
      "t": 40,
      "b": 50
    }
  }
}
```

</details>



> 本图展示了可训练参数和训练过程中典型 GPU 内存使用量的比较，（以完全微调 (fine-tuning)为 100%）。QLoRA 因基础模型量化 (quantization)而显示出最低的内存使用量。提示调优的可训练参数最少。实际值很大程度上取决于模型大小、配置和硬件。

**权衡：**

- **性能与效率：** 通常，调整更多参数（更接近完全微调）或更直接修改权重 (weight)（LoRA、适配器）的方法，在复杂任务上倾向于达到更高的峰值性能，与提示/前缀调优相比。然而，它们在训练期间会带来更高的计算成本（QLoRA 的内存优势除外）。
- **训练内存：** QLoRA 在最小化训练内存方面明显更优，使得大型模型微调变得可行。提示/前缀调优的内存占用也非常小。LoRA 和适配器与完全微调相比节省大量资源，但比 QLoRA 或基于提示的方法需要更多内存。
- **推理 (inference)延迟：** 已合并的 LoRA 提供最佳推理性能（无延迟增加）。提示/前缀调优增加的开销极小。适配器会带来不可忽略的延迟惩罚。QLoRA 的推理延迟取决于是否使用反量化或专用核。
- **灵活性：** 所有 PEFT 方法都允许存储小型、任务专用模块，从而可以轻松切换任务，而无需复制大型基础模型。

**方法选择：**

- 对于**最大训练内存效率**，尤其是在大型模型上：**QLoRA**。
- 对于**性能和效率的平衡以及零推理延迟**（合并后）：**LoRA**。
- 作为 **LoRA 的替代方案**，如果合并不是优先事项，或者如果经验结果表明它更适合特定任务/架构：**适配器**。
- 对于**绝对最少的可训练参数和存储空间**，或者当不希望对基础模型进行修改时：**提示调优**或**前缀调优**。

最佳选择通常取决于您的特定模型、任务和数据集的经验结果。尝试不同的 PEFT 方法和配置以找到最适合您要求的方法是常见做法。接下来的实践部分将提供实现 LoRA 和 QLoRA 的实际经验，它们是目前最广泛使用和有效的两种 PEFT 技术。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  提出了 QLoRA，一种用于量化大型语言模型的参数高效微调方法，显著降低了内存需求。
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly (2019)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1902.00751](https://doi.org/10.48550/arXiv.1902.00751)
  介绍了适配器模块，作为自然语言处理中迁移学习的一种参数高效方法。

---

[上一节](06-%E5%89%8D%E7%BC%80%E5%BE%AE%E8%B0%83.md) · [下一节](08-%E4%BD%BF%E7%94%A8%20Hugging%20Face%20PEFT%20%E5%BA%93%E8%BF%9B%E8%A1%8C%E5%AE%9E%E7%8E%B0.md)
