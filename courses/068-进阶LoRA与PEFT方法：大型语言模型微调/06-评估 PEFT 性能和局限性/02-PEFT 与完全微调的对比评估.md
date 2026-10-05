# PEFT 与完全微调的对比评估

来源：[原文](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-6-evaluating-peft-limitations/peft-vs-full-finetuning-benchmarks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

了解参数 (parameter)高效微调 (fine-tuning)（PEFT）方法与传统完全微调方式的比较情况，对于模型适应中的明智决策很重要。虽然PEFT承诺大幅减少计算成本和内存占用，但量化 (quantization)其对下游任务性能的潜在影响（如果有的话）同样重要。对比评估提供了这一重要的比较视角。

### 建立基线：完全微调 (fine-tuning)的表现

在完全微调中，预训练 (pre-training)模型的所有参数 (parameter)都会在训练期间更新，这通常代表了给定模型架构和数据集的性能上限。它允许模型最大限度地灵活调整其内部表示，以适应目标任务的细节。然而，如第一章所述，这会带来相当大的成本，特别是对于拥有数十亿参数的大型语言模型（LLM）而言。这种高性能成为衡量PEFT方法的主要对比标准。

### 主要比较维度

PEFT与完全微调 (fine-tuning)的对比评估通常涉及从多个维度进行评估：

1. **任务表现：** PEFT适应的模型在目标下游任务上的表现与完全微调的模型相比如何？这通常是主要考量。
2. **参数 (parameter)效率：** 实际训练了多少参数？这量化 (quantization)了可训练参数的减少量，是衡量效率的直接指标。
3. **计算成本（训练）：** 训练时间、GPU内存需求（峰值和平均）以及总浮点运算次数（FLOPs）有何差异？
4. **计算成本（推理 (inference)）：** 推理延迟或吞吐量 (throughput)有差异吗？（如果PEFT权重 (weight)已合并，通常影响很小，但如果适配器保持独立，则有关联。）
5. **存储：** 适应后的权重需要多少存储空间？PEFT方法每个任务只需存储一小组适配器权重，而完全微调则需要完整的模型副本。

### 常见对比评估和任务

比较通常使用已建立的自然语言理解（NLU）和生成（NLG）对比评估套件进行。范例包括：

- **GLUE（通用语言理解评估）：** 一组多样化的NLU任务，包括情感分析、文本蕴含和相似性。
- **SuperGLUE：** 一组更具挑战性的NLU任务，旨在拓展语言理解模型的性能限制。
- **SQuAD（斯坦福问答数据集）：** 抽取式问答。
- **摘要数据集（例如，CNN/Daily Mail, XSum）：** 抽象式文本摘要。
- **翻译数据集（例如，WMT）：** 机器翻译。
- **指令遵循数据集（例如，Alpaca, Dolly）：** 评估遵循自然语言指令的能力。

使用多样化的任务集有助于提供全面的情况，因为相对表现可能因任务类型和复杂性而异。

### 分析性能权衡

众多研究和经验结果已将各种PEFT技术（如LoRA、适配器微调 (fine-tuning)、前缀微调等）与完全微调在不同模型尺寸和任务上进行了比较。重要的观察结果通常包括：

- **接近可比的表现：** 对于许多任务，特别是在GLUE/SuperGLUE等NLU对比评估中，配置得当的PEFT方法（如LoRA）通常能达到与完全微调非常接近的性能，有时甚至能达到完全微调分数的95-100%。
- **模型规模的影响：** 随着基础模型尺寸的增加，PEFT与完全微调之间的性能差距趋于缩小。对于非常大的模型（100亿+参数 (parameter)），PEFT不仅高效，而且常常是必需的，并且性能保持非常强劲。
- **任务敏感性：** 在需要大量知识注入或复杂多步推理 (inference)的任务上，性能差距可能略大，因为更新更多参数*可能*会提供优势。然而，即使在复杂任务上，PEFT方法也经常提供有竞争力的结果。
- **数据条件：** 在数据量较少的情况下，PEFT方法有时可以胜过完全微调，这可能归因于一种正则化 (regularization)效应，通过限制可训练参数的数量来防止对小数据集过拟合 (overfitting)。



[交互图表：表现与可训练参数对比](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-6-evaluating-peft-limitations/peft-vs-full-finetuning-benchmarks#plot-14dqh35)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "表现与可训练参数对比",
    "xaxis": {
      "title": "可训练参数（对数尺度）",
      "type": "log",
      "tickformat": ".1s"
    },
    "yaxis": {
      "title": "下游任务准确率（%）",
      "range": [
        75,
        100
      ]
    },
    "legend": {
      "title": "方法"
    },
    "margin": {
      "l": 60,
      "r": 30,
      "t": 40,
      "b": 50
    }
  },
  "data": [
    {
      "x": [
        10000000000
      ],
      "y": [
        92
      ],
      "mode": "markers",
      "marker": {
        "color": "#fa5252",
        "size": 12
      },
      "name": "完全微调"
    },
    {
      "x": [
        50000000
      ],
      "y": [
        90.5
      ],
      "mode": "markers",
      "marker": {
        "color": "#4263eb",
        "size": 12
      },
      "name": "LoRA (r=16)"
    },
    {
      "x": [
        70000000
      ],
      "y": [
        89
      ],
      "mode": "markers",
      "marker": {
        "color": "#12b886",
        "size": 12
      },
      "name": "适配器微调"
    },
    {
      "x": [
        40000000
      ],
      "y": [
        88
      ],
      "mode": "markers",
      "marker": {
        "color": "#f59f00",
        "size": 12
      },
      "name": "前缀微调 (L=20)"
    }
  ]
}
```

</details>



> 分类任务上的表现与微调期间更新的参数数量对比。PEFT方法达到了有竞争力的准确率，同时训练的参数数量比完全微调少几个数量级。

### 效率提升量化 (quantization)

PEFT的主要动机是效率，对比评估清楚地显示了这些优势：

- **参数 (parameter)：** PEFT方法通常只训练总模型参数的远低于1%（通常低于0.1%）。对于像GPT-3（1750亿参数）这样的模型，完全微调 (fine-tuning)会更新所有1750亿参数，而LoRA可能只更新数千万或数亿参数。
- **内存：** 可训练参数的减少直接转化为更低的GPU内存需求，用于存储优化器状态（例如，Adam/AdamW为每个参数维护动量和方差项）。像QLoRA这样的技术通过量化基础模型，进一步大幅减少内存使用。
- **训练时间：** 尽管训练时间并非完全取决于参数数量（数据加载、通过完整模型的前向/反向传播 (backpropagation)仍然发生），但优化器开销的减少和潜在的更快收敛可以带来显著的加速，特别是在与量化结合时。



[交互图表：训练成本比较](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-6-evaluating-peft-limitations/peft-vs-full-finetuning-benchmarks#plot-wi1g50)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "训练成本比较",
    "xaxis": {
      "title": "方法"
    },
    "yaxis": {
      "title": "相对成本（对数尺度）",
      "type": "log",
      "range": [
        0,
        3
      ]
    },
    "margin": {
      "l": 60,
      "r": 30,
      "t": 40,
      "b": 50
    },
    "barmode": "group"
  },
  "data": [
    {
      "x": [
        "完全微调",
        "LoRA",
        "适配器微调",
        "QLoRA"
      ],
      "y": [
        1000,
        10,
        15,
        2
      ],
      "type": "bar",
      "name": "GPU内存",
      "marker": {
        "color": "#7048e8"
      }
    },
    {
      "x": [
        "完全微调",
        "LoRA",
        "适配器微调",
        "QLoRA"
      ],
      "y": [
        500,
        20,
        25,
        8
      ],
      "type": "bar",
      "name": "训练时间",
      "marker": {
        "color": "#f76707"
      }
    }
  ]
}
```

</details>



> 与不同微调方法相关的相对训练内存和时间成本（对数尺度）。PEFT方法，特别是QLoRA，与完全微调相比，提供了大幅减少。

### 适用各方法的场景

- **选择PEFT的情况：**

  - 计算资源（GPU内存、时间）有限时。
  - 对于同一个基础模型需要执行多个任务或进行多项定制时（存储小型适配器而非完整的模型副本）。
  - 需要快速实验和迭代时。
  - 接近完全微调 (fine-tuning)的性能可以接受时（通常如此）。
  - 部署到内存受限的边缘设备是一个考量时（如果适用，在合并权重 (weight)后）。
- **考虑完全微调的情况：**

  - 性能最大化是绝对的优先事项，即使是小百分比的提升也很重要时。
  - 计算资源充足时。
  - 任务需要深度适应，可能受益于调整所有模型权重时。
  - 只需要一个或很少的适应任务时。

实际中，PEFT方法，特别是LoRA及其变体如QLoRA，已成为适应大型预训练 (pre-training)模型的默认选择，因为它们在高性能和大幅降低的计算需求之间取得了良好的平衡。与完全微调的对比评估提供了必要的验证，即这种效率在大多数应用中不会以任务准确性为代价产生过高的成本。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: ICLR 2022; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了LoRA，展示了其在各种NLP任务中与全量微调相当的效率和性能。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: NeurIPS 2023; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  提出了QLoRA，一种微调量化LLM的方法，它显著降低了内存需求，同时保持了性能。
- [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods](https://arxiv.org/abs/2303.15647) — Vladislav Lialin, Vijeta Deshpande, Xiaowei Yao, Anna Rumshisky (2023)
  Journal: arXiv preprint arXiv:2303.15647; DOI: [10.48550/arXiv.2303.15647](https://doi.org/10.48550/arXiv.2303.15647)
  一个统一的框架和经验性概述，涵盖了各种PEFT方法，包括它们的性能比较和实现细节。

---

[上一节](01-PEFT%E8%AF%84%E4%BC%B0%E7%9A%84%E5%B8%B8%E7%94%A8%E6%8C%87%E6%A0%87.md) · [下一节](03-%E5%88%86%E6%9E%90%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8A%97%E5%B9%B2%E6%89%B0%E8%83%BD%E5%8A%9B%E5%92%8C%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B.md)
