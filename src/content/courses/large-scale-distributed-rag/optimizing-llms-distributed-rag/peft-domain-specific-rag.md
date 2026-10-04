---
course: "large-scale-distributed-rag"
chapter: "optimizing-llms-distributed-rag"
lesson: "peft-domain-specific-rag"
sourceId: 6187
sourceUrl: "https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-3-optimizing-llms-distributed-rag/peft-domain-specific-rag"
title: "领域专用RAG的参数高效微调"
description: "应用参数高效微调（PEFT）技术，以调整LLMs适应RAG中的特定应用方向。"
order: 2
plots: []
sourceHash: "d505e0a2f3bef09abc5f5fe2a51041d3c8560bb5626bb39b38e2a88856aa72de"
sourceCorrections: []
---

尽管预训练 (pre-training)的大型语言模型（LLMs）具备全面的能力，但要在针对特定应用方向的检索增强生成（RAG）系统中达到最佳性能和相关性，通常需要进一步调整。对这些大型模型进行全面微调 (fine-tuning)以适应每个细分应用方向或任务，计算成本高昂且操作繁琐，尤其是在处理多样化信息需求的分布式环境中。参数 (parameter)高效微调（PEFT）技术提供了一种实用且有效的方案，使得模型可以显著地进行专业化，而其计算成本和存储开销仅为传统全面微调的一小部分。对如何应用PEFT方法来优化LLMs以适应特定应用方向的RAG进行了详细说明，旨在提高它们理解检索到的上下文 (context)并生成准确、相关回复的能力。

### PEFT在大规模RAG中的理由

在RAG框架内调整LLMs以适应特定应用方向具有多项益处：

- **提升相关性**：LLM能更好地适应目标应用方向的术语和常见查询模式。
- **增强忠实性**：模型可以学习更有效地基于所提供的特定应用方向检索上下文 (context)来生成回复。
- **减少幻觉 (hallucination)**：应用方向专业化可以降低生成看似合理但实际错误信息的可能性，因为模型对该应用方向变得更加熟悉。
- **更好的风格匹配**：LLM可以被训练成以适合特定应用方向（例如，法律RAG的正式风格，客户支持RAG的易懂风格）的风格或语气生成回复。

然而，全面微调 (fine-tuning)（涉及更新大型LLM的所有参数 (parameter)）在分布式RAG环境中带来显著的挑战：

- **高计算成本**：训练需要大量的GPU资源和时间。
- **大存储需求**：每个全面微调后的模型都与基础模型一样大，如果需要许多特定应用方向的版本，会导致存储问题。
- **部署复杂性**：管理和提供众多大型、不同的模型实例会增加运行开销。

PEFT方法通过仅更新模型参数的一小部分，或者引入少量新的可训练参数，同时保持大部分预训练 (pre-training)模型参数冻结来应对这些挑战。

> 全面微调与参数高效微调（PEFT）的训练范围比较。PEFT更新的参数集显著更小。

### RAG中常用的PEFT方法

几种PEFT技术已获得认可。它们对RAG的适用性取决于所需的调整程度、计算预算以及具体的LLM架构等因素。

#### 1. 低秩适配（LoRA）

LoRA是一种被广泛采用的PEFT方法，它在预训练 (pre-training)LLM的层中引入可训练的低秩分解矩阵。对于LLM中给定的权重 (weight)矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA将其更新 $\Delta W$ 表示为两个较小矩阵$B \in \mathbb{R}^{d \times r}$和$A \in \mathbb{R}^{r \times k}$的乘积，其中秩 $r \ll \min(d, k)$。修改后的前向传播变为：


$$
h = W_0 x + \alpha B A x
$$


这里，$x$是输入，$W_0$代表冻结的预训练权重，而$B$和$A$是可训练的LoRA适配器权重。标量$\alpha$是一个缩放因子。

- **优点**：
  - 可训练参数 (parameter)大幅减少（通常小于总量的1%）。
  - LoRA适配器小巧便携，通过更换适配器可以轻松地在不同任务或应用方向之间切换。
  - 推理 (inference)延迟增加极小。
  - 在许多任务上通常能达到与全面微调 (fine-tuning)相当的性能。
- **在RAG中的应用**：非常适合为LLM创建多个特定应用方向版本。例如，一个金融RAG系统可以使用在财务报告和查询上训练的LoRA适配器，而一个医疗保健RAG系统则使用在医学文献上训练的不同适配器。

#### 2. 提示微调及其变体（P-Tuning，前缀微调）

基于提示的PEFT方法不修改模型权重，而是学习“软提示”或“虚拟令牌嵌入 (embedding)”，这些提示或嵌入被附加到输入序列或LLM的内部隐藏状态之前。

- **提示微调**：学习一系列连续的任务专用向量 (vector)（软提示），这些向量被添加到嵌入输入中。基础LLM参数完全冻结。
- **P-Tuning**：类似于提示微调，但也可以涉及可训练的“提示编码器”（例如，一个小型MLP或LSTM）来生成虚拟令牌，提供更强的表达能力。
- **前缀微调**：将可训练的前缀向量添加到每个Transformer层中自注意力 (self-attention)块的键和值上。这允许对模型的内部表示产生更直接的影响。
- **优点**：

  - 参数效率极高，因为只训练提示/前缀嵌入。
  - 非侵入性，因为基础LLM架构和权重未被触动。
- **在RAG中的应用**：对于在不需要大量再训练的情况下指导LLM在特定RAG任务中的行为很有用。例如，可以训练一个软提示来鼓励LLM总结检索到的上下文 (context)、提取特定实体或根据应用方向遵守某种回复格式。

#### 3. 适配器模块（例如，AdapterHub）

适配器模块涉及在预训练LLM的每个Transformer块中插入小型瓶颈状神经网络 (neural network)层（适配器）。在微调期间，只更新这些新添加的适配器层的参数，而原始LLM权重保持冻结。

- **优点**：
  - 模块化；适配器可以针对不同任务进行训练和组合。
  - 参数效率高，尽管通常比LoRA或提示微调的参数多。
- **在RAG中的应用**：与LoRA类似，适配器可以用于应用方向专业化训练。AdapterFusion技术允许在推理时结合多个已学习到的能力，这对于复杂的RAG场景可能有利。

以下表格总结了这些常见的PEFT方法：

| 方法 | 主要机制 | 参数效率 | 基础模型修改 | 在RAG中的典型应用 |
| --- | --- | --- | --- | --- |
| LoRA | 向现有权重添加低秩矩阵 | 极高 | 极小（添加权重） | 应用方向调整，任务专用行为修正 |
| 提示微调 | 学习输入“软提示”（虚拟令牌） | 极其高 | 无（输入级别） | 指导生成风格，任务指令 |
| 前缀微调 | 学习内部Transformer层的前缀 | 极高 | 无（输入级别） | 类似于提示微调，可能更具表达力 |
| 适配器 | 在层之间插入小型前馈网络 | 高 | 中等（添加层） | 应用方向/任务调整，模块化能力注入 |

### 为特定应用方向的RAG实施PEFT

成功应用PEFT来为RAG系统中的特定应用方向定制LLM，需要细致的数据准备、选择合适的PEFT方法以及专注的训练过程。

#### 数据准备

微调 (fine-tuning)数据集的质量和性质极为重要。对于特定应用方向的RAG，理想的数据集由三元组组成：`(应用方向查询, 应用方向检索上下文, 应用方向理想答案)`。

- **应用方向专用性**：查询、上下文 (context)和答案都应能代表目标应用方向。
- **任务匹配**：数据应反映LLM如何预期使用上下文来回答查询。如果任务是基于检索片段进行总结，那么理想答案应是总结。
- **数据来源**：
  - 现有的特定应用方向问答数据集。
  - 使用强大的LLM（例如GPT-4）根据应用方向文档合成生成问题和答案的数据，随后进行人工审查和改进。
  - 来自现有系统的用户交互日志，经过仔细整理和筛选。
- **数据量**：尽管PEFT所需数据量少于全面微调，但仍需要足够数量的高质量示例（数百到数千，取决于任务复杂性和应用方向）。

#### 训练流程

1. **选择基础LLM**：选择一个具有强大通用能力且适合您的RAG系统规模的预训练 (pre-training)LLM。
2. **选择PEFT方法**：根据您对参数 (parameter)效率、期望的性能提升和实施复杂性的要求，选择一种PEFT技术（LoRA、提示微调等）。像Hugging Face的`peft`这样的库极大地简化了这一过程。
3. **配置训练**：
   - 为PEFT方法设置超参数 (hyperparameter)（例如，LoRA的秩 $r$，提示微调的提示长度）。
   - 设置标准训练超参数（学习率、批次大小、训练轮数）。由于可训练参数数量少，PEFT通常需要更少的训练轮数，并且可以使用更小的批次大小。
4. **训练**：执行微调过程。只有PEFT参数的梯度会被计算和更新。LLM的大部分权重 (weight)保持冻结。
5. **评估**：在预留的特定应用方向RAG评估集上评估PEFT调整后的LLM的性能。衡量指标不仅应包括标准语言模型的困惑度，还应包括RAG专用指标，例如：
   - **答案相关性**：生成的答案与查询和应用方向的关联程度如何？
   - **忠实性/基于性**：答案是否准确反映了检索到的上下文中的信息？
   - **特定应用方向准确性**：对于事实问答，特定应用方向问题的准确率是多少？
   - 减少特定应用方向幻觉 (hallucination)。

### 在分布式RAG中运行PEFT调整过的模型

PEFT为在大规模分布式RAG系统中部署应用方向专业化LLM提供了显著优势。

- **适配器管理**：
  - 由于PEFT适配器很小（LoRA的只有几兆字节，提示微调 (fine-tuning)的甚至更小），可以高效存储许多不同的应用方向或任务适配器。
  - 在分布式服务环境中，基础LLM的单个实例可以加载到内存中。当特定应用方向的请求到达时，相应的PEFT适配器可以动态加载并应用于基础模型以处理请求。
  - 这种“基础模型+可切换适配器”架构与部署多个全面微调的LLM相比，大幅减少了内存占用。现代LLM服务框架已开始支持此类动态适配器加载。
- **推理 (inference)性能**：
  - **LoRA和适配器**：可能会由于额外的矩阵乘法或适配器层引入少量（通常可以忽略不计）的额外延迟。这通常可以通过融合操作或仔细的实现来优化。
  - **提示微调**：通常对推理延迟几乎没有影响，因为它只修改输入嵌入 (embedding)。
  - 批量处理使用相同PEFT适配器的请求可以提高整体吞吐量 (throughput)。
- **简化A/B测试**：新的应用方向调整或任务专用PEFT模块可以通过将新适配器与现有适配器一起部署，并将一部分流量路由到它来轻松进行A/B测试。

### 优势与注意事项

参数 (parameter)高效微调 (fine-tuning)为在分布式RAG系统中对LLM进行专业化提供了有吸引力的途径：

**优势**：

- **计算成本降低**：微调所需GPU时间和内存显著减少。
- **存储开销更低**：PEFT适配器很小，可以存储许多专业化版本。
- **更快的迭代**：新应用方向或任务专业化内容的快速开发和部署。
- **减轻灾难性遗忘**：由于基础模型权重 (weight)大部分被冻结，模型比在小数据集上进行全面微调更能保留其通用能力。
- **模块化专业化**：可以为不同的能力或应用方向训练不同的适配器。

**注意事项**：

- **性能上限**：尽管通常具有竞争力，PEFT方法可能无法总是达到全面微调所能实现的绝对最佳性能，尤其是在应用方向差异极大或任务非常复杂的情况下。然而，这种权衡通常是有利的。
- **超参数 (hyperparameter)敏感性**：PEFT的有效性可能取决于方法的选择及其特定的超参数（例如，LoRA秩、提示长度）。通常需要经验性调整。
- **任务干扰**：当使用多个适配器时，如果适配器旨在组合使用或其效果可能产生负面干扰，则需要仔细设计。
- **集成复杂性**：尽管库简化了PEFT，但将动态适配器加载和管理集成到生产LLM服务堆栈中需要精细的工程设计。

通过采用PEFT，组织可以构建更通用、高效且有用的大规模RAG系统，这些系统能够满足众多特定应用方向和信息需求，而无需承担传统微调方法带来的高昂成本。这使得在专业信息检索和生成任务中，能够更敏捷地应对不断变化的业务要求和用户期望。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了LoRA，一种在预训练模型权重中添加低秩矩阵的参数高效微调方法。
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) — Brian Lester, Rami Al-Rfou, Noah Constant (2021)
  Journal: EMNLP (Conference on Empirical Methods in Natural Language Processing); DOI: [10.48550/arXiv.2104.08691](https://doi.org/10.48550/arXiv.2104.08691)
  提出了Prompt Tuning，其中仅优化少量软提示参数，同时冻结预训练模型。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成（RAG）框架，结合了预训练检索器与序列到序列模型。
