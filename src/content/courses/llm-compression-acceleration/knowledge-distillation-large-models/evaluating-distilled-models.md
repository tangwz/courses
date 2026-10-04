---
course: "llm-compression-acceleration"
chapter: "knowledge-distillation-large-models"
lesson: "evaluating-distilled-models"
sourceId: 4374
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-4-knowledge-distillation-large-models/evaluating-distilled-models"
title: "评估蒸馏模型性能"
description: "建立评估蒸馏大型语言模型性能、泛化能力和忠实度的指标和规程。"
order: 7
plots: ["plots/4374-0.json"]
sourceHash: "2d32ec973dcd5c5941ed30b6881a8ca553a5bdb1f8b1d2e0f327cac3a0d43a1f"
sourceCorrections: []
---

成功通过知识蒸馏 (knowledge distillation)训练学生模型后，下一个重要步骤是严格的评估过程。蒸馏的主要目标是创建一个更小、更快的模型，使其性能接近更大的教师模型。因此，评估必须全面衡量其*忠实度*（学生模型在预期任务上的表现）和*效率*提升（在模型大小、延迟和计算成本上的缩减）。仅仅实现模型变小是不够的；我们必须量化 (quantization)涉及的权衡。

### 评估任务表现与忠实度

衡量学生模型的能力，需要评估其在教师模型所设计的相同任务上，或在蒸馏过程中所针对的特定下游任务上的表现。

#### 标准基准与特定应用任务

如果目标是创建通用型压缩模型，评估应涵盖适合模型模态的各种标准基准。对于自然语言理解 (NLU) 模型，这通常包括 GLUE (通用语言理解评估) 或 SuperGLUE 等套件。对于生成模型，在保留文本语料库上的困惑度仍是一个常见但有局限的内在指标。

非常重要的一点是，在使用*完全相同的评估设置*下，直接将学生模型在这些基准上的得分与教师模型的得分进行比较。这为性能下降建立了清晰的基线。

如果蒸馏是针对特定下游应用（例如，情感分析、文档摘要、代码生成）进行定制的，则应优先使用与这些任务最相关的指标进行评估，例如准确率 (Accuracy)、F1分数 (F1-Score)、ROUGE、BLEU、CodeBLEU 或精确匹配 (Exact Match, EM)。

#### 生成模型的评估难点

评估蒸馏后的生成式大型语言模型 (LLM) 在标准分类或回归指标上带来特有难题。虽然 BLEU、ROUGE 和 METEOR 等自动化指标为翻译或摘要等任务提供了可扩展的比较方式，但它们与人类对质量、连贯性和事实准确性的判断关联度不高。

- **困惑度 (PPL)：** 较低的困惑度通常表示更好的流畅性和分布匹配，但不能保证生成内容的质量或实用性。学生模型可能通过过度重复或过于保守来达到低困惑度。
- **高级自动化指标：** 使用嵌入 (embedding)（embeddings）（例如 BERTScore）或使用其他大型语言模型作为评估器（例如 G-Eval，使用 GPT-4 评分输出）的技术提供了更细致的评估，但引入了对评估器模型的依赖。
- **人工评估：** 这仍是衡量创造力、连贯性、指令遵循能力、安全对齐 (alignment)和整体实用性等方面最可靠的方法。设计明确的评估标准，并使用多位标注者以确保一致性。尽管资源密集，但对于高风险应用，人工评估通常是必需的。

#### 评估鲁棒性与公平性

更详细地分析学生模型的行为表现：

- **分布外 (OOD) 泛化能力：** 在与训练/蒸馏数据分布略有不同的数据集上评估学生模型。学生模型能否像教师模型一样保持性能，还是性能下降得更剧烈？如果学生模型在蒸馏数据集上对教师模型的特定输出模式过拟合 (overfitting)，蒸馏有时会损害鲁棒性。
- **错误分析：** 对学生模型与教师模型相比所犯的错误类型进行分类。蒸馏是否引入了新的失败模式？某些能力是否受到不成比例的影响？
- **公平性与偏见：** 使用公平性指标和偏见检测数据集（例如 BOLD、StereoSet）评估学生模型在不同人口群体或敏感属性方面是否存在潜在偏见。将这些结果与教师模型进行比较，以了解蒸馏是否减轻、保留或加剧了现有偏见。确保效率提升不会以增加不公平为代价，这一点非常重要。

### 评估效率提升

量化 (quantization)通过蒸馏实现的效率提升通常更直接，但需要在实际部署场景中进行仔细测量。

- **模型大小：** 测量最终学生模型检查点的参数 (parameter)数量和存储占用空间（例如，磁盘上的兆字节或千兆字节）。将其直接与教师模型大小进行比较。
- **推理 (inference)延迟：** 测量在目标硬件（例如，特定 CPU、GPU、TPU 或专用 NPU）上处理单个输入或一批输入的平均时间。指定测量条件，包括批大小、序列长度和硬件配置，因为这些会显著影响延迟。测量首个令牌延迟（用于交互式应用）以及每输出令牌延迟或总生成时间（用于生成任务）。
- **吞吐量 (throughput)：** 测量每秒完成的推理次数（或生成的令牌数量），通常在持续负载下进行。这是服务系统的一个重要指标。
- **计算成本 (FLOPs)：** 估算每次推理所需的浮点运算次数。这提供了一种与硬件无关的计算复杂度衡量标准，对理论比较有用。存在根据模型架构估算 FLOPs 的工具。
- **内存占用：** 测量推理过程中的峰值 RAM 或 VRAM 消耗。这通常是一个严格限制，尤其是在移动或边缘设备上部署时。同时考虑静态模型权重 (weight)内存和动态激活内存。

### 比较分析与可视化

有效的评估包括不仅将学生模型与教师模型比较，还要与相关基线进行比较。

- **学生模型与教师模型权衡：** 这是核心分析。量化 (quantization)模型大小、延迟或 FLOPs 的百分比缩减与任务性能指标的百分比变化（通常是下降）。将这种权衡可视化通常很有用。
- **学生模型与非蒸馏基线：** 将蒸馏后的学生模型与未经蒸馏的传统训练（从头开始或标准微调 (fine-tuning)）的*大小和架构相似*的其他模型进行比较。学生模型是否表现优于这些大小匹配的基线？这种比较分离出从更大的教师模型迁移知识所带来的特定好处。
- **消融研究：** 如果尝试了多种蒸馏技术（不同的损失函数 (loss function)、温度设置、中间层匹配），进行消融研究以了解每个组件对最终性能和效率的贡献。

一种常见的性能-效率权衡可视化方式是使用散点图比较模型在这两个轴上的表现。



![模型性能与推理延迟](plots/4374-0.json)



> 教师模型、各种蒸馏学生模型以及未经蒸馏训练的大小匹配基线的准确率与延迟对比。理想区域是左上角（高准确率，低延迟）。蒸馏模型（蓝色圆圈）通常优于从头训练的大小匹配基线（黄色菱形），展现了知识迁移的价值。

### 建立评估规程

为确保结果有意义且可靠：

- **一致性：** 比较不同模型时，使用相同的数据集、预处理步骤、指标计算脚本和硬件环境。
- **统计有效性：** 使用不同的随机种子（用于模型初始化、数据打乱等）多次运行评估，并报告平均结果以及标准偏差或置信区间。在适当情况下进行统计显著性检验，尤其是在比较性能相似的模型时。
- **目标环境：** 在可能的情况下，在实际目标硬件上，并在最终部署场景中预期的条件下（例如，批大小、量化 (quantization)）进行效率测量（延迟、吞吐量 (throughput)、内存）。性能在不同的硬件平台和软件执行引擎（例如 PyTorch eager 模式、TorchScript、ONNX Runtime、TensorRT）之间可能差异显著。

归根结底，评估蒸馏模型是一个多方面过程。它需要仔细测量任务特定性能，考虑潜在的性能下降、鲁棒性和公平性影响，同时量化计算效率的提升。一个成功的蒸馏策略会产生一个学生模型，在这些因素之间取得可接受的平衡，从而满足目标应用和部署环境的特定要求。

## 参考资料

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Geoffrey Hinton, Oriol Vinyals, Jeff Dean (2015)
  Journal: arXiv preprint arXiv:1503.02531; DOI: [10.48550/arXiv.1503.02531](https://doi.org/10.48550/arXiv.1503.02531)
  介绍知识蒸馏概念的开创性论文，为评估学生模型对教师模型的忠实度提供了基础。
- [GLUE: A Multi-Task Benchmark for Natural Language Understanding](https://arxiv.org/abs/1804.07461) — Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman (2018)
  Journal: ICLR 2019 (published as arXiv preprint); DOI: [10.48550/arXiv.1804.07461](https://doi.org/10.48550/arXiv.1804.07461)
  介绍了GLUE基准测试，这是评估NLU模型的标准套件，对于评估蒸馏模型的任务性能和忠实度至关重要。
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) — Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi (2019)
  Journal: ICLR 2020 (published as arXiv preprint); DOI: [10.48550/arXiv.1904.09675](https://doi.org/10.48550/arXiv.1904.09675)
  提出了BERTScore，一种利用上下文嵌入与人类判断更相关的文本生成自动评估指标。
- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu (2023)
  Journal: arXiv preprint arXiv:2303.16634; DOI: [10.48550/arXiv.2303.16634](https://doi.org/10.48550/arXiv.2303.16634)
  介绍了G-Eval，一种利用大型语言模型作为评估器来评估自然语言生成的方法，提高了与人类判断的一致性。
