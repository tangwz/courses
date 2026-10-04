---
course: "lora-peft-efficient-llm-training"
chapter: "evaluating-peft-limitations"
lesson: "peft-computational-cost-revisited"
sourceId: 4180
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-6-evaluating-peft-limitations/peft-computational-cost-revisited"
title: "计算成本分析再议"
description: "详细比较不同 PEFT 策略的训练时间、推理延迟和内存使用情况。"
order: 5
plots: ["plots/4180-0.json", "plots/4180-1.json"]
sourceHash: "69ae2ae1841dfd3331913d8bd53780d247c0e6605af63ff3ba034e9fac8da46e"
sourceCorrections: []
---

参数 (parameter)高效微调 (fine-tuning) (PEFT) 方法因其带来的效率提升而得到广泛认可，这也是其主要动因。对计算成本进行量化 (quantization)分析，对于在实际应用中做出明智决策是必需的。选择正确的 PEFT 策略通常需要在下游任务的性能与 GPU 内存、训练时间、推理 (inference)延迟和存储容量等实际资源限制之间取得平衡。这里将提供不同 PEFT 技术之间的计算成本详细比较，并将其与完整微调进行对比。

### 内存使用：训练与推理 (inference)

内存占用仍然是一个重要瓶颈，尤其是在处理日益庞大的模型时。PEFT 方法在此方面提供了显著优势，主要体现在训练阶段。

#### 训练内存

完整微调 (fine-tuning)需要存储*所有*模型参数 (parameter)的权重 (weight)、梯度和优化器状态。对于拥有数十亿参数的模型，这会迅速消耗数十甚至数百千兆字节的 GPU 内存，即使对于适中的批处理大小也常需要多 GPU 设置。内存使用主要由以下几项构成：

1. **模型参数：** 基础 LLM 的权重。
2. **梯度：** 在反向传播 (backpropagation)过程中为每个可训练参数计算。大小与参数相同。
3. **优化器状态：** 像 AdamW 这样的优化器会存储额外的动量和方差信息，对于 32 位精度，通常需要两倍于参数本身的内存。
4. **激活值：** 前向传播过程中保存的中间结果，用于反向传播中的梯度计算。这会随着批处理大小、序列长度和模型深度/宽度而变化。

PEFT 方法大幅减少了梯度和优化器状态所需的内存。通过冻结基础模型并仅训练少量适配器参数（例如，LoRA 矩阵、适配器层、前缀），与可训练参数相关的内存开销显著缩小。

- **LoRA：** 只有低秩矩阵 $A$ 和 $B$ 需要梯度和优化器状态。如果原始权重矩阵是 $W \in \mathbb{R}^{d \times k}$ 并且 LoRA 使用秩 $r$，则每层的可训练参数数量从 $d \times k$ 变为 $r \times (d+k)$，这对于 $r \ll \min(d, k)$ 来说会大幅减小。
- **适配器微调：** 需要为适配器层的参数、梯度和优化器状态分配内存。大小取决于适配器的瓶颈维度。
- **前缀/提示微调：** 只训练前缀或提示嵌入 (embedding)，导致可训练参数数量非常少。
- **QLoRA：** 通过以下方式在训练期间实现进一步显著的内存减少：
  - 将基础模型参数量化 (quantization)为 4 位（使用 NF4 格式）。
  - 对量化常数使用双重量化。
  - 采用分页优化器，在 GPU 内存耗尽时将优化器状态卸载到 CPU 内存。

与完整微调甚至标准 LoRA 相比，这使得在普通硬件上训练显著更大的模型成为可能。



![相对训练内存占用比较](plots/4180-0.json)



> 训练期间内存组成的示例性比较。QLoRA 通过量化显著减少了基础模型内存，而所有 PEFT 方法都大幅削减了梯度/优化器状态内存。激活内存很大程度上取决于批处理大小和序列长度，在此为便于比较假定为常数。

#### 推理内存

在推理期间，主要的内存消耗者是模型权重。

- **完整微调：** 每个特定任务模型都是一个完整副本，每个部署的模型都需要大量内存。
- **PEFT（LoRA、适配器等）：** 允许部署单个基础模型副本，并为不同任务动态加载小套 PEFT 参数（适配器）。这在多任务或多租户场景中显著减少了内存占用。基础模型使用大部分内存，每个适配器仅增加数兆字节。
- **合并 LoRA：** 如果 LoRA 适配器在训练后合并到基础模型权重中，推理内存占用与原始基础模型相同。这消除了多适配器的优势，但如果只需要一个任务，则简化了部署。
- **QLoRA：** 如果使用 4 位量化的基础模型进行部署，QLoRA 为基础权重提供了最低的推理内存占用，尽管适配器权重通常以更高精度存储。

### 训练时间和计算量 (FLOPs)

尽管 PEFT 显著减少了可训练参数 (parameter)，但其对每步原始训练 FLOPs（浮点运算）的影响，不如内存节省可能表明的那么显著。

- **前向传播：** 前向传播计算主要由冻结的基础模型层中的大型矩阵乘法（例如，注意力、前馈网络）主导。所有方法，包括 PEFT，都需要承担此成本，因为需要完整的模型上下文 (context)。像适配器微调 (fine-tuning)这样的方法由于插入的层而增加了少量额外的 FLOPs。LoRA 增加了极小的 FLOPs（$BAx$ 计算）。QLoRA 由于前向传播中的反量化 (quantization)操作而增加了开销。
- **反向传播 (backpropagation)与优化器步骤：** PEFT 方法在此处显示出显著节省。梯度计算和优化器更新仅对一小部分适配器参数执行，与更新整个模型相比，这大幅减少了与这些步骤相关的 FLOPs。

尽管前向传播占据主导地位，PEFT 仍常导致更快的*实际*训练时间，原因在于：

1. **更大的批处理大小：** 内存使用量减少使得在相同硬件上可以使用更大的批处理大小，从而提高 GPU 利用率，并可能加快每轮的收敛速度。
2. **更快的收敛：** 对于某些任务，PEFT 方法可能比完整微调在更少的步骤或轮次内收敛，尽管这取决于任务。
3. **减少通信（分布式训练）：** 在分布式设置中，仅同步 PEFT 参数的梯度会显著减少通信带宽要求，相比于同步整个模型的梯度。

QLoRA 的量化/反量化为每一步增加了计算开销，但内存节省通常允许采用抵消此成本的训练配置（更大的模型、更大的批处理），从而在内存受限的硬件上实现更快的整体训练。

### 推理 (inference)延迟

推理延迟是模型接收输入后生成响应所需的时间。这对于面向用户的应用程序很重要。

- **完整微调 (fine-tuning)：** 根据基础模型架构和大小设定基线延迟。
- **合并 LoRA：** 增加了*零*延迟开销，因为适配器权重 (weight)已融合到基础模型层中。推理与使用原始基础模型相同。
- **未合并 LoRA：** 由于额外的矩阵乘法（$A$ 和 $B$）而增加了少量延迟开销。其影响取决于秩 $r$ 和所适配的具体层，但对于典型秩来说通常是极小的。
- **适配器微调：** 引入延迟，因为输入必须顺序通过插入的适配器层。这种开销通常比未合并 LoRA 更大。
- **前缀/提示微调：** 增加了极小的延迟，主要轻微影响初始嵌入 (embedding)查找或注意力计算。
- **QLoRA：** 由于在前向传播期间需要对权重进行反量化 (quantization)，因此会增加延迟。除非使用专用硬件或优化过的内核（如 NVIDIA 的 FasterTransformer 或 TensorRT-LLM）来加速混合精度或量化计算，否则这可能会很明显。



![相对推理延迟开销](plots/4180-1.json)



> 不同 PEFT 方法相对于基础模型增加的推理延迟开销比较。实际值很大程度上取决于实现、硬件和配置（例如，LoRA 秩、适配器大小、QLoRA 优化）。

### 存储成本

PEFT 方法在存储空间方面提供了巨大的节省。

- **完整微调 (fine-tuning)：** 每个微调后的模型检查点都会保存*整套*模型权重 (weight)，导致文件大小可能达到数百千兆字节。管理多个特定任务版本会变得存储密集。
- **PEFT：** 只需保存训练过的适配器参数 (parameter)。这些参数通常比基础模型小几个数量级（兆字节对千兆字节）。这使得存储和管理与单个基础模型相关联的大量特定任务适配器变得高效。

### 成本权衡总结

最佳 PEFT 选择很大程度上取决于您项目的具体限制和优先级：

- **最小训练内存：** QLoRA 是首选，它使得在有限硬件上微调 (fine-tuning)非常大的模型成为可能。
- **零推理 (inference)延迟开销：** 合并 LoRA 是理想选择，前提是合并与部署策略兼容。
- **多任务部署效率（内存）：** 任何将适配器保持分离的 PEFT 方法（未合并 LoRA、适配器、前缀/提示微调）都允许共享基础模型，与部署多个完整微调模型相比，大幅减少了推理内存。
- **存储效率：** 所有 PEFT 方法都比完整微调具有显著优势。
- **简洁性与兼容性：** LoRA 在 Hugging Face 的 PEFT 等流行库中得到了广泛支持。

将这些计算成本与任务性能（在本章其他部分讨论）一同分析，可以进行全面的评估。理解每种 PEFT 技术的内存、计算、延迟和存储影响，是您在考虑资源的环境中有效选择、实施和部署这些强大的微调策略的重要组成部分。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了 LoRA，一种通过添加低秩矩阵显著减少 LLM 微调所需训练参数和内存的方法。
- [Parameter-Efficient Fine-tuning (PEFT) documentation](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Hugging Face PEFT 库的官方文档，提供了各种 PEFT 方法的实际实现细节和使用指南。
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly (2019)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1902.00751](https://doi.org/10.48550/arXiv.1902.00751)
  介绍了适配器层作为一种参数高效的方法，用于将预训练模型适应下游任务。
