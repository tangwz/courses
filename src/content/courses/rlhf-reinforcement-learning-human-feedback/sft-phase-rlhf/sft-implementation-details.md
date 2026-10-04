---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "sft-phase-rlhf"
lesson: "sft-implementation-details"
sourceId: 5106
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-2-sft-phase-rlhf/sft-implementation-details"
title: "SFT 实现细节"
description: "LLM 微调的技术方面，包括超参数和训练设置。"
order: 3
plots: ["plots/5106-0.json"]
sourceHash: "bd2a9e146d59506462d2c934810cbe2925a609ae9dd3e1a89a7783a4295deeee"
sourceCorrections: []
---

实施大型语言模型 (LLM) 的监督微调 (fine-tuning) (SFT) 过程，需要理解其具体实践。这个阶段涉及到一个预训练 (pre-training)的基础模型，并使用特定的演示数据集对其进行进一步训练，以在更复杂的强化学习 (reinforcement learning) (RL) 阶段之前调整其行为。仔细关注实施细节对于获得表现良好的 SFT 模型十分重要，这为成功的 RLHF 奠定了基础。

### 选择基础模型和框架

SFT 的起点通常是一个大型的预训练 (pre-training)语言模型。常用的选择包括 GPT 系列、Llama、Mistral 模型或其他开源替代方案。选择取决于任务要求、计算资源和许可限制等因素。

大多数 SFT 实施会利用 PyTorch 或 TensorFlow 等已有的深度学习 (deep learning)框架，通常结合专门为 Transformer 模型设计的库，例如 Hugging Face 的 `transformers`。这些库提供预置的模型架构、分词 (tokenization)器 (tokenizer)和训练工具，大大简化了微调 (fine-tuning)过程。使用 `transformers`，你只需几行代码就能加载一个预训练模型及其对应的分词器：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf" # 示例模型
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 数据准备和格式化

由提示-响应对组成的整理好的 SFT 数据集需要为模型正确格式化。这包括：

1. **结构化输入：** 将提示和响应连接成一个单一序列。一种常用格式是这样构造输入：`"<s>[INST] 提示文本 [/INST] 响应文本 </s>"`。特殊标记 (token)（如 `<s>`、`</s>`、`[INST]`、`[/INST]`）用于区分输入的不同部分，并标识回合或序列的开始/结束。这些标记取决于所使用的具体基础模型；请查阅模型文档。
2. **分词 (tokenization)：** 将结构化文本序列转换为模型可以理解的数值输入 ID。必须使用与基础模型关联的分词器 (tokenizer)。请注意填充（添加特殊标记以使批次中的序列长度相同）和截断（裁剪超出模型最大上下文 (context)长度的序列）。

   ```python
   # 使用 Hugging Face 分词器的示例
   prompt = "Explain the difference between supervised and unsupervised learning."
   response = "Supervised learning uses labeled data..."
   formatted_text = f"<s>[INST] {prompt} [/INST] {response} </s>"

   # 对格式化文本进行分词
   inputs = tokenizer(formatted_text, return_tensors="pt", padding=True, truncation=True, max_length=1024) 
   # inputs['input_ids']，inputs['attention_mask']
   ```
3. **标签遮蔽：** 在训练期间，模型应只学习预测响应标记，而不是提示标记。我们通过将与提示标记对应的标签 ID 设置为一个特殊值（通常是 -100）来实现此目的，该值会被标准交叉熵损失函数 (loss function)忽略。只有响应标记对损失计算有贡献。

### 训练配置

配置训练循环包括设置优化器、损失函数 (loss function)以及控制学习过程的各种超参数 (parameter) (hyperparameter)。

#### 优化器和损失函数

- **优化器：** AdamW 优化器是训练 Transformer 模型的常用且有效的选择。它通过分离的权重 (weight)衰减扩展了 Adam 优化器，这可以提高模型的泛化能力。
- **损失函数：** 对于 SFT 这样的语言建模任务，标准损失函数是交叉熵损失。它衡量模型预测的下一个标记 (token)的概率分布与序列中实际的下一个标记之间的差异。PyTorch 等库能高效处理计算，尤其是在结合上述标签遮蔽时。

#### 超参数

调整超参数对于稳定有效的 SFT 训练非常重要。其中一些最重要的包括：

- **学习率：** 这决定了优化过程中的步长。对于微调 (fine-tuning)大型模型，通常使用较小的学习率（例如 $1e^{-5}$ 到 $5e^{-5}$），以避免过于剧烈地破坏预训练 (pre-training)权重。学习率调度，通常包括一个预热阶段，随后是线性或余弦衰减，是标准做法。这有助于初步稳定训练，并允许后期进行更精细的调整。



![示例学习率调度（预热 + 线性衰减）](plots/5106-0.json)



> 典型的学习率调度从零开始，在预热期线性增加，然后在剩余的训练步数中线性衰减至零。

- **批次大小：** 每个训练步中处理的样本数量。较大的批次大小提供更稳定的梯度估计，但需要更多 GPU 内存。如果内存有限，梯度累积可以通过在几个较小的步骤中累积梯度，然后在执行优化器更新之前，来模拟一个更大的批次大小。
- **周期数：** 一个周期是对整个训练数据集的一次完整遍历。微调通常只需要几个周期（例如 1-5 个），因为模型已经预训练过。训练时间过长可能导致在 SFT 数据集上过拟合 (overfitting)。
- **权重衰减：** 一种正则化 (regularization)技术（常包含在 AdamW 优化器中），它惩罚大权重，有助于防止过拟合。典型值很小，例如 0.01。
- **梯度裁剪：** 通过限制梯度的最大范数来防止梯度爆炸（梯度变得过大）。这提高了训练稳定性，尤其是在训练初期。常用的裁剪值是 1.0。
- **序列长度：** 模型一次处理的最大标记数。较长的序列能捕获更多上下文 (context)，但需要显著更多的内存和计算。选择一个适合你任务和硬件限制的长度，通常是对较长样本进行截断或使用滑动窗口。

### 参数 (parameter)高效微调 (fine-tuning) (PEFT)

完全微调（即更新所有模型参数）对于亿级参数模型来说计算成本可能很高。参数高效微调 (PEFT) 方法提供了替代方案，它们只更新一小部分参数或添加新的可训练模块。

- **低秩适应 (LoRA)：** 一种常用的 PEFT 技术。LoRA 会冻结原始的预训练 (pre-training)权重 (weight)，并向特定层（通常是注意力层）注入更小的、可训练的“低秩分解”矩阵。训练期间，只有这些较小的矩阵会得到更新，从而大幅减少可训练参数的数量和内存需求。原始权重保持不变，保留了大部分预训练知识。在推理 (inference)时，学习到的 LoRA 矩阵通常可以合并回原始权重，而不会增加延迟。Hugging Face 的 `peft` 等库提供了 LoRA 的简便实现。

> 说明 LoRA 机制的图示。输入 `x` 由冻结的预训练权重 `W0` 和低秩分解矩阵 `A` 和 `B` 共同处理。训练期间只更新 `A` 和 `B`。

### 硬件和效率考量

- **GPU：** 微调 (fine-tuning)大型 LLM 通常需要拥有大内存的强大 GPU（例如 NVIDIA A100 或 H100）。
- **分布式训练：** 对于非常大的模型或数据集，可能需要 DeepSpeed 或 PyTorch 的 Fully Sharded Data Parallel (FSDP) 等分布式训练框架，以便将模型和数据分布到多个 GPU 或机器上。
- **混合精度训练：** 使用 FP16（16 位浮点）或 BF16 (bfloat16) 等低精度浮点格式可以显著加快训练速度并减少内存使用，同时对模型质量的影响很小。库通常提供简单的标志来启用此功能。

### 保存和检查点

在训练期间定期保存模型检查点。这可以让你：

- 如果中断可以恢复训练。
- 根据在验证集上的评估指标选择表现最好的模型。
- 跟踪模型的不同版本。

当使用 LoRA 等 PEFT 方法时，你通常只需保存小的适配器权重 (weight)，这些权重比完整模型小得多，这使得检查点保存非常高效。

通过仔细配置这些实施细节，你可以有效地在演示数据上微调 (fine-tuning)你的基础 LLM，创建一个强大的 SFT 模型，为 RLHF 流水线中后续的奖励模型和 RL 微调阶段做好准备。下一节将介绍如何评估这个 SFT 模型的性能。

## 参考资料

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  `transformers`库的官方文档，提供了LLM模型加载、分词和训练工具的详细信息，与本节的代码示例直接相关。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang and Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR 2022); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA），这是一种参数高效微调方法，显著减少了大型语言模型的可训练参数数量和内存占用。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR 2019); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  提出了AdamW优化器，该优化器包含解耦权重衰减，用于改进深度神经网络训练时的正则化和泛化能力。
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  `peft`库的官方文档，提供了各种参数高效微调技术（包括LoRA）的实用实现和指南，与本节内容相关。
