# 量化 LoRA (QLoRA) 与 4 位训练

来源：[原文](https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-4-parameter-efficient-fine-tuning-peft/quantized-lora-qlora-4-bit-training)

[返回章节目录](README.md) · [返回课程目录](../README.md)

低秩自适应 (LoRA) 减少了训练期间更新的参数 (parameter)数量，但基础模型本身仍然占用大量显存 (VRAM)。一个以 16 位精度加载的标准 70 亿参数语言模型，仅在闲置状态下就需要约 14 GB 的内存。一旦加上优化器状态、梯度和批次数据，显存需求会迅速超过标准 24 GB 消费级 GPU 的限制。量化 (quantization) LoRA 正是为了解决这个显存瓶颈。

量化通过降低表示模型权重 (weight)所用数字的精度来发挥作用。量化不再将每个参数存储为 16 位或 32 位浮点数，而是将这些值映射到更低位的表示形式，如 8 位或 4 位整数。从 16 位降低到 4 位可以使模型的内存占用减少近 75%。然而，标准的量化可能会导致信息丢失，从而降低模型的性能。

QLoRA 引入了一种优化方法，可以在严格的内存限制下运行，同时保持原有的性能水平。它的工作原理是将预训练 (pre-training)的基础模型以特殊的 4 位格式加载并冻结其权重。然后，将标准的 16 位 LoRA 适配器附加到这些冻结层上。在前向传播过程中，4 位权重会临时反量化为 16 位，以便与输入进行矩阵乘法，并将结果与 16 位 LoRA 适配器的输出合并。

为了在不降低文本生成质量的情况下实现这一点，QLoRA 采用了三种特定的技术：

**4 位正态浮点数 (NF4)**
神经网络 (neural network)的权重通常遵循以零为中心的止正态分布。NF4 是一种专门为此分布设计的、信息论上最优的数据类型。它在权重最集中的零附近分配了更多的位表示，从而确保了最常用值的更高精度。

**双重量化**
量化模型需要缩放因子（称为量化常数）来进行数值的往返映射。在数百万个参数块中存储这些常数会消耗额外的内存。双重量化对这些常数进行第二次量化，将其内存占用从 32 位降低到 8 位。

**分页优化器**
优化器状态可能会在训练期间导致显存突然激增。QLoRA 集成了分页优化器，当显存达到上限时，它会自动在 GPU 显存和系统 CPU 内存之间传输数据。这可以防止在剧烈的梯度更新期间出现显存溢出错误。

> QLoRA 中的前向传播架构，其中冻结的 4 位权重被临时反量化，以便与 16 位 LoRA 适配器的输出合并。

标准的 LoRA 前向传播接收输入向量 (vector) $x$，并同时应用基础权重矩阵 $W$ 以及适配器矩阵 $B$ 和 $A$。标准等式如下：

$h = Wx + BAx$

在 QLoRA 中，基础权重 $W$ 被量化为 4 位，表示为 $W_{NF4}$。在前向传播过程中，反量化函数会将 $W_{NF4}$ 转换回更高精度的计算格式（如 bfloat16）来进行数学运算。更新后的等式为：

$h = 	ext{反量化}(W_{NF4})x + BAx$

在这里，$W_{NF4}$ 作为 4 位数据固定在内存中。适配器 $B$ 和 $A$ 保持 16 位浮点格式，以便在反向传播 (backpropagation)期间准确地累积梯度。

在 Python 中实现这一点需要将 bitsandbytes 库与 Hugging Face Transformers 结合使用。在代码中设置量化配置对象时，需要指定四个主要参数：

1. **以 4 位加载**：指示流水线在加载时立即量化基础模型权重。
2. **量化类型**：配置系统使用 NF4 数据类型而不是标准的 4 位浮点数。
3. **计算数据类型**：设置临时反量化步骤的精度，通常为 bfloat16 以匹配适配器。
4. **使用双重量化**：激活缩放常数的二次量化，以最大限度地节省内存。

这种配置可以确保你的训练脚本能够适应标准硬件的内存限制。通过将 4 位基础模型量化与低秩自适应相结合，你可以在严格执行硬件效率的同时，保持参数化语言模型的高性能。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://papers.nips.cc/paper_files/paper/2023/hash/1f898b688304975525501861c8a14b35-Abstract-Conference.html) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 36; Pages: 10088-10115; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍 4-bit NormalFloat (NF4)、双重量化和分页优化器的原始研究论文。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (2022)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  低秩自适应（LoRA）的基础论文，为 QLoRA 提供了架构基础。
- [Quantization (Hugging Face Transformers Documentation)](https://huggingface.co/docs/transformers/main_classes/quantization) — Hugging Face (2024)
  使用 bitsandbytes 和 NF4 数据类型实现 4-bit 量化的官方技术文档。
- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861) — Tim Dettmers, Mike Lewis, Sam Shleifer, Luke Zettlemoyer (2022)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.2110.02861](https://doi.org/10.48550/arXiv.2110.02861)
  确立了 QLoRA 工作流中使用的分块量化和内存高效优化器有效性的研究。

---

[上一节](02-%E4%BD%8E%E7%A7%A9%E8%87%AA%E9%80%82%E5%BA%94%20%28LoRA%29%20%E5%8E%9F%E7%90%86.md) · [下一节](04-%E9%85%8D%E7%BD%AE%E7%9B%AE%E6%A0%87%E6%A8%A1%E5%9D%97%E4%B8%8E%E7%A7%A9.md)
