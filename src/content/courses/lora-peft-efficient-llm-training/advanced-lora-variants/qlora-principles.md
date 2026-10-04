---
course: "lora-peft-efficient-llm-training"
chapter: "advanced-lora-variants"
lesson: "qlora-principles"
sourceId: 4148
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-4-advanced-lora-variants/qlora-principles"
title: "量化LoRA (QLoRA)：原理"
description: "阐述QLoRA背后的原理，包含4比特NormalFloat (NF4) 量化和双重量化。"
order: 3
plots: []
sourceHash: "22627bcbba480d040df57903312c929cb8dc9c1114567c55e7fba0436aaf36f2"
sourceCorrections: []
---

虽然标准LoRA大幅减少了*可训练*参数 (parameter)的数量，但微调 (fine-tuning)大型语言模型仍构成主要的内存难题。主要瓶颈通常不是适配器权重 (weight)本身，而是加载庞大*基础模型*并进行计算所需的内存。即使模型被冻结，基础模型的权重（通常为FP16或BF16等16比特格式）也会占用大量GPU内存。此外，在前向和反向传播 (backpropagation)过程中计算的激活值会显著增加内存占用，这使得在没有高端多GPU配置的情况下，微调亿级参数模型通常不可行。

QLoRA（量化 (quantization)低秩适配）直接解决了这个内存瓶颈。它引入了一种技术，通过大幅减少基础模型的内存占用，而不显著牺牲性能来微调大型语言模型。核心思想是将预训练 (pre-training)的基础模型加载为极低精度（通常为4比特）的量化权重，同时以更高精度格式（如BFloat16）训练LoRA适配器。

### 低比特量化 (quantization)的挑战

简单地将模型量化到4比特然后进行微调 (fine-tuning)，通常会导致性能大幅下降。标准量化方法在如此低的比特宽度下，往往难以保留必要信息。引入的量化误差可能会干扰LoRA旨在进行的微调，阻碍微调过程。QLoRA通过几项专门的改进来克服这一限制，这些改进旨在量化和微调过程中最大程度地保留信息。

### QLoRA的原理

QLoRA的有效性源于三个主要组成部分：4比特NormalFloat (NF4) 量化 (quantization)、双重量化 (DQ) 以及与分页优化器的配合使用（尽管分页优化器将单独介绍）。

#### 1. 4比特NormalFloat (NF4) 量化

QLoRA的核心是NF4数据类型。与标准整数或浮点数量化方案不同，NF4专门为通常围绕零点正态分布的权重 (weight)设计，这是预训练 (pre-training)神经网络 (neural network)权重的一个常见特点。

NF4基于**分位数量化**。核心思想是确定目标分布（例如，标准正态分布 $N(0, 1)$）的分位数，然后根据这些分位数分配量化区间。输入权重首先被归一化 (normalization)（缩放），以适应NF4分位数所覆盖的范围。然后，每个归一化权重被映射到最近的NF4分位数值。

为什么这有效？分位数量化确保了$2^4 = 16$个可能的4比特值中的每一个都代表原始分布中相等比例的权重。这意味着它为大多数权重所在的区域（靠近均值）分配了更多精度（量化层级），而为尾部区域分配了较少精度，使其在信息理论上对于正态分布数据是最优的。NF4通常设计为以零为中心且对称，这与权重分布非常吻合。

与更简单的4比特量化方法相比，这种方法保留了关于原始权重分布的更多信息，这对于在微调 (fine-tuning)过程中保持基础模型的能力至关重要。

#### 2. 双重量化 (DQ)

虽然将权重量化到4比特大幅减少了内存，但量化过程本身会引入一些额外开销。每块权重（例如，64个权重块）通常需要自己的量化常数，这通常是一个以FP32等更高精度格式存储的缩放因子。对于大型模型，这些常数累积起来，会占用不容忽视的内存量（例如，如果对64个权重块使用32比特常数，平均每参数 (parameter)占用0.5比特）。

双重量化 (DQ) 通过对量化常数本身进行量化来进一步减少此开销。此过程包括：

1. 使用NF4将基础模型权重 $W$ 量化到4比特，为每个块生成一级量化常数 $c_1^{FP32}$（例如，缩放因子）。
2. 使用二级低精度量化方案（例如，对常数本身使用块大小为256的8比特浮点数）对这些一级常数 $c_1^{FP32}$ 进行量化，生成二级常数 $c_2^{FP8}$ 和相关元数据。

这第二个量化步骤显著压缩了量化元数据所需的内存。例如，使用块大小为256的8比特量化常数，将开销从大约每参数0.5比特减少到大约 $32 / (64 * (256/8)) + 8/64 \approx 0.14$ 比特/参数，从而实现了进一步的内存节省，而不明显影响模型性能。

#### 3. 微调期间的运行流程

这些组件在QLoRA微调过程中配合运行方式如下：

1. **加载与量化：** 预训练的基础模型 $W$ 被加载，其权重立即被量化为NF4格式，即 $W^{NF4}$。双重量化技术用于压缩相关量化常数。原始高精度权重被丢弃，释放大量GPU内存。$W^{NF4}$ 权重被冻结，训练期间不会更新。
2. **初始化适配器：** LoRA适配器（$A$ 和 $B$ 矩阵）被添加到目标层（例如，注意力层），并以更高精度格式（通常为BFloat16 (BF16)）初始化。这些适配器权重是*唯一*将进行训练的参数。
3. **前向传播：** 处理输入 $x$ 时：
   - 输入通过冻结的 $W^{NF4}$ 层传播。
   - 对于包含LoRA适配器的层，4比特基础模型权重 $W^{NF4}$ 的必要块会*即时*反量化回计算精度 (BF16)。
   - 输出计算为基础模型输出和LoRA适配器输出的总和：
     
     $$
     y = \text{反量化}(W^{NF4})x + \alpha \cdot x B A
     $$
     
     其中 $\text{反量化}(W^{NF4})$ 表示反量化的基础模型权重块，$B$ 和 $A$ 是BF16格式的LoRA矩阵，$\alpha$ 是LoRA缩放因子。矩阵乘法和加法运算在BF16精度下进行。
4. **反向传播 (backpropagation)：** 计算梯度，但它们仅流经适配器权重 $A$ 和 $B$。梯度计算通常在BF16中进行。由于 $W^{NF4}$ 被冻结，不会为基础模型计算或存储梯度，从而节省大量内存和计算资源。
5. **优化器更新步：** 优化器（通常是AdamW，可能是与分页优化器一起使用的8比特AdamW等内存高效变体）仅根据计算出的梯度更新LoRA适配器权重 $A$ 和 $B$。

> QLoRA层在前向传播期间的计算流程。基础模型权重保持量化状态，直到需要计算时，而仅训练LoRA适配器权重。

### 优势与影响

QLoRA的主要优势在于GPU内存使用量的显著减少。通过将最大的组成部分（基础模型）以4比特精度存储，QLoRA使得在VRAM有限的硬件上微调 (fine-tuning)以前无法访问的模型成为可能。例如，一个650亿参数 (parameter)的模型，仅FP16权重 (weight)就可能需要超过130GB，而使用4比特量化 (quantization)（加上额外开销）大约33GB即可加载。这使得在具有48GB甚至24GB VRAM的单GPU上微调此类模型成为可行，从而普及了大型模型微调的访问。

重要的是，由于NF4和双重量化的有效性，QLoRA在实现内存减少的同时，在许多基准测试上保持了与16比特LoRA甚至16比特全量微调非常接近的性能水平。效率和性能的这种结合使QLoRA成为参数高效微调领域广泛采用的技术。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  这篇基础论文介绍了 QLoRA，详细阐述了 4 位 NormalFloat (NF4) 量化、双重量化 (DQ) 和分页优化器，用于大型语言模型内存高效的微调。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了原始的低秩适应（LoRA）方法，这是一种参数高效的微调技术，是 QLoRA 基于适配器训练的基础。
