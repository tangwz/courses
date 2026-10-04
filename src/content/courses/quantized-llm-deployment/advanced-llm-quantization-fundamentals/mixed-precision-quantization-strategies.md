---
course: "quantized-llm-deployment"
chapter: "advanced-llm-quantization-fundamentals"
lesson: "mixed-precision-quantization-strategies"
sourceId: 4605
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-1-advanced-llm-quantization-fundamentals/mixed-precision-quantization-strategies"
title: "混合精度量化方法"
description: "将不同精度应用于大型语言模型各部分以实现最佳平衡的技术。"
order: 6
plots: ["plots/4605-0.json"]
sourceHash: "44b3f2f085386a16037d1f3a4b5e045c83d35063a54639bfebe641961362605e"
sourceCorrections: []
---

统一地将单一低比特量化 (quantization)格式（如INT4或INT8）应用于整个大型语言模型，虽然能提供理论上最大的压缩比和潜在的加速效果，但这通常会导致不可接受的精度下降。大型语言模型在不同层和参数 (parameter)类型上对量化噪声表现出不同的敏感度。某些部分，如注意力机制 (attention mechanism)或特定的前馈网络层，可能对激进量化具有较好的容忍度，而其他部分，例如嵌入 (embedding)层或最终输出层，可能需要更高的精度以保持模型准确性。这种现象促使人们采用混合精度量化方法。

其主要思想很简单：对模型的不同部分应用不同的数值精度（比特宽度和格式），以在效率（速度、内存）和精度之间取得更好的平衡。而不是采用一刀切的方式，我们根据每个组件的敏感度来调整量化级别。

### 识别敏感度

在应用混合精度之前，我们需要一种方式来确定模型哪些部分对量化 (quantization)误差敏感。常见的方法有：

1. **经验分析：** 这涉及到系统地逐个（或分组）量化不同的层或模块，同时将其他部分保持在更高的精度（例如FP16或FP32）。针对每种配置，会测量其对相关指标（例如校准集上的困惑度或下游任务的准确性）的影响。量化后导致显著下降的层会被标记 (token)为敏感。这种方法虽然直接，但计算量可能很大。
2. **基于Hessian的分析：** 诸如最佳大脑量化（OBQ）及其后续方法（如GPTQ）中使用的技术，能够近似量化引入的二阶误差。Hessian矩阵（或其近似值）提供了损失函数 (loss function)相对于权重 (weight)的曲率信息。曲率越高，敏感度越大。与较大Hessian特征值相关的参数 (parameter)或层通常对扰动（包括量化噪声）更敏感。这种分析可以指导精度分配，为更敏感的权重建议更高的精度。
3. **激活值分析：** 观察激活值的分布和范围也能提供线索。激活值表现出大离群值或宽动态范围的层，可能更容易受到低比特量化引入的裁剪和舍入误差的影响，并可能从更高精度或专门的量化方案中获益。

### 常见的混合精度方法

一旦敏感度得到理解（或基于常见启发式规则），可以采用以下几种方法：

- **基于层类型的精度分配：** 一种常见的启发式方法是根据层类型分配精度。例如：
  - **嵌入 (embedding)层和输出层：** 通常保持较高精度（例如FP16或BFloat16），因为这里的误差会显著影响后续计算或最终预测。
  - **注意力层（QKV投影、输出投影）：** 敏感度各不相同。有时这些层可以进行激进量化 (quantization)（例如INT4/INT8），但其影响需要仔细评估。
  - **前馈网络（FFN）：** 通常是模型中参数 (parameter)和计算量最大的部分。它们是激进量化（例如INT4）的常见目标。
- **敏感度导向的精度分配：** 根据前述分析，被识别为高度敏感的层被分配更高的精度（例如INT8或FP16），而不太敏感的层则进行更激进的量化（例如INT4）。这提供了一种比简单层类型规则更量身定制的方法。
- **混合格式：** 除了混合比特宽度（如INT8和INT4）外，我们还可以混合数值格式。例如，对敏感层使用FP16，对中等敏感层使用INT8，对最具弹性的层使用INT4甚至NF4/FP4等格式。这需要硬件和内核对所选格式的支持。
- **组件特定量化：** 超越层的层面，可以在单个层内应用不同的方案。例如，在注意力机制 (attention mechanism)中，查询（Q）和键（K）投影的量化方式可能与值（V）投影或输出投影不同，这取决于经验结果或理论考量。

### 权衡与考量

混合精度引入了复杂性，但也提供了显著的灵活性。主要目标是比统一量化 (quantization)更有效地平衡模型大小/性能与准确性之间的取舍。



![精度与模型大小的权衡](plots/4605-0.json)



> 模型大小缩减与精度下降（以困惑度衡量）在不同量化方法下的权衡。与统一低比特方法相比，混合精度旨在获得这条曲线上更好的表现点。

考量因素包括：

- **复杂性：** 设计和实现混合精度方法比统一量化需要更多的工作。敏感度分析、配置管理和调试变得更加复杂。
- **硬件支持：** 性能优势在很大程度上取决于目标硬件对所选精度和混合精度操作高效内核的原生支持。在现代GPU上，混合INT8和INT4可能得到良好支持，而混合更不常见的格式可能需要特定的库或硬件加速器。
- **框架兼容性：** 部署框架（如TensorRT-LLM、vLLM、ONNX Runtime）需要高效支持所选的混合精度配置。这通常涉及能够平稳处理不同数据类型之间转换的专用内核。
- **寻找最佳配置：** 确定*最佳*混合精度配置通常是一个迭代过程，需要针对目标硬件和任务进行实验和基准测试。

混合精度量化代表了一种实用的LLM优化方法。通过认识到模型组件敏感度的异质性，它使得实践者能够获得显著的效率提升，同时谨慎地管理对模型准确性的影响，超越了统一量化方案的局限性。这种灵活性在目标是INT4或更低比特宽度时尤为重要。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了4位NormalFloat (NF4) 和一种高效的量化LLM微调方法，这对高效部署至关重要，并与混合不同数字格式相关。
- [A Survey on Quantization of Neural Networks](https://ieeexplore.ieee.org/document/9121633) — Jiya Ren, Jingjing Li, Hongliang Li, Gang Li, Meng Wang, and Jiancheng Lv (2020)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 43; Pages: 1802-1823; DOI: [10.1109/TPAMI.2020.3006096](https://doi.org/10.1109/TPAMI.2020.3006096)
  全面概述了各种神经网络量化技术，包括混合精度，为该主题提供了广泛的背景信息。
