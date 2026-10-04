---
course: "mixture-of-experts-advanced-implementation"
chapter: "efficient-moe-inference"
lesson: "moe-quantization-techniques"
sourceId: 7070
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference/moe-quantization-techniques"
title: "MoE层量化技术"
description: "应用量化方法以减少路由网络和专家网络的内存占用并加速计算。"
order: 5
plots: []
sourceHash: "2256bb1fc976f84becf07b763ed8030429e20de24818e56eda9d27e7bdd57ade"
sourceCorrections: []
---

量化 (quantization)旨在解决如何将大型MoE模型放入有限GPU内存的难题，通过缩小模型权重 (weight)来处理此问题。这是一种降低模型参数 (parameter)以及在某些情况下激活值数值精度的一系列方法。通过使用较少的比特来表示权重（例如，8位整数（INT8）或4位整数（INT4），而非32位或16位浮点数），这项技术可以显著减少模型的内存占用，并通常在支持的硬件上加速计算。

然而，对MoE架构应用量化并非像对整个模型统一降低精度那么简单。其特有结构，包括敏感的路由网络和独立的专家，需要更细致的方法来保持模型性能。

### MoE中的组件特定量化 (quantization)

MoE有效量化的核心思路是根据组件对数值误差的敏感程度，对其应用不同的精度级别。一个MoE层可以分为两个主要部分：路由网络和专家池。

#### 量化专家

专家，通常是前馈网络（FFNs），包含MoE模型参数 (parameter)的绝大部分。例如，像Mixtral-8x7B这样的模型，其每层八个专家中包含了其470亿有效参数的大部分。这使得它们成为激进量化的主要对象。

由于每个专家都作为一个标准FFN运作，我们可以对每个专家应用GPTQ或AWQ等现代训练后量化（PTQ）技术。这些方法有效，因为它们通过分析权重 (weight)分布和激活模式来减少量化引入的误差。

专家常用的量化格式有：

- **INT8：** 相比FP16，权重内存减少2倍。它在压缩和精度保持之间提供了良好平衡。
- **INT4：** 相比FP16，内存减少4倍。这是一种非常激进的量化形式，如果操作不慎，有时会导致明显的质量下降。QLoRA方法中使用的NormalFloat 4 (NF4) 等技术，旨在更好地处理神经网络 (neural network)权重的分布。

该过程涉及量化每个专家FFN的权重，同时通常将激活值保留在BF16或FP16等更高精度的格式中。在前向传播过程中，INT4/INT8权重在矩阵乘法之前即时进行反量化。此操作计算效率高，因为较小的量化权重从VRAM到GPU处理单元的内存传输速度快得多。

> 不同量化方案下专家权重相对内存占用图示。从16位浮点数转向4位整数，存储需求减少4倍。

#### 保护路由网络

路由网络，或称路由器，情况则不同。它的作用是计算对数几率，从而决定由哪些专家处理每个令牌。路由决策是一种离散的、影响重大的操作。路由器输出对数几率中的微小扰动，可能由量化噪声引起，都可能导致令牌被发送到完全不同且不合适的专家。相比单个专家FFN内部的微小精度误差，这可能对最终输出产生大得多的负面影响。

由于这种敏感性，行业通用做法是**避免量化路由网络**。路由器的参数只占模型总大小的一小部分，因此将其保持在FP16或BFloat16等更高精度的格式中，内存开销可忽略不计，但却能为路由机制提供显著的稳定性。

### 混合精度架构

因此，最有效的方法是混合精度方案。您可以在收益最大的地方（专家）应用激进量化 (quantization)，并在最需要的地方（路由器）保持高精度。

推理 (inference)时MoE Transformer层的典型混合精度配置如下：

1. **输入与自注意力 (self-attention)：** 激活值和权重 (weight)采用16位格式（例如，BF16）。
2. **路由网络：** 路由器使用BF16/FP16权重和激活值计算其对数几率。
3. **专家选择：** 前k个令牌被路由到其指定的专家。
4. **专家计算：** 选定的专家被激活。其FFN权重以INT4/INT8加载，但会在与BF16激活张量进行矩阵乘法之前反量化为BF16。
5. **输出：** 专家输出被组合，仍为BF16格式。

此策略将低精度表示限制在内存中“静止”的参数 (parameter)，从而获得内存节省，同时以更稳定的16位格式进行实际计算。

> 混合精度MoE层内的流程。路由网络以高精度运行以确保路由稳定，而大型专家网络使用量化权重以节省内存。

### Transformers中的实际应用

现代库如Hugging Face `transformers`，结合 `bitsandbytes`，使得实现这种混合精度策略变得简单。您可以指定一个全局量化 (quantization)配置，该配置将应用于模型专家中的线性层，而框架足够智能，通常会排除敏感组件，如模型的头部或在此例中的路由器。

以下是如何在加载模型时配置4位量化方案的示例。

```python
from transformers import BitsAndBytesConfig
import torch

# 定义4位量化配置
# 这将应用于每个专家内部的FFN
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载模型时，传入此配置。
# 库会自动将torch.nn.Linear层
# 替换为根据配置量化后的版本。
# 对于MoE模型，这主要针对专家。
# model = AutoModelForCausalLM.from_pretrained(
#     "mistralai/Mixtral-8x7B-v0.1",
#     quantization_config=quantization_config,
#     device_map="auto"
# )
```

在此配置中：

- `load_in_4bit=True`：启用4位加载。
- `bnb_4bit_quant_type="nf4"`：指定“NormalFloat4”数据类型，该类型针对神经网络 (neural network)中常态分布的权重 (weight)进行了优化。
- `bnb_4bit_use_double_quant=True`：一种对量化常数本身进行二次量化的技术，每个参数 (parameter)可额外节省0.4位。
- `bnb_4bit_compute_dtype=torch.bfloat16`：设置计算数据类型。INT4权重将在矩阵乘法之前反量化为BFloat16。

通过审慎应用量化，您可以实现显著的内存节省，从而可以在消费级或单服务器GPU上运行超大型MoE模型。这种技术与专家卸载和推测解码等其他技术结合使用，为使稀疏模型在生产推理 (inference)中实用提供了强大的方法集。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: arXiv preprint arXiv:2210.17323; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍了GPTQ，一种用于大型语言模型的训练后量化方法，与专家量化相关。
- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了NF4数据类型和4比特量化，实现了大型语言模型的高效微调。
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://doi.org/10.48550/arXiv.2208.07339) — Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer (2022)
  Journal: NeurIPS 2022; DOI: [10.48550/arXiv.2208.07339](https://doi.org/10.48550/arXiv.2208.07339)
  描述了大型语言模型中`bitsandbytes`库使用的8比特矩阵乘法和量化技术。
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: MLSys 2024; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  提出了AWQ，一种激活感知权重 量化方法，通过保护重要权重来保持模型性能。
- [Quantization (bitsandbytes)](https://huggingface.co/docs/transformers/quantization) — Hugging Face (2024)
  Publisher: Hugging Face documentation
  官方文档，解释了如何在Hugging Face Transformers库中通过`bitsandbytes`应用量化。
