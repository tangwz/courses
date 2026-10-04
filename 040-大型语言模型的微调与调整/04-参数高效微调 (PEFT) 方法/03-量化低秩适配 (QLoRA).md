# 量化低秩适配 (QLoRA)

来源：[原文](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning/qlora-introduction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

低秩适配 (LoRA) 是一种与完全微调 (fine-tuning)相比能大幅降低可训练参数 (parameter)数量的方法。然而，即使使用了LoRA，加载和微调大型语言模型所需的内存占用仍然可能很大。基础模型的权重 (weight)通常以16位精度（如float16或bfloat16）存储，会占用大量GPU显存 (VRAM)。量化 (quantization)低秩适配 (QLoRA) 通过将LoRA的参数效率与基础模型的激进量化相结合，直接解决了这一内存瓶颈。

QLoRA 的主要思路是将预训练 (pre-training)的基础大型语言模型以量化格式（通常是4位）加载，从而大幅减少其内存需求。重要的是，虽然基础模型以这种低精度格式保存，LoRA适配器本身却以更高精度（例如bfloat16）进行训练。这种方法可以在不导致性能严重下降的情况下，实现大量内存节省，使得在VRAM有限的硬件（例如单块消费级GPU）上微调大型模型（例如650亿参数）成为可能。

### QLoRA 的核心技术

QLoRA 通过若干相互配合的技术实现了其卓越的效率：

1. **4位NormalFloat (NF4) 量化 (quantization)：** 这是随QLoRA引入的一项重要改进。与标准整数量化方法不同，NF4 专门为遵循以零为中心的正态分布数据而设计，这是预训练 (pre-training)后神经网络 (neural network)权重 (weight)的一个常见属性。NF4 是一种理论上对正态分布数据最优的量化方案。它采用分位数量化，意味着每个量化区间都表示目标正态分布中相等预期数量的值。这使得 NF4 能够比标准4位整数 (Int4) 量化更精确地表示原始权重分布，从而更有效地维持模型性能。基础模型权重在加载时被转换为NF4格式，并在整个微调 (fine-tuning)过程中保持该格式不变。
2. **双重量化 (DQ)：** 为了进一步节省内存，QLoRA 采用了双重量化。这项技术进一步压缩了量化元数据本身。标准量化需要为每个权重块保存量化常数（如缩放因子或零点）。DQ 对这些常数进行量化，通常使用8位浮点格式，块大小为256。这个次级量化步骤只增加了少量开销，但平均每个参数 (parameter)可以节省大约0.4到0.5位，这对于大型模型而言节省效果会很可观。
3. **分页优化器：** 微调，特别是使用梯度检查点等方法时，可能造成内存使用量突然飙升，尤其是在优化器状态（例如，存储动量和方差值的Adam优化器状态）方面。即使平均内存使用量可控，这些飙升也可能导致内存不足（OOM）错误。QLoRA 使用 NVIDIA 的统一内存功能来实现分页优化器。这项方法会在 GPU 显存 (VRAM)和 CPU 内存之间自动分页优化器状态，类似于操作系统在 RAM 和磁盘之间管理内存分页的方式。当 GPU 在潜在的内存峰值期间耗尽内存时，优化器状态会被移动到 CPU 内存，并在需要时被重新加载回 GPU。这避免了 OOM 错误，并确保了比原本能适应GPU内存的更大规模的模型也能稳定训练。

### QLoRA 微调 (fine-tuning)流程

以下是 QLoRA 微调的流程概述：

1. **加载与量化 (quantization)：** 加载预训练 (pre-training)的LLM，立即将其参数 (parameter)量化为4位NF4精度。如果启用，对量化常数应用双重量化。这些量化后的权重 (weight)是冻结的。
2. **注入适配器：** 将 LoRA 适配器（可训练的低秩矩阵 $A$ 和 $B$）插入到指定层（通常是注意力机制 (attention mechanism)的线性变换层）。这些适配器以更高精度（例如 bfloat16）保存。
3. **前向传播：** 在前向传播过程中，对于涉及原始权重的计算，所需的4位权重块会被即时反量化到计算数据类型（例如 bfloat16）。然后执行标准的前向计算，并结合高精度 LoRA 适配器的输出：
   
   $$
   h = \text{反量化}(W_{NF4})x + BAx
   $$
   
   此处，$W_{NF4}$ 代表冻结的4位基础模型权重，$B$ 和 $A$ 是可训练的 bfloat16 LoRA 适配器权重。
4. **反向传播 (backpropagation)与梯度计算：** 根据模型的输出计算损失。重要的是，梯度仅通过 LoRA 适配器权重 ($A$ 和 $B$) 进行计算和反向传播。基础模型权重保持冻结，因此不计算或存储它们的梯度。适配器的梯度以计算精度（例如 bfloat16）进行计算和存储。
5. **参数更新：** 仅使用计算出的梯度更新 LoRA 适配器权重 ($A$ 和 $B$)。如果使用分页优化器，此步骤中的任何内存压力都通过将优化器状态分页到 CPU 内存来处理。

### 优势与权衡

**优势：**

- **内存大幅减少：** QLoRA 大幅降低了微调 (fine-tuning)所需的显存 (VRAM)，使得可以在易于获取的硬件（例如具有24GB或48GB显存的GPU）上调整规模很大的模型（300亿、650亿+参数 (parameter)）。
- **性能得以保持：** 尽管基础模型进行了4位量化 (quantization)，QLoRA 在许多基准测试和任务上通常能达到与16位完全微调或16位LoRA非常接近的性能。NF4 的使用和 bfloat16 的计算对于这一点非常重要。
- **兼容性好：** 它与 Hugging Face 的 `transformers` 和 `peft` 库等现有生态系统良好兼容，通常只需更改配置即可。



[交互图表：大约内存使用对比（以650亿参数模型为例）](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning/qlora-introduction#plot-ngvhar)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "大约内存使用对比（以650亿参数模型为例）"
    },
    "xaxis": {
      "title": {
        "text": "微调方法"
      }
    },
    "yaxis": {
      "title": {
        "text": "预估GPU显存 (GB)"
      },
      "range": [
        0,
        85
      ]
    },
    "bargap": 0.3
  },
  "data": [
    {
      "type": "bar",
      "x": [
        "完全微调（16位）",
        "LoRA（16位基础）",
        "QLoRA（4位基础）"
      ],
      "y": [
        80,
        50,
        22
      ],
      "marker": {
        "color": [
          "#fa5252",
          "#5c7cfa",
          "#40c057"
        ]
      }
    }
  ]
}
```

</details>



> 使用不同方法微调650亿参数LLM的预估GPU显存需求。QLoRA 大幅降低了内存占用。实际用量可能因序列长度、批次大小和具体实现方式而异。

**权衡：**

- **潜在吞吐量 (throughput)降低：** 前向传播过程中的即时反量化会带来计算开销。虽然 QLoRA 节省内存，但如果内存并非主要限制，与标准 LoRA 相比，它可能会造成训练步骤变慢（吞吐量降低）。
- **轻微性能差异：** 尽管通常能达到相似性能，但在特定任务上，与16位方法相比，某些衡量标准可能会出现少量下降。
- **实现依赖：** 需要依赖 `bitsandbytes` 等特定库来实现 NF4 量化、DQ 和分页优化器。

### 实现说明

在使用 Hugging Face `peft` 等库时，启用 QLoRA 通常需要设置一些参数 (parameter)，包括模型加载和 `LoraConfig` 设置。常用参数包括：

- `load_in_4bit=True`：指示 `transformers` 库通过 `bitsandbytes` 使用4位量化 (quantization)加载基础模型。
- `bnb_4bit_quant_type="nf4"`：指定4位量化类型。NF4 通常推荐使用。“fp4”（标准4位浮点）是另一个选项，但通常效果较差。
- `bnb_4bit_use_double_quant=True`：启用双重量化功能以进一步节省内存。
- `bnb_4bit_compute_dtype=torch.bfloat16`：设置用于反量化权重 (weight)计算和 LoRA 适配器的数据类型。在现代硬件上，`bfloat16` 因其在范围和精度上的良好兼顾而更受青睐，这对于保持性能有很大帮助。

QLoRA 是一项重要进展，它使得大规模模型调整变得更易实现。通过巧妙地将量化与 LoRA 的有针对性更新结合起来，它突破了现有硬件能力的限制，使得定制先进大型语言模型的能力得以普及。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍QLoRA的原创论文，详细阐述了4位NormalFloat（NF4）量化、双重量化和分页优化器，以实现高效的LLM微调。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA）方法，这是QLoRA的基础，展示了大型语言模型参数高效的适应技术。
- [bitsandbytes GitHub Repository](https://github.com/TimDettmers/bitsandbytes) — bitsandbytes-foundation (2024)
  bitsandbytes的官方软件仓库，该库提供优化的CUDA函数，用于8位和4位量化，包括QLoRA中使用的NF4、双重量化和分页优化器。
- [Parameter-Efficient Fine-tuning (PEFT) library documentation](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  Hugging Face PEFT库的官方文档，提供了实现如QLoRA等参数高效微调方法的指南和API参考。

---

[上一节](02-%E4%BD%8E%E7%A7%A9%E9%80%82%E9%85%8D%20%28LoRA%29.md) · [下一节](04-%E9%80%82%E9%85%8D%E5%99%A8%E6%A8%A1%E5%9D%97.md)
