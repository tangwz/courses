# 量化LoRA (QLoRA)

来源：[原文](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-5-parameter-efficient-fine-tuning-peft/quantized-lora-qlora)

[返回章节目录](README.md) · [返回课程目录](../README.md)

低秩适配 (LoRA) 相较于完全微调 (fine-tuning)，能够显著减少大型语言模型微调所需的*可训练*参数 (parameter)数量。尽管如此，一个主要的内存瓶颈依然存在：大型基础模型的全精度权重 (weight)。尽管在LoRA训练期间这些权重是冻结的，但它们必须驻留在GPU内存中，以便与LoRA适配器的输出结合。对于非常大的模型，这种静态内存需求可能会非常高，轻易超出即使高端加速器的容量。例如，以BFloat16 (BF16) 精度加载一个拥有700亿参数的模型，仅权重就需要超过140GB的GPU内存，这限制了微调的可用性。

量化 (quantization)低秩适配 (QLoRA) 通过将LoRA与基础模型权重的激进量化相结合，直接解决了这一内存问题。这种方法显著降低了微调超大型模型所需的内存占用，使得在VRAM显著较少的硬件（包括某些情况下的单块消费级GPU）上适配这些模型成为可能。

## QLoRA工作原理

QLoRA的基本策略是加载预训练 (pre-training)的基础模型，其参数 (parameter)被量化 (quantization)为非常低的精度格式，通常是4位。这些量化权重 (weight)，占模型参数的绝大多数，在整个微调 (fine-tuning)过程中保持冻结。之后，将小得多的LoRA适配器权重（低秩矩阵$A$和$B$）添加到模型架构中，通常在注意力或前馈模块内部，并使用标准的高精度格式（如BFloat16）进行训练。

在前向传播过程中，操作需要将冻结的基础模型层的输出与可训练LoRA适配器生成的输出结合。设$W$表示某层的原始权重，$x$为输入激活。在标准LoRA中，计算过程为：


$$
h = Wx + \Delta W x = Wx + BAx
$$


在QLoRA中，基础权重$W$不以全精度存储。它们以量化格式$W^{q}$（例如，4位）存储。为了执行前向传播，当$W^{q}$的对应块或部分在与$x$进行矩阵乘法之前，会被动态地反量化回计算精度（例如BF16）。然后，这个反量化后的权重用于计算，与LoRA适配器的输出($BAx$)结合，并且反量化版本在该层使用后通常可以立即丢弃，从而最大程度地减少峰值内存使用。


$$
h = \text{反量化}(W^{q})x + BAx
$$


重要的方面是，整套基础模型参数($W^q$)以低位格式驻留在GPU内存中，从而带来了显著的内存节省。实际的训练（包括梯度计算和权重更新）仅针对小部分的LoRA参数（$A$和$B$）进行，这些参数保持在较高精度。

## QLoRA实现方法

QLoRA的有效性依赖于引入的几个相互关联的组件，以最大程度地节省内存，同时最小化对模型性能的影响：

### 4位NormalFloat (NF4) 量化 (quantization)

标准量化技术通常假设值呈均匀分布，这与预训练 (pre-training)神经网络 (neural network)中权重 (weight)的典型分布（通常以零为中心，呈钟形）不符。QLoRA引入了4位NormalFloat (NF4) 数据类型。NF4专门设计为遵循标准正态分布($N(0, 1)$)的数据，其信息理论上最佳的量化格式。

它通过分位数量化实现这一点。NF4不是使用均匀间隔的量化级别，而是根据$N(0, 1)$分布的分位数定义级别。这确保了如果输入数据呈正态分布，每个量化 bin 都有相同数量的预期值分配给它。在应用NF4之前，量化块内的权重（例如，一组64个权重）会被归一化 (normalization)为零均值和单位方差。这种定制化方法与应用于LLM权重的标准4位整数或浮点量化方案相比，保留了更多信息。

### 双重量化 (quantization) (DQ)

量化张量不仅需要存储量化值，还需要存储量化参数 (parameter)（元数据），例如用于将原始范围映射到量化范围的缩放因子或零点。对于使用块级量化（每个权重 (weight)块使用独立的量化参数）的大型模型，这些参数合起来可能会占用相当可观的内存量（例如，对于大型LLM，可达数GB）。

双重量化 (DQ) 通过对*量化参数本身*应用第二层量化来解决此问题。例如，如果第一步量化对每个块使用32位浮点缩放因子，DQ可能会使用8位浮点格式对这些32位浮点数进行量化，第二次量化可能还会使用自己的块大小。这种方法进一步压缩了模型的内存占用，平均每个原始模型参数可额外节省约0.4-0.5位。

### 分页优化器

微调 (fine-tuning)需要为每个可训练参数 (parameter)维护优化器状态（例如，Adam/AdamW中的动量和方差向量 (vector)）。尽管LoRA显著减少了可训练参数的*数量*，但内存使用量仍可能出现不可预测的峰值，特别是在使用梯度检查点等技术时，这些技术在前向传播期间临时存储激活以供反向传播 (backpropagation)期间重新计算。这些峰值可能导致内存不足 (OOM) 错误，从而中断训练过程。

QLoRA使用分页优化器，它们利用NVIDIA的统一虚拟内存系统。这使得GPU驱动程序能够自动管理内存页，根据需要将其在GPU显存 (VRAM)和锁定CPU内存之间传输。如果GPU在内存峰值期间尝试为优化器状态或激活分配空间时内存不足，最近最少使用的页面将被移至CPU内存，从而防止OOM错误并允许训练继续。这使得微调过程更能应对临时内存峰值。

## 优点与性能考量

QLoRA最显著的优势是大幅减少了微调 (fine-tuning)大型模型所需的GPU内存。通过将基础模型权重 (weight)存储为4位NF4并采用双重量化 (quantization)，静态内存成本显著降低（相较于16位精度大约减少了4倍）。结合分页优化器处理动态内存峰值，QLoRA使得在具有24GB或48GB显存 (VRAM)的单个GPU上微调拥有数百亿参数 (parameter)的模型（如Llama 2 70B）成为可能，而这项任务此前需要更大规模的多GPU设置。



[交互图表：近似微调内存使用比较](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-5-parameter-efficient-fine-tuning-peft/quantized-lora-qlora#plot-1oc4bw1)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "近似微调内存使用比较",
    "xaxis": {
      "title": "微调方法"
    },
    "yaxis": {
      "title": "相对内存 (%)",
      "range": [
        0,
        100
      ]
    },
    "legend": {
      "traceorder": "reversed",
      "orientation": "h",
      "yanchor": "bottom",
      "y": -0.3,
      "xanchor": "center",
      "x": 0.5
    },
    "barmode": "stack",
    "height": 450,
    "width": 600,
    "template": "plotly_white",
    "shapes": [
      {
        "type": "rect",
        "xref": "paper",
        "yref": "paper",
        "x0": 0,
        "y0": 0,
        "x1": 1,
        "y1": 1,
        "line": {
          "width": 1,
          "color": "#dee2e6"
        }
      }
    ]
  },
  "data": [
    {
      "type": "bar",
      "name": "基础模型权重",
      "x": [
        "Full FT (BF16)",
        "LoRA (BF16)",
        "QLoRA (NF4)"
      ],
      "y": [
        65,
        65,
        16.25
      ],
      "marker": {
        "color": "#74c0fc"
      },
      "text": [
        "65%",
        "65%",
        "16%"
      ],
      "textposition": "inside"
    },
    {
      "type": "bar",
      "name": "优化器与梯度",
      "x": [
        "Full FT (BF16)",
        "LoRA (BF16)",
        "QLoRA (NF4)"
      ],
      "y": [
        25,
        2,
        2
      ],
      "marker": {
        "color": "#f06595"
      },
      "text": [
        "25%",
        "2%",
        "2%"
      ],
      "textposition": "inside"
    },
    {
      "type": "bar",
      "name": "激活值与其他",
      "x": [
        "Full FT (BF16)",
        "LoRA (BF16)",
        "QLoRA (NF4)"
      ],
      "y": [
        10,
        10,
        10
      ],
      "marker": {
        "color": "#96f2d7"
      },
      "text": [
        "10%",
        "10%",
        "10%"
      ],
      "textposition": "inside"
    }
  ]
}
```

</details>



> 微调期间内存使用情况的示意性分解。QLoRA大幅减少了基础模型权重占用的内存，使其成为三者中最节省内存的选项。请注意，“激活值与其他”部分的变化性很大。

值得注意的是，研究表明QLoRA微调通常能达到与使用全精度（例如BF16）基础模型的LoRA微调几乎相同的性能水平。这表明NF4量化、仅在高精度下训练适配器以及大型预训练 (pre-training)模型固有的鲁棒性的结合，使得QLoRA即使在对基础权重进行激进的4位压缩后也能保持高保真度。

然而，仍有一些实际考量：

- **计算开销：** 前向传播过程中基础模型权重的动态反量化，相较于使用已存储在计算精度（例如BF16）中的权重，会引入计算开销。尽管`bitsandbytes`库等提供的高度优化核函数可以缓解这一点，但如果内存不是主要限制，推理 (inference)或训练速度可能略低于类似LoRA设置。主要优势在于*使得*在内存受限的硬件上进行微调成为可能。
- **潜在的轻微保真度损失：** 尽管实验结果表现良好，但量化到4位等极低位宽本质上是有损的。对于对模型输出微小变化高度敏感的任务，建议将QLoRA与更高精度的微调方法进行对比评估，以确保满足性能要求。

QLoRA代表了一项强大的进展，它使最先进LLM的适配变得更易于使用且更具成本效益。通过巧妙结合参数高效微调与专门的量化技术，它扩展了在广泛可用硬件资源上可实现的目标。Hugging Face（`transformers`、`peft`、`bitsandbytes`）等流行生态系统中的工具提供了实用实现，使工程师和研究人员能够有效地应用QLoRA。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  本文介绍了QLoRA，详细阐述了4比特NormalFloat（NF4）量化、双重量化（DQ）和分页优化器。它为本节提供了技术基础。
- [bitsandbytes: 8-bit Optimizers and Quantization Techniques for PyTorch](https://github.com/TimDettmers/bitsandbytes) — Tim Dettmers (2023)
  bitsandbytes库的GitHub仓库，该库实现了4比特NormalFloat和其他对QLoRA至关重要的量化技术的优化CUDA内核。
- [Parameter-Efficient Fine-Tuning (PEFT) library](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，该库提供了用于应用LoRA、QLoRA和其他参数高效微调方法的高级API。

---

[上一节](04-%E4%BD%8E%E7%A7%A9%E9%80%82%E5%BA%94%EF%BC%88LoRA%EF%BC%89.md) · [下一节](06-%E7%BB%84%E5%90%88PEFT%E6%96%B9%E6%B3%95.md)
