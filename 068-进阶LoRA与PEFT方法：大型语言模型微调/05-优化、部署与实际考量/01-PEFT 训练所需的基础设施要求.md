# PEFT 训练所需的基础设施要求

来源：[原文](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-5-peft-optimization-deployment/peft-infrastructure-requirements)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然参数 (parameter)高效微调 (fine-tuning)相较于全量微调显著降低了门槛，但使用 LoRA 或 QLoRA 等方法成功训练模型，仍需仔细考量你的基础设施。了解这些要求对规划实验、选择硬件以及确保训练顺利进行来说具有基础性作用。这里将分析必要的硬件和软件先决条件。

### 硬件要求

PEFT 的主要硬件瓶颈围绕图形处理器 (GPU)，特别是它们的内存容量和计算能力。

#### GPU 内存 (VRAM)

这通常是最主要的限制。尽管 PEFT 技术显著减少了*可训练*参数 (parameter)的数量，但整个基础模型的权重 (weight)仍需驻留在 GPU 内存中，以进行前向和反向传播 (backpropagation)。此外，激活值、适配器参数的梯度和优化器状态还会占用额外的显存 (VRAM)。

- **基础模型大小：** 最大的内存消耗者是预训练 (pre-training)模型本身。一个使用标准 `float32` 精度的 70 亿参数模型大约需要 28GB (70 亿参数 \* 4 字节/参数)。在 `bfloat16` 或 `float16` 中，这个需求减半到约 14GB。加载更大的模型（130 亿、300 亿、700 亿以上）会按比例增加这项基本要求。
- **量化 (quantization) (QLoRA)：** QLoRA 等技术，以 4 位精度加载基础模型（使用 `bitsandbytes` 等库），能显著减少基础内存占用。一个 70 亿参数的 4 位模型可能只需约 4-5GB 的权重内存，这使得在消费级 GPU 上微调 (fine-tuning)大型模型成为可能。
- **批量大小和序列长度：** 更大的批量大小和更长的输入序列会增加前向传播期间存储激活值所需的内存，这些激活值在反向传播中计算梯度时会用到。
- **LoRA 秩 (`r`)：** 更高的 LoRA 秩 `r` 意味着更大的 LoRA 矩阵（A 和 B），从而增加可训练参数的数量，以及它们梯度和优化器状态所需的内存。然而，这种影响通常远小于基础模型或激活值。
- **优化器状态：** AdamW 等标准优化器会为每个可训练参数存储动量（均值和方差），这会占用大量内存（通常是参数本身在 FP32 中内存的两倍）。使用 8 位优化器（例如 `bitsandbytes` 中的 AdamW8bit）能显著降低这项开销。QLoRA 通常采用分页优化器，以在更新期间进一步节省内存。

以下图表提供了不同情况下显存需求的*粗略估计*。实际使用量将根据具体实现、序列长度、批量大小和库开销而有所不同。



[交互图表：微调 (70亿参数模型) 估算显存需求](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-5-peft-optimization-deployment/peft-infrastructure-requirements#plot-1p27q2y)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "微调 (70亿参数模型) 估算显存需求",
    "xaxis": {
      "title": "微调方法"
    },
    "yaxis": {
      "title": "估算最小显存 (GB)"
    },
    "barmode": "group"
  },
  "data": [
    {
      "type": "bar",
      "name": "全量微调 (FP16)",
      "x": [
        "Full",
        "LoRA",
        "QLoRA"
      ],
      "y": [
        30
      ],
      "marker": {
        "color": "#fa5252"
      }
    },
    {
      "type": "bar",
      "name": "LoRA (BF16 基础模型 + LoRA)",
      "x": [
        "Full",
        "LoRA",
        "QLoRA"
      ],
      "y": [
        null,
        18
      ],
      "marker": {
        "color": "#4c6ef5"
      }
    },
    {
      "type": "bar",
      "name": "QLoRA (4 位基础模型 + LoRA)",
      "x": [
        "Full",
        "LoRA",
        "QLoRA"
      ],
      "y": [
        null,
        null,
        10
      ],
      "marker": {
        "color": "#40c057"
      }
    }
  ]
}
```

</details>



> 70 亿参数模型在不同微调方法下的显存估算。全量微调需要为*所有*参数的模型、梯度和优化器状态分配显存。LoRA 能显著减少梯度/优化器内存占用。QLoRA 进一步降低了基础模型的内存占用。这些值假设采用中等批量大小和序列长度。

**建议：** 对于使用 LoRA 处理 70 亿至 130 亿参数模型的严肃工作，建议选用至少 16-24GB 显存的 GPU（例如 NVIDIA RTX 3090、4090、A4000、A6000）。对于这些模型上的 QLoRA，12-16GB 可能已足够，对于较小模型或经过高度优化的设置，甚至 8-10GB 即可。使用 LoRA 的大型模型（300 亿以上）通常需要 40GB 以上显存（例如 NVIDIA A100 40GB），而 QLoRA 使它们在 24-40GB 显存卡上也能运行。A100 80GB 或 H100 GPU 为更大的模型和批量大小提供了更多空间。

#### GPU 计算能力

虽然内存通常是首要障碍，但计算能力（以 TFLOPS 衡量）决定了训练速度。

- **前向/反向传播：** 即使 PEFT 中只有一小部分参数得到更新，计算过程仍涉及激活值通过*整个*基础模型。因此，具有更高 TFLOPS 和如 Tensor Cores（NVIDIA Volta、Turing、Ampere、Hopper 架构及更新版本提供）等功能的 GPU 将显著加速训练。
- **减少梯度计算：** PEFT 的计算节省主要来源于仅为适配器参数计算和更新梯度，这比全量微调中对所有参数进行操作快得多。
- **量化影响：** QLoRA 在前向和反向传播过程中引入了量化和反量化操作，从而带来了计算开销。虽然它节省了内存，但如果硬件未能充分发挥 4 位操作的性能，其每迭代速度有时会比 `bfloat16` 基础模型上的标准 LoRA 稍慢。然而，由于内存节省而能够使用更大的批量大小，通常能弥补这一不足，从而带来更快的整体训练收敛。

**建议：** 虽然 PEFT 可以在显存充足的旧款 GPU 上运行，但现代 GPU（NVIDIA Ampere 或 Hopper 系列）由于 TFLOPS 的提升和专用硬件加速（例如为 BF16 和 INT8/FP8 等低精度优化的 Tensor Cores），提供了显著的速度提升。

### 软件要求

确保正确的软件环境与拥有合适的硬件同样重要。兼容性问题可能经常出现，尤其是在使用最新功能时。

- **核心框架：** 需要最新版本的 PyTorch（推荐）或 TensorFlow。PEFT 的开发通常与 PyTorch 版本保持同步。
- **Hugging Face 生态系统：**
  - `transformers`：提供预训练 (pre-training)模型和标准 Transformer 架构的访问。
  - `peft`：实现 LoRA、QLoRA、适配器微调 (fine-tuning)等技术的主要库。它与 `transformers` 紧密结合。
  - `accelerate`：简化了 PyTorch 代码在混合精度和分布式设置（多 GPU、TPU）上的运行。对于管理设备放置和扩展训练非常必要。
  - `datasets`：用于加载和预处理训练数据。
- **量化 (quantization)库：**
  - `bitsandbytes`：对 QLoRA 很重要。提供 4 位量化、反量化和 8 位优化器的 CUDA 实现。确保安装的版本与你的 CUDA 工具包和 GPU 架构兼容。安装有时可能需要编译自定义 CUDA 内核。
- **CUDA 工具包：** 你安装的 NVIDIA 驱动和 CUDA 工具包必须与 PyTorch、`bitsandbytes` 及其他 GPU 加速库所需的版本兼容。查阅这些库的文档以了解具体的版本要求。使用 Docker 等容器化工具可以帮助有效管理这些依赖项。

**建议：** 从 Hugging Face 库（`transformers`、`peft`、`accelerate`、`datasets`）的最新稳定版本开始。如果你打算使用 QLoRA，请仔细查阅 `bitsandbytes` 的安装说明和兼容性矩阵。使用针对深度学习 (deep learning)优化的预构建 Docker 容器（如 NVIDIA 的 PyTorch 容器）通常可以节省大量设置时间。

### 存储和网络

虽然在实际训练迭代中不如 GPU 资源那么关键，但仍需考虑以下几点：

- **模型存储：** 下载大型基础模型（数十到数百 GB）需要充足的磁盘空间和网络带宽。
- **数据集存储：** 数据集大小也可能从兆字节到吉字节甚至更大。
- **检查点：** 保存训练检查点（包括适配器权重 (weight)和可能的优化器状态）需要磁盘空间。适配器权重较小（兆字节），但适配器的优化器状态会累积，尽管远少于全量微调 (fine-tuning)的状态。
- **网络带宽：** 主要对初始下载模型/数据集以及分布式训练设置中节点间梯度或参数 (parameter)的通信很重要。

规划你的基础设施需要平衡这些要求与你的预算和项目目标。PEFT 显著提高了微调的可行性，但了解具体的内存、计算和软件需求对于成功实施仍然非常必要。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了LoRA方法，解释了它如何减少可训练参数以及微调所需的资源。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  详细介绍了4比特NormalFloat (NF4) 量化、双重量化和分页优化器，这些技术减少了微调的VRAM需求。
- [Parameter-Efficient Fine-Tuning (PEFT) library](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  `peft`库的官方指南，提供了关于实现、API使用以及与Hugging Face模型兼容性的实用信息。
- [Accelerate training with 🤗 Accelerate](https://huggingface.co/docs/accelerate/en/index) — Hugging Face (2024)
  Hugging Face Accelerate的官方文档，该库协助PyTorch的设备分配、混合精度和分布式训练。

---

[上一节](../04-%E8%BF%9B%E9%98%B6LoRA%E5%AE%9E%E7%8E%B0%E6%96%B9%E6%B3%95%E5%92%8C%E5%8F%98%E4%BD%93/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%20QLoRA.md) · [下一节](02-%E7%94%A8%E4%BA%8EPEFT%E7%9A%84%E4%BC%98%E5%8C%96%E5%99%A8%E5%92%8C%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6%E5%99%A8.md)
