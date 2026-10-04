# DeepSpeed 介绍

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-16-implementing-distributed-training-frameworks/introduction-deepspeed)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要在大规模模型上实现数据并行（DP）、张量并行（TP）和流水线并行（PP）等分布式训练策略，需要用到实用的工具。虽然PyTorch等框架提供了基本的分布式数据并行（`DistributedDataParallel`），但训练拥有数千亿甚至数万亿参数 (parameter)的模型，需要更精密的内存管理和扩展能力。为此，专门的库变得不可或缺。

DeepSpeed由微软研究院开发，是一个开源的深度学习 (deep learning)优化库，旨在显著提高大型模型训练的速度和规模，同时最大限度地减少所需的硬件资源。它与PyTorch结合使用，提供了一系列优化功能，解决了大型训练中遇到的主要瓶颈，特别是GPU内存限制。

标准数据并行中的根本问题是内存冗余。参与DP的每个GPU通常都包含模型权重 (weight)、梯度和优化器状态（如Adam中的动量和方差缓冲区）的完整副本。对于拥有数十亿参数的模型，仅优化器状态所需的内存（通常以32位精度（FP32）存储）就可以超过即使是高端GPU的可用内存。考虑一个拥有 $1$ 亿参数的模型。权重可能需要 $4$ GB（FP32）或 $2$ GB（FP16）。然而，Adam优化器状态通常每个参数需要额外 $16$ 字节（动量 $4$ 字节，方差 $4$ 字节，FP32主权重 $4$ 字节，如果使用FP16，梯度 $4$ 字节），这意味着 *每个GPU* 额外增加 $16$ GB。这很快就变得难以维持。

DeepSpeed通过几种创新的技术来应对这一问题及其他扩展难题：

1. **ZeRO（零冗余优化器）：** 这可以说是DeepSpeed最受认可的贡献。ZeRO是一系列优化方法，旨在消除数据并行训练中的内存冗余。ZeRO不是在所有数据并行GPU上复制优化器状态、梯度，甚至可能模型权重，而是将这些状态分区到可用设备上。这意味着每个GPU只保留整体状态的一部分，从而大幅减少了每个设备的内存占用。我们将在下一节中查看ZeRO的不同阶段（阶段1、阶段2和阶段3）。
2. **内存卸载：** DeepSpeed允许将训练状态的部分（优化器状态、激活或参数）从GPU内存卸载到主机CPU的主内存甚至NVMe固态硬盘。虽然访问CPU内存或NVMe比GPU HBM慢，但卸载不常访问的状态可以释放宝贵的GPU内存，从而能够训练比原本能容纳的更大的模型。
3. **高效流水线并行：** 除了用于数据并行的ZeRO之外，DeepSpeed还包含其自身高度优化的流水线并行实现。这使用户能够将模型的层分区到多个GPU上，减少每个GPU激活所需的内存，并通过分散计算图来支持更大的模型。
4. **自定义内核和优化器：** DeepSpeed通常包含高度优化的CUDA内核，用于大型模型中常见的操作，例如自定义Transformer层或像FusedAdam这样高效的优化器，它们将多个步骤组合到一个内核启动中以获得更好的性能。
5. **简化的混合精度训练：** 它提供了易于使用的工具，用于管理FP16或BF16混合精度训练，包括自动损失缩放。

DeepSpeed的设计目标之一是易于集成。对于许多常见的使用场景，特别是使用ZeRO时，对现有PyTorch训练脚本只需进行最小的改动。核心修改通常包括使用DeepSpeed的`initialize`函数封装模型、优化器和数据加载器。

以下是DeepSpeed集成的示意图：

```python
import torch
import deepspeed

# 假设模型、优化器、数据加载器已经是已定义的PyTorch对象
# DeepSpeed设置的配置字典（例如，ZeRO阶段，批次大小）
config_params = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5
        }
    },
    "fp16": {
        "enabled": True # 示例：启用混合精度
    },
    "zero_optimization": {
        "stage": 2 # 示例：启用ZeRO阶段2
    }
    # ... 其他DeepSpeed配置
}

# 初始化DeepSpeed引擎
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer, # 传递原始优化器
    config_params=config_params
)

# 训练循环修改：
# 将 model(inputs) 替换为 model_engine(inputs)
# 将 loss.backward() 替换为 model_engine.backward(loss)
# 将 optimizer.step() 替换为 model_engine.step()

for batch in dataloader:
    inputs, labels = batch
    inputs = inputs.to(model_engine.local_rank) # 将数据移动到正确的设备
    labels = labels.to(model_engine.local_rank)

    outputs = model_engine(inputs)
    loss = calculate_loss(outputs, labels) # 你的损失计算

    model_engine.backward(loss)
    model_engine.step()
```

在此代码片段中，`deepspeed.initialize`接受标准PyTorch模型和优化器，以及一个配置字典（`config_params`），并返回一个`model_engine`。这个引擎在训练循环中取代了原始模型，并且它的方法（`backward`、`step`）根据提供的配置处理分布式训练、梯度累积、混合精度和ZeRO优化的复杂性。

DeepSpeed提供了一整套工具，旨在使大型语言模型的训练更易于进行且更高效。它通过ZeRO专注于内存优化，结合对卸载和流水线并行的支持，使其成为工程师扩展模型规模的有力选择。我们现在将更仔细地查看ZeRO优化器的不同阶段。

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://doi.org/10.1109/SC44002.2020.9251842) — Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He (2020)
  Journal: SC '20: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: IEEE; Pages: 1–16; DOI: [10.1109/SC44002.2020.9251842](https://doi.org/10.1109/SC44002.2020.9251842)
  介绍了ZeRO（零冗余优化器）技术系列，这是DeepSpeed在大型模型训练中大幅减少内存占用的基础性贡献，通过对优化器状态、梯度和参数进行分区实现。
- [DeepSpeed Documentation](https://deepspeed.readthedocs.io/) — Microsoft DeepSpeed Team (2024)
  Publisher: Microsoft
  DeepSpeed的官方综合文档，提供安装、配置、API使用以及包括ZeRO、内存卸载和流水线并行在内的所有功能的详细指南和实践示例。

---

[上一节](01-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E5%BA%93%E6%A6%82%E8%BF%B0.md) · [下一节](03-%E4%BD%BF%E7%94%A8%20DeepSpeed%20ZeRO%20%E4%BC%98%E5%8C%96.md)
