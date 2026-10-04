---
course: "planning-optimizing-ai-infrastructure"
chapter: "foundations-ai-compute-infrastructure"
lesson: "memory-for-large-models"
sourceId: 6970
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure/memory-for-large-models"
title: "内存及其对大型模型的重要性"
description: "分析训练和部署大规模人工智能和深度学习模型所需的内存（RAM）需求。"
order: 6
plots: []
sourceHash: "582662fe110852bb3064460a5bf113874e83481cd296bd35759acd1e63b26d0c"
sourceCorrections: []
---

虽然CPU和GPU的算力 (compute)是主要考量，但可用内存量通常成为处理大型模型时的第一个实际瓶颈。如果处理器无法获取其计算所需的数据和模型参数 (parameter)，它将是无用的。因此，了解内存需求对于训练和部署现代人工智能系统都非常必要。

在人工智能硬件背景下，我们主要指两种类型的内存：

1. **系统内存 (随机存取内存)：** 这是服务器的主内存，CPU可以直接访问。它用于存储操作系统、数据预处理脚本以及等待被送入加速器的整个数据集。
2. **GPU显存 (VRAM) (视频内存)：** 这是直接位于GPU卡上的高速内存。它比系统内存明显更快，是所有加速深度学习 (deep learning)活动发生的地方。GPU只能处理存在于其自身显存中的数据。

对于GPU加速工作负载，显存几乎总是更重要的限制。如果训练任务的组成部分无法放入单个GPU的显存中，或者无法跨多GPU配置的汇总显存，该任务将失败。

### 训练期间显存 (VRAM)被什么占用？

在神经网络 (neural network)训练期间，多个组成部分必须同时驻留在显存中。总内存占用远大于模型本身。

> 模型训练周期中GPU显存的主要占用者。所需的总内存显著超出仅模型参数 (parameter)本身的大小。

我们来分解这些组成部分：

- **模型参数：** 这些是神经网络的权重 (weight)和偏差。其大小取决于参数数量和存储它们所用的精度（例如，32位浮点数，即FP32）。
- **梯度：** 在反向传播 (backpropagation)期间，必须为模型中的每个参数计算梯度。因此，梯度所需的内存通常等于模型参数所需的内存。
- **优化器状态：** 像Adam这样的现代优化器会维护状态，以调整每个参数的学习率。例如，Adam为每个参数存储两个移动平均值（动量和方差），这有效地使模型参数的内存占用再次翻倍。
- **激活值：** 这些是前向传播期间计算的每个层的中间输出。它们需要存储在内存中，以便在随后的反向传播期间用于梯度计算。激活值的大小是动态的，取决于输入数据的批次大小和序列长度。
- **输入数据：** 当前正在处理的数据批次也必须加载到显存中。

### 实际内存计算

为了使其具体化，我们来估计训练一个使用Adam优化器和标准32位精度（FP32）的70亿参数 (parameter)语言模型所需的显存 (VRAM)，其中每个参数需要4字节。

1. **模型参数：**
   
   $$
   70 \text{ 亿参数} \times 4 \text{ 字节/参数} = 28 \text{ GB}
   $$
   
2. **梯度：** 梯度的尺寸与参数相同。
   
   $$
   70 \text{ 亿参数} \times 4 \text{ 字节/参数} = 28 \text{ GB}
   $$
   
3. **优化器状态 (Adam)：** Adam为每个参数存储两个状态。
   
   $$
   2 \times 70 \text{ 亿参数} \times 4 \text{ 字节/参数} = 56 \text{ GB}
   $$
   

仅这三个组成部分所需的总显存为：


$$
28 + 28 + 56 = 112 \text{ GB}
$$


此计算甚至不包括激活值内存或CUDA内核开销，但它已经超出NVIDIA A100 80GB等高端GPU的容量。这就是为什么你经常听到“内存不足”（OOM）错误的原因。一个OOM错误简单来说意味着你尝试分配给GPU显存的数据量超出了其可用量。这可能是因为模型过大、`batch_size`过高（这会增加激活内存），或者多种因素组合造成的。

### 推理 (inference)内存

当模型部署用于推理（即进行预测）时，内存需求要低得多。你不再需要存储梯度或优化器状态。主要内存占用者是模型参数 (parameter)本身，加上当前输入的激活值。

对于我们的70亿参数模型，推理的基础内存将是模型权重 (weight)所需的28 GB，再加上单个推理请求的激活值内存。这更易于管理，并解释了为什么可以在一个太小而无法从头训练模型的GPU上运行该模型的推理。

了解这种分解是优化的前提。混合精度训练（使用16位浮点数）或选择不同的优化器等技术可以大幅减少这种内存占用，这些我们将在后续章节中讨论。目前，主要收获是：在规划基础设施时，模型大小和训练配置直接决定了你的最小内存需求。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了 Adam 优化器，详细阐述了其设计以及每个参数的两个一阶和二阶矩估计，解释了其内存占用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，涵盖了神经网络训练、反向传播以及激活值和参数的内存注意事项。
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  官方指南，详细介绍了在 NVIDIA GPU 上优化深度学习性能的最佳实践，包括内存管理考量。
- [ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models](https://arxiv.org/abs/1910.02054) — Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He (2020)
  Journal: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '20); Publisher: ACM; Pages: 1–16; DOI: [10.1145/3410464.3410714](https://doi.org/10.1145/3410464.3410714)
  介绍了 ZeRO，这是一系列内存优化技术，通过在多个 GPU 之间高效分发模型状态，对于训练数十亿参数的模型至关重要。
