---
course: "advanced-diffusion-architectures"
chapter: "consistency-models"
lesson: "consistency-model-architecture"
sourceId: 5458
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-5-consistency-models/consistency-model-architecture"
title: "一致性模型的架构考量"
description: "训练一致性模型时常用的架构选择和修改。"
order: 6
plots: []
sourceHash: "4863579d2cc9fb8cf9c7c8478b505cc75db6e55947f7d68eef529f6cbf889f5e"
sourceCorrections: []
---

尽管一致性模型的核心思想在于其训练目标和采样过程，但神经网络 (neural network)架构的选择仍是影响性能、训练稳定性和推理 (inference)速度的重要因素。通常，在标准扩散模型中成功应用的架构，可作为一致性模型的良好起点，它们已采用成熟的设计模式。

### 复用现有扩散架构

一致性模型，无论是通过蒸馏训练还是独立训练，通常都会采用那些已被证明在扩散建模中有效的骨干架构。这通常指使用：

1. **U-Net变体**：带有跳跃连接的深度卷积网络，擅长获取图像中的多尺度空间信息。高级U-Net结合了自注意力 (self-attention)与交叉注意力机制 (attention mechanism)（如第2章所述），常被使用，特别是高分辨率图像生成。
2. **Transformer架构**：如扩散Transformer（DiT，第3章已介绍）之类的模型，将卷积层替换为Transformer块，也是可行的选择。它们擅长处理远距离依赖并可扩展模型能力。

The理由很简单：这些架构已针对基于有噪声的输入 $x_t$ 和时间 $t$ 预测输出（如噪声 $\epsilon$ 或数据 $x_0$）的任务进行优化。一致性模型只是将这种预测能力重新定向，以学习一致性映射 $f_{\theta}(x_t, t) \approx x_0$。

### 时间嵌入 (embedding)集成

与标准扩散模型一样，一致性模型需要关于“去噪”过程中当前点的信息，这由时间 $t$（或噪声水平 $\sigma$）表示。嵌入 $t$ 的方法通常继承自基础扩散架构：

- **正弦嵌入**：使用固定正弦位置嵌入的常见方法。
- **可学习傅里叶特征**：其中频率和相位可学习的变体。
- **MLP投影**：将时间嵌入投影到适合集成到网络层中的特征空间（例如，通过相加或自适应归一化 (normalization)）。

对于连续时间一致性模型，确保网络能平稳处理连续 $t$ 值是很重要的。嵌入机制必须有效表示这个连续变量。

### 条件机制

一致性模型中的条件生成（例如，基于文本提示或类别标签）通常遵循与父级扩散模型中相同的策略：

- **交叉注意力**：将条件信息（如文本嵌入 (embedding)）集成到Transformer或U-Net注意力层中。
- **自适应归一化 (normalization)（AdaLN/AdaGN）**：根据条件嵌入调整归一化层参数 (parameter)。
- **拼接**：将条件嵌入直接拼接至输入特征图（在最先进的模型中较不常见）。

条件机制的选择很大程度上取决于基础架构（U-Net vs. Transformer）和条件信号的类型。一致性训练目标本身并不要求对条件信息如何融入网络前向传播进行根本性改变。

### 网络容量与蒸馏

在使用一致性蒸馏时，会出现一个重要考量：一致性模型（学生模型）是否需要与预训练 (pre-training)扩散模型（教师模型）相同的架构复杂度和参数 (parameter)量？

- **蒸馏**：通常，较小的学生网络可以有效地从较大的教师模型中学习到一致性映射。这是一个主要优点，因为它能带来更快、更节省内存的推理 (inference)模型。蒸馏过程传递了这种核心映射，可能允许架构简化（例如，更少的层、更小的隐藏维度）。
- **独立训练**：从头开始训练时，网络必须有足够的容量来直接学习从不同 $t$ 值下的噪声输入 $x_t$ 到 $x_0$ 的复杂映射。所需容量将取决于数据复杂度和期望的输出质量，可能与为相同任务训练的标准扩散模型所需容量相似。

> 此图描绘了在蒸馏过程中，大型教师扩散模型与可能更小的学生一致性模型之间的关系。两者常共享架构模式，但在大小和预测目标上有所不同。

### 输出层配置

标准扩散模型常被参数 (parameter)化以预测在步骤 $t$ 添加的噪声 $\epsilon$。网络输出 $\epsilon_{\theta}(x_t, t)$ 随后用于估计 $x_0$。一致性模型，顾名思义，旨在直接将轨迹上的任意点 $x_t$ 映射到其原点 $x_0$。因此，一致性模型 $f_{\theta}(x_t, t)$ 的输出层通常被配置为直接生成与输入数据 $x_0$ 具有相同维度和值范围的输出。与预测 $\epsilon$ 的扩散模型相比，这可能涉及对最终激活函数 (activation function)的调整（例如，如果数据被归一化 (normalization)到 [-1, 1] 则使用 `tanh`）。

### 架构要点总结

- **基础**：标准扩散模型中的U-Net和Transformer是很好的起点。
- **时间/条件**：复用有效的嵌入 (embedding)和条件技术（时间使用正弦/MLP，条件使用交叉注意力/AdaLN）。
- **容量**：相较于教师模型，蒸馏能使学生模型可能更小、更快。独立训练需要足够的容量。
- **输出**：网络的最后一层必须调整为直接预测 $x_0$（或表示 $x_0$ 的值）。

尽管一致性模型引入了一种更快采样的办法，但其架构组成部分通常依然常见，使得实践者可以运用现有扩散模型架构的知识，同时专注于一致性目标的具体内容及其对模型大小和输出表示的影响。

## 参考资料

- [Consistency Models](https://arxiv.org/abs/2303.01469) — Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever (2023)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.2303.01469](https://doi.org/10.48550/arXiv.2303.01469)
  介绍了作为本节核心内容的一致性模型的核心概念和方法论。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020)
  Journal: arXiv; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  这是一篇关于扩散模型的基础论文，确立了U-Net架构和正弦时间嵌入技术，这些技术被广泛采纳作为一致性模型的骨干网络。
