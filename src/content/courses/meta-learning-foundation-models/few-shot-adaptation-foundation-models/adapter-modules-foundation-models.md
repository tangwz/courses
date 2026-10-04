---
course: "meta-learning-foundation-models"
chapter: "few-shot-adaptation-foundation-models"
lesson: "adapter-modules-foundation-models"
sourceId: 4357
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-5-few-shot-adaptation-foundation-models/adapter-modules-foundation-models"
title: "基础模型适配器模块"
description: "用于高效调优的适配器模块的架构与训练。"
order: 2
plots: []
sourceHash: "9e39386f0f1b7048ef252f7b5917582fbd60b406c2c3fd774fcf3623a127d140"
sourceCorrections: []
---

对拥有数十亿参数 (parameter)的基础模型进行适应训练，会带来巨大的计算挑战，尤其是在数据稀缺的少样本场景中。由于内存限制和灾难性遗忘的风险，全面微调 (fine-tuning)通常不可行。参数高效微调（PEFT）方法通过仅修改模型参数的一小部分来提供解决方案。在最早且最有影响力的PEFT技术中，适配器模块是其中之一。

适配器的主要思想简洁明了：不是修改大型预训练 (pre-training)模型的现有权重 (weight)，而是在冻结的基础模型层*之间*插入小型、可训练的神经网络 (neural network)模块。在适应训练期间，只有这些新添加的适配器模块的参数会被更新，而原始的基础模型权重保持不变。

### 适配器架构与插入

适配器模块通常由一系列操作组成，这些操作旨在将输入激活投影到较低维空间（瓶颈），施加非线性变换，然后将其投影回原始维度。一种常见的架构是：

1. **降维投影：** 一个线性层（$W_{down}$）将输入激活 $h$（来自前一层）映射到更小的维度 $m$。
2. **非线性变换：** 应用一个激活函数 (activation function) $\sigma$（例如，ReLU，GeLU）。
3. **升维投影：** 另一个线性层（$W_{up}$）将结果映射回 $h$ 的原始维度。
4. **残差连接：** 适配器的输出被加回到原始输入激活 $h$。

在数学上，适配器模块对中间激活 $h \in \mathbb{R}^d$ 应用的变换可以表示为：


$$
\text{适配器}(h) = W_{up}(\sigma(W_{down}(h)))
$$


其中 $W_{down} \in \mathbb{R}^{m \times d}$ 且 $W_{up} \in \mathbb{R}^{d \times m}$。维度 $m$ 是适配器的瓶颈维度，这是一个非常重要的超参数 (parameter) (hyperparameter)，通常选择使得 $m \ll d$。这个瓶颈是参数效率的来源。包含适配器的层的最终输出 $h'$ 是：


$$
h' = h + \text{适配器}(h)
$$


残差连接非常重要。它允许适配器学习对层功能的修改。如果适配器学习输出零（这可以通过初始化来促成），则适应后的层行为与原始预训练 (pre-training)层完全相同。

适配器可以插入到 Transformer 块内的各个位置。常见的策略包括将它们依次放置在多头自注意力 (self-attention)（MHSA）子层和前馈网络（FFN）子层之后。

> 适配器模块在标准 Transformer 块内的插入位置。适配器通常放置在主要子层（MHSA、FFN）之后，并通过残差连接集成。原始 Transformer 块的参数保持冻结。

### 适配器训练

在针对新任务进行适应训练时，基础模型的参数 (parameter)（$\theta_{FM}$）被冻结。只有插入的适配器模块的参数（$\theta_{adapter}$，具体而言是所有适配器中的 $W_{down}$ 和 $W_{up}$）是可训练的。训练过程涉及相对于 $\theta_{adapter}$ 最小化任务特定的损失函数 (loss function) $\mathcal{L}_{task}$（例如，分类的交叉熵）：


$$
\min_{\theta_{adapter}} \mathcal{L}_{task}(\text{模型}(X; \theta_{FM}, \theta_{adapter}), Y)
$$


其中 $(X, Y)$ 代表目标任务的少样本训练数据。

一种常见的初始化策略是随机初始化 $W_{down}$（例如，使用 Xavier 或 He 初始化），并将 $W_{up}$ 初始化为接近零的值。这确保了在训练开始时，适配器模块的影响极小（$\text{适配器}(h) \approx 0$），并且模型行为类似于原始预训练 (pre-training)的基础模型，为优化提供了一个稳定的起点。

### 优点与考量

适配器为少样本适应提供了几个引人注目的优点：

- **高参数 (parameter)效率：** 适配器引入的可训练参数数量通常相对于基础模型的总参数数量非常少（通常小于5%，可通过瓶颈维度 $m$ 进行调整）。这大大减少了训练期间存储优化器状态和梯度所需的内存。
- **降低计算成本：** 训练速度更快，因为梯度只需针对少量适配器参数计算。尽管仍然需要通过冻结的基础模型进行反向传播 (backpropagation)，但优化器步骤的成本要低得多。
- **模块化：** 由于基础模型未改变，针对不同任务训练的适配器可以独立存储。在特定任务的推理 (inference)时，相应的适配器模块可以与基础模型一起加载。这避免了存储大型基础模型的多个副本的需要，并防止了在学习顺序任务时发生灾难性遗忘。一个基础模型可以通过简单地替换适配器来服务多个任务。

然而，也有一些考量：

- **推理延迟：** 与原始模型或LoRA等方法（通常可以将其修改合并到原始权重 (weight)中）相比，添加适配器模块在推理过程中引入了额外的计算步骤。虽然每个适配器的计算量相对较小，但跨多层的累积效应可能会轻微增加延迟。适配器融合等技术旨在通过组合顺序适配器来减轻这一点。
- **潜在性能限制：** 尽管非常有效，尤其是在低数据量场景下，但如果下游任务有足够的数据可用，适配器有时可能会表现不如全面微调 (fine-tuning)。其表示能力受到瓶颈维度和基础模型固定性质的限制。
- **超参数 (hyperparameter)敏感性：** 性能可能取决于瓶颈维度 $m$ 的选择、特定的非线性函数 $\sigma$、插入策略（在 MHSA、FFN 或两者之后）以及适配器训练所用的学习率。

适配器是PEFT中一项基本的技术。它们表明通过仅引入和训练少量新参数来适应大型模型是可行的，这为LoRA和提示调优等其他方法提供了途径，这些方法研究了实现参数效率的不同方式，我们将在以下章节中看到。

## 参考资料

- [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html) — Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly (2019)
  Journal: Proceedings of the 36th International Conference on Machine Learning; Publisher: PMLR; Volume: 97; Pages: 2790-2799; DOI: [10.48550/arXiv.1902.00751](https://doi.org/10.48550/arXiv.1902.00751)
  这篇基础性论文提出了适配器模块作为自然语言处理任务中参数高效微调的方法，介绍了其核心概念和架构。
