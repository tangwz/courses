---
course: "foundations-transformers-architecture"
chapter: "encoder-decoder-stacks"
lesson: "stacking-encoder-decoder-layers"
sourceId: 2025
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-5-encoder-decoder-stacks/stacking-encoder-decoder-layers"
title: "多层堆叠"
description: "通过堆叠多个相同编码器和解码器层来增加模型深度的影响。"
order: 9
plots: []
sourceHash: "ff7549c610596cd1a458ff240eb1afe8a4546d27caea42b39f149f8f63d56852"
sourceCorrections: []
---

单个编码器层和解码器层尽管具有内部结构，但它们对输入表示执行的转换通常是有限的。Transformer架构的真正威力源于顺序组合这些层，为编码器和解码器构建深层堆叠。例如，原始Transformer模型为编码器使用了$N=6$个相同层的堆叠，为解码器也使用了$N=6$个相同层的堆叠。

### 为何堆叠层？

堆叠层使得模型能够分层学习输入数据渐趋复杂的表示。正如卷积神经网络 (neural network) (CNN)在不同层中从简单边缘到复杂对象构建表示一样，堆叠的Transformer层逐步优化序列表示。

1. **分层处理：** 初始层可能专注于序列中的局部上下文 (context)和依赖关系。后续层可以整合更长距离的信息，借助较低层的优化表示来捕捉更多全局关系和抽象特征。每层内的多头注意力 (multi-head attention)机制 (attention mechanism)允许不同的头关注不同方面，而堆叠使模型能够逐层在这些多元视角上进行构建。
2. **模型能力增强：** 每层都增加了计算深度和参数 (parameter)复杂性。层数越多，模型就具备更大的能力来近似机器翻译或文本生成等任务所需的复杂函数。通过多层顺序应用自注意力 (self-attention)、交叉注意力（在解码器中）和前馈转换，可实现从输入序列到输出序列的高度非线性和强大映射。

### 堆叠机制

在标准Transformer架构中，编码器和解码器均由指定数量$N$个相同层连续堆叠组成。尽管这些层共享相同的结构（相同的子层和维度），但每层都有其独有的可训练权重 (weight)集。

- **编码器堆叠：** 输入序列（词元 (token)嵌入 (embedding) + 位置编码 (positional encoding)）首先由编码器层1处理。层1的输出与输入具有相同维度，并作为层2的输入，以此类推，直到层$N$。最终编码器层（层$N$）的输出张量封装了整个输入序列的丰富表示。这个最终编码器输出随后被重要地用作$N$个解码器层中*每个*交叉注意力子层的键（$K$）和值（$V$）输入。
- **解码器堆叠：** 类似地，解码器堆叠处理目标序列嵌入（加位置编码）。解码器层$i$的输出成为解码器层$i+1$的输入。每个解码器层对目标序列执行带掩码的自注意力 (self-attention)，接着与最终编码器输出进行交叉注意力，最后通过一个位置前馈网络处理结果。最终解码器层（层$N$）的输出随后被输入到最终线性变换和softmax层，以生成词汇表 (vocabulary)上的输出概率。

> 数据通过堆叠的编码器和解码器层流动。最终编码器输出通过交叉注意力机制 (attention mechanism)为所有解码器层提供上下文 (context)（键和值）。

### 启用多层堆叠：残差连接与归一化 (normalization)

简单地堆叠层可能导致训练困难，特别是深度网络中常见的梯度消失问题，即梯度变得过小，无法有效更新早期层的权重 (weight)。Transformer架构在每层中引入了两个重要机制来减轻此问题：

1. **残差连接（加）：** 每个子层（自注意力 (self-attention)、前馈）周围都有一个残差连接。子层的输入$x$在经过dropout后，与子层的输出$\text{子层}(x)$相加。这为梯度在网络中反向流动创建了一条直接路径，或“捷径”。这通过确保梯度在通过多层反向传播 (backpropagation)时不会过度减小，从而大幅简化了深度模型的优化。在原始的Post-LN公式中，该操作形式上定义为$\text{层归一化}(x + \text{Dropout}(\text{子层}(x)))$。
2. **层归一化（归一）：** 应用于残差连接路径内（在Pre-LN变体中是在子层之前，在原始Post-LN公式中是在相加之后）。层归一化通过对每个位置在嵌入 (embedding)维度上独立地归一化特征来稳定每层内的激活。这有助于防止激活值爆炸或消失，减少对初始化的敏感度，并且通常允许更快、更稳定的训练，特别是对于更多层堆叠。

如果没有这些组件，训练具有大量层（例如$N > 2$）的Transformer将极其困难，甚至不可能。它们确保信息和梯度即使通过数十个堆叠层也能有效传播。

### 模型深度带来的影响

增加层数$N$直接影响模型：

- **性能：** 通常，更深的模型（更大的$N$）在复杂序列任务上能取得更佳的性能，直到出现边际效益递减或优化困难。最佳深度通常取决于任务复杂性、可用训练数据量以及计算预算。
- **计算成本：** 训练时间和推理 (inference)时间都大致与$N$呈线性关系。将层数加倍，通过编码器和解码器堆叠进行一次前向传播所需的计算量也大致加倍。
- **参数 (parameter)：** 假设每层结构相同（这是标准做法），总参数量也与$N$呈线性关系。这会增加训练期间存储模型检查点和激活值的内存需求。

选择$N$是Transformer设计中的一个基本超参数 (hyperparameter)。虽然原始论文使用了$N=6$，但现代大型语言模型常采用更多层堆叠（例如，$N=24, 48, 96$甚至更多）。这种深度的增加得益于对庞大数据集的访问、大量计算资源以及架构细节（如广泛采用Pre-LN归一化 (normalization)以提高稳定性）和训练技术的持续改进。

总之，通过堆叠多个结构相同的编码器和解码器层，Transformer模型实现了高性能所需的深度。这种深度允许对序列信息进行分层处理，并为复杂序列建模任务提供了所需的模型能力。这些多层堆叠的成功训练很大程度上依赖于每个组成块中残差连接和层归一化的精心整合。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: NeurIPS; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇基础论文介绍了Transformer架构，详细阐述了其编码器-解码器堆栈、多头注意力机制、残差连接和层归一化。
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)
  Journal: CVPR; Pages: 770-778; DOI: [10.48550/arXiv.1512.03385](https://doi.org/10.48550/arXiv.1512.03385)
  这篇论文介绍了残差网络，一种通过缓解梯度消失问题来训练深度神经网络的方法。
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (2016)
  Journal: arXiv; DOI: [10.48550/arXiv.1607.06450](https://doi.org/10.48550/arXiv.1607.06450)
  这篇论文介绍了层归一化，一种通过在每层内部归一化激活来稳定和加速深度神经网络训练的技术。
