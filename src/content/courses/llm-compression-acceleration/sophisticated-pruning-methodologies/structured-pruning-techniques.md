---
course: "llm-compression-acceleration"
chapter: "sophisticated-pruning-methodologies"
lesson: "structured-pruning-techniques"
sourceId: 4344
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-3-sophisticated-pruning-methodologies/structured-pruning-techniques"
title: "结构化剪枝技术"
description: "专门针对Transformer架构实现块剪枝、通道剪枝和注意力头剪枝。"
order: 4
plots: []
sourceHash: "2c0003a131b0c1fe6144d40933be6051a0d670b382e69a3fe5088f7d6c3b40f7"
sourceCorrections: []
---

结构化剪枝是一种通过移除整个预定义的参数 (parameter)组来压缩机器学习 (machine learning)模型的技术。这种方法会产生更规则的稀疏模式，标准硬件（如GPU和TPU）通常能更高效地处理，从而直接减少计算和内存带宽需求，而无需专用内核。非结构化剪枝则通过移除单个权重 (weight)提供了最大灵活性，但其不规则的稀疏模式通常需要专用硬件或软件支持（如稀疏矩阵库）才能转化为实际的推理 (inference)速度提升。

其核心在于识别并移除对模型整体性能贡献最小的较大网络结构组件。对于基于Transformer的大型语言模型，有几种结构化剪枝技术尤为重要：

### 前馈网络 (FFN) 中的神经元剪枝

每个Transformer块中的前馈网络 (FFN) 子层通常由两次线性变换和一个激活函数 (activation function)构成。神经元剪枝针对该FFN的中间层，移除第一个权重 (weight)矩阵的整行以及第二个权重矩阵的对应列。这有效地移除了中间表示中的特定“神经元”或维度。

假设一个FFN定义为：


$$
\text{前馈网络}(x) = \text{线性变换}_2(\text{激活}(\text{线性变换}_1(x)))
$$


其中 $x$ 是输入，$\text{线性变换}_1$ 从维度 $d_{model}$ 映射到 $d_{ffn}$，$\text{线性变换}_2$ 从 $d_{ffn}$ 映射回 $d_{model}$。剪枝中间层的一个神经元意味着减小维度 $d_{ffn}$。如果我们移除第 $k$ 个神经元，我们将移除 $\text{线性变换}_1$ 中权重矩阵的第 $k$ 行和 $\text{线性变换}_2$ 中权重矩阵的第 $k$ 列。

**重要性判据：** 我们如何决定剪枝哪些神经元？常见的判据包括：

- **Lp-范数：** 计算与每个中间神经元相关的权重的L1或L2范数（即第一个矩阵中对应的行和第二个矩阵中对应的列）。范数较小的神经元被认为重要性较低。
- **激活统计：** 分析神经元在校准数据集上的平均激活值。持续激活值较低的神经元可能被剪枝。
- **梯度信息：** 使用梯度信息或泰勒展开近似法，衡量神经元对损失函数 (loss function)的影响。

剪枝神经元直接减小了FFN层的宽度 ($d_{ffn}$)，缩小了权重矩阵，并减少了推理 (inference)过程中所需的浮点运算 (FLOPs) 数量。

### 注意力头剪枝

多头注意力 (multi-head attention) (MHA) 使模型能够联合关注来自不同表示子空间、不同位置的信息。注意力头剪枝涉及从MHA层中移除整个注意力头。

回想一下，在MHA中，输入查询 ($Q$)、键 ($K$) 和值 ($V$) 通过为每个头学习到的不同投影矩阵，进行 $h$ 次（其中 $h$ 是头的数量）线性投影。注意力机制 (attention mechanism)独立应用于每个头，结果被连接起来并投影回模型维度。


$$
\text{多头}(Q, K, V) = \text{连接}(\text{头}_1, ..., \text{头}_h)W^O
$$


$$
\text{其中 头}_i = \text{注意力}(QW_i^Q, KW_i^K, VW_i^V)
$$


剪枝第 $j$ 个注意力头意味着移除对应的投影矩阵 $W_j^Q, W_j^K, W_j^V$ 以及输出投影矩阵 $W^O$ 的对应部分。

**重要性判据：** 选择要剪枝的头通常涉及比简单权重 (weight)范数更复杂的指标：

- **头重要性得分：** 衡量每个头对模型在特定任务上的输出或性能的贡献。这可以通过逐个屏蔽头并观察校准数据集上损失或评估指标的变化来估计。
- **冗余分析：** 分析不同头产生的注意力模式之间的相似性。高度相似的头可能是冗余的，从而允许剪枝其中一些。
- **基于梯度的敏感性：** 评估模型损失对每个头参数 (parameter)的敏感程度。

移除注意力头直接降低了注意力机制的计算成本，因为执行的独立注意力计算更少，并且随后的连接和输出投影涉及更小的矩阵。

> 比较非结构化剪枝产生的不规则稀疏性与结构化剪枝中移除整个神经元或注意力头产生的规则稀疏性。已剪枝的组件以灰色显示。

### 其他结构化方法

虽然神经元剪枝和注意力头剪枝是Transformer中最常见的方法，但还存在其他粒度级别：

- **层剪枝：** 移除整个Transformer块（层）。这是一种粗粒度的方法，对模型容量影响很大，通常会导致显著的准确性下降，除非原始模型存在严重的参数 (parameter)冗余。
- **块剪枝：** 在权重 (weight)矩阵内部，移除连续的权重块。这有时可以在非结构化剪枝的细粒度与神经元/注意力头剪枝的硬件友好性之间提供一个折中方案。

### 实现与微调 (fine-tuning)

结构化剪枝通常迭代进行：

1. **训练/加载模型：** 从预训练 (pre-training)的大型语言模型开始。
2. **评估结构：** 计算相关层中目标结构（神经元、注意力头）的重要性得分。
3. **剪枝：** 移除得分最低的结构，直至达到所需的稀疏度。这通常涉及创建掩码或直接修改架构以移除参数 (parameter)。
4. **微调：** 对剪枝后的模型进行一段时间的再训练（通常远短于原始预训练时间），以恢复剪枝过程中损失的准确性。
5. **重复（可选）：** 步骤2-4可以重复进行，以逐步达到更高的稀疏度。

微调是重要的一步。移除模型的大部分参数最初可能导致性能显著下降。微调使剩余参数能够适应并补偿被移除的组件。

### 硬件效率提升

结构化剪枝的主要优势在于其与硬件效率的直接对应。移除整个注意力头或FFN神经元会带来更小的密集矩阵乘法和更低的内存需求。标准深度学习 (deep learning)库和硬件加速器可以高效执行这些较小的密集操作，而无需专用稀疏计算支持。这通常意味着延迟和内存占用方面的可衡量减少，使其在资源受限环境中部署大型语言模型具有吸引力。

### 权衡与考量

结构化剪枝强制在比非结构化剪枝更粗的粒度上进行选择。尽管它产生更硬件友好的模型，但它可能会移除非结构化剪枝会保留的一些参数 (parameter)，可能导致在相同参数数量减少的情况下，准确性下降幅度略大。非结构化剪枝和结构化剪枝之间的选择通常取决于目标硬件、所需的压缩级别以及可接受的性能下降。将结构化剪枝与其他技术（如量化 (quantization)）结合，可以进一步提高效率，下一节将对此进行讨论。

## 参考资料

- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) — Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, Hans Peter Graf (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1608.08710](https://doi.org/10.48550/arXiv.1608.08710)
  引入了通过移除卷积神经网络中的整个滤波器进行结构化剪枝的概念，为硬件高效的模型压缩奠定了关键方法。
- [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.09418) — Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov (2019)
  Journal: ACL; Publisher: Association for Computational Linguistics; DOI: [10.48550/arXiv.1905.09418](https://doi.org/10.48550/arXiv.1905.09418)
  研究了Transformer模型中注意力头的冗余性和重要性，为有效的注意力头剪枝提供了见解和方法。
- [Sparsity in Deep Learning: Pruning and beyond](https://www.semanticscholar.org/paper/Sparsity-in-Deep-Learning%3A-Pruning-and-beyond-Blalock-Ortiz/857731a26d71317d7b140cfc34ae3b94a08080a9) — Jacob Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, John Guttag (2020)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 108; Pages: 1993-2011; DOI: [10.1109/JPROC.2020.3031112](https://doi.org/10.1109/JPROC.2020.3031112)
  一篇关于深度学习稀疏性的综合调查报告，涵盖了各种剪枝技术，包括结构化剪枝及其在现代神经网络架构中的应用。
