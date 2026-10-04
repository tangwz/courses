---
course: "introduction-to-transformer-models"
chapter: "training-implementing-transformers"
lesson: "loss-functions-sequence-tasks"
sourceId: 3026
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-4-training-implementing-transformers/loss-functions-sequence-tasks"
title: "序列任务的损失函数"
description: "回顾训练Transformer时常用的损失函数，比如交叉熵损失。"
order: 3
plots: []
sourceHash: "70b8fee3c8ea5bc2a08c95f22a220d319922aa370d7964f70d854f561a247f56"
sourceCorrections: []
---

为了有效地训练Transformer模型，准确衡量其性能是必不可少的。训练的目标是调整模型的参数 (parameter)（权重 (weight)和偏差），使其预测结果越来越接近实际的目标序列。为了指导这个过程，我们需要一种方法来量化 (quantization)预测输出与真实输出之间的“误差”或“差异”。这就是损失函数 (loss function)（有时也称为成本函数或目标函数）的作用。

对于Transformer处理的大多数序列到序列任务，例如机器翻译、文本摘要或问答，模型需要预测一系列标记 (token)（词语、子词 (subword)、字符）。在输出序列的每一步，模型实际上都执行一个分类任务：它尝试从整个词汇表 (vocabulary)中预测正确的下一个标记。

### 交叉熵损失：常用选择

鉴于每个时间步的这种分类性质，用于训练Transformer进行序列生成任务的最常用损失函数 (loss function)是**交叉熵损失**。你可能还记得简单神经网络 (neural network)中分类问题里的交叉熵。在这里，我们将其重复应用于整个序列。

让我们拆解它在Transformer背景下的工作原理：

1. **模型输出：** 对于输出序列中的每个位置，Transformer的解码器（具体来说，是最终的线性层加上一个softmax函数）生成一个覆盖整个目标词汇表 (vocabulary)的概率分布。如果我们的词汇表有$V$个标记 (token)，则位置$t$的输出是一个概率向量 (vector)$P_t = [\hat{y}_{t,1}, \hat{y}_{t,2}, ..., \hat{y}_{t,V}]$，其中$\hat{y}_{t,i}$是模型预测位置$t$处的标记是词汇表中第$i$个标记的概率。这些概率之和为1（$\sum_{i=1}^{V} \hat{y}_{t,i} = 1$）。
2. **目标输出：** 实际目标序列提供“真实情况”。对于给定位置$t$，我们知道正确的标记。这通常隐式表示为一个整数索引，对应词汇表中正确标记的位置；或者显式表示为一个大小为$V$的独热编码向量$Y_t$。这个向量在正确标记的索引处为1，其他地方为0。例如，如果正确标记是词汇表中的第5个词，则$Y_t = [0, 0, 0, 0, 1, 0, ..., 0]$。
3. **计算损失：** 交叉熵损失衡量预测概率分布$P_t$与真实分布（由$Y_t$表示）之间的差异。对于单个位置$t$，公式是：

   
   $$
   H(Y_t, P_t) = - \sum_{i=1}^{V} Y_{t,i} \log(\hat{y}_{t,i})
   $$
   

   由于$Y_t$是一个独热向量，只有与正确标记对应的项（假设其索引为$c$）是非零的（$Y_{t,c} = 1$）。因此，求和简化为：

   
   $$
   H(Y_t, P_t) = - \log(\hat{y}_{t,c})
   $$
   

   这意味着单个位置的损失简单地是模型分配给*正确*标记的概率的负对数。

   - 如果模型非常自信并为正确标记分配了高概率（接近1），则$\log(\hat{y}_{t,c})$接近0，损失很小。
   - 如果模型为正确标记分配了低概率（接近0），则$\log(\hat{y}_{t,c})$是一个大的负数，并且损失$- \log(\hat{y}_{t,c})$变得很大。

### 跨序列和批次的损失汇总

我们为目标序列中的*每个*位置（填充位置除外，见下文）计算此交叉熵损失。为了得到单个序列的总损失，我们通常会平均该序列中所有有效（非填充）位置的损失值。

最后，由于我们使用小批次进行训练，整个批次的损失通常是该批次中所有序列损失的平均值。这个最终的平均批次损失是优化算法（例如Adam）通过反向传播 (backpropagation)更新模型参数 (parameter)时尝试最小化的值。

### 处理填充

还记得我们为了使批次中的序列长度一致而添加的那些填充标记 (token)吗？我们不希望模型因为在这些填充位置上的预测而受到惩罚。它们只是批处理过程的产物，不代表实际内容。

因此，在计算损失时，我们需要确保这些填充位置的损失贡献被忽略。大多数深度学习 (deep learning)框架都提供了轻松处理此问题的方法：

1. **使用掩码：** 我们可以使用注意力掩码（或创建一个特定的损失掩码）来识别填充位置。然后修改损失计算，在平均之前将每个位置的损失乘以掩码值（填充为0，实际标记为1），从而有效地将填充部分的损失清零。
2. **内置功能：** 许多损失函数 (loss function)实现（例如PyTorch中的`torch.nn.CrossEntropyLoss`或TensorFlow中的`tf.keras.losses.SparseCategoricalCrossentropy`）都包含参数 (parameter)（例如`ignore_index`或通过使用掩码张量），这些参数允许您指定填充标记的索引，从而自动将其从损失计算中排除。

### 关于标签平滑的说明

虽然标准交叉熵工作良好，但有时用它训练的模型可能会对预测过于自信，尤其是在大型数据集上。它们可能会为预测的标记 (token)分配一个非常接近1.0的概率。这有时会损害泛化能力。

在Transformer中与交叉熵损失一同使用的常用正则化 (regularization)技术是**标签平滑**。标签平滑不是要求模型以100%的概率预测目标标记（使用$[0, 0, 1, 0]$这样的硬性独热目标），而是稍微修改目标分布。它为真实标记分配一个稍低的概率（例如，$1 - \epsilon$），并将剩余的小概率质量$\epsilon$分配到词汇表 (vocabulary)中的所有其他标记上。

例如，当平滑因子$\epsilon = 0.1$时，正确类别的目标分布可能看起来更像$[0.0001, 0.0001, 0.9, 0.0001, ...]$。这鼓励模型稍微降低确定性，这通常能带来更好的性能和稳定性。我们在这里不会讨论实现细节，但这是一种在训练大型Transformer模型时值得了解的有用技术。

总之，交叉熵损失是训练Transformer进行序列生成任务的得力工具。它有效地衡量模型在每个位置预测下一个标记的准确程度，为学习提供梯度信号，并且可以轻松适应以处理填充并纳入标签平滑等技术。理解这种损失的计算方式对于解释训练过程和诊断潜在问题很重要。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer模型的奠基性论文，详细阐述了其架构、训练目标（包括交叉熵损失）以及标签平滑的应用。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本权威教材，全面解释了交叉熵损失、其数学基础以及在各种深度学习模型中的应用。
- [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — PyTorch Contributors (2023)
  Publisher: PyTorch Foundation
  PyTorch交叉熵损失函数（CrossEntropyLoss）的官方文档，详细介绍了诸如`ignore_index`等参数，用于处理序列任务中的填充。
