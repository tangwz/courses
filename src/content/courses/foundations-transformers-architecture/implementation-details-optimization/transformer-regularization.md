---
course: "foundations-transformers-architecture"
chapter: "implementation-details-optimization"
lesson: "transformer-regularization"
sourceId: 2063
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/transformer-regularization"
title: "正则化方法 (Dropout, 标签平滑)"
description: "在Transformer子层内应用Dropout，并使用标签平滑处理输出分布。"
order: 5
plots: ["plots/2063-0.json"]
sourceHash: "d21df262edb8d8d4b1e8f3e533365e89d4a99fbbebea2f805f5f267cd9e09bef"
sourceCorrections: []
---

训练大型Transformer模型时，管理过拟合 (overfitting)以及确保模型对未见过的数据有良好的泛化能力，是令人关注的问题。仅仅构建一个大型网络并在大量数据集上训练是不够的；需要技巧来防止模型仅仅记住训练样本。正则化 (regularization)方法是实现工具箱中获得稳定表现的必不可少的工具。两种广泛用于Transformer的技巧是Dropout和标签平滑。

### Dropout

Dropout是一种简单而有效的正则化 (regularization)技术，最初是为了对抗前馈神经网络 (neural network)中的过拟合 (overfitting)而提出的。其核心思想是在每次训练更新时，随机地将一部分神经元输出“置零”。这可以防止单元过度依赖于特定的其他单元，促使网络学习到更分散、更具弹性的表示。

在标准的Transformer架构中，Dropout应用于以下几个位置：

1. **在嵌入 (embedding)和位置编码 (positional encoding)相加之后：** 应用于合并的输入表示进入编码器或解码器堆栈之前。
2. **在每个子层内部：** 应用于多头注意力 (multi-head attention)操作之后（在残差连接和层归一化 (normalization)之前）以及位置前馈网络之后（同样，在残差连接和层归一化之前）。
3. **可选地，在注意力权重 (weight)上：** 有时，Dropout直接应用于注意力权重 $softmax(QK^T/\sqrt{d_k})$，然后才与 $V$ 相乘。这有时被称为注意力Dropout。

将单元输出置零的概率 $p_{drop}$ 是一个超参数 (parameter) (hyperparameter)。典型值范围为0.1到0.3，尽管最佳值取决于具体的模型大小、数据集和任务。在训练期间，Dropout**之前**那一层的输出会以 $p_{drop}$ 的概率随机置零。其余的输出通常会按 $1 / (1 - p_{drop})$ 的因子进行缩放，以保持输出的期望总和。这种缩放确保了下一层的期望输入在训练和推理 (inference)之间保持一致。

在推理或评估期间，Dropout被禁用。所有单元都被使用，并且不应用任何缩放（假设缩放已在训练期间完成）。这确保了给定输入的确定性输出。

考虑一个子层内的简化示例：

```python
# 示例（例如，使用PyTorch）
import torch
import torch.nn as nn

dropout_prob = 0.1
# 假设 'sublayer_output' 是MHA或FFN之后的张量
dropout_layer = nn.Dropout(p=dropout_prob)
# 'x' 是子层的输入（用于残差连接）
# 'layer_norm' 是归一化层

# 训练期间
# 在残差连接和归一化之前应用dropout
output_dropped = dropout_layer(sublayer_output)
normalized_output = layer_norm(x + output_dropped)

# 评估期间 (model.eval() 模式)
# Dropout层自动将输入直接传递，不进行修改
output_no_drop = dropout_layer(sublayer_output) # 表现为恒等映射
normalized_output = layer_norm(x + output_no_drop)
```

通过这种方式注入噪声，Dropout促使模型发展出冗余性，并防止神经元之间复杂的共同适应，从而提高了泛化表现。

### 标签平滑

标签平滑处理了过拟合 (overfitting)的另一个方面，它与模型对其预测的置信度相关。在分类任务中（例如在语言建模中预测下一个词元 (token)），模型通常使用交叉熵损失进行训练，并采用硬性、独热编码的目标标签。例如，如果正确的下一个词对应词汇表 (vocabulary)中大小为 $K$ 的索引5，则目标向量 (vector)为 `[0, 0, 0, 0, 1, 0, ..., 0]`。

使用这种硬性目标进行训练会促使模型将正确类别对应的logit推向正无穷，而将所有其他类别对应的logit推向负无穷，导致预测类别具有极高的置信度（概率接近1.0）。这种过度的置信度可能是有害的：

- 它使模型适应性变差。
- 如果模型即使给正确类别分配了很小的概率，但对错误类别却非常自信，这可能会在训练期间对其进行严厉惩罚。
- 它可能无法准确体现语言或数据中固有的真实不确定性。

标签平滑正则化 (regularization)（LSR）修改目标标签，以纳入少量的不确定性。我们不再要求模型将1.0的概率分配给正确类别，而是将一个小的概率质量 $\epsilon$（epsilon）均匀地分配给所有类别，包括不正确的类别。

原始的独热目标分布 $y_k$（即当真实类别为 $k=t$ 时 $y_k=1$，否则 $y_k=0$）被平滑分布 $y'_k$ 所取代：


$$
y'_{k} = y_{k} (1 - \epsilon) + \frac{\epsilon}{K}
$$


这里，$K$ 是类别的总数（例如，词汇表大小）。真实类别现在具有 $1 - \epsilon + \epsilon/K$ 的目标概率，而所有其他类别都具有 $\epsilon/K$ 的目标概率。

我们通过一个小的例子来说明这一点。假设我们有 $K=5$ 个类别，真实类别是索引2（基于0）。当 $\epsilon=0.1$ 时：

- **独热目标：** `[0.0, 0.0, 1.0, 0.0, 0.0]`
- **平滑目标：**
  - 索引2： $1 \times (1 - 0.1) + 0.1 / 5 = 0.9 + 0.02 = 0.92$
  - 其他索引（0, 1, 3, 4）： $0 \times (1 - 0.1) + 0.1 / 5 = 0.0 + 0.02 = 0.02$
  - 结果向量： `[0.02, 0.02, 0.92, 0.02, 0.02]` (注意：总和为1.0)



![独热编码 vs. 标签平滑 (K=5, ε=0.1)](plots/2063-0.json)



> 比较了5类别问题中独热目标向量和标签平滑目标向量（当 $\epsilon=0.1$ 时）。真实类别的概率质量被降低，并均匀地分配给所有类别。

在计算交叉熵损失时，模型现在会因对正确预测过于自信而受到惩罚，并被鼓励为其他合理输出分配小的非零概率。损失函数 (loss function)变为：


$$
L_{LS} = - \sum_{k=1}^{K} y'_{k} \log(p_k)
$$


这里 $p_k$ 是模型对类别 $k$ 预测的概率。这促使正确类别和不正确类别之间logit的差异是有限的，起到正则化作用。

$\epsilon$ 的常用值是0.1。研究表明，标签平滑通常能改善序列到序列任务中的困惑度和BLEU分数，并能带来更好的模型校准。

Dropout和标签平滑都是大型Transformer模型训练方案中的标准组成部分。它们与其他元素（如适当的权重 (weight)初始化、优化算法（AdamW）和学习率调度）相互配合，以稳定训练并提高最终模型在未见过数据上的表现。为 $p_{drop}$ 和 $\epsilon$ 选择合适的值通常需要根据验证集表现进行实验和调整。

## 参考资料

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Volume: 15; Pages: 1929-1958
  引入神经网络Dropout正则化技术的原始论文。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构的基础论文，详细阐述了Dropout和标签平滑的应用。
- [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) — Rafael Müller, Simon Kornblith, Geoffrey Hinton (2019)
  Journal: NeurIPS 2019; DOI: [10.48550/arXiv.1906.02629](https://doi.org/10.48550/arXiv.1906.02629)
  分析了标签平滑正则化，探讨其对模型校准和泛化的益处。
- [Dropout (nn.Dropout)](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) — PyTorch Authors (2024)
  Publisher: PyTorch
  PyTorch Dropout层（nn.Dropout）的官方文档，包含使用细节。
