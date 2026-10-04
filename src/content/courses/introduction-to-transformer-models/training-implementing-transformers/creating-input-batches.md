---
course: "introduction-to-transformer-models"
chapter: "training-implementing-transformers"
lesson: "creating-input-batches"
sourceId: 3023
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-4-training-implementing-transformers/creating-input-batches"
title: "构建输入批次"
description: "说明如何准备分词后的序列、注意力掩码和填充，以进行批次训练。"
order: 2
plots: []
sourceHash: "9720d396e8a4dc3b323466433ddf44c40b9e286f147bc880c0bdd1fa82daca7e"
sourceCorrections: []
---

为了在训练时将分词 (tokenization)后的输入文本（表示为数字ID序列）馈送到Transformer模型，需要将这些序列组织成批次。仅仅将原始的词元 (token)ID序列分组是不够的，主要是因为批次中的序列长度几乎总是不相同。神经网络 (neural network)，特别是那些在PyTorch或TensorFlow等框架中实现的，通常要求输入是形状统一的张量。因此，填充和注意力掩码变得非常重要。

### 填充的必要性

设想您希望使用包含以下两个分词 (tokenization)句子的批次来训练模型：

- 句子 A: `[101, 7592, 2077, 2003, 102]` (5 个词元 (token))
- 句子 B: `[101, 2023, 2003, 1037, 4077, 3681, 102]` (7 个词元)

您不能直接将这些堆叠成一个单一的矩形张量。为解决此问题，我们使用填充。我们为批次选择一个最大序列长度（通常是该特定批次中最长序列的长度），并在较短序列的末尾添加特殊的“填充”词元，直到它们都达到这个最大长度。

大多数分词器 (tokenizer)会为填充保留一个特定的ID，通常是 `0`。使用这个填充词元（假设其ID为 `0`），并填充到句子B的长度（7个词元），我们的批次将如下所示：

- 句子 A (已填充): `[101, 7592, 2077, 2003, 102, 0, 0]`
- 句子 B (已填充): `[101, 2023, 2003, 1037, 4077, 3681, 102]`

现在，这两个序列可以组合成一个形状为 `(batch_size, sequence_length)` 的单一张量，本例中为 `(2, 7)`。

### 注意力掩码：忽略填充

填充解决了结构问题，但带来了计算问题。自注意力 (self-attention)机制 (attention mechanism)，如第2章所述，根据序列中所有词元 (token)之间的相互影响计算分数。我们不希望模型关注这些人工填充词元，因为它们不包含有意义的信息，关注它们可能会对性能产生负面影响。

此时，**注意力掩码**就发挥作用了。它是一个二进制张量，与输入ID张量形状相同（或在注意力计算期间兼容广播的维度）。该掩码指示模型*应该*关注哪些词元以及*应该忽略*哪些词元。

常用的约定是使用：

- `1` 表示真实词元（关注）。
- `0` 表示填充词元（忽略）。

对于我们的示例批次，对应的注意力掩码将是：

- 掩码 A: `[1, 1, 1, 1, 1, 0, 0]`
- 掩码 B: `[1, 1, 1, 1, 1, 1, 1]`

这些掩码通常在注意力机制中，在 Softmax 步骤*之前*应用。通过向与填充位置（掩码为 `0` 的地方）对应的注意力分数添加一个很大的负数（例如 -10000 或负无穷），随后的 Softmax 操作会有效地将这些位置的概率分配为接近零。这确保了填充词元不参与注意力头计算的上下文 (context)向量 (vector)的构成。

### 训练批次的组成部分

综合来看，用于训练标准序列到序列Transformer（例如用于机器翻译）的典型输入批次通常包含几个组成部分，通常以字典或类似结构组织：

1. **`input_ids`**: 源序列（例如，原始语言中的句子）的词元 (token)ID，在批次内部填充到统一长度。形状: `(batch_size, source_sequence_length)`。
2. **`attention_mask`**: 源序列的注意力掩码，指示填充词元。形状: `(batch_size, source_sequence_length)`。
3. **`decoder_input_ids`**: 目标序列（例如，翻译后的句子）的词元ID，也经过填充。重要的是，在训练时，这通常是目标序列右移（通常以一个特殊的序列开始词元开头）并截断后的结果。这为解码器在每个步骤提供了输入。形状: `(batch_size, target_sequence_length)`。
4. **`decoder_attention_mask`**: 目标序列的注意力掩码。这个掩码有双重目的：它屏蔽填充词元，*并*实现解码器中因果自注意力 (self-attention)所需的先行掩码。先行机制防止解码器在训练期间关注未来的词元，确保它只使用之前的词元来预测下一个词元。形状: `(batch_size, target_sequence_length)`。
5. **`labels`**: 模型应该预测的实际目标序列ID，用于计算损失函数 (loss function)。这通常是 `decoder_input_ids` 左移后的结果，不包含初始的序列开始词元，并且可能在损失计算本身中屏蔽填充词元。形状: `(batch_size, target_sequence_length)`。

这里是这些组成部分对于一个翻译任务“Hello world” -> “Bonjour le monde”的单个训练示例（批次大小为1）的简化文本表示，假设进行了简化分词 (tokenization)、填充到长度5以及特定的词元ID：

```text
源: "Hello world" -> 词元: [101, 87, 99, 102]
目标: "Bonjour le monde" -> 词元: [201, 150, 160, 170, 202]

填充词元ID: 0
最大长度: 5

input_ids:             [101,  87,  99, 102,   0]
attention_mask:        [  1,   1,   1,   1,   0]

# 解码器输入通常右移（起始词元 201）
decoder_input_ids:     [201, 150, 160, 170, 202] # 假设此处目标长度恰好为 5

# 解码器掩码需要填充 AND 先行
# (简化：此处仅显示填充掩码)
decoder_attention_mask:[  1,   1,   1,   1,   1]

# 标签是模型应预测的目标词元
labels:                [150, 160, 170, 202,   0] # 用于损失计算的填充标签
```

> 单个示例的简化批次组成部分。请注意，实际实现中的 `decoder_attention_mask` 还会包含因果（先行）掩码。`labels` 通常是 `decoder_input_ids` 左移并带有填充的结果。

### 框架支持

现代深度学习 (deep learning)库提供了有用的工具来管理这个过程。例如，Hugging Face `transformers` 等库中的分词 (tokenization)器 (tokenizer)通常在编码文本时自动生成注意力掩码。此外，数据加载工具（PyTorch 中的 `DataLoader`，TensorFlow 中的 `tf.data`）通常包含整理函数，可以动态地将每个批次内的序列填充到该特定批次所需的最大长度，这比将所有序列填充到固定的全局最大长度更有效率。

了解这些批次是如何构建的，包括填充和注意力掩码的作用，对于正确准备数据和有效训练Transformer模型来说是根本性的。有了格式正确的批次，我们现在可以进行下一步，定义将指导模型学习过程的损失函数 (loss function)。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  描述了Transformer架构，包括注意力机制以及在编码器和解码器中使用掩码来管理序列信息。
- [Preprocessing data](https://huggingface.co/docs/transformers/preprocessing) — Hugging Face (2024)
  提供了使用Hugging Face `transformers` 库为Transformer模型进行分词、填充和注意力掩码创建的实用指南。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2023)
  Publisher: Stanford University
  一本全面的教材，涵盖了核心NLP概念，包括序列模型的数据准备技术以及Transformer架构的细节，这些架构隐含地依赖于填充和掩码。
- [Writing Custom Datasets, DataLoaders and Transforms](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) — PyTorch Team (2024)
  解释了PyTorch中`Dataset`和`DataLoader`的基本概念，这对于包括Transformer在内的深度学习模型的数据准备流程中的批处理和自定义填充逻辑至关重要。
