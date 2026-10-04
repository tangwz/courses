---
course: "rnns-and-sequence-modeling"
chapter: "sequence-modeling-applications"
lesson: "text-generation-models"
sourceId: 2641
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-9-sequence-modeling-applications/text-generation-models"
title: "文本生成模型"
description: "详细说明如何设置模型以预测序列中的下一个元素来生成文本。"
order: 6
plots: []
sourceHash: "0a9336dd2186e160a345d431f0880241b815419e6532233f11493c2341eee8c1"
sourceCorrections: []
---

建立在序列生成的一般思想之上，文本生成是循环神经网络 (neural network) (RNN)的一种常见且直观的应用。这里的目标是训练一个模型，使其能够根据从示例文本语料库中学习到的模式，逐字符或逐词地生成类似人类的文本。应用范围从自动补全句子、生成代码，到协助创意写作和提供聊天机器人回复。

### 核心原理：预测下一个词元 (token)

循环神经网络 (neural network)（RNN）进行文本生成的基本原理是：预测给定前序元素的序列中的下一个元素（一个字符或一个词，常称为“词元”）。RNN的隐藏状态充当记忆，概括之前的序列，以便为下一步的预测提供信息。

设想你有一个序列“The quick brown fox jumps”。当模型接收到这个序列时，其任务是预测下一个最有可能的词元，例如“over”。我们通过向模型提供海量文本来训练它，并在每个位置要求它预测训练数据中实际出现的下一个词元。

### 文本生成模型架构

文本生成模型的常见设置包含以下层：

1. **输入层：** 接收词元 (token)ID序列（表示字符或词的整数）。
2. **嵌入 (embedding)层：** 将这些整数ID转换为稠密向量 (vector)表示。此层学习词汇表 (vocabulary)中每个词元的有意义表示（如第8章所述）。
3. **循环层（LSTM/GRU）：** 逐步处理嵌入序列，并更新其隐藏状态。此层捕获文本中的时间依赖性。在此使用LSTM或GRU是标准做法，以处理可能很长的序列并减轻梯度消失问题。如果堆叠循环层或后续层需要完整的序列输出，请设置 `return_sequences=True`；但对于预测下一个词元的基本生成任务，通常在处理完输入序列后只需要最终输出（尽管框架在逐步预测时通常会隐式处理）。
4. **全连接输出层：** 一个全连接层，其单元数量等于词汇量大小（唯一字符或词的数量）。它接收RNN层在每个时间步的输出。
5. **Softmax激活：** 应用于输出层，以在整个词汇表上生成概率分布。输出向量中的每个元素表示相应词元是序列中*下一个*词元的概率。

模型训练旨在最小化损失函数 (loss function)，通常是分类交叉熵，该函数衡量预测概率分布与训练数据中实际的下一个词元（表示为独热编码向量）之间的差异。

例如，如果输入序列由词元ID `[10, 34, 5]`（“the cat sat”）表示，词汇量为1000，RNN会处理这些输入。如果下一个实际词是“on”（ID 12），模型在处理“sat”后的输出理想情况下应该是一个概率分布，其中第12个元素接近1，而所有其他元素接近0。损失函数促使模型学习能达到此目的的权重 (weight)。

### 生成新文本：采样过程

模型训练完成后，我们可使用它来生成新颖的文本。这通常是一个迭代过程：

1. **提供起始文本：** 以一段初始文本序列（“起始文本”或“提示”）开始，例如“The weather today is”。
2. **预处理：** 将起始序列转换为模型所需的数字格式（词元 (token)ID）。
3. **预测：** 将起始序列输入到训练好的模型中。模型输出下一个词元的词汇表 (vocabulary)概率分布。
4. **采样：** 根据预测概率选择下一个词元。这是重要一步，可采用不同策略（下文讨论）。
5. **添加：** 将选定的词元添加到当前序列的末尾。
6. **迭代：** 使用新扩展的序列作为下一个预测步骤的输入。重复步骤3-5，逐词元生成文本，直到达到期望长度或生成特殊的序列结束词元。

> 使用已训练的RNN模型进行迭代文本生成过程的视图。

### 采样策略

我们如何选择下一个词元 (token)（上述步骤4）会显著影响生成文本的质量和风格：

- **贪婪搜索：** 在每一步只选择概率最高的词元。这种方法是确定性的，常导致重复或可预测的文本。
- **温度采样：** 这是一种控制输出随机性的常用技术。在应用softmax或采样之前，对数（softmax之前全连接层的原始输出）会除以一个`温度`值（$T$）。
  - 低温度（$T < 1$，例如0.5）使分布“更尖锐”，增加了最可能候选词的概率。当$T \to 0$时，这会接近贪婪搜索，生成更可预测、更集中的文本。
  - 温度为$T = 1$时使用原始概率。
  - 高温度（$T > 1$，例如1.2）使分布变得扁平，使得不太可能的词元概率增加。这会生成更出人意料、随机且可能富有创意（或无意义）的文本。
    调整后的概率 $p_i'$ 是根据原始概率 $p_i$（来自对数）计算的，如下所示：

  
  $$
  p_i' = \frac{\exp(\log(p_i) / T)}{\sum_j \exp(\log(p_j) / T)}
  $$
  
  然后我们从这个修改后的分布 $p'$ 中进行采样。调整温度是平衡连贯性和创造性的一种实用方法。
- **Top-k采样：** 在每一步将采样池限制为$k$个最有可能的词元，然后重新分配它们的概率质量并采样。这避免了选择极不可能的词元，同时仍允许一定的变体。
- **核（Top-p）采样：** 选择累积概率超过阈值$p$的最小词元集合，然后重新分配它们的概率质量并采样。这会根据概率分布的形状调整采样池的大小。

### 字符级模型与词级模型

你可以构建不同粒度的文本生成模型：

- **字符级：**
  - **词汇表 (vocabulary)：** 由单个字符（字母、标点符号、空格等）组成。通常较小（例如，< 100）。
  - **优点：** 可以创造新词，自然处理拼写错误或罕见词，捕获细粒度的风格元素（如标点符号模式）。
  - **缺点：** 需要建模更长的序列来捕获含义，生成有意义的文本计算量更大，有时会生成语法无效的词。要求RNN从头开始学习词结构。
- **词级：**
  - **词汇表：** 由训练语料库中的独特词组成。可达数万或数十万。通常需要处理未知词（使用`<UNK>`词元 (token)）或子词 (subword)单元（如BPE或WordPiece）。
  - **优点：** 更直接地建模语义，通常需要更短的序列来生成连贯文本，每个语义单元的计算强度较低。
  - **缺点：** 不能生成词汇表之外的词（除非使用子词技术），词汇量可能变得非常大，对词汇表的构建方式敏感。

字符级与词级（或子词级）生成之间的选择取决于具体应用、训练数据的性质和计算资源。

设置和训练这些模型需要仔细的数据准备（词元化，通常使用文本上的滑动窗口创建输入/目标对），并采用深度学习 (deep learning)框架提供的RNN实现（LSTM、GRU层），如前几章所述。生成步骤随后在推理 (inference)循环中使用训练好的模型，并采用选定的采样策略。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆 (LSTM) 架构，这对于处理文本等序列数据中的长距离依赖关系至关重要。
- [Sequence to Sequence Learning with Neural Networks](https://doi.org/10.48550/arXiv.1409.3215) — Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 27); Publisher: Curran Associates; Pages: 3104-3112; DOI: [10.48550/arXiv.1409.3215](https://doi.org/10.48550/arXiv.1409.3215)
  一篇开创性论文，展示了 LSTM 在序列到序列学习中的有效性，这是许多高级序列生成任务的核心方法。
- [The Curious Case of Neural Text Degeneration](https://openreview.net/forum?id=Byfnk0olH) — Ari Holtzman, Jan Philipp Fritsch, Stefan Riezler, and Antoine Bosselut (2019)
  Journal: International Conference on Learning Representations (ICLR); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 119; DOI: [10.34789/BS566H](https://doi.org/10.34789/BS566H)
  讨论了贪婪搜索的局限性，并介绍了核采样（Top-p 采样）作为生成多样化且连贯文本的有效方法。
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  一本关于自然语言处理的综合教科书，包括对循环神经网络、语言建模和文本生成技术的讨论。
