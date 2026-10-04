# 联结主义时序分类 (CTC) 损失

来源：[原文](https://apxml.com/zh/courses/applied-speech-recognition/chapter-3-acoustic-modeling-deep-learning/connectionist-temporal-classification-ctc)

[返回章节目录](README.md) · [返回课程目录](../README.md)

语音识别中的一个主要难题是，输入音频特征的长度与输出文本转录的长度不匹配。例如，一个800毫秒的音频片段可能会产生数千个特征向量 (vector)，但对应的转录可能只是一个像“OK Google”这样的短语。此外，没有明确的逐帧对齐 (alignment)来告诉模型音频的哪一部分对应哪个字符。标准的交叉熵损失函数 (loss function)需要输入和目标之间存在一对一的映射，因此不适用于此任务。

联结主义时序分类 (CTC) 损失旨在解决这个问题。它是一种损失函数，使得神经网络 (neural network)能够针对输入和输出之间对齐方式未知的序列到序列任务进行训练。

### 对齐 (alignment)问题详述

以一个声学模型为例，例如LSTM，它处理一个音频特征序列。对于每个时间步$t$的每个特征向量 (vector)，网络输出我们词汇表 (vocabulary)中所有可能字符的概率分布。如果我们的输入有$T$个时间步，我们将得到$T$个概率分布。

现在，假设我们的目标转录是单词“CAT”。它的长度为$N=3$。输入特征序列的长度为$T$，这几乎肯定远大于3。我们如何计算损失？这$T$个输出步中，哪些应该对应“C”，哪些对应“A”，哪些对应“T”？那些对应单词间的静默或元音的拉长发音的时间步又该如何处理呢？

CTC巧妙地解决了这个问题，它通过在词汇表中添加一个特殊的**空白符**来实现，通常表示为`<B>`或`\u03b5`。模型现在允许在任何时间步预测这个空白符。

### 从网络输出到转录

有了空白符，网络的输出序列现在与输入特征序列的长度相同，即$T$。然后我们可以定义一套简单的规则，将这个较长的输出序列（称为“路径”）转换成最终的、较短的转录。

折叠过程如下：

1. **合并重复字符：** 首先，将任何连续的相同字符序列折叠成一个字符。例如，`C-C-A-A-A-T` 变为 `C-A-T`。
2. **移除空白符：** 其次，从序列中移除所有空白符。例如，`C-<B>-A-A-<B>-T` 变为 `C-A-A-T`。

我们来结合这些规则。如果网络输出路径`<B>-C-C-<B>-A-T-T`，转换结果将是：

- **初始路径：** `<B>-C-C-<B>-A-T-T`
- **步骤1（合并重复字符）：** `<B>-C-<B>-A-T`
- **步骤2（移除空白符）：** `C-A-T`

空白符很重要，因为它分隔了本应不同但重复出现的字符。例如，在单词“HELLO”中，两个“L”在网络的路径中必须由一个空白符分隔，以避免被合并成一个“L”。一个有效路径可以是`H-E-L-<B>-L-O`。而像`H-E-L-L-O`这样的路径会被错误地折叠为“HELO”。

这种多对一映射意味着许多路径都可以得到相同的最终转录。以下图表说明了来自网络输出层的不同路径如何都能折叠成目标“CAT”。

> 不同的网络输出序列（路径）在应用CTC折叠规则后，能正确解码为目标转录“CAT”。

### CTC损失计算

CTC损失函数 (loss function)的核心思想是将所有能正确映射到目标转录的可能路径的概率相加。我们把目标转录表示为$Y$，输入特征序列表示为$X$。

1. **路径概率：** 在每个时间步$t$，网络（例如，LSTM）输出词汇表 (vocabulary)（字符加上空白符）中每个标记 (token)$k$的概率分布$p_t(k)$。长度为$T$的单个路径$\pi$的概率是每个时间步上标记概率的乘积：

   
   $$
   P(\pi | X) = \prod_{t=1}^{T} p_t(\pi_t)
   $$
   

   其中$\pi_t$是路径$\pi$在时间步$t$的标记。
2. **目标总概率：** 目标转录$Y$的总概率是可以折叠成$Y$的所有路径$\pi$的概率之和。

   
   $$
   P(Y | X) = \sum_{\pi \in \text{有效路径}} P(\pi | X)
   $$
   
3. **损失函数：** CTC损失是这个总概率的负对数似然。通过最小化此损失，我们最大化了网络输出解码为正确文本的路径的概率。

   
   $$
   \mathcal{L}_{CTC} = - \log P(Y | X)
   $$
   

计算所有可能路径的总和似乎计算上难以处理，因为路径数量随序列长度呈指数级增长。然而，这个求和可以通过一种称为前向-后向算法的动态规划方法高效计算，该算法类似于隐马尔可夫模型 (HMM) 中使用的算法。幸运的是，TensorFlow和PyTorch等深度学习 (deep learning)框架提供了优化实现，所以您只需调用函数，无需担心底层复杂性。

### 解码训练好的CTC模型

模型训练完成后，我们可以将其用于推断，将新的音频转换成文本。网络生成一个大小为$T \times (\text{字符数} + 1)$的概率矩阵。从该矩阵生成文本的过程称为解码。

最简单的解码方法是**贪心解码**（也称为最佳路径解码）。在每个时间步$t$，我们简单地选择概率最高的标记 (token)。这给了我们一个单一的、最可能的路径。然后我们使用CTC规则折叠这条路径。

例如，如果每个时间步的`argmax`产生路径`H-H-<B>-E-L-L-<B>-L-O-O`，贪心解码会将其折叠为：

- `H-H-<B>-E-L-L-<B>-L-O-O` -> `H-<B>-E-L-<B>-L-O`（合并重复字符）
- `H-<B>-E-L-<B>-L-O` -> `HELO`（移除空白符）

虽然速度快，但贪心解码不是最优的，因为最可能的路径可能不会产生最可能的转录。许多“好”路径的概率之和可能高于单个“最佳”路径的概率。更复杂的算法，例如我们将在第5章讨论的**束搜索**，通过在解码过程中考察更多高概率路径，可以取得更好的结果。

## 参考资料

- [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://dl.acm.org/doi/10.1145/1143844.1143891) — Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber (2006)
  Journal: Proceedings of the 23rd International Conference on Machine Learning (ICML '06); Publisher: Association for Computing Machinery; Pages: 369-376; DOI: [10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891)
  介绍连接时间分类（CTC）损失函数的原始论文，它实现了无需显式对齐的序列到序列学习。
- [tf.nn.ctc_loss](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss) — TensorFlow Developers (2024)
  TensorFlow CTC 损失函数实现的官方文档，提供实际使用细节。
- [torch.nn.CTCLoss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch CTC 损失函数实现的官方文档，详细说明其 API 和用法。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  一本全面的教科书，涵盖语音识别基础知识，包括 CTC 的详细解释。

---

[上一节](03-%E4%BD%BF%E7%94%A8LSTM%E5%92%8CGRU%E5%A4%84%E7%90%86%E5%BA%8F%E5%88%97%E9%9A%BE%E9%A2%98.md) · [下一节](05-%E5%AE%9E%E7%8E%B0%E5%9F%BA%E4%BA%8ECTC%E7%9A%84ASR%E6%A8%A1%E5%9E%8B.md)
