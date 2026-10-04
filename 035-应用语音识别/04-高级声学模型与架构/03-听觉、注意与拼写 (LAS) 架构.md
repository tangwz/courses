# 听觉、注意与拼写 (LAS) 架构

来源：[原文](https://apxml.com/zh/courses/applied-speech-recognition/chapter-4-advanced-acoustic-models/listen-attend-spell-architecture)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然连接主义时间分类 (CTC) 提供了一种无需明确对齐 (alignment)即可训练声学模型的方法，但其输出预测之间条件独立的核心假设是一个主要限制。语音本身有结构；“apple”中“p”的发音受到前面的“a”和后面的“l”的影响。为了捕捉这些关联，我们需要一个直接对输出序列进行建模的架构。听觉、注意与拼写 (LAS) 模型正是一个开创性的端到端架构，实现了这一点。

由谷歌研究人员开发，LAS 将语音识别视为一个序列到序列 (Seq2Seq) 问题，这与机器翻译非常相似。它直接将音频特征序列转换为字符或单词序列。其名称本身简洁地描述了三个主要组成部分：“听者”负责处理音频，“注意力”机制负责关注音频的相关部分，“拼写者”负责生成文本记录。

### LAS 的三个组成部分

LAS 架构由编码器、解码器和一个连接它们的注意力机制 (attention mechanism)精巧构成。

1. **听者 (编码器):** 该组件的作用是接收输入音频特征，例如对数梅尔频谱图，并将其转换为更高层级的表示。它“听取”整个语音片段。
2. **注意力机制:** 这是听者和拼写者之间的桥梁。在生成输出的每一步，它决定编码音频的哪个部分最相关。
3. **拼写者 (解码器):** 该组件接收编码表示，并在注意力机制的引导下，一次生成一个标记 (token)（例如，一个字符或词块）的输出文本记录。它“拼写出”文本记录。

接下来，让我们详细了解一下这些部分。

### 听者：一个声学编码器

听者充当声学模型的编码器。其主要目标是学习输入语音的丰富、紧凑表示。通常，听者实现为循环神经网络 (neural network) (RNN)的堆叠，最常见的是双向长短期记忆网络 (LSTM) (BLSTM)。

听者的输入是特征向量 (vector)序列，$X = (x_1, x_2, ..., x_T)$，其中 $T$ 是音频中的时间步数。BLSTM 处理这个序列，并生成一组高级特征向量或编码器隐藏状态，$H = (h_1, h_2, ..., h_T')$。

由于 LSTM 处理序列，它们非常适合捕捉语音中的时间模式。使用*双向*LSTM 尤其有效，因为它在前向和后向两个方向上处理音频，使得每个隐藏状态 $h_i$ 能够包含关于整个语音片段的信息，而不仅仅是过去的信息。

通常，听者包含一个金字塔结构 (pBLSTM)，其中连续的时间步在更高层中合并。这逐步缩短了序列的时间长度 ($T' < T$)，从而创建了更紧凑的表示，并降低了注意力机制 (attention mechanism)的计算负担。

> 听者通过堆叠的 pBLSTM 层处理输入特征，以生成高级隐藏状态。

### 拼写者：一个注意力解码器

拼写者是一个自回归 (autoregressive)解码器，这意味着它一次生成一个标记 (token)的输出序列，并且每个新的预测都取决于先前生成的标记。LAS 在此处明确建模了 CTC 所忽略的输出关联。

拼写者通常是一个单向 LSTM 或 GRU。在每个解码步骤 $i$，它执行两个动作：

1. 它使用注意力机制 (attention mechanism)计算一个**上下文 (context)向量 (vector)**，$c_i$。该向量是听者输出状态的汇总，根据每个状态与生成当前输出标记 $y_i$ 的相关性进行加权。
2. 它将其自身的先前隐藏状态 ($s_{i-1}$)、先前生成的标记 ($y_{i-1}$) 和新的上下文向量 ($c_i$) 作为其 RNN 单元的输入。然后它更新其隐藏状态至 $s_i$，并预测当前标记的概率分布，$P(y_i | y_{<i}, X)$。

该过程从一个特殊的序列开始标记 `<SOS>` 开始，并持续进行，直到生成序列结束标记 `<EOS>` 为止。

注意力机制是该过程的核心。对于每个解码步骤 $i$，它将当前解码器状态 $s_{i-1}$ 与所有编码器隐藏状态 $H = (h_1, h_2, ..., h_T')$ 进行比较。这种比较产生一组注意力分数或权重 (weight)，然后用于计算编码器状态的加权平均。这个平均值就是上下文向量 $c_i$。


$$
\alpha_{ij} = \frac{\exp(\text{score}(s_{i-1}, h_j))}{\sum_{k=1}^{T'} \exp(\text{score}(s_{i-1}, h_k))}
$$


$$
c_i = \sum_{j=1}^{T'} \alpha_{ij} h_j
$$


这使得拼写者能够“关注”当前正在转录音素对应的特定音频片段。例如，当生成“apple”中的字符“p”时，注意力权重对于原始音频中 /p/ 音对应的编码器隐藏状态将是最高的。

### 完整的 LAS 架构

综合来看，LAS 模型在推理 (inference)过程中以循环方式运行。

1. **听:** 整个音频输入一次性通过听者（编码器），以获取完整的隐藏状态集合 $H$。
2. **注意与拼写:** 拼写者（解码器）开始其生成过程。
   - 在步骤 $i$，解码器的状态 $s_{i-1}$ 用于查询编码器状态 $H$。
   - 注意力机制 (attention mechanism)计算权重 (weight)并生成一个上下文 (context)向量 (vector) $c_i$。
   - 解码器 RNN 接收 $c_i$ 和前一个标记 (token) $y_{i-1}$ 来预测下一个标记 $y_i$。
   - 这个过程重复，直到生成一个 `<EOS>` 标记。

> 完整的 LAS 架构。听者编码音频，对于每个输出步骤，拼写者使用注意力机制从编码器状态生成一个上下文向量，以预测下一个字符。

### 训练与局限

LAS 模型使用预测标记 (token)概率与真实文本记录之间的标准交叉熵损失进行端到端训练。训练期间常用一种名为**教师强制**的技术，即解码器在每一步都被输入来自真实文本记录的正确前一个标记，而非其自身（可能不正确的）预测。这有助于稳定训练，并帮助模型更快地学习。对于推理 (inference)，由于真实数据不可用，会使用像**束搜索**这样的算法来考虑多个候选转录，并找到最有可能的那个。

虽然 LAS 是一大进步，但它存在局限。拼写者严格从左到右的自回归 (autoregressive)性质在推理时可能速度较慢，并且在处理非常长的语音片段时可能表现不佳，此时注意力机制 (attention mechanism)可能失去聚焦。这些挑战促使人们转向像 Transformer 这样的完全基于注意力的模型，我们将在后面讨论。

## 参考资料

- [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211) — William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals (2015)
  Journal: arXiv preprint arXiv:1508.01211; DOI: [10.48550/arXiv.1508.01211](https://doi.org/10.48550/arXiv.1508.01211)
  介绍了用于端到端语音识别的Listen, Attend, and Spell (LAS)模型，详细阐述了其带有注意力的编解码器架构。
- [Sequence to Sequence Learning with Neural Networks](https://neurips.cc/paper/2014/file/a14ac55a4f27472c112fd74898170a59-Paper.pdf) — Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014)
  Journal: Advances in Neural Information Processing Systems 27 (NIPS 2014); Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS); Volume: 27; Pages: 3104-3112; DOI: [10.48550/arXiv.1409.3215](https://doi.org/10.48550/arXiv.1409.3215)
  提出了基础的序列到序列学习框架，该框架是LAS模型处理自动语音识别方法的基础。
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (2014)
  Journal: International Conference on Learning Representations (ICLR 2015); DOI: [10.48550/arXiv.1409.0473](https://doi.org/10.48550/arXiv.1409.0473)
  介绍了注意力机制，这是一个核心组件，允许解码器有选择地关注输入序列的特定部分，并被LAS模型采用。
- [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://dl.acm.org/doi/10.1145/1150447.1150493) — Alex Graves, Santiago Fernández, Faustino Gomez, Jürgen Schmidhuber (2006)
  Journal: Proceedings of the 23rd International Conference on Machine Learning (ICML '06); Publisher: Association for Computing Machinery; Pages: 369-376; DOI: [10.1145/1150447.1150493](https://doi.org/10.1145/1150447.1150493)
  描述了连接时序分类 (CTC)，通过强调早期的序列建模方法，为LAS的发展提供了重要背景。

---

[上一节](02-%E8%87%AA%E5%8A%A8%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB%20%28ASR%29%20%E4%B8%AD%E7%9A%84%E5%BA%8F%E5%88%97%E5%88%B0%E5%BA%8F%E5%88%97%20%28Seq2Seq%29%20%E6%A8%A1%E5%9E%8B.md) · [下一节](04-%E8%87%AA%E5%8A%A8%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB%E4%B8%AD%E7%9A%84%20Transformer%20%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0.md)
