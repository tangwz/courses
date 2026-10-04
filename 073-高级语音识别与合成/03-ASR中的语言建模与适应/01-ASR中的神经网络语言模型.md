# ASR中的神经网络语言模型

来源：[原文](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-3-language-modeling-adaptation-asr/neural-lm-asr)

[返回章节目录](README.md) · [返回课程目录](../README.md)

声学模型给出给定词序列$W$时音频特征$X$的概率$P(X|W)$，而语言模型（LM）给出词序列$W$本身的先验概率$P(W)$。将两者结合有助于ASR系统识别出不仅声学上合理，而且语言上也合乎情理的序列。传统语言模型，如n-gram模型，通过查看前置词的短固定长度历史来估算$P(W)$（例如，三元模型考虑前两个词）。尽管计算效率高，并且在一定程度上有效，但n-gram模型存在两个主要局限：

1. **数据稀疏性：** 估算更长的n-gram（如四元或五元模型）的概率变得困难，因为许多序列在训练文本语料库中从未出现过，导致概率为零。平滑技术部分缓解了这一问题，但并未完全解决。
2. **语境受限：** n-gram模型无法捕获固定窗口大小“n”之外的依赖关系。真实语言通常包含长程依赖关系，其中一个词的选择可能依赖于前几个词甚至几句话的语境。

神经网络 (neural network)语言模型（NLMs）通过使用词的连续表示（词向量 (vector)）以及神经网络结构来建模在给定可能长得多的历史条件下下一个词的概率分布，从而解决了这些局限。

### 循环神经网络 (neural network) (RNN)语言模型（RNN-LMs）

循环神经网络天然适合语言建模等序列建模任务。RNN-LM逐词处理输入序列，并维持一个隐藏状态向量 (vector)，该向量概括了目前所见的信息。

在每个时间步$t$，RNN将当前词$w_t$的词向量和前一个隐藏状态$h_{t-1}$作为输入，并计算新的隐藏状态$h_t$。这个隐藏状态随后通常会通过一个线性层，再通过softmax函数，以生成词表中下一个词$w_{t+1}$的概率分布。

$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$
$P(w_{t+1} | w_1, ..., w_t) = \text{softmax}(W_{hy} h_t + b_y)$

$x_t$是词$w_t$的词向量，$f$是非线性激活函数 (activation function)（如tanh或ReLU），且$W_{hh}, W_{xh}, W_{hy}, b_h, b_y$是可学习的权重 (weight)矩阵和偏置 (bias)。

变体，如长短期记忆（LSTM）和门控循环单元（GRU），常被用于替代简单的RNNs，因为它们包含了门控机制，有助于减轻梯度消失问题，使它们能更有效地学习长程依赖关系。

> 一个用于语言建模的基本循环神经网络（RNN）结构。隐藏状态`h(t)`（蓝色）根据当前输入词`w(t)`（青色）和前一个隐藏状态`h(t-1)`计算得出。该状态随后被用于预测后续词`P(w(t+1)|...)`（橙色）的概率分布。

RNN-LM相比n-gram模型具有显著优势：

- **连续表示：** 词向量使模型能够理解词之间的语义相似性。
- **更长语境：** 循环状态理论上允许模型运用整个前置语境，尽管实际使用中仍有局限。
- **更好的泛化能力：** 由于更平滑的概率估算，它们在未见序列上的表现通常优于n-gram模型。

### Transformer 语言模型

近年来，完全基于自注意力 (self-attention)机制 (attention mechanism)的Transformer架构在许多自然语言处理任务（包括语言建模）中占据主导地位。与RNNs顺序处理序列不同，Transformer可以在训练期间并行处理语境中的所有词。

自注意力机制使模型在预测下一个词时，能够衡量语境中不同词的重要性，而无需考虑它们之间的距离。这使得Transformer特别善于捕获非常长程的依赖关系。典型的Transformer语言模型使用堆叠的解码器层，每个层应用多头自注意力，然后是前馈网络。

Transformer语言模型在困惑度（语言模型性能的常用衡量指标）方面常取得顶尖成果，并在文本生成方面展现出显著能力。

### 将NLMs整合到ASR解码中

无论使用RNN还是Transformer架构，NLM在ASR中的主要作用是在解码过程（通常是束搜索）中提供$P(W)$得分。ASR系统寻找词序列$W^*$，使其最大化声学模型得分和语言模型得分的组合：

$W^* = \arg \max_W P(X|W) P(W)^\lambda$
或，更常见的是在对数空间中：
$W^* = \arg \max_W \log P(X|W) + \lambda \log P(W)$

$\lambda$是语言模型权重 (weight)（有时称为LM比例因子），它是一个控制LM相对于声学模型影响的超参数 (parameter) (hyperparameter)。它通常在开发集上进行调整。

在束搜索期间：

1. 假设（部分词 (tokenization)序列）被保留在束中。
2. 为了用潜在的下一个词`w_next`扩展假设`W_prefix`，会查询NLM来计算$P(w_{\text{next}} | W_{\text{prefix}})$。
3. 这个概率（或其对数）会乘以权重$\lambda$，然后加到扩展假设的当前得分上。
4. 具有最高组合得分的假设被保留在束中，以供下一步使用。

对束中每个假设的每个可能下一个词查询大型NLM可能计算成本很高，相比查询n-gram模型（后者通常涉及预编译结构中更简单的查找）。将NLMs整合到ASR系统中时，这种计算成本是一个重要的考量，特别是对于实时应用。诸如缓存常见前缀的NLM状态或使用专用硬件等方法可以帮助缓解此问题。

结合这些得分的方法，如浅层结合和深层结合，将在下一节讨论。这些方法明确了NLM得分*如何*以及*在哪个阶段*被纳入整个ASR解码图或搜索过程。使用NLMs通常会使词错误率（WER）相比传统n-gram模型显著降低，特别是对于语言复杂或句子较长的任务，这使得在许多应用中增加的计算需求变得合理。

## 参考资料

- [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin (2003)
  Journal: Journal of Machine Learning Research; Volume: 3; Pages: 1137-1165; DOI: [10.1162/jmlr.2003.3.6.1137](https://doi.org/10.1162/jmlr.2003.3.6.1137)
  本文介绍了神经网络语言模型的概念，利用分布式词表示和神经网络进行概率估计。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这篇基础论文介绍了LSTM架构，它通过解决梯度消失问题改进了RNN，并实现了更长序列的学习。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇论文介绍了完全基于注意力机制的Transformer架构，它彻底改变了序列建模和语言模型。
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson
  一本综合性教材，涵盖了传统和神经网络语言模型及其在语音识别中的应用，并包含了Transformer等最新进展。（第三版草稿）

---

[上一节](../02-%E9%AB%98%E7%BA%A7ASR%E5%A3%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AB%AF%E5%88%B0%E7%AB%AFASR%E6%A8%A1%E5%9E%8B.md) · [下一节](02-%E6%B5%85%E5%B1%82%E9%9B%86%E6%88%90%E4%B8%8E%E6%B7%B1%E5%BA%A6%E9%9B%86%E6%88%90.md)
