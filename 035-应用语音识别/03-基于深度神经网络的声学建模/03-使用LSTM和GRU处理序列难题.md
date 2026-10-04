# 使用LSTM和GRU处理序列难题

来源：[原文](https://apxml.com/zh/courses/applied-speech-recognition/chapter-3-acoustic-modeling-deep-learning/addressing-challenges-with-lstms-grus)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然简单的循环神经网络 (neural network)（RNN）是为序列数据设计的，但在处理语音中长而变化的序列时，它们会遇到显著的限制。它们的主要缺点是难以在较长时间步中学习和保留信息，这个问题通常被称为处理长距离依赖。这个问题主要源于梯度消失问题，即随着序列变长，过去输入对当前输出的影响呈指数级减小。

对于ASR系统而言，这是一个严重不足。模型必须能够将相隔几秒的声音联系起来，以形成连贯的词或短语。例如，为了正确转录句子的末尾，模型可能需要回溯开头的信息。简单的RNN无法很好地完成这项任务。为了应对这些难题，我们转向更精巧的循环架构：长短期记忆（LSTM）和门控循环单元（GRU）网络。

### 长短期记忆 (LSTM)

LSTM被明确设计用于长时间记住信息。它们引入了一个`单元状态`，它就像一条信息传送带，沿着整个序列直线传输，只进行少量线性交互。这种结构使得信息更容易保持不变地流动。

LSTM的独特之处在于它们能够使用三个称为门的特殊组件来调节信息流。这些门是小型神经网络 (neural network)，它们会学习哪些信息需要添加到单元状态、哪些需要保留、哪些需要移除。

1. **遗忘门**: 这个门决定了从前一个单元状态中丢弃哪些信息。它会查看前一个隐藏状态 $h_{t-1}$ 和当前输入 $x_t$，并为前一个单元状态 $C_{t-1}$ 中的每个数值输出一个介于0和1之间的数。1表示“完全保留”，而0表示“完全丢弃”。在语音中，这可能意味着忘记单词之间短暂静音的声学特征。

   
   $$
   f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
   $$
   
2. **输入门**: 这个门决定了哪些新信息将被存储到单元状态中。它包含两部分：一个sigmoid层，用于决定我们要更新哪些值（即`输入门`）；以及一个`tanh`层，用于生成可能添加到状态中的新候选值向量 (vector) $\tilde{C}_t$。模型就是这样加入新的声音的，例如新音素的开始。

   
   $$
   i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
   $$
   
   
   $$
   \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
   $$
   
3. **输出门**: 这个门决定了网络的输出。输出将是单元状态的过滤版本。首先，我们运行一个sigmoid层来决定单元状态的哪些部分将被输出。然后，我们将单元状态通过`tanh`（将值推到-1到1之间）并乘以sigmoid门的输出，这样我们只输出我们决定输出的部分。

   
   $$
   o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
   $$
   
   
   $$
   h_t = o_t * \tanh(C_t)
   $$
   

新的单元状态 $C_t$ 是通过结合旧状态（乘以遗忘门）和新候选值（乘以输入门）来计算的。


$$
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
$$


这种门控机制允许LSTM选择性地记忆或遗忘信息，使其在建模语音复杂的时序动态方面非常有效。

> 一个LSTM单元。最上面的线表示单元状态，它在时间步之间传递信息。门（红色）控制信息如何添加到或从该状态中移除。

### 门控循环单元 (GRU)

门控循环单元，简称GRU，是LSTM的一种较新且略简单的替代方案。它将遗忘门和输入门合并为一个`更新门`，并融合了单元状态和隐藏状态。这使得模型在计算上更高效。

一个GRU单元有两个主要门：

1. **更新门 ($z_t$)**: 这个门的功能与LSTM的遗忘门和输入门相似。它决定了保留多少过去的信息（来自前一个隐藏状态），以及添加多少新信息。
2. **重置门 ($r_t$)**: 这个门决定了如何将新输入与前一个隐藏状态结合。它本质上决定了要遗忘多少过去的信息。

GRU的简化结构意味着它们比LSTM拥有更少的参数 (parameter)，这使得它们在训练时更快，并且在较小数据集上更不容易出现过拟合 (overfitting)。在许多ASR任务中，GRU已表现出与LSTM相当的性能。

> 一个GRU单元。它通过合并门控和融合单元状态与隐藏状态来简化LSTM设计，从而形成更精简的架构。

### 双向和堆叠架构

为了进一步增强模型理解上下文 (context)的能力，我们可以采用两种额外的策略：双向处理和堆叠。

**双向RNN**
在语音中，上下文不仅是历史性的，也是前瞻性的。一个词的读音或含义可能受其后续词的影响。例如，为了区分“I read the book”和“I will read the book”中的同音词，模型通过看到整个句子而获得优势。

双向RNN（Bi-LSTM或Bi-GRU）以两个方向处理输入序列。一个循环层从头到尾处理序列（前向传播），而第二个独立的层从尾到头处理序列（反向传播 (backpropagation)）。在每个时间步，两个层的输出会被拼接起来形成最终的表示。这使得模型能够获得序列中每个点的完整周边上下文信息。

**堆叠RNN**
就像其他类型的神经网络 (neural network)一样，我们可以通过将循环层堆叠起来增加模型的深度。在堆叠RNN中，第一层的输出序列成为第二层的输入序列，依此类推。

这种方法使得网络能够学习数据的分层表示。第一层可能学习检测基本的声学特征和音素。第二层可以学习将这些音素表示组合成音节或词段，后续层可以学习更高级别的语言结构。典型的ASR声学模型可能会使用2到6个堆叠的双向LSTM或GRU层。

通过使用LSTM或GRU，通常采用堆叠、双向的配置，我们创建了一个声学模型，该模型在捕捉人类语音复杂、长距离模式方面非常有效。这种出色的序列到序列架构，是我们下一节构建基于CTC训练流程的立足点。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆（LSTM）架构，为读者提供了对其设计及其在学习长程依赖方面的有效性的基础理解。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179/) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724-1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  这篇开创性论文介绍了门控循环单元（GRU）作为LSTM的更简单替代方案，详细阐述了其架构和在序列建模任务中的表现。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面涵盖深度学习理论基础和实践方面的教科书，包括对循环神经网络、LSTM和GRU的详细解释。

---

[上一节](02-%E4%BD%BF%E7%94%A8%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%9E%84%E5%BB%BA%E5%A3%B0%E5%AD%A6%E6%A8%A1%E5%9E%8B.md) · [下一节](04-%E8%81%94%E7%BB%93%E4%B8%BB%E4%B9%89%E6%97%B6%E5%BA%8F%E5%88%86%E7%B1%BB%20%28CTC%29%20%E6%8D%9F%E5%A4%B1.md)
