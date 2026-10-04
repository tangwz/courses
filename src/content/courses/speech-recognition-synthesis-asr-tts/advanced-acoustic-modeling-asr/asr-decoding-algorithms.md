---
course: "speech-recognition-synthesis-asr-tts"
chapter: "advanced-acoustic-modeling-asr"
lesson: "asr-decoding-algorithms"
sourceId: 3136
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-2-advanced-acoustic-modeling-asr/asr-decoding-algorithms"
title: "解码算法比较"
description: "分析不同模型类型的波束搜索变体、WFST解码及格生成。"
order: 7
plots: []
sourceHash: "944ecf1cb844c395d34d6f26731e9602630859f420f89ae2a98bc909beddf30f"
sourceCorrections: []
---

当一个高级声学模型，如CTC、基于注意力的编码器-解码器或RNN Transducer训练完成后，它会根据输入音频特征生成表示不同语言单元（如字符或音素）可能性的输出。然而，这些原始输出，通常是随时间或序列变化的概率分布，并非最终的转录结果。重要的下一步是**解码**：即寻找与模型输出相对应的最可能词序列的过程。这个搜索过程不简单，因为可能的序列数量可能非常大。不同的声学模型架构需要不同的解码策略，每种策略都有其特点，并在准确性、计算成本和实时转录等应用的适用性方面存在权衡。

### ASR解码中的搜索问题

核心在于，解码的目标是寻找输出序列 $Y$（例如字符或词），使其在给定输入声学特征 $X$ 的情况下概率最大化。对于端到端模型，这通常表示为寻找 $\hat{Y} = \arg\max_Y P(Y|X)$。对于混合系统，这涉及结合来自声学模型（AM）、发音词典（L）和语言模型（LM）的概率：$\hat{Y} = \arg\max_Y P(X|Y) P(Y)$。

寻找全局最优序列通常计算上不可行，因为搜索空间非常庞大。因此，实际的解码器依赖启发式搜索算法，主要是波束搜索的变体，以高效地找到高概率序列。解码算法的选择和配置对最终的ASR性能有显著影响。

### 使用混合HMM-DNN系统进行解码：WFSTs

传统的混合系统，将DNN与隐马尔可夫模型（HMM）结合，通常依赖加权有限状态转换器（WFST）进行解码。此方法巧妙地将不同知识源结合到一个搜索图中。

1. **组成部分：** 此过程涉及组合多个WFST：
   - **H (HMM)：** 表示声学上下文 (context)相关的HMM状态转换。DNN提供发射概率（给定HMM状态下声学特征的可能性）。
   - **C (上下文相关性)：** 将上下文相关单元（例如三音素）映射到上下文无关单元（例如单音素）。
   - **L (词典)：** 将音素序列映射到词。
   - **G (语法/语言模型)：** 表示词序列的概率，通常是将n-gram语言模型转换为WFST形式。
2. **组合：** 这些WFST被组合成一个单一的大型搜索图：$S = H \circ C \circ L \circ G$。此图编码了从声学状态到词序列的所有有效路径，并根据组合概率进行加权。
3. **搜索：** 解码涉及在给定声学特征向量 (vector)序列 $X$ 的情况下，寻找通过组合图 $S$ 的最佳路径。这通常使用令牌传递算法完成，该算法本质上是直接在WFST上操作的维特比搜索或波束搜索的优化形式。

基于WFST的解码功能强大，允许顺畅集成复杂的词典和大型n-gram语言模型。然而，构建和优化这些WFST可能很复杂，而且此方法将声学模型与预定义的HMM状态和发音词典紧密耦合。

### 使用连接主义时间分类（CTC）进行解码

CTC模型通过隐式学习对齐 (alignment)来简化过程。它们在每个时间帧 $t$ 输出每个标记 (token) $k$（包括特殊空白符号 $\epsilon$）的概率分布 $P(k|t, X)$。解码涉及通过考虑所有可能的对齐 $\pi$ 来寻找最可能的输出序列 $Y$，这些对齐在移除重复的非空白符号和空白符号后映射到 $Y$。

1. **贪婪（最佳路径）解码：** 最简单的方法。在每个时间步 $t$，选择概率最高的标记 $k$，即 $P(k|t, X)$。连接这些标记，然后应用CTC折叠规则（移除重复项，移除空白）。这非常快但通常不是最优的，因为它做局部决策而没有考虑完整的序列概率。
2. **波束搜索解码：** 一种更有效的方法，它在每个时间步保持一个包含 $B$ 个最可能候选前缀序列的集合（即“波束”）。
   - **初始化：** 从一个概率为1.0的空前缀开始。
   - **扩展：** 在每个时间步 $t$，通过考虑附加下一个可能的输出标记（包括空白）来扩展波束中的每个前缀。使用CTC输出概率 $P(k|t, X)$ 和前一步前缀的概率来计算扩展前缀的概率。处理在折叠后产生相同输出序列的路径的合并（例如，`A` -> `AA` 和 `A` -> `A$\epsilon$` 都产生 `A`）。
   - **剪枝：** 只保留波束中前 $B$ 个最可能的前缀用于下一步。
   - **终止：** 继续直到输入序列结束。最终波束中概率最高的序列即为结果。
3. **语言模型集成：** 标准的n-gram或神经语言模型可以在CTC波束搜索期间集成。扩展前缀时，可以通过添加来自语言模型的建议下一个词（或字符）的对数概率来调整概率，通常由超参数 (parameter) (hyperparameter) $\lambda$ 加权。这有时被称为前缀波束搜索或浅融合。

> CTC波束搜索随时间步扩展的视图，维护候选前缀及其概率。实际实现根据CTC折叠规则处理路径合并。

CTC解码比WFST解码更简单，因为它避免了显式HMM状态和词典组合，但波束搜索中有效的语言模型集成对高准确率很重要。

### 使用注意力编码器-解码器（AED）进行解码

AED模型，例如听、注意和拼写（LAS）或基于Transformer的ASR模型，自回归 (autoregressive)地生成输出序列 $Y = (y_1, y_2, ..., y_L)$。也就是说，下一个标记 (token) $y_i$ 的概率取决于编码后的声学特征 $H_{enc}$ 和先前生成的标记 $y_{<i}$：$P(y_i | H_{enc}, y_{<i})$。

1. **贪婪解码：** 在每个输出步 $i$，只需选择概率最高的标记 $y_i$，即 $P(y_i | H_{enc}, y_{<i})$。这很快，但易于在序列早期出现错误，且这些错误之后无法纠正。
2. **波束搜索解码：** 序列生成模型的标准方法。
   - **初始化：** 从序列开始标记（`<sos>`）作为初始假设开始。
   - **扩展：** 在每个步骤 $i$，取出波束中当前 $B$ 个假设（部分序列）的集合。对于每个假设，使用模型计算下一个可能标记 $y_i$ 的概率分布。通过附加前 $k$ 个最可能的下一个标记来扩展每个假设（其中 $k$ 可能等于波束宽度 $B$ 或更小）。
   - **评分：** 扩展假设的分数（通常是对数概率）是其组成标记的对数概率之和。$Score(y_1...y_i) = \sum_{j=1}^{i} \log P(y_j | H_{enc}, y_{<j})$。
   - **剪枝：** 收集所有扩展的假设，只保留总得分最高的前 $B$ 个。
   - **终止：** 当假设达到序列结束标记（`<eos>`）或最大长度时停止。选择得分最高（可能根据长度调整）的已完成假设。
3. **语言模型集成：** 语言模型通常被集成以提高流畅性和识别罕见词。
   - **浅融合：** 在波束搜索期间，通过添加加权的语言模型对数概率来修改假设的分数：$Score_{fused} = Score_{AM} + \lambda \cdot Score_{LM}$。语言模型独立地计算分数 $P_{LM}(y_i | y_{<i})$。
   - **深融合：** 语言模型的内部状态在预测下一个标记概率*之前*与ASR模型的解码器状态结合。这允许更紧密的集成，但增加了模型复杂性。

相比贪婪解码，波束搜索显著提高了AED性能，但由于需要检查多条路径，会引入延迟。

### 使用RNN Transducers (RNN-T) 进行解码

RNN-T模型逐个标记 (token)地预测输出，同时决定是否在输入音频时间步长上前进，天然支持流式推理 (inference)。模型输出下一个输出标记 $y_i$ 和发出空白符号 $\epsilon$ 的联合分布：$P(k | t, u)$，其中 $t$ 是当前音频帧索引，$u$ 是先前预测的非空白标记的索引。

1. **贪婪解码：** 在隐式对齐 (alignment)网格的每一步中，选择概率最高的动作：要么发出一个特定的标记 $k \neq \epsilon$（推进 $u$），要么发出 $\epsilon$（推进 $t$）。
2. **波束搜索解码：** 标准波束搜索需要根据RNN-T的结构进行调整。
   - **假设状态：** 波束中的每个假设不仅需要跟踪生成的标记序列 $y$，还需要跟踪当前音频帧索引 $t$ 以及与最后预测的非空白标记 $u$ 相关的索引/状态。
   - **扩展：** 对于波束中的每个假设 $(y, t, u)$，使用 $P(k | t, u)$ 计算所有可能的下一个标记 $k$（包括 $\epsilon$）的概率。如果 $k = \epsilon$，新状态为 $(y, t+1, u)$。如果 $k \neq \epsilon$，新状态为 $(y \cdot k, t, u+1)$，其中 $y \cdot k$ 表示将 $k$ 附加到 $y$。
   - **剪枝：** 根据累积概率保留前 $B$ 个假设。通常使用对齐长度同步解码等专门算法来提高效率。
3. **语言模型集成：** 类似于CTC，语言模型可以在波束搜索期间集成，通常通过浅融合，当提出非空白标记时调整分数。

RNN-T解码非常适合低延迟流式ASR，因为可以随着音频的到达进行预测。由于需要管理时间和输出序列的推进，其波束搜索算法比标准AED波束搜索更复杂。

### 比较与权衡

| 特性 | WFST（混合） | CTC | AED（注意力/Transformer） | RNN-T |
| --- | --- | --- | --- | --- |
| **架构** | DNN + HMM/GMM | 端到端（含空白） | 端到端（Seq2Seq） | 端到端（流式） |
| **对齐 (alignment)** | 显式（HMM状态） | 隐式（通过空白） | 隐式（注意力） | 隐式（对齐网格） |
| **主要解码器** | WFST搜索 | 波束搜索 | 波束搜索 | 专门的波束搜索 |
| **流式** | 困难/需要技巧 | 可能（有延迟） | 困难（自回归 (autoregressive)） | 自然 |
| **语言模型集成** | WFST组合 | 浅融合/前缀BS | 浅/深融合 | 浅融合/前缀BS |
| **复杂性** | 高（WFST组合） | 中等 | 中等 | 高（波束搜索状态） |
| **依赖** | 词典，HMM拓扑 | 最少 | 最少 | 最少 |

> 不同ASR架构的解码方法比较。

**考量因素：**

- **延迟与准确性：** 贪婪解码最快但准确性最低。波束搜索以计算和延迟为代价提高准确性。更大的波束宽度（$B$）通常能提高准确性但增加了成本。RNN-T为低延迟流式传输提供了最佳的原生支持。
- **语言模型集成：** 集成外部语言模型几乎总是有利于提高准确性。浅融合在不同架构中更容易实现，而深融合（主要用于AED）可能提供更紧密的集成。
- **计算资源：** WFST解码可能需要大量内存用于组合图。波束搜索的复杂性随波束宽度和序列长度而变化。RNN-T波束搜索需要仔细的状态管理。
- **实现：** Kaldi等库提供工具。ESPnet、NeMo和SpeechBrain等工具包提供CTC、AED和RNN-T解码的实现。

归根结底，解码算法的选择与声学模型架构紧密相关。理解WFST、CTC波束搜索、序列到序列波束搜索和RNN-T解码背后的原理，可以帮助您为特定的ASR系统和应用需求选择和配置合适的搜索策略。高效准确的解码与训练有素的声学模型同样重要，可实现最先进的语音识别性能。

## 参考资料

- [Speech Recognition with Weighted Finite-State Transducers](https://ieeexplore.ieee.org/document/1026040) — Mehryar Mohri, Fernando Pereira, Michael Riley (2002)
  Journal: Proceedings of the 2002 IEEE Workshop on Machine Learning for Signal Processing; Publisher: IEEE; Pages: 3-10; DOI: [10.1109/MLSP.2002.1026040](https://doi.org/10.1109/MLSP.2002.1026040)
  奠基性论文，详细阐述了加权有限状态转换器在构建和解码语音识别系统中的应用。
- [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://doi.org/10.1145/1143844.1143891) — Alex Graves, Santiago Fernandez, Faustino Gomez, Jürgen Schmidhuber (2006)
  Journal: Proceedings of the 23rd International Conference on Machine Learning (ICML); Publisher: Association for Computing Machinery; Pages: 369-376; DOI: [10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891)
  介绍了连接时序分类（CTC）损失函数及其相关解码算法，对现代端到端ASR至关重要。
- [Sequence Transduction with Recurrent Neural Networks](https://proceedings.mlr.press/v27/graves12a/graves12a.pdf) — Alex Graves (2012)
  Journal: Proceedings of the International Conference of Machine Learning (ICML) 2012 Workshop on Representation Learning; Volume: 27; Pages: 368-376
  提出了循环神经网络换能器（RNN-T）架构，提供了一种无需显式对齐即可将输入序列直接映射到输出序列的框架，对流式ASR至关重要。
- [Listen, Attend and Spell: A Neural Network for Large Vocabulary Conversational Speech Recognition](https://arxiv.org/pdf/1508.01211.pdf) — William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals (2016)
  Journal: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP); Publisher: IEEE; Pages: 5775-5779; DOI: [10.1109/ICASSP.2016.7472651](https://doi.org/10.1109/ICASSP.2016.7472651)
  开创性论文，介绍了用于ASR的端到端基于注意力的编码器-解码器模型（LAS），展示了序列到序列学习与注意力的有效性。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  Publisher: Stanford University
  这本全面的教材涵盖了语音识别的基本概念，包括声学建模、语言建模以及Viterbi和Beam搜索等解码算法。
