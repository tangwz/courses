# 联结主义时间分类 (CTC)

来源：[原文](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-2-advanced-acoustic-modeling-asr/ctc-acoustic-modeling)

[返回章节目录](README.md) · [返回课程目录](../README.md)

传统的混合隐马尔可夫模型-深度神经网络 (neural network)（HMM-DNN）系统虽然有效，但涉及不同的组成部分（声学模型、词典、语言模型），并且训练时依赖于音频帧和语音状态之间预先计算的对齐 (alignment)。这个对齐过程本身可能很复杂且效果不佳。联结主义时间分类（CTC）提供了一种替代方法，它使声学模型能够直接从输入音频特征到输出字符或单词序列进行端到端训练，无需帧级别对齐。

### 对齐 (alignment)的难题

设想将变长音频输入帧 $x = (x_1, x_2, ..., x_T)$ 映射到短得多的变长目标标签序列 $y = (y_1, y_2, ..., y_U)$，其中 $T$（特征帧数量）通常远大于 $U$（输出字符数量）。例如，一段1秒的发音可能有100个特征向量 (vector)，但只对应10个字符。核心问题在于确定哪些输入帧对应哪些输出标签。CTC巧妙地避免了这种明确的对齐要求。

### 引入空白标签

CTC引入一个特殊的`空白`符号，通常表示为 $\epsilon$ 或 `<b>`。该符号表示在某个时间步“无输出标签”。声学模型，通常是RNN或Transformer，处理输入序列 $x$ 并在每个时间步 $t$ 输出一个概率分布 $p(k|t, x)$，该分布覆盖了所有可能的输出标签词汇表 (vocabulary)*以及*空白符号。设扩展词汇表为 $V' = V \cup \{\mathbf{b}\}$。网络的输出是一个概率分布序列 $p_1, p_2, ..., p_T$，其中每个 $p_t$ 是一个大小为 $|V'|$ 的向量 (vector)。

### 从路径到概率

考虑网络在所有时间步上的输出预测序列，$\pi = (\pi_1, \pi_2, ..., \pi_T)$，其中每个 $\pi_t \in V'$。这称为一个CTC*路径*。给定输入，特定路径 $\pi$ 的概率通过假设不同时间步的预测之间条件独立来计算：

$P(\pi | x) = \prod_{t=1}^{T} p(\pi_t | t, x)$

现在，重要的一步是定义一个多对一映射 $\mathcal{B}$，将所有可能路径 $\pi$ 的集合映射到目标标签序列 $y$。这个映射 $\mathcal{B}$ 首先移除路径中所有连续的重复标签，然后移除所有空白符号。

例如，如果我们的目标标签序列是 $y = (\text{c, a, t})$，词汇表 (vocabulary)是 \{\text{a, c, t, \mathbf{b}}\}，则有几条路径可以映射到 $y$：

- $\pi^1 = (\mathbf{b}, \text{c}, \mathbf{b}, \text{a}, \text{a}, \mathbf{b}, \text{t}, \mathbf{b})$ -> $\mathcal{B}(\pi^1) = (\text{c, a, t})$
- $\pi^2 = (\text{c}, \text{c}, \mathbf{b}, \mathbf{b}, \text{a}, \text{t}, \text{t})$ -> $\mathcal{B}(\pi^2) = (\text{c, a, t})$
- $\pi^3 = (\text{c}, \text{a}, \text{t})$ -> $\mathcal{B}(\pi^3) = (\text{c, a, t})$ (假设T=3)

给定输入 $x$，观察到目标序列 $y$ 的概率是所有映射到 $y$ 的路径 $\pi$ 的概率之和：

$P(y | x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} P(\pi | x)$

其中 $\mathcal{B}^{-1}(y)$ 表示所有会折叠成目标序列 $y$ 的路径的集合。

### CTC损失函数 (loss function)

直接对所有有效路径求和来计算 $P(y|x)$ 在计算上是不可行的，因为此类路径的数量随输入长度 $T$ 呈指数增长。CTC采用一种动态规划算法，类似于HMM中使用的前向-后向算法，来高效计算这个和。

我们定义 $y'$ 为目标序列 $y$ 在开头和每个标签之间穿插了空白符号。对于 $y = (\text{c, a, t})$， $y' = (\mathbf{b}, \text{c}, \mathbf{b}, \text{a}, \mathbf{b}, \text{t}, \mathbf{b})$。设 $|y'| = 2U + 1$。

核心思想是计算前向概率 $\alpha(t, u)$，它表示在时间步 $t$ 结束于 $y'$ 的状态 $u$（要么是空白，要么是字符）的所有路径的总概率。类似地，后向概率 $\beta(t, u)$ 表示从时间 $t$ 的状态 $u$ 开始完成序列 $y$ 的总概率。

$\alpha(t, u)$ 的递归计算涉及对来自 $t-1$ 时间步的有效先行状态的概率求和：

- 状态 $u$ 可以在 $t-1$ 时间步从状态 $u$ 到达。
- 状态 $u$ 可以在 $t-1$ 时间步从状态 $u-1$ 到达。
- 如果 $y'_u$ 是空白，或者如果 $y'_{u-2} = y'_u$，状态 $u$ 也可以在 $t-1$ 时间步从状态 $u-2$ 到达。这处理了涉及空白和重复字符的转移。

基本情况是 $\alpha(1, 1) = p(y'_1 | 1, x)$ 和 $\alpha(1, 2) = p(y'_2 | 1, x)$，当 $u > 2$ 时，$\alpha(1, u) = 0$。

通过对 $y'$ 的最后两个状态（最终标签和最终空白）在最终时间步 $T$ 的前向概率求和，可以高效计算总概率 $P(y|x)$：

$P(y | x) = \alpha(T, |y'|) + \alpha(T, |y'|-1)$

CTC损失函数就是正确标签序列的负对数似然：

$L_{CTC}(x, y) = - \ln P(y | x)$

训练涉及使用梯度下降 (gradient descent)和反向传播 (backpropagation)，通过声学模型（例如RNN）和CTC计算层来最小化这个损失函数。梯度 $\frac{\partial L_{CTC}}{\partial p(k|t, x)}$ 也可以使用前向（$\alpha$）和后向（$\beta$）变量高效计算。

### 基于CTC模型的解码

模型训练完成后，我们需要一种方法来在给定新输入 $x$ 的情况下，推断出最有可能的输出序列 $y^*$。

$y^* = \arg \max_y P(y | x)$

找到精确的最大值很困难。常见的解码策略包括：

1. **最佳路径解码（贪心搜索）：** 这是最简单的方法。在每个时间步 $t$，选择概率最高的标签 $\pi^*_t$。将这些标签连接起来形成路径 $\pi^* = (\pi^*_1, ..., \pi^*_T)$。最后，应用映射 $\mathcal{B}$ 将这条路径折叠成输出序列 $y^* = \mathcal{B}(\pi^*)$。虽然快速，但这种方法并非最优，因为它没有考虑完整路径的组合概率，并且可能做出局部最优的选择，从而导致全局非最优的序列。
2. **集束搜索解码：** 这种启发式搜索算法查看搜索空间中更有希望的部分。它在每个时间步维护 $k$ 个最有可能的候选序列集合（束）。对于束中的每个序列，它查看可能的下一个标签（包括空白），计算扩展序列的概率，并只保留整体前 $k$ 个。结束于相同折叠序列的概率会被合并。这比贪心搜索显著提升性能，但增加了计算成本。

> CTC集束搜索解码的简化图示。节点代表不同时间步的局部假设（折叠后的序列）。边表示基于网络输出 $P(\text{label}|t)$ 的转移，搜索只保留最有可能的假设（束）。与通向相同折叠序列的路径相关的概率会被求和。

3. **语言模型集成：** CTC的条件独立假设意味着它内部对语言依赖的建模效果不佳。集束搜索的输出序列可以使用外部语言模型（LM）进行重新评分，以偏向更符合语言规则的假设。浅层组合（在集束搜索期间组合分数）或深度集成（将LM集成到网络中）等方法是常见的，将在第3章中进一步讨论。

### 优点和缺点

**优点：**

- **无对齐 (alignment)训练：** 无需预对齐数据，简化了训练流程。
- **端到端：** 学习从声学特征到标签序列的直接映射。
- **灵活性：** 自然处理变长输入/输出。

**缺点：**

- **条件独立假设：** 在给定输入的情况下，时间 $t$ 的输出预测被认为是独立于其他输出的，这限制了其捕捉语言上下文 (context)的能力。需要外部LM才能获得最佳性能。
- **多对一映射复杂：** 折叠机制有时会使模型难以学习某些序列，特别是那些由短暂停顿分隔的重复字符序列。
- **解码复杂：** 贪心解码通常不足够；为了获得良好的性能，通常需要结合LM的集束搜索。

### 总结

联结主义时间分类提供了一个强大而精巧的框架，用于构建无需帧级别对齐 (alignment)的端到端声学模型。通过引入空白符号和通过动态规划计算的专门损失函数 (loss function)，CTC允许像RNN和Transformer这样的模型直接在音频序列及其转录的配对上进行训练。尽管它有局限性，特别是条件独立假设，但CTC是朝着现代端到端ASR迈出的重要一步，并且仍然是一种重要的技术，通常作为更复杂架构中的一个组成部分或作为强大的基线。理解CTC为理解后续端到端方法（如基于注意力的模型和RNN Transducers）的动机和设计提供了重要的背景。

## 参考资料

- [Connectionist Temporal Classification: Labelling Sequences without Alignment](https://dl.acm.org/doi/10.1145/1143812.1143862) — Alex Graves, Santiago Fernandez, Faustino Gomez, and Jürgen Schmidhuber (2006)
  Journal: Proceedings of the 23rd International Conference on Machine Learning - ICML '06; Publisher: ACM; Pages: 369-376; DOI: [10.1145/1143812.1143862](https://doi.org/10.1145/1143812.1143862)
  CTC算法的原始奠基性论文，介绍了其数学公式、空白标签的概念、用于损失计算的动态规划方法和基本解码策略。
- [Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567) — Awni Hannun, Carl Case, Jared Casper, Bryan Catanzaro, Greg Diamos, Erich Elsen, Ryan Prenger, Sanjeev Satheesh, Shubho Sengupta, Adam Coates, Andrew Y. Ng (2014)
  Journal: arXiv preprint arXiv:1412.5567; DOI: [10.48550/arXiv.1412.5567](https://doi.org/10.48550/arXiv.1412.5567)
  这篇论文展示了CTC在大规模端到端语音识别系统中的实际应用，详细介绍了其实现、训练以及包括结合语言模型的束搜索在内的有效解码策略。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/index.html) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Online Draft
  这本权威教材提供了CTC在自动语音识别背景下的教学性和全面说明，适合寻求结构化概述的进阶学生。
- [Supervised Sequence Labelling with Recurrent Neural Networks](https://link.springer.com/book/10.1007/978-3-642-24797-2) — Alex Graves (2012)
  Publisher: Springer; Volume: 385; DOI: [10.1007/978-3-642-24797-2](https://doi.org/10.1007/978-3-642-24797-2)
  本书基于作者的博士论文，对CTC、其基本原理以及其与使用循环神经网络的其他序列标注技术的联系进行了深入的理论阐述。

---

[上一节](01-%E6%B7%B7%E5%90%88HMM-DNN%E7%B3%BB%E7%BB%9F.md) · [下一节](03-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E7%9A%84%E7%BC%96%E7%A0%81%E5%99%A8-%E8%A7%A3%E7%A0%81%E5%99%A8%E6%A8%A1%E5%9E%8B.md)
