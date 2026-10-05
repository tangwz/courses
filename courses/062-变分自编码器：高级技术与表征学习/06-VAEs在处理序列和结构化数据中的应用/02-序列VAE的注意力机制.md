# 序列VAE的注意力机制

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-6-vaes-sequential-structured-data/vaes-attention-sequences)

[返回章节目录](README.md) · [返回课程目录](../README.md)

循环变分自编码器（RVAE）常用于对序列数据中的时间依赖性进行建模。然而，当这些依赖性跨度很长时，它们可能会遇到难题。标准的循环架构，包括LSTM或GRU，可能难以在长时间滞后中传播信息，导致对久远事件的表示变得稀薄。在这种问题下，注意力机制 (attention mechanism)带来了显著改善，使得模型在处理或生成信息时，能够动态地侧重于输入序列的相关部分，而无需考虑它们之间的距离。

### 将注意力机制 (attention mechanism)融入序列VAE

注意力机制使得模型在产生输出或形成表示时，能够权衡输入序列不同部分的重要性。在序列VAE的背景下，注意力可以被引入到编码器、解码器，或者两者。

#### 解码器端注意力

最常见的引入方式是在VAE解码器中加入注意力机制。在此，在生成输出序列元素 $y_i$ 的每一步，解码器会关注输入序列的编码表示 $H = (h_1, h_2, \ldots, h_{L_x})$，其中 $h_j$ 是VAE编码器对第$j$个输入元素的输出，$L_x$ 是输入序列的长度。

这个过程通常如下进行：

1. VAE编码器处理输入序列 $X$，以生成编码器输出序列 $H$。它还计算近似后验分布 $q_\phi(z|X)$ 的参数 (parameter)（例如，均值 $\mu_z$ 和对数方差 $\log \sigma_z^2$）。
2. 使用重参数化技巧从 $q_\phi(z|X)$ 中采样潜在变量 $z$。这个 $z$ 捕获全局序列特征。
3. 解码器（通常是RNN）逐步生成输出序列 $Y = (y_1, y_2, \ldots, y_{L_y})$。在每个步骤 $i$：
   - 解码器当前的隐藏状态 $s_{i-1}$ 作为查询。
   - 在 $s_{i-1}$ 和每个编码器输出 $h_j$ 之间计算一个**对齐 (alignment)分数** $e_{ij}$。一种常见的评分函数是加性（Bahdanau式）注意力：
     
     $$
     e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j + b_a)
     $$
     
     或乘性（Luong式）注意力：
     
     $$
     e_{ij} = s_{i-1}^T W_a h_j
     $$
     
     其中 $W_a$、$U_a$、$v_a$ 和 $b_a$ 是可学习参数。
   - 这些分数通过softmax函数归一化 (normalization)以生成**注意力权重 (weight)** $\alpha_{ij}$：
     
     $$
     \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{L_x} \exp(e_{ik})}
     $$
     
     这些权重总和为1，并表明每个输入部分 $h_j$ 对生成当前输出 $y_i$ 的重要性。
   - **上下文 (context)向量 (vector)** $c_i$ 作为编码器输出的加权和计算：
     
     $$
     c_i = \sum_{j=1}^{L_x} \alpha_{ij} h_j
     $$
     
   - 解码器的隐藏状态被更新，并使用 $s_{i-1}$、前一个输出 $y_{i-1}$、上下文向量 $c_i$ 和全局潜在变量 $z$ 预测输出 $y_i$。例如，步骤 $i$ 时RNN单元的输入可以是连接 $[y_{i-1}, c_i, z]$。

潜在变量 $z$ 可以以多种方式影响这个过程：它可能初始化解码器的状态，在每个解码器步骤中连接到输入（如上述示例所示），甚至参与注意力分数计算。要点在于 $z$ 提供全局条件，而注意力则提供对输入序列特定部分的细粒度、动态对齐。

> 带有注意力机制的VAE解码器中的信息流，说明了一个解码步骤。编码器处理输入序列以生成隐藏状态和潜在变量 $z$ 的参数。然后解码器使用 $z$、其先前状态、先前输出以及一个上下文向量（通过对编码器状态的注意力获得）来生成当前输出。

#### 编码器端注意力（自注意力 (self-attention)）

注意力也可以提升编码器性能。由Transformer架构推广的自注意力机制，使得编码器在计算每个元素的表示时，能够权衡*输入序列内部*不同元素的重要性。这意味着 $h_j$ 不仅仅是 $x_j$ 和 $h_{j-1}$ 的函数（如在RNN中），而是所有 $x_1, \ldots, x_{L_x}$ 的函数，根据它们与 $x_j$ 的相关性进行加权。

使用Transformer风格的编码器可以产生强大的表示 $H$，能够捕获复杂的序列内部依赖性。这些丰富的编码器输出 $H$ 随后可以被上述的注意力解码器使用，或者它们可以被聚合（例如，通过特殊的 `[CLS]` 符号的表示或通过池化）来形成 $q_\phi(z|X)$ 的参数。

### 主要注意力机制 (attention mechanism)

- **加性（Bahdanau）和乘性（Luong）注意力**：这些是通常用于基于RNN的序列到序列模型的基础注意力机制。它们的主要区别在于对齐 (alignment)分数 $e_{ij}$ 的计算方式。两者都能有效地使得解码器侧重于（可能经过RNN编码的）输入序列的相关部分。
- **自注意力 (self-attention)（Transformer风格）**：这种机制使得序列中的每个元素都可以关注同一序列中的所有其他元素。多头自注意力是Transformer的核心组成部分，它并行运行多个自注意力操作并连接它们的结果，使得模型能够同时关注来自不同表示子空间、不同位置的信息。Transformer可以作为VAE中的强大编码器或解码器，替代或增强RNN。例如，VAE可以使用Transformer编码器来生成 $H$ 并参数 (parameter)化 $z$，以及一个以 $z$ 为条件的Transformer解码器（自回归 (autoregressive)的，使用掩码自注意力和对 $H$ 的交叉注意力）。

### 序列VAE的优点

将注意力融入序列数据的VAE带来了多项优点：

- **更好地建模长距离依赖性**：注意力直接解决了捕获序列中遥远元素之间关系的难题，从而产生更准确的模型。
- **增强样本质量**：通过侧重于相关上下文 (context)，解码器可以生成更连贯、高保真度的序列，特别是对于长文本 (long context)段落或细致音频等复杂数据。
- **更佳的表示学习**：带有自注意力 (self-attention)的编码器能够更好地归纳输入序列，从而创建更详细的潜在表示 $z$。解码器端的注意力使得 $z$ 可以侧重于全局方面，将局部细节交给注意力机制 (attention mechanism)。
- **可解释性**：注意力权重 (weight) $\alpha_{ij}$ 可以被可视化，提供关于模型在生成特定输出时认为输入序列哪些部分重要的洞察。这对于调试和理解模型行为有价值。

### 挑战和设计考量

虽然强大，注意力机制 (attention mechanism)也引入了一些挑战：

- **计算成本**：自注意力 (self-attention)的计算复杂度为 $O(L^2 \cdot d)$，其中 $L$ 是序列长度，$d$ 是表示维度。这对于非常长的序列来说可能成本过高。稀疏注意力或局部注意力等技术旨在缓解这个问题。RNN输出上的传统注意力机制的计算复杂度为 $O(L_x \cdot L_y)$。
- **模型复杂度增加**：带有注意力的VAE在设计、实现和调优方面更为复杂。更多的超参数 (parameter) (hyperparameter)和架构选择需要仔细考量。
- **平衡潜在变量的使用**：强大的注意力机制，特别是在解码器中，可能会学会有效地复制或严重依赖编码器输出 $H$，从而可能使得全局潜在变量 $z$ 被边缘化。如果ELBO中的KL散度项过强，或者 $z$ 没有提供足够独特的信息，这可能导致 $z$ 被忽略（一种“后验坍塌”的形式）。需要仔细的正则化 (regularization)和架构设计，以确保 $z$ 对生成过程有意义的贡献。

### 典型应用

带有注意力的VAE特别适用于：

- **自然语言处理**：可控文本生成、抽象摘要和对话建模等任务得益于注意力处理长上下文 (context)和依赖性的能力。
- **语音合成和识别**：生成或理解长话语，其中来自久远音素或单词的上下文很重要。
- **音乐生成**：创建具有连贯长期结构以及音符或乐句之间依赖性的音乐作品。
- **时间序列分析**：建模复杂时间序列，其中过去事件，即使是遥远的事件，也能以非简单的方式影响未来值。

通过融入注意力，VAE在处理序列数据中复杂的依赖性方面变得显著更熟练。这使得生成更真实的序列和学习更具表达力的潜在表示成为可能，从而拓展了VAE在处理复杂、有序信息方面的能力范围。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  变分自编码器（VAE）和重参数化技巧的基础论文。
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1409.0473](https://doi.org/10.48550/arXiv.1409.0473)
  引入了用于序列到序列模型的加性注意力机制（Bahdanau风格），支持动态对齐。
- [Effective Approaches to Attention-based Neural Machine Translation](https://aclanthology.org/D15-1166/) — Thang Luong, Hieu Pham, Christopher D. Manning (2015)
  Journal: Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing; Publisher: Association for Computational Linguistics; Pages: 1412-1421; DOI: [10.18653/v1/D15-1166](https://doi.org/10.18653/v1/D15-1166)
  比较了神经机器翻译中不同的注意力机制，包括乘法注意力（Luong风格）。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS 2017); Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  引入了Transformer架构，该架构完全依赖自注意力（多头注意力）并改变了序列建模。

---

[上一节](01-%E5%BE%AA%E7%8E%AF%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%20%28RVAE%29%20%E7%94%A8%E4%BA%8E%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E5%BB%BA%E6%A8%A1.md) · [下一节](03-%E5%9B%BEVAEs%E7%94%A8%E4%BA%8E%E7%BB%93%E6%9E%84%E5%8C%96%E6%95%B0%E6%8D%AE%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0.md)
