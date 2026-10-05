# 门控循环单元 (GRUs)

来源：[原文](https://apxml.com/zh/courses/nlp-fundamentals/chapter-5-nlp-sequence-models-intro/gru-networks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

循环神经网络 (neural network) (RNN) 是处理序列数据的基本架构，但简单的 RNN 经常遇到梯度消失问题，这阻碍了它们捕获长距离依赖关系的能力。长短期记忆 (LSTM) 网络作为一种解决方案应运而生，通过引入门控机制来调节长序列中的信息流动。然而，LSTM 相当复杂，它具有三个不同的门（输入门、遗忘门、输出门）和一个独立的单元状态。门控循环单元 (GRU) 由 Cho 及其同事于 2014 年提出，它是一种更简单的替代方案。GRU 通常能达到与 LSTM 相似的性能，同时计算效率更高。

GRU 与 LSTM 类似，都使用门控机制来控制信息流动，但 GRU 通过更精简的架构实现了这一点。GRU 单元不再使用三个门和一个独立的单元状态，而是仅采用两个门：**更新门**和**重置门**。它还将单元状态和隐藏状态合并为一个隐藏状态向量 (vector) $h_t$。

让我们看看这些组件在给定时间步 $t$ 下如何共同作用，将当前输入 $x_t$ 和前一隐藏状态 $h_{t-1}$ 作为输入。

### 重置门

重置门 ($r_t$) 决定如何将新输入 $x_t$ 与前一隐藏状态 $h_{t-1}$ 组合。具体来说，它控制在计算*下一个*隐藏状态的候选值时，*前一*隐藏状态有多少应该被“遗忘”或忽略。如果重置门对于 $h_{t-1}$ 的某些维度值接近 0，它会有效地使该单元表现得像是首次读取输入一样，忽略与当前语境不相关的历史信息。

重置门的激活值使用 Sigmoid 函数 ($\sigma$) 计算，该函数输出 0 到 1 之间的值：


$$
r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)
$$


其中，$[h_{t-1}, x_t]$ 表示前一隐藏状态和当前输入向量 (vector)的连接。$W_r$ 是权重 (weight)矩阵，$b_r$ 是重置门的偏置 (bias)项。

### 更新门

更新门 ($z_t$) 扮演着与 LSTM 中遗忘门和输入门的组合相似的角色。它决定了前一隐藏状态 $h_{t-1}$ 有多少信息应该传递到当前隐藏状态 $h_t$。同时，它也控制了*新计算出的*候选隐藏状态 ($\tilde{h}_t$，接下来会解释) 应该被加入多少。

如果 $z_t$ 接近 1，则前一隐藏状态 $h_{t-1}$ 大部分被复制到新的隐藏状态 $h_t$。如果 $z_t$ 接近 0，则新的候选状态 $\tilde{h}_t$ 主要构成新的隐藏状态 $h_t$。这种机制使得 GRU 能够保留来自遥远过去时间步的信息。

更新门的激活值也使用 Sigmoid 函数计算：


$$
z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)
$$


其中 $W_z$ 和 $b_z$ 分别是更新门的权重 (weight)矩阵和偏置 (bias)项。

### 候选隐藏状态

GRU 计算一个候选隐藏状态 $\tilde{h}_t$，它表示当前时间步的提议更新。此计算受重置门 $r_t$ 影响。重置门决定了前一隐藏状态 $h_{t-1}$ 的哪些部分用于此计算。候选状态通常使用双曲正切 ($\tanh$) 激活函数 (activation function)计算，该函数输出 -1 到 1 之间的值。


$$
\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)
$$


请注意重置门 $r_t$ 和前一隐藏状态 $h_{t-1}$ 之间的元素级乘法 ($\odot$)。这是重置门在将信息与当前输入 $x_t$ 组合之前，选择性地过滤来自 $h_{t-1}$ 信息的地方。$W_h$ 和 $b_h$ 是相应的权重 (weight)矩阵和偏置 (bias)。

### 最终隐藏状态

最后，当前时间步的实际隐藏状态 $h_t$ 是通过对前一隐藏状态 $h_{t-1}$ 和候选隐藏状态 $\tilde{h}_t$ 进行线性插值计算得出的。更新门 $z_t$ 控制此插值过程：


$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$


此公式说明了更新门如何作用：

- 当 $z_t$ 接近 1 时，$h_t$ 主要由候选状态 $\tilde{h}_t$ 构成。
- 当 $z_t$ 接近 0 时，$h_t$ 主要由前一状态 $h_{t-1}$ 构成。

这使得网络能够通过在多个时间步保持 $z_t$ 接近 0 来维持长期信息，或者通过将 $z_t$ 设置为接近 1 来根据新输入快速更新。

下图展示了 GRU 单元的结构：

> 门控循环单元 (GRU) 单元的内部结构，展示了信息如何通过重置门 ($r_t$) 和更新门 ($z_t$) 流动以计算下一个隐藏状态 ($h_t$)。

### GRU 与 LSTM 对比

尽管 GRU 和 LSTM 都有效地解决了梯度消失问题并捕获了长距离依赖，但它们在内部结构上有所不同：

- **复杂性：** GRU 拥有两个门并合并了单元和隐藏状态。LSTM 有三个门并保持独立的单元状态。这使得 GRU 在结构上更简单。
- **参数 (parameter)：** 由于门数量较少，在相同的隐藏状态大小下，GRU 通常比 LSTM 拥有更少的训练参数。这可以导致更快的训练时间，并可能减少过拟合 (overfitting)，尤其是在较小的数据集上。
- **信息流：** LSTM 凭借其专用的单元状态和输出门，可能提供对内存内容和暴露更细粒度的控制。这在理论上可能对某些复杂任务更有利，尽管经验证据通常显示相似的性能。
- **性能：** 实际上，在所有任务中，没有任何一种架构能够始终优于另一种。性能通常取决于具体数据集、任务和超参数 (hyperparameter)调整。在模型开发过程中，通常会尝试两种架构。

### GRU 的实际应用

与 LSTM 类似，GRU 在 TensorFlow (通过 Keras) 和 PyTorch 等标准深度学习 (deep learning)框架中很容易作为层使用。您可以轻松地在模型定义中用 `GRU` 层替换 `LSTM` 层。诸如堆叠多个 GRU 层（以创建更深的网络）或使用双向 GRU（以处理正向和反向序列）等思路同样适用于 LSTM。

总而言之，GRU 为序列建模提供了 LSTM 的一个有吸引力的替代方案。其更简单的架构通常意味着更高的计算效率，而对于许多 NLP 任务，性能没有显著牺牲。通过使用重置门和更新门，它们有效地管理随时间的信息流，使它们能够学习序列间的依赖关系，这与它们的 LSTM 同类非常相似。在构建序列模型时，同时考虑 LSTM 和 GRU 是模型选择过程的标准部分。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://doi.org/10.3115/v1/D14-1179) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724–1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  介绍了门控循环单元（GRU）架构，作为LSTM在序列建模方面的一种更简单的替代方案。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  提供了循环神经网络（包括LSTM和GRU）在序列建模背景下的详细理论基础。

---

[上一节](04-%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%EF%BC%88LSTM%EF%BC%89%E7%BD%91%E7%BB%9C.md) · [下一节](06-%E5%B0%86%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E4%BA%8E%E6%96%87%E6%9C%AC.md)
