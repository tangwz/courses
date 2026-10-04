---
course: "mixture-of-experts-advanced-implementation"
chapter: "foundations-of-moe-models"
lesson: "mathematical-formulation-moe-layer"
sourceId: 7047
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-1-foundations-of-moe-models/mathematical-formulation-moe-layer"
title: "MoE层的数学表述"
description: "对完整MoE层的详细数学分解，结合了门控和专家计算。"
order: 4
plots: []
sourceHash: "c06044d3d640547b5f3948f7b846ecfd16608dd72179de385e694b8541513d7e"
sourceCorrections: []
---

门控网络和专家网络是高层组成部分，它们可以被组合成一个精确的数学模型。这个模型详细说明了单个token通过稀疏MoE层的完整前向计算过程，概述了具体的计算步骤。

### 门控网络：从输入到路由权重 (weight)

该过程始于门控网络，也称作路由器。它的作用是确定哪些专家应处理当前的输入token。输入是一个token嵌入 (embedding)，表示为一个向量 (vector) $x \in \mathbb{R}^d$，其中$d$是模型的隐藏维度。

门控网络本身是一个简单的线性层，由一个权重矩阵 $W_g \in \mathbb{R}^{d \times N}$定义，其中$N$是专家总数。该层将输入token投射到一个$N$维空间，为每个专家生成一个logit值。


$$
h(x) = x \cdot W_g
$$


得到的向量$h(x)$包含$N$个原始得分。为了将这些得分转换为概率分布，我们应用softmax函数：


$$
g(x) = \text{softmax}(h(x))
$$


输出$g(x)$是一个稠密的$N$维向量，其中每个元素$g(x)_i$表示路由器将token分配给专家$i$的置信度。$g(x)$中所有元素的和为1。

### 通过Top-K门控实现稀疏性

一个稠密的$g(x)$向量 (vector)意味着每个专家都会对输出做出贡献，这与MoE计算效率的目标相悖。为了强制稀疏性，我们采用`TopK`操作。不使用所有专家，我们选择得分最高的专家中的一小部分固定数量$k$。

对于给定的token，我们识别出$g(x)$中Top $k$个值的索引，并将所有其他门控值设为零。这就创建了一个稀疏门控向量$G(x)$。$k$的选择是一个重要超参数 (parameter) (hyperparameter)。在Switch Transformers中，$k=1$，意味着每个token被路由到一个专家。更常见的选择是$k=2$，这为学习更复杂的函数提供了途径，并增加了一定程度的冗余。

这一操作有效地修剪了每个token的计算图。如果$k=2$且我们有$N=64$个专家，我们只需对其中2个进行前向计算，忽略其余62个。

### 专家网络

$N$个专家中的每一个通常是一个独立的自前馈网络（FFN）。尽管它们都共享相同的架构，但它们不共享权重 (weight)。每个专家$E_i$都有一套自己的参数 (parameter)。一个标准的双层FFN专家可以表示为：


$$
E_i(x) = \text{ReLU}(x \cdot W_{1,i}) \cdot W_{2,i}
$$


这里，$W_{1,i}$和$W_{2,i}$分别是专家$i$的第一和第二线性层的权重矩阵。正是这种独立的专家权重集合导致模型总参数量大幅增加。

### 完整的前向计算

我们现在可以结合这些步骤来定义MoE层的最终输出$y(x)$。输出是来自所选专家的输出的加权和，使用`TopK`操作得到的稀疏门控权重 (weight)。


$$
y(x) = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)
$$


因为$G(x)$是稀疏的，只有$k$个非零值，这个求和是计算高效的。我们只需计算路由器选择的$k$个专家的$E_i(x)$。

单个token的完整数据流可以如下所示：

> 单个token通过MoE层的数据流。输入`x`被送往门控网络以生成稀疏权重，并同时送往所选专家进行处理。

### 关于可微分性和权重 (weight)重新归一化 (normalization)的说明

`TopK`函数是不可微分的，这给反向传播 (backpropagation)带来了问题。实际中，这通过直通估计器处理。在前向计算时，我们应用离散的`TopK`选择。在反向传播时，我们使梯度通过顶部$k$个门，就好像选择是一个简单的乘法。稠密门控输出$g(x)$用于计算门控权重$W_g$的梯度。

此外，从初始softmax输出$g(x)$中选择Top $k$个值后，它们的和不再保证为1。为了形成一个合适的凸组合，这些$k$个值通常会被重新归一化。这通常通过仅对$h(x)$中选定的Top $k$个logit应用第二次softmax来完成。这确保了最终求和中使用的权重准确反映其相对重要性并和为1。

这种表述提供了一个参数 (parameter)量庞大的模型，但每个token的计算成本是恒定的，由$k$而非$N$决定。然而，这种精巧的结构带来了一个重大挑战：如果门控网络学习到将大多数token路由到少数几个专家，其他专家将不会收到训练信号。这导致了专家退化问题，我们接下来通过引入负载均衡损失来解决。

## 参考资料

- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/v23/120.html) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 23; Pages: 1-39; DOI: [10.55979/v23/21-0990](https://doi.org/10.55979/v23/21-0990)
  介绍了Switch Transformer架构，该架构显著推动了稀疏MoE的实际应用，并讨论了Top-K门控（k=1和k=2）和负载均衡策略。
