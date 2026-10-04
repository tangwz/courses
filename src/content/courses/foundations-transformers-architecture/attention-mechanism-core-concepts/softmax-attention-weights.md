---
course: "foundations-transformers-architecture"
chapter: "attention-mechanism-core-concepts"
lesson: "softmax-attention-weights"
sourceId: 1943
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-2-attention-mechanism-core-concepts/softmax-attention-weights"
title: "注意力权重的Softmax函数"
description: "softmax函数在将注意力分数归一化为概率分布中的作用。"
order: 5
plots: ["plots/1943-0.json"]
sourceHash: "89662663db87ed96ff2931ed4a71f868816a2ce5b88c57d558bccd59e69c0863"
sourceCorrections: []
---

查询向量 (vector) ($Q$) 和键向量 ($K$) 之间的缩放点积分数计算会产生一个原始对齐 (alignment)分数矩阵。此计算结果为 $\frac{QK^T}{\sqrt{d_k}}$。尽管这些分数反映了查询和键向量之间的匹配程度，但它们未经归一化 (normalization)处理，值域可以是任意范围，这使得它们难以直接被视为贡献权重 (weight)。

为了将这些原始分数转换为一组可用的、表示注意力分布的权重，我们对分数矩阵的每一行独立应用softmax函数。对于特定的查询 $q_i$ (对应 $Q$ 的第 $i$ 行)，它与键 $k_j$ (对应 $K^T$ 的第 $j$ 列) 对齐的原始分数记作 $s_{ij} = \frac{q_i k_j^T}{\sqrt{d_k}}$。softmax函数将查询 $q_i$ 与所有 $N$ 个键的分数向量 $s_i = [s_{i1}, s_{i2}, ..., s_{iN}]$ 转换为注意力权重向量 $\alpha_i = [\alpha_{i1}, \alpha_{i2}, ..., \alpha_{iN}]$，其中每个权重 $\alpha_{ij}$ 的计算方式如下：


$$
\alpha_{ij} = \text{softmax}(s_{ij}) = \frac{\exp(s_{ij})}{\sum_{l=1}^{N} \exp(s_{il})}
$$


这里，$N$ 代表键/值对的序列长度。

### 特性和解释

应用softmax函数会为得到的注意力权重 (weight) $\alpha_{ij}$ 带来几个重要特性：

1. **归一化 (normalization)：** 分母 $\sum_{l=1}^{N} \exp(s_{il})$ 保证了给定查询 $q_i$ 在所有键上的注意力权重总和为1。即 $\sum_{j=1}^{N} \alpha_{ij} = 1$。这使得我们可以将这些权重理解为概率分布。
2. **非负性：** 由于指数函数 $\exp(x)$ 对于任何实数输入 $x$ 总是正数，每个单独的注意力权重 $\alpha_{ij}$ 都保证是正数。
3. **概率解释：** 对于查询 $q_i$ 的权重集合 $\{\alpha_{i1}, \alpha_{i2}, ..., \alpha_{iN}\}$ 代表了在输入序列上的注意力概率分布。权重 $\alpha_{ij}$ 表示模型在计算第 $i$ 个位置的输出表示时，分配给第 $j$ 个输入元素 (由其值向量 (vector) $v_j$ 代表) 的注意力比例。
4. **突出重要性：** 指数函数本质上会放大更大的分数，使其比小分数更为突出。如果分数 $s_{ik}$ 在行 $s_i$ 中明显大于其他分数，其对应的权重 $\alpha_{ik}$ 将接近1，而其他权重将接近0。这种机制使得模型能够有效地关注通过点积评分确定的最相关输入元素。

### 在计算最终输出中的作用

这些计算出的注意力权重 (weight) $\alpha_{ij}$ 是用于计算值向量 (vector) ($V$) 加权和的系数。注意力机制 (attention mechanism)对第 $i$ 个查询的输出，是通过将注意力权重分布 $\alpha_i$ 与值矩阵 $V$ 相乘而获得的：


$$
\text{输出}_i = \sum_{j=1}^{N} \alpha_{ij} v_j
$$


在矩阵表示中，这直接对应于缩放点积注意力公式的最后一步：


$$
\text{注意力}(Q, K, V) = A V = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


其中 $A$ 是注意力权重矩阵，其元素为 $A_{ij} = \alpha_{ij}$。因此，softmax函数在将原始相似性分数转换为归一化 (normalization)分布方面起着重要作用，该分布决定了如何聚合来自输入序列不同部分 (由值向量表示) 的信息以形成输出。

### 可视化示例

考虑一个简化的例子，单个查询有4个原始对齐 (alignment)分数：$s = [1.0, 0.5, 2.5, -0.1]$。应用softmax函数会转换这些分数：

- $\exp(1.0) \approx 2.718$
- $\exp(0.5) \approx 1.649$
- $\exp(2.5) \approx 12.182$
- $\exp(-0.1) \approx 0.905$

指数之和为 $2.718 + 1.649 + 12.182 + 0.905 = 17.454$。

得到的softmax权重 (weight)是：

- $\alpha_1 = 2.718 / 17.454 \approx 0.156$
- $\alpha_2 = 1.649 / 17.454 \approx 0.094$
- $\alpha_3 = 12.182 / 17.454 \approx 0.698$
- $\alpha_4 = 0.905 / 17.454 \approx 0.052$

请注意，最高的原始分数 (2.5) 对应着主导的注意力权重 (0.698)，在此示例中，有效地将注意力机制 (attention mechanism)集中在第三个输入元素上。这些权重之和约为1 ($0.156 + 0.094 + 0.698 + 0.052 = 1.000$)。



![原始分数与Softmax注意力权重对比](plots/1943-0.json)



> 单个查询的原始对齐分数与应用softmax函数后得到的注意力权重的对比。Softmax将分数转换为概率分布，突出显示最相关的位置。

总之，softmax函数在注意力机制中是一个重要的归一化 (normalization)步骤。它将原始对齐分数转换为概率分布，使得模型能够基于查询-键相似性，选择性地加权并组合来自值向量 (vector)的信息，为Transformer如何处理序列信息奠定基础。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构和缩放点积注意力机制的原始论文，包含softmax的作用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6, Section 6.2.2.3 (Output Units)
  基础教科书，详细解释了softmax函数的数学原理和特性。
- [Stanford CS224N: Natural Language Processing with Deep Learning, Lecture 10: Transformers](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture08-transformers.pdf) — John Hewitt, Anna Goldie (2023)
  Publisher: Stanford University
  课程讲义，提供了Transformer和注意力机制的教学概述，说明了softmax的应用。
- [Neural Networks and Deep Learning, Chapter 3: Improving the way neural networks learn (Softmax)](http://neuralnetworksanddeeplearning.com/chap3.html#softmax) — Michael Nielsen (2019)
  Publisher: Determination Press
  易读的在线书籍，解释了softmax函数，其作为概率分布的解释以及实际应用。
