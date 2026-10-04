---
course: "mixture-of-experts-advanced-implementation"
chapter: "advanced-routing-mechanisms"
lesson: "hash-based-routing"
sourceId: 7054
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-2-advanced-routing-mechanisms/hash-based-routing"
title: "基于哈希的确定性选择路由"
description: "考察基于哈希的路由，作为学习型门控的一种确定性替代方案，用于将令牌分配给专家。"
order: 3
plots: ["plots/7054-0.json"]
sourceHash: "632d429e586e4bea0dfee2cc624a59c17ce9cfb811c71ab16c27962a6814ec3a"
sourceCorrections: []
---

尽管学习型门控网络提供动态路由，但它们也带来了一系列复杂性，包括可训练参数 (parameter)、潜在不稳定性和对辅助损失函数 (loss function)的需求。另一种方法通过从路由器中移除学习过程来避免这些问题。基于哈希的路由是一种确定性方法，它使用固定的、非学习的函数将令牌分配给专家，以此来换取绝对的简单和稳定性，尽管这可能牺牲智能专业化的潜力。

### 基于哈希的路由机制

其核心思想是，基于哈希的路由很简单。我们不通过线性层来计算路由逻辑值，而是对令牌的某个属性应用标准的哈希函数。然后，使用模运算将所得的哈希值映射到专家索引。

单个令牌 $x$ 的处理流程是：

1. **选择特征：** 选择要进行哈希处理的令牌特征。这通常是令牌在词汇表 (vocabulary)中的唯一ID，但也可以是其ID和在序列中位置的组合。
2. **应用哈希函数：** 计算所选特征的哈希值。我们将哈希函数表示为 $H$。
3. **确定专家索引：** 使用模运算符和专家总数 $N$ 来获取分配的专家索引 $E_i$。


$$
E_i = H(\text{令牌特征}) \pmod N
$$


此操作计算量很小，无需可训练权重 (weight)。相同的输入令牌特征将始终映射到同一个专家，使得路由决策在训练和推理 (inference)过程中保持静态且可预测。

> 基于哈希的路由流程图。令牌的特征通过哈希函数和模运算符，以确定性方式选择一个专家。

### 原理：为何选择确定性路由？

选择非学习型路由器可能看似违反直觉，因为它放弃了模型学习智能路由模式的能力。然而，这种方法提供了重要的工程优势。

#### 简单性和参数 (parameter)效率

最明显的优点是完全取消了门控网络。这意味着：

- **零路由器参数：** 模型变得稍微更小，占用更少内存，因为没有与路由器相关的权重 (weight)或偏置 (bias)。
- **无路由器计算：** 前向传播避免了标准门控网络所需的矩阵乘法，取而代之的是快得多的哈希和模运算。

#### 固有的负载均衡

一个选择得当的哈希函数能将令牌自然地均匀分布到所有可用专家上。这种统计均匀性意味着，在大量令牌批次中，每个专家预计会收到类似数量的分配。

此特性直接解决了困扰学习型路由器的负载不平衡问题。因此，**辅助负载均衡损失不再必需**。总损失函数 (loss function)简化为仅包含主要任务损失（例如，交叉熵）：


$$
L_{总} = L_{任务}
$$


通过移除辅助损失及其相关超参数 (hyperparameter)，我们也消除了训练不稳定性的常见来源，例如上一章讨论的路由器z-loss问题。

### 不可避免的权衡：性能与简单性

基于哈希的路由的主要缺点是它对模型性能的影响。学习型门控允许模型形成语义专业化；例如，一个专家可能擅长处理标点和语法，而另一个则专注于科学术语。路由器学习将正确的令牌发送给正确的专家。

基于哈希的路由打破了这种联系。由于分配是伪随机的，每个专家都被迫成为通才。它必须学习处理整个数据分布的随机子集，从而阻止精细专业化的形成。与使用训练有素的学习型路由器的稀疏MoE模型相比，这通常会导致模型质量较低（例如，更高的困惑度或更低的准确率）。

下面的图表显示了这种权衡。尽管基于哈希的MoE比密集模型提供了更好的性能与计算比，但它通常不如具有学习型路由的标准MoE模型。



![模型质量与计算成本](plots/7054-0.json)



> 密集模型、学习型MoE和基于哈希的MoE模型质量比较。基于哈希的路由提供了一个折中方案，它在密集模型的基础上有所改进，但未能达到学习型、专业化路由的性能水平。

在实际应用中，基于哈希的路由是一个重要的实验基线。它有助于区分性能提升是来源于学习型稀疏激活，还是仅仅因为增加了更多参数 (parameter)。如果一个复杂的学习型MoE模型无法超越基于哈希的同等模型，这通常表明训练设置或路由机制本身存在问题。

## 参考资料

- [Sparsely-Gated Mixture-of-Experts Layers](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了现代稀疏专家混合架构及其学习门控机制，并探讨了负载均衡挑战。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2021)
  Journal: Journal of Machine Learning Research (JMLR); Volume: 22; Pages: 1-36; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  该文关注专家混合模型向超大规模参数的扩展，提供了关于MoE设计、训练和效率的实用见解，有助于理解路由选择的背景。
