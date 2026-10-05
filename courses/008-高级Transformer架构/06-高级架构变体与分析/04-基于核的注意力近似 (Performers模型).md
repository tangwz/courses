# 基于核的注意力近似 (Performers模型)

来源：[原文](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/kernel-based-attention-performers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管标准自注意力 (self-attention)提供了出色的序列建模能力，但其二次方的计算和内存复杂度（序列长度为 $N$ 时为 $O(N^2)$）仍然是一个主要的限制，尤其是在处理高分辨率图像生成、文档摘要或基因组数据分析等场景中遇到的长序列时。Choromanski 等人 (2020) 提出的 Performer 架构，通过以线性复杂度近似注意力机制 (attention mechanism)，提供了一种有效的解决办法。它通过一种巧妙的技术，称为“通过正交随机特征实现快速注意力”（FAVOR+），实现了这一点。

### 限制：显式注意力矩阵计算

回顾标准缩放点积注意力：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


这里，$Q, K, V$ 分别是查询、键和值向量 (vector)的矩阵，每个矩阵有 $N$ 行（序列长度）和 $d_k$（或 $d_v$）列（嵌入 (embedding)维度）。主要计算成本在于计算 $N \times N$ 的注意力矩阵 $A = \text{softmax}(QK^T / \sqrt{d_k})$。这需要 $O(N^2 d_k)$ 次操作和 $O(N^2)$ 内存，随着 $N$ 的增加，这很快变得难以承受。

### 使用核与随机特征近似注意力

Performer 模型的核心思路是避免显式计算 $N \times N$ 矩阵 $A$。相反，它寻求使用核方法来近似注意力机制 (attention mechanism)。让我们将注意力机制的第 $i$ 个输出向量 (vector)（归一化 (normalization)之前）重写为：


$$
\text{注意力}'(Q, K, V)_i = \sum_{j=1}^N \exp\left(\frac{q_i^T k_j}{\sqrt{d_k}}\right) v_j
$$


标准注意力输出通过使用分母 $D_i = \sum_{j=1}^N \exp(q_i^T k_j / \sqrt{d_k})$ 对此和进行归一化而得到。

Performer 模型利用了函数 $\text{相似度}(q_i, k_j) = \exp(q_i^T k_j / \sqrt{d_k})$ 类似于核函数的思路，特别是在经过适当缩放和变换后的高斯核。核方法通常允许使用特征映射隐式计算相似度。Performer 模型提出寻找一个特征映射 $\phi: \mathbb{R}^{d_k} \rightarrow \mathbb{R}^r$，其中 $r$ 是随机特征的维度（通常 $r \ll N$），使得核相似度可以通过特征空间中的内积来近似：


$$
\exp\left(\frac{q_i^T k_j}{\sqrt{d_k}}\right) \approx \phi(q_i)^T \phi(k_j)
$$


如果存在这样的特征映射 $\phi$，注意力计算可以巧妙地重新排序，以避免 $N \times N$ 的计算。未归一化的注意力求和变为：


$$
\text{注意力}'(Q, K, V)_i \approx \sum_{j=1}^N (\phi(q_i)^T \phi(k_j)) v_j = \phi(q_i)^T \sum_{j=1}^N (\phi(k_j) v_j^T)
$$


类似地，分母可以近似为：


$$
D_i \approx \sum_{j=1}^N \phi(q_i)^T \phi(k_j) = \phi(q_i)^T \sum_{j=1}^N \phi(k_j)
$$


请注意操作顺序的重要变化。我们可以首先计算求和 $\sum_{j=1}^N (\phi(k_j) v_j^T)$（一个 $r \times d_v$ 矩阵）和 $\sum_{j=1}^N \phi(k_j)$（一个 $r \times 1$ 向量）。这些计算只需处理键和值一次，大约需要 $O(N r d_k + N r d_v)$ 的时间。然后，对于每个查询 $q_i$，我们计算 $\phi(q_i)$ 并执行两次矩阵-向量乘法，每个查询需要 $O(r d_k + r d_v)$ 的时间。总时间复杂度近似为 $O(N r (d_k + d_v))$，如果 $r$ 被视为常数或增长远慢于 $N$，则在 $N$ 上呈线性关系。内存复杂度也降低到 $O(N r + N d_k + N d_v)$，主要用于存储映射后的查询、键、值和中间求和结果。

> 计算流程对比。标准注意力需要形成代价高昂的 $N \times N$ 矩阵（红色节点）。Performer 模型使用特征映射 $\phi$ 来计算中间求和（黄色节点），这些操作具有线性复杂度 $O(N)$，从而避免了二次方的限制。

### 构建特征映射 $\phi$

Performer 模型的有效性取决于能否找到一个合适的特征映射 $\phi$。FAVOR+ 机制通过使用随机特征构建 $\phi$ 来实现这一点，其灵感来源于用于近似高斯核的技术（如随机傅里叶特征）。Performer 模型的一个重要贡献是开发了能保证非负性（$\phi(x)^T \phi(y) \ge 0$）的特征映射，这有助于保持稳定性并更好地近似 softmax 函数的特性（其输出分量是非负的）。

具体来说，Performer 模型基于随机投影和三角函数定义特征映射，例如：


$$
\phi(x) = \frac{h(x)}{\sqrt{m}} \left[ f_1(w_1^T x), \dots, f_m(w_m^T x), f'_1(w_1^T x), \dots, f'_m(w_m^T x) \right]^T
$$


其中 $w_i$ 是随机采样的向量 (vector)，$h(x)$ 是像 $\exp(\|x\|^2 / 2)$ 这样的函数，而 $f_l, f'_l$ 是像 $(\sin, \cos)$ 或 $(\exp, \exp)$ 这样的对。特征映射 $r$ 的维度是 $2m$。精确的构造确保了 softmax 核的无偏或近无偏估计，并具有非负输出。

### 优点与注意事项

**优点：**

- **线性复杂度：** 将注意力的时间和空间复杂度从 $O(N^2)$ 降低到 $O(N)$，使得能够应用于更长的序列。
- **强大的理论保证：** 为 softmax 核提供了可证明的近似界限。
- **良好的实证表现：** 通常能达到与标准 Transformer 模型相当的性能，同时在长序列任务上效率显著提高。
- **兼容性：** 可以作为标准注意力模块的替代品进行集成，对整个 Transformer 架构的修改很小。

**注意事项：**

- **近似质量与成本：** 近似质量取决于随机特征维度 $r$。较大的 $r$ 会产生更好的近似，但会增加线性复杂度中的常数因子（$O(N r d)$）。选择 $r$ 需要在精度和计算成本之间进行权衡。 $r$ 的典型值可能在 64 到 256 甚至更高，具体取决于任务和资源。
- **随机性：** 使用随机特征会引入随机性。然而，方差通常较低，特别是在 $r$ 的选择合理时，并且 Performer 模型采用正交随机特征等技术来进一步降低这种方差。在实际应用中，不同随机种子下的结果通常是稳定的。

通过使用非负随机特征近似 softmax 核，Performer 模型提供了一种计算高效且有理论依据的方法，能够将 Transformer 模型扩展到以前认为不可行的长度，显著拓宽了其应用范围。

## 参考资料

- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) — Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller (2021)
  Journal: International Conference on Learning Representations (ICLR 2021); DOI: [10.48550/arXiv.2009.14794](https://doi.org/10.48550/arXiv.2009.14794)
  介绍Performer架构和FAVOR+机制的原始研究论文，实现了线性时间复杂度的注意力近似。
- [Random Features for Large-Scale Kernel Machines](https://proceedings.neurips.cc/paper/2007/hash/0cf11cce2009ad49d79677e112d6a8ab-Abstract.html) — Ali Rahimi, Benjamin Recht (2007)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS Foundation; Volume: 20; Pages: 1177-1184; DOI: [10.5591/978-1-57735-703-5.2016.1177](https://doi.org/10.5591/978-1-57735-703-5.2016.1177)
  一篇开创性论文，介绍了随机傅里叶特征的概念，用于近似平移不变核函数，是Performer核近似的直接灵感来源。
- [Efficient Transformers: A Survey](https://dl.acm.org/doi/10.1145/3530811) — Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler (2022)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery (ACM); Volume: 55; Pages: 1-28; DOI: [10.1145/3530811](https://doi.org/10.1145/3530811)
  全面概述了提高Transformer模型效率的各种技术，包括Performer等不同的线性注意力机制。

---

[上一节](03-%E8%BF%91%E4%BC%BC%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%EF%BC%9A%E7%BA%BF%E6%80%A7Transformer.md) · [下一节](05-%E4%BD%8E%E7%A7%A9%E6%8A%95%E5%BD%B1%E6%96%B9%E6%B3%95%EF%BC%88Linformer%EF%BC%89.md)
