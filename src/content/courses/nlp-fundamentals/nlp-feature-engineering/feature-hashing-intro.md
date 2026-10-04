---
course: "nlp-fundamentals"
chapter: "nlp-feature-engineering"
lesson: "feature-hashing-intro"
sourceId: 2484
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-2-nlp-feature-engineering/feature-hashing-intro"
title: "特征哈希简介"
description: "了解特征哈希（哈希技巧）作为一种内存高效的特征表示方法。"
order: 4
plots: []
sourceHash: "a9e2931d9aa7e04247d997b74fca89f4323892480aba7820b0dc08ce2b0e36a4"
sourceCorrections: []
---

TF-IDF和N-gram等方法在将文本数字表示时，通常会产生维度极高的特征向量 (vector)。试想一个包含数十万甚至数百万个独特词语或N-gram的词汇表 (vocabulary)。存储和处理拥有如此多列的矩阵，需要大量的内存和计算资源。在处理大型数据集或流式数据（其中完整词汇表可能并非预先已知）时，这一问题尤为突出。

特征哈希，有时被称为“哈希技巧”，提供了一种巧妙的替代方法。它不构建和存储显式的词汇映射（如词语 -> 列索引），而是使用哈希函数将特征名称（如词语或N-gram）直接映射到固定大小向量中的索引。

### 哈希机制

特征哈希的核心工作方式如下：

1. **选择向量 (vector)大小：** 您预先定义输出向量所需的维度（列）数量，我们称之为 $D$。这个大小通常远小于潜在的词汇量。
2. **应用哈希函数：** 对于每个特征（例如，“predict”这个词或“machine learning”这个二元组），应用一个哈希函数（例如MurmurHash3），它会输出一个整数。
3. **确定索引：** 取哈希函数输出的绝对值，并计算其除以所选向量大小 $D$ 的余数。这会给出一个介于0和 $D-1$ 之间的索引。
   
   $$
   索引 = \text{哈希}(\text{"特征名称"}) \pmod{D}
   $$
   
4. **更新向量：** 与特征关联的值（例如，其词频或TF-IDF分数）随后被添加到输出向量中该计算出的 `index` 位置的值上。

考虑一个目标向量大小为 $D=10$ 的简单例子：

- `哈希("learning") % 10 = 3`
- `哈希("algorithm") % 10 = 7`
- `哈希("model") % 10 = 3`

如果我们的文档中每个词出现一次（TF=1），那么我们得到的向量（最初全为零）可能如下所示：
`[0, 0, 0, 2, 0, 0, 0, 1, 0, 0]`

请注意，“learning”和“model”都哈希到了索引3。这被称为**哈希冲突**。

> 此图说明了不同特征如何通过哈希函数映射到固定大小向量中的索引。请注意“learning”和“model”映射到相同索引（3）的冲突。

### 处理冲突

冲突是特征哈希的固有属性。多个不同的特征可能会映射到输出向量 (vector)中的同一个索引。这些发生冲突的特征的计数或权重 (weight)会简单地在该索引处累加。

- **权衡：** 目标向量大小 $D$ 越小，节省的内存越多，但发生冲突的概率也越高。
- **影响：** 尽管冲突意味着会丢失一些信息（模型无法区分哈希到同一索引的特征），但在实践中，对于像文本这样的稀疏高维数据，对模型性能的影响通常出乎意料地小，特别是当 $D$ 足够大时（例如 $2^{18}$ 到 $2^{24}$）。固有的稀疏性意味着许多索引无论如何可能只有一个特征映射到它们。
- **带符号哈希：** 一种常见的改进方法是使用第二个哈希函数来确定添加到索引的值的*符号*（+1 或 -1）。这样，冲突的特征有机会部分相互抵消，而不是总是正向累加，这有时可以减轻冲突的负面影响。

### 特征哈希的优势

1. **内存效率：** 与存储完整的词汇映射相比，它所需的内存显著减少，特别是对于庞大的特征集合。内存需求由所选的向量 (vector)维度 $D$ 固定。
2. **在线学习：** 它非常适合数据顺序到达（流式）的场景。您不需要预先看到整个数据集来构建词汇表 (vocabulary)；后来遇到的新特征可以直接哈希到现有的向量空间中。
3. **简单性：** 它避免了管理词汇字典的复杂性。

### 特征哈希的缺点

1. **可解释性：** 结果特征向量 (vector)难以解释。索引 `i` 处的激活可能源于任何哈希到 `i` 的特征。您会失去从向量索引直接回溯到特定词语或N-gram的映射关系。
2. **冲突影响：** 如果哈希空间 ($D$) 相对于唯一特征的数量过小，过多的冲突会通过将过多不相关的特征映射到一起而降低模型性能。选择合适的 $D$ 通常需要通过实践确定。

特征哈希为传统的基于词汇的向量化 (quantization)方法（如词袋模型或TF-IDF）提供了一种强大的、内存高效的替代方案，在大规模或资源受限的环境中尤其有价值。它在向量化过程中*隐式地*实现降维，这与PCA或SVD等在创建可能很大的初始特征矩阵*之后*应用的技巧不同。

## 参考资料

- [Feature Hashing for Large Scale Multitask Learning](https://arxiv.org/pdf/0902.2206.pdf) — Kilian Weinberger, Anirban Dasgupta, John Langford, Alex Smola, Josh Attenberg (2009)
  Journal: Proceedings of the 26th International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research (PMLR); Pages: 1-4; DOI: [10.48550/arXiv.0902.2206](https://doi.org/10.48550/arXiv.0902.2206)
  这篇开创性论文介绍了并推广了“哈希技巧”，用于机器学习中高效的特征表示，解决了高维数据的可伸缩性问题。
- [Mining of Massive Datasets](http://www.mmds.org/) — Jure Leskovec, Anand Rajaraman, Jeff Ullman (2014)
  Publisher: Cambridge University Press
  一本关于大规模数据分析技术的综合教科书，其中包含了基于哈希的降维方法讨论，为特征哈希提供了更广阔的背景。
- [sklearn.feature_extraction.FeatureHasher](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html) — scikit-learn developers (2023)
  scikit-learn中特征哈希实现的官方文档，提供了实际使用细节、参数说明以及具体的应用示例。
