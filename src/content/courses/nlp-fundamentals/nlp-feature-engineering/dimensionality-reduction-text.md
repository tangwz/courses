---
course: "nlp-fundamentals"
chapter: "nlp-feature-engineering"
lesson: "dimensionality-reduction-text"
sourceId: 2485
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-2-nlp-feature-engineering/dimensionality-reduction-text"
title: "文本特征的降维"
description: "应用主成分分析（PCA）和奇异值分解（SVD）等方法到文本特征。"
order: 5
plots: []
sourceHash: "505ba9e0d0d03d0373134be2e82768a9f7912aab2d1918b227e08089cd574582"
sourceCorrections: []
---

TF-IDF和N-gram等方法能将文本转换为数值特征向量 (vector)。然而，这通常会生成维度极高的数据。如果词汇表 (vocabulary)包含数万或数十万个独立词项，TF-IDF矩阵将拥有同样多的列。处理这类高维稀疏矩阵（大多由零填充的矩阵）会带来一些难题：

1. **计算成本：** 在非常宽的矩阵上训练机器学习 (machine learning)模型，计算开销大且占用大量内存。
2. **维度灾难：** 高维度会增加算法识别模式的难度，可能需要指数级增长的数据才能实现良好的泛化。高维空间 (high-dimensional space)中的点往往稀疏且距离很远。
3. **噪声：** 许多维度可能包含很少有用的信息，甚至包含噪声，这会影响模型表现。

降维方法旨在将这些高维特征向量转换为低维表示，同时尽量保留有意义的信息。这能缩短训练时间、减少内存占用，有时通过去除噪声甚至能改进模型表现。我们来看看为此目的使用的两种常见线性代数方法：主成分分析（PCA）和奇异值分解（SVD）。

## 主成分分析（PCA）

PCA是一种广泛应用的降维方法。它的核心思路是找出数据中方差最大的方向（主成分）。然后，它将原始数据投影到一个由选定数量的主成分定义的新、低维子空间上。

PCA执行以下步骤：

1. **标准化：** 它通常要求数据进行中心化（从每个特征中减去均值）。
2. **协方差矩阵：** 它计算标准化数据的协方差矩阵。
3. **特征值分解：** 它计算协方差矩阵的特征向量 (vector)和特征值。特征向量表示主成分（最大方差方向），特征值表示每个成分捕获的方差量。
4. **投影：** 它选择与最大特征值对应的最主要的 $k$ 个特征向量，并将原始标准化数据投影到这些特征向量所张成的子空间上。结果是一个只有 $k$ 个维度的新特征矩阵。

尽管PCA功能强大，但直接将其应用于稀疏的TF-IDF矩阵可能会有问题。初步的标准化步骤（中心化）需要从每个数据点中减去每个特征（列）的均值。由于TF-IDF矩阵通常非常稀疏（大多为零），减去非零均值会使矩阵变得密集，可能导致内存问题，从而抵消从稀疏表示开始的优势。

因此，尽管PCA是一种基本的降维方法，但在直接处理TF-IDF等稀疏文本特征时，通常更倾向于选择SVD的变体。

## 奇异值分解（SVD）

奇异值分解是另一种强大的矩阵分解方法，在降维方面有应用。SVD将原始矩阵 $X$ （例如，我们大小为 $m \times n$ 的TF-IDF矩阵，$m$ 是文档数量，$n$ 是词项数量）分解为三个独立的矩阵：


$$
X = U \Sigma V^T
$$


此处：

- $U$ 是一个 $m \times m$ 的正交矩阵（文档-主题矩阵）。
- $\Sigma$ 是一个 $m \times n$ 的对角矩阵，包含 $X$ 的奇异值，按降序排列。这些奇异值表示不同维度或“主题”的重要性。
- $V^T$ 是一个 $n \times n$ 的正交矩阵（词项-主题矩阵）。

为了降维，我们使用一种变体，称为**截断SVD**。截断SVD不是计算完全分解，而是只计算前 $k$ 个奇异值以及 $U$ 和 $V$ 中对应的向量 (vector)。


$$
X \approx U_k \Sigma_k V_k^T
$$


这里，$U_k$ 是 $m \times k$ 的，$\Sigma_k$ 是 $k \times k$ 的（包含前 $k$ 个奇异值），$V_k^T$ 是 $k \times n$ 的。原始数据的转换后的低维表示通常取为 $X_k = U_k \Sigma_k$（或有时仅为 $U_k$），这会生成一个 $m \times k$ 的矩阵。

> 截断SVD用于降维的概览。原始矩阵被分解，然后只保留与最大奇异值对应的最主要的 $k$ 个分量，以形成低维表示。

截断SVD的一个重要优点，尤其是scikit-learn等库中的实现（`TruncatedSVD`），是它能够直接处理稀疏矩阵，而不需要PCA所需的那个有问题的中心化步骤。这使得它非常适合NLP中常见的大型TF-IDF或计数矩阵。

将SVD应用于词项-文档矩阵也称为\*\*潜在语义分析（LSA）\*\*或潜在语义索引（LSI）。LSA旨在找出文本数据中潜在的语义结构（“主题”）。生成的维度（$k$）表示这些潜在主题，值表示每个文档与每个主题的关联强度。

## 维度数量的选择（k）

一个实际问题是如何选择要保留的维度数量，$k$。通常需要权衡：

- 维度太少可能导致大量信息丢失。
- 维度太多可能无法充分减少计算负担，或可能保留噪声。

常见的方法包括：

1. **解释方差：** 绘制主成分（对于PCA）或奇异值之和（对于SVD）解释的累积方差随 $k$ 增加的变化图。选择一个 $k$ 值，使其能够捕捉到总方差/信息中所需的百分比（例如，80-95%）。
2. **后续任务表现：** 将 $k$ 视为一个超参数 (parameter) (hyperparameter)。试验不同的 $k$ 值，并使用交叉验证评估最终机器学习 (machine learning)模型（例如，文本分类器）的表现。选择在验证集上产生最佳表现的 $k$ 值。
3. **计算限制：** 有时，$k$ 的选择取决于可用内存或期望的训练速度。

## 考量因素

尽管降维是有益的，但请记住以下几点：

- **可解释性：** PCA或SVD生成的新维度是原始特征（词项）的线性组合。它们通常缺乏原始TF-IDF分数或N-gram计数的直接可解释性。
- **信息丢失：** 通过减少维度，一些信息不可避免地会被丢弃。目标是丢弃噪声或冗余信息，同时保留最重要的模式。
- **算法选择：** 由于中心化问题，对于稀疏文本特征，通常更倾向于截断SVD而非标准PCA。

总之，降维方法，例如PCA以及在文本处理中更常见的截断SVD，是管理TF-IDF和N-gram等文本特征表示中固有的高维度问题的宝贵工具。通过将数据投影到低维空间，它们可以使后续的模型训练更有效，并通过关注最显著的信息来潜在地提高表现。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  全面涵盖统计学习方法，包括PCA和SVD作为降维技术的详细解释。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  Publisher: Online Draft
  自然语言处理的权威教材，其中包含详细介绍潜在语义分析 (LSA) 和SVD在文本特征应用的部分。
- [sklearn.decomposition.TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) — scikit-learn developers (2024)
  Publisher: scikit-learn
  scikit-learn TruncatedSVD的官方文档，解释了其用法、参数以及处理稀疏矩阵的能力，这对于文本特征至关重要。
