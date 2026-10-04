---
course: "nlp-fundamentals"
chapter: "nlp-word-embeddings"
lesson: "word-embedding-concepts"
sourceId: 2497
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-4-nlp-word-embeddings/word-embedding-concepts"
title: "词嵌入基本原理"
description: "理解将词语表示为连续空间中稠密向量的核心思路。"
order: 3
plots: ["plots/2497-0.json"]
sourceHash: "2a14ab689370c26cc0a9d1baaee67fc6bda89202c4ff374ffd96e9265c69efa4"
sourceCorrections: []
---

TF-IDF 等方法基于词频表示文档。尽管有用，但这些表示将词语视为独立的单元。它们本质上并不理解“猫”和“小猫”是相关的，或者“跑步”和“慢跑”语义相近。如果两个相关词语在训练集中未出现在非常相似的文档中，基于频率的模型会认为它们完全不同。这限制了它们对需要对语言含义有更细致认识的任务的效用。

为了解决这个问题，我们需要一种能捕获词语语义关联 (semantic relationship)的表示方法。这引出了**词嵌入 (embedding)**的思路，它建立在**分布式语义**的理念之上。其核心主张常概括为“知词识伴”，指明了出现在相似语境中的词语很可能含义相关。例如，“咖啡”和“茶”等词语经常出现在“一杯\_\_\_\_”、“喝了一些\_\_\_\_”或“热\_\_\_\_”等语境中。通过分析大量文本中的这些共现模式，我们可以学到一种表示，使得“咖啡”和“茶”在空间中比“咖啡”和“C++编程语言”更接近。

### 从稀疏向量 (vector)到稠密向量

不同于词袋模型或TF-IDF等技术生成的高维稀疏向量（这些向量大部分元素为零，且长度等于词汇量大小），词嵌入 (embedding)将词语表示为低维空间中的**稠密向量**。通常，一个词 $w$ 会被映射到一个向量 $v_w \in \mathbb{R}^d$。其中 $d$ 是嵌入维度（一个超参数 (parameter) (hyperparameter)，通常在50到300之间）。


$$
v_{\text{词语}} = [x_1, x_2, \dots, x_d]
$$


此向量中的每个维度 $x_i$ 都表示从数据中学到的词语含义的潜在特征。与人工设计的特征不同，这些维度通常无法用人类可理解的术语来解释（例如，维度1不明确表示“是动物”或“可食用”）。相反，它们共同捕获了词语含义、用法和关联性的复杂而精妙之处，这些信息源自词语在训练语料库中的上下文 (context)模式。重点是，*整个向量*表示了词语与其他词语之间的含义关联。

### 几何关联的威力

词嵌入 (embedding)的魅力在于这些向量 (vector)在$d$维空间中的几何关联。

1. **语义相似性：** 含义相似的词语，它们的向量往往相互接近。我们可以使用欧氏距离等距离度量来衡量这种接近程度，更常用的是**余弦相似度**。余弦相似度衡量的是两个向量之间夹角的余弦值，范围从-1（含义相反）到1（含义/语境相同），0表示不相关（正交）。

   
   $$
   \text{相似度}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{d} A_i B_i}{\sqrt{\sum_{i=1}^{d} A_i^2} \sqrt{\sum_{i=1}^{d} B_i^2}}
   $$
   

   因此，“汽车”和“轿车”的向量应该具有较高的余弦相似度，而“汽车”和“香蕉”的相似度应该较低。
2. **语义类比：** 值得注意的是，这些向量空间常能捕捉到关联相似性。经典的例子是类比：“男人之于女人，犹如国王之于女王。”在向量空间中，这种关系可能会表示为：

   
   $$
   v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}
   $$
   

   这表明向量差 $v_{\text{king}} - v_{\text{man}}$ 捕捉到了类似“王权减去男性特征”这样的属性，加上 $v_{\text{woman}}$ 后，结果向量会非常接近 $v_{\text{queen}}$。类似的类比，如“法国之于巴黎，犹如德国之于柏林”（$v_{\text{Paris}} - v_{\text{France}} + v_{\text{Germany}} \approx v_{\text{Berlin}}$），也能够被观察到。

设想一个简化的2D可视化图，显示某些词语在嵌入空间中彼此间的相对位置：



![2D词嵌入空间](plots/2497-0.json)



> 这是一个简化的图示，说明了相关词语（国王/男人、女王/女人、苹果/橙子）如何在嵌入空间中聚类，并与不相关词语（汽车）分开。实际的嵌入空间维度要高得多。

### 下游任务的输入

这些稠密向量 (vector)表示可作为各种应用于自然语言处理任务的机器学习 (machine learning)模型的有效输入特征。我们可以使用相应的词嵌入 (embedding)，而非将独热编码向量或TF-IDF分数输入到分类器或序列模型中。这通常会带来显著的性能提升，因为模型接收的特征已经包含了语义信息，从而减轻了模型从头开始学习这些关联的负担。

在接下来的章节中，我们将研究Word2Vec和GloVe等特定算法，它们能从大型文本语料库中学习这些向量表示。我们还将了解如何使用预计算的嵌入，这能节省大量的训练时间，并凭借从海量数据集中获取的知识。

## 参考资料

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean (2013)
  DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  这篇论文介绍了Word2Vec模型，它通过高效地生成高质量的稠密向量表示来捕捉词语的语义和句法关系，从而推动了词嵌入学习的发展。
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) — Jeffrey Pennington, Richard Socher, Christopher Manning (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1532-1543; DOI: [10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)
  这篇论文提出了GloVe，这是一种将全局矩阵分解和局部上下文窗口方法相结合的词嵌入方法，用于从词语共现统计中学习向量表示。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  一本全面且权威的教材，涵盖了自然语言处理的基础理论和实际应用，其中包含关于词嵌入及相关概念的详细章节。（第四版草稿）
