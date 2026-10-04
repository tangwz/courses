---
course: "nlp-fundamentals"
chapter: "nlp-word-embeddings"
lesson: "visualizing-word-embeddings"
sourceId: 2500
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-4-nlp-word-embeddings/visualizing-word-embeddings"
title: "词嵌入的可视化"
description: "将高维词向量在2D或3D空间中可视化的方法。"
order: 6
plots: ["plots/2500-0.json"]
sourceHash: "50265700490f1732b34964297ad674a9074c8bc31cc015feba307f146a21f107"
sourceCorrections: []
---

词嵌入 (embedding)，例如Word2Vec或GloVe生成的词嵌入，将词表示为稠密向量 (vector)，这些向量通常存在于数百维的空间中（例如，$w \in \mathbb{R}^{300}$）。尽管这些高维表示包含了丰富的语义信息，但其本质使我们人类无法直接观察。

我们如何才能直观地理解这些模型学习到的关系？我们如何验证预期相似的词语在嵌入空间中是否确实彼此靠近？

此时，降维技术就成为了不可或缺的工具。通过将高维词向量投影到二维或三维空间，我们可以创建散点图，帮助我们直观地检查学到的关联。为此，主成分分析（PCA）和t-分布随机邻域嵌入（t-SNE）是两种广泛使用的方法。

### 主成分分析（PCA）

PCA是一种线性技术，它识别数据中能够捕捉到最大方差的方向（主成分）。它将原始高维向量 (vector)转换到较低维空间，同时尽量保留最多的全局结构和方差。

设想将一个三维点云表示在二维纸张上。PCA会找到观察该点云的最佳“角度”，使得投影到纸上的点尽可能地分散，从而显示主要的变异轴。

尽管PCA在计算上高效且具有确定性（对于相同数据总是产生相同结果），但其对方差的侧重意味着它可能并非总能很好地保留原始高维空间 (high-dimensional space)中局部结构或相邻点之间的具体相似之处。

### t-分布随机邻域嵌入 (embedding)（t-SNE）

t-SNE是一种非线性降维技术，特别适合在低维空间（通常是2D或3D）中展示高维数据。与PCA不同，t-SNE明确专注于模拟点之间的相似性。它努力确保在高维空间 (high-dimensional space)中彼此接近的点在低维图中仍然保持接近，而相距较远的点则仍然相距较远。

t-SNE通常能产生更具说服力的可视化结果，展现数据内的群组和局部结构。如果Word2Vec学习到了有意义的关联，t-SNE通常能使这些关联变得明显，显示相关词语聚集在一起。

然而，t-SNE有一些需要注意的地方：

- **计算成本：** 它可能比PCA慢得多，尤其是在大型数据集上。
- **非确定性：** 由于其概率性和优化过程，在相同数据上多次运行t-SNE可能会产生略微不同的可视化结果。
- **全局结构：** t-SNE图中群组*之间*的距离并非总是有意义。它擅长显示*哪些*点彼此靠近，但不同群组的相对位置可能会产生误导。请侧重于分组情况，而非群组间精确的距离或群组的大小。

### 可视化结果的解释

无论使用何种方法（PCA或t-SNE），目标都是将词向量 (vector)投影到2D或3D空间，并将其绘制为点。然后我们可以用对应的词语标记 (token)这些点，并检查生成的散点图。

我们可能会看到什么？

- **语义群组：** 意思相近的词语应该出现在一起。例如，'cat'（猫）、'dog'（狗）、'hamster'（仓鼠）可能会形成一个群组，与‘car’（汽车）、‘truck’（卡车）、‘bicycle’（自行车）的群组分开。国家可能会聚集成群，城市可能会聚集成群，与移动相关的动词可能会聚集成群等等。
- **作为方向的关联：** 有时，几何排列反映了类比关系。经典的例子是，从“man”到“woman”的向量偏移可能与从“king”到“queen”的偏移非常相似。这表明$vector(\text{'国王'}) - vector(\text{'男人'}) + vector(\text{'女人'}) \approx vector(\text{'女王'})$。虽然不能保证完美的几何类比，但可视化有时能够显示这些方向性关联。

### 可视化示例

假设我们已经训练好了嵌入 (embedding)，并选择了一小部分与国家、首都和王室相关的词语。在应用t-SNE来降低它们的向量 (vector)维度（例如，从100维降到2维）后，我们可能会得到如下图所示的图表。



![词嵌入的2D t-SNE投影](plots/2500-0.json)



> 2D t-SNE可视化图，显示了语义聚类。王室术语（紫色）、欧洲首都（蓝色）和欧洲国家（青色）形成了不同的群组。

在这张图中，我们观察到与王室相关的词语（“国王”、“女王”、“王子”）聚集在一起。同样，首都（“巴黎”、“伦敦”、“柏林”）形成了另一个群组，而国家（“法国”、“英国”、“德国”、“西班牙”）形成了第三个群组。这种视觉上的分离证实了嵌入已捕获到训练数据中预期的一些语义相似性与区别。这样的可视化为词嵌入的质量提供了有益的反馈。它们有助于理解模型学到了什么，有时还能显示嵌入空间中意料之外的关联或问题。

## 参考资料

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (2013)
  Journal: arXiv preprint arXiv:1301.3781; DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  介绍了Word2Vec模型，这是一种学习捕获语义关系的密集词嵌入的基础方法。
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) — Jeffrey Pennington, Richard Socher, Christopher Manning (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1532-1543; DOI: [10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)
  提出了GloVe，一种用于获取词向量表示的无监督学习算法，结合了全局矩阵分解和局部上下文窗口方法。
- [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) — Laurens van der Maaten and Geoffrey Hinton (2008)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 9; Pages: 2579-2605
  介绍了t-SNE（t分布随机邻域嵌入）的原始论文，这是一种广泛用于数据可视化的非线性降维技术。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 738
  机器学习领域的标准教科书，全面阐述了降维技术，包括主成分分析（PCA），这对高维数据可视化至关重要。
