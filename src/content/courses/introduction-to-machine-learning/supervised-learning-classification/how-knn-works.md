---
course: "introduction-to-machine-learning"
chapter: "supervised-learning-classification"
lesson: "how-knn-works"
sourceId: 1498
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-4-supervised-learning-classification/how-knn-works"
title: "KNN 的运作方式"
description: "说明使用KNN算法进行预测的步骤。"
order: 6
plots: ["plots/1498-0.json"]
sourceHash: "679fcab329f51f54ddf349d574be0c25043833013764d22b7b058d51b995ef74"
sourceCorrections: []
---

K近邻（KNN）算法在分类时采用一种直接且基于实例的方法。这种方法与逻辑回归等算法形成对比，后者旨在学习特定函数（例如Sigmoid）来定义决策边界。KNN常被称为“惰性学习器”，因为它在训练阶段并没有真正建立一个明确的模型。相反，它会记住整个训练数据集。实际的分类工作发生在其被要求对一个*新的*、未见过的数据点进行分类时。

以下是KNN对新数据点进行分类的逐步过程：

1. **选择K：** 首先，你需要确定'K'的值，它表示要考虑的邻居数量。这是你作为实践者预先设定的一个数值（使其成为一个超参数 (parameter) (hyperparameter)，正如我们在第2章中讨论过的）。假设你选择$K=3$。
2. **计算距离：** 当一个新的数据点出现时（我们称之为查询点），KNN会计算该查询点与存储的训练数据集中*每一个数据点*之间的距离。衡量这种距离的常用方法是使用标准欧几里得距离，它本质上是特征空间中两点之间的直线距离。如果你在$n$维空间中有两个点$p$和$q$，欧几里得距离计算如下：
   $d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_n - q_n)^2}$
   或更简洁地：
   $d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$
   虽然欧几里得距离很常见，但有时也可以根据数据使用其他距离度量方法。核心思想是根据新点与每个训练点的特征，量化 (quantization)它们的“相似”或“接近”程度。
3. **找出K个最近邻居：** 一旦所有距离都计算完毕，算法会找出与查询点距离最小的'K'个训练数据点。这些就是它的“最近邻居”。如果我们选择$K=3$，KNN会找出距离新点最近的3个训练点。
4. **多数投票：** 现在，查看这'K'个最近邻居的类别标签。KNN在这些邻居中进行简单的多数投票。K个邻居中最常见的类别将成为新查询点的预测类别。例如，如果$K=3$，并且两个邻居属于A类，一个属于B类，则新点将被分类为A类。如果$K=5$，并且三个邻居是B类，两个是A类，则预测为B类。
5. **分配类别：** 投票结果就是新数据点的最终分类。

让我们将其可视化。想象我们有根据两个特征绘制的属于两个类别（A类为蓝色，B类为橙色）的数据点。一个新的、未分类的点（绿色星形）出现。如果我们设置$K=5$，KNN会找出距离星形最近的5个训练点。



![KNN分类示例 (K=5)](plots/1498-0.json)



> K=5的KNN示例。绿色星形是新点。算法找出5个最近的点（用紫色圈出）。四个邻居是B类（橙色），一个邻居是A类（蓝色）。通过多数投票，KNN将新点分类为B类。

这就是核心机制。'K'的选择很重要。小的'K'（例如$K=1$）会使模型对噪声或异常值非常敏感；分类结果可能仅凭一个距离很近、可能异常的点就显著改变。非常大的'K'会使决策边界更平滑，但可能会模糊局部模式，或者根据数据集中整体的多数类别对点进行分类，而忽略附近的点。选择合适的'K'通常需要尝试多个值，并查看哪个在独立的验证数据集上表现最佳（这是我们在第2章中提及的一个原理，并将再次讨论）。

因为KNN高度依赖距离计算，所以所有特征都处于可比较的尺度也很重要。这就是为什么特征缩放（我们将在第6章介绍）在使用KNN之前通常是一个必要的预处理步骤。如果特征值过大且未正确缩放，它们可能会不成比例地影响距离计算。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFihiIm3T5Unzvstx1GXhJXyZ1IE_7aFnNWrHZfrX2-VqEdF6hZHZ8Bm27LmkJmDsZH84dwebDYfeNeuZfvRyGOeo5TwX-NKfWTwALge7nvv2zvwBkfY8b7SvZRpFt68Xy5FcFQmadEKfRadbCmYb2KA5b2kurQoy-VpEn4WaD92xLoG_Z0f6qSUGIhJ0Buphb7uO71cBAqug==) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本综合性教材，详细解释了K-近邻算法，包括其理论基础和实际应用。
- [Nearest neighbor pattern classification](https://ieeexplore.ieee.org/document/1053964) — Thomas Cover and Peter Hart (1967)
  Journal: IEEE Transactions on Information Theory; Publisher: IEEE; Volume: 13; Pages: 21-27; DOI: [10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964)
  引入并形式化K-近邻算法的开创性论文，奠定了其理论基础。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer
  一本广泛使用的教材，全面介绍了模式识别和机器学习概念，其中包含K-近邻算法的专门章节。
- [CS229 Lecture Notes: Supervised Learning (K-Nearest Neighbors section)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG33tix_QtYtnwP-xvkSxDOkWujiiNqvM__RW_G9rg-Nc0R0Qu0tswvOkBDB6EcT68v97HYwGPMSdYaOpc30emkHIEH-3U392jzfW4r9y0HuZKjPXBEKl2fbc79P3rRhaTUg7OnsDUrsoO5Ja33IvzQA1BrJ7scfO-_tw==) — Stanford CS229 Course Staff (2018)
  Publisher: Stanford University; Pages: 3-11+
  斯坦福大学机器学习课程讲义，在监督学习背景下清晰介绍了K-近邻算法。
