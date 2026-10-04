---
course: "basics-model-evaluation-metrics"
chapter: "metrics-for-classification"
lesson: "f1-score-metric"
sourceId: 3981
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-2-metrics-for-classification/f1-score-metric"
title: "F1分数：结合精确率和召回率"
description: "定义F1分数为精确率和召回率的调和平均值，并学习其计算方法。"
order: 9
plots: []
sourceHash: "f9299d074211b80ffdd0c1ed191482e3508d626ffd416e4331843949bd471563"
sourceCorrections: []
---

您已经了解了精确率和召回率，这是评估分类模型的两个重要指标。精确率说明模型做出的正向预测中有多少是实际正确的，而召回率则说明模型识别出了多少实际的正向实例。如同我们在上一节讨论过的，这二者常常存在权衡：提升精确率有时会降低召回率，反之亦然。

那么，如果您需要一个能同时衡量这两者的单一数值该怎么办？如果您的应用场景中，找到*所有*相关项（高召回率）以及确保找到的项*确实*相关（高精确率）都显得重要呢？此时F1分数就能发挥作用。

### 什么是F1分数？

F1分数是一种将精确率和召回率结合成一个单一指标的方法。它的计算方式是精确率和召回率的*调和平均值*。

### 为什么使用调和平均值？

您可能会想，为什么不直接取精确率和召回率的简单平均值（算术平均值）呢？调和平均值具有一个有用特性：它会降低对一个指标非常高而另一个指标非常低的模型的权重 (weight)。即使一个指标表现不佳，简单平均值可能看起来还不错，但调和平均值会惩罚这种不平衡。

考虑一个模型，其具有：

- 精确率 = 0.9 (90%)
- 召回率 = 0.1 (10%)

算术平均值为 $(0.9 + 0.1) / 2 = 0.5$。这个分数看起来不算太差。

然而，F1分数（我们接下来会看到如何计算）会低得多，这反映了其糟糕的召回率。调和平均值将综合分数拉向较低值，从而确保模型必须在精确率和召回率*两者*上都表现良好，才能获得较高的F1分数。只有当精确率*和*召回率都高时，它的分数才会高。

### 计算F1分数

F1分数的公式如下：


$$
F1 = 2 \times \frac{精确率 \times 召回率}{精确率 + 召回率}
$$


让我们使用之前的例子（精确率 = 0.9，召回率 = 0.1）：


$$
F1 = 2 \times \frac{0.9 \times 0.1}{0.9 + 0.1} = 2 \times \frac{0.09}{1.0} = 0.18
$$


请注意，F1分数（0.18）远低于算术平均值（0.5），并且更接近较低的指标（召回率 = 0.1）。这显示了模型在召回率方面的不足，而简单平均值掩盖了这一点。

您也可以直接使用真阳性 ($TP$)、假阳性 ($FP$) 和假阴性 ($FN$) 来表示F1分数：


$$
F1 = \frac{2 \times TP}{2 \times TP + FP + FN}
$$


这个公式是通过将精确率 ($精确率 = \frac{TP}{TP + FP}$) 和召回率 ($召回率 = \frac{TP}{TP + FN}$) 的定义代入第一个F1分数方程而得出的。

### F1分数的解读

像精确率和召回率一样，F1分数的范围从0到1。

- F1分数为1表示精确率和召回率都完美。
- F1分数为0表示精确率或召回率（或两者）为零。

较高的F1分数表示精确率和召回率之间有更好的平衡。当出现以下情况时，它是一个特别有用的指标：

1. **存在类别不平衡：** 在一个类别数量明显多于其他类别的数据集中，准确率可能会产生误导。高F1分数可确保模型在少数类别上表现良好，同时考虑假阳性和假阴性。
2. **假阳性和假阴性都重要：** 如果将阴性实例错误地识别为阳性 ($FP$) 和未能识别阳性实例 ($FN$) 都是不理想的结果，F1分数会提供一个均衡的评估。例如，在医疗诊断中，漏诊疾病（低召回率）和误诊健康患者（低精确率）都可能产生重大后果。

### 计算示例

让我们回顾一个混淆矩阵：

|  | 预测为正 | 预测为负 |
| --- | --- | --- |
| **实际为正** | TP = 80 | FN = 20 |
| **实际为负** | FP = 10 | TN = 90 |

由此，我们计算得出：

- 精确率 = $\frac{TP}{TP + FP} = \frac{80}{80 + 10} = \frac{80}{90} \approx 0.89$
- 召回率 = $\frac{TP}{TP + FN} = \frac{80}{80 + 20} = \frac{80}{100} = 0.80$

现在，我们计算F1分数：


$$
F1 = 2 \times \frac{0.89 \times 0.80}{0.89 + 0.80} = 2 \times \frac{0.712}{1.69} \approx 2 \times 0.421 \approx 0.84
$$


F1分数为0.84表明精确率（0.89）和召回率（0.80）之间有良好的平衡。

总而言之，F1分数提供了一个单一且便捷的指标，通过平衡精确率和召回率来概括分类器的性能。当处理不平衡类别或两种分类错误（假阳性和假阴性）都需要最小化时，它尤其有价值。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  为读者提供了分类算法及其评估指标（包括F1分数）在统计学习框架下的全面且严谨的阐述。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2019)
  Publisher: O'Reilly Media
  面向实践者的优秀资源，使用流行的机器学习库，清晰地解释了包括F1分数在内的分类指标并提供了实际示例。第二版。
- [A systematic analysis of performance measures for classification tasks](https://doi.org/10.1016/j.ipm.2009.03.002) — Marina Sokolova, Guy Lapalme (2009)
  Journal: Information Processing & Management; Publisher: Elsevier; Volume: 45; Pages: 427-437; DOI: [10.1016/j.ipm.2009.03.002](https://doi.org/10.1016/j.ipm.2009.03.002)
  这篇同行评审论文系统地概述和分析了各种分类性能度量，包括对精确率、召回率和F度量的详细讨论及其应用场景。
