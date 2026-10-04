---
course: "basics-model-evaluation-metrics"
chapter: "preparing-data-evaluation"
lesson: "train-test-split-procedure"
sourceId: 4027
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-4-preparing-data-evaluation/train-test-split-procedure"
title: "训练-测试集划分步骤"
description: "描述将数据集划分为训练和测试部分的常见步骤。"
order: 4
plots: []
sourceHash: "56e37d95c8f10e7648f12153ed6af211da30d58248da3d5dc0919c9abf24e671"
sourceCorrections: []
---

你现在明白了*为什么*我们需要独立的训练集和测试集：为了公平评估模型在新数据、未见过的数据上的表现。你也清楚每个集合的用途：训练集用于教导模型，测试集则用于后续的评估。

但我们如何实际进行这种划分呢？这个过程本身很简单。以下是标准训练-测试集划分的逐步指南：

### 划分过程

1. **从你的整个带标签数据集开始：** 设想你已收集所有数据，包括特征（输入）和目标变量（你希望预测的内容，如“垃圾邮件”/“非垃圾邮件”或房价）。这个完整的数据集是你的起点。
2. **打乱数据（通常建议）：** 在划分之前，通常非常推荐随机打乱数据集中行（即单个数据点或样本）的顺序。为什么？有时数据是按特定顺序收集或存储的。例如，可能所有“垃圾邮件”都列在前面，或者房价数据按社区排序。如果你在未打乱的情况下划分有序数据，你的训练集可能只包含一种类型的样本，而测试集包含另一种，这会导致训练效果不佳和评估结果有偏差。打乱数据可确保不同类型的样本能够随机分布在训练集和测试集中。我们将在本章后面更多地讨论这种随机性的意义。
3. **选择划分比例：** 你需要决定你的数据中多少比例用于训练，多少比例用于测试。这个比例通常以百分比表示，例如 80/20（80% 用于训练，20% 用于测试）或 70/30。选择取决于几个因素，包括数据集的总大小。我们将在下一节讨论常见的比例。
4. **执行划分：** 根据所选比例，将你打乱后的数据集分成两个独立、不重叠的子集。

   - **训练集：** 包含数据中较大的部分（例如，80%）。这些数据将用于训练你的机器学习 (machine learning)模型。模型从这个集合中学习模式、关系和规则。
   - **测试集：** 包含剩余的较小部分（例如，20%）。这些数据被预留起来，在训练期间*不*向模型展示。它作为未见过的数据，用于评估。
5. **保持测试集独立：** 这是一个非常重要的步骤。划分完成后，你应该像测试集不存在一样对待它，直到你有一个最终的、已训练好的模型准备好进行评估。**不要**使用测试集来决定如何构建或调整模型（例如选择要使用的特征或调整模型参数 (parameter)）。在模型构建过程中使用来自测试集的信息会污染它，你的最终评估将无法反映在真正新数据上的真实表现。

### 划分的可视化

可以把它想象成：你拿出完整的数据卡片，彻底洗牌，然后将一定比例的卡片分发到“训练堆”，其余的卡片分发到“测试堆”。

> 一个流程图，显示数据集被打乱后，再分成独立的训练集和测试集。

大多数机器学习 (machine learning)库都提供了函数，可以轻松执行这种打乱和划分操作。例如，在 Python 的 scikit-learn 库中，`train_test_split` 函数通过一个命令就可以处理打乱和划分，它将你的特征、目标变量和所需的测试集大小作为输入。

遵循此步骤，你便能在用于学习的数据与用于公正评估的数据之间建立必要的区分，这对于理解模型在实际中可能表现如何非常重要。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本基础教材，涵盖统计学习的理论基础，包括模型评估和独立测试集对泛化误差估计的必要性。
- [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — scikit-learn developers (2023)
  `train_test_split` 函数的官方文档，提供了将数据集划分为训练和测试子集的实用指南。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125969/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广受欢迎的实践指南，展示了如何使用流行库实现机器学习概念，包括训练-测试集划分。
