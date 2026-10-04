---
course: "introduction-to-machine-learning"
chapter: "building-first-ml-model"
lesson: "choosing-right-algorithm"
sourceId: 1534
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-7-building-first-ml-model/choosing-right-algorithm"
title: "选择合适的算法"
description: "关于如何根据问题类型（回归/分类）选择合适算法的指南。"
order: 2
plots: []
sourceHash: "634824ab704326eaffb759cba3fdf3272a6bd5e7b3716a38131f8e7526c16d0a"
sourceCorrections: []
---

机器学习 (machine learning)涉及多种任务，并使用不同的算法，例如线性回归、K近邻 (KNN) 和 K均值。在构建机器学习模型时，一个主要问题是选择哪种算法。

选择合适的算法是机器学习流程中的一个基础步骤。虽然经验丰富的实践者通常会尝试多种算法，但对于你的最初模型，选择过程通常始于清晰地识别你试图解决的问题类型。

### 从你的目标开始

缩小算法选择范围最直接的方法是查看你希望用数据实现什么。问问自己：

1. **我是否有有标签数据？** 也就是说，我的数据集是否已经包含模型需要预测的正确答案或结果？

   - 如果**是**，你可能正在处理一个**监督学习 (supervised learning)**问题。
   - 如果**否**，并且你正在寻找数据本身固有的模式或分组，那很可能是一个**无监督学习 (unsupervised learning)**问题。
2. **如果是监督学习，我需要哪种类型的输出？**

   - 我是在预测一个**连续数值**（比如房价、温度或销售额）吗？这是一个**回归**任务。
   - 我是在预测一个**离散类别或类标签**（比如“垃圾邮件”与“非垃圾邮件”、“猫”与“狗”，或不同类型的客户）吗？这是一个**分类**任务。
3. **如果是无监督学习，我正在寻找哪种模式？**

   - 我是否试图根据数据的特征将**相似数据点聚合在一起**，而事先不知道分组情况？这是一个**聚类**任务。

### 问题与算法的匹配 (基于本课程)

根据我们在这门入门课程中迄今为止讨论的问题类型和具体算法，你可以做出初步选择：

- **对于回归问题**（预测一个数字）：**线性回归**是我们介绍的基础算法。它通过找到输入特征与连续输出值之间的最佳直线关系来工作。
- **对于分类问题**（预测一个类别）：
  - **逻辑回归：** 尽管名字如此，它用于分类（特别是二元分类，如是/否）。它计算一个实例属于特定类别的概率。
  - **K近邻 (KNN)：** 该算法根据新数据点在特征空间中与其“k”个最近邻居中的多数类别进行分类。它直观且在传统意义上无需复杂的训练阶段。
- **对于聚类问题**（对无标签数据进行分组）：**K均值聚类**是我们介绍的主要算法。它的目标是将数据划分为 'K' 个不同的组，其中组内数据点彼此相似。

### 简单决策指南

你可以这样看待这个过程：

> 一个流程图，指导根据数据标签和预测目标进行初步算法选择。

### 这只是一个起点

对于这门入门课程，侧重于将问题类型（回归、分类、聚类）与我们学过的算法（线性回归、逻辑回归、KNN、K均值）相匹配是主要的选择方式。

请记住：

1. **先求简单：** 通常，最好从适合你任务的更简单算法（如这里介绍的算法）开始。如果它的表现足够好，你初期可能不需要任何更复杂的东西。
2. **数据很重要：** 数据本身的特性（特征数量、数据点数量、是否存在异常值）也可能影响算法选择和表现，但理解任务类型是第一个筛选条件。
3. **下一步是评估：** 选择并训练算法后，下一个重要步骤是评估其表现，我们很快就会介绍。评估有助于你理解所选算法是否在你的特定数据上表现良好。

在接下来的章节中，我们将使用一个库（如Scikit-learn），一旦你根据问题类型做出初步选择，它将使实现和切换这些基本算法变得相对容易。现在，请专注于正确识别你面对的是回归、分类还是聚类任务。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  这本实用指南演示了如何使用Scikit-learn构建机器学习模型，涵盖了从问题定义、算法选择到训练和评估各种任务的工作流程。
- [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map.html) — scikit-learn developers (2023)
  Scikit-learn官方文档提供了一个决策树（流程图），指导用户根据数据特征和问题类型选择合适的机器学习算法。
- [Machine Learning](http://www.cs.cmu.edu/~tom/mlbook.html) — Tom Mitchell (1997)
  Publisher: McGraw Hill
  这是一本系统介绍机器学习概念的经典教材，包括对监督学习和无监督学习的详细解释，为算法选择提供了背景知识。
