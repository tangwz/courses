---
course: "introduction-to-machine-learning"
chapter: "supervised-learning-classification"
lesson: "understanding-classification-problems"
sourceId: 1490
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-4-supervised-learning-classification/understanding-classification-problems"
title: "理解分类问题"
description: "定义分类任务，其目标是预测离散类别或类别标签。"
order: 1
plots: ["plots/1490-0.json"]
sourceHash: "6e26607bc53393ef344d9ca323af0e05c1fdb8984ea9dd22ae32228cfa61de66"
sourceCorrections: []
---

在上一章中，我们学习了回归，其目标是预测一个连续的数值，例如房屋的价格或明天的温度。现在，我们将注意力转向另一种监督学习 (supervised learning)任务：**分类**。

想象一下，你不是在试图预测一个具体的数字，而是试图将某物归入一个组或类别。这就是分类的核心。其目的是学习将输入变量（特征）映射到预定义的离散类别或`类`。

请思考这些常见情景：

- **垃圾邮件检测：** 传入的电子邮件是`垃圾邮件`还是`非垃圾邮件`？
- **图像识别：** 图像中包含的是`猫`、`狗`还是`汽车`？
- **医疗诊断：** 根据症状和检测结果，患者患有特定的`疾病`还是`无疾病`？
- **客户行为：** 客户是可能`流失`（停止使用服务）还是`不流失`？

在每种情况下，预测都不是一个连续刻度上的数字；它是一个从有限可能性集合中选出的不同标签。

## 什么定义了分类问题？

分类问题的定义特征是，我们想要预测的**目标变量**是**类别型的**。这意味着它取代表不同组或类的值。

- **特征：** 就像回归中一样，我们使用输入`特征`来进行预测。对于垃圾邮件检测，特征可能包括某些词语的频率、发件人地址或电子邮件是否包含附件。对于医疗诊断，特征可以是患者的年龄、血压或特定实验室测试的结果。
- **类别标签：** 我们预测的输出称为`类别标签`（或简称为`类`或`类别`）。这些标签是预定义的。示例包括`{'垃圾邮件', '非垃圾邮件'}`、`{'猫', '狗', '汽车'}`、`{'疾病 A', '疾病 B', '健康'}`。

## 二元分类与多元分类

分类问题大致可以分为两种主要类型：

"1. **二元分类：** 这是最简单的形式，其中只有*两个*可能的输出类别。许多问题都属于这一类别，通常被表述为是/否决策。"
\* 示例：垃圾邮件检测（`垃圾邮件`/`非垃圾邮件`）、医学测试（`阳性`/`阴性`）、客户流失预测（`流失`/`不流失`）。

2. **多元分类：** 在这种情况下，有*三个或更多*可能的输出类别。任务是将一个实例分配到这些多个类别中的一个。
   - 示例：手写数字识别（`0`、`1`、`2`、...、`9`）、图像中的物体识别（`人`、`汽车`、`树`、`建筑物`）、文档分类（`体育`、`政治`、`科技`、`商业`）。

分类是一种监督学习 (supervised learning)任务，旨在预测数据的离散类别或标签，而非连续数值。例如，它可以预测电子邮件是否为垃圾邮件，或将图像归类到特定对象类别。一个简单的分类情景是，数据点具有两个特征（特征1和特征2），并且每个点都属于两个类（类A或类B）中的一个。



![简单分类示例数据](plots/1490-0.json)



> 两个不同类别的数据点根据两个特征进行绘制。分类算法的目标是学习如何根据特征区分这些类别。

分类算法的任务是学习一个“规则”或“边界”（通常称为决策边界，我们稍后会讨论），根据特征中的模式来分离不同的类别。当一个新的、未见过的数据点出现时，算法会使用这个学到的规则将其分配给最可能的类别。

在本章中，我们将研究专门为这些任务设计的特定算法，首先是逻辑回归，它常用于二元分类，然后是K近邻（KNN），这是一种适用于二元和多元问题的多用途算法。我们还将介绍如何衡量分类模型的表现如何。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
  一本广泛使用的入门教材，涵盖统计学习的核心概念，包括分类与回归的区别以及基础分类算法。
- [Pattern Recognition and Machine Learning](https://www.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/b93563](https://doi.org/10.1007/b93563)
  一本全面而基础的教材，详细阐述了分类及其他机器学习概念的理论基础，并侧重于概率方法。
- [User Guide: Classification](https://scikit-learn.org/stable/supervised_learning.html#classification) — scikit-learn developers (2023)
  scikit-learn的官方文档，提供了分类任务、常见算法以及在主流机器学习库中的使用示例的实用概述。
- [Supervised Learning (Lecture Notes)](https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf) — Andrew Ng (2018)
  Publisher: Stanford University
  斯坦福大学一门极具影响力的机器学习课程的讲义，对监督学习（包括分类问题）进行了清晰且数学严谨的介绍。
