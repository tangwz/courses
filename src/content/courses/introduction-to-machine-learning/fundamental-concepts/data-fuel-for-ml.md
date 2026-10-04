---
course: "introduction-to-machine-learning"
chapter: "fundamental-concepts"
lesson: "data-fuel-for-ml"
sourceId: 1462
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-2-fundamental-concepts/data-fuel-for-ml"
title: "数据：机器学习的燃料"
description: "了解数据为何对训练机器学习模型不可或缺。"
order: 1
plots: []
sourceHash: "da644139efe77fdaf907aaa07a93ec5619baa3daa9ce0dbb595a88015acbfec5"
sourceCorrections: []
---

机器学习 (machine learning)算法像人类一样，通过经验学习。但对算法来说，什么构成了“经验”？答案是**数据**。如果把机器学习模型比作一台引擎，那么数据就是它的燃料。没有数据，引擎就无法运行；它将无从学习。

在机器学习中，数据通常指观察结果或示例的集合。对于许多入门问题，这些数据通常以结构化格式组织，如表格或电子表格，其中每行代表一个单独的观察结果（也称为**样本**、**实例**或**数据点**），每列代表该观察结果的一个特定属性或**特征**。

假设您想构建一个系统来预测电子邮件是否为垃圾邮件。您的数据可能看起来像这样：

> 电子邮件数据的简单呈现。每行是一个电子邮件实例。蓝色列是用于预测的特征，红色列是模型希望学习预测的标签。

我们用这个例子来分析一下这些术语：

- **实例（或样本、数据点、记录、观察结果）：** 表格中的每一行都代表一个特定的电子邮件。它是一个模型可以从中学习的单一例子。
- **特征：** 这些是实例的可衡量属性或特性，用作模型的输入。在我们的例子中，“包含‘伟哥’”、“主题全大写”和“已知发件人”都是特征。它们是模型用于进行预测的输入变量（$x$）。根据问题的不同，您可以拥有少量或多达数千个特征。
- **标签（或目标、输出、响应变量）：** 这是我们希望机器学习模型预测的结果或数值。在垃圾邮件的例子中，“是垃圾邮件？”这一列就是标签。它在学习阶段（在监督学习 (supervised learning)中）代表每个实例的“正确答案”。模型学习特征与标签（$y$）之间的关联。请注意，并非所有类型的机器学习都涉及标签；我们将在后面讨论无监督学习 (unsupervised learning)。

数据的质量和数量对机器学习项目的成功极为重要。

1. **数量：** 通常，更多的数据能让模型学习到更可靠的模式。与在数百万封电子邮件上训练的模型相比，仅在十封电子邮件上训练的模型可能无法很好地推广到新的、未见过的电子邮件。
2. **质量：** 数据必须相关且准确。如果收集的特征与电子邮件是否为垃圾邮件没有关联，模型将无法有效学习。同样，如果标签经常不正确（例如，将合法电子邮件标记 (token)为垃圾邮件），模型就会学到错误的模式。这引出了该领域一句常见的说法：“垃圾进，垃圾出。” 劣质数据必然导致劣质模型。

数据可以有多种类型，例如数值型（如温度、身高）、类别型（如颜色名称、电子邮件发件人域名）或文本型。将这些原始数据准备成适合算法的格式是机器学习过程的重要组成部分，我们将在第 6 章“准备数据”中讨论。

目前，最重要的一点是数据构成了构建机器学习模型的基础。理解数据的结构方式，包括特征和标签之间的区别，是理解这些模型如何学习的第一步。

## 参考资料

- [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2013)
  Publisher: Springer
  介绍统计学习概念，包括数据、特征和响应变量的详细解释。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=q_eJEAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  机器学习实践指南，通过实例介绍基本数据概念、特征和标签。
- [Machine Learning](https://www.cs.cmu.edu/~tom/mlbook.html) — Tom Mitchell (1997)
  Publisher: McGraw-Hill
  一本定义机器学习核心原理的经典教材，包括算法如何从数据、特征和目标概念中学习。
