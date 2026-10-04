---
course: "getting-started-with-scikit-learn"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-5-model-selection-evaluation"
sourceId: 521
chapter: "model-selection-evaluation"
title: "模型选择与评估"
order: 5
description: "学习如何在Scikit-learn中使用训练-测试集划分和交叉验证方法来正确评估机器学习模型。"
hasQuiz: true
---

在学会实现监督学习模型并进行数据预处理之后，下一步自然是弄清楚这些模型表现如何以及如何选择最佳配置。仅仅在模型训练所用的数据上评估它可能会产生误导。模型可能在训练数据上表现完美，但在新的、未见过的数据上表现很差。本章将应对这个问题。

我们将考察常见的过拟合问题（模型对训练数据，包括其噪声，学习得过于充分）和欠拟合问题（模型过于简单以至于无法捕捉到数据背后的规律）。你将学会方法来对模型性能进行更实际的评估。

具体来说，本章涵盖：

*   **过拟合**和**欠拟合**的含义。
*   使用Scikit-learn的`train_test_split`函数，将数据集划分为**训练集和测试集**进行初步评估。
*   应用**交叉验证**，尤其是**K折交叉验证**和**分层K折交叉验证**，以更可靠地估计模型在未见过的数据上的表现。
*   使用**网格搜索（`GridSearchCV`）**，系统地调整模型超参数并找到最佳设置。

到本章结束时，你将能够严谨地评估你的机器学习模型，并对模型选择和配置做出明智的决定。
