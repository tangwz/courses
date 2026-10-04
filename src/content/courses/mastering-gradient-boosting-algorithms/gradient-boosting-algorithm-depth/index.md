---
course: "mastering-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-2-gradient-boosting-algorithm-depth"
sourceId: 528
chapter: "gradient-boosting-algorithm-depth"
title: "梯度提升算法详解"
order: 2
description: "详细解析梯度提升机 (GBM) 算法、损失函数以及梯度的作用。"
hasQuiz: false
---

在了解了集成方法和加法模型的基本原理后，本章将着重讲解标准梯度提升机 (GBM) 算法的运作机制。我们将从一般思路出发，循序渐进地理解这些模型是如何构建的。

您将学会从优化角度看待提升过程，特别是在函数空间中进行的梯度下降。我们将正式推导通用GBM算法，阐明残差或伪残差如何指导每个顺序基学习器（通常是决策树）的训练。

涵盖的主要方面包括：

*   GBM算法的数学推导。
*   对常见损失函数（例如回归中的平方误差或分类中的对数损失）的分析，以及它们的梯度（$ \nabla L $）如何推动模型拟合过程。
*   收缩（即学习率参数）在控制每棵树贡献和辅助正则化方面的作用。
*   诸如行和列子采样（随机梯度提升）等技术，以提高模型泛化能力和计算效率。
*   使用Scikit-learn的`GradientBoostingRegressor`和`GradientBoostingClassifier`进行实际实现。

学完本章，您将对经典的GBM算法有一个详细的操作理解，为您后续讨论的更高级实现做好准备。一个实践环节将指导您使用Python构建您的第一个GBM模型。
