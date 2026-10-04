---
course: "introduction-to-machine-learning"
chapter: "building-first-ml-model"
lesson: "recap-ml-workflow-steps"
sourceId: 1533
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-7-building-first-ml-model/recap-ml-workflow-steps"
title: "回顾：机器学习工作流步骤"
description: "总结创建机器学习模型的端到端步骤。"
order: 1
plots: []
sourceHash: "63980dc1fb4fd85e591ec7f890b273359aa3b9466ce4c214daa8f61d094b5c2a"
sourceCorrections: []
---

好的，让我们把目前学到的内容汇总起来。构建机器学习 (machine learning)模型不仅仅是选择一个算法；它是一个系统的过程，通常被称为机器学习工作流。可以把它看作一张路线图，指引你从原始数据到能进行预测的实用模型。尽管具体细节会因项目而异，但核心阶段保持相对一致。

在之前的章节中，我们已经单独讲过这些阶段。我们从了解什么是机器学习开始（第1章），以及数据、特征、模型和评估指标等基本要点（第2章）。然后我们介绍了特定类型的问题和算法：用于预测数值的回归（第3章），用于预测类别的分类（第4章），以及用于在无标签数据中寻找群组的聚类（第5章）。最近，我们着重介绍了准备数据的重要一步（第6章），内容包括处理缺失值和特征缩放等任务。

现在，让我们正式概括一下启动和运行机器学习模型通常涉及的步骤：

1. **问题定义与规划：** 在编写任何代码之前，你需要清楚地了解你试图解决的问题。你要提出什么问题？这是一个回归、分类还是聚类任务？你需要什么数据？成功的标志是什么？虽然我们没有专门用一整章来介绍这个，但了解目标是告知所有后续行动的必要第一步。
2. **数据获取与认识：** 问题定义后，你需要数据。这可能涉及收集新数据或访问现有数据集。这个阶段的重要部分是分析数据以认识其结构，识别潜在问题（比如我们在第6章讨论的缺失值），并获得初步见解。
3. **数据准备（预处理）：** 原始数据很少能直接用于机器学习模型。如第6章所述，此阶段涉及数据清洗（处理缺失值、纠正错误）、特征转换（缩放、编码分类变量），以及将数据拆分为训练集和测试集（在第2章介绍并在第6章回顾）。这通常是工作流中最耗时的部分，但对于构建有效模型是绝对必要的。
4. **模型选择：** 根据问题定义（回归、分类等）和你对数据的认识，你选择一个或多个候选算法进行尝试。我们已经介绍了线性回归、逻辑回归、KNN和K-均值作为基本实例。实际中，你可能会考虑多种选择。
5. **模型训练（拟合）：** 这是“学习”发生的地方。你将准备好的训练数据提供给所选算法。算法在数据中学习模式或关系，调整其内部参数 (parameter)以最小化误差（正如我们在第3章线性回归中看到的梯度下降 (gradient descent)）。这个过程的结果就是训练好的模型。
6. **模型评估：** 模型学习得如何？你使用单独的测试集（模型在训练期间未见过）来评估模型的表现。我们使用适用于任务的指标，例如回归的均方误差或分类的准确率和混淆矩阵（如第2章和第4章所述），来衡量模型对新的、未见过的数据的泛化能力。这有助于发现过拟合 (overfitting)或欠拟合 (underfitting)等问题（第2章）。

"7. **预测（推断）：** 如果评估结果令人满意，模型就可以用于其预期目的：对新数据点进行预测。"

记住，这个工作流并非总是严格线性的。通常，评估结果可能会让你回到更早的步骤。也许模型表现不佳，这表明你需要尝试不同的数据准备技术、选择不同的算法，甚至重新审视问题定义。这种迭代式地改进数据、模型和参数的过程在实际机器学习中很常见。

> 迭代机器学习工作流阶段的图示概览。

在接下来的章节中，我们将具体介绍这些步骤，使用标准库来处理数据加载、准备、模型训练和评估，将所有部分整合到一个实际的例子中。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的指南，涵盖了使用流行Python库进行机器学习的整个工作流程，从数据准备和模型训练到评估和部署。
- [Machine Learning Yearning](https://www.deeplearning.ai/machine-learning-yearning/) — Andrew Ng (2017)
  Publisher: deeplearning.ai
  提供了关于如何构建和指导机器学习项目的战略建议，特别强调工作流程的迭代性质和问题框架。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGwFWcCPADWeh9Iw4swYbgBXqkfEVcOj7P4X6jCyeoSyL055-lMIA2YbfwoJiYJZIsW1WRS3-rG9zjYhTzXZlC8Al0VBOW-KyazegO1nzI674nOa7vMWAI7HgynpjVn-UIWY-CFDtlcFoKExcRPam6udwP2bdKgTXVtZCVNSKa6JcEBEvYiVshlhYyfatLtD03wuLgxDpzSnRTRdw==) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media; Pages: 218
  探讨了创建和转换特征的技术，这是机器学习工作流程中构建有效模型的关键阶段。
