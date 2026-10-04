---
course: "basics-model-evaluation-metrics"
chapter: "basic-evaluation-workflow"
lesson: "training-simple-model-conceptual"
sourceId: 4041
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-5-basic-evaluation-workflow/training-simple-model-conceptual"
title: "训练一个简单模型"
description: "对评估流程中模型训练阶段的概述。"
order: 4
plots: []
sourceHash: "10f4e98bf184763854da1ea714c2f39d8aee4cccf6b54326c1cfd4a0dde18e2f"
sourceCorrections: []
---

我们已成功将数据集分成两个不同的部分：训练集和测试集。评估流程的下一个合理步骤是使用训练集来真正教授我们的机器学习 (machine learning)模型。这个阶段被称为**模型训练**或**模型拟合**。

可以把训练集想象成你为考试而学习的课本和练习题。模型就像学生一样，研习这些材料，从而学习其中的道理和规律。

### 学习过程

在训练期间，机器学习 (machine learning)算法会接收训练数据（包括输入特征和相应的已知结果或标签），并重复调整其内部设置，这些设置通常被称为参数 (parameter)或权重 (weight)。所使用的具体算法取决于问题的类型（分类或回归）以及所选的模型（例如线性回归、逻辑回归、决策树等）。

例如：

- 如果我们正在构建一个**回归模型**来预测房价，训练算法可能会调整参数，以找到最能表示房屋特征（如面积、卧室数量）与训练数据中价格之间关系的直线（或曲线）。目标是最小化此训练数据上的预测误差。
- 如果我们正在构建一个**分类模型**来检测垃圾邮件，算法可能会学习规则或识别电子邮件文本（来自训练集）中的模式，从而区分垃圾邮件和非垃圾邮件。它会自行调整，以尽可能正确地分类更多的训练邮件。

核心想法是模型尝试*仅*基于训练集中提供的示例，找到从输入特征到输出目标的最佳可能对应关系。

> 训练过程会使用训练数据，并使用机器学习算法来生成一个能够做出预测的已训练模型。

### “已训练”意味着什么？

训练阶段的输出不是数据；而是模型本身，现在它已配置了学习到的参数 (parameter)。这个“已训练模型”包含了在训练数据中发现的模式。它现在可以应用于新的、未见过的数据来做出预测。

### 保持测试集独立

绝对有必要记住，**测试集在训练期间绝不能展示给模型**。测试集就像期末考试。如果学生（我们的模型）在学习（训练）时看到了考题（测试数据），那么他们在考试中的表现就无法真实反映他们的学习情况。同样，在训练期间将模型暴露给测试数据，将导致对其在新数据上的表现评估过于乐观和不可靠。

这个训练步骤纯粹是为了从数据的指定训练部分中学习。一旦模型训练完成，我们就会进入流程的下一步：使用这个已训练模型对未见过的测试集进行预测，这将使我们能够评估它真正学习泛化的程度。

## 参考资料

- [An Introduction to Statistical Learning: with Applications in R (2nd Edition)](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
  这本基础教材清晰地解释了统计学习概念，包括训练集和测试集的作用以及模型拟合的过程。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (3rd Edition)](https://books.google.com/books?id=f_GZEAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  这本实用指南涵盖了整个机器学习流程，包括模型训练的详细解释以及使用独立训练集和测试集的重要性。
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — Scikit-learn developers (2024)
  Scikit-learn 官方用户指南解释了标准的机器学习工作流程，强调了数据分割在训练和评估中的作用。
