---
course: "intro-feature-engineering"
chapter: "feature-engineering-foundations"
lesson: "ml-workflow-revisit"
sourceId: 1299
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-1-feature-engineering-foundations/ml-workflow-revisit"
title: "重温机器学习工作流程"
description: "回顾标准机器学习流程以及特征工程在其中的位置。"
order: 1
plots: []
sourceHash: "1676f375e2fe558dae42f97ea6ef43e4fc1610864253d70f430935121a4a7080"
sourceCorrections: []
---

如果您以前从事过机器学习 (machine learning)项目，那么您可能熟悉构建和部署模型所涉及的一般步骤顺序。虽然具体问题和背景不同会导致一些差异，但常见的机器学习工作流程是一个有用的框架。了解特征工程在此过程中的位置，对于理解其作用非常重要。

让我们快速回顾一下典型阶段：

1. **问题定义与数据收集：** 明确定义您要解决的问题（例如，分类、回归）并收集相关的原始数据。这些数据可能来自数据库、API、日志、传感器或各种其他来源。
2. **数据审查与准备：** 这是进行初步数据处理工作的地方。它包括审查数据以理解其结构、分布和潜在问题。这一阶段特别包含**特征工程**，其中包括：
   - *数据清洗：* 处理缺失值、纠正错误和处理异常值。
   - *特征转换：* 数值特征缩放、应用数学转换（例如对数或Box-Cox）和编码类别变量。
   - *特征生成：* 从现有特征生成新特征（例如，交互项、多项式特征、从日期中提取信息）。
   - *特征选择：* 识别并选择模型最相关的特征，可能移除冗余或不相关的特征。
3. **模型选择与训练：** 根据问题和准备好的数据选择合适的机器学习算法（例如，线性回归、随机森林、神经网络 (neural network)）。然后，模型根据处理过的特征和相应的目标标签进行训练（在监督学习 (supervised learning)中）。
4. **模型评估：** 在模型训练期间未见过的独立测试数据集上，使用适当的度量指标（例如，准确率、精确率、召回率、均方根误差）评估训练模型的性能。此步骤有助于判断模型对新数据的泛化能力。
   "5. **模型部署与监控：** 使训练好的模型可用于对新数据进行预测。这通常涉及将其集成到应用程序或系统中。需要持续监控以确保模型的性能不会随时间下降。"

以下图表显示了此工作流程，突出了数据准备和特征工程的核心作用：

> 典型的机器学习工作流程，显示特征工程作为数据准备阶段的重要部分，为模型选择和训练提供输入。请注意迭代循环，它们表示根据评估结果进行的完善。

需要认识到，这个工作流程并非严格线性。模型评估阶段的结果常常会把您带回到先前的步骤。糟糕的模型性能可能表明：

- 需要更多或不同的数据（步骤1）。
- 特征未得到适当处理，需要调整特征缩放、编码或转换（步骤2b）。
- 需要生成新特征，或选择不同的特征（步骤2b）。
- 所选模型或其超参数 (parameter) (hyperparameter)需要调优（步骤3/4）。

因此，特征工程处于一个重要的交汇点。在此做出的选择直接决定了输入到学习算法的数据质量。正如章节引言所强调的，原始数据很少是最佳的。数值数据可能跨越非常不同的尺度，类别数据需要数值表示，而有价值的信息可能隐藏在现有变量的组合或转换中。本课程专门关注步骤2b中使用的技术，使您能够有效地为成功的机器学习应用准备数据。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的指南，涵盖完整的机器学习工作流程，从数据收集和预处理到模型训练、评估和部署，并包含特征工程技术的具体章节。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  一本基础书籍，全面讨论了各种特征工程技术及其在机器学习工作流程中的重要性。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供了构建机器学习系统的整体视图，强调了工作流程的迭代性质、数据质量和整个生命周期中的特征管理。
- [Machine Learning Engineering](https://leanpub.com/machine-learning-engineering) — Andriy Burkov (2020)
  Publisher: True Positive Inc.; Pages: 310
  提供构建和维护机器学习系统的实用和全面视角，涵盖从问题制定到部署和监控的端到端流水线。
