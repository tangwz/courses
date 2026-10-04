---
course: "basics-model-evaluation-metrics"
chapter: "introduction-model-evaluation"
lesson: "overview-evaluation-process"
sourceId: 3956
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-1-introduction-model-evaluation/overview-evaluation-process"
title: "模型评估过程概述"
description: "机器学习模型评估步骤概览。"
order: 6
plots: []
sourceHash: "700865671398c54ce58618c3b2821ad3db2e39f9c94962528afc8689d9a25f65"
sourceCorrections: []
---

既然我们了解了*为什么*需要评估机器学习 (machine learning)模型，以及它们解决的基本问题类型（分类和回归），接下来就看一下评估过程本身的常用步骤。可以将其视为检验模型表现的指引。

从整体来看，评估机器学习模型通常遵循以下步骤：

1. **数据准备**：从数据集开始。此处首要的一步是将数据分成至少两部分：**训练集**和**测试集**。
2. **模型训练**：使用*训练集*来训练模型。模型从这些数据中学习模式、关系或规则。
3. **模型预测**：使用训练好的模型在*测试集*上进行预测。由于模型从未见过这些测试数据，这些预测能公平地评估其能力。
4. **性能评估**：将模型在测试集上的预测与测试集中的实际真实值进行比较。此处会用到**评估指标**。您会计算具体的得分（例如分类的准确率或回归的均方误差），以量化 (quantization)模型的性能。
5. **结果解读**：最后，您需要分析这些指标得分。它们是否符合您的要求？模型是否足以满足其预期用途？这一步能帮助您决定模型是否已准备就绪，或者是否需要进一步改进。

让我们将这个基本流程可视化：

> 机器学习模型评估工作流程的简化视图。

"此处最重要的原则是在模型训练期间未遇到的数据（测试集）上评估模型。这种分离有助于避免过于乐观的结果，并能更实际地估计模型在面对新数据时的表现。"

在接下来的章节中，我们将研究步骤4中用于分类（第2章）和回归（第3章）问题的具体指标。我们还将详细介绍数据分割技术（第4章），以确保评估的可靠性。本章提供了关于我们通常*为什么*以及*如何*进行模型评估的基本理解。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125968/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media, Inc.; Pages: 864
  一本实用指南，通过清晰的示例介绍了数据分割、模型训练、预测和评估指标。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本提供机器学习统计基础的教科书，涵盖了使用测试集进行模型评估和选择。
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) — Andrew Ng, deeplearning.ai, and Stanford Online (2022)
  Publisher: DeepLearning.AI and Stanford Online
  一门更新的在线课程，涵盖机器学习概念，包括评估工作流程、数据分割和性能指标。
