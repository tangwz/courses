---
course: "introduction-to-machine-learning"
chapter: "what-is-machine-learning"
lesson: "machine-learning-workflow-overview"
sourceId: 1459
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-1-what-is-machine-learning/machine-learning-workflow-overview"
title: "机器学习工作流程概述"
description: "简要了解机器学习项目中涉及的典型步骤。"
order: 6
plots: []
sourceHash: "11335d7e66e5990c7f6a558d012cd503c0d357817b68a9c08cd7b47605848dc7"
sourceCorrections: []
---

建立一个机器学习 (machine learning)系统并非魔法，而是一个有条理的过程。尽管具体细节可能因问题和数据而异，但大多数机器学习项目都遵循一套通用的步骤。可以将其视为一个路线图，指导你从初步构思到构建出可用的模型。了解这个工作流程，能为你在这门课程中学习的各种知识点和技巧提供背景信息。

以下是机器学习工作流程的典型概览：

1. **明确问题并把握整体：** 该解决方案将实现什么业务目标？如何衡量模型的表现？这是哪种类型的问题（例如：监督学习 (supervised learning)、无监督学习 (unsupervised learning)、分类、回归）？事先回答这些问题能明确范围和成功标准。例如，如果你想预测房价，这是一个监督学习中的回归问题，你可能会通过预测值与实际售价的接近程度来衡量成功。
2. **获取数据：** 机器学习模型从数据中学习，因此获取正确的数据必不可少。这可能涉及查询数据库、使用公共数据集、（合法合规地）抓取网页或使用API。数据的质量和数量显著影响模型的表现。
3. **了解和准备数据：** 原始数据通常是杂乱的。这一步通常最耗时，包含：

   - **数据初步分析（EDA）：** 通过可视化和统计摘要来理解数据。寻找规律、关联、异常值和分布。
   - **数据清洗：** 处理缺失值、纠正错误和解决不一致之处。
   - **特征工程/选择：** 从现有特征中创建新特征，或为模型选择最相关的特征。
   - **数据转换：** 对数值特征进行缩放，或将分类特征编码成适合算法的格式。我们将在第6章介绍具体方法。
4. **选择并训练模型：** 根据问题类型和数据初步分析，选择一个或多个候选模型（例如：用于预测值的线性回归、用于分类的K近邻、用于聚类的K均值）。然后，通过向模型提供准备好的数据（训练集）来“训练”它。在训练过程中，算法学习数据中的规律或关系。线性回归、KNN和K均值等模型的具体内容将在第3、4、5章详细介绍。
5. **评估模型：** 模型训练完成后，你需要评估其表现。这通过使用模型未见过的数据（测试集）来完成。你使用与问题类型相关的特定指标（例如：分类的准确率、回归的均方误差）来衡量表现。这一步有助于确定模型是否足够好，或者是否需要进一步优化。第2章介绍了基本指标，我们将在后续章节中再次讨论评估。
6. **调整与迭代：** 根据评估结果，你可能需要调整模型（例如：修改称为超参数 (parameter) (hyperparameter)的设置，第2章有简要提及），甚至回到之前的步骤。也许你需要更多数据、更好的特征，或者完全不同的模型。机器学习通常是一个迭代过程，包含训练、评估和调整的循环。
7. **提出解决方案并部署（本课程可选）：** 一旦对模型的表现满意，你就可以展示你的成果，并可能将模型部署到生产环境中，以便它能够对新的实时数据进行预测。

这个工作流程并非总是严格线性的。随着你对数据或模型表现的了解增加，你可能会重新审视之前的步骤。

> 机器学习项目中常见步骤的简化视图。注意其迭代性，通常需要回到早期阶段进行完善。

了解这些通用步骤提供了一个框架，有助于我们在后续章节中学习具体算法和技术。我们将在第7章从头到尾构建一个简单模型时，再次讨论这个工作流程。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  使用流行库提供机器学习工作流程的实践指南，涵盖从数据获取、准备到模型训练、评估和部署。
- [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2013)
  Publisher: Springer
  这本基础教科书解释了各种机器学习方法的统计原理，并间接涵盖了它们在结构化项目中的作用。
- [Python for Data Analysis](https://wesmckinney.com/book) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  对于理解数据获取、清洗、探索和准备至关重要，这些是任何机器学习工作流程中基础且耗时的步骤。
- [CS229 Lecture Notes](https://cs229.stanford.edu/) — Andrew Ng (2020)
  Journal: Stanford CS229: Machine Learning Course; Publisher: Stanford University
  提供机器学习概念的高层次概述，包括项目背景下的问题制定和模型评估。
