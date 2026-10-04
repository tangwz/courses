---
course: "intro-eda-course"
chapter: "advanced-visualization-feature-engineering"
lesson: "intro-feature-engineering"
sourceId: 1183
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-5-advanced-visualization-feature-engineering/intro-feature-engineering"
title: "特征工程思想介绍"
description: "理解EDA的洞察如何指导新特征的创建（特征工程）。"
order: 3
plots: []
sourceHash: "06ebebf31378c6b0f135b61879dc3114b687b26ee0e095be83698d7e9cb2ba26"
sourceCorrections: []
---

探索性数据分析（EDA）不仅关乎理解现有数据；它也关乎为接下来的步骤，通常是机器学习 (machine learning)建模，准备数据。查看数据分布、可视化数据关联以及识别数据质量问题提供的信息，直接指导数据集的优化。这个旨在改善预测模型输入信号的优化过程，被称为**特征工程**。

可以将特征工程视为一门艺术和科学，它将原始数据转换成能够更好地向预测模型呈现其核心问题的特征，从而提高模型的准确性和表现。它是连接EDA期间所形成的理解与机器学习算法需求之间的桥梁。

### 为什么特征工程是必要的？

机器学习 (machine learning)算法从它们获得的数据中学习模式。这些输入数据的质量和形式显著影响它们有效学习的能力。原始数据，即使经过清洗，可能也并非最佳格式。以下是特征工程成为数据科学标准工作流程一部分的原因：

1. **提升模型表现：** 经过良好设计的特征能够更清晰地向算法展现数据的内在结构，帮助它们更有效地学习模式，从而带来更好的预测。有时，组合或转换现有特征能够显现之前隐藏的关联。
2. **满足算法要求：** 许多算法对输入数据有特定的要求。例如：
   - 像scikit-learn这样的库中，大多数算法需要数值输入。分类特征通常需要转换为数值表示（编码）。
   - 对特征尺度敏感的算法（如支持向量 (vector)机、主成分分析或使用梯度下降 (gradient descent)的算法），当特征被缩放到一个共同的范围或分布时，通常表现更好。
   - 一些模型假设数据遵循特定分布（如正态性）。转换可以帮助满足这些假设。
3. **融入领域知识：** 特征工程允许你将领域专长融入模型。例如，在零售场景中，你可能基于交易日期创建一个`days_since_last_purchase`（上次购买以来的天数）这样的特征，运用你对客户行为的理解。
4. **简化模型：** 有时，创建高效的特征可以使更简单的模型也能表现良好，从而使模型更易于解释和维护。

### 连接EDA与特征工程思路

你的EDA工作直接提供了潜在的特征工程步骤：

- **单变量分析：** 你是否观察到数值变量的分布偏斜？这可能暗示可以应用对数或平方根转换（稍后在数据转换部分讨论）。你是否识别出具有多级别分类变量？这将指导你选择编码策略。是否检测到异常值？特征工程可能涉及为异常值创建标记 (token)，或使用对异常值具有鲁棒性的转换。
- **双变量分析：** 散点图是否显示了数值变量之间的非线性关联？这可能促使创建多项式特征（$x^2$, $x^3$）或交互项（$x_1 * x_2$）。数值变量与分类变量之间的比较是否显示出跨类别的不同分布？这强调了分类特征的重要性以及进行适当编码的必要性。
- **数据类型和结构：** 日期或时间戳等特征以原始形式很少有用。EDA帮助你了解它们的范围和潜在模式，提示可以派生出像`day_of_week`（星期几）、`month`（月份）、`is_weekend`（是否周末）或时间差等特征。文本数据可能需要专门的特征提取技术（如TF-IDF或词嵌入 (embedding)），这通常由EDA期间的初步文本分析引导。

本质上，EDA突显你数据中的特点和潜在问题，而特征工程提供解决这些问题的方法，调整数据集以进行有效建模。以下部分将介绍具体的技术，例如从现有特征创建新特征、数值数据缩放以及分类变量编码，所有这些都依据此处讨论的原理。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  这本实用书籍涵盖了完整的机器学习工作流程，包括专门的数据预处理和特征工程章节。
- [Feature Engineering and Selection: A Practical Approach for Predictive Models](https://www.routledge.com/Feature-Engineering-and-Selection-A-Practical-Approach-for-Predictive-Models/Kuhn-Johnson/p/book/9781032090856) — Max Kuhn, Kjell Johnson (2019)
  Publisher: CRC Press; Pages: 314
  一本专门介绍特征工程的综合指南，涵盖了策略、技术以及特征选择在预测模型中的重要性。
- [CS109a: Introduction to Data Science (Fall 2023 Course Website)](https://harvard-iacs.github.io/cs109a/) — Pavlos Protopapas, Kevin A. Rader, Chris Tanner (2023)
  Publisher: Harvard University
  提供了数据科学和机器学习概念的概述，包括机器学习工作流程中基本的数据预处理和特征工程步骤。
