---
course: "getting-started-with-scikit-learn"
chapter: "intro-setup-scikit-learn"
lesson: "what-is-scikit-learn"
sourceId: 1824
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-1-intro-setup-scikit-learn/what-is-scikit-learn"
title: "Scikit-learn 是什么？"
description: "Scikit-learn 库概述及其在 Python 科学计算栈中的定位。"
order: 1
plots: []
sourceHash: "a85f39eec13cdd87b88ef15225702d572d2e45fa200ea2f4ddbf803f8c5b8e60"
sourceCorrections: []
---

Scikit-learn 在 Python 中常被导入为 `sklearn`，它是执行机器学习 (machine learning)任务的核心库。这是一个开源项目，意味着它可以免费获取并由社区协作开发。它的主要目标是提供简单高效的预测性数据分析工具，这些工具对所有人可用，并可在多种情境下重复使用。Scikit-learn 是 Python 科学计算体系的一部分，能与 NumPy（用于数值运算）和 Pandas（用于数据处理）等其他常用库良好配合。

可以将 Scikit-learn 视为 Python 数据科学生态系统中的特定层。最底层是 Python 本身。在此之上，NumPy 提供了基本的 N 维数组对象和运算，SciPy 则提供更专业的科学和技术计算例程。Scikit-learn 大量依赖这些底层库，特别是 NumPy 数组，以实现其数据结构和高效计算。它专门专注于提供机器学习所需的算法和工具，而非 NumPy 的通用数值计算功能或 SciPy 更广泛的科学功能。通常，Matplotlib 或 Seaborn 等库会与 Scikit-learn 一起用于数据和模型结果的可视化。

> Scikit-learn 在常见 Python 数据科学生态系统中的定位，它构建于 NumPy 和 SciPy 之上，常与 Pandas 一起用于数据输入，并与 Matplotlib/Seaborn 一起用于可视化。虚线表示常见的使用模式而非严格的依赖关系。

Scikit-learn 之所以高效且受欢迎，原因在于其多项设计原则：

1. **一致性：** 它具有非常一致的应用程序编程接口（API）。无论是使用线性回归模型、支持向量 (vector)机，还是像缩放器这样的预处理工具，实例化、拟合（训练）和使用这些对象的方式都遵循可预测的模式。我们将在后续章节中研究核心 API 组件（估计器、预测器、转换器）。这种一致性大大降低了学习不同算法的难度。
2. **简单与高效：** 该库优先考虑易用性而不牺牲性能。许多核心算法的内部实现都使用了优化代码（通常是 Cython 或 C），这使得它们在合理规模的数据集上计算高效。
3. **功能全面：** Scikit-learn 提供了广泛的机器学习任务实现，包括：
   - **分类：** 识别对象所属的类别。
   - **回归：** 预测与对象关联的连续值属性。
   - **聚类：** 将相似对象自动分组。
   - **降维：** 减少需要考虑的随机变量数量，以提高计算效率或避免维度灾难。
   - **模型选择：** 比较、验证和选择参数 (parameter)及模型。
   - **预处理：** 特征提取和归一化 (normalization)。
4. **开源与社区驱动：** 作为遵循宽松 BSD 许可证的开源项目，它促进了在学术研究和商业应用中的广泛采用。它得益于庞大活跃的用户社区和专注的开发团队，后者持续改进和维护该库。

Scikit-learn 专为需要将机器学习算法应用于数据的开发者、数据科学家和研究人员设计。开始使用它并不要求您掌握每个算法的深奥数学细节，尽管理解其基本原理对有效应用总是有好处的。在本课程中，我们将广泛地使用 Scikit-learn 来加载数据、预处理、训练各种监督学习 (supervised learning)模型（如回归和分类）、评估它们的性能，并使用流水线构建高效的工作流。本章旨在帮助您完成设置，并熟悉该库的基本结构和约定。

## 参考资料

- [Scikit-learn Documentation](https://scikit-learn.org/stable/) — Scikit-learn Developers (2024)
  Scikit-learn的综合官方文档，涵盖从安装到API参考和用户指南的所有方面。
- [Hands-On Machine Learning with Scikit-Learn, Keras, & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广受好评的实践指南，用于使用Scikit-learn以及深度学习框架实现机器学习算法。
