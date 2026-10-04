---
course: "getting-started-with-scikit-learn"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-4-data-preprocessing-feature-engineering"
sourceId: 520
chapter: "data-preprocessing-feature-engineering"
title: "数据预处理与特征工程"
order: 4
description: "学习Scikit-learn中重要的数据预处理技术，包括缩放、编码分类数据和处理缺失值。"
hasQuiz: true
---

机器学习模型很少能直接处理原始数据并取得好效果。实际数据集通常包含不一致、缺失值以及度量尺度差异大或非数值格式的特征。许多算法要求数据干净、数值化并进行适当缩放以达到最佳表现。例如，计算点之间距离的算法（如K近邻）或使用梯度下降优化的算法（如带正则化的线性回归）对输入特征的尺度很敏感。同样，大多数算法需要数值输入，因此有必要将分类文本数据转换为合适的格式。

本章侧重介绍使用Scikit-learn工具进行数据准备的重要技术。您将学习如何：

*   **缩放数值特征**：使用标准化（$$(X - \mu) / \sigma$$）和归一化（缩放到 $[0, 1]$ 范围）等方法，确保特征对模型训练有恰当的作用。
*   **编码分类特征**：使用独热编码和序数编码等策略，将分类特征转换为数值表示。
*   **处理缺失值**：通过填充策略，用估计值或统计值替换缺失项。

掌握这些预处理步骤，对于构建有效的机器学习模型非常重要。我们将学习Scikit-learn的变换器API，以便高效地应用这些技术。
