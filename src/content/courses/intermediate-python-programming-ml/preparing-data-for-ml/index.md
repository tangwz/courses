---
course: "intermediate-python-programming-ml"
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-5-preparing-data-for-ml"
sourceId: 562
chapter: "preparing-data-for-ml"
title: "为机器学习准备数据"
order: 5
description: "学习使用 Python 和 Scikit-learn 进行重要的数据准备技术，包括特征工程、缩放和划分。"
hasQuiz: true
---

原始数据通常不适合直接输入到机器学习算法中。模型需要干净、格式正确的数值数据才能有效运行。本章介绍将原始数据集转换为适用于模型训练的格式的重要技术。

你将了解典型机器学习项目中的标准步骤，特别是数据准备阶段。我们将介绍以下实用方法：

*   **特征工程：** 从现有数据构建有意义的输入变量。
*   **处理类别数据：** 使用独热编码等技术将非数值特征（如文本标签）转换为数值表示。
*   **特征缩放：** 规范化或标准化数值特征，以确保它们处于可比较的尺度上，这通常是算法所要求的。
*   **数据划分：** 将数据集正确划分为训练集和测试集，以准确评估模型性能。
*   **使用 Scikit-learn 流水线：** 将这些预处理步骤简化为一致且可复用的工作流程。

在本章结束时，你将能够使用 Pandas 和 Scikit-learn 等 Python 库应用这些预处理技术，从而有效准备机器学习任务所需的数据。
