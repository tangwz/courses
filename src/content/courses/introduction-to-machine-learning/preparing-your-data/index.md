---
course: "introduction-to-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-6-preparing-your-data"
sourceId: 470
chapter: "preparing-your-data"
title: "准备数据"
order: 6
description: "学习必要的数据预处理步骤，包括处理缺失值、特征缩放和分类数据编码。"
hasQuiz: true
---

机器学习模型高度依赖于其训练数据的质量。通常，您会遇到的数据集在被算法有效使用之前，需要大量的准备工作。实际数据通常是不完整、不一致的，或者格式不适合处理。

本章介绍数据预处理的基本技术。您将学习以下实用方法：

*   识别并处理数据集中缺失的值。
*   应用特征缩放，例如归一化 (Normalization)
    $$
    x' = \frac{x - \min(x)}{\max(x) - \min(x)}
    $$
    和标准化 (Standardization)
    $$
    x' = \frac{x - \mu}{\sigma}
    $$
    以将数值特征调整到统一的范围。
*   将分类（非数值）特征编码为模型可理解的数值表示，使用独热编码 (One-Hot Encoding) 等方法。
*   实现将数据划分为训练集和测试集。

学完本章后，您将明白为什么这些步骤是必需的，以及如何执行基本的数据清洗和转换任务，为机器学习模型准备数据。
