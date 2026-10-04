---
course: "introduction-to-machine-learning"
chapter: "preparing-your-data"
lesson: "importance-data-preprocessing"
sourceId: 1522
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-6-preparing-your-data/importance-data-preprocessing"
title: "数据预处理的重要性"
description: "理解数据清洗和准备为何是模型训练前的重要步骤。"
order: 1
plots: []
sourceHash: "43aaa39dc710997f6b61bd189865eef5d92db68249426e21027f681eeef70bcd"
sourceCorrections: []
---

将构建机器学习 (machine learning)模型想象成烹饪一道美食。你可能拥有最好的菜谱（算法）和最棒的厨师（你自己！），但如果食材（原始数据）是腐坏或不合适的，最终的菜品很可能不会美味。“垃圾进，垃圾出”（GIGO）的原则在机器学习中尤为适用。

“机器学习算法功能强大，但它们本质上是数学过程，要求数据以非常特定、整洁和一致的格式呈现。原始数据很少能直接满足这些要求。以下是为什么数据准备不仅仅是初步的杂务，而是取得成功的基础步骤：”

1. **算法不明白混乱：** 大多数算法设计用于处理数值数据。它们无法直接处理缺失的条目（通常表示为 `NaN`、`None` 或空单元格）。将包含缺失值的数据输入到许多算法中，只会导致错误或产生不可靠的结果。同样，文本类（如“红色”、“绿色”、“蓝色”）或标签类（如“是”、“否”）的分类数据需要转换为算法能够理解的数值格式。
2. **不一致的数据导致学习不准：** 想象一个数据集，其中温度有时以摄氏度记录，有时以华氏度记录，没有明确的指示。或者一项调查中，“是”的回复被不一致地输入为“Y”、“yes”或“1”。除非你对它们进行标准化，否则算法会将其视为不同的值。不一致性会引入噪声和混淆，阻碍模型学习真实的数据规律。
3. **特征尺度很重要：** 依赖距离（如 K-近邻）或基于梯度的优化（如使用梯度下降 (gradient descent)训练的线性回归）的算法对特征的尺度很敏感。如果一个特征的范围是 0 到 1，而另一个特征的范围是 0 到 1,000,000，那么范围较大的特征可能会不成比例地影响结果或显著减慢学习过程。特征缩放，例如通过归一化 (normalization) ($x' = \frac{x - \min(x)}{\max(x) - \min(x)}$) 和标准化 ($x' = \frac{x - \mu}{\sigma}$) 引入的方法，有助于使各特征的影响力趋于平衡。
4. **不相关或冗余信息会影响性能：** 有时数据集中包含对预测任务无用的特征，或者彼此高度相关（冗余）的特征。虽然本章侧重于清洗和格式化，但数据准备的后续阶段可能涉及选择最相关的特征，以提高模型性能和效率。

> “垃圾进，垃圾出”原则在机器学习中的体现。预处理将原始数据转换为可用格式，从而大幅提升模型效果。

忽视数据预处理就像在不稳固的地基上盖房子。结构可能能够建成，但它不会稳固或可靠。花时间清洗和整理数据，确保你的算法有最好的机会来学习有意义的模式并做出准确的预测。本章介绍的技术构成了实际机器学习开发的基础。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实践指南，涵盖数据清洗、缺失值处理、分类特征编码和特征缩放（标准化、归一化），并附有Python示例。
- [Introduction to Machine Learning with Python: A Guide for Data Scientists](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/) — Andreas C. Müller and Sarah Guido (2016)
  Publisher: O'Reilly Media
  解释了数据预处理技术、其作用以及如何使用scikit-learn进行实践应用。
- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2024)
  官方文档，详细说明了包括缩放、编码和填充在内的各种预处理方法，并提供了具体解释和示例。
