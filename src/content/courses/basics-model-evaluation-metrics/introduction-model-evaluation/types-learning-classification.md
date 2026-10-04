---
course: "basics-model-evaluation-metrics"
chapter: "introduction-model-evaluation"
lesson: "types-learning-classification"
sourceId: 3951
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-1-introduction-model-evaluation/types-learning-classification"
title: "学习问题的类型：分类"
description: "定义机器学习中的分类任务，并提供简单示例。"
order: 4
plots: []
sourceHash: "10dfb9207d700206e4f5f8d31c3d0f42b0562afe094098b664ecedc8c749141b"
sourceCorrections: []
---

在机器学习 (machine learning)中，一类常见的问题涉及训练计算机将事物分到不同的组或类别中。这项任务称为**分类**。可以把它想象成一个数字分院帽：你提供一些信息（一个输入），模型会告诉你该输入属于哪个预设类别。

分类模型的目的是学习从输入特征（数据的特点）到特定输出标签（常称为类别）的映射。这些标签代表离散、不同的类别。

### 分类常见例子

你经常遇到分类问题，可能都没有意识到：

1. **电子邮件垃圾邮件检测**：电子邮件服务会查看邮件内容、发件人及其他特征，并将其归类为`垃圾邮件`或`非垃圾邮件`（正常邮件）。这些是两种可能的类别。
2. **图像识别**：模型分析图像并根据其内容进行分类，例如识别图片中是否包含`猫`、`狗`或`鸟`。
3. **医学诊断**：根据患者症状和检测结果（特征），模型可能会分类患者是否患有某种`疾病`或`未患病`。
4. **情感分析**：分析一段文本（例如产品评论）以将其表达的情感分类为`积极`、`消极`或`中性`。

在每种情况下，模型的输出都是从有限的可能性集合中选出的特定类别标签。

### 分类模型如何运作

从宏观上看，分类模型从已知正确类别的数据中学习模式（这称为标注训练数据）。例如，要构建一个垃圾邮件检测器，我们会向模型展示许多电子邮件示例，每封都已标记 (token)为`垃圾邮件`或`非垃圾邮件`。模型会研究这些邮件的特征（如特定词语、发件人信誉等），并学习区分垃圾邮件与正常邮件的规则或模式。

一旦训练完成，模型可以接收一封*新*的、未曾见过的邮件，检查其特征，并预测它属于哪个类别。

> 此流程图说明了分类模型如何处理输入特征以生成预测类别标签。

### 分类与回归

将分类与回归（我们接下来会讨论）进行比较很有用。分类将数据点分配到离散类别（如`垃圾邮件`/`非垃圾邮件`、`猫`/`狗`），而回归模型则预测连续的数值（如房价、明天的温度或学生的考试分数）。输出类型（类别与数值）是根本区别。

理解分类很重要，因为评估这些模型需要特定的衡量标准。我们不仅需要知道预测是否正确，还经常需要了解模型产生的错误*类型*。例如，在垃圾邮件检测中，错误地将一封正常邮件分类为垃圾邮件（误报）可能比让一封垃圾邮件通过（漏报）问题更大。为分类设计的衡量标准帮助我们准确地衡量这种性能，我们将在下一章详细研究。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  这本全面的教科书涵盖了监督学习的核心概念，包括各种分类方法及其理论基础。
- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/b100721](https://doi.org/10.1007/b100721)
  这本经典著作从概率角度介绍了机器学习，广泛涵盖了分类算法。
