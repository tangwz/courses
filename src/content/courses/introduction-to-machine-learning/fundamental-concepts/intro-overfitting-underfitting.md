---
course: "introduction-to-machine-learning"
chapter: "fundamental-concepts"
lesson: "intro-overfitting-underfitting"
sourceId: 1472
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-2-fundamental-concepts/intro-overfitting-underfitting"
title: "过拟合与欠拟合简介"
description: "了解过拟合（过于复杂）和欠拟合（过于简单）模型的基本要点。"
order: 6
plots: ["plots/1472-0.json"]
sourceHash: "eb01cf2785a3969d2907e2f439e66e6a8ca0468c5fd86928ef006d7749762422"
sourceCorrections: []
---

当我们训练一个机器学习 (machine learning)模型时，我们的目标不仅仅是在它已见过的数据（训练数据）上表现良好。我们希望模型能有很好的*泛化能力*，这意味着它应该也能对新的、未见过的数据做出准确预测。可以把它想象成准备考试：你不想只记住练习题的答案；你希望理解这些要点，这样你就能在实际考试中回答*不同*的问题。

两个常见问题会阻碍模型良好泛化：欠拟合 (underfitting)与过拟合 (overfitting)。它们代表了模型从数据中学习的两个极端情况。

### 欠拟合 (underfitting)：过于简单

想象一下，你试图画一条直线，穿过明显呈曲线分布的数据点。这条直线过于简单，无法捕捉到数据背后呈现的形态。这就是**欠拟合**。

欠拟合的模型不够复杂，无法学到训练数据中的主要趋势。它不仅在训练数据上表现差，在新的、未见过的数据（例如测试集）上表现也差。它未能掌握特征与标签之间的关联。

- **表现：** 训练集和测试集上的错误率都很高。
- **比喻：** 考试前你只看了章节概述，没有细致学习。你在练习题和实际考试中都不会考好，因为你没有学到足够的内容。

### 过拟合 (overfitting)：过于复杂

现在想象一下，你画了一条狂野的、蜿蜒的线，完美地穿过训练集中的*每一个*数据点。虽然它在训练数据上看起来表现出色，但这条线可能不仅学到了数据背后的形态，还学到了随机噪声以及该特定数据集的独有特点。这就是**过拟合**。

过拟合的模型过于复杂。它实质上是记住了训练数据，包括其中的噪声，而不是学习到普遍的形态。当面对新数据时，这些新数据不会有完全相同的噪声和独有特点，模型的表现就会很差。

- **表现：** 训练集上的错误率非常低，但测试集上的错误率很高。
- **比喻：** 你记住了每一个练习题及其准确答案，包括任何错别字。你练习题得了满分，但当实际考试出现略有不同的题目，考察相同的要点时，你就会很吃力，因为你没有学到背后的原理。

### 最佳状态：良好拟合

理想的模型介于这两种极端之间。它足够复杂，能捕捉到数据背后呈现的趋势，又足够简单，避免记住噪声。这种模型达到**良好拟合**，并能很好地泛化到新数据上。

- **表现：** 训练集上的错误率较低，测试集上的错误率也同样较低。
- **比喻：** 你学习要点并练习不同类型的问题。你在练习题上表现良好，并将学到的知识运用到实际考试中取得成功。

以下图表说明了这三种情况：



![欠拟合 vs. 良好拟合 vs. 过拟合](plots/1472-0.json)



> 散点代表训练数据。蓝色虚线（欠拟合 (underfitting)）过于简单。绿色实线（良好拟合）呈现普遍趋势。粉色点线（过拟合 (overfitting)）过于紧密地跟随训练数据点，包括噪声。

识别并避免过拟合和欠拟合是机器学习 (machine learning)中的一个主要难题。我们使用的技巧，例如将数据分成训练集和测试集、选择合适的模型复杂度（这与参数 (parameter)和超参数 (hyperparameter)有关），以及使用性能指标（接下来会讲到），都是为了帮助我们找到那个“最佳状态”，并构建在新数据上真正有用的模型。

## 参考资料

- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: Chapter 1, Chapter 3
  经典教材，为机器学习提供了全面的理论基础，在早期章节中详细解释了过拟合和欠拟合。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 4
  一本实用指南，通过实际案例和现代机器学习框架清晰解释了过拟合和欠拟合，提供了一个实践视角。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 5
  深度学习领域的奠基性教材，对机器学习基础知识进行了严谨的理论处理，包括模型容量、过拟合和欠拟合。
