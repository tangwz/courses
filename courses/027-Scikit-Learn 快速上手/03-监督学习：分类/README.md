# 第 3 章：监督学习：分类

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification)

[返回课程目录](../README.md)

之前，我们侧重于使用回归来预测连续数值。本章将转向分类，其目标是将数据点分配到预定义的类别。与回归中目标变量 $y$ 是连续的不同，在分类中，$y$ 属于一个有限的离散类别集合，例如 {垃圾邮件, 非垃圾邮件} 或 {猫, 狗, 鸟}。

我们将考察 Scikit-learn 中几种常见的分类算法。你将学习实现逻辑回归（一种适用于分类任务的线性模型）、K 近邻 (KNN)（一种基于实例的方法），以及支持向量机 (SVM) 的基本知识。一个重要的方面是有效评估这些模型。我们将介绍专门为分类设计的指标，包括准确率、精确率、召回率、F1 分数以及混淆矩阵，并演示如何使用 Scikit-learn 函数计算它们。你将获得构建和评估标准分类模型的实践经验。

## 小节

- 1. [分类问题介绍](01-%E5%88%86%E7%B1%BB%E9%97%AE%E9%A2%98%E4%BB%8B%E7%BB%8D.md)
- 2. [逻辑回归用于分类](02-%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%94%A8%E4%BA%8E%E5%88%86%E7%B1%BB.md)
- 3. [K-近邻算法 (KNN)](03-K-%E8%BF%91%E9%82%BB%E7%AE%97%E6%B3%95%20%28KNN%29.md)
- 4. [在 Scikit-learn 中实现 KNN](04-%E5%9C%A8%20Scikit-learn%20%E4%B8%AD%E5%AE%9E%E7%8E%B0%20KNN.md)
- 5. [支持向量机 (SVM) 基本原理](05-%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%20%28SVM%29%20%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md)
- 6. [使用Scikit-learn实现SVM](06-%E4%BD%BF%E7%94%A8Scikit-learn%E5%AE%9E%E7%8E%B0SVM.md)
- 7. [分类评估指标](07-%E5%88%86%E7%B1%BB%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md)
- 8. [在 Scikit-learn 中计算指标](08-%E5%9C%A8%20Scikit-learn%20%E4%B8%AD%E8%AE%A1%E7%AE%97%E6%8C%87%E6%A0%87.md)
- 9. [实践操作：构建分类模型](09-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/quiz)
