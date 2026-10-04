---
course: "getting-started-with-scikit-learn"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification"
sourceId: 516
chapter: "supervised-learning-classification"
title: "监督学习：分类"
order: 3
description: "使用 Scikit-learn 实现和评估逻辑回归、KNN 和基本 SVM 等常见分类算法。"
hasQuiz: true
---

之前，我们侧重于使用回归来预测连续数值。本章将转向分类，其目标是将数据点分配到预定义的类别。与回归中目标变量 $y$ 是连续的不同，在分类中，$y$ 属于一个有限的离散类别集合，例如 {垃圾邮件, 非垃圾邮件} 或 {猫, 狗, 鸟}。

我们将考察 Scikit-learn 中几种常见的分类算法。你将学习实现逻辑回归（一种适用于分类任务的线性模型）、K 近邻 (KNN)（一种基于实例的方法），以及支持向量机 (SVM) 的基本知识。一个重要的方面是有效评估这些模型。我们将介绍专门为分类设计的指标，包括准确率、精确率、召回率、F1 分数以及混淆矩阵，并演示如何使用 Scikit-learn 函数计算它们。你将获得构建和评估标准分类模型的实践经验。
