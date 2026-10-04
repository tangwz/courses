---
course: "getting-started-with-scikit-learn"
chapter: "supervised-learning-regression"
lesson: "intro-regression-problems"
sourceId: 1838
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-2-supervised-learning-regression/intro-regression-problems"
title: "回归问题简介"
description: "理解机器学习中回归的目标：预测连续目标变量。"
order: 1
plots: ["plots/1838-0.json"]
sourceHash: "0cd224f7840300ff2afcefd3d7d4a79cce92fd8884f71990e32fd96f853664f1"
sourceCorrections: []
---

在监督机器学习 (machine learning)中，我们的目标通常是学习从输入特征到输出变量的映射关系。当我们试图预测的输出变量是连续的数值时，我们处理的是一个**回归**问题。这与下一章将介绍的分类问题形成对比，分类问题的目标是预测一个离散的类别或标签（例如“垃圾邮件”与“非垃圾邮件”，或“猫”与“狗”）。

可以将回归视为尝试回答“有多少？”或“什么值？”。以下是回归技术的一些典型应用场景：

- **预测房价：** 给定房屋面积、卧室数量和位置等特征，预测房屋的售价。
- **预测需求：** 根据过去的销售数据、广告支出和季节性，估算下个月将售出的产品数量。
- **估算能耗：** 根据室外温度、一天中的时间点和入住率等因素，预测建筑物的用电量。
- **评估金融风险：** 根据贷款申请人的财务历史，计算其信用评分（一个连续值）。
- **预测作物产量：** 根据天气模式、土壤类型和肥料用量，估算作物产量。

在每个例子中，我们旨在预测的目标变量（价格、销售量、千瓦时、信用评分、产量）都可以取一系列广泛的连续值。

### 回归问题的核心组成部分

从数学角度看，我们可以这样来描述一个回归问题：我们有一个包含 $n$ 个观测值的数据集。每个观测值 $i$ 都有一组输入**特征**（也称为自变量或预测变量），表示为向量 (vector) $X_i$，以及一个对应的连续**目标变量**（或因变量、响应变量），表示为 $y_i$。

我们的目标是学习一个模型，本质上是一个函数 $f$，它能够近似特征 $X$ 和目标 $y$ 之间的关系：


$$
y \approx f(X)
$$


函数 $f$ 是从训练数据中学习得到的。一旦学习完成，该模型就可以用来预测新的、未见过特征输入 $X_{new}$ 的目标值 $y_{new}$。

### 回归的可视化

对于简单情况，特别是只有一两个特征时，我们可以可视化特征与目标之间的关系。如果我们只有一个特征 $x$ 和一个目标 $y$，我们可以创建一个散点图，其中每个点代表一个观测值 $(x_i, y_i)$。在这种可视化背景下，回归模型的目标通常是找到一条最能拟合这些点模式的直线或曲线。



![回归数据示例](plots/1838-0.json)



> 一个散点图，显示了单个特征（学习时长）与连续目标变量（考试分数）之间可能存在的关系。回归模型旨在捕捉这种趋势。

这个难点，以及线性回归等算法（我们将在下文介绍）的目的，在于找到函数 $f$ 的特定参数 (parameter)，以便根据提供的数据最准确地表示这种潜在关系。Scikit-learn 提供了各种算法的高效实现，专门用于解决这类问题。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in Python](https://link.springer.com/book/10.1007/978-3-031-38747-0) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38747-0](https://doi.org/10.1007/978-3-031-38747-0)
  一本全面且易懂的教科书，涵盖了统计学习的基础概念，包括各种回归技术。
- [User Guide: Supervised learning](https://scikit-learn.org/stable/supervised_learning.html) — scikit-learn developers (2024)
  官方文档中介绍监督学习算法（包括回归模型）的部分，提供了实际实现细节。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本使用 Scikit-learn 实现机器学习算法（包括回归）的实用指南，提供了清晰的解释和代码示例。
