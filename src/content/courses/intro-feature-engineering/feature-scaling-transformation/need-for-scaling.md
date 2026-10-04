---
course: "intro-feature-engineering"
chapter: "feature-scaling-transformation"
lesson: "need-for-scaling"
sourceId: 1337
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation/need-for-scaling"
title: "特征缩放的必要性"
description: "理解特征缩放对于基于距离的算法和梯度下降优化的重要原因。"
order: 1
plots: ["plots/1337-0.json"]
sourceHash: "a97f034931c5445003546869a833581433dd574117d8063fae53aef8941f2419"
sourceCorrections: []
---

你已经学习了如何为机器学习 (machine learning)模型准备分类数据。现在，我们来看数值特征。你可能会问：“为什么我们不能直接将原始数值输入到算法中呢？”毕竟它们已经是数字了。尽管有些模型能够很好地处理原始数值输入，但当数值特征经过缩放后，许多模型的表现会好得多，或者收敛速度会快得多。

想象一个包含两个特征的数据集：一个人的`年收入`（例如，从20,000美元到200,000美元）和他们的`工作经验年限`（从0到40年）。如果我们在某些算法中直接使用这些特征，`年收入`值的大小可能会掩盖`工作经验年限`的贡献，仅仅因为其数值大得多。这并非必然是因为收入本身更重要，而纯粹是尺度差异造成的影响。特征缩放旨在通过使所有数值特征处于可比较的水平来防止这种扭曲。

### 对基于距离的算法的影响

许多机器学习 (machine learning)算法都依赖于计算数据点之间的距离。例子包括：

- **K-近邻算法 (KNN):** 根据其最近邻居的多数类别来对一个点进行分类。
- **K-均值聚类:** 根据最小化到聚类中心的距离将点分组到簇中。
- **支持向量 (vector)机 (SVM) 采用径向基函数 (RBF) 核:** 使用距离来确定支持向量的影响。

考虑二维空间（$x$ 和 $y$）中两点 $p$ 和 $q$ 之间的标准欧几里得距离计算公式：

$\text{距离}(p, q) = \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2}$

如果特征 $x$（例如，`年收入`）的范围是100,000，而特征 $y$（例如，`工作经验年限`）的范围是40，那么收入相差10,000 ($p_x - q_x = 10000$) 将对平方根内的和贡献 $(10000)^2 = 100,000,000$。经验相差10 ($p_y - q_y = 10$) 只贡献 $10^2 = 100$。收入特征完全主导了距离计算。缩放确保两个特征对距离度量做出更公平的贡献，更准确地反映它们的相对重要性。

### 对基于梯度下降 (gradient descent)算法的影响

使用梯度下降进行优化的算法也从特征缩放中获得很大益处。这包括：

- **线性回归**
- **逻辑回归**
- **神经网络 (neural network)**
- **正则化 (regularization)模型 (Lasso, Ridge)**

梯度下降通过采取与成本函数负梯度成比例的步骤来更新模型参数 (parameter)（权重 (weight)）。权重 $w_j$ 的更新规则通常如下所示：

$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$

这里，$\alpha$ 是学习率，$\frac{\partial J}{\partial w_j}$ 是成本函数 $J$ 对权重 $w_j$ 的偏导数。如果特征的尺度差异很大，相应的梯度也会有很大差异。

- **收敛速度：** 尺度较大的特征可能会产生较大的梯度，导致优化过程主要集中在调整这些特征的权重，而较小尺度特征的权重则收敛缓慢。缩放有助于使梯度正常化，从而实现更平衡、通常更快的收敛到最佳解。
- **稳定性：** 对于未缩放的特征，成本函数的表面可能会变得细长（像一个长而窄的山谷）。梯度下降可能会在狭窄的轴线上来回振荡，难以有效地找到最小值。可能需要更小的学习率来防止发散，从而进一步减慢训练速度。缩放通常会使成本函数表面更接近球形或条件更好，从而允许梯度下降更直接地走向最小值，通常可以安全地使用更大的学习率。

考虑两个参数的成本函数曲面形状。如果没有缩放，如果特征的范围差异很大，等高线可能会显得细长。经过缩放后，等高线变得更接近圆形，使得梯度下降更容易找到最小值。



![成本函数等高线 (未缩放 vs. 缩放特征)](plots/1337-0.json)



> 左侧，细长的等高线表示特征尺度不同时的成本曲面，这可能减慢梯度下降的速度。右侧，更接近圆形的等高线表示缩放后的成本曲面，这使得收敛更直接。

### 对尺度不那么敏感的算法呢？

基于树的算法，例如决策树、随机森林和梯度提升机，通常被认为对特征的尺度不那么敏感。这是因为它们通过基于特征阈值（$\text{特征} > \text{阈值}$）来划分数据。阈值的选择与特征的整体尺度无关。

然而，即使对于基于树的模型，缩放有时也可能间接有益，例如：

- 如果在树算法中应用了正则化 (regularization)技术。
- 对于可能包含距离计算或线性组件的特定实现或相关算法。
- 为了预处理流程中的一致性，特别是当比较不同模型类型时。

### 总结

特征缩放是为许多机器学习 (machine learning)算法准备数值数据的一个基础步骤。它解决了值范围较大的特征可能不成比例地影响模型训练的问题，这并非因为它们固有的重要性，而仅仅因为它们的尺度。通过将特征带到相似的尺度，我们帮助基于距离的算法进行更有意义的比较，并使基于梯度下降 (gradient descent)的算法更快、更稳定地收敛。接下来的部分将讨论实现这一目标的具体技术，如标准化、归一化 (normalization)和鲁棒缩放。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R (2nd Edition)](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
  一本基础教科书，解释了各种机器学习算法，含蓄地说明了特征尺度对基于距离和梯度下降方法的算法的影响。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (3rd Edition)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，涵盖了特征缩放对各种算法的重要性，包括使用Scikit-Learn的示例。
- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2024)
  官方文档，提供了各种特征缩放技术的实践细节和示例，以及它们在机器学习中的应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的教科书，涵盖了输入标准化和缩放对于优化深度学习模型和提高收敛稳定性的作用。
