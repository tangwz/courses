---
course: "getting-started-with-scikit-learn"
chapter: "supervised-learning-classification"
lesson: "intro-classification-problems"
sourceId: 1853
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/intro-classification-problems"
title: "分类问题介绍"
description: "理解分类的目标：预测离散的类别标签。"
order: 1
plots: ["plots/1853-0.json"]
sourceHash: "0e024d23fc1b603a9e13b450148ca660231c81ef430d3a25a14e1d04539226bf"
sourceCorrections: []
---

在研究了目标是预测连续数值的回归问题之后，我们现在转向**分类**。在分类任务中，目标根本不同：我们旨在将每个输入数据点分配到几个预设的、不同的类别或**类**中的一个。

分类是机器学习 (machine learning)中的一项任务，目标变量 $y$ 是离散的，属于一个有限的标签集。例如，一封电子邮件可能被归类为 {`垃圾邮件`, `非垃圾邮件`}，一个肿瘤可能被诊断为 {`良性`, `恶性`}，或者一个客户交易可能被标记 (token)为 {`欺诈`, `非欺诈`}。这些是**二元分类**的例子，其中只有两种可能的结果。

分类问题也可以涉及两个以上的类别。这被称为**多类别分类**。例子包括：

- 识别手写数字 ({`0`, `1`, `2`, ..., `9`})。
- 将新闻文章分类为主题 ({`体育`, `科技`, `政治`, `商业`})。
- 对动物图片进行分类 ({`猫`, `狗`, `鸟`, `鱼`})。

### 分类的目标

形式上，给定一个包含输入特征 $X$ 和相应类别标签 $y$ 的数据集，分类算法（常被称为**分类器**）的目标是学习一个映射函数 $f$。该函数将新的、未见过的数据点的特征 $X$ 作为输入，并预测其类别标签 $\hat{y}$：

$\hat{y} = f(X)$

其中 $\hat{y}$ 属于预设的可能类别集合 $C = \{c_1, c_2, ..., c_k\}$。分类器本质上是在特征空间中学习一个决策边界，以分离不同的类别。

从视觉上看差异：回归旨在拟合一条穿过数据点的线或曲线，而分类旨在找到数据点组之间的边界。



![比较：回归与分类](plots/1853-0.json)



> 回归旨在预测连续值（蓝色线拟合蓝色点），而分类旨在将数据分离成不同的组（红色虚线将绿色圆形与紫色方形分离）。

### 分类为何重要

分类是监督机器学习 (machine learning)的一个核心组成部分，应用广泛。构建准确的分类器使我们能够自动化决策过程，识别模式，并从带标签的数据中获得认识。从过滤垃圾邮件到辅助医疗专业人员以及了解客户行为，分类模型都是不可或缺的工具。

在本章接下来的部分中，我们将研究 Scikit-learn 提供的用于处理分类任务的特定算法，包括逻辑回归、K-近邻和支持向量 (vector)机。我们还将学习如何使用合适的度量标准来评估这些分类器的性能，因为仅仅测量“准确率”通常不足以了解模型的真正效果。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://doi.org/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer Science+Business Media; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  提供理解分类和回归的统计框架。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供使用 Scikit-learn 解决分类问题的实践示例和代码。
- [Classification](https://scikit-learn.org/stable/supervised_learning.html#classification) — scikit-learn developers (2024)
  Scikit-learn 库中分类功能和算法的官方指南。
