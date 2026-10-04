---
course: "introduction-to-machine-learning"
chapter: "fundamental-concepts"
lesson: "parameters-hyperparameters"
sourceId: 1470
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-2-fundamental-concepts/parameters-hyperparameters"
title: "参数与超参数"
description: "区分从数据中学习到的模型参数和训练前设置的超参数。"
order: 5
plots: []
sourceHash: "747224461eca30f7b393a8aaca9ee2dfd42b8f4048054efd7e649e58c9e1b839"
sourceCorrections: []
---

在上一节中，我们明确了机器学习 (machine learning)模型从数据中学习规律。但模型到底*学到*了什么，我们又如何控制这个学习过程呢？这引出了与模型相关的两种基本类型的设置：参数 (parameter)和超参数 (hyperparameter)。了解它们之间的区别对于有效地训练和调优模型非常必要。

### 参数 (parameter)：模型学习到的内容

参数是模型的内部变量，其值直接从训练数据中估算或学得。可以把它们想象成模型在训练过程中自动调整的旋钮和表盘，目的是使误差最小化并做出更好的预测。这些变量捕捉了数据中发现的模式。

机器学习 (machine learning)模型从数据中学习模式。模型究竟学习了什么，以及如何控制其学习过程？与模型相关的两种主要设置是参数和超参数 (hyperparameter)。理解它们之间的差异对于有效训练和调整模型非常重要。例如，考虑一个试图将直线拟合到数据点上的简单线性回归模型，直线的方程通常写作 $y = mx + b$。

- $m$（直线的斜率）是一个参数。
- $b$（直线的y轴截距）也是一个参数。

在训练过程中，线性回归算法会迭代地调整 $m$ 和 $b$ 的值，以找到最能代表训练数据中输入 ($x$) 和输出 ($y$) 之间关系的直线。最终学得的 $m$ 和 $b$ 值定义了通过训练过程产生的特定模型。

参数的其他例子包括：

- 神经网络 (neural network)中的权重 (weight)和偏置 (bias)。
- 逻辑回归中的系数。

主要理解点是，参数是由算法本身*从数据中学习*的。你无需手动设置它们的值；训练过程会决定它们。

### 超参数 (parameter) (hyperparameter)：模型如何学习

另一方面，超参数是外部于模型的配置设置，其值无法从数据中估算。它们由机器学习 (machine learning)实践者在训练过程开始*之前*设置。可以把它们看作更高层次的控制，用于定义学习过程本身应如何运作或限制模型的复杂度。

你不能仅仅通过查看数据就知晓最佳超参数值；它们通常需要实验和调优（我们稍后会提及的一个过程）。

让我们考虑一些与课程大纲中提到的算法相关的例子：

- **学习率（用于梯度下降 (gradient descent)）：** 在使用梯度下降（我们将会讨论的一种优化技术）的算法中，例如线性回归或逻辑回归，学习率控制着更新参数（例如 $m$ 和 $b$）时的调整步长大小。学习率太小可能会使训练非常缓慢，而学习率太大则可能导致模型无法找到最佳参数值。这个学习率是你提前设置的一个超参数。
- **邻居数量（KNN中的K）：** 在K近邻算法（用于分类）中，'K' 代表算法在对新点进行预测时所考虑的相邻数据点的数量。它是应该查看3个最近的邻居？还是5个？或是10个？'K' 的值是你必须在运行KNN算法之前选择的超参数。
- **簇的数量（K均值中的K）：** 类似地，在K均值聚类算法中，'K' 代表你希望算法在数据中找到的簇的数量。你需要在开始聚类过程*之前*决定这个值。

其他例子包括神经网络 (neural network)中的层数或神经元数量、要使用的正则化 (regularization)类型（一种防止过拟合 (overfitting)的技术），或决策树的最大深度。

### 为什么要区分它们？

区分参数 (parameter)和超参数 (hyperparameter)很重要：

| 特点 | 参数 | 超参数 |
| --- | --- | --- |
| **来源** | 训练时从数据中学习 | 训练开始*前*手动设置 |
| **作用** | 定义特定模型的预测逻辑 | 配置训练过程、模型结构 |
| **确定方式** | 由学习算法估算 | 由实践者选择（通常通过调优） |
| **例子** | 系数（线性回归）、权重 (weight)（神经网络 (neural network)） | 学习率、K（在KNN/K均值中）、树深度 |

超参数控制着学习算法的整体行为和能力。找到好的超参数对实现良好的模型性能通常非常重要。你通常会尝试超参数的不同组合，为每种组合训练单独的模型，并评估它们（通常使用验证集，如前所述），以找到最适合你特定问题和数据的设置。

总而言之，参数是模型*从*数据中学习的，而超参数则由你*设置*，以指导模型如何学习。两者都是构建和优化机器学习 (machine learning)模型不可或缺的部分。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本经典的教材，涵盖机器学习的统计学基础，包括对线性回归等各种算法中模型参数的详细解释。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的综合性资源，详细阐述了神经网络中的参数（权重、偏差）和超参数（学习率、网络架构）。
- [CS229 Lecture Notes: Machine Learning](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGxC8womKJjgHMz_b9oGzu0h8LIOx0mWMWbrPtjfwnaV3O4Iw9NVT) — Andrew Ng, Tengyu Ma (2023)
  Publisher: Stanford University
  著名机器学习课程的讲义，清晰地解释了线性回归中的参数以及学习率等超参数的基本概念。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的机器学习指南，通过K-最近邻和神经网络等各种算法的例子，清晰地区分了参数和超参数。
