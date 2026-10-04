---
course: "basics-model-evaluation-metrics"
chapter: "introduction-model-evaluation"
lesson: "types-learning-regression"
sourceId: 3954
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-1-introduction-model-evaluation/types-learning-regression"
title: "学习问题类型：回归"
description: "定义机器学习中的回归任务，并提供简单示例。"
order: 5
plots: ["plots/3954-0.json"]
sourceHash: "cfc55faefaabffef6891081930e409af7a5f096d71f65f5b673cc6b380642c80"
sourceCorrections: []
---

让我们把话题从分类问题（我们给数据分配如‘猫’或‘狗’这样的标签）转向另一种常见的机器学习 (machine learning)问题：**回归**。

分类预测的是事物属于*哪个类别*，而回归预测的是*连续的数值*。可以把它想象成预测“多少”或“有多少”，而不是“什么种类”。回归模型的输出不是固定的标签，而是一个可以在某个范围内变化的数字。

以下是回归常用于的一些场景：

### 预测房价

假设你想估算一套房产的市场价值。这是一个经典的回归问题。你会向模型提供关于房产的信息，这些信息被称为*特征*。这些特征可能包含：

- 面积（例如，平方英尺或平方米）
- 卧室数量
- 浴室数量
- 房屋年龄
- 位置（可能以数字形式编码）

模型从包含房屋特征及其实际售价的历史数据中学习模式。基于这些模式，它预测出新房产的具体价格，例如345,000美元或280,000欧元。这个预测价格是一个连续的数值输出。

### 预测气温

预测明天的气温是另一个例子。特征可能包括今天的气温、湿度水平、风速、大气压、一年中的时间以及历史天气模式。回归模型处理这些信息，输出一个数值预测，例如摄氏22.5度或华氏72度。

### 估算客户支出

一家公司可能希望预测某个客户下个月可能花费多少。特征可能涉及客户过去的购买记录、网站浏览行为、人口统计信息以及参与营销活动的情况。模型的输出会是一个预测的支出金额，例如85.50美元。

### 核心理念：寻找关系

在所有这些例子中，回归模型都试图理解并量化 (quantization)输入特征与连续输出值之间的关系。它基本上是学习一个将输入映射到输出的数学函数。



![回归示例：值预测](plots/3954-0.json)



> 一张图表，显示回归任务中预测值与实际值的对比。每个点代表一个预测。精确落在虚线上的点表示完美的预测。点到线的距离表示预测误差。

### 这与分类有何不同？

主要差异在于输出的性质。分类输出的是离散标签（例如，‘垃圾邮件’、‘非垃圾邮件’、‘蓝色’、‘绿色’、‘红色’）。回归输出的是连续值（例如，10.5、350000、-5.2）。这种差异意味着我们需要完全不同的方式来衡量回归模型的表现，与分类模型相比。我们不能简单地用同样的方式计算有多少预测是“正确的”。相反，我们需要衡量我们的数值预测与实际值有多*接近*。我们将在第三章中查看为此目的而设计的具体评估指标。

在构建和评估任何机器学习 (machine learning)模型之前，明白你的问题是需要预测类别（分类）还是预测数量（回归），这是最基本的第一步。

## 参考资料

- [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-2965-0](https://doi.org/10.1007/978-1-0716-2965-0)
  这本基础教材为统计学习提供了易于理解的介绍，详细涵盖了回归、分类和模型评估。
- [CS229: Machine Learning](https://cs229.stanford.edu/) — Jehangir Amjad (2025)
  Publisher: Stanford University
  斯坦福大学广受认可的机器学习课程，提供了关于监督学习的全面材料，包括回归问题及其应用的介绍。
- [1. Supervised learning (scikit-learn user guide)](https://scikit-learn.org/stable/supervised_learning.html) — scikit-learn developers (2024)
  scikit-learn库的官方文档，提供了监督学习算法（包括各种回归模型）的实用解释和示例。
