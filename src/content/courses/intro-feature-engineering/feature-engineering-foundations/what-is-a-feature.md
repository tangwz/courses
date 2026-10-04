---
course: "intro-feature-engineering"
chapter: "feature-engineering-foundations"
lesson: "what-is-a-feature"
sourceId: 1301
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-1-feature-engineering-foundations/what-is-a-feature"
title: "什么是特征？"
description: "定义机器学习数据集中的特征及其特点。"
order: 2
plots: []
sourceHash: "9fdf46420ea07498b1b292ed71e3cf111170cb4ed6dd5df8417a64425058254d"
sourceCorrections: []
---

正如本章引言所述，特征工程在机器学习 (machine learning)工作流程中占据重要位置。但在此背景下，特征究竟*是*什么呢？

核心而言，**特征**是被观测现象的一个可衡量的个体属性或特点。可以将其视为机器学习模型用于做出预测或判断的输入变量。在机器学习常用的结构化数据集（如表格或电子表格）中，特征通常对应于列。

考虑一个简单例子：预测房价。您收集的原始数据可能包含每套房屋的各种信息。部分原始数据可能可以直接用作特征，而另一些特征可能需要创建或转换。

- **直接可用特征：** 房屋的面积、卧室数量、浴室数量。这些都是可衡量的属性，很可能可以直接输入到模型中。
- **需要转换的特征：** 地址可能是原始文本。这对于大多数模型来说无法直接使用。您可能需要从中构建特征，例如邮政编码、到最近学校的距离，或者一个代表社区的分类特征。
- **需要创建的特征：** 您可能拥有房屋的建造年份。虽然可用，但房屋的*年龄*（当前年份 - 建造年份）可能是一个信息量更大的特征。您从`year_built`数据中创建这个`age`特征。同样，拥有地块大小和房屋面积，可能让您能够创建`yard_size`特征（地块大小 - 房屋面积）或`house_ratio`特征（房屋面积 / 地块大小）。

主要观点是，特征是为学习算法量身定制的原始数据的**信息丰富的表示**。原始数据通常杂乱，包含不相关信息，或者格式不适合算法（例如，文本地址、原始时间戳）。特征是经过提炼的、从这些原始数据中派生出来的数值或分类输入。

下面是一个说明其区别的小表格：

| 原始数据点 | 可派生的潜在特征 | 特征类型 |
| --- | --- | --- |
| `交易时间戳` | `一天中的小时`, `一周中的天`, `是否周末` (二元) | 数值型, 数值型, 分类型 |
| `客户地址` | `邮政编码`, `到商店的距离` (英里) | 分类型, 数值型 |
| `产品描述` (文本) | `关键词数量`, `情感分数` | 数值型, 数值型 |
| `销售价格`, `成本价格` | `利润率` ($(销售价格 - 成本价格) / 销售价格$) | 数值型 |
| `卧室数量` | `卧室数量` | 数值型 |
| `颜色` (例如, "红色", "蓝色") | `颜色编码` (例如, 使用编码的0, 1) | 分类型/数值型 |

> 原始数据点转换为潜在机器学习特征的示例。

特征工程的目标，也是我们将在本课程中会详细介绍的，是从可用数据中构建最有效的特征集。这包括选择相关信息、将其转换为合适的格式（如将分类数据转换为数字）、处理缺失值，有时还会创建全新的特征，以比单独使用原始数据更有效地捕捉潜在模式。这些特征的质量通常比模型算法本身的选择对模型性能更为重要。理解什么是好的特征，是此过程的第一步。

## 参考资料

- [Feature Engineering for Machine Learning](https://learning.oreilly.com/library/view/feature-engineering-for-machine/9781491953242/) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  理解和创建有效机器学习模型特征的基础指南，涵盖核心定义和实用方法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供实用示例和背景，说明特征在完整机器学习工作流中如何使用和准备。
- [CS229 Lecture Notes](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFbyhut-fJzU2ksYgpisvonYswOhQ5UQFwxmr_QyGNSePGhjPjSTc-Ryyolw8xzKmR3Y9Mr0cJbJDiOY_OAdcWBTEAMm1eA6Oah-Rw4lIUiUA5WIUgi8MUKGbzFWuee2l3ZKHo=) — Andrew Ng, Tengyu Ma (2023)
  斯坦福大学的基础机器学习课程笔记，涵盖学习问题的设置以及数据和特征的作用。
- [An Introduction to Statistical Learning: With Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38914-1](https://doi.org/10.1007/978-3-031-38914-1)
  统计学习的综合介绍，阐述数据、变量和特征如何在各种建模技术中使用。
