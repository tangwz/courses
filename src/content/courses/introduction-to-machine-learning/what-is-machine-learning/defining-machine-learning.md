---
course: "introduction-to-machine-learning"
chapter: "what-is-machine-learning"
lesson: "defining-machine-learning"
sourceId: 1449
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-1-what-is-machine-learning/defining-machine-learning"
title: "定义机器学习"
description: "理解机器学习（ML）的基本定义和目标。"
order: 1
plots: []
sourceHash: "e91318d528976bf25bbf42c2c77aa2f029af2b7fb712c5b0f8254039f06d6a65"
sourceCorrections: []
---

本质上，机器学习 (machine learning)是关于创建能够从数据中学习的系统。程序员不需要为每种可能的情况编写明确的、按部就班的指令，而是由机器学习系统识别数据中的模式，并根据这些模式建立自己的逻辑。

设想一下，你会如何编写一个传统程序来过滤垃圾邮件。你需要创建一长串规则：如果邮件包含某些词语（如“伟哥”、“免费”、“优惠”），就将其标记 (token)为垃圾邮件；如果它来自特定发件人，就标记为垃圾邮件；如果它的格式很奇怪，也标记为垃圾邮件，等等。这份规则列表会变得非常冗长且难以维护。垃圾邮件发送者不断改变策略，因此你需要持续更新你的规则。

机器学习则采取了不同的方式。你无需自己编写规则，而是向机器学习系统提供大量已被标记为“垃圾邮件”或“非垃圾邮件”（正常邮件）的电子邮件。该系统利用特定算法分析这些示例，并*学习*区分垃圾邮件和合法邮件的模式。它可能会学习到某些词语、发件人域名或特征组合强烈表明是垃圾邮件，而无需你明确编程那些具体规则。结果便是一个*模型*，它能够分类新的、未曾见过的邮件。

人工智能和计算机游戏方面的先驱亚瑟·塞缪尔（Arthur Samuel）早在1959年就提供了一个简洁的定义：机器学习是“赋予计算机无需明确编程即可学习能力的一项研究。”

一个更正式、经常被引用的定义来自汤姆·米切尔（Tom Mitchell）（1997年）：“如果一个计算机程序在某类任务T上，通过性能指标P衡量，其在任务T上的表现随着经验E的增加而提升，那么就称该程序从经验E中学习。”

让我们来分析米切尔的定义：

- **任务 (T):** 系统应该做什么？示例包括将电子邮件分类为垃圾邮件或非垃圾邮件、根据面积和位置等特征预测房价，或识别手写数字。
- **经验 (E):** 系统从什么数据中学习？这通常是示例数据集。对于垃圾邮件过滤，经验就是标记好的电子邮件集合。对于房价预测，则是包含房屋特征和相应价格的历史销售数据。
- **性能指标 (P):** 我们如何评估系统表现得好不好？对于垃圾邮件过滤，P可能是正确分类的电子邮件百分比。对于房价预测，P可以是预测价格与实际价格之间的平均差异。

目标是，随着系统获得更多经验（更多数据），它在任务上的表现应该有所提升。

我们可以将传统编程和机器学习的基本区别可视化：

> 在传统编程中，规则（程序）处理数据以获得输出。在机器学习中，数据和预期输出被学习算法用来创建规则（模型）。

这种从数据中学习的能力使得机器学习功能非常多样。它使我们能够处理以下问题：规则过于复杂，无法手动编写；规则随时间变化；或者我们需要从海量信息中发现细微的模式。“学习”这部分很重要；系统根据所提供的数据进行调整，使其与静态的、基于规则的系统有着根本的不同。

## 参考资料

- [Some Studies in Machine Learning Using the Game of Checkers](https://doi.org/10.1147/rd.33.0210) — Arthur L. Samuel (1959)
  Journal: IBM Journals & Magazine; Volume: 3; Pages: 210-229; DOI: [10.1147/rd.33.0210](https://doi.org/10.1147/rd.33.0210)
  这篇基础性论文通过开发自学习跳棋程序展示了机器学习，文中提及的定义是对该领域开创性工作的总结。
- [Machine Learning](https://www.cs.cmu.edu/~tom/mlbook.html) — Tom Mitchell (1997)
  Publisher: McGraw Hill
  这本权威教科书提供了广为引用的机器学习正式定义，并涵盖了基本概念。
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) — Andrew Ng (2023)
  Publisher: DeepLearning.AI and Stanford Online
  这一广受欢迎且易于理解的在线专业课程为初学者提供了机器学习概念和应用的绝佳介绍。
