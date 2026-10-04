---
course: "introduction-to-machine-learning"
chapter: "what-is-machine-learning"
lesson: "types-of-machine-learning-systems"
sourceId: 1457
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-1-what-is-machine-learning/types-of-machine-learning-systems"
title: "机器学习系统种类"
description: "主要类别介绍：监督学习、无监督学习和强化学习。"
order: 5
plots: []
sourceHash: "a703d5013c8567ddfaadc998a64f1e442a74effe1b292e061b0f6c5e9d99ac01"
sourceCorrections: []
---

你已经了解到，机器学习 (machine learning)让系统能够从数据中学习规律，而无需为每一个规则进行明确编程。但这些系统究竟是如何学习的呢？实际上，学习方式不止一种。机器学习算法通常根据它们执行的学习任务种类和所学习数据的特点来分类。可以把这些类别视为我们学习机器的不同教学方法。

你会遇到的三种最常见类别是：

1. **监督学习 (supervised learning)**
2. **无监督学习 (unsupervised learning)**
3. **强化学习 (reinforcement learning)**

接下来我们逐一看看。

## 监督学习 (supervised learning)：用标签进行学习

想象一下你在教一个孩子识别不同的水果。你给他看一个苹果，说“苹果”，再给他看一个香蕉，说“香蕉”，以此类推。你提供了例子（水果图片）和正确答案（水果名称）。这就是监督学习的核心。

在监督学习中，机器学习 (machine learning)算法在数据集上进行训练，每个数据点都包含输入特征*和*对应的正确输出或“标签”。目标是让算法学习一个映射函数，以便能够为新的、未见过的输入特征预测输出标签。之所以称之为“监督”，是因为正确标签的存在指导着学习过程，很像老师监督学生一样。

监督学习问题主要有两种类型：

- **分类：** 目标是预测一个离散的类别或类标签。例如“这封邮件是垃圾邮件还是非垃圾邮件？”，“这个肿瘤是恶性还是良性？”，或者“这张图片里是什么类型的动物（猫、狗、鸟）？”。输出是一个特定类别。
- **回归：** 目标是预测一个连续的数值。例如“明天的气温会是多少？”，“这栋房子会卖多少钱？”，或者“下周会有多少顾客光顾商店？”。输出是一个数值。

> 监督学习的基本流程：算法从带标签数据中学习，以创建一个能够预测新的、未带标签数据标签的模型。

如今，多数机器学习的实际运用都采用监督学习，因为拥有带标签的数据通常能为特定任务带来更准确的预测。第3章和第4章将进一步介绍回归和分类方法。

## 无监督学习 (supervised learning) (unsupervised learning)：在无标签数据中发现结构

现在，想象一下你给同一个孩子一大盒混杂的乐高积木，没有任何说明或标签。孩子可能会开始根据颜色、形状或大小对它们进行分类，在积木中发现自然的 分组。这与无监督学习类似。

在无监督学习中，算法获得输入数据，*没有*任何对应的输出标签。目标是让算法自行分析数据，找出有意义的结构、模式或关系。之所以称之为“无监督”，是因为没有老师或正确答案来指导这个过程。

无监督学习任务的常见类型包括：

- **聚类：** 根据数据点的特征，将相似的数据点归为一组。例如，将具有相似购买习惯的客户进行分组，以便进行精准营销；或将关于同一主题的新闻文章进行分组。
- **降维：** 通过减少特征（维度）的数量来简化数据，同时保留重要信息。这对于数据可视化或提升其他机器学习 (machine learning)算法的性能很有用。
- **关联规则学习：** 找出描述大型数据集中项目之间关系的规则。一个经典例子是发现购买尿布的顾客也经常购买啤酒。

> 无监督学习的基本流程：算法分析未带标签的数据，以找出隐藏的模式或分组。

无监督学习常用于数据分析，并且可以发现意想不到的洞察。第5章将介绍一种流行的聚类算法。

## 强化学习 (reinforcement learning)：通过试错学习

想想你是如何训练狗的：你告诉它“坐下”，如果它坐下了，你就给它零食（奖励）。如果它没坐下，就没有零食（或者可能会有一个温和的纠正，一种惩罚）。随着时间的推移，狗会学习哪些行为能带来奖励。强化学习（RL）的工作原理类似。

在强化学习中，一个“智能体”（学习算法）与一个“环境”进行交互。智能体观察环境的当前状态并选择一个动作。基于这个动作，环境会转换到一个新状态，并以奖励（积极）或惩罚（消极）的形式向智能体提供反馈。智能体的目标是学习一个被称为“策略”的方法，使它随时间最大化其累积奖励。

强化学习的主要组成部分：

- **智能体：** 学习者或决策者。
- **环境：** 智能体与之交互的环境。
- **状态：** 环境的当前情况或配置。
- **动作：** 智能体在给定状态下做出的选择。
- **奖励/惩罚：** 来自环境的反馈，指示动作的后果。

> 强化学习中的交互循环：智能体行动，环境以新状态和反馈进行响应，智能体学习优化其动作。

强化学习特别适合于需要顺序做出决策，并且动作影响可能不会立即显现的问题。例子包括：

- 教机器人行走或执行任务。
- 开发游戏AI（如AlphaGo）。
- 优化交通灯控制系统。
- 随时间个性化推荐系统。

尽管非常强大，强化学习的实施通常比监督学习 (supervised learning)或无监督学习 (unsupervised learning)更复杂，并且通常在更高级的课程中讲授。

## 其他变体

尽管这三者是主要类别，但你偶尔可能还会听到其他一些：

- **半监督学习 (supervised learning) (semi-supervised learning)：** 结合使用少量带标签数据和大量无标签数据。当获取标签成本高或耗时时，这种方法很有用。
- **自监督学习 (self-supervised learning)：** 一种无监督学习 (unsupervised learning)类型，标签从输入数据本身自动生成。例如，根据前文预测句子中的下一个词。

理解这些机器学习 (machine learning)系统的基本类型，为你思考不同的机器学习问题及其解决方法提供了框架。随着你学习本课程，你将主要看到监督学习和无监督学习的例子，它们构成了许多常见应用的根本。

## 参考资料

- [Machine Learning](https://www.cs.cmu.edu/~tom/mlbook.html) — Tom Mitchell (1997)
  Publisher: McGraw Hill
  一本经典的教科书，一般性地介绍了机器学习原理，包括监督学习和无监督学习的核心概念。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本易于理解的指南，使用流行库介绍了监督学习、无监督学习和其他学习类型的实际实现，对初学者有益。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  强化学习领域的标准教科书，详细阐述了其基本概念和算法。
- [Machine Learning (CS229) Lecture Notes](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFRoPImFUBDzMyqSOzL-zIzATfxHpRXOmmjWL2DmhHNRahlxZXOq148e2PBNaiCDaPn8nFbPZcdmFLt49PbYtd0cIIxwhVURfBmMlUoDeVJ-LpvHuYPmI5LOL4UptKUDhy78Q==) — Andrew Ng, Tengyu Ma (2023)
  来自受尊敬大学课程的讲义，提供了监督学习、无监督学习和强化学习的理论基础。
