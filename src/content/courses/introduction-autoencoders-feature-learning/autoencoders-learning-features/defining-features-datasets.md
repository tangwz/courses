---
course: "introduction-autoencoders-feature-learning"
chapter: "autoencoders-learning-features"
lesson: "defining-features-datasets"
sourceId: 6421
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-4-autoencoders-learning-features/defining-features-datasets"
title: "数据集中特征的界定"
description: "了解数据集中特征的含义以及它们在机器学习中的重要性。"
order: 1
plots: []
sourceHash: "6f93cc5d8ee5baaaadea7c73a90c844519da3e5e4e55b6254eaa51764459d5c2"
sourceCorrections: []
---

在讨论自编码器如何从数据中学习新表示之前，有必要先弄清数据集中“特征”的含义。可以把特征看作你正在处理的数据的单个、可衡量的属性或特点。它们是描述每条信息的基本构成，机器学习 (machine learning)模型利用这些特征进行预测或找出规律。

### 表格数据中的特征：列即特点

理解特征最直接的方式或许是查看结构化数据，这类数据常以表格形式组织，比如电子表格。在这样的表格中：

- 每**行**通常表示一个单独的观测值或项目（例如：一位顾客、一个产品、一个病人、一栋房子）。
- 每**列**表示这些项目的一个特定**特征**或属性。

例如，如果你有一个关于顾客的数据集，特征可能包含：

- `年龄` (例如：34岁)
- `年收入` (例如：50,000美元)
- `居住城市` (例如：“纽约”)
- `是否已订阅` (例如：真/假)

这些列中的每一列都提供了关于每位顾客的一条独特信息（一个特征）。机器学习 (machine learning)模型可以利用这些特征，例如预测新顾客是否可能订阅某项服务。

请看以下一个数据集的简单表示：

> 在这个房屋数据集示例中，“建筑面积”、“卧室数量”和“到市中心的距离”是描述每栋房屋的特征。“价格”可能是我们希望预测的目标。

### 其他类型数据中的特征

特征不限于表格中整齐的列。

- **图像数据**：对于图像来说，最基本的特征是像素值。一张28x28像素的灰度图像（例如MNIST手写数字数据集中的图像）有$28 \times 28 = 784$个特征，其中每个特征是特定像素的强度值。对于彩色图像，每个像素通常有三个值（红、绿、蓝），因此一张100x100的彩色图像将有$100 \times 100 \times 3 = 30,000$个原始特征。
- **文本数据**：在文本中，特征可以是单个词、词的频率（“词袋”模型）、词序列（N-gram），或更复杂的表示，比如词嵌入 (embedding)（我们后面会看到它们可以被学习到）。例如，在情感分析任务中，像“高兴”或“失望”这类词的出现或缺失可能是重要的特性。
- **音频数据**：对于声音，特征可以从波形中提取，例如梅尔频率倒谱系数（MFCCs），它们表示声音的短期功率谱。

共同点在于，特征是从原始数据中提取的数值或分类输入，机器学习 (machine learning)算法利用它们来执行任务。

### 理解特征的重要性

特征的选择和质量是任何机器学习 (machine learning)项目成功的前提。

1. **模型的输入**：特征是模型“看到”的直接输入。如果特征不包含当前任务的相关信息，即使最复杂的模型也会表现不佳。
2. **维度**：特征数量决定了数据的维度。特征过多（特别是那些不相关的）有时会使学习变得更困难（这常被称为“维度灾难”）。这是自编码器可以提供帮助的一个方面，我们将在稍后的维度降低部分讨论。
3. **可解释性**：选得好或学得好的特征有时能使模型的决策更易于理解。

简而言之，特征是机器学习模型看待数据的视角。

在本章后续内容中，我们将看到自编码器不仅使用预设特征；它们的优势在于能够从初始的原始输入特征中*学习*新的、通常更紧凑和更有信息量的特征。自编码器中的瓶颈层（我们之前讨论过）正是这些学习到的特征所在之处。对“特征”基本含义的清晰认识为理解自编码器如何转换它们奠定了基底。

## 参考资料

- [Pattern Recognition and Machine Learning](https://www.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/b139367](https://doi.org/10.1007/b139367)
  一本经典教科书，严谨地阐述了数据表示、特征及其在机器学习模型中的作用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本权威书籍全面概述了深度学习，包括特征的定义以及从数据中学习新表示的动因。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGOx6s5n4tdSMDFtCeBdVraIqEirStk8-8oykS2BEZKfrP2RrZZluFV1jf0A7aTVNJn8Ny-7sbwd0uJaKNraHx-7DhhkYf06PHcyuFSyATYHDksaqXY8GrtWU4eijv0U0eTdYz3i5IYNkEgWwnLnRPArz5WfYGqw0FNCU_Y5ncU_B1trH9zafP9QjraEbEA2dW-piH0cpmA9T1_oSyo) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  这本实用指南通过具体示例，清晰地解释了特征、各种特征类型及其在机器学习工作流中的重要性。
- [CS229 Lecture Notes: Machine Learning](https://www.studylib.net/doc/25732168/cs229-lecture-notes--machine-learning) — Andrew Ng, Tengyu Ma (2023)
  Journal: Course Notes; Publisher: Stanford University
  这份来自领先机器学习课程的讲义涵盖了基本概念，包括数据表示和特征的作用，内容易于理解。
