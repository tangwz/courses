---
course: "introduction-to-deep-learning"
chapter: "neural-network-foundations"
lesson: "ml-to-dl-transition"
sourceId: 5018
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-1-neural-network-foundations/ml-to-dl-transition"
title: "从机器学习到深度学习"
description: "了解传统机器学习与深度学习方法之间的区别与联系。"
order: 1
plots: []
sourceHash: "782322188598bcd4cec4289a10a6eba2f6c453e53bd3a25cb25dce70baf3ddaa"
sourceCorrections: []
---

您可能已经有一些机器学习 (machine learning)的经验。您了解其典型流程：收集数据，精心设计输入特征，选择合适的算法（如支持向量 (vector)机、决策树或逻辑回归），训练模型，并评估其表现。在许多传统机器学习应用中，大量精力投入到*特征工程*中。这需要运用领域知识，手动设计并从原始数据中提取有信息量的特征，以帮助模型做出准确预测。例如，在构建垃圾邮件检测器时，您可以设计诸如特定词语的频率、全大写文本的存在或感叹号数量等特征。

这些传统模型的成功通常很大程度上取决于这些手动设计的特征的质量。创建好的特征可能耗时，需要特定问题领域的大量专业知识，并且可能无法捕捉数据中所有复杂、细微的模式，特别是对于涉及感知的任务，如图像识别或自然语言理解。

深度学习 (deep learning)是一种机器学习分支，它提供了一种独特的学习方法。其核心是受大脑结构和功能启发的算法，称为人工神经网络 (neural network)（ANN）。深度学习模型，特别是*深层*神经网络的显著特点是，它们能够通过分层过程*直接从数据中*学习相关特征。

深度学习模型不是依靠人类来定义数据的最佳表示，而是自动学习多层次的表示，从低级特征开始，逐步构建更复杂、抽象的表示。想象一个图像分类任务。一个深度学习模型可能首先在其初始层中学习检测简单的边缘和纹理。随后的层可能将这些边缘组合起来，以识别角点和基本形状。更高层可以进一步整合这些形状，以识别物体部件（如眼睛或轮子），最终在最后层中识别完整的物体（如人脸或汽车）。这个过程常被称为*表示学习*。

> 传统机器学习和深度学习典型流程的比较，强调深度学习中自动化特征学习的特点。

这种自动学习特征的能力使得深度学习在处理涉及图像、音频信号和文本等非结构化数据的复杂问题时尤为有效，在这些情况下，手动设计有效特征非常困难。当有大量带标签数据可用于训练这些多层网络时，它也能表现出色。

然而，重要的是要明白深度学习并非传统机器学习的通用替代品。传统方法通常表现良好，需要更少的数据，计算成本较低，并且更具可解释性，特别是对于结构化或表格数据。深度学习代表了更广泛的机器学习领域中一组强大的工具，在特定类型的难题任务上提供了当前最好的性能。

本章主要介绍这些深度学习模型的基本组成部分。我们将从其生物学启发开始，并定义最简单的处理单元——人工神经元，然后再研究这些单元如何连接形成网络。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖深度学习的理论基础、算法和应用，包括其与传统机器学习的区别以及表征学习的概念。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael Nielsen (2015)
  Publisher: Determination Press
  一本易于理解的在线书籍，从基本原理介绍神经网络和深度学习的核心概念，阐释自动化特征学习的动机。
- [CS230: Deep Learning (Fall 2018 Lecture Notes)](http://cs230.stanford.edu/fall2018/) — Andrew Ng and Kian Katanforoosh (2018)
  Publisher: Stanford University
  斯坦福大学一门基础课程的讲义，介绍深度学习概念，包括传统机器学习与深度神经网络中自动化特征学习作用的区别。
- [Representation Learning: A Review and New Perspectives](https://ieeexplore.ieee.org/document/6472238) — Yoshua Bengio, Aaron Courville, and Pascal Vincent (2013)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 35; Pages: 1798-1828; DOI: [10.1109/TPAMI.2013.29](https://doi.org/10.1109/TPAMI.2013.29)
  一篇开创性综述文章，正式介绍并讨论表征学习，强调其在深度学习中的重要性以及如何解决传统机器学习中手动特征工程的局限性。
