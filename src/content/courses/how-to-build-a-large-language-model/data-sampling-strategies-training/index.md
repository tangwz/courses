---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-9-data-sampling-strategies-training"
sourceId: 1087
chapter: "data-sampling-strategies-training"
title: "训练中的数据取样策略"
order: 9
description: "检视训练期间数据取样的不同策略，包括混合数据来源和课程学习等方法。"
hasQuiz: false
---

在收集和处理了大量文本数据集之后，下一步是决定如何在漫长的训练过程中有效地将这些数据提供给模型。简单地将所有数据打乱混合可能并非最佳。数据呈现的构成和顺序可以显著影响模型的学习过程及其最终表现。

本章将介绍训练期间数据取样的方法。你将了解到：

*   混合不同来源数据的重要性以及如何构成这些混合数据。
*   为不同数据源分配权重，以强调或弱化某些文本类型的方法。
*   使用基于温度的取样方法来控制抽取数据批次的分布。
*   课程学习的思想，即数据可能以特定顺序呈现，例如从简单到更复杂的例子。
*   数据配速和退火等策略，涉及在整个训练期间动态改变数据混合。

理解这些取样策略对于指导训练过程并塑造最终大语言模型的能力非常重要。
