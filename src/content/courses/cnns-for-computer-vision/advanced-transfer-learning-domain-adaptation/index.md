---
course: "cnns-for-computer-vision"
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-6-advanced-transfer-learning-domain-adaptation"
sourceId: 616
chapter: "advanced-transfer-learning-domain-adaptation"
title: "高级迁移学习与域适应"
order: 6
description: "学习精巧的迁移学习策略、域适应、少样本学习和自监督预训练，面向视觉任务。"
hasQuiz: false
---

从零开始训练大型计算机视觉模型需要大量数据和计算资源。迁移学习提供了一种实用方法，通过重复使用预训练模型的知识。尽管基本的迁移学习技术（如基本微调）很有效，但在许多实际使用场景中，当模型需要适应与原始训练数据显著不同的新任务或数据集时，需要更精巧的方法。

本章考察模型适应的高级策略。我们将考察在微调和特征提取之间进行选择的优化方法，包括层冻结模式。您将学习域适应技术，处理目标数据分布($P_{target}(X, Y)$)与源数据分布($P_{source}(X, Y)$)不同的情况，以及相关的域泛化方法，以提升在完全未见域上的性能。此外，我们涵盖了少样本学习方法，用于使用极少量标注样本构建有效模型，并介绍自监督预训练方法，这些方法学习有用的视觉表示，而无需依赖手动标注。目的是为您提供方法，以便有效应用和调整预训练模型，以适应专门任务和不同数据环境。
