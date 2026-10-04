---
course: "pytorch-for-tensorflow-developers"
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users"
sourceId: 1045
chapter: "pytorch-data-loading-for-tf-users"
title: "数据加载与预处理：从 tf.data 到 torch.utils.data"
order: 3
description: "将您的 TensorFlow `tf.data` 技能迁移到 PyTorch 的 `torch.utils.data`，以实现高效的数据处理和预处理。"
hasQuiz: true
---

有效的数据管理是任何机器学习项目的重要组成部分。作为一名 TensorFlow 开发者，您可能对使用 `tf.data` 构建输入管道非常熟悉。本章侧重介绍 PyTorch 处理数据加载和预处理的方式，帮助您调整现有技能。

我们将学习 PyTorch 的 `torch.utils.data` 模块，它为此目的提供了主要工具。您将学习如何使用 `Dataset` 类定义自定义数据源，并使用 `DataLoader` 有效地批量处理您的数据。我们还将介绍 `torchvision.transforms`，用于应用常见的数据预处理和增强技术。在本章结束时，您将能够为您的 PyTorch 模型构建灵活且高性能的数据管道，并结合您在 TensorFlow 的经验进行比较。
