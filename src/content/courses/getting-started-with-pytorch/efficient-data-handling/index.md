---
course: "getting-started-with-pytorch"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling"
sourceId: 560
chapter: "efficient-data-handling"
title: "高效数据处理"
order: 5
description: "在 PyTorch 中使用 `Dataset` 和 `DataLoader` 类高效管理数据集，以实现数据批处理、打乱和转换。"
hasQuiz: true
---

既然你已经了解如何使用 `torch.nn` 构建模型以及如何用 Autograd 计算梯度，下一步就是有效地为这些模型提供数据。处理大型数据集、进行必要的预处理以及在不耗尽内存的情况下分批加载数据，是深度学习工作中常见的问题。

本章将讲解 PyTorch 管理数据流程的方案：即 `torch.utils.data` 模块。你将学习如何：

*   使用 `Dataset` 类来组织数据。
*   使用预设数据集，例如 `torchvision` 中提供的。
*   使用 `torchvision.transforms` 进行数据转换和增强。
*   使用 `DataLoader` 类高效地分批加载数据、打乱数据，并可能并行加载。

学完本章后，你将能够为你的 PyTorch 项目构建高效的数据流程。
