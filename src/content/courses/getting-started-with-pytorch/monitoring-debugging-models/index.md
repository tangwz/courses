---
course: "getting-started-with-pytorch"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-8-monitoring-debugging-models"
sourceId: 565
chapter: "monitoring-debugging-models"
title: "监控与调试模型"
order: 8
description: "学习调试 PyTorch 模型和使用 TensorBoard 等工具可视化训练进度的方法。"
hasQuiz: true
---

构建和训练模型是重要的进展，但通常，初次尝试并不能达到预期效果或运行不顺畅。模型可能收敛缓慢、生成无意义的输出，或遇到运行时错误。本章讨论监控训练过程和调试 PyTorch 应用程序的实际需求。

我们将介绍诊断和解决常见问题的系统方法，包含张量形状不匹配以及与 CPU/GPU 设备分配相关的错误。你将学习如何检查梯度，以发现训练稳定性问题，比如梯度消失或梯度爆炸。此外，本章介绍监控训练动态的方法，特别是使用 TensorBoard 可视化例如损失和准确率随时间的变化。我们还将讨论集成基本日志记录，以及使用 Python 调试器 (`pdb`) 进行逐步代码检查。到本章结束时，你将掌握一个工具集，能够有效地排查问题并观察你的 PyTorch 模型。
