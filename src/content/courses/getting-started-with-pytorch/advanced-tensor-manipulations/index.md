---
course: "getting-started-with-pytorch"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-2-advanced-tensor-manipulations"
sourceId: 550
chapter: "advanced-tensor-manipulations"
title: "高级张量操作"
order: 2
description: "掌握 PyTorch 张量操作：索引、切片、重塑、广播，以及在 CPU 和 GPU 之间移动张量。"
hasQuiz: true
---

在前面介绍的基础张量操作之上，本章将讲解更高级的张量操作方法。对张量结构、数据类型和设备放置进行精细控制，对于有效准备数据和实现复杂的深度学习模型来说非常重要。

在本章中，你将学会：

*   使用索引和切片方法，精确选择和修改张量元素。
*   使用 `view()`、`reshape()` 和 `permute()` 重构张量，而不改变其数据。
*   使用 `cat()`、`stack()`、`split()` 和 `chunk()` 组合多个张量或将单个张量分割成不同部分。
*   使用 PyTorch 的广播机制，对形状兼容但不同的张量之间执行操作。
*   处理不同的数值数据类型（例如 `float`、`int`），并相应地转换张量类型。
*   使用 `.to(device)` 在 CPU 内存和 GPU 加速器之间传输张量。

掌握这些操作对于处理深度学习工作流中遇到的多样化数据格式和计算要求是必要的。
