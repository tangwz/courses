---
course: "advanced-pytorch"
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability"
sourceId: 639
chapter: "custom-extensions-interoperability"
title: "自定义扩展与互操作性"
order: 6
description: "为 PyTorch 构建自定义 C++ 和 CUDA 扩展，与 NumPy 交互，并扩展 `nn.Module` 和 `optim.Optimizer` 等核心组件。"
hasQuiz: false
---

PyTorch 提供了丰富的工具集，但特定应用场景常常需要标准库中没有的操作或性能优化。
本章将介绍如何将 PyTorch 的功能扩展到其标准 Python 应用编程接口之外。

我们将构建自定义运算符，使用 C++ 和 CUDA 来应对对计算效率要求高或需要专用算法的场合。您将获得直接操作 PyTorch C++ 后端 (ATen) 的经验，学习管理 PyTorch 张量和 NumPy 数组之间的数据传输，并通过扩展 `torch.nn.Module` 和 `torch.optim.Optimizer` 来组织自定义网络组件和优化方法。此外，还将介绍使用外部函数接口 (FFI) 与现有 C 库进行交互的技术。学完本章，您将能够把自定义代码和外部库集成到您的 PyTorch 工作流程中。
