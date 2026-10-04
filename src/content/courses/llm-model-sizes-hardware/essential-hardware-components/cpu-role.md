---
course: "llm-model-sizes-hardware"
chapter: "essential-hardware-components"
lesson: "cpu-role"
sourceId: 4204
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-2-essential-hardware-components/cpu-role"
title: "中央处理器 (CPU)"
description: "说明 CPU 在通用计算中的功能及其在大型语言模型中扮演的角色。"
order: 1
plots: []
sourceHash: "31645903df3ee2670932ae5c6c160c8ac7363606af8b1a41a81f32aad2b2de55"
sourceCorrections: []
---

可以把中央处理器（CPU）看作是电脑的主要处理器或“大脑”。它是几乎所有计算设备（从笔记本电脑、台式机到服务器和智能手机）中都必不可少的一个硬件部件。CPU 的主要作用是执行程序指令、进行计算并管理系统的整体运行。

在一部普通电脑中，CPU 处理许多不同种类的任务。当你上网、编写文档或运行大多数标准软件时，CPU 承担着大部分工作。它的设计用途很广，能够非常快速地一个接一个处理复杂的顺序指令。你可以把它想象成乐队的指挥，指挥系统中的不同部分，管理资源，并确保任务按正确顺序完成。大多数 CPU 都有少数几个性能强大的核心，这些核心经过优化，能有效执行这些不同的指令。

那么，在处理大型语言模型时，CPU 是如何发挥作用的呢？虽然我们很快就会了解到图形处理器（GPU）承担了 LLM 最耗时的计算，但 CPU 仍有不可或缺的辅助功能。

运行 LLM 时，CPU 通常会执行以下操作：

1. **系统管理：** 它管理操作系统和 LLM 应用程序运行的整体软件环境。
2. **任务安排：** 它常统筹这个过程，启动 LLM 的加载，管理输入和输出数据流（如你提供的文本提示和模型生成的文本），并可能配合包括 GPU 在内不同硬件部件之间的操作。
3. **数据处理：** 根据具体设置，CPU 可能会在数据发送到 GPU 之前或结果返回之后，处理一些初步的数据准备或后处理任务。
4. **运行应用程序代码：** 你用来与 LLM 交互的应用程序（如聊天机器人界面或代码补全工具）在 CPU 上运行。

然而，使 LLM 运作的计算任务，例如大型矩阵乘法和同时执行数十亿次计算，并不太适合 CPU 的架构。CPU 擅长处理复杂的单一任务，但 LLM 需要许多相对简单的计算同时进行。这种同时进行的运算正是 GPU 的优势所在。

因此，虽然 CPU 对于电脑正常运行和管理 LLM 的整体过程来说必不可少，但它通常不是在推理 (inference)（生成文本的过程）期间为大型模型执行主要计算的部件。可以把它看作是统筹工作的管理者，而专业的“工作人员”（GPU，我们接下来会讨论）则处理 LLM 本身所需的最繁重、重复的劳动。明白这种差异对于理解为何针对不同 AI 任务推荐特定硬件很有价值。

## 参考资料

- [Computer Organization and Design: The Hardware/Software Interface](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHrNIJFrOdy-D3yRvOWfumvWoelIDH5eHNT4sKVVG3Gvjp1poavkMPKqAYLQqfRs0H8IEFBmMavaPSSegRn0F_p5eBphcv41u1BnHYCgwiOhNFd5cqWo-ViijWRcUKx-gCBFHbKYRaji1VwoaFCWHmn-R62RPzRMAffG0XVTxssZdpUQiMufra28kgjLTNfEx2esVgmNECaZJEevmcS) — David A. Patterson and John L. Hennessy (2020)
  Publisher: Morgan Kaufmann
  解释CPU架构、指令执行及其与其他组件的交互，构成了对计算机工作原理的核心理解。
- [Computer Architecture: A Quantitative Approach](https://www.biblio.com/book/computer-architecture-quantitative-approach-hennessy-john/d/1487661642) — John L. Hennessy, David A. Patterson (2017)
  Publisher: Morgan Kaufmann
  深入分析现代计算机架构，包括通用CPU和GPU等专用加速器在并行计算方面的详细比较。
