---
course: "getting-started-local-llms"
chapter: "running-first-local-llm"
lesson: "intro-local-llm-runners"
sourceId: 4254
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-4-running-first-local-llm/intro-local-llm-runners"
title: "本地LLM运行器介绍"
description: "简化本地LLM下载与运行软件概览。"
order: 1
plots: []
sourceHash: "866390f7cffc4ca90d846a9619d529a35839e5df8cbc3f1db95c336352c9ccec"
sourceCorrections: []
---

大型语言模型（LLM）是强大的工具，在本地运行它们具有诸多优势。模型文件通常可以在Hugging Face Hub等平台找到。现在的问题是：如何在自己的电脑上实际*运行*这些模型？

虽然可以直接使用编程库与LLM交互，但这通常涉及到复杂的配置、依赖管理以及针对不同模型格式和硬件的特定命令行指令。对于初学者来说，这可能是一个不小的障碍。

这就是**本地LLM运行器**的用处所在。可以将它们看作是专门的应用程序，旨在简化在本地下载、管理和使用LLM的整个流程。它们充当用户和底层模型执行复杂性之间的友好层。就像媒体播放器应用程序允许你播放各种视频文件，而无需理解视频编解码器的复杂细节一样，LLM运行器让你能够使用不同的模型，而无需对它们的内部操作有深刻的技术背景。

使用专用运行器应用程序有几个优点：

- **设置简便：** 运行器通常会捆绑所需组件，并为不同操作系统（Windows、macOS、Linux）提供直接的安装说明。
- **模型管理便捷：** 它们提供界面（图形界面或命令行界面），以便轻松查找、下载和整理LLM模型文件（例如我们之前提到的`.gguf`格式）。
- **用户友好型交互：** 它们提供即用型聊天界面或命令提示符，让你在下载模型后几乎可以立即开始与模型进行交互。
- **硬件抽象：** 许多运行器会尝试自动检测你的硬件（CPU、内存、GPU），并配置模型执行设置以获得合理的性能，尽管通常可以手动调整。

> 本地设置中LLM运行器作用的简化视图。你与运行器交互，运行器则处理底层引擎、模型文件和硬件使用。

本章接下来的部分，我们将着重介绍两种流行且适合初学者的运行器：

1. **Ollama：** 一款主要通过命令行操作的工具。它以其在下载和运行模型时配置需求低而闻名。
2. **LM Studio：** 一款图形桌面应用程序。它提供可视化界面，用于查找、下载和与模型聊天，并提供更方便的配置选项。

许多这类工具，包括 Ollama 和通常 LM Studio（在幕后），都依赖于高效的推理 (inference)引擎来执行实际计算。在本地LLM技术中，一个非常有影响力的引擎是\*\*`llama.cpp`\*\*。这是一个 C/C++ 库，经过优化，能有效运行LLM在标准消费者硬件（CPU和GPU）上。虽然在使用 Ollama 或 LM Studio 等运行器时，你通常不会直接与 `llama.cpp` 交互，但了解它的存在有助于理解这些工具如何实现良好的性能。运行器提供便捷的界面，而像 `llama.cpp` 这样的引擎则承担了主要工作。

现在，我们来学习安装和使用这些运行器的实际步骤，以便你首次在本地成功运行LLM。

## 参考资料

- [Ollama](https://ollama.com/) — Ollama Team (2023)
  Ollama 的官方网站和文档，这是一个用于在本地运行大型语言模型的流行命令行工具。
- [LM Studio](https://lmstudio.ai/) — LM Studio Team (2023)
  LM Studio 的官方网站和下载，这是一个用于在本地运行和管理大型语言模型的图形桌面应用程序。
- [llama.cpp: LLM inference in C/C++](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov and the llama.cpp contributors (2023)
  llama.cpp 的 GitHub 仓库，这是一个高度优化的 C/C++ 推理引擎，是许多本地 LLM 运行器和 GGUF 模型格式的基础。
