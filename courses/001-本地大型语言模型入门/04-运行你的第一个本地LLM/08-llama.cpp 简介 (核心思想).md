# llama.cpp 简介 (核心思想)

来源：[原文](https://apxml.com/zh/courses/getting-started-local-llms/chapter-4-running-first-local-llm/intro-llama-cpp)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在自己的电脑上运行强大的大型语言模型正变得相当可行，这得益于Ollama和LM Studio等工具。您可能想知道，在不需要大型服务器的情况下，幕后有什么让这成为可能。通常，一个名为`llama.cpp`的重要软件会参与其中。

可以将`llama.cpp`看作一个专门为运行特定类型大型语言模型而打造的高效引擎，而不是像LM Studio那样用户友好的应用程序。它是一个主要用C++编程语言编写的库。

### 为什么要用C++？性能很重要

为什么要用C++？主要原因是性能。C++代码编译后运行速度非常快，可以直接与电脑硬件交互。这很重要，因为大型语言模型生成文本需要大量的计算。`llama.cpp`经过优化，可以尽可能快地执行这些计算，尤其是在每台电脑都有的标准中央处理器（CPU）上。虽然图形处理器（GPU）可以进一步加速大型语言模型（如第2章所述），但`llama.cpp`使得仅用CPU和RAM运行中等大小的模型成为可能，从而降低了使用门槛。

### 幕后引擎

许多易于使用的工具，包括可能Ollama或LM Studio使用的后端，都在内部使用`llama.cpp`。想象一下您的大型语言模型运行程序（如LM Studio）是一辆汽车。您操作方向盘、踏板和仪表盘。`llama.cpp`就像引擎盖下的发动机——您通常不直接与其交互，但它正在进行处理模型和根据您的提示生成文本的重要工作。

> 一个简化视图，展示了用户界面如何通常依赖于像`llama.cpp`这样的底层引擎来与模型文件交互。

### 与GGUF模型的关联

还记得我们在第3章讨论的GGUF模型格式吗？`llama.cpp`与它紧密相关。GGUF格式是与`llama.cpp`一起开发的，专门设计用于被该引擎高效加载和运行。GGUF文件以一种`llama.cpp`可以轻松地在CPU和GPU上使用的方式打包模型权重 (weight)（通常经过量化 (quantization)以节省空间和内存）。这种密切关系是GGUF成为本地共享和运行模型的流行标准的原因。

### llama.cpp的优点

因此，尽管您可能不会直接输入`llama.cpp`命令（除非您选择之后了解更高级的用法），但了解它的存在很重要，因为它为本地大型语言模型社区提供了多项好处：

- **CPU效率高：** 使得在标准硬件上运行有能力的大型语言模型成为可能。
- **跨平台：** 可在Windows、macOS和Linux上运行。
- **重要支持：** 为许多用户友好的工具提供了核心推理 (inference)能力。
- **优化：** 与GGUF等量化 (quantization)模型格式有效配合，减少了资源需求。

总的来说，`llama.cpp`是一个重要的C++库，专注于在消费级硬件上高效运行大型语言模型，尤其是在GGUF格式下。它是本章所学工具能将大型语言模型的能力直接带到您的台式机或笔记本电脑上的一个重要原因。了解它的作用有助于弄清这些模型如何在本地运行。

## 参考资料

- [GGML - Tensor library for Machine Learning](https://github.com/ggerganov/ggml) — Georgi Gerganov and the GGML contributors (2024)
  用于在 CPU 上高效执行张量运算的基础 C 库，是 `llama.cpp` 和 GGUF 格式开发的核心。

---

[上一节](07-%E5%9C%A8LM%20Studio%E4%B8%AD%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%BF%9B%E8%A1%8C%E8%81%8A%E5%A4%A9.md) · [下一节](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%BF%90%E8%A1%8C%E6%A8%A1%E5%9E%8B.md)
