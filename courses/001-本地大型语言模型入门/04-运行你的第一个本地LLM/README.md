# 第 4 章：运行你的第一个本地LLM

来源：[原章节](https://apxml.com/zh/courses/getting-started-local-llms/chapter-4-running-first-local-llm)

[返回课程目录](../README.md)

在你的系统准备就绪并了解了大型语言模型及其获取途径之后，本章将侧重于在你的机器上运行一个LLM的实际操作步骤。

你将了解到专门为了简化本地LLM的下载、管理和交互过程而设计的软件工具。我们将介绍两种常用工具的设置和基本使用方法：Ollama（一个命令行工具）和LM Studio（一个图形界面应用）。你会看到如何用这些工具下载模型文件（通常是`.gguf`格式），然后加载它来进行交互。我们还将简单提及`llama.cpp`，这是一个支持许多这类工具的核心库。

本章的目标是指导你下载并运行你的第一个LLM，从而支持在你的电脑上直接进行基本的文本生成。

## 小节

- 1. [本地LLM运行器介绍](01-%E6%9C%AC%E5%9C%B0LLM%E8%BF%90%E8%A1%8C%E5%99%A8%E4%BB%8B%E7%BB%8D.md)
- 2. [设置 Ollama](02-%E8%AE%BE%E7%BD%AE%20Ollama.md)
- 3. [用 Ollama 下载模型](03-%E7%94%A8%20Ollama%20%E4%B8%8B%E8%BD%BD%E6%A8%A1%E5%9E%8B.md)
- 4. [使用 Ollama 运行模型 (命令行)](04-%E4%BD%BF%E7%94%A8%20Ollama%20%E8%BF%90%E8%A1%8C%E6%A8%A1%E5%9E%8B%20%28%E5%91%BD%E4%BB%A4%E8%A1%8C%29.md)
- 5. [设置 LM Studio](05-%E8%AE%BE%E7%BD%AE%20LM%20Studio.md)
- 6. [在 LM Studio 中查找和下载模型](06-%E5%9C%A8%20LM%20Studio%20%E4%B8%AD%E6%9F%A5%E6%89%BE%E5%92%8C%E4%B8%8B%E8%BD%BD%E6%A8%A1%E5%9E%8B.md)
- 7. [在LM Studio中加载模型并进行聊天](07-%E5%9C%A8LM%20Studio%E4%B8%AD%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%BF%9B%E8%A1%8C%E8%81%8A%E5%A4%A9.md)
- 8. [llama.cpp 简介 (核心思想)](08-llama.cpp%20%E7%AE%80%E4%BB%8B%20%28%E6%A0%B8%E5%BF%83%E6%80%9D%E6%83%B3%29.md)
- 9. [动手实践：运行模型](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%BF%90%E8%A1%8C%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-local-llms/chapter-4-running-first-local-llm/quiz)
