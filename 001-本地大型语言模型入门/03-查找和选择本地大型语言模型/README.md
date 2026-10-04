# 第 3 章：查找和选择本地大型语言模型

来源：[原章节](https://apxml.com/zh/courses/getting-started-local-llms/chapter-3-finding-selecting-local-llms)

[返回课程目录](../README.md)

在上一章中准备好系统后，你现在需要一个模型来运行。本章侧重于了解专为本地使用设计的大型语言模型的各种选项。你将学习在哪里可以找到模型，主要关注Hugging Face Hub等资源。

我们将检查影响你选择的重要因素，包括模型大小（通常用参数表示，例如$7B$或$13B$）、为本地推理优化的GGUF等文件格式，以及用于减少资源需求的量化方法。此外，理解模型卡和软件许可证对于恰当的选择和使用必不可少。目标是为你提供所需条件，以选择一个与你的硬件能力和目标相符的合适初始模型。

## 小节

- 1. [寻找LLM模型：Hugging Face Hub](01-%E5%AF%BB%E6%89%BELLM%E6%A8%A1%E5%9E%8B%EF%BC%9AHugging%20Face%20Hub.md)
- 2. [理解模型大小与参数](02-%E7%90%86%E8%A7%A3%E6%A8%A1%E5%9E%8B%E5%A4%A7%E5%B0%8F%E4%B8%8E%E5%8F%82%E6%95%B0.md)
- 3. [模型格式：GGUF及其他](03-%E6%A8%A1%E5%9E%8B%E6%A0%BC%E5%BC%8F%EF%BC%9AGGUF%E5%8F%8A%E5%85%B6%E4%BB%96.md)
- 4. [量化：缩小模型](04-%E9%87%8F%E5%8C%96%EF%BC%9A%E7%BC%A9%E5%B0%8F%E6%A8%A1%E5%9E%8B.md)
- 5. [了解模型卡片中的信息](05-%E4%BA%86%E8%A7%A3%E6%A8%A1%E5%9E%8B%E5%8D%A1%E7%89%87%E4%B8%AD%E7%9A%84%E4%BF%A1%E6%81%AF.md)
- 6. [模型许可与使用限制](06-%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E4%B8%8E%E4%BD%BF%E7%94%A8%E9%99%90%E5%88%B6.md)
- 7. [选择你的第一个模型](07-%E9%80%89%E6%8B%A9%E4%BD%A0%E7%9A%84%E7%AC%AC%E4%B8%80%E4%B8%AA%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-local-llms/chapter-3-finding-selecting-local-llms/quiz)
