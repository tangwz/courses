# 第 2 章：模型、提示与解析器

来源：[原章节](https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-2-models-prompts-parsers)

[返回课程目录](../README.md)

任何LangChain应用的中心在于您的代码与大型语言模型（LLM）之间的通信。本章介绍处理这种通信的三个主要部分：模型、提示与解析器。

您将首先学习如何连接不同类型的模型，特别是LLM和聊天模型，并知道何时使用它们。之后，我们将说明如何使用`PromptTemplates`为这些模型构建准确且灵活的指令。您还将看到通过在提示中提供示例来改善模型响应的方法，这种方法称为少样本提示。最后，我们将介绍输出解析器，它们对于将模型自由格式的文本响应转换为有组织且可用的格式（例如JSON）是必需的。

这些部分构成了一个标准的调用顺序，通常表示为$Prompt \rightarrow Model \rightarrow Parser$。本章以一个实用的练习结束，您将在其中组合这些部分以制作一个从文本块中获取有组织数据的程序。

## 小节

- 1. [与LLM和聊天模型交互](01-%E4%B8%8ELLM%E5%92%8C%E8%81%8A%E5%A4%A9%E6%A8%A1%E5%9E%8B%E4%BA%A4%E4%BA%92.md)
- 2. [使用 PromptTemplates 管理提示词](02-%E4%BD%BF%E7%94%A8%20PromptTemplates%20%E7%AE%A1%E7%90%86%E6%8F%90%E7%A4%BA%E8%AF%8D.md)
- 3. [实现少样本提示](03-%E5%AE%9E%E7%8E%B0%E5%B0%91%E6%A0%B7%E6%9C%AC%E6%8F%90%E7%A4%BA.md)
- 4. [使用解析器结构化输出](04-%E4%BD%BF%E7%94%A8%E8%A7%A3%E6%9E%90%E5%99%A8%E7%BB%93%E6%9E%84%E5%8C%96%E8%BE%93%E5%87%BA.md)
- 5. [动手实践：构建结构化数据提取器](05-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E7%BB%93%E6%9E%84%E5%8C%96%E6%95%B0%E6%8D%AE%E6%8F%90%E5%8F%96%E5%99%A8.md)
