# 第 8 章：构建自主代理

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-8-developing-autonomous-agents)

[返回课程目录](../README.md)

到目前为止，我们已将大型语言模型（LLMs）用于直接的、单轮任务，例如生成文本或依据给定背景信息回答问题。本章将讲解构建能够更自主运作以达成目的的系统。自主代理使用大型语言模型（LLM）作为推理引擎来确定一系列步骤，并通常会与外部工具交互以收集信息或执行操作。此过程可表示为思维、行动和观察的循环。

在本章中，您将学习使用 `agent` 模块来构建这些代理。我们将从介绍 ReAct（推理与行动）模式开始，它是一个用于组织代理行为的常用架构。接着，您将实现一个 ReAct 代理，并学习如何为其提供工具，例如搜索功能或一个简单的计算器，以扩展其功能。核心循环可简化为以下步骤：

$$
\text{Goal} \rightarrow \text{Thought} \rightarrow \text{Action} \rightarrow \text{Observation} \rightarrow \dots
$$

接下来，我们将讲解另一种架构，名为“规划-执行”代理。最后，我们将讨论允许多个代理合作以处理更复杂问题的系统构建原则。本章结束时，您将能够构建能将问题分解、使用工具寻找处理办法并代表您行事的代理。

## 小节

- 1. [大型语言模型智能体简介](01-%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%99%BA%E8%83%BD%E4%BD%93%E7%AE%80%E4%BB%8B.md)
- 2. [ReAct 思维与行动模式](02-ReAct%20%E6%80%9D%E7%BB%B4%E4%B8%8E%E8%A1%8C%E5%8A%A8%E6%A8%A1%E5%BC%8F.md)
- 3. [构建 ReAct 智能体](03-%E6%9E%84%E5%BB%BA%20ReAct%20%E6%99%BA%E8%83%BD%E4%BD%93.md)
- 4. [定义和使用工具](04-%E5%AE%9A%E4%B9%89%E5%92%8C%E4%BD%BF%E7%94%A8%E5%B7%A5%E5%85%B7.md)
- 5. [实施规划-执行型智能体](05-%E5%AE%9E%E6%96%BD%E8%A7%84%E5%88%92-%E6%89%A7%E8%A1%8C%E5%9E%8B%E6%99%BA%E8%83%BD%E4%BD%93.md)
- 6. [协调多智能体系统](06-%E5%8D%8F%E8%B0%83%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E7%B3%BB%E7%BB%9F.md)
