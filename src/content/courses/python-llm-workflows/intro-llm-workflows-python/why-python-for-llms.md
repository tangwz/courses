---
course: "python-llm-workflows"
chapter: "intro-llm-workflows-python"
lesson: "why-python-for-llms"
sourceId: 5088
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-1-intro-llm-workflows-python/why-python-for-llms"
title: "为什么选择Python进行LLM开发？"
description: "讨论使用Python开发LLM工作流程和应用的优势。"
order: 3
plots: []
sourceHash: "4074a59d31ad4a8fe1db4bf1720c2ba956e81c228121a511ae9a5bbfc751e66e"
sourceCorrections: []
---

构建功能性LLM工作流程需要连接其不同的组成部分。尽管理论上可以使用多种编程语言，Python已成为LLM开发的主要标准。Python的流行并非偶然；其适用性由多个因素共同促成。

### AI和数据科学的丰富生态系统

Python拥有一个为数据处理、科学计算、机器学习 (machine learning)以及越来越多的LLM操作专门设计的无与伦比的库和框架生态系统。这个庞大的集合大大加快了开发速度：

- **核心库：** 像NumPy（用于数值运算）和Pandas（用于数据处理）这样的工具是任何数据密集型应用的必备，包括那些为LLM准备数据的应用。
- **机器学习集成：** Scikit-learn提供多种算法和工具，可以补充LLM工作流程，例如在数据预处理或评估输出方面。
- **专用LLM框架：** 像LangChain和LlamaIndex（它们是本课程的核心）这样的库主要是为Python构建的。它们为常见的LLM任务提供高级抽象，例如提示管理、与模型交互、数据索引（如检索增强生成 (RAG)中所见）和代理创建。
- **模型中心访问：** Hugging Face的`transformers`库可方便地直接在Python中访问数千个预训练 (pre-training)模型，包括许多LLM。

这种丰富的环境意味着开发人员通常无需从头开始构建基本组件。相反，他们可以运用现有且经过充分测试的工具来组装复杂的应用程序。围绕这些库的庞大活跃社区也提供了充足的文档、教程和支持。

> 图示说明了Python在LLM工作流程中连接各个组件的作用。

### 简洁性和可读性

Python的语法因其清晰性及与纯英语的相似性而受到称赞。这使得代码相对容易编写、理解和维护，在处理LLM工作流程潜在复杂的逻辑时，这是一个重要优势。构建提示、解析响应、根据LLM输出管理条件逻辑以及将多个操作连接起来，都可能变得错综复杂。Python的可读性有助于管理这种复杂性。此外，其解释型特性促进快速原型制作和迭代，使开发人员能够快速尝试不同的提示、模型和工作流程结构。

### 强大的集成能力

LLM应用很少孤立存在。它们需要与各种外部系统、数据源和API交互。Python作为一种集成语言表现出色，常被称为“胶水代码”。它提供直接的方法来：

- 向LLM提供商API发送HTTP请求。
- 连接并与不同类型的数据库交互，包括RAG系统中使用的专用向量 (vector)数据库。
- 处理多种数据格式，如JSON、CSV、PDF、HTML等。
- 与Web框架（如Flask或FastAPI）集成，将LLM应用作为服务公开。
- 调用其他系统工具或库。

这种灵活性使构建将LLM作为更大系统中一个组件的端到端应用变得实际。

### 出色的LLM API支持

大多数主要LLM提供商（OpenAI、Anthropic、Google、Cohere等）通过提供官方软件开发工具包（SDK）来优先支持Python。这些SDK简化了与API交互的过程，处理了认证、请求格式、响应解析和错误处理等细节。使用官方SDK通常比发送原始HTTP请求更方便且不易出错，尽管Python的`requests`等库在需要时使后者方法完全可行。

### 强大的数据处理能力

由于LLM常与特定数据集结合使用（例如在RAG系统中），有效的数据处理非常必要。Python，以及Pandas等库和LlamaIndex中提供的专用数据加载器及索引器，为摄取、清洗、转换和准备供LLM使用的数据提供了强大的工具。从读取文本文件和网页到计算用于向量 (vector)存储的文本嵌入 (embedding)等任务，都在Python生态系统中得到良好支持。

### 行业标准和社区

Python是数据科学和机器学习 (machine learning)等更广泛方面的主导语言。因此，大量与AI相关的研究、工具开发和实际实现都首先在Python中进行。选择Python进行LLM开发可使您的工作与行业标准保持一致，更容易找到相关资源和人才，并确保获取该方向的最新进展。

总之，Python广泛的库、易用性、集成优势、强大的API支持、数据处理能力以及作为行业标准的地位，使其成为开发、试用和部署LLM工作流程及应用的非常合适的选择。本课程将大量使用Python及其生态系统来构建实用的LLM驱动系统。

## 参考资料

- [Transformers Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  `transformers`库的官方文档，该库允许Python用户访问包括大型语言模型在内的众多预训练模型。
- [LangChain Python Documentation](https://python.langchain.com/docs/get_started/introduction) — LangChain (2024)
  LangChain的官方文档，LangChain是一个用于开发由大型语言模型驱动的应用程序的框架，展示了Python在构建工作流中的作用。
- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/) — LlamaIndex (2024)
  LlamaIndex的官方文档，LlamaIndex是一个面向大型语言模型应用程序的数据框架，展示了Python在大型语言模型数据处理方面的优势。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本介绍Python核心数据科学库（Pandas、NumPy）的基础书籍，这些库对于大型语言模型工作流中的数据准备至关重要。
