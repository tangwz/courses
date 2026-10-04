---
course: "python-llm-workflows"
chapter: "advanced-langchain-chains-agents"
lesson: "langchain-agent-tools"
sourceId: 5160
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-5-advanced-langchain-chains-agents/langchain-agent-tools"
title: "代理可用工具"
description: "了解代理可用的标准工具（如搜索、计算等）。"
order: 4
plots: []
sourceHash: "434cdf9d1567f741820dcff448ce90075aee52c4363c5c37a5e3a15c79f3fdb4"
sourceCorrections: []
---

代理不只进行简单的文本生成，它们的能力因此得以提升。它们通过**工具**进行交互。可以将工具看作是代理可以选择使用的专业功能或服务，以完成任务的特定部分。如果说大型语言模型（LLM）是代理的大脑，负责决定*做什么*，那么工具就是它的手，让它能够*执行*操作。LangChain 提供了一系列预构建工具，并且可以方便地定义自己的工具。

让代理能够使用工具，会显著增加其能力范围。仅凭大型语言模型本身，无法浏览当前的互联网、执行精确计算或查询私有数据库中的信息。工具弥补了这一不足。

### 常用工具分类

LangChain 提供了多种工具，通常分为几个实用的类别：

1. **搜索工具：** 这类工具让代理能够查询 Google、DuckDuckGo、Bing 等搜索引擎，或专门的搜索 API（例如维基百科、ArXiv）。当代理需要获取最新信息或训练数据中没有的事实时，这类工具非常必要。例如，当代理被问及今天的天气时，就会使用搜索工具。设置这些工具通常需要从相应的搜索提供商获取 API 密钥，并安全地配置它们（如第 2 章中说明的）。
2. **计算工具：** 尽管大型语言模型理解数学思想，但它们在精确算术或复杂计算方面常有不足。计算工具解决了这个问题。

   - `LLMMathChain`：此工具使用大型语言模型确定*需要进行什么*计算，然后使用一个独立的、性能稳定的计算器（如 Python 的 `numexpr` 库）来获得精确答案。
   - `Python REPL Tool`：这为代理提供了访问 Python 读取-评估-打印循环（Read-Eval-Print Loop）的能力。代理可以编写和执行 Python 代码来执行计算、操作数据或进行 Python 允许的任何其他操作。*请极其谨慎地使用此工具*，因为它允许任意代码执行。
3. **数据库工具：** 代理可以与存储在 SQL 数据库中的结构化数据进行交互。`SQLDatabaseToolkit`（或特定工具如 `QuerySQLDataBaseTool`）允许代理查看数据库模式、根据自然语言请求构建 SQL 查询、执行这些查询并理解结果。这使得询问公司数据库中存储的业务指标等应用成为可能。
4. **API 交互工具：** 许多现代应用程序依赖于 API。代理可以配备与特定 API 交互的工具。这可能包括：

   - 通用 HTTP 请求工具（如 `requests` 库的封装），代理可以在其中生成 API 调用。
   - 专门为与特定 API 端点交互而构建的自定义工具（例如，一个通过公司内部 API 查询客户订单历史的工具）。
5. **文件系统工具：** 代理可以获得在本地文件系统上读取、写入和列出文件的工具。与 Python REPL 工具类似，这需要仔细考虑安全方面的问题，因为它可能授予广泛的访问权限。
6. **网页浏览工具：** 除了简单的搜索，有些工具还允许代理加载网页内容，可能从中提取特定信息，甚至在网站中进行导航。这在底层使用了 `requests` 或 `BeautifulSoup` 等库。

### 代理如何使用工具

当你初始化一个代理时，你需要向它提供一个可用 `Tool` 对象列表。每个 `Tool` 通常都有一个 `name`（例如，“搜索”、“计算器”）和一个 `description`。代理的大型语言模型利用这些描述来决定哪个工具（如果有的话）适合满足用户的请求或帮助达成其目标。代理的内部逻辑决定了序列：接收输入，决定采取行动（使用工具或直接回应），执行行动，观察结果，然后重复，直到任务完成。

> 一个代理接收用户输入，分析所需动作，选择并执行合适的工具（如搜索），观察结果，并组织最终回应的过程。

### 自定义工具

你不仅限于使用 LangChain 提供的工具。此框架的一个重要优势在于能够轻松地从自己的 Python 函数创建自定义工具。通过添加一个描述函数功能的特定文档字符串（docstring），你可以将其封装在一个 `Tool` 对象中，赋予其名称和描述，并使其可供代理使用。这使你能够整合专有逻辑、内部服务或应用程序所需的任何专业功能。

了解可用工具以及代理如何选择和使用它们，对于构建能够有效解决复杂多步问题的智能工作流程非常重要。接下来的部分将指导你如何实现代理并在实际中应用这些工具。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: ICLR; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  提出了一种通用框架，使语言模型能够交错进行推理轨迹和任务特定行动，通过工具使用在解决问题和事实核查方面展现出增强的能力。
- [Generative AI with LangChain: Build great LLM-powered applications with LangChain](https://www.oreilly.com/library/view/generative-ai-with/9781098150499/) — Ashwin Ramachandran, Raymond Hsiao, Anurag Singh, Anubhav Singh (2024)
  Publisher: O'Reilly Media
  一本实用指南，提供了使用LangChain开发应用的详细示例和实践，包括关于代理和集成各种工具的章节。
