---
course: "langchain-production-llm"
chapter: "evaluation-monitoring-observability"
lesson: "langsmith-debugging"
sourceId: 3908
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-5-evaluation-monitoring-observability/langsmith-debugging"
title: "使用 LangSmith 进行调试和根本原因分析"
description: "借助 LangSmith 追踪记录来识别并解决链和代理中的问题。"
order: 4
plots: []
sourceHash: "0ea4f0eebb7df7de95875ad1e433e2f4f6a029ca578923f05c1e697c9d10b240"
sourceCorrections: []
---

LangSmith 为监控和评估提供了核心功能。当生产问题不可避免地出现时，其细粒度追踪在调试时尤其有效。与传统软件相比，调试使用大型语言模型 (LLM) 构建的应用会带来独特的难题。LLM 响应的非确定性、多步骤链的构造特点以及提示词 (prompt)或数据处理中可能出现的微小错误，都需要专用工具来有效进行根本原因分析。LangSmith 正好提供了这种对 LangChain 应用内部运作情况的查看能力。

当应用行为异常时，无论是抛出错误、产生不正确答案还是耗时过长，通常第一步都应是检查 LangSmith 中对应的追踪记录。每条追踪记录都提供了详细的、分层执行日志，记录了处理请求所涉每个组件的输入、输出、时间以及可能的错误。

### 理解调试追踪视图

LangSmith 追踪记录不仅仅是日志；它是应用执行流程的结构化表示。调试时需留意的要素包含：

1. **运行层级：** 追踪记录常有嵌套结构。顶层运行（例如，代理执行）包含子运行（例如，LLM 调用、工具执行、检索器查询）。这种层级结构即刻展现了不同操作之间的顺序和关系，使应用逻辑的追踪更简单。
2. **输入和输出：** 对于追踪记录中的每个步骤（运行），LangSmith 都会显示收到的确切输入和生成的输出。这价值极大。你可以看到发送给 LLM 的准确提示词 (prompt)、收到的原始文本响应、传递给工具的数据、检索器返回的文档以及最终解析的输出。预期与实际输入/输出之间的不一致，通常能指出错误的源头。
3. **延迟：** 每次运行都会显示其持续时间。这有助于识别性能瓶颈。如果追踪记录显示总延迟过高，你可以进一步查看子运行，以确定是否是某个特定的 LLM 调用、工具执行或数据检索步骤造成。
4. **状态和错误：** 运行会被标明状态（成功、错误）。如果发生错误，LangSmith 会捕获异常类型和消息，并将其直接与失败的组件关联。这即刻告知你故障发生在*何处*。

> 一个简化的代理执行流程，说明了 LangSmith 追踪记录中捕获的步骤。工具执行期间的错误被突出显示。

### 使用 LangSmith 调试的常见情况

让我们思考如何使用 LangSmith 追踪记录处理典型问题：

- **情况：意外或不正确的输出：** 应用无错误运行，但产生无意义或事实不准确的信息。

  - **分析：** 逐步检查追踪记录。
    - 检查发送给生成输出的 LLM 的最终提示词 (prompt)。它格式良好吗？是否包含了必要上下文 (context)？
    - 检查前一步骤的输出。如果是 RAG 应用，检索器是否检索到了相关文档？它们是否正确融入了提示词？
    - 查看中间的 LLM 调用（例如，代理中的思考过程）。模型推理 (inference)正确吗？是否误解了指令或上下文？
    - 检查所有输出解析器。它们是否正确处理了原始 LLM 响应，还是误解了结构？
- **情况：工具执行故障：** 追踪记录显示工具运行的错误状态。

  - **分析：**
    - 在追踪层级中找到失败的工具运行。
    - 检查提供的错误消息。LangSmith 通常会捕获 Python 异常。
    - 查看传递给工具的 `inputs`。通常，错误是由于 LLM 或前序逻辑生成的格式错误的输入（例如，不正确的参数 (parameter)、错误的数据类型）引起的。
    - 如果错误暗示连接问题，验证工具的外部依赖（例如，API 端点）是否正常运行。
- **情况：输出解析错误：** 最后一步失败，因为输出解析器无法处理 LLM 的响应。

  - **分析：**
    - 找到失败的输出解析器运行。
    - 检查其 `input`，这通常是前一个 LLM 调用的原始字符串输出。
    - 将此原始输出与解析器期望的结构（例如，JSON、编号列表）进行比较。通常，LLM 未遵循提示词中的格式指令。这可能需要调整提示词或使解析器更强大。
- **情况：高延迟：** 用户报告应用运行缓慢。

  - **分析：**
    - 打开慢请求的追踪记录。总运行时间显示在顶部。
    - 检查追踪视图中每个子运行的延迟（常可视化为时间线或甘特图）。
    - 识别耗时最长的步骤。是某个特定的 LLM 调用吗？一个耗时的工具执行？还是一个慢速向量 (vector)数据库查询？这缩小了优化工作应集中的范围（例如，提示词调优、缓存、优化检索、并行化调用）。

### 使用 LangSmith 功能高效调试

在检查单个追踪记录的同时，LangSmith 提供了帮助在多条运行记录中调试的功能：

- **筛选和搜索：** 生产应用会生成大量追踪记录。使用筛选功能来筛选出相关的记录。可以按状态（`错误`）、延迟（`> 5s`）、您添加的特定标签（例如，`user_segment: premium`、`chain_type: RAG`）、元数据或用户反馈评分进行筛选。搜索特定的错误消息也能快速归类类似的故障。
- **数据集整理：** 当你识别出失败的追踪记录时，可以直接将其添加到 LangSmith 数据集。这会捕获边缘情况，并允许你系统地针对其测试修复，确保问题得到解决且不破坏其他功能。
- **Playground 集成：** 你通常可以将追踪记录中的特定运行（例如，带有确切输入的 LLM 调用）直接发送到 LangSmith Playground。这允许你独立试验提示词 (prompt)变体或模型设置，以查看是否能纠正行为，而无需重新运行整个应用。
- **反馈关联：** 如果你收集用户反馈（例如，赞/踩）并将其记录到 LangSmith，你可以根据负面反馈筛选追踪记录。点击反馈评分通常会直接带你到关联的追踪记录，为用户可能不满意的原因提供即时背景。

LangChain 中的调试通常是一个迭代过程：发现问题，使用 LangSmith 追踪并查明根本原因，修改代码（调整提示词、修正工具逻辑、改进解析），重新部署，然后再次监控 LangSmith 以确认修复并确保未引入回退。通过提供对执行流程的透彻了解，LangSmith 将调试从猜测转变为系统性调查。

## 参考资料

- [LangSmith Documentation: Debugging](https://docs.langchain.com/docs/langsmith/) — LangChain Team (2024)
  Publisher: LangChain
  官方文档提供了关于如何使用 LangSmith 的追踪功能和专门的调试特性来排查 LangChain 应用程序的全面指南。
- [Designing Machine Learning Systems: An Introduction to MLOps](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书提供了 MLOps 的基本原则，涵盖了复杂机器学习系统的系统监控、评估和调试策略，这些原则也高度适用于基于 LLM 的应用程序。
- [LangChain Documentation: Monitoring & Evaluation](https://python.langchain.com/docs/langsmith/evaluation/) — LangChain Team (2024)
  解释了 LangChain 如何与 LangSmith 集成以进行全面的监控和评估，为本节中讨论的调试功能提供了背景信息。
