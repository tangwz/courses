---
course: "getting-started-rag"
chapter: "building-basic-rag-pipeline"
lesson: "running-queries"
sourceId: 3663
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-5-building-basic-rag-pipeline/running-queries"
title: "在流程中执行查询"
description: "演示如何输入用户查询并从流程中获取增强回复。"
order: 6
plots: []
sourceHash: "7e63d1dd3f938760f4edf771414bd7f68740f3506a4cdce7f784c1367ff63058"
sourceCorrections: []
---

检索器和生成器组件在组装和连接后（可能使用LangChain或LlamaIndex等框架），一个基本的检索增强生成 (RAG)流程便可投入使用。该流程的目的是首先在特定的文档集中查找相关信息，然后利用这些信息生成连贯且基于上下文 (context)的回复来回答查询。

### 查询的处理流程

当您向RAG流程提交查询时，它会启动一系列操作，旨在借助您的外部知识库：

1. **查询输入：** 您将一个问题或提示以字符串形式提供给流程。
2. **检索步骤：** 流程首先将您的查询传递给检索器组件。检索器在包含已索引文档片段的向量 (vector)数据库中查找嵌入 (embedding)与查询嵌入最相似的片段。它返回一组这些相关文档片段。
3. **上下文 (context)增强：** 检索到的片段（即上下文）与您的原始查询结合。这通常涉及将它们格式化为较大的提示模板中的特定结构，该模板指导语言模型（LLM）如何使用上下文。
4. **生成步骤：** 这个增强提示（包含原始查询和检索到的上下文）被发送到LLM（即生成器组件）。
5. **最终回复：** LLM处理增强提示并生成回复。理想情况下，此回复直接回答您的查询，主要从提供的上下文中获取信息，而非仅仅依赖其内部预训练 (pre-training)知识。

这个流程确保生成的答案基于您在数据准备阶段提供的具体文档。

> 图示：查询通过RAG流程的路径，从初始输入到最终生成的回复。

### 实际执行查询

假设您已将您的RAG流程组合实例化为一个对象（为保持一致，我们称之为`rag_chain`，尽管实际名称取决于前面步骤中使用的框架或代码），执行查询通常很简单。

您将查询字符串传递给`rag_chain`对象的相应方法。LangChain等框架常为此目的使用`invoke()`或`stream()`等方法。

```python
# 假设 'rag_chain' 是您之前配置的 RAG 流程对象
# （例如，使用 LangChain 或 LlamaIndex 创建）

# 定义您的查询
user_query = "What are the main challenges mentioned in the latest project status report?"

# 通过流程执行查询
try:
    response = rag_chain.invoke(user_query)
    print("流程回复:")
    print(response)

except Exception as e:
    print(f"发生错误: {e}")

# 带有潜在流式输出的示例（如果链/LLM支持）
# try:
#     print("流式流程回复:")
#     for chunk in rag_chain.stream(user_query):
#         # 处理每个到达的片段（例如，打印它）
#         print(chunk, end="", flush=True)
#     print("\n--- 流结束 ---")
# except Exception as e:
#     print(f"\n流式处理过程中发生错误: {e}")
```

### 理解输出

上面代码中的`response`变量将包含LLM生成的最终答案。设想一个场景，我们的索引文档包含“阿尔法项目”的状态报告。

**查询：** `"最新的阿尔法项目状态报告中提到了哪些主要挑战？"`

**可能的检索上下文 (context)（简化）：**

- 片段1：“...阿尔法项目状态 - 10月26日...与传统支付系统的集成依然复杂...”
- 片段2：“...第四季度资源分配比预期更紧张，可能影响测试时间表...”
- 片段3：“...对贝塔团队API交付进度的依赖引入了风险...”

**可能的`rag_chain`回复：**
`"根据最新的阿尔法项目状态报告，提到的主要挑战是与传统支付系统集成时的复杂性，第四季度资源分配比预期更紧张可能影响时间表，以及与贝塔团队API交付进度相关的依赖风险。"`

请注意，此回复如何直接回应查询，并综合了从检索到的片段中找到的信息。它具体且基于文档内容。如果没有RAG，标准的LLM可能会提供关于项目挑战的通用答案，或者声明它无法访问具体的实时项目报告。

### 尝试与后续步骤

运行一次查询可以确认流程有效，但真正的理解来自于尝试。尝试不同类型的查询：

- **特定事实检索：** 询问您知道文档中包含的精确细节。
- **总结：** 请求总结特定文档部分或多个文档中涵盖的主题。
- **比较：** 询问文档中讨论的两个理念有何不同。
- **无直接答案的查询：** 输入一个答案不太可能在索引文档中的查询。观察流程如何回复。它是否指出信息在上下文 (context)中不可用，还是试图仅基于LLM的通用知识回答？期望的行为通常取决于创建流程时所用的提示工程 (prompt engineering)。

执行查询是RAG系统验证效果的时刻。它表明了将有针对性的信息检索与LLM的生成能力相结合的实际价值，从而根据您的具体数据来源产生相关且具备上下文感知的答案。在下一章中，我们将研究评估您的流程运行情况的方法，以及提高其效率的策略。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 33; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了原始的检索增强生成（RAG）架构，描述了其组件以及它们如何协同工作以提供事实性和上下文知情的回复。
- [Question Answering with RAG](https://python.langchain.com/docs/use_cases/question_answering/) — LangChain Team (2024)
  Publisher: LangChain
  LangChain 框架构建 RAG 应用的官方文档，包含设置管道和运行查询的示例。
- [Building a RAG Application](https://docs.llamaindex.ai/en/stable/getting_started/concepts/rag.html) — LlamaIndex Team (2024)
  Publisher: LlamaIndex
  LlamaIndex 框架的官方文档，提供了构建检索增强生成管道的指南和代码示例。
