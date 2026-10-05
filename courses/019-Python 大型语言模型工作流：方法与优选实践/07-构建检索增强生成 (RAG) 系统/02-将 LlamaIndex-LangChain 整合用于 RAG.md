# 将 LlamaIndex/LangChain 整合用于 RAG

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-7-building-rag-systems/integrating-llamaindex-langchain-rag)

[返回章节目录](README.md) · [返回课程目录](../README.md)

检索增强生成（RAG）通过在大型语言模型生成回复前为其提供相关外部信息来提升其表现。此过程包含两个主要环节：一是获取相关数据，二是根据原始查询和获取到的数据生成回答。像 LangChain 和 LlamaIndex 这样的 Python 库经常配合使用，以有效地搭建 RAG 系统。尽管每个库都能执行一些重叠功能，但它们各自具备的主要优势使得它们在构建 RAG 流程时高度互补。

LlamaIndex 主要擅长 RAG 过程中数据摄取、建立索引和数据获取的部分。它的强项在于连接各种数据源（文件、数据库、API），解析文档，并创建结构化的索引，尤其是针对语义搜索优化的向量 (vector)索引。在典型的 RAG 设置中，LlamaIndex 负责接收用户查询并从您的知识库中找出最相关的信息片段。

另一方面，LangChain 为编排整个 LLM 工作流程提供了一个全面的框架。它提供与 LLM 交互、管理提示、将组件连接起来以及定义代理的抽象。在 RAG 的语境下，LangChain 通常管理着整体流程：它接收用户查询，使用一个检索器（通常由 LlamaIndex 提供支持）来获取相关上下文 (context)，利用其模板功能将此上下文与原始查询一同格式化为提示，将组合好的提示发送给 LLM，并可能解析 LLM 的输出。

### 常见整合模式

将这些库结合起来最常见的方式，是将 LlamaIndex 的索引和查询引擎用作 LangChain 链中的 `Retriever`。LangChain 定义了一个标准 `Retriever` 接口，该接口明确了如何根据查询获取相关文档。LlamaIndex 提供了与自身索引直接配合的该接口实现。

1. **LlamaIndex 作为 LangChain 检索器：** 您首先使用 LlamaIndex 构建您的数据索引（例如，一个 `VectorStoreIndex`）。然后，使用 LlamaIndex 的 `as_retriever()` 方法从该索引创建一个检索器。这个检索器对象可以直接用于设计用于问答的 LangChain 链，例如 `RetrievalQA` 或使用 LangChain 表达式语言 (LCEL) 构建的链。LangChain 会处理与 LLM 的交互，自动将获取到的文档（通过 LlamaIndex 检索器获得）传入提示上下文 (context)。
2. **LlamaIndex 查询引擎作为 LangChain 工具：** 对于涉及代理的更复杂工作流程，LlamaIndex `QueryEngine` 可以被包装为 LangChain `Tool`。代理可以决定何时使用此工具来查询知识库。这使得代理能够在推理 (inference)过程中需要时动态地获取特定信息。

### 流程示意

以下图表展示了一个使用 LangChain 进行编排、LlamaIndex 进行获取的标准 RAG 流程：

> 用户查询触发 LlamaIndex 检索器，该检索器从索引中获取相关文档。LangChain 将这些文档与原始查询一同格式化为 LLM 的提示词 (prompt)，LLM 随后生成最终回答。

### 简化整合示例

我们来看一个将 LlamaIndex 检索器整合到 LangChain `RetrievalQA` 链中的简化 Python 示意。假设您已构建了一个 LlamaIndex `index` 对象（在 LlamaIndex 基础章节中有所涉及）。

```python
# 假设 'index' 是一个预先构建好的 LlamaIndex 索引对象
# 假设 'llm' 是一个预先配置好的 LangChain LLM 对象

from langchain.chains import RetrievalQA
# LlamaIndex 提供了检索器接口的实现
llama_retriever = index.as_retriever()

# LangChain 链使用 LlamaIndex 检索器
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 仅为演示使用的简单链类型
    retriever=llama_retriever
)

# 使用整合后的链
query = "使用向量数据库进行 RAG 的主要优点是什么？"
response = rag_chain.run(query) # LangChain 编排调用

print(response)
# 输出将是 LLM 的回答，其依据是 LlamaIndex 获取到的文档
```

在该代码片段中：

1. 我们直接从 LlamaIndex `index` 中获得一个 `retriever` 对象。
2. 我们实例化一个 LangChain `RetrievalQA` 链，并将 LLM 接口和 `llama_retriever` 都传递给它。
3. 当调用 `rag_chain.run()` 时，LangChain 会使用 `llama_retriever`（其内部调用 LlamaIndex）来获取与 `query` 相关的文档。接着，它会构建提示词 (prompt)，调用 `llm`，并返回结果。

通过整合这些库，您可以发挥 LlamaIndex 在高效数据处理和数据获取方面的独特优势，同时结合 LangChain 灵活的框架来构建和管理整体 LLM 应用逻辑。这种职责分离有助于 RAG 系统更清晰、更易于维护，并且通常性能也更好。后续章节将详细介绍向量 (vector)存储的设置和这些流程的构建步骤。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了RAG框架应用于大型语言模型。
- [Question Answering with RAG](https://python.langchain.com/docs/use_cases/question_answering/) — LangChain Developers (2024)
  Publisher: LangChain
  使用LangChain构建RAG应用程序的官方指南。
- [LangChain Integration](https://docs.llamaindex.ai/en/stable/integrations/frameworks/langchain.html) — LlamaIndex Developers (2024)
  关于将LlamaIndex组件与LangChain结合使用的详细信息。
- [Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2023)
  Journal: arXiv preprint arXiv:2312.10997; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  对RAG系统及其进展的全面回顾。

---

[上一节](01-%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90%E5%8E%9F%E7%90%86.md) · [下一节](03-%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93%E5%92%8C%E5%B5%8C%E5%85%A5%E6%8A%80%E6%9C%AF%E6%A6%82%E8%BF%B0.md)
