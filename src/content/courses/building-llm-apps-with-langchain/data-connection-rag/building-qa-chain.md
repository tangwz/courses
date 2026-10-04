---
course: "building-llm-apps-with-langchain"
chapter: "data-connection-rag"
lesson: "building-qa-chain"
sourceId: 7433
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-5-data-connection-rag/building-qa-chain"
title: "构建问答链"
description: "将所有 RAG 组成部分使用 RetrievalQA 链组合起来，构建一个完整的问答应用。"
order: 6
plots: []
sourceHash: "3d8b8519b212f809c6ca0b84f9a66f06832a8242fb424df1723275812740b083"
sourceCorrections: []
---

构建一个检索增强生成 (RAG)系统需要用到几个重要的组成部分。这些部分包括一个设计用来从向量 (vector)存储中获取相关文档的检索器，以及一个能生成文本的大语言模型 (LLM)。将这些元素结合起来，形成一个紧密的应用，使其能够接收用户问题并给出有根据的回答。

LangChain 提供特定的辅助函数来构建这个工作流程。当前普遍的做法是构建一个 `RetrievalChain`，它负责管理文档的检索以及对语言模型的后续调用。这自动化了定义 RAG 的“先检索后读取”模式。

过程直接：用户的查询首先发送给检索器。检索器从向量存储中获取最相关的文档。这些文档随后连同原始查询一起被格式化为一个提示词 (prompt)，并发送到大语言模型，大语言模型会根据给定的上下文 (context)生成最终回答。

> 检索链中的流程。用户的查询有两个作用：寻找相关文档，并作为发送给大语言模型的最终提示词的一部分。

### 实现检索链

创建检索链包含两个步骤：首先，创建一个将文档组合成提示词 (prompt)的链（问答部分），其次，将其连接到检索器。

我们使用两个工厂函数：

1. `create_stuff_documents_chain`：这接收文档列表，将它们格式化为提示词，并将它们传递给大语言模型 (LLM)。这实现了常规的“填充”策略。
2. `create_retrieval_chain`：这连接检索器到文档链，自动处理文档的获取和传递。

让我们看看实际操作。假设你有一个大语言模型实例（如 `ChatOpenAI`）和一个从你的向量 (vector)存储创建的检索器，你可以这样构建这个链：

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 假设你已经加载了文档并创建了一个向量存储 'db'
# 例如：
# db = Chroma.from_documents(split_docs, OpenAIEmbeddings())

# 1. 实例化大语言模型
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 2. 从向量存储创建检索器
retriever = db.as_retriever(search_kwargs={"k": 3})

# 3. 创建提示词模板
system_prompt = (
    "你是一个问答任务助手。 "
    "请使用以下检索到的上下文来回答问题。 "
    "如果你不知道答案，请说你不知道。 最多使用三句话，并保持回答简洁。"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 4. 创建链
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. 提问
question = "没有RAG的大语言模型有哪些局限性？"
response = rag_chain.invoke({"input": question})

print(response["answer"])
```

当你运行这段代码时，LangChain 会执行整个 RAG 流程。检索器会找到与问题相关的排名前3的文档，`question_answer_chain` 将它们插入到提示词的 `{context}` 占位符中，大语言模型则用此来回答。结果是一个基于你的特定文档信息的回答。

### 文档组合策略

上述实例使用了“填充”策略，通过 `create_stuff_documents_chain` 实现。这是最普遍的方法，但根据你的使用场景和文档数量，你可能会遇到其他模式：

- **填充**：这是最直接的方法。它接收所有检索到的文档，将它们插入到提示词 (prompt)模板中，并通过一次 API 调用将整个文本块发送给大语言模型 (LLM)。

  - **优点**：快速高效，因为它只需对大语言模型进行一次调用。它让模型能够一次性看到所有上下文 (context)。
  - **缺点**：如果你检索了大量文档或文档本身很大，很容易超出模型的上下文窗口限制。
- **映射-归约**：这种方法适用于大量文档。它首先对每个文档单独运行一个初始提示词（映射步骤）。然后，将每个文档的输出在一个单独的调用中进行组合和总结（归约步骤）。

  - **优点**：可扩展处理海量文档，远超单个上下文窗口所能容纳的数量。
  - **缺点**：需要对大语言模型进行多次调用，使其更慢、成本更高。由于每个文档在最终组合前是独立处理的，它也可能丢失一些细节。
- **精炼**：这种方法也迭代地处理文档。它首先对第一个文档运行一个提示词来生成初始回答。然后，它循环处理剩余文档，将之前的回答和新文档提供给大语言模型，以逐步精炼回答。

  - **优点**：通过在之前的回答基础上构建，可以比“映射-归约”方法整合更多细节。
  - **缺点**：也需要多次大语言模型调用，并且最终输出可能受文档处理顺序的影响。

对于大多数常规问答任务，“填充”策略是最佳起点，因为它简单且性能好。只有当你频繁遇到上下文长度错误时，才应该选择其他架构。

### 获取源文档

RAG 的一个显著优势是能够将回答追溯到其源材料。这有助于建立信任，并让用户验证信息。

`create_retrieval_chain` 自动保留检索到的文档，并将它们包含在输出字典的 `context` 键下。你不需要任何额外配置来获取它们。

```python
# 输入是一个带有 'input' 键的字典
question = "向量存储如何实现语义搜索？"
result = rag_chain.invoke({"input": question})

# 输出是一个包含答案和上下文（源文档）的字典
print("回答：")
print(result["answer"])
print("\n源文档：")
for doc in result["context"]:
    print(f"- 页面内容: {doc.page_content[:150]}...")
    print(f"  来源: {doc.metadata.get('source', 'N/A')}\n")
```

这个链的输出包含生成的 `answer` 和一个 `context` 列表。这种结构化输出对构建显示引用或让用户查看原始文本的用户界面非常有帮助。

拥有构建完整问答链并检查其来源的能力，你现在可以构建功能强大、感知数据的应用了。下一节提供一个动手实践，通过在你选择的文档上构建问答系统来巩固这些技能。

## 参考资料

- [Retrieval QA](https://python.langchain.com/docs/langchain_expression_language/cookbook/retrieval_qa_migration/) — LangChain (2024)
  Publisher: LangChain
  LangChain官方文档，提供了实现问答链的详细信息和API参考，涵盖了各种链类型和配置选项。
- [Generative AI with LangChain: Build Large Language Model Applications with Python](https://www.oreilly.com/library/view/generative-ai-with/9781098670494/) — Ben Auffarth (2024)
  Publisher: O'Reilly Media
  一本使用LangChain构建LLM应用程序的实用指南，包含RAG实现和链类型选择的具体示例和解释。
- [Prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2024)
  Publisher: OpenAI
  提供了构建有效提示词的实用建议和最佳实践，这对于理解检索到的文档如何整合到语言模型的上下文中非常重要。
