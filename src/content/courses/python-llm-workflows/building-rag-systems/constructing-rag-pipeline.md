---
course: "python-llm-workflows"
chapter: "building-rag-systems"
lesson: "constructing-rag-pipeline"
sourceId: 5189
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-7-building-rag-systems/constructing-rag-pipeline"
title: "构建 RAG 流程"
description: "构建端到端 RAG 工作流所涉及的步骤。"
order: 5
plots: []
sourceHash: "6465983eb676e189ff137758098725479a623a64aaf928d623b0fc8d3d8a0119"
sourceCorrections: []
---

构建检索增强生成（RAG）流程需要理解 RAG 的主要理念以及向量 (vector)存储、嵌入 (embedding)等组件的作用。在实践中构建 RAG 流程，是将外部数据源与语言模型相连接，使其能够生成有依据且上下文 (context)相关的回应。

RAG 系统的基本流程是：根据用户的查询检索相关信息，然后将这些信息连同原始查询一同提供给大型语言模型（LLM），以生成最终答案。

> 典型的 RAG 流程：查询 -> 检索 -> 增强 -> 生成。

我们来将构建过程分解为不同的阶段，这些阶段通常使用 LangChain 或 LlamaIndex 等库来实现。

### 步骤 1：加载和准备数据

首先，您需要摄取外部知识源。这可以是文本文件、PDF、网页、数据库条目或其他格式。LlamaIndex 或 LangChain 的文档加载器提供了数据加载功能，可以简化此过程。

加载后，原始文本通常需要分割成更小、更易管理的文本块。这很重要，因为：

1. 嵌入 (embedding)模型通常对输入序列长度有限制。
2. 更小的文本块往往更集中，从而带来更好的检索相关性。
3. 它可以防止向 LLM 提供过大、可能不相关的上下文 (context)块。

常见策略包括按段落、句子分割，或使用递归字符分割，后者会尝试保持语义连贯性。

```python
# 使用 LangChain 文本分割器的示例
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 假设 'my_document.txt' 包含您的数据
loader = TextLoader('my_document.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个文本块的最大字符数
    chunk_overlap=50   # 文本块之间的重叠字符数
)
text_chunks = text_splitter.split_documents(documents)

print(f"已加载 {len(documents)} 个文档。")
print(f"已分割成 {len(text_chunks)} 个文本块。")
# 输出可能为：
# 已加载 1 个文档。
# 已分割成 42 个文本块。
```

### 步骤 2：生成嵌入 (embedding)

每个文本块都需要转换为数值向量 (vector)表示，称为嵌入。这个向量捕获文本的语义。OpenAI 的 `text-embedding-ada-002` 或 Hugging Face 的 `sentence-transformers` 库中的开源模型是常用选择。嵌入模型的选择会显著影响检索质量。

```python
# 通过 LangChain 使用 OpenAI 嵌入的示例
from langchain_openai import OpenAIEmbeddings

# 确保您的环境中已设置 OPENAI_API_KEY
embeddings_model = OpenAIEmbeddings()

# 通常您会在这里嵌入 text_chunks，
# 但向量存储通常在索引期间隐式处理此操作。
# 嵌入单个文本的示例：
# sample_embedding = embeddings_model.embed_query("This is a sample text chunk.")
# print(f"嵌入维度：{len(sample_embedding)}")
# 输出可能为：嵌入维度：1536
```

### 步骤 3：在向量 (vector)存储中索引数据

生成的嵌入 (embedding)及其对应的文本块被存储在一个称为向量存储的专门数据库中。该存储针对高效的相似性搜索进行了优化。当查询到来时，向量存储可以快速找到其嵌入与查询嵌入最接近（最相似）的文本块。

常用选项包括 Chroma、FAISS（通常在本地使用）、Pinecone、Weaviate 等。

```python
# 使用 LangChain 设置 Chroma 向量存储的示例
from langchain_community.vectorstores import Chroma

# 这将已准备好的 text_chunks 的嵌入和索引结合起来
vector_store = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings_model,
    persist_directory="./chroma_db" # 可选：保存到磁盘
)

print("向量存储已创建并填充。")
```

> LlamaIndex 提供了类似的简化过程，可以使用其 `VectorStoreIndex` 抽象来加载、分割、嵌入和索引数据到各种向量存储中。

### 步骤 4：实现检索器

检索器是负责根据用户查询从向量 (vector)存储中获取相关信息的组件。它接收查询，使用与索引时相同的模型生成其嵌入 (embedding)，并对向量存储中的嵌入执行相似性搜索（例如，余弦相似度或点积）。它通常返回最相似的前 `k` 个文本块。

```python
# 从 LangChain 向量存储创建检索器的示例
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # 检索最相似的前 3 个文本块

# 检索查询对应文档的示例
query = "What are the best practices for LLM deployment?"
retrieved_docs = retriever.invoke(query)

print(f"已为该查询检索到 {len(retrieved_docs)} 个文档。")
# for doc in retrieved_docs:
#     print(f" - {doc.page_content[:100]}...") # 打印检索内容的开头
```

### 步骤 5：增强提示

RAG 的核心理念是*增强*发送给 LLM 的提示。您不是仅仅发送用户的原始查询，而是构建一个新提示，其中包含上一步检索到的文档内容。这为 LLM 提供了必要的上下文 (context)。

一种常用模式是使用提示模板：

```python
from langchain.prompts import PromptTemplate

template = """
仅根据所提供的上下文回答以下问题：

上下文：
{context}

问题：
{question}

回答：
"""

rag_prompt = PromptTemplate.from_template(template)
```

`context` 占位符将填充 `retrieved_docs` 中的内容，而 `question` 将是原始用户查询。

### 步骤 6：生成回应

最后，增强后的提示被发送给 LLM（例如 GPT-4、Claude 3、Llama 3）以生成最终答案。LLM 使用原始问题和提供的上下文 (context)来构建其回应，使其基于检索到的信息。

LangChain 等框架提供抽象（例如 `RetrievalQA` 链或使用 LangChain 表达式语言 - LCEL），可以整齐地将这些步骤串联起来。

```python
# 使用 LangChain 的 RetrievalQA 链的示例
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 初始化 LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 创建 RetrievalQA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 'stuff' 链类型将所有检索到的文档直接放入上下文
    retriever=retriever,
    chain_type_kwargs={"prompt": rag_prompt} # 使用我们自定义的提示
)

# 使用用户查询运行链
query = "What are the main challenges in testing LLM systems?"
response = qa_chain.invoke({"query": query})

print("\nLLM 回应：")
print(response['result'])
```

> `stuff` 链类型虽然简单，但如果检索到的文档超出 LLM 的上下文窗口限制，它可能会失败。其他链类型，如 `map_reduce` 或 `refine`，以不同方式处理更大量的上下文。LlamaIndex 查询引擎提供类似的功能。

加载、分割、嵌入 (embedding)、索引、检索、增强和生成这一序列构成了 RAG 流程的骨干。虽然这概述了基本结构，但构建有效的 RAG 系统通常需要对每个步骤进行细致调整，包括文本块分割策略、嵌入模型的选择、检索参数 (parameter)以及提示本身，我们将在后面进一步阐述。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了RAG架构，展示了其在结合参数化和非参数化记忆以完成生成任务方面的有效性。
- [Retrieval-Augmented Generation (RAG)](https://python.langchain.com/docs/tutorials/rag/) — LangChain Team (2024)
  提供使用LangChain框架及其组件构建基于RAG的问答系统的实用教程。
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) — OpenAI (2024)
  Publisher: OpenAI
  介绍了如何使用OpenAI的嵌入模型，这些模型在RAG管道中常用于将文本转换为向量表示以进行语义搜索。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2023)
  DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  一份全面的学术综述，详细介绍了大型语言模型检索增强生成技术的现状、挑战和未来发展方向。
