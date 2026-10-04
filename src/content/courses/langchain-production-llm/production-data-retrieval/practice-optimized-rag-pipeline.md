---
course: "langchain-production-llm"
chapter: "production-data-retrieval"
lesson: "practice-optimized-rag-pipeline"
sourceId: 3901
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-4-production-data-retrieval/practice-optimized-rag-pipeline"
title: "实践：构建优化的RAG管道"
description: "构建并评估一个结合先进索引和检索方法的RAG系统。"
order: 7
plots: []
sourceHash: "cd314e02724b628f07938b4c2f1ecde7e7dfe7fa0325bf7a7e6ebe7730dbc009"
sourceCorrections: []
---

本实践练习着重于构建一个在基础向量 (vector)搜索上改进的检索增强生成（RAG）管道。它结合了适用于生产环境的先进文档处理、混合检索和重排序技术。目标是通过使用多种检索信号并在生成前优化检索到的上下文 (context)，来构建一个提供更相关、更准确答案的系统。

假设您有一系列文档（例如，PDF或Markdown格式的技术文章、项目说明或研究报告），希望将其用作问答系统的知识库。

**先决条件：** 请确保已安装所需的库：

```bash
pip install langchain langchain-openai langchain-community langchain-chroma langchain-text-splitters sentence-transformers tiktoken pypdf rank_bm25
pip install unstructured # For more document types
```

您还需要访问一个大型语言模型（如OpenAI的模型），并可能需要将API密钥设置为环境变量。

### 1. 高级文档加载与分块

我们不使用简单的固定大小分块，而是采用`RecursiveCharacterTextSplitter`，它首先尝试根据语义边界（段落、句子）进行拆分，然后再按字符计数。这通常能更好地保持上下文 (context)。我们还需要处理加载可能不同类型的文档。

```python
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 根据您的文档路径和类型进行配置
DOCS_PATH = "./your_documents"
# 使用 DirectoryLoader 实现灵活性，如果需要，可按文件类型配置加载器
# 示例：为简化起见，此处仅加载PDF
loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
documents = loader.load()

# 高级分块策略
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, # 重叠有助于维持分块间的上下文
    length_function=len,
    add_start_index=True, # 对后续潜在的父文档检索有用
)
chunks = text_splitter.split_documents(documents)

print(f"已加载 {len(documents)} 份文档。")
print(f"已拆分为 {len(chunks)} 个分块。")

# 初始化嵌入模型（如果使用OpenAI，请确保已设置 OPENAI_API_KEY）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

如果您有其他文件类型，请考虑使用`UnstructuredMarkdownLoader`或`langchain_community.document_loaders`中的其他特定加载器。`RecursiveCharacterTextSplitter`的参数 (parameter)（`chunk_size`、`chunk_overlap`）通常需要根据您的文档特点和下游大型语言模型上下文窗口大小进行调整。

### 2. 混合搜索索引

一个高效的RAG系统通常能从结合密集（向量 (vector)）和稀疏（基于关键词）检索中获益。我们将设置这两种方式。

**a) 密集检索（向量存储）**

我们将文档分块索引到向量存储中。此处我们使用`langchain-chroma`作为本地示例，但您可以根据需要替换为自己偏好的生产级向量存储（如Pinecone、Weaviate等）。

```python
# 初始化向量存储并索引分块
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_hybrid" # 选择一个目录来保存索引
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) # 最初检索更多以进行重排序

print("向量存储已初始化，分块已索引。")
```

**b) 稀疏检索（BM25）**

BM25 (Best Matching 25) 是一种流行的基于关键词的算法。LangChain通过社区包提供了相应的检索器。

```python
from langchain_community.retrievers import BM25Retriever

# BM25 需要分块的原始文本
# 确保您的“chunks”包含 page_content 属性
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10 # 根据关键词检索前10个结果

print("BM25 检索器已初始化。")
```

### 3. 使用EnsembleRetriever实现混合搜索

现在，使用`EnsembleRetriever`组合密集和稀疏检索器。这可以对每种方法的贡献进行加权。最佳权重 (weight)通常取决于特定的数据集和查询类型，并且需要实验来确定。

```python
from langchain.retrievers import EnsembleRetriever

# 初始化集成检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6] # 示例权重：稍微优先向量搜索
)

print("集成检索器已创建。")

# 测试检索（可选）
# sample_query = "什么是提示工程的最佳实践？"
# retrieved_docs = ensemble_retriever.invoke(sample_query)
# print(f"为示例查询检索到 {len(retrieved_docs)} 份文档。")
```

### 4. 实现重排序

混合搜索会检索出各种文档，但其中一些可能仅是边缘相关的。使用计算密集型交叉编码器模型进行重排序，可以显著提升传递给大型语言模型的最终上下文 (context)质量。

我们使用`ContextualCompressionRetriever`，它封装了我们的集成检索器并应用了一个重排序模型。

```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever

# 初始化交叉编码器模型
# 像 'cross-encoder/ms-marco-MiniLM-L-6-v2' 这样的模型高效且有成效
model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

# 初始化重排序压缩器
compressor = CrossEncoderReranker(model=model, top_n=5) # 重排序后保留最相关的5份文档

# 创建压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever # 使用混合检索器作为基础
)

print("重排序检索器已配置。")

# 测试带重排序的检索（可选）
# reranked_docs = compression_retriever.invoke(sample_query)
# print(f"为示例查询重排序到 {len(reranked_docs)} 份文档。")
# 比较 reranked_docs 与 retrieved_docs 的内容
```

`top_n`参数 (parameter)控制重排序后保留的文档数量。这有助于大型语言模型聚焦于最有用的信息，并符合上下文窗口的限制。

### 5. 使用LCEL构建完整的RAG链

最后，使用LangChain表达式语言（LCEL）将优化后的检索器集成到完整的RAG链中。该链使用我们的`compression_retriever`获取上下文 (context)，将其格式化为提示，发送给大型语言模型，并解析输出。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 初始化大型语言模型
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # 或 gpt-4, claude 等

# 定义提示模板
template = """您是一个问答助手。
仅使用以下检索到的上下文来回答问题。
如果您无法从上下文中得知答案，请直接说您不知道。
请保持回答简洁，并直接基于提供的信息。

上下文：
{context}

问题: {question}

回答："""
prompt = ChatPromptTemplate.from_template(template)

# 辅助函数，用于格式化检索到的文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 构建RAG链
rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("优化后的RAG链已构建。")
```

这种LCEL结构清晰地定义了数据流：用户的提问（`RunnablePassthrough()`）被传递，同时也被`compression_retriever`用于获取和格式化上下文。然后，这些内容在提示模板中组合，由大型语言模型处理，并解析最终的字符串输出。

### 6. 调用与评估方法

您现在可以使用用户提问来调用链：

```python
query = "请解释RAG中的混合搜索概念。"
final_answer = rag_chain.invoke(query)
print("\n--- 最终回答 ---")
print(final_answer)

query_2 = "法国的首都是哪里？" # 超出上下文的问题示例
final_answer_2 = rag_chain.invoke(query_2)
print("\n--- 最终回答 2 ---")
print(final_answer_2)
```

**流程可视化：**

这是我们构建的管道图：

> 优化后的RAG管道流程，在大型语言模型生成最终答案之前，结合了混合搜索和重排序阶段。虚线表示可选的查询转换。

**评估：** 这个优化后的管道应比仅进行基础向量 (vector)搜索的RAG产生更好的结果。然而，细致的评估是必要的。如第5章所述，可以使用LangSmith等工具来追踪执行过程，并定义在代表性的问答对数据集上评估的指标（例如，RAGAS指标，如忠实性、上下文 (context)准确性、上下文召回率、答案相关性）。将此管道的性能与更简单的基线进行比较，以量化 (quantization)混合搜索和重排序所带来的改进。调整集成权重 (weight)（`EnsembleRetriever`中的`weights`）以及重排序后保留的文档数量（`CrossEncoderReranker`中的`top_n`）是根据评估结果进行的常见优化步骤。

本练习提供了一个面向生产的RAG系统模板。您可以通过集成查询转换（如HyDE）、实现父文档检索策略以获得更好的上下文，以及针对您的文档语料库优化数据加载/分块来进一步提升它。请记住，监控（第5章）和部署实践（第7章）对于在生产环境中保持性能和可靠性是必要的。

## 参考资料

- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) — Rodrigo Nogueira, Kyunghyun Cho (2019)
  Journal: arXiv preprint arXiv:1901.04085; DOI: [10.48550/arXiv.1901.04085](https://doi.org/10.48550/arXiv.1901.04085)
  这项有影响力的工作展示了如何使用像BERT这样的大型预训练语言模型进行检索到的段落重排序，为交叉编码器重排序器奠定了基础。
- [LangChain Documentation](https://python.langchain.com/docs/) — LangChain Developers (2024)
  官方文档提供了使用LangChain构建RAG管道的实践指南、API参考和示例，包括高级检索和使用LCEL构建链。
- [RAGAS: A Framework for Automated Evaluation of Retrieval Augmented Generation Systems](https://arxiv.org/abs/2309.15217) — Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2309.15217](https://doi.org/10.48550/arXiv.2309.15217)
  这篇论文介绍了RAGAS，一个用于自动评估检索增强生成（RAG）系统的框架，提供了评估忠实度和上下文准确性等方面的指标。
