---
course: "building-llm-apps-with-langchain"
chapter: "data-connection-rag"
lesson: "practice-qa-over-documents"
sourceId: 7434
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-5-data-connection-rag/practice-qa-over-documents"
title: "动手实践：基于您文档的问答"
description: "一个动手实践，用于构建一个完整的 RAG 系统，根据提供的 PDF 文档回答问题。"
order: 7
plots: []
sourceHash: "39da4bf201939caa106b9a73a23efede68caf226773541db98f0cb15f57197ba"
sourceCorrections: []
---

通过结合检索增强生成 (RAG) 系统的各组成部分，可以构建一个完整的问答应用。这需要加载文档并生成检索器。此程序的目的是，通过首先获取相关资料，然后使用大型语言模型 (LLM) 组织回答，来解答关于 PDF 文档的问题。

本次实践环节将带您走过整个工作流程，让您了解每个部分如何协作，从而构成一个能处理数据的应用。

### 项目目标：基于研究论文的问答

为本次练习，我们将使用介绍 Transformer 结构的论文《Attention Is All You Need》。我们的应用将加载这份 PDF，对其进行处理，并回答关于其内容的具体问题。您可以从其 [arXiv 页面](https://arxiv.org/pdf/1706.03762.pdf) 下载此 PDF，并将其保存到您的项目目录中，命名为 `attention-is-all-you-need.pdf`。

### 步骤 1：环境配置

首先，请确保您已安装所需的库。我们将使用 `langchain` 及其社区扩展包，`openai` 用于模型，`pypdf` 用于加载 PDF 文件，`langchain-chroma` 用于向量 (vector)库，以及 `tiktoken` 用于文本分词 (tokenization)。

```bash
pip install langchain langchain-community langchain-openai langchain-chroma pypdf tiktoken
```

接下来，我们来设置 Python 脚本。我们将导入需要的模块并配置 OpenAI API 密钥。为保障安全，请确认您的 API 密钥已作为环境变量设置。

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 设置您的 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
```

### 步骤 2：加载和切分文档

我们的首要任务是加载 PDF。我们将为此使用 `PyPDFLoader`。加载后，将文档切分成更小的文本块非常必要。这能保证获取到的上下文 (context)符合大型语言模型 (LLM) 的上下文窗口大小，并提高搜索结果的准确性。我们将使用 `RecursiveCharacterTextSplitter`，这是一个不错的通用选项。

```python
# 1. 加载文档
loader = PyPDFLoader("attention-is-all-you-need.pdf")
documents = loader.load()
print(f"从 PDF 加载了 {len(documents)} 页。")

# 2. 将文档切分为文本块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, 
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)
print(f"将文档切分成了 {len(splits)} 个文本块。")
```

此处，`chunk_size` 设为 1200 个字符，`chunk_overlap` 设为 200 个字符。这种重叠有助于保持文本块间的语义关联 (semantic relationship)，从而保证句子或观点不会被突然中断。

### 步骤 3：创建嵌入 (embedding)和向量 (vector)库

文本块准备好后，我们需要将它们转换为数值向量（嵌入），以便进行语义搜索的索引。我们将为此使用 `OpenAIEmbeddings`。这些嵌入随后会存储在 `Chroma` 向量库中。

以下代码会初始化嵌入模型，并在一个步骤中从文档切分中生成向量库。

```python
# 3. 创建嵌入和向量库
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embedding_model
)
```

`Chroma.from_documents` 方法负责为每个文本块生成嵌入并将其索引到向量数据库中的过程。现在，我们文档的内容已存储，并能高效获取了。

### 步骤 4：构建检索链

现在我们来组装最终的链。在现代 LangChain 中，我们将一个“文档链”（负责回答组织）与一个“检索链”（负责获取数据）结合起来。

首先，我们从向量 (vector)库中生成一个检索器。检索器是一种接口，它在接收到查询后，会给出相关文档。

接下来，我们定义一个提示模板，指示模型使用所给上下文 (context)来回答问题。最后，我们构建这些链。

```python
# 4. 初始化大型语言模型 (LLM) 和检索链
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 定义一个提示模板
prompt = ChatPromptTemplate.from_template("""请仅根据所给上下文回答以下问题：

<context>
{context}
</context>

问题: {input}""")

# 创建文档链 (组织答案)
document_chain = create_stuff_documents_chain(llm, prompt)

# 创建检索链 (获取文档并将其传递给文档链)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

我们使用 `create_stuff_documents_chain`，它会获取检索到的文档，将它们合并成一个上下文，然后将该上下文与用户的问题一起传递给大型语言模型 (LLM)。

下图描绘了我们刚刚构建的完整 RAG 工作流程。

> RAG 流水线包含两个主要阶段。摄取阶段处理文档并将其索引到向量库中。查询阶段根据用户查询获取相关文档，并利用这些文档生成最终回答。

### 步骤 5：提出问题

我们的系统已准备就绪。我们可以使用包含输入内容的字典来调用 `retrieval_chain`。现在，让我们提出一些与论文内容直接关联的问题。

```python
# 5. 提出问题
query1 = "在《Attention Is All You Need》这篇论文中，提出的主要架构是什么？"
response1 = retrieval_chain.invoke({"input": query1})
print("问题 1:", query1)
print("回答 1:", response1['answer'])

print("\n" + "="*50 + "\n")

query2 = "描述编码器和解码器堆栈中的两个主要子层。"
response2 = retrieval_chain.invoke({"input": query2})
print("问题 2:", query2)
print("回答 2:", response2['answer'])
```

#### 预期输出

输出将由大型语言模型 (LLM) 根据从 PDF 获取的文本块生成。它会与以下内容相似：

```text
问题 1: 在《Attention Is All You Need》这篇论文中，提出的主要架构是什么？
回答 1: 《Attention Is All You Need》论文中提出的主要架构是 Transformer，这是一种模型结构，它避免了递归，而是完全依赖于注意力机制来捕捉输入和输出之间的全局关联。

==================================================

问题 2: 描述编码器和解码器堆栈中的两个主要子层。
回答 2: 在编码器和解码器堆栈中，每个层都有两个主要的子层。第一个是多头自注意力机制，第二个是一个简单的、位置感知型全连接前馈网络。在两个子层周围都使用了残差连接，随后是层归一化。
```

这些回答准确无误，直接源自原始文档，展现了 RAG 方法的成效。模型不只是依赖其预训练 (pre-training)的知识；它利用所给上下文 (context)，提供具体且基于事实的回答。

### 总结与后续步骤

在本次实践练习中，您成功构建了一个完整的检索增强生成 (RAG)系统。您学会了如何将文档加载器、文本切分器、嵌入 (embedding)模型、向量 (vector)库和大型语言模型 (LLM) 串联起来，以构建一个强大的私人文档问答应用。

这种模式是目前大型语言模型 (LLM) 最常见且成效显著的应用之一。我们建议您通过以下方式进行更多尝试：

- 使用不同的 PDF 或文档类型。
- 调整 `chunk_size` 和 `chunk_overlap`，看看它对效果有何影响。
- 对于非常大的文档，可以试用 `map_reduce` 或 `refine` 等不同的链类型。

有了这个根本，您现在已能构建精巧的应用，这些应用可以对您的数据源进行推断并与之互动。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: NeurIPS; Pages: 5998-6008; DOI: [10.5555/3295222.3295349](https://doi.org/10.5555/3295222.3295349)
  引入Transformer架构的开创性论文，该架构是大多数现代大型语言模型的基础，也是本实践中使用的特定文档。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Narsimha Chilkuri, Michael Grave, Pasquale Minervini, and Sebastian Riedel (2020)
  Journal: Advances in Neural Information Processing Systems 33; Publisher: NeurIPS; Volume: 33; Pages: 6102-6113; DOI: [10.5555/3495289.3495444](https://doi.org/10.5555/3495289.3495444)
  介绍了检索增强生成（RAG）架构，该架构结合了参数化和非参数化记忆，显著提升了知识密集型自然语言处理任务的性能。
- [LangChain Python Library Documentation](https://python.langchain.com/docs/) — LangChain Developers (2024)
  LangChain Python库的官方文档，提供了构建大型语言模型应用的全面指南、教程和API参考。
- [OpenAI Embeddings Guide](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGPOw5tm9OU6TU3iHcne8YnxgssZd6afhsh_Gn85QTWQYmg0nQi3g3J54mbeljLcJwDeysVkUehI4jzPqT3_LMd5jTHxcAWNYQCAri3CSCU-DMytfnJ_dKBnE6TeN4tL28FtwaiLc3XCNaTBp1fq2xp9_KEqykrVJhQ00TM-zBdVA==) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI的官方指南，解释了文本嵌入的概念以及如何使用其嵌入模型，这些模型是RAG系统中向量搜索的核心。
