---
course: "prompt-engineering-llm-application-development"
chapter: "integrating-llms-external-data-rag"
lesson: "practice-build-rag-qa-system"
sourceId: 3072
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-6-integrating-llms-external-data-rag/practice-build-rag-qa-system"
title: "实操：构建文档RAG问答系统"
description: "实操创建R一个基于所给文档并使用RAG技术回答问题的系统。"
order: 9
plots: []
sourceHash: "bdf8ba340d5a6d8ca06179e1b7cf081fbc1b60da2803112ab853867797acfecf"
sourceCorrections: []
---

本次实践将引导您构建一个功能性的问答系统，它通过组合检索增强生成（RAG）的各个组成部分实现。您将创建一款Python应用，该应用能根据所给文本文档的内容回答问题。我们将使用常用库来处理文档、生成嵌入 (embedding)、存储向量 (vector)以及与大型语言模型（LLM）API交互。

**前置条件：**

在开始之前，请确保您已安装所需的库。您通常会需要：

- 大型语言模型（LLM）客户端库（例如，用于OpenAI模型的`openai`，用于Anthropic模型的`anthropic`）。
- 嵌入库（例如，`sentence-transformers`，或者如果客户端库本身提供嵌入功能则使用客户端库）。
- 向量存储库（例如，用于本地FAISS索引的`faiss-cpu`，或`chromadb`）。
- 文本分割库，通常包含在`langchain`等框架中。对于本例，我们将假定可使用文本分割功能。

您还需要为您选择的LLM提供商准备一个API密钥。请记住安全地管理您的API密钥，例如，通过使用环境变量。

```python
# 示例环境设置
import os
# 从环境变量加载API密钥
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
# os.environ["ANTHROPIC_API_KEY"] = "YOUR_API_KEY_HERE"
```

**步骤1：准备文档**

首先，我们需要一个要查询的文档。让我们创建一个名为`my_document.txt`的简单文本文件，其中包含一些关于Python字典的内容：

```text
# my_document.txt 文件内容
Python dictionaries are versatile data structures used for storing key-value pairs.
Keys must be immutable types like strings, numbers, or tuples (if they contain only immutable elements).
Values can be of any data type. Dictionaries are unordered in older Python versions (before 3.7),
but maintain insertion order in modern Python.

You can create a dictionary using curly braces {} or the dict() constructor.
Accessing values is done using square bracket notation with the key, e.g., my_dict['my_key'].
If a key is not found, a KeyError is raised. The .get() method can be used for safer access,
allowing a default value to be returned if the key is absent.

Common dictionary methods include .keys(), .values(), and .items() to retrieve views of keys,
values, and key-value tuples, respectively. The 'in' keyword checks for key existence.
Dictionaries are mutable, meaning you can add, remove, or update key-value pairs after creation.
```

现在，让我们在Python中加载并分割这个文档。我们需要将其分解成更小的文本块，以便有效地生成嵌入并进行检索。

```python
# 假定文本分割函数可用
# 例如：from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

def load_and_split_document(file_path):
    """加载文本文件并将其分割成文本块。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # 文本分割机制的占位符
        # 在实际情况中，请使用LangChain的文本分割器等库
        # 为简化起见，这里我们按段落分割。根据需要进行调整。
        chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        print(f"文档已加载并分割成{len(chunks)}个文本块。")
        return chunks
    except FileNotFoundError:
        print(f"错误：未在 {file_path} 找到文档")
        return []

document_chunks = load_and_split_document('my_document.txt')

# 示例输出（取决于分割逻辑）：
# ['Python dictionaries are versatile data structures...',
#  'You can create a dictionary using curly braces {}...',
#  'Common dictionary methods include .keys(), .values()...']
```

文本块分割策略很重要。按段落分割对于密集文档来说可能过于粗糙。使用LangChain等库可提供更精细的分割器（例如，`RecursiveCharacterTextSplitter`），它们会考虑句子结构或固定字符长度并带有重叠，这通常能带来更好的检索结果。

**步骤2：生成嵌入并创建向量索引**

接下来，我们将文本块转换为能捕捉其语义的数字表示（嵌入）。然后，我们将这些嵌入存储在向量存储中，以便高效搜索。

我们将使用`sentence-transformers`库来生成嵌入，并使用`faiss-cpu`作为简单的本地向量存储。

```python
# 必需：pip install sentence-transformers faiss-cpu
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 加载嵌入模型
#    常用选择：'all-MiniLM-L6-v2' - 快速且效果较好
model_name = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(model_name)
print(f"已加载嵌入模型：{model_name}")

# 2. 为每个文本块生成嵌入
if document_chunks:
    chunk_embeddings = embedding_model.encode(document_chunks, convert_to_numpy=True)
    print(f"Generated embeddings of shape: {chunk_embeddings.shape}") # (文本块数量, 嵌入维度)

    # 3. 创建FAISS索引
    embedding_dimension = chunk_embeddings.shape[1]
    # IndexFlatL2 使用欧几里得距离进行相似度计算
    index = faiss.IndexFlatL2(embedding_dimension)

    # 4. 将嵌入添加到索引
    index.add(chunk_embeddings)
    print(f"已创建包含 {index.ntotal} 个向量的FAISS索引。")
else:
    print("没有可处理的文档文本块。")
    index = None # 如果没有数据，确保索引为None
```

至此，我们的`index`（FAISS向量存储）已保存文档文本块的数值表示，随时可供搜索。我们还需要保留原始的`document_chunks`列表，因为索引只存储向量，不存储文本本身。该索引使我们能够快速找到与查询向量最接近的向量（以及相应的文本块）。

**步骤3：检索相关文本块**

当用户提出问题时，我们首先使用*相同的*嵌入模型对查询进行嵌入。然后，我们使用向量存储（我们的FAISS索引）来查找与查询嵌入最相似的嵌入（以及相应的文本块）。

```python
def retrieve_relevant_chunks(query, index, embedding_model, chunks, top_k=3):
    """嵌入查询并检索最相关的 top_k 个文本块。"""
    if index is None or index.ntotal == 0:
        print("向量索引未初始化或为空。")
        return []
    if not chunks:
        print("没有可用的文档文本块。")
        return []

    # 1. 嵌入查询
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # 2. 搜索索引
    #    D: 距离, I: 最近邻居的索引
    distances, indices = index.search(query_embedding, top_k)

    # 3. 获取对应的文本块
    relevant_chunks = [chunks[i] for i in indices[0]] # indices[0] 是因为我们只查询了一个
    print(f"为查询 '{query}' 检索到 {len(relevant_chunks)} 个相关文本块。")
    return relevant_chunks

# 示例用法
user_query = "How do I safely access dictionary values in Python?"
if index:
    retrieved_chunks = retrieve_relevant_chunks(user_query, index, embedding_model, document_chunks)
    # 示例输出：
    # 为查询 'How do I safely access dictionary values in Python?' 检索到 3 个相关文本块
    # ['You can create a dictionary using curly braces {}...',
    #  'Common dictionary methods include .keys(), .values()...',
    #  'Python dictionaries are versatile data structures...'] # 顺序取决于相似度
else:
    retrieved_chunks = []
```

`top_k`参数 (parameter)决定了要检索多少个文本块。选择合适的`k`需要权衡；太少可能遗漏必要的信息，而太多则可能稀释提示或超出上下文 (context)窗口限制。

**步骤4：增强提示并生成答案**

现在，我们将检索到的文本块（我们的上下文）与原始用户查询组合起来，为大型语言模型（LLM）构建一个详细的提示。这个提示指示LLM*仅*根据所提供的上下文回答查询。

```python
# 假定 'llm_client' 是您选择的LLM API的已初始化客户端
# 例如：from openai import OpenAI; llm_client = OpenAI()
# 或者：from anthropic import Anthropic; llm_client = Anthropic()

def generate_answer(query, retrieved_chunks, llm_client):
    """构建提示并使用LLM生成答案。"""
    if not retrieved_chunks:
        return "我无法在文档中找到相关信息来回答您的问题。"

    # 1. 构建上下文字符串
    context = "\n\n".join(retrieved_chunks)

    # 2. 创建提示
    prompt = f"""
    仅根据以下上下文回答问题。
    如果上下文中不包含答案，请说“我无法根据提供的上下文回答此问题。”

    上下文：
    {context}

    问题：{query}

    答案：
    """

    print("\n--- 正在向LLM发送提示 ---")
    # print(prompt) # （可选）打印完整提示用于调试
    print("--- 提示结束 ---")

    try:
        # 3. 调用LLM API（使用OpenAI聊天补全的示例）
        # 根据需要调整参数（模型、最大令牌数、温度）
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo", # 或其他合适的模型
            messages=[
                {"role": "system", "content": "您是一个根据所提供上下文回答问题的有用助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # 降低温度可获得更事实性的答案
            max_tokens=150
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"调用LLM API时出错：{e}")
        return "抱歉，生成答案时发生错误。"

# 示例用法（假定llm_client已配置）
# final_answer = generate_answer(user_query, retrieved_chunks, llm_client)
# print(f"\nAnswer:\n{final_answer}")

# 示例预期输出：
# --- 正在向LLM发送提示 ---
# --- 提示结束 ---
#
# 答案：
# 为了在Python中安全地访问字典值并避免在找不到键时出现KeyError，您可以使用.get()方法。此方法允许您在键不存在时提供一个默认值。例如，`my_dict.get('non_existent_key', 'default_value')`。
```

这种提示结构清楚地区分了指令、源自*您的*文档的上下文以及用户的问题。指示LLM*仅*依赖于所提供的上下文对于将响应基于文档信息而言非常重要。

**整合所有部分：一个基本的RAG流程**

让我们将这些步骤整合到一个脚本中。请记住替换API密钥的占位符，并可能根据您的具体设置调整模型名称或库的使用。

```python
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI # 或您偏好的LLM客户端

# --- 配置 ---
DOCUMENT_PATH = 'my_document.txt'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_STORE_DIMENSION = 384 # 'all-MiniLM-L6-v2' 的维度
TOP_K_RESULTS = 3
LLM_MODEL = "gpt-3.5-turbo" # 或其他有能力的模型

# --- 设置 ---
# 安全加载API密钥（请替换为您的方法）
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI() # 假定OPENAI_API_KEY已在环境中设置

# --- 函数（来自前述步骤） ---
def load_and_split_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        print(f"文档已加载并分割成{len(chunks)}个文本块。")
        return chunks
    except FileNotFoundError:
        print(f"错误：未在 {file_path} 找到文档")
        return []

def setup_rag_components(chunks, model_name, dimension):
    if not chunks:
        return None, None
    try:
        embedding_model = SentenceTransformer(model_name)
        chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings)
        print(f"已初始化包含 {index.ntotal} 个向量的FAISS索引。")
        return embedding_model, index
    except Exception as e:
        print(f"RAG组件初始化出错：{e}")
        return None, None

def retrieve_relevant_chunks(query, index, embedding_model, chunks, top_k):
    if index is None or index.ntotal == 0 or not chunks:
        print("向量索引或文本块不可用于检索。")
        return []
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)
        relevant_chunks = [chunks[i] for i in indices[0]]
        print(f"已检索到 {len(relevant_chunks)} 个相关文本块。")
        return relevant_chunks
    except Exception as e:
        print(f"检索过程中出错：{e}")
        return []

def generate_answer(query, retrieved_chunks, llm_client, llm_model):
    if not retrieved_chunks:
        return "我无法在文档中找到相关信息来回答您的问题。"
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    仅根据以下上下文回答问题。
    如果上下文中不包含答案，请说“我无法根据提供的上下文回答此问题。”

    上下文：
    {context}

    问题：{query}

    答案：
    """
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "您是一个根据所提供上下文回答问题的有用助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"调用LLM API时出错：{e}")
        return "抱歉，生成答案时发生错误。"

# --- 主程序执行 ---
if __name__ == "__main__":
    # 1. 加载和准备文档
    document_chunks = load_and_split_document(DOCUMENT_PATH)

    # 2. 设置嵌入和向量存储
    embedding_model, vector_index = setup_rag_components(
        document_chunks, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIMENSION
    )

    if embedding_model and vector_index:
        # 3. 获取用户查询
        user_query = input("请输入您关于文档的问题：")

        # 4. 检索相关上下文
        retrieved_context = retrieve_relevant_chunks(
            user_query, vector_index, embedding_model, document_chunks, TOP_K_RESULTS
        )

        # 5. 生成答案
        final_answer = generate_answer(
            user_query, retrieved_context, client, LLM_MODEL
        )

        print("\n--- 最终答案 ---")
        print(final_answer)
    else:
        print("RAG组件设置失败。退出。")
```

> 本次实践中构建的RAG系统的基本工作流程。

**总结**

您已成功构建一个基本的检索增强生成系统！这个应用展示了如何将LLM与文档中包含的外部知识联系起来。通过根据语义相似性检索相关文本块并将其作为上下文提供，LLM可以生成基于特定信息的答案，克服其静态训练数据的局限。

这是一个基础示例。RAG系统通常涉及更精密的技巧，例如：

- 更进阶的文本块分割策略。
- 具有更多功能（元数据过滤、云托管）的不同向量存储。
- 在发送给LLM之前，对检索到的结果进行相关性重排。
- 结合关键词和语义搜索的混合搜索方法。
- 评估检索和生成质量的评估框架。

尝试使用不同的文档、问题、嵌入模型和LLM参数，以了解它们对结果的影响。这个实践基础为您构建更复杂和有效的RAG应用做好了准备。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems 33 (NeurIPS 2020); Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成（RAG）的概念，一个结合检索与序列到序列模型的框架。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410.pdf) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3981-3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  提出了Sentence-BERT，一种生成语义上有意义的句子嵌入的方法，用于语义搜索等任务。
- [Billion-scale similarity search with GPUs](https://ieeexplore.ieee.org/document/8023168) — Jeff Johnson, Matthijs Douze, Hervé Jégou (2017)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Volume: 40; Pages: 3022-3037; DOI: [10.1109/TPAMI.2017.2747600](https://doi.org/10.1109/TPAMI.2017.2747600)
  描述了FAISS，一个用于密集向量高效相似性搜索和聚类的库。
- [OpenAI API Documentation](https://platform.openai.com/docs/) — OpenAI (2024)
  Publisher: OpenAI
  与OpenAI模型交互的官方文档，包含API参考和使用示例。
