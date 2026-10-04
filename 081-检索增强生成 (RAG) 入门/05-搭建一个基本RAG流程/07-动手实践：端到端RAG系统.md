# 动手实践：端到端RAG系统

来源：[原文](https://apxml.com/zh/courses/getting-started-rag/chapter-5-building-basic-rag-pipeline/hands-on-end-to-end-rag)

[返回章节目录](README.md) · [返回课程目录](../README.md)

好的，让我们把我们讨论过的思路和组成部分结合起来，构建并运行一个完整但基础的检索增强生成 (RAG)系统。这个实践练习展示了检索器和生成器如何一起运作，以使用外部知识回答问题。我们将使用常用模式和库模拟这些步骤，侧重于流程，而非详尽的实现细节。

假设Python环境已设置并安装了所需库。这通常包括用于向量 (vector)嵌入 (embedding)的库（如`sentence-transformers`）、一个向量数据库客户端（如`chromadb`或`faiss-cpu`）以及一个大型语言模型接口（如`openai`或`huggingface_hub`）。为简化起见，我们可能用伪代码或高级框架调用来说明RAG系统的实现方法（例如LangChain或LlamaIndex中的调用）。

### 准备示例数据

首先，我们需要一个小型知识库。我们设想有几段关于太阳系不同行星的文本片段。在实际场景下，您会从文件中加载这些内容（如第3章所讨论），但在这里我们将直接定义它们：

```python
# 代表我们知识库的示例文档
documents = [
    "Mercury is the smallest planet in our Solar System and closest to the Sun.",
    "Venus has a thick, toxic atmosphere filled with carbon dioxide and is perpetually shrouded in thick, yellowish clouds of sulfuric acid.",
    "Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
    "Mars is often called the 'Red Planet' because of its reddish appearance, due to iron oxide prevalent on its surface."
]

# 在实际场景中，我们会在这里应用分块策略。
# 为简化起见，我们将每个句子视为一个“块”。
# 假设每个块都关联着元数据，例如{'source': 'solar_system_facts.txt', 'doc_id': i}。
document_chunks = documents
```

在这里，我们通过将每个句子视为一个文档块来保持简单。请记住第3章中提到的，有效的文本分块对于大型文档很重要。

### 初始化核心组件

接下来，我们初始化主要组成部分：嵌入 (embedding)模型、向量 (vector)存储和大型语言模型。

```python
# 1. 嵌入模型
# 加载预训练模型以生成嵌入
# （例如，来自sentence-transformers库）
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 向量存储（内存版ChromaDB示例）
# import chromadb
# client = chromadb.Client() # 内存客户端
# vector_store = client.create_collection("solar_system")

# 3. 大型语言模型
# 初始化与大型语言模型的连接
# （例如，使用OpenAI的库或Hugging Face的transformers）
# llm = OpenAI(api_key="YOUR_API_KEY") # 或加载本地模型
```

这些步骤设置了所需工具：一个用于将文本转换为向量，一个用于高效存储和检索这些向量，以及一个用于根据输入提示生成文本。

### 索引数据

现在，我们处理文档块并将其存储在向量 (vector)存储中。这包括为每个块生成嵌入 (embedding)，并将块文本、其嵌入以及任何关联的元数据添加到向量数据库。

```python
# （示例继续）

# 为每个块生成嵌入
# chunk_embeddings = embedding_model.encode(document_chunks)

# 准备用于索引的数据（文本、嵌入、元数据）
# ids = [f"chunk_{i}" for i in range(len(document_chunks))]
# metadatas = [{'source': 'solar_system_facts.txt', 'doc_id': i} for i in range(len(document_chunks))]

# 添加到向量存储
# vector_store.add(
#     embeddings=chunk_embeddings,
#     documents=document_chunks,
#     metadatas=metadatas,
#     ids=ids
# )

print("数据已成功索引到向量存储中。")
```

完成此步骤后，我们的知识库就可以被查询了。向量存储现在可以执行相似性搜索，找到与用户问题相关的块。

### 定义RAG流程

我们现在定义核心逻辑，它接收用户查询，检索相关上下文 (context)，并使用大型语言模型生成答案。这可以封装在一个函数中，或者在使用框架时更常见地定义为一个链或序列。

```python
# （示例）

def answer_query_with_rag(query: str):
    # 1. 嵌入查询
    # query_embedding = embedding_model.encode([query])[0]

    # 2. 检索相关块
    # 在向量存储中搜索top_k个最相似的块
    # retrieved_results = vector_store.query(
    #     query_embeddings=[query_embedding],
    #     n_results=2 # 检索最相关的2个块
    # )
    # retrieved_docs = retrieved_results['documents'][0]
    # print(f"Retrieved Documents: {retrieved_docs}") # 可选：检查检索到的上下文

    # 3. 构建提示
    # 将原始查询与检索到的上下文结合
    # context_string = "\n\n".join(retrieved_docs)
    # prompt_template = f"""
    # 基于以下上下文，回答查询。
    # 如果上下文中没有答案，请说明。

    # 上下文：
    # {context_string}

    # 查询：{query}

    # 答案：
    # """
    # print(f"Generated Prompt:\n{prompt_template}") # 可选：检查提示

    # 4. 使用大型语言模型生成响应
    # response = llm.complete(prompt=prompt_template) # 或等效的大型语言模型API调用
    # generated_answer = response # 根据需要处理响应对象

    # 为了演示，我们将返回占位符值
    generated_answer = "This is a placeholder answer based on retrieved context."
    retrieved_docs = ["Document 1 text...", "Document 2 text..."]

    return generated_answer, retrieved_docs
```

此函数概述了RAG流程：嵌入 (embedding)查询，搜索向量 (vector)存储，使用结果构建一个详细的提示，最后，用这个增强的提示调用大型语言模型。

### 运行和测试流程

我们来测试一下基础RAG系统，使用与我们已索引数据相关的问题。

```python
# 示例查询
user_query = "Which planet is closest to the sun?"

# 执行RAG流程
final_answer, retrieved_context = answer_query_with_rag(user_query)

print("\n" + "="*30)
print(f"User Query: {user_query}")
# 在实际运行时，retrieved_context将包含实际的文档文本
print(f"Retrieved Context: {retrieved_context}")
print(f"Generated Answer: {final_answer}")
print("="*30 + "\n")

# 另一个示例查询
user_query_2 = "What is Mars known for?"
final_answer_2, retrieved_context_2 = answer_query_with_rag(user_query_2)

print("="*30)
print(f"User Query: {user_query_2}")
print(f"Retrieved Context: {retrieved_context_2}")
print(f"Generated Answer: {final_answer_2}")
print("="*30)
```

如果这被完整实现，第一个查询理想情况下应检索到关于水星的块，大型语言模型应生成类似“水星是离太阳最近的行星”的答案。第二个查询应检索到关于火星的块，并生成一个提到其红色外观或“红色星球”昵称的答案。

### 流程可视化

我们刚刚实现的流程可以按以下方式可视化：

> 一张图示，显示了在此实践练习中构建的基础RAG流程中的操作顺序。

### 总结与后续步骤

恭喜！您已经完整体验了端到端RAG流程的构建与执行。我们将数据准备、向量 (vector)存储、检索、提示工程 (prompt engineering)和大型语言模型生成连接起来。

“这个示例刻意保持简单。实际应用通常涉及更复杂的数据加载，更高级的分块策略，可能不同的嵌入 (embedding)模型或向量数据库，更复杂的提示模板，以及对未找到相关上下文 (context)等边缘情况的处理。”

虽然它能运作，但这个基础系统的质量和可靠性仍有待确认。我们如何知道检索器是否找到了*最好的*上下文？我们如何评估大型语言模型生成的答案是否准确且忠实于检索到的信息？这些问题直接引向下一章，我们将研究评估RAG系统的方法以及提升其表现的策略。

## 参考资料

- [ChromaDB Documentation](https://docs.trychroma.com/) — The ChromaDB Authors (2024)
  ChromaDB的官方文档，这是一种流行的开源向量数据库，用于存储和检索向量嵌入，在RAG实践示例中有所展示。
- [LangChain Documentation](https://docs.langchain.com/) — Harrison Chase and the LangChain Team (2024)
  LangChain的官方文档，这是一个广泛使用的框架，简化了构建大型语言模型应用（包括RAG管道）的过程，在内容中被提及为高级框架示例。

---

[上一节](06-%E5%9C%A8%E6%B5%81%E7%A8%8B%E4%B8%AD%E6%89%A7%E8%A1%8C%E6%9F%A5%E8%AF%A2.md) · [下一节](../06-RAG%E7%B3%BB%E7%BB%9F%E7%9A%84%E8%AF%84%E4%BC%B0%E4%B8%8E%E6%94%B9%E8%BF%9B/01-%E8%AF%84%E4%BC%B0RAG%E7%9A%84%E6%8C%91%E6%88%98.md)
