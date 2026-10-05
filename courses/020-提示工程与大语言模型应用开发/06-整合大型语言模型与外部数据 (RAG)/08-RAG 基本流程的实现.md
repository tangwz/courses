# RAG 基本流程的实现

来源：[原文](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-6-integrating-llms-external-data-rag/basic-rag-pipeline-implementation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

检索增强生成（RAG）包含多个独立的组成部分：加载文档、分割文档、创建嵌入 (embedding)、将嵌入存储在向量 (vector)存储中以及获取相关文本块。将这些部分组合成一个可用的流程，可以构建一个系统。该系统连接这些组件，以便基于提供的外部数据回答用户查询，而不是仅仅依赖于大型语言模型的内部知识。

主要思路很简单：当用户提出问题时，我们首先从文档存储中获取相关信息，然后将这些信息连同原始问题一起提供给大型语言模型，以生成一个有依据的回答。

### RAG 的工作流程

我们来描绘一下基本 RAG 流程中信息的典型流向：

> 该图表显示了 RAG 流程中的步骤顺序，从接收用户查询到使用获取的上下文 (context)和大型语言模型生成最终回答。

### 步骤的实现

我们来分解一下这个流程的实现，假设您已经按照前面的章节设置了必要的组件：文档加载器、文本分割器、嵌入 (embedding)模型接口、已填充的向量 (vector)存储和大型语言模型接口。

1. **接收用户查询：** 当应用程序从用户那里接收到查询时，该过程开始。

   ```python
   user_query = "大型语言模型输出一致性的主要挑战是什么？"
   ```
2. **嵌入查询：** 使用与嵌入文档块时*相同*的嵌入模型，将用户的查询转换为向量表示。这里的一致性对于语义搜索正确运行非常重要。

   ```python
   # 假设 'embedding_model' 是一个已初始化的接口
   query_vector = embedding_model.embed_query(user_query)
   # query_vector 可能看起来像 [-0.012, 0.987, ..., -0.345]
   ```
3. **执行向量搜索：** 使用生成的查询向量搜索向量存储。目标是找到其嵌入与查询嵌入最相似（例如，使用余弦相似度或点积）的文档块。通常，您会获取最相关的 `k` 个顶部块。

   ```python
   # 假设 'vector_store' 是一个已初始化并填充的向量存储接口
   # 获取最相关的 3 个文档块
   k = 3
   retrieved_chunks = vector_store.similarity_search(query_vector, k=k)
   # retrieved_chunks 将是一个 Document 对象/字符串的列表
   # 块内容示例：
   # “大型语言模型输出可能不可预测。可变性源于...”
   ```
4. **增强提示：** 将获取的文档块与原始用户查询结合，为大型语言模型创建一个新的、包含丰富上下文 (context)的提示。一种常用方法是使用提示模板。

   ```python
   # 将获取的块格式化为单个上下文字符串
   context_string = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])

   # 定义一个提示模板
   prompt_template = """
   请根据以下上下文回答问题。如果上下文不包含答案，请说明。

   上下文：
   {context}

   问题：
   {question}

   回答：
   """

   # 创建最终的增强提示
   augmented_prompt = prompt_template.format(context=context_string, question=user_query)
   ```

   这个模板清楚地指示大型语言模型如何使用提供的上下文来回答问题。
5. **生成回答：** 将增强的提示发送给大型语言模型。模型将使用其内部知识和提供的特定上下文来构建回答。

   ```python
   # 假设 'llm' 是一个已初始化的大型语言模型接口
   response = llm.invoke(augmented_prompt)
   # 回答可能为：“根据提供的上下文，大型语言模型输出一致性的主要挑战包括不可预测性和源于模型参数的可变性...”
   ```
6. **返回回答：** 将大型语言模型生成的回答呈现给用户。

### 整合起来 (示例)

以下是这些步骤在一个简化函数中组合起来的样子：

```python
# 假设必要的组件 (embedding_model, vector_store, llm) 已初始化

def answer_query_with_rag(user_query: str, k: int = 3) -> str:
    """
    使用基本 RAG 流程回答用户查询。
    """
    # 1. 嵌入查询
    query_vector = embedding_model.embed_query(user_query)

    # 2. 获取相关文档
    retrieved_chunks = vector_store.similarity_search(query_vector, k=k)

    # 3. 格式化上下文
    context_string = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])

    # 4. 创建增强提示
    prompt_template = """
    上下文：
    {context}

    问题：{question}

    仅根据提供的上下文回答：
    """
    augmented_prompt = prompt_template.format(context=context_string, question=user_query)

    # 5. 使用大型语言模型生成回答
    response = llm.invoke(augmented_prompt)

    return response

# 使用示例：
user_question = "RAG 中用于文档分割的方法有哪些？"
final_answer = answer_query_with_rag(user_question)
print(final_answer)
```

这个基本流程构成了大多数 RAG 系统的基础。通过动态获取相关信息，它使得大型语言模型能够回答关于特定外部数据源的问题，克服了其静态训练知识的局限性。接下来的章节将在此基础上进行，讨论构建 RAG 应用程序的更高级技术和考量。请记住，获取步骤的质量直接影响最终生成回答的质量。因此，优化文档准备、嵌入 (embedding)选择和获取策略对于有效的 RAG 实现非常重要。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: NeurIPS 2020; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  这篇论文介绍了检索增强生成（RAG）范式，阐述了其架构和优势。
- [A Survey on Approximate Nearest Neighbor Search](https://ieeexplore.ieee.org/document/9440618) — Jianqiu Xu, Yang Cao, Yulong Huang, Bin Wang, Xinjie Zhang, Ruihui Zhao (2020)
  Journal: Big Data Mining and Analytics; Publisher: Tsinghua University Press; Volume: 3; Pages: 203-221; DOI: [10.26599/BDMA.2020.1030005](https://doi.org/10.26599/BDMA.2020.1030005)
  讨论了近似最近邻搜索的各种技术和算法，这是RAG向量搜索组件的基础。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 33; Volume: 33; DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  这篇论文介绍了情境学习，展示了大型语言模型如何在给定少量示例的情况下执行新任务，为提示工程奠定了基础。
- [LangChain Documentation](https://www.langchain.com/) — Harrison Chase (2024)
  Publisher: LangChain
  提供了构建LLM应用的全面指南和示例，包括RAG管道的详细实现模式。

---

[上一节](07-%E7%BB%93%E5%90%88%E6%A3%80%E7%B4%A2%E5%88%B0%E7%9A%84%E5%86%85%E5%AE%B9%E4%B8%8E%E6%8F%90%E7%A4%BA%E8%AF%8D.md) · [下一节](09-%E5%AE%9E%E6%93%8D%EF%BC%9A%E6%9E%84%E5%BB%BA%E6%96%87%E6%A1%A3RAG%E9%97%AE%E7%AD%94%E7%B3%BB%E7%BB%9F.md)
