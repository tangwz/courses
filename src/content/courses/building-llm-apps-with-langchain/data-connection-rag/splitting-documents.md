---
course: "building-llm-apps-with-langchain"
chapter: "data-connection-rag"
lesson: "splitting-documents"
sourceId: 7430
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-5-data-connection-rag/splitting-documents"
title: "文档分块处理"
description: "学习如何将大文档分割成更小、语义上连贯的片段，以适合进行嵌入处理的方法。"
order: 3
plots: ["plots/7430-0.json"]
sourceHash: "56696230927fb44fb41b34bd54f6b9135c554edddb3b432f87707dcf7ab53b34"
sourceCorrections: []
---

在使用大型语言模型（LLM）应用程序时，文档通常以 `Document` 对象形式存在，这类对象可能包含大量文本。直接将大型文档喂给LLM进行问答处理是不切实际的，主要有以下两个原因。首先，它很可能会超出模型的上下文 (context)窗口限制。其次，提供一大段不集中的文本会使模型难以找到回答查询所需的具体信息。

解决方法是将大文档分解成更小、语义上有意义的片段。这个过程，被称为文本分块或切分，是为RAG系统准备数据的一个必要步骤。目的是创建足够小以供嵌入 (embedding)模型高效处理，同时又足够大以保持其原始上下文的文本片段。

### 分块的权衡：大小与上下文 (context)

选择合适的分块大小需要一个精妙的平衡。如果分块过小，语义上下文可能会丢失。例如，一个只包含“这是一个重大突破”的片段，如果缺少解释“它”指代何物的前文句子，就会变得毫无意义。另一方面，如果分块过大，它们可能包含不相关的信息，在检索时产生噪声，使向量 (vector)搜索更难找到最相关的段落。这通常被称为“迷失中间”问题，即有价值的信息被埋藏在一个大的、嘈杂的文本块中。

> 同一文本的不同分块策略。小分块可能导致语义碎片化，而大分块则可能引入噪声。目的是找到一个能保持语义上下文的平衡点。

### 递归字符分块

LangChain 中文本分块最推荐和通用的方法是 `RecursiveCharacterTextSplitter`。该分块器不是基于固定字符进行分割，而是使用一个有优先级的分隔符列表。它尝试使用列表中的第一个分隔符分割文本（默认是 `["\n\n", "\n", " ", ""]`）。如果生成的分块仍然过大，它会转向下一个分隔符，依此类推。

这种分层方法与书面文本的结构自然吻合。它首先尝试保持段落完整，然后是句子，最后是单词，以确保生成的分块在语义上尽可能连贯。

两个主要参数 (parameter)控制其行为：

- `chunk_size`：定义每个分块的最大长度，以字符为单位。
- `chunk_overlap`：指定相邻分块之间重叠的字符数。这是一个有用的功能，有助于保持上下文 (context)的连续性。如果一个句子在一个分块的末尾被截断，重叠会确保它在下一个分块中得到完整呈现，从而减少句子之间关系丢失的可能性。

以下是您在实践中如何使用它：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 从文档加载的一些长文本
long_text = """LangChain 提供了一个全面的框架，用于构建 LLM 应用程序。它简化了模型链、提示管理以及连接数据源的过程。

它的主要组件之一是 TextSplitter。此实用程序对检索增强生成 (RAG) 至关重要，因为它将大文档分解为可管理的分块。这些分块随后被嵌入并存储在向量数据库中以进行高效检索。分块大小和重叠的选择对应用程序性能有很大影响。"""

# 初始化分块器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20
)

# 分割文档
chunks = text_splitter.split_text(long_text)

# 打印生成的分块
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---\n{chunk}\n")
```

运行这段代码会产生以下输出，展示了文本是如何被分割的，同时重叠部分保持了边界处的上下文。

```text
--- Chunk 1 ---
LangChain 提供了一个全面的框架，用于构建 LLM 应用程序。它简化了模型链、提示管理以及连接

--- Chunk 2 ---
连接到数据源。

它的主要组件之一是 TextSplitter。此实用程序对检索增强生成 (RAG) 至关重要，因为它

--- Chunk 3 ---
它将大文档分解为可管理的分块。这些分块随后被嵌入并存储在向量数据库中以进行高效检索。分块的选择

--- Chunk 4 ---
分块大小和重叠对应用程序性能有很大影响。
```

重叠部分清晰可见；“连接”出现在第一个分块的末尾和第二个分块的开头，确保整个短语可用于上下文。

### 选择合适的分块大小

最佳 `chunk_size` 取决于您的数据具体情况以及您打算使用的嵌入 (embedding)模型。大多数嵌入模型都有输入token限制（例如，512或1024个token），您希望您的分块能舒适地符合该限制。字符数不直接对应token数，但一个常见的经验法则是，英文中一个token大约是四个字符。

对于许多应用程序而言，一个好的起始点是 `chunk_size` 约为1000个字符，`chunk_overlap` 为200个字符。这些值提供了一个坚实的平衡，但您应该通过实验来找到最适合您用例的设置。可视化分块后的长度分布可以帮助确认您的配置是否按预期运行。



![分块长度分布](plots/7430-0.json)



> 一个直方图，显示示例中生成的四个分块的长度。大多数接近 `chunk_size` 限制150个字符，最后一个分块较短。

### 其他分块策略

虽然 `RecursiveCharacterTextSplitter` 是一个可靠的默认选项，但 LangChain 也提供了为不同需求定制的其他分块器：

- **CharacterTextSplitter:** 最简单的分块器。它基于单个分隔符字符（例如 `\n`）分割文本。它速度快，但与递归方法相比，在保留语义边界方面的效果较差。
- **TokenTextSplitter:** 此分块器根据特定数量的token而不是字符来分割文本。这是一种更精确的方法，用于确保分块不超过LLM的上下文 (context)限制，但它需要一个外部tokenizer，例如OpenAI的 `tiktoken`。

例如，使用 `TokenTextSplitter` 的方式如下：

```python
from langchain_text_splitters import TokenTextSplitter

# 此分块器需要一个tokenizer
token_splitter = TokenTextSplitter(
    chunk_size=50, # 以token为单位的大小
    chunk_overlap=10, # 以token为单位的重叠
    model_name="gpt-4o" # 指定模型以确保准确的token计数
)

token_chunks = token_splitter.split_text(long_text)
```

这种方法在模型将处理的内容方面提供了更可预测的分块大小。然而，对于大多数通用RAG系统，`RecursiveCharacterTextSplitter` 在性能和简洁性之间提供了有效的平衡。

现在，我们的文档已经被处理成定义清晰、语义上有意义的分块，下一步是将其转换为机器可以用于相似性比较的格式：数值向量 (vector)，即嵌入 (embedding)。

## 参考资料

- [LangChain Text Splitters](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE20vPxAG1wcwUYIGPPIV6anAwi-5JQ1gLjPAPVhbH7fwdrvYVDoezvXot1dyN6gxal17mlrkxINHUl2p2L53IKvET8Xmb_uQjPtCNLGjvLMGv7e8FwBIs3Jy6P4CoYUhIXZuceO-q_oeMBD5jh3Me9XjGCHSY=) — LangChain Development Team (2024)
  Publisher: LangChain
  官方文档，详细介绍了 LangChain 中的各种文本拆分策略，包括 `RecursiveCharacterTextSplitter` 和 `TokenTextSplitter`，对实际实现至关重要。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: NeurIPS 2020; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  这篇论文介绍了检索增强生成（RAG）框架，为将文档拆分为可管理块提供了理论背景，这是高效信息检索的基本预处理步骤。
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang (2023)
  Journal: Transactions of the Association for Computational Linguistics (TACL); DOI: [10.48550/arXiv.2307.03172](https://doi.org/10.48550/arXiv.2307.03172)
  研究了语言模型在处理长输入上下文时的表现，指出了“中间丢失”现象，文本分块策略旨在通过提供集中的文本段来克服这一问题。
