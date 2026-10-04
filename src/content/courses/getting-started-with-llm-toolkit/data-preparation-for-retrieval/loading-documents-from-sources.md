---
course: "getting-started-with-llm-toolkit"
chapter: "data-preparation-for-retrieval"
lesson: "loading-documents-from-sources"
sourceId: 7764
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-4-data-preparation-for-retrieval/loading-documents-from-sources"
title: "从不同来源加载文档"
description: "使用 document 模块加载和处理来自 PDF、TXT 等文件以及网页的数据。"
order: 2
plots: []
sourceHash: "e02e761c65ece6e260597f2c090b9ccf769e3d521974e8e5648b38d900d14c42"
sourceCorrections: []
---

构建检索增强生成 (RAG) 系统时，首要步骤是加载源信息。这些数据可能存在多种格式：纯文本文件、结构化Markdown文档、网页，甚至CSV文件中的表格数据。为了给大型语言模型 (LLM) 准备这些多样化的信息，首先需要将其转换为一致的标准化格式。

这就是 `document` 模块的作用。它提供了一种直接的方法，可以从不同来源摄取内容，并将其表示为通用的 `Document` 对象。此对象作为一个标准容器，存储文本内容和任何相关联的元数据，从而简化了后续的分块、嵌入 (embedding)和检索步骤。

### Document 对象：一个标准容器

可以将 `Document` 对象看作一个包装器，它将原始数据标准化。无论原始来源是简单的 `.txt` 文件还是复杂的 `.json` 文件，处理流程都将其视为一个 `Document`。此对象主要由两部分组成：

- **content**：一个字符串，存储源材料的完整文本。
- **metadata**：一个字典，包含额外信息，例如文件路径、格式或加载期间提取的结构化数据。

这种抽象使得 RAG 处理流程的其余部分能够在一致的数据结构上运行，而无需了解每种原始文件格式的具体细节。

> Document 对象标准化了摄取的数据，将原始文本与其相关联的元数据分离。

### 从常见文件格式加载

加载文件最便捷的方式是使用 `load_document()` 函数。它会自动根据文件扩展名检测文件类型并使用相应的加载器，使你的数据摄取代码整洁简单。

#### 纯文本和 Markdown

对于非结构化或半结构化文本，加载器会提取内容和任何可用的元数据。我们从一个纯文本文件 `sample.txt` 开始。

```python
from kerb.document import load_document

# 假设 'sample.txt' 存在于你的项目目录中
doc = load_document("sample.txt")

print(f"内容预览: {doc.content[:50]}...")
print(f"元数据: {doc.metadata}")
```

输出会显示文本内容和包含文件源路径的元数据。

Markdown 文件也以类似方式处理，但有一个额外优势：加载器会自动解析 YAML frontmatter。这是一种常见的模式，用于将标题、作者或标签等元数据直接存储在文档中。

考虑一个名为 `article.md` 的文件，其内容如下：

```markdown
---
title: RAG Systems Explained
author: AI Assistant
tags: [rag, llm, python]
---

# Introduction to RAG

Retrieval-Augmented Generation is a technique...
```

当你加载此文件时，frontmatter 会被提取到 `metadata` 字典中。

```python
from kerb.document import load_markdown

md_doc = load_markdown("article.md")

print(f"标题: {md_doc.metadata['frontmatter']['title']}")
print(f"作者: {md_doc.metadata['frontmatter']['author']}")
print(f"标签: {md_doc.metadata['frontmatter']['tags']}")
```

这种自动元数据提取对于构建复杂的 RAG 系统很有意义，因为它让你能够在处理流程的后期根据文档属性进行过滤或搜索。

#### JSON 和 CSV

尽管 RAG 系统通常与非结构化文本相关联，但它们也能从 JSON 或 CSV 文件等结构化来源获取数据。这些格式的加载器会将结构化数据转换为 `content` 字段的人类可读字符串，同时在 `metadata` 中保留原始结构。

加载 JSON 文件时，原始字符串放置在 `content` 中，解析后的字典则存储在 `metadata['parsed_content']` 中。

```python
from kerb.document import load_json

# 假设 'product.json' 包含 {"name": "AI Toolkit", "version": "1.2"}
json_doc = load_json("product.json")

print(f"内容:\n{json_doc.content}")
print(f"解析后的名称: {json_doc.metadata['parsed_content']['name']}")
```

类似地，对于 CSV 文件，加载器会将表格转换为 `content` 的字符串，并用标题和行列表丰富 `metadata`，其中每行是一个字典。

```python
from kerb.document import load_csv

# 假设 'data.csv' 包含列：id, category, relevance
csv_doc = load_csv("data.csv")

print(f"标题行: {csv_doc.metadata['headers']}")
print(f"行数: {csv_doc.metadata['num_rows']}")
print(f"第一行: {csv_doc.metadata['rows'][0]}")
```

这种双重表示为你提供了灵活性，可以将文本用于语义检索，同时保留结构化数据用于过滤或其他处理任务。

### 处理网页和内存数据

你的数据不总是存在于本地文件中。它可能来自网页、数据库查询或 API 响应。在这些情况下，你可以以编程方式创建 `Document` 对象。

例如，要处理网页，你首先会使用 `requests` 等库获取其 HTML 内容。然后，你会使用 `preprocessing` 模块中的函数清理 HTML 并提取主要文本。最后，你会用此文本实例化一个 `Document` 对象。

```python
from kerb.document import Document, DocumentFormat
# from kerb.preprocessing import extract_text_from_html
# import requests # This is an external library

# 步骤 1：获取网页内容（使用外部库）
# html_content = requests.get("https://example.com/article").text

# 步骤 2：提取干净文本（使用我们稍后会介绍的函数）
# clean_text = extract_text_from_html(html_content)
clean_text = "这是从网页中提取的干净文本。"

# 步骤 3：以编程方式创建 Document 对象
web_doc = Document(
    content=clean_text,
    metadata={
        "source": "https://example.com/article",
        "fetch_date": "2024-10-26"
    },
    format=DocumentFormat.HTML
)

print(f"来源: {web_doc.metadata['source']}")
print(f"内容预览: {web_doc.content[:50]}...")
```

这种模式非常通用。任何可以表示为字符串的数据都可以加载到 `Document` 中，使其成为你 RAG 处理流程的通用入口。

原始数据现已加载到标准化的 `Document` 对象中，下一个难题是将这些文档分解成更小、更易于管理的部分，以适合大型语言模型使用。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems Foundation, Inc.; Volume: 33; Pages: 9459-9474; DOI: [10.55917/cb.2023-1123](https://doi.org/10.55917/cb.2023-1123)
  介绍了检索增强生成（RAG）架构，这是理解本节讨论的系统数据加载的基础。
- [Document loaders](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEzdMCI93yl6HbMrDP04-mSNbQHIZpFnCGnu_LcGY1auRMr4ZoasDUsqK9jmx86TNpoPzik4oBq4z2cbLZjMLOnenyvd-Bc0NTnQvqddPoCwYT8UUl2tf0VD5OzaMYPGf8nlvyMWZl6hTD5nmBGoXCcQFPzU8CjXhV-gew=) — LangChain (2024)
  Publisher: LangChain
  官方文档，涵盖LangChain如何加载多样化文档类型并将其表示为`Document`对象，这是RAG系统中的常见方法。
- [Ingestion Pipeline](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/) — LlamaIndex (2024)
  Publisher: LlamaIndex
  官方指南，详细介绍了LlamaIndex的数据摄取过程，展示了原始数据如何转换为`Node`对象（类似于`Document`对象）并提取元数据用于RAG。
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — Daniel Jurafsky and James H. Martin (2023)
  Publisher: Online (3rd Edition Draft); Pages: Chapter 10
  权威NLP教材的第10章“信息抽取”，讨论从文本中抽取结构化信息和元数据的方法，与RAG数据准备相关。
