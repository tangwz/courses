---
course: "building-llm-apps-with-langchain"
chapter: "data-connection-rag"
lesson: "document-loaders"
sourceId: 7429
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-5-data-connection-rag/document-loaders"
title: "使用文档加载器加载数据"
description: "学习如何使用LangChain的文档加载器从文本文件、PDF、网站和数据库等各种来源摄取数据。"
order: 2
plots: []
sourceHash: "1431bc5201eaae2190f0ebab25a8b261c1a621e7edd5e1caccb342c85ff8feb2"
sourceCorrections: []
---

任何RAG系统的根本在于其所依赖的数据。在大型语言模型（LLM）能够处理您的信息之前，这些信息必须以系统可理解的方式加载和组织。LangChain的`DocumentLoader`组件负责这一初步步骤，即数据摄取。它们提供了一种标准化的方式，可以从多种来源导入数据，将原始内容转换为统一的格式，供框架其余部分处理。

### LangChain的文档抽象

LangChain数据处理的核心是`Document`对象。可以把它看作是一个标准化的容器，用于存放一段文本及其相关信息。每个`Document`对象都有两个主要属性：

1. `page_content`：一个字符串，包含实际的文本内容。
2. `metadata`：一个Python字典，包含内容的补充信息，例如其来源、页码或创建日期。这些元数据对于在最终应用中进行筛选、追踪出处和提供引用非常有用。

下面是如何手动创建一个`Document`对象的方法：

```python
from langchain_core.documents import Document

doc = Document(
    page_content="LangChain provides a standard interface for document loaders.",
    metadata={"source": "internal_docs/intro.md", "chapter": 1}
)

print(doc.page_content)
# LangChain provides a standard interface for document loaders.

print(doc.metadata)
# {'source': 'internal_docs/intro.md', 'chapter': 1}
```

尽管可以手动创建`Document`对象，但真正的便利之处在于使用`DocumentLoaders`从数据源自动生成它们。`DocumentLoader`是一个对象，旨在从源获取数据并将其转换为`Document`对象列表。

> 文档加载器作为一个统一接口，将各种数据源转换为标准化的`Document`对象列表，以供后续处理。

### 从常见来源加载

LangChain提供多种文档加载器，每个都针对特定数据源进行定制。我们来看几个最常用的。

#### 文本文件

最简单的情况是从纯文本`.txt`文件加载数据。为此，我们使用`TextLoader`。

首先，我们创建一个示例文件。

```bash
# 在你的终端中
echo "This is the first sentence. This is the second." > sample.txt
```

现在，我们可以在Python中使用`TextLoader`加载它。`.load()`方法执行操作并返回`Document`对象列表。对于`TextLoader`，这通常是一个只包含整个文件内容的`Document`对象的列表。

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.txt")
docs = loader.load()

print(len(docs))
# 1

print(docs[0].page_content)
# This is the first sentence. This is the second.

print(docs[0].metadata)
# {'source': 'sample.txt'}
```

#### PDF 文档

一个常见需求是从PDF文件摄取数据。`PyPDFLoader`集成通过加载PDF并按页拆分其内容来处理此问题，为每个页面创建一个`Document`对象。这是一个合理的默认设置，因为它会自动保留页面边界。

假设您有一个名为`product_manual.pdf`的PDF文件：

```python
# 您可能需要安装pypdf库：pip install pypdf
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("product_manual.pdf")
pages = loader.load()

# 假设PDF有多个页面
print(f"从PDF加载了{len(pages)}页。")

# 检查第一页的文档
first_page_doc = pages[0]
print(first_page_doc.page_content[:200]) # 打印前200个字符
# ...第一页的内容...

print(first_page_doc.metadata)
# {'source': 'product_manual.pdf', 'page': 0}
```

注意元数据现在包含一个`page`键，这对于引用原始来源非常有用。

#### 网页

要直接从URL加载内容，可以使用`WebBaseLoader`。它从给定URL获取HTML，解析它，并将文本内容提取到`Document`中。这是获取在线文章、博客文章或文档的一种高效方式。

```python
# 您可能需要安装BeautifulSoup：pip install beautifulsoup4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.oreilly.com/about/")
docs = loader.load()

print(docs[0].page_content[:500])
# ...网页内容的纯文本版本...

print(docs[0].metadata)
# {'source': 'https://www.oreilly.com/about/', 'title': 'About O’Reilly - O’Reilly'}
```

`WebBaseLoader`的元数据通常包括源URL和页面标题，这使您可以引用原始内容。

### 加载整个目录

在许多应用程序中，您需要从一个目录加载所有文档。`DirectoryLoader`提供了一种便捷的方式来完成此操作。您需要指定一个路径和一个`glob`模式来选择要加载的文件。您还必须提供应用于这些选定文件的加载器类。

例如，要使用`TextLoader`从`documentation/`文件夹加载所有Markdown文件（`.md`）：

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 假设documentation/包含多个.md文件
loader = DirectoryLoader(
    path="documentation/", 
    glob="**/*.md",          # 加载所有子目录中的所有.md文件
    loader_cls=TextLoader,   # 对这些文件使用TextLoader
    show_progress=True       # 显示进度条
)

docs = loader.load()
print(f"从目录加载了{len(docs)}个文档。")
```

这种将`DirectoryLoader`与特定文件加载器结合的模式是批量数据摄取的一种强大而高效的方法。

现在，我们的数据已加载到`Document`对象中，我们面临一个新的挑战。单个文档，例如PDF中的整章或一篇长网页文章，很容易超出大多数语言模型的上下文 (context)窗口。为了解决这个问题，我们必须将这些大型文档拆分为更小、语义连贯的片段。这个过程，称为文本拆分，是我们下一节的重点。

## 参考资料

- [LangChain Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/) — LangChain (2024)
  Publisher: LangChain
  LangChain 官方文档，详细介绍了各种 DocumentLoader 类及其在 LangChain 中用于数据摄取的方法。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; Pages: 9449-9463; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  一篇介绍检索增强生成（RAG）范式的基础论文，为需要数据摄取系统的顶层设计提供支持。
- [Document Object API Reference](https://api.python.langchain.com/en/stable/documents/langchain_core.documents.base.Document.html) — LangChain (2024)
  LangChain 官方 Document 对象 API 参考，详细说明了其 page_content 和 metadata 等属性。
- [Data Engineering for AI: A Practical Guide to Building Data Pipelines for LLMs, Generative AI, and Machine Learning](https://www.oreilly.com/library/view/data-engineering-for/9781098150488/) — Andrew C. Ng (2024)
  Publisher: O'Reilly Media
  涵盖 AI 系统的数据工程原则，包括数据收集、预处理和构建与 LLM 应用相关的实践方法。
