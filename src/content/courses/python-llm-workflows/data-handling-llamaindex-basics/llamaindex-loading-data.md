---
course: "python-llm-workflows"
chapter: "data-handling-llamaindex-basics"
lesson: "llamaindex-loading-data"
sourceId: 5172
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-6-data-handling-llamaindex-basics/llamaindex-loading-data"
title: "加载数据 (文档、网页)"
description: "使用 LlamaIndex 数据加载器从各类来源摄取信息。"
order: 2
plots: []
sourceHash: "5c38c865d2cc0bf4b1ee72dbed1eb6b871a5b770d08fab84efb8e9929c080b48"
sourceCorrections: []
---

在大型语言模型 (LLM) 使用你的特定数据之前，这些数据必须先转换为 LlamaIndex 能理解的格式。这一第一步是将原始信息（无论是简单的文本文件、PDF 文档、网页，甚至是数据库）从其来源转换为标准化表示。LlamaIndex 通过通常称为 `Readers` (读取器) 或 `Data Loaders` (数据加载器) 的组件来处理这个数据摄取过程。

LlamaIndex 中数据的基本单位是 `Document` 对象。`Document` 通常包含从源中提取的文本内容，以及相关的元数据（例如文件名或 URL）。读取器的主要作用是从特定来源获取输入数据，并生成一个或多个 `Document` 对象。

让我们看看如何从常见来源加载数据。

### 使用读取器加载数据

读取器隐藏了访问和解析不同文件类型及数据源的复杂性。LlamaIndex 为许多常见格式提供了内置读取器，并在 LlamaIndex Hub 中维护了一个更大的集合，用于更专用来源。

一般模式包括导入相应的读取器，实例化它（通常指向数据源），然后调用其 `load_data()` 方法。

```python
# 示例结构（具体读取器不同）
from llama_index.core import SimpleDirectoryReader # 或其他特定读取器

# 实例化读取器，指向数据源
# 对于基于文件的读取器，这通常是目录路径
reader = SimpleDirectoryReader("./data_directory")

# 将数据加载到 Document 对象中
documents = reader.load_data()

# 现在 'documents' 是一个 LlamaIndex Document 对象的列表
print(f"Loaded {len(documents)} document(s).")
# 示例输出：访问第一个文档的文本
if documents:
    print(f"First document text snippet: {documents[0].text[:100]}...")
```

### 读取本地文件 (`SimpleDirectoryReader`)

最常见的任务之一是从本地机器上存储的文件加载数据。`SimpleDirectoryReader` 是一个处理此任务的多功能工具。默认情况下，只要安装了必要的底层库，它就可以处理各种文件类型，例如 `.txt`、`.pdf`、`.docx`、`.json`、`.md` 等。

```python
# 确保特定文件类型可能需要必要的库
# pip install llama-index pypdf # 例如：支持 PDF

from llama_index.core import SimpleDirectoryReader

# 将 SimpleDirectoryReader 指向包含你文档的文件夹
reader = SimpleDirectoryReader("./path/to/your/docs")
documents = reader.load_data()

print(f"Successfully loaded {len(documents)} documents from the directory.")
```

`SimpleDirectoryReader` 会遍历指定目录。对于它找到的每个受支持文件，它会提取文本内容并创建一个 `Document` 对象。文档的元数据通常会自动包含文件路径和名称。你可以配置它以递归搜索子目录或过滤特定文件扩展名。

### 加载网页

为了直接从网页摄取内容，LlamaIndex 提供了专为网页抓取设计的读取器。一个常用选择是 `SimpleWebPageReader`（或者有时是其他使用 `BeautifulSoup` 等库的读取器或框架中的集成功能）。

通常你需要安装用于网络请求和 HTML 解析的库。

```python
# 安装必要的库
# pip install llama-index beautifulsoup4

from llama_index.core import SimpleWebPageReader

# 要加载的 URL 列表
urls = ["https://www.example.com/page1", "https://www.anothersite.org/article"]

# 实例化读取器
# 设置 html_to_text=True 以尝试进行基本的 HTML 标签移除
reader = SimpleWebPageReader(html_to_text=True)

# 从 URL 加载数据
# 注意：这将返回一个 Document 列表，每个 URL 对应一个
web_documents = reader.load_data(urls)

print(f"Loaded content from {len(web_documents)} web page(s).")
if web_documents:
    print(f"Snippet from first web page: {web_documents[0].text[:150]}...")
```

> 网页读取器从每个 URL 获取 HTML 内容，可选地处理它以提取主要文本内容，并为每个页面创建一个 `Document` 对象。URL 通常存储在文档的元数据中。

### 图示：数据加载过程

> 来自各类来源（如文本文件、PDF 或网页）的数据由相应的 LlamaIndex 读取器（例如 `SimpleDirectoryReader`、`SimpleWebPageReader`）处理，以创建包含文本和元数据的标准化 `Document` 对象。

### 处理多样化数据源

LlamaIndex 支持除简单文件和网页之外的多种数据源。通过内置读取器以及 LlamaIndex Hub（一个社区注册中心）上提供的丰富集合，你可以连接到：

- 数据库 (SQL, NoSQL)
- API (Slack, Google Drive 等)
- 结构化数据格式 (CSV, JSON)
- 音频和视频 (需要转录服务)

查看 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/module_guides/loading/connector.html) 或 [LlamaIndex Hub](https://llamahub.ai/) 可以看到可用连接器的广度。安装和使用这些通常遵循类似模式：安装依赖项，导入特定的读取器，使用必要的配置（如 API 密钥或连接字符串）实例化它，然后调用 `load_data()`。

这个加载阶段的结果是一致的：一个 `Document` 对象列表。这些对象作为标准化输入，用于为 LLM 准备数据的下一个重要步骤：索引。

## 参考资料

- [LlamaIndex Documentation: Data Connectors](https://docs.llamaindex.ai/en/stable/module_guides/loading/data_connectors.html) — LlamaIndex (2024)
  LlamaIndex数据加载器的官方指南，包括`SimpleDirectoryReader`和`SimpleWebPageReader`，以及连接到各种数据源。
- [LlamaIndex Documentation: Document](https://docs.llamaindex.ai/en/stable/module_guides/concepts/document.html) — LlamaIndex (2024)
  定义了LlamaIndex中的`Document`对象、其结构和元数据处理。
- [LlamaHub: Community Data Loaders](https://llamahub.ai/) — LlamaIndex (2024)
  一个社区贡献的数据加载器仓库，用于将各种数据源与LlamaIndex集成。
