---
course: "getting-started-rag"
chapter: "data-preparation-for-rag"
lesson: "loading-documents"
sourceId: 3645
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-3-data-preparation-for-rag/loading-documents"
title: "从不同来源加载文档"
description: "涵盖从不同文件类型（PDF、TXT、HTML）和来源加载数据的方法。"
order: 1
plots: []
sourceHash: "03ed6e3f22451692c45f45daf5760c3f1302a4cd823df4c0657d251271c0a790"
sourceCorrections: []
---

您的RAG系统的效力直接取决于您为其提供的知识的质量和可获取性。这些知识很少以完全准备好的状态出现，它们通常存在于多种文件格式和位置。准备数据的第一个实际步骤是*摄取*或*加载*：将这些不同来源的原始文本内容引入您的处理流程。

将此阶段视为收集原始材料。在提炼黄金之前，您首先需要开采矿石。同样，在为检索分块和嵌入 (embedding)文本之前，您必须首先从其容器中提取该文本，无论是简单文本文件、结构化PDF还是动态网页。

### 处理纯文本文件 (.txt)

纯文本文件通常是最简单的处理格式。标准Python库提供了直接的方法来读取它们的内容。

```python
def load_text_file(file_path):
    """从文本文件中读取内容。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"错误：文件未在 {file_path} 找到")
        return None
    except Exception as e:
        print(f"读取文件 {file_path} 时出错：{e}")
        return None

# 使用示例：
file_path = 'documents/my_document.txt'
text_content = load_text_file(file_path)

if text_content:
    print(f"成功从 {file_path} 加载内容")
    # 继续处理text_content...
```

在使用文本文件时，一个重要考虑点是字符编码。虽然UTF-8是广泛接受的标准，但您可能会遇到以不同编码（如`latin-1`或`cp1252`）保存的文件。如果您看到乱码文本或遇到`UnicodeDecodeError`，您可能需要在打开文件时指定正确的编码。确定正确的编码有时需要检查或反复尝试。

### 从PDF (.pdf) 中提取文本

PDF（便携式文档格式）文件普遍用于共享文档并保留布局，但以编程方式从中提取文本可能比从纯文本文件复杂。PDF可以包含文本、图片、矢量图和复杂的布局信息。

像`pypdf`（一个常用Python库，用于处理PDF）或LangChain和LlamaIndex等框架中更高级的工具可以简化文本提取。

```python
# 以pypdf为例（安装命令：pip install pypdf）
from pypdf import PdfReader

def load_pdf_text(file_path):
    """从PDF文件中提取文本内容。"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: # 检查文本提取是否成功
                text += page_text + "\n" # 在页之间添加换行符
        return text.strip() # 移除开头/结尾的空白字符
    except FileNotFoundError:
        print(f"错误：文件未在 {file_path} 找到")
        return None
    except Exception as e:
        print(f"读取PDF文件 {file_path} 时出错：{e}")
        return None

# 使用示例：
pdf_path = 'documents/research_paper.pdf'
pdf_text_content = load_pdf_text(pdf_path)

if pdf_text_content:
    print(f"成功从 {pdf_path} 提取文本")
    # 继续处理pdf_text_content...
```

PDF的难题包括：

- **扫描文档：** 有些PDF实质上是文本图片。提取文本需要光学字符识别（OCR）软件（如Tesseract，可通过库集成），这增加了另一个处理层面和潜在的不准确性。
- **复杂布局：** 多列布局、表格、页眉/页脚和图表有时会混淆文本提取逻辑，导致混乱或不完整的文本。如果结构化数据很重要，您可能需要更复杂的解析逻辑或专门用于表格提取的库。
- **密码保护：** 加密或密码保护的PDF在没有正确密码的情况下无法读取。

### 获取网页内容 (HTML)

网页是另一种常见的信息来源。加载需要两个步骤：从URL获取HTML内容，然后解析HTML以提取相关文本，并丢弃导航菜单、广告和页脚等样板内容。

Python的`requests`库常用于获取网页内容，而像`BeautifulSoup`这样的库则擅长解析HTML。

```python
# requests和BeautifulSoup使用示例
# 安装命令：pip install requests beautifulsoup4
import requests
from bs4 import BeautifulSoup

def load_web_page_text(url):
    """从网页URL获取并提取文本内容。"""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}) # 做一个有礼貌的爬虫
        response.raise_for_status() # 对于错误的HTTP状态码（4xx或5xx）抛出异常

        soup = BeautifulSoup(response.content, 'html.parser')

        # 基本文本提取（可以进行大量优化）
        # 移除脚本和样式元素
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # 获取文本，使用分隔符确保元素之间有空格
        text = soup.get_text(separator=' ', strip=True)
        return text

    except requests.exceptions.RequestException as e:
        print(f"获取URL {url} 时出错：{e}")
        return None
    except Exception as e:
        print(f"从 {url} 解析HTML时出错：{e}")
        return None

# 使用示例：
web_url = 'https://example-documentation-site.com/page'
web_text_content = load_web_page_text(web_url)

if web_text_content:
    print(f"成功从 {web_url} 加载文本")
    # 继续处理web_text_content...
```

网页抓取可能很脆弱。网站结构会改变，仅提取主要文章内容通常需要通过检查页面源代码来特定定位HTML标签（如`<article>`、`<main>`或特定的CSS类）。请注意网站的`robots.txt`文件和关于自动化访问的服务条款。

### 其他格式和文档加载器的想法

您可能会遇到其他格式，如Microsoft Word（`.docx`）、Markdown（`.md`）、JSON或CSV文件，它们包含与您的RAG系统相关的文本数据。对于大多数常见格式，存在Python库来解析它们（例如，`python-docx`用于Word文档，`json`用于JSON，`csv`用于CSV）。

一般模式保持不变：

1. 确定文件格式。
2. 找到合适的库来读取/解析该格式。
3. 提取核心文本内容。
4. 处理潜在错误（文件未找到、解析错误、编码问题）。

许多RAG框架（如LangChain和LlamaIndex）提供便捷的抽象，称为**文档加载器**。这些加载器封装了读取不同数据源（文件、网页、数据库、Notion或Slack等API）的逻辑，并将它们转换为标准化的`Document`对象。`Document`通常包含提取的文本（`page_content`）和相关的元数据（例如，`source`文件路径或URL、页码、章节）。

```python
# 示例（语法可能因框架而异）
# from rag_framework.document_loaders import PyPDFLoader, WebBaseLoader

# pdf_loader = PyPDFLoader("documents/research_paper.pdf")
# web_loader = WebBaseLoader("https://example-documentation-site.com/page")

# pdf_documents = pdf_loader.load() # 返回Document对象列表
# web_documents = web_loader.load() # 返回Document对象列表

# print(pdf_documents[0].page_content)
# print(pdf_documents[0].metadata) # {'source': 'documents/research_paper.pdf', 'page': 0}
```

使用这些加载器可以通过为许多常见数据源提供预构建的集成来大幅加快开发。然而，理解文件I/O和解析的原理（如上所述）对于故障排除和处理不常见或自定义格式很有用。

无论使用何种方法，此加载阶段的目标是将您的不同知识来源转换为一致的内存表示，通常是字符串或这些`Document`对象，为下一个重要步骤——分块做好准备。

## 参考资料

- [Built-in Functions - open()](https://docs.python.org/3/library/functions.html#open) — Python Software Foundation (2024)
  Python `open()` 函数的官方文档，详细说明文件处理和字符编码选项。
- [pypdf Documentation](https://pypdf.readthedocs.io/en/stable/) — pypdf developers (2024)
  `pypdf` 库的全面文档，包含从 PDF 文件提取文本的示例。
- [Beautiful Soup 4 Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) — Leonard Richardson (2024)
  Beautiful Soup 4 的使用指南，用于解析 HTML 和 XML 文档以提取内容。
- [Document loaders](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG3YMikPdRYPEydmJbfJCNK_A8irKr3slflE11dAO1qZxhoJrN9bpZ-thLOnXKd68vwJQrSDANHjSyiXA9dG6fM1YzTwprBx0Pog69lx3IXNl5VZTEgsut57-I6EEvv22NZODZNqk6fGO77Oh2J_QO0pV3DSFyrX9Irbg==) — LangChain (2024)
  描述 LangChain 的文档加载系统，支持 RAG 应用程序的多种数据源。
