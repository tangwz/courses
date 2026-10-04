---
course: "building-advanced-llm-agent-tools"
chapter: "advanced-tool-functionality"
lesson: "web-browsing-extraction-tools"
sourceId: 6602
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-5-advanced-tool-functionality/web-browsing-extraction-tools"
title: "开发网页浏览与内容提取工具"
description: "创建能让LLM智能体浏览网站、获取信息并与网页内容交互的工具。"
order: 2
plots: []
sourceHash: "f8320133f44585cf38d5dcc10a817335fe869154eafdd72ca05f8b142494bb54"
sourceCorrections: []
---

将LLM智能体配备网页浏览和信息获取能力，能使其从静态知识库转变为可访问和处理实时数据的动态实体。此能力对于需要最新信息、数据查阅或与基于网络的服务的交互的任务来说非常基本。构建高效的网页浏览和内容提取工具不仅仅是获取网页；它需要仔细考虑如何以对LLM有用且易于理解的方式获取、解析、清理和呈现网页内容。

### 网页交互的核心技术

从根本上说，网页浏览工具需要执行两个主要功能：获取网页内容并解析它以提取有意义的信息。有几个Python库常用于这些任务。

#### 使用 `requests` 获取网页内容

对于许多直接在服务器上渲染内容（静态网站）的网站，`requests` 库是一个很好的选择。它允许您向URL发送HTTP请求（GET、POST等），并接收原始HTML内容、JSON数据或其他资源。

```python
import requests

def fetch_url_content(url: str) -> str | None:
    try:
        response = requests.get(url, timeout=10) # 设置超时
        response.raise_for_status() # 对于不良响应（4XX或5XX）引发HTTPError
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
```

使用 `requests` 时，设定一个识别您的机器人的用户代理字符串是很必要的，并且要处理潜在的异常，例如网络错误、超时或HTTP错误状态。

#### 使用 `BeautifulSoup` 解析HTML

获取原始HTML后，您需要解析它以理解其结构并获取特定的信息片段。`BeautifulSoup` 是用于此目的的常用库。它可以解析格式不正确的HTML，并提供方便的方法来搜索和遍历解析树。

```python
from bs4 import BeautifulSoup

def extract_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 移除脚本和样式元素
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
        
    # 获取文本并进行清理
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text
```

此示例显示了提取所有文本内容的基本方法。为了更具针对性的提取，您可以使用 `BeautifulSoup` 的查找方法，配合CSS选择器或其他条件。

#### 使用无头浏览器处理动态内容

许多现代网站高度依赖JavaScript来动态加载和渲染内容。一个简单的 `requests.get()` 调用可能只返回初始HTML骨架，而缺少客户端脚本加载的实际内容。为了处理此类网站，您需要一个能够执行JavaScript的工具，本质上是一个没有图形用户界面运行的浏览器。`Playwright` 或 `Selenium` 等库可以控制无头浏览器（例如Chrome、Firefox、WebKit）。

使用无头浏览器比直接HTTP请求更占用资源，因此应在必要时才使用。一般的工作流程包括：

1. 导航到URL。
2. 等待特定元素加载或JavaScript执行完成。
3. 提取页面源（现在包含动态渲染的内容）或直接与元素交互以获取其文本或属性。

```python
# Playwright的简化流程
# from playwright.sync_api import sync_playwright

# def fetch_dynamic_content(url: str) -> str | None:
#     with sync_playwright() as p:
#         browser = p.chromium.launch()
#         page = browser.new_page()
#         try:
#             page.goto(url, wait_until="networkidle", timeout=30000) # 等待网络活动停止
#             content = page.content() # JS执行后获取完整的HTML
#         except Exception as e:
#             print(f"Error fetching {url} with Playwright: {e}")
#             content = None
#         finally:
#             browser.close()
#         return content
```

使用无头浏览器时，请注意设置中的额外复杂性，如果等待条件定义不明确可能导致的不稳定性，以及资源消耗的增加。

### 内容提取策略

仅仅获取页面通常是不够的。LLM需要相关信息，而不是整个HTML模板。

#### 使用选择器

CSS选择器和XPath表达式是定位页面上特定元素的有效方法。例如，您可能希望提取所有段落标签（`<p>`）、具有特定类名（例如`class="article-title"`）的元素，或特定ID标识的`<div>`内的内容。

```python
# BeautifulSoup支持CSS选择器：
# soup = BeautifulSoup(html_content, 'html.parser')
# headlines = [h.get_text() for h in soup.select('h2.headline-class')]
# main_content = soup.select_one('#main-article-div').get_text(separator='\n', strip=True) if soup.select_one('#main-article-div') else ""
```

#### 为LLM简化内容

原始HTML可能包含许多干扰信息。在将内容传递给LLM之前，考虑对其进行简化：

- \*\*转换为Markdown：\*\*像 `markdownify` 这样的库可以将HTML转换为Markdown，这通常对LLM来说更具可读性和简洁性。
- \*\*提取主要内容：\*\*实施启发式方法或使用 `trafilatura` 或 `goose3` 等库，这些库旨在从网页中提取主要文章文本，去除导航、广告和页脚。
- \*\*分块：\*\*如果提取的内容非常长，它可能会超出LLM的上下文 (context)窗口。将内容分成更小、连贯的块可能是必要的。

#### LLM辅助提取

对于非常复杂的提取任务，或者当目标网站的结构差异很大时，您甚至可以使用另一次LLM调用作为工具的一部分。浏览工具会获取内容，或许进行一些初步清理，然后将其传递给一个LLM，并附带特定的提示，以结构化格式获取所需的信息片段。这会增加延迟和成本，但能提供更大的灵活性。

### 设计面向LLM的网页浏览工具接口

LLM将使用特定参数 (parameter)调用您的网页浏览工具，并期待一个结构良好的响应。

#### 工具输入

网页浏览工具的常见输入包括：

- `url`：要获取的具体URL。
- `query`（可选）：如果工具集成了搜索功能（例如，首先使用搜索引擎API查找相关URL），这将是搜索词。
- `extraction_instructions`（可选）：关于要寻找哪些特定信息的自然语言提示或更结构化的请求（例如，“获取主要文章文本”、“查阅产品X的当前价格”）。这在您的工具中需要更复杂的解析逻辑。

#### 工具输出

输出应设计为供LLM使用：

- \*\*清理后的文本：\*\*包含相关提取文本的字符串，没有HTML标签和不必要的模板。
- \*\*结构化数据：\*\*如果获取特定字段（例如，产品名称、价格、作者），返回JSON对象或字典。
- \*\*摘要：\*\*对于长页面，工具可以在返回之前生成摘要（通过启发式方法或另一次LLM调用）。
- \*\*错误消息：\*\*如果URL无法获取、未找到内容或解析失败，则显示清晰的消息。

目标是为LLM提供可直接用于其任务的信息，最大限度地减少LLM自行解析复杂原始数据的需要。考虑您LLM的令牌限制；除非必要并有指示，否则避免返回过长的内容。

> LLM智能体使用网页浏览和内容提取工具的流程。

### 重要的操作考量

构建一个负责任的网页浏览工具需要关注几个操作细节。

#### 尊重 `robots.txt`

网站使用 `robots.txt` 文件来表明哪些网站部分不应被网络爬虫访问。您的工具应解析并遵守这些指令，以避免服务器过载或访问受限区域。Python中存在库（例如 `robotexclusionrulesparser`）来协助完成此任务。

#### 速率限制和用户代理

在短时间内向单个服务器发出过多请求可能导致您的IP被阻止。在您的工具中实施速率限制。始终在HTTP头中设置一个描述性的 `User-Agent` 字符串，以便在必要时网站管理员可以识别请求的来源。

#### 错误处理

网页交互容易出错：网络问题、超时、网站结构变化导致选择器失效，或内容根本不在您预期位置。您的工具必须妥善处理这些错误，并向LLM返回有用的消息，使其能够潜在地重试或尝试不同的方法。

#### 获取内容的安全性

如果您的工具处理或渲染从网络获取的内容，请务必小心，特别是使用无头浏览器等执行JavaScript的技术时。尽管主要目标是为LLM提取文本，但要确保您的工具环境不容易受到恶意脚本的攻击，例如，如果您尝试渲染页面的一部分或提取可能包含可执行代码的属性。沙盒策略，如针对代码执行工具讨论的那样，如果与网页内容的交互变得非常复杂，则可能适用。通常，对于文本提取，只要您在解析而非在特权上下文 (context)中执行任意获取的脚本，风险就较低。

通过开发网页浏览和内容提取工具，您能显著增强LLM智能体收集信息、进行数据查阅和保持信息更新的能力。这些工具连接了LLM的训练知识与互联网上动态变化的信息之间的空白。

## 参考资料

- [Requests: HTTP for Humans™](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGhuWWFEr3iXeVXOqpA0PwkWa2HsTdNC7IyW2tuWweQlNpo53pPzE1BqOKfxCfSmafaRMuseBER5yIY6uDsqxyb5nZPI2XEGDVneiXY7eJAZO_xwyBUl6LazbQ=) — Kenneth Reitz (Ongoing)
  Python 'requests' 库的官方文档，对于通过 HTTP 请求获取网页内容至关重要。
- [Beautiful Soup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) — Leonard Richardson (Ongoing)
  'BeautifulSoup' 的官方指南，该库是用于解析 HTML 和 XML 文档的常用 Python 库。
- [Playwright Documentation](https://playwright.dev/python/docs/intro) — Microsoft (Ongoing)
  Playwright 的官方文档，涵盖了其在浏览器自动化和处理动态网页内容方面的应用。
- [Web Scraping with Python: Collecting Data from the Modern Web](https://www.oreilly.com/library/view/web-scraping-with/9781491998520/) — Ryan Mitchell (2018)
  Publisher: O'Reilly Media
  一本全面涵盖网络爬虫各个方面的书籍，包括道德考量、解析和处理动态内容。它是网络爬虫实践者的基础指南。
