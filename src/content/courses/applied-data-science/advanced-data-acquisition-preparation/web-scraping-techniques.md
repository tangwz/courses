---
course: "applied-data-science"
chapter: "advanced-data-acquisition-preparation"
lesson: "web-scraping-techniques"
sourceId: 1832
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-1-advanced-data-acquisition-preparation/web-scraping-techniques"
title: "网页抓取结构化数据的技巧"
description: "学习负责任的网页抓取方法，从HTML网站提取数据。"
order: 3
plots: []
sourceHash: "a47f152cc0a2ad8eba6d9c932e587c3d97e0cd7879afb603aa59184816837941"
sourceCorrections: []
---

API提供了结构化数据访问，数据库保存了整理好的信息，但有时您需要的数据直接存在于网页上，而没有专门的API。网页抓取是指通过编程方式从HTML网站提取这些信息的行为。这是一种有效的数据收集方法，适用于无法通过其他途径获取数据的情况，例如电子商务网站上的商品价格、新闻门户的文章或公共网页表格中的统计数据。

然而，抓取伴随着重大责任。在尝试抓取任何网站之前，您必须考虑其道德和法律层面的影响。

### 道德与法律考量

1. **检查 `robots.txt`**：大多数网站在其域名根目录下都有一个名为`robots.txt`的文件（例如，`www.example.com/robots.txt`）。此文件规定了自动化爬虫（机器人）的规则，指明网站的哪些部分不应被访问。务必遵守这些规则。忽略`robots.txt`可能导致您的IP地址被阻止。
2. **审查服务条款（ToS）**：网站的服务条款或使用政策通常明确规定了自动化访问和数据提取的规则。违反服务条款进行抓取可能会产生法律后果。
3. **保持尊重**：抓取可能给目标网站的服务器带来负担。请以合理的速率发送请求（在请求之间加入延迟），在请求头中使用适当的User-Agent字符串标识您的机器人，并尽可能避免在高峰流量时段进行抓取。不要使服务器过载；对待它就像您手动浏览一样，只是速度更快。过度抓取可能被视为拒绝服务攻击。

未能负责任地抓取可能会损害网站所有者，并可能导致法律诉讼或永久性阻止。始终将道德考量放在首位。

### 理解用于抓取的HTML结构

为了有效抓取，您需要对网页如何使用HTML（超文本标记 (token)语言）进行结构化有一个基本理解。HTML使用标签来定义页面上的元素。数据通常嵌套在这些标签中。重要事项包括：

- **标签**：定义标题（`<h1>`、`<h2>`）、段落（`<p>`）、链接（`<a>`）、列表（`<ul>`、`<ol>`、`<li>`）、表格（`<table>`、`<tr>`、`<td>`、`<th>`）和通用容器（`<div>`、`<span>`）等元素。
- **属性**：提供关于标签的额外信息。常见属性包括`id`（元素的唯一标识符）、`class`（用于分组元素以进行样式设置或选择的非唯一标识符）和`href`（链接的URL）。

例如，考虑以下表示表格行的一个简单HTML片段：

```html
<tr class="data-row">
  <td id="item-name">产品A</td>
  <td class="price">$19.99</td>
  <td><a href="/details/item1">详情</a></td>
</tr>
```

这里，我们有一个带有`data-row`类的表格行（`<tr>`）。它包含表格数据单元格（`<td>`）。其中一个单元格具有唯一的`id`，另一个具有`class`，最后一个包含带有`href`属性的链接（`<a>`）。理解这种结构使您能够定位特定的信息片段。

### 用于网页抓取的Python库

有几个Python库有助于网页抓取：

1. **Requests**：用于向Web服务器发送HTTP请求并获取网页的原始HTML内容。
2. **Beautiful Soup 4 (bs4)**：用于解析HTML（和XML）文档的主力工具。它从页面源代码创建解析树，可用于遍历、搜索和修改该树，从而方便地提取数据。
3. **Scrapy**：一个更全面的框架，用于构建大规模抓取项目（爬虫）。它更全面地处理请求、解析、数据提取和数据保存，但学习曲线较陡。对于从特定页面提取结构化数据，Requests和Beautiful Soup通常就足够了。

我们来着重讲解`Requests`和`Beautiful Soup`的使用。

### 使用Requests获取网页内容

首先，您需要获取HTML内容。`requests`库让这变得简单。

```python
import requests

# 定义您想要抓取的页面URL
url = 'https://example.com/data-page' # 替换为真实的URL（请遵守robots.txt！）

# 定义User-Agent头以标识您的请求
headers = {
    'User-Agent': 'MyDataScienceBot/1.0 (contact@example.com)' # 请如实填写！
}

try:
    # 发送HTTP GET请求
    response = requests.get(url, headers=headers, timeout=10) # 添加超时

    # 检查请求是否成功（状态码200）
    response.raise_for_status() # 对于错误状态码（4xx或5xx）抛出异常

    # 获取HTML内容
    html_content = response.text
    print("成功获取HTML内容。")
    # print(html_content[:500]) # 打印前500个字符以检查

except requests.exceptions.RequestException as e:
    print(f"请求{url}时出错: {e}")
    html_content = None # 如果请求失败，确保html_content为None
```

始终包含错误处理（`try...except`）并检查响应状态码。User-Agent有助于标识您的脚本，这被认为是良好实践。

### 使用Beautiful Soup解析HTML并提取数据

一旦您有了`html_content`，就可以使用Beautiful Soup来解析它。

```python
from bs4 import BeautifulSoup

if html_content: # 仅在获取成功时继续
    # 创建BeautifulSoup对象，指定解析器
    soup = BeautifulSoup(html_content, 'html.parser')

    # --- 示例：查找页面标题 ---
    page_title = soup.title.string if soup.title else "未找到标题"
    print(f"页面标题: {page_title}")

    # --- 示例：查找第一个标题 (h1) ---
    first_h1 = soup.find('h1')
    if first_h1:
        print(f"第一个H1: {first_h1.get_text(strip=True)}")

    # --- 示例：查找所有段落 (<p>) ---
    # paragraphs = soup.find_all('p')
    # print(f"\n找到 {len(paragraphs)} 个段落:")
    # for i, p in enumerate(paragraphs[:3]): # 打印前3个
    #     print(f"  {i+1}. {p.get_text(strip=True)[:100]}...") # 获取文本内容，去除空白

    # --- 示例：按类查找元素 ---
    # 假设数据项位于类名为'product-item'的div中
    product_items = soup.find_all('div', class_='product-item') # 类
    print(f"\n找到 {len(product_items)} 个类名为'product-item'的元素")
    # for item in product_items:
        # 提取每个项目中的特定数据，例如名称和价格
        # name = item.find('h3', class_='product-name')
        # price = item.find('span', class_='price')
        # if name and price:
        #     print(f"  - 名称: {name.get_text(strip=True)}, 价格: {price.get_text(strip=True)}")

    # --- 示例：按ID查找元素 ---
    # 假设有一个ID为'summary-table'的特定表格
    summary_table = soup.find('table', id='summary-table') # ID
    if summary_table:
        print("找到ID为'summary-table'的表格。")
        # 您可以迭代此表格中的行（tr）和单元格（td）
        # data_rows = summary_table.find_all('tr')
        # for row in data_rows:
        #     cells = row.find_all(['td', 'th']) # 查找数据和标题单元格
        #     cell_texts = [cell.get_text(strip=True) for cell in cells]
        #     print("  " + " | ".join(cell_texts))
    else:
        print("未找到ID为'summary-table'的表格。")

else:
    print("HTML内容为空，跳过解析。")
```

**Beautiful Soup方法：**

- `BeautifulSoup(html_content, 'html.parser')`：创建soup对象。`'html.parser'`是Python内置的解析器。`'lxml'`（需要安装）速度更快，通常更受欢迎。
- `soup.find('tag_name', attribute='value')`：返回符合条件的*第一个*元素。
- `soup.find_all('tag_name', class_='class_name')`：返回所有符合条件的元素的*列表*。请注意`class_`中的下划线，因为`class`是Python的保留字。
- `element.get_text(strip=True)`：从元素及其子元素中提取可读文本，并去除前导/尾随空白。
- `element['attribute_name']`：访问属性的值（例如，`link_tag['href']`）。

### 提取结构化数据：表格

HTML表格（`<table>`）是结构化数据的常见来源。通常，您会遍历行（`<tr>`），然后遍历每行中的单元格（数据用`<td>`，表头用`<th>`）。

```python
import pandas as pd

if summary_table: # 假设summary_table已找到
    headers = []
    data = []

    # 提取表头（假设它们位于第一个<tr>行中的<th>标签内）
    header_row = summary_table.find('tr')
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all('th')]

    # 提取数据行（假设它们是后续带有<td>标签的<tr>元素）
    data_rows = summary_table.find_all('tr')[1:] # 如果表头已单独处理，则跳过表头行
    for row in data_rows:
        row_data = [td.get_text(strip=True) for td in row.find_all('td')]
        if len(row_data) == len(headers): # 确保行数据与表头数量匹配
             data.append(row_data)

    # 创建一个Pandas DataFrame
    if headers and data:
        df = pd.DataFrame(data, columns=headers)
        print("\n提取到的表格数据:")
        print(df.head())
    else:
        print("\n未能提取结构化表格数据（表头或数据缺失/不匹配）。")
```

### 应对挑战

- **动态内容（JavaScript）**：许多现代网站在接收初始HTML页面*之后*使用JavaScript加载数据。`Requests`只获取初始HTML源代码。如果您需要的数据不在源代码中，抓取就会变得更复杂。这时可能需要Selenium或Playwright等控制真实网络浏览器的工具。这些工具会执行JavaScript，但速度较慢且资源消耗更多。有时，您可以通过检查浏览器开发者工具中的网络流量来找到JavaScript调用的底层API端点，如果可行，这通常是更有效的方法。
- **网站结构变化**：网站会随着时间改变其布局。HTML结构的变化（例如，类名、标签层级）可能会导致您的抓取器失效。通常需要定期维护。
- **反抓取措施**：网站可能会采用一些技术来检测和阻止抓取器（例如，验证码、IP速率限制、检查类似浏览器的请求头和行为）。这可能使某些网站的抓取变得困难或无法进行。

网页抓取是一种有用的技能，适用于其他方法无法获取数据的情况，但它需要细致地处理技术、道德和法律方面的考量。始终负责任地进行抓取，遵守网站政策，并准备好应对目标网站结构的变化。通过抓取获取的数据通常需要大量的清洗和结构化，这使得本主题与本章中讨论的数据准备后续步骤连接起来。

## 参考资料

- [Beautiful Soup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) — Leonard Richardson (2024)
  Beautiful Soup 4 的官方文档，介绍了其用于解析 HTML 和 XML 文档以及提取数据的方法。
- [Requests: HTTP for Humans™](https://requests.readthedocs.io/en/latest/) — Kenneth Reitz and others (2024)
  Requests 库的官方文档，详细说明了如何发送 HTTP 请求以可靠地获取网页内容。
- [Web Scraping with Python: Collecting Data from the Modern Web](https://openlibrary.org/works/OL17377281W/Web_Scraping_with_Python) — Ryan Mitchell (2018)
  Publisher: O'Reilly Media
  一本全面的书籍，解释了网页抓取原理、道德考量、使用 Beautiful Soup 和 Scrapy 等 Python 库的实用技术以及处理动态内容。
- [Scrapy Documentation](https://docs.scrapy.org/en/latest/) — Scrapy developers (2024)
  Scrapy 的官方文档，Scrapy 是一个用于 Python 的快速、高级网络爬虫和抓取框架，适用于大型项目。
