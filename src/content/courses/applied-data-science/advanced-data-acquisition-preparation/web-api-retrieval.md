---
course: "applied-data-science"
chapter: "advanced-data-acquisition-preparation"
lesson: "web-api-retrieval"
sourceId: 1828
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-1-advanced-data-acquisition-preparation/web-api-retrieval"
title: "使用Web API获取数据"
description: "使用Python库从Web API获取结构化数据的技术。"
order: 2
plots: []
sourceHash: "103f77c6db95a391bd7260dd2ef969c872e11ac40a1af9aff87a349e9d776b6a"
sourceCorrections: []
---

数据库提供对内部管理数据的访问，但许多组织和服务通过应用程序编程接口（API）对外提供数据。API充当标准化约定，使不同的软件系统能够进行通信。对数据科学家来说，Web API是结构化数据的常用来源，这些数据通常是实时数据，涵盖社交媒体活动、金融市场、天气预报和政府统计数据等。

与抓取为人类设计的HTML网页不同，API通常以JSON（JavaScript对象表示法）或XML等机器可读格式提供数据，这使得数据提取更整洁、更可靠。我们将把重点放在REST（表征状态传输）API上，这是一种使用标准HTTP方法的Web API常用架构风格。

### 了解API交互

与REST API的交互通常遵循请求-响应模式：

1. **客户端请求：** 您的Python脚本（客户端）向API服务器上的特定URL（端点）发送HTTP请求。此请求包括：

   - 一个**HTTP方法：** 通常是 `GET`，用于获取数据。其他方法如 `POST`、`PUT`、`DELETE` 用于创建、更新或删除资源，但对于简单的数据获取来说不太常用。
   - **请求头：** 关于请求的元数据，例如预期的响应格式 (`Accept: application/json`) 或认证凭据 (`Authorization: Bearer YOUR_API_KEY`)。
   - **参数 (parameter)（可选）：** 添加到URL（查询参数）中的键值对，用于过滤或指定请求的数据（例如，`?city=London&units=metric`）。
   - **请求体（可选）：** 随请求发送的数据，通常与 `POST` 或 `PUT` 一起使用，对于获取数据的 `GET` 请求通常不需要。
2. **服务器响应：** API服务器处理请求并返回HTTP响应，其中包含：

   - 一个**状态码：** 表示成功或失败（例如，`200 OK`、`404 Not Found`、`401 Unauthorized`、`500 Internal Server Error`）。
   - **响应头：** 关于响应的元数据。
   - **响应体：** 请求的数据，通常为JSON格式。

> 使用HTTP GET请求从Web API获取数据时的客户端-服务器交互简化视图。

### 使用Python的`requests`库发出API请求

`requests`库是Python中发出HTTP请求的实际标准库。如果您尚未安装，通常可以使用pip进行安装：

```bash
pip install requests
```

接下来看看如何使用它来获取数据。我们将使用JSONPlaceholder API（`https://jsonplaceholder.typicode.com`），这是一个免费的模拟在线REST API，用于测试和原型开发。

**基本GET请求**

要获取帖子列表，您可以向`/posts`端点发出GET请求：

```python
import requests
import pandas as pd

# 定义API端点URL
url = "https://jsonplaceholder.typicode.com/posts"

try:
    # 发送GET请求
    response = requests.get(url)

    # 对于错误的HTTP状态码（4xx或5xx）抛出异常
    response.raise_for_status()

    # 如果请求成功（状态码200）
    print("请求成功！")

    # 解析JSON响应体
    data = response.json() # 返回字典列表

    # 可选：转换为Pandas DataFrame
    df_posts = pd.DataFrame(data)
    print(f"成功获取了 {len(df_posts)} 条帖子。")
    print(df_posts.head())

except requests.exceptions.RequestException as e:
    # 处理连接错误、超时等
    print(f"请求失败: {e}")
except requests.exceptions.HTTPError as e:
    # 处理特定的HTTP错误（如404 Not Found, 401 Unauthorized）
    print(f"HTTP错误发生: {e}")
    print(f"状态码: {e.response.status_code}")
    # 您可能需要检查 e.response.text 以获取API的更多详细信息
except requests.exceptions.JSONDecodeError:
    # 处理响应体不是有效JSON的情况
    print("解析JSON响应失败。")
    print("响应文本:", response.text) # 记录原始文本
```

**添加URL参数 (parameter)**

许多API允许您使用附加到URL的查询参数（例如，`?userId=1`）来过滤或自定义结果。`requests`库通过使用接受字典的`params`参数，使这变得容易。

```python
import requests
import pandas as pd

# 仅获取userId = 5的帖子
params = {"userId": 5}
url = "https://jsonplaceholder.typicode.com/posts"

try:
    response = requests.get(url, params=params)
    response.raise_for_status() # 检查HTTP错误

    data = response.json()
    df_user5_posts = pd.DataFrame(data)
    print(f"\n获取了 {len(df_user5_posts)} 条 userId=5 的帖子：")
    print(df_user5_posts.head())

except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
except requests.exceptions.HTTPError as e:
    print(f"HTTP错误发生: {e} (状态: {e.response.status_code})")
```

`requests`会自动编码字典并正确地将其附加到URL（例如，`https://jsonplaceholder.typicode.com/posts?userId=5`）。

**自定义请求头和认证**

API通常需要认证，通常通过API密钥。您可能还需要设置其他请求头，例如`User-Agent`或`Accept`。请求头作为字典传递给`headers`参数。

常见的认证模式包括：

1. **请求头中的API密钥：** 通常使用`Authorization`请求头（例如，`Authorization: Bearer YOUR_KEY` 或 `Authorization: ApiKey YOUR_KEY`）或自定义请求头（例如，`X-API-Key: YOUR_KEY`）。
2. **查询参数中的API密钥：** 安全性较低，但有时会使用（例如，`?apiKey=YOUR_KEY`）。

让我们模拟添加一个API密钥和一个自定义User-Agent请求头（尽管JSONPlaceholder不需要它们）：

```python
import requests

# 如果其他API需要，请将'YOUR_ACTUAL_API_KEY'替换为实际密钥
api_key = "YOUR_ACTUAL_API_KEY"
headers = {
    "Authorization": f"Bearer {api_key}",
    "User-Agent": "MyDataScienceApplication/1.0",
    "Accept": "application/json"
}

url = "https://jsonplaceholder.typicode.com/todos/1" # 示例：获取单个'todo'项目

try:
    # 传递headers字典
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    todo_item = response.json()
    print("\n获取了单个待办事项（含模拟请求头）：")
    print(todo_item)

except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
except requests.exceptions.HTTPError as e:
    print(f"HTTP错误发生: {e} (状态: {e.response.status_code})")
    # 如果您收到401或403错误，很可能是认证问题。
```

始终查阅特定API的文档，了解正确的认证方法和所需的请求头。

### 处理响应

成功发出请求只是过程的一半。您需要正确地解释响应。

- **检查状态码：** 始终检查`response.status_code`或使用`response.raise_for_status()`来确保请求成功（通常是状态码`200`）。`4xx`代码表示客户端错误（错误请求、缺少认证、资源未找到），而`5xx`代码表示服务器错误。
- **解析JSON：** 如果API返回JSON（最常用格式），请使用`response.json()`将其解析为Python字典或列表。如果响应不是有效的JSON，请准备好处理`requests.exceptions.JSONDecodeError`。
- **错误处理：** 将您的API调用包装在`try...except`块中，以优雅地处理网络问题（`requests.exceptions.RequestException`）、HTTP错误（`requests.exceptions.HTTPError`）和JSON解析错误。在出现错误时记录状态码和响应文本（`response.text`）有助于调试。

### API使用规范：速率限制和服务条款

API是共享资源。大多数API实行**速率限制**，限制您在特定时间窗口内可以发出的请求数量（例如，每分钟100个请求）。超过这些限制可能会导致暂时性封锁（例如，状态码`429 Too Many Requests`）。

始终：

1. **阅读API文档：** 了解使用限制、认证要求和数据使用政策。
2. **实施延迟：** 如果发出大量请求，请在它们之间添加暂停（例如，使用`time.sleep()`）。
3. **处理`429`错误：** 如果您收到`429`状态码，您的脚本应该等待（响应中的`Retry-After`请求头通常会指示等待时长），然后再尝试。
4. **缓存结果：** 如果数据不经常变化，请避免重复获取相同数据。

遵守API条款可确保持续访问和负责任的数据获取。

熟练掌握使用`requests`等库与Web API的交互，您可以获取到全面的动态数据集，显著扩展了可纳入您数据科学项目的数据范围。这构成了数据获取工具包的重要组成部分，是对从数据库和文件获取数据的补充。

## 参考资料

- [Architectural Styles and the Design of Network-based Software Architectures](https://www.ics.uci.edu/~fielding/pubs/dissertation/fielding_dissertation.pdf) — Roy Thomas Fielding (2000)
  Publisher: University of California, Irvine
  这篇基础论文介绍了代表性状态传输 (REST) 及其定义，作为分布式超媒体系统的一种架构风格，它是现代Web API的基石。
- [Requests: HTTP for Humans™](https://requests.readthedocs.io/en/latest/) — Kenneth Reitz and the Requests Community (2023)
  Python `requests` 库的官方文档，提供了进行HTTP请求、处理响应和管理各种API交互模式的全面指南和示例。
- [Data Wrangling with Python: Learn how to gather, store, and make sense of data](https://www.oreilly.com/library/view/data-wrangling-with/9781491948781/) — Jacqueline Kazil, Katharine Jarmul, Mike Dewar (2016)
  Publisher: O'Reilly Media
  这本更新版书籍提供了从各种网络源（包括使用Python的Web API）获取数据的实用指导，对数据科学家很重要。
