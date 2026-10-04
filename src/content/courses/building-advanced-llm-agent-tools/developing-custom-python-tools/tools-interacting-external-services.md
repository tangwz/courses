---
course: "building-advanced-llm-agent-tools"
chapter: "developing-custom-python-tools"
lesson: "tools-interacting-external-services"
sourceId: 6559
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-2-developing-custom-python-tools/tools-interacting-external-services"
title: "与外部服务交互：API 和数据库"
description: "开发用于连接和与外部 API 及数据库交互的工具，扩展代理能力。"
order: 3
plots: []
sourceHash: "348e0c5fb827182a70378bbe329f6e55151121bf23d22cc6e669610c119ed6f4"
sourceCorrections: []
---

你的LLM代理通常需要处理或获取在其当前编程环境中无法直接获得的信息。为了真正增强它们的能力，你的自定义Python工具将经常需要与外部服务通信。你将遇到的两种最常见的外部服务是应用程序编程接口（API）和数据库。构建能够高效地与这些资源对接的工具，让你的代理能够获取实时数据、访问专门的功能并处理持久化信息。

## 连接外部API

API是通往网络上各种功能和数据源的通道。无论是获取最新的金融新闻、翻译文本、获取天气更新，还是与专有业务系统交互，API都允许你的工具以编程方式访问这些服务。`requests`库是Python中进行HTTP请求的事实标准，HTTP请求是大多数Web API的底层协议。

如果你尚未安装它，可以使用pip将`requests`添加到你的项目中：
`pip install requests`

### 发起API请求

大多数API交互都涉及向特定URL（一个端点）发送HTTP请求，然后处理响应。你将使用的两种最常见的HTTP方法是`GET`（用于获取数据）和`POST`（用于发送数据或触发操作）。

**GET 请求**

`GET`请求用于从API获取信息。你可以在URL的查询字符串中传递参数 (parameter)。

```python
import requests

def get_current_weather(api_key, city_name):
    """使用示例天气API获取给定城市的当前天气数据。"""
    base_url = "http://api.exampleweather.com/v1/current.json" # 虚构的API端点
    params = {
        "": api_key,
        "q": city_name
    }
    try:
        response = requests.get(base_url, params=params, timeout=10) # 10秒超时
        response.raise_for_status()  # 对错误响应（4XX或5XX）引发HTTPError
        weather_data = response.json() # 假设API返回JSON
        # 处理并返回相关天气信息
        description = weather_data.get("current", {}).get("condition", {}).get("text", "Not available")
        temp_c = weather_data.get("current", {}).get("temp_c", "N/A")
        return f"Current weather in {city_name}: {description}, Temperature: {temp_c}°C"
    except requests.exceptions.Timeout:
        return f"错误：天气API请求超时。"
    except requests.exceptions.HTTPError as http_err:
        return f"错误：发生HTTP错误：{http_err} - 状态：{response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"错误：获取天气数据时发生错误：{e}"

# 示例用法（API密钥应安全管理）
# print(get_current_weather("YOUR_API_KEY", "London"))
```

在此示例中，`requests.get()`发送请求。`response.raise_for_status()`是一种方便的方式来检查请求是否成功（状态码200-299）；否则，它会引发异常。API响应通常是JSON格式，可以使用`response.json()`轻松解析为Python字典。

**POST 请求**

`POST`请求用于向API发送数据，可能是为了创建新资源或触发操作。数据通常在请求体中发送，常为JSON格式。

```python
import requests
import json

def submit_user_feedback(api_endpoint, user_id, feedback_text):
    """向示例反馈API提交用户反馈。"""
    payload = {
        "user_id": user_id,
        "feedback": feedback_text,
        "source": "LLM_Agent_Tool"
    }
    headers = {
        "Content-Type": "application/json"
        # "Authorization": "Bearer YOUR_ACCESS_TOKEN" # 如果需要认证
    }
    try:
        response = requests.post(api_endpoint, data=json.dumps(payload), headers=headers, timeout=10)
        response.raise_for_status()
        # 假设API返回JSON响应，例如包含提交ID
        return f"反馈提交成功。响应：{response.json()}"
    except requests.exceptions.RequestException as e:
        return f"提交反馈时出错：{e}"

# 示例用法
# feedback_api_url = "http://api.examplefeedback.com/v1/submit" # 虚构的API端点
# print(submit_user_feedback(feedback_api_url, "user123", "The agent was very helpful!"))
```

在这里，`json.dumps(payload)`将Python字典序列化为JSON字符串，并作为请求体发送。`Content-Type`头部告诉服务器我们正在发送JSON。

### 处理API响应和认证

始终检查`response.status_code`以了解请求的结果。常见状态码包括：

- `200 OK`：请求成功。
- `201 Created`：请求成功，并创建了新资源（常用于POST/PUT）。
- `400 Bad Request`：服务器因无效语法无法理解请求。
- `401 Unauthorized`：需要认证，但认证失败或尚未提供。
- `403 Forbidden`：服务器理解请求，但拒绝执行。
- `404 Not Found`：未找到请求的资源。
- `429 Too Many Requests`：你已达到请求频率限制。
- `500 Internal Server Error`：服务器遇到意外情况。

许多API需要认证。常见方法包括API密钥（作为头部或查询参数 (parameter)发送）或OAuth令牌。安全管理这些凭据非常重要；避免直接在工具中硬编码它们。使用环境变量、配置文件或专门的秘密管理服务。第4章“将外部API集成到工具中”将更详细地介绍认证和其他API集成方面，例如速率限制。

## 查询数据库

数据库对于需要持久存储、检索和管理结构化数据的工具来说必不可少。你的LLM代理可能需要一个工具来查找产品信息、检索客户历史或记录交互细节。

Python有一个用于连接SQL数据库的标准API，称为DB-API 2.0（PEP 249）。大多数用于PostgreSQL、MySQL和SQLite等关系型数据库的Python数据库驱动程序都遵循此规范。

### 使用`sqlite3`简化操作

SQLite是一个轻量级的、基于文件的数据库引擎，通过`sqlite3`模块内置于Python的标准库中，这使得它非常适合用于示例和简单的应用程序。

让我们设想一个需要从SQLite数据库获取客户详细信息的工具。

首先，你需要建立连接并创建一个游标对象。游标允许你执行SQL查询。

```python
import sqlite3

def get_customer_details(db_path, customer_id):
    """使用客户ID从SQLite数据库检索客户详细信息。"""
    conn = None  # 为finally块将conn初始化为None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 使用参数化查询以防止SQL注入！
        query = "SELECT name, email, join_date FROM customers WHERE id = ?;"
        cursor.execute(query, (customer_id,)) # 元组 (customer_id,) 很重要

        customer_data = cursor.fetchone() # 获取一行

        if customer_data:
            name, email, join_date = customer_data
            return f"客户ID：{customer_id}\n姓名：{name}\n电子邮件：{email}\n加入日期：{join_date}"
        else:
            return f"未找到ID为{customer_id}的客户。"

    except sqlite3.Error as e:
        return f"发生数据库错误：{e}"
    finally:
        if conn:
            conn.close() # 始终关闭连接

# 示例：假设'company.db'存在并有一个'customers'表。
# 用于测试设置：
# def setup_dummy_db(db_file):
#     conn_setup = sqlite3.connect(db_file)
#     cursor_setup = conn_setup.cursor()
#     cursor_setup.execute("DROP TABLE IF EXISTS customers")
#     cursor_setup.execute("""
#         CREATE TABLE customers (
#             id INTEGER PRIMARY KEY,
#             name TEXT NOT NULL,
#             email TEXT NOT NULL UNIQUE,
#             join_date TEXT
#         )
#     """)
#     cursor_setup.execute("INSERT INTO customers (id, name, email, join_date) VALUES (?, ?, ?, ?)",
#                          (1, 'Alice Wonderland', 'alice@example.com', '2023-01-15'))
#     cursor_setup.execute("INSERT INTO customers (id, name, email, join_date) VALUES (?, ?, ?, ?)",
#                          (2, 'Bob The Builder', 'bob@example.com', '2022-11-05'))
#     conn_setup.commit()
#     conn_setup.close()

# setup_dummy_db('company.db')
# print(get_customer_details('company.db', 1))
# print(get_customer_details('company.db', 3))
```

这里强调的一个重要方面是使用**参数 (parameter)化查询**（`cursor.execute(query, (customer_id,))`）。你应该始终使用占位符（例如`sqlite3`的`?`或某些其他驱动程序的`%s`），并将值作为单独的元组或字典传递。这种做法对于防止SQL注入漏洞很重要，否则恶意输入可能会改变你的SQL查询。

执行`SELECT`查询后，你可以使用以下方法检索结果：

- `fetchone()`：获取查询结果集的下一行，返回单个序列；如果没有更多数据，则返回`None`。
- `fetchall()`：获取查询结果的所有剩余行，返回序列列表。
- `fetchmany(size)`：获取下一组行，返回列表。

对于修改数据库的操作（如`INSERT`、`UPDATE`、`DELETE`），你需要使用`conn.commit()`提交事务以保存更改。始终确保连接被关闭（通常在`finally`块中），以释放资源。

### 使用其他数据库和ORM

尽管`sqlite3`很方便，但对于更多应用程序，你可能会使用PostgreSQL或MySQL等其他数据库系统。连接过程和查询执行是类似的，但你需要安装它们各自的Python驱动程序（例如，PostgreSQL的`psycopg2-binary`，MySQL的`mysql-connector-python`）。

对于复杂应用程序，像SQLAlchemy或Django ORM这样的对象关系映射器（ORM）可以提供比直接SQL更高层次的抽象。ORM允许你使用Python对象和方法与数据库交互，这可以简化开发并提高代码可维护性，特别是对于处理复杂数据模型的工具。

## 图示：工具与外部API的交互

以下图示说明了LLM代理使用Python工具与外部API交互的典型流程：

> 此图显示了一个LLM代理请求股票价格信息。代理调用一个Python工具，该工具又查询外部股票API。该工具处理API的响应，并将结构化结果返回给LLM。

## 外部交互工具的设计考量

在构建与外部服务交互的工具时，请记住以下几点：

1. **明确的输入**：设计工具的输入参数 (parameter)，使其清晰地映射到API或数据库查询所需的内容。例如，如果一个工具查询天气API，其输入可能是`location`和`date`。
2. **数据转换**：来自API或数据库的原始数据可能冗长，或者不处于LLM的理想格式。你的工具应该处理并将这些数据转换为对代理而言简洁、易懂且有用的格式。这与后面“为LLM构建复杂工具输出”一节相关。
3. **错误处理**：外部服务可能会失败。网络问题、API速率限制、数据库停机或无效的API密钥都很常见。实施全面的错误处理（例如`try-except`块）来捕获这些问题，并向LLM返回有用的错误消息，使其能够尝试重试或采用替代方法。
4. **超时设置**：网络请求有时可能会无限期挂起。始终为API调用和数据库连接设置合理的超时时间，以防止你的工具长时间阻塞代理。
5. **幂等性**：在可能的情况下，设计与外部系统交互的工具应具有幂等性。这意味着使用相同的输入参数多次进行相同的调用，其效果与调用一次相同。如果代理在临时故障后可能重试工具调用，这一点尤其重要。
6. **配置**：将API端点、数据库连接字符串和凭据等配置外部化。避免直接在工具逻辑中硬编码这些内容。环境变量或配置文件是更好的替代方案。

熟练掌握与API和数据库的交互，将大大拓宽你的LLM代理可执行任务的范围。这些工具成为代理在数字环境中的感知和执行器官，使它们能够获取最新信息并在外部系统中触发操作。随着你的学习推进，你将把这些基本技能与更高级的错误处理、输入验证和输出结构化技术结合起来，构建真正强大和可靠的工具。

## 参考资料

- [Requests: HTTP for Humans™](https://requests.readthedocs.io/en/latest/) — Kenneth Reitz (2024)
  `requests`库的官方文档，提供关于在Python中进行HTTP请求、处理响应和管理会话的全面指南。
- [\`sqlite3\` - DB-API 2.0 interface for SQLite databases](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGbxOqUt8YM9IHvXTe7GB2TtXI_fl8xQxsYrFQVi4Di_EOJikcax6rzJtm9GQCiHckidvkUYoeSvWO3aHSb-NXGNGxCGhsR5jp6I1AnsubljjZ1H-LTh9_hJlmeqoNNSU8nRYgD5eMzsQ==) — Python Software Foundation (2024)
  Publisher: Python.org
  描述Python中SQLite接口的标准库文档，包含连接管理、游标操作和安全的参数化查询执行。
- [OWASP Top 10 Web Application Security Risks](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGB8soRhC12tQsuJfl4QUItDs15C8U1QpTuVvj-fIWDHcN9SvuCE0_CkC7EunlXMF7Rl7qVfSNdc9zlHV0kAfsteX3hmioUccnDfVWQDrLvNvEzJNbMGR9XoAp7SzRHC9cWXCF-ukX2mTI5hTArbDbzkKE=) — OWASP Foundation (2021)
  识别最关键Web应用程序安全风险的权威资源，提供详细解释和缓解策略，包括SQL注入。
- [Python Database API Specification v2.0](https://www.python.org/dev/peps/pep-0249/) — Marc-André Lemburg (2000)
  Publisher: Python Software Foundation
  定义Python数据库驱动标准接口的官方规范，确保各种实现之间数据库访问方法的一致性。
