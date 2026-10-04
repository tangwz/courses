# API 端点创建 (FastAPI, Flask)

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-10-deployment-operational-practices/api-endpoint-creation-fastapi-flask)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要让 LLM 应用程序可供用户或其他服务访问，尤其是在其已打包（例如在 Docker 容器中）的情况下，就需要公开其功能。通常，这是通过创建 Web 应用程序编程接口 (API) 来完成的。API 充当协议，定义了外部客户端如何通过网络（通常通过 HTTP）与您的应用程序交互。这种做法将您的核心 LLM 逻辑与任何特定的用户界面分离，并允许各种客户端（Web 应用程序、移动应用程序、其他后端服务）使用其能力。

Python 提供了几个优秀的 Web 框架来构建 API。我们将侧重于两种流行的选择：FastAPI 和 Flask。它们都提供了定义 API 端点（特定 URL）、处理传入请求、处理数据、与您的 LLM 工作流组件交互以及发送回响应所需的工具。

### 使用 FastAPI

FastAPI 是一个现代、高性能的 Python Web 框架，构建于标准 Python 类型提示之上。它以其速度（可与 NodeJS 和 Go 媲美）、使用 Pydantic 进行的自动数据验证、依赖注入功能以及自动交互式 API 文档（Swagger UI 和 ReDoc）而著称。它对异步操作 (`async`/`await`) 的原生支持使其特别适合 I/O 密集型任务，例如向外部 LLM API 发出请求，从而防止您的应用程序在等待 LLM 响应时阻塞。

让我们创建一个简单的 FastAPI 端点，以与 LLM 查询函数交互。

首先，请确保您已安装 FastAPI 和像 Uvicorn 这样的 ASGI 服务器：

```bash
pip install fastapi uvicorn pydantic openai # 或您的特定 LLM 客户端库
```

现在，创建一个 Python 文件（例如 `main.py`）：

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
# 假设我们有一个来自之前模块的 setup_llm_chain() 函数
# 该函数返回一个配置好的 LangChain 链或类似的可调用对象
# from my_llm_logic import setup_llm_chain

# 如果 setup_llm_chain 不可用，此为演示占位符
def dummy_llm_call(query: str) -> str:
    print(f"模拟 LLM 调用：{query}")
    # 在实际应用程序中，这将调用您的链、代理或直接 API
    # e.g., return llm_chain.invoke({"input": query})
    if "hello" in query.lower():
        return "您好！今天有什么可以帮您的吗？"
    else:
        return f"我收到了您的查询：'{query}'。正在处理..."

# 使用 Pydantic 定义请求体结构
class QueryRequest(BaseModel):
    text: str
    user_id: str | None = None # 示例可选字段

# 定义响应体结构
class QueryResponse(BaseModel):
    answer: str
    request_text: str

# 初始化 FastAPI 应用
app = FastAPI(
    title="LLM 查询服务",
    description="与我们的 LLM 工作流交互的 API 端点。"
)

# 加载或初始化您的 LLM 交互逻辑（例如 LangChain 链）
# 在实际应用程序中，请妥善管理此对象的生命周期。
# llm_chain = setup_llm_chain()

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    接收用户查询，通过 LLM 工作流处理它，并返回响应。
    """
    print(f"收到来自用户 {request.user_id or 'anonymous'} 的查询")
    try:
        # 将 dummy_llm_call 替换为您的实际 LLM 交互
        # result = await llm_chain.ainvoke({"input": request.text}) # 如果使用异步 LangChain
        result = dummy_llm_call(request.text) # 同步示例

        # 假设结果是字符串或可以通过 result['output_key'] 访问
        llm_answer = result # 根据您的实际返回结构进行调整

        return QueryResponse(answer=llm_answer, request_text=request.text)

    except Exception as e:
        # 在此处记录异常详情
        print(f"处理查询时出错：{e}")
        raise HTTPException(status_code=500, detail="处理查询时发生内部服务器错误。")

# 可选：添加一个简单的根端点用于健康检查
@app.get("/")
def read_root():
    return {"status": "LLM API 正在运行"}
```

要运行此应用程序，请使用 Uvicorn：

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

`--reload` 标志在检测到代码更改时自动重启服务器，这在开发期间很有用。将浏览器指向 `http://localhost:8000/docs` 将显示由 FastAPI 自动生成的交互式 Swagger UI 文档。您可以直接从那里测试 `/query` 端点。

所示优点：

1. **类型提示：** `request: QueryRequest` 清楚地定义了预期的输入结构。
2. **数据验证：** FastAPI 自动根据 `QueryRequest` 模型验证传入的 JSON。如果 `text` 字段缺失或不是字符串，它将返回 422 Unprocessable Entity 错误。
3. **序列化：** 返回值自动根据 `QueryResponse` 模型进行序列化。
4. **异步支持：** 使用 `async def` 定义路由函数允许使用 `await` 进行非阻塞调用（例如，如果您的链支持异步，则使用 `await llm_chain.ainvoke(...)`）。
5. **自动文档：** `/docs` 端点提供非常有用的交互式文档。

### 使用 Flask

Flask 是另一个广泛使用的 Python Web 框架。它通常被认为比 FastAPI 更简单、更明确，遵循微框架理念。它提供路由和请求处理的基本功能，将数据验证、异步支持（可通过扩展或 ASGI 服务器实现）以及其他功能的选择更多地留给开发者。

以下是使用 Flask 的一个类似示例：

首先，安装 Flask 和可能用于生产的 WSGI 服务器，例如 Gunicorn：

```bash
pip install Flask gunicorn openai # 或您的特定 LLM 客户端库
```

创建一个 Python 文件（例如 `app.py`）：

```python
from flask import Flask, request, jsonify
import os
# 假设我们有一个来自之前模块的 setup_llm_chain() 函数
# from my_llm_logic import setup_llm_chain

# 演示占位符
def dummy_llm_call(query: str) -> str:
    print(f"模拟 LLM 调用：{query}")
    # 在实际应用程序中，这将调用您的链、代理或直接 API
    # e.g., return llm_chain.invoke({"input": query})
    if "hello" in query.lower():
        return "您好！今天有什么可以帮您的吗？"
    else:
        return f"我收到了您的查询：'{query}'. 正在处理..."

# 初始化 Flask 应用
app = Flask(__name__)

# 加载或初始化您的 LLM 交互逻辑
# llm_chain = setup_llm_chain()

@app.route("/query", methods=['POST'])
def process_query():
    """
    接收用户通过 JSON 发送的查询，处理它，并返回 JSON 响应。
    """
    if not request.is_json:
        return jsonify({"error": "请求必须是 JSON"}), 400

    data = request.get_json()
    query_text = data.get('text')
    user_id = data.get('user_id', 'anonymous') # 示例可选字段

    if not query_text:
        return jsonify({"error": "请求体中缺少 'text' 字段"}), 400

    print(f"收到来自用户 {user_id} 的查询")

    try:
        # 将 dummy_llm_call 替换为您的实际 LLM 交互
        llm_answer = dummy_llm_call(query_text) # 同步示例

        response_data = {
            "answer": llm_answer,
            "request_text": query_text
        }
        return jsonify(response_data), 200

    except Exception as e:
        # 在此处记录异常详情
        print(f"处理查询时出错：{e}")
        return jsonify({"error": "处理查询时发生内部服务器错误。"}), 500

# 可选：添加一个简单的根端点用于健康检查
@app.route("/")
def index():
    return jsonify({"status": "LLM API 正在运行"}), 200

if __name__ == '__main__':
    # 仅用于开发服务器
    app.run(host='0.0.0.0', port=8000, debug=True)
```

要在开发中运行此应用：

```bash
python app.py
```

对于生产环境，您通常会使用 WSGI 服务器，例如 Gunicorn：

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

这里，`-w 4` 启动 4 个工作进程。

Flask 示例的重要方面：

1. **明确的请求处理：** 您手动检查 `request.is_json` 并使用 `request.get_json()` 来访问数据。
2. **手动验证：** 输入验证 (`data.get('text')`) 是在路由函数内部明确完成的。更复杂的验证通常涉及 Marshmallow 或 Cerberus 等库。
3. **JSON 响应：** `jsonify()` 用于将 Python 字典转换成 JSON 响应。
4. **默认为同步：** 标准 Flask 路由是同步的。异步操作通常需要像 Flask-Executor 这样的扩展，或者使用像 Hypercorn 这样的 ASGI 服务器运行 Flask。

### API 端点的考虑事项

- **输入/输出结构：** 为您的请求和响应体使用清晰一致的结构。FastAPI 中的 Pydantic 模型会自动强制执行此结构。在 Flask 中，定义清晰的 JSON 结构并仔细验证它。
- **错误处理：** 实现错误处理。返回适当的 HTTP 状态码（例如，400 表示错误请求，401/403 表示身份验证问题，500 表示服务器错误），并在响应体中提供信息的错误消息（避免泄露敏感细节）。
- **身份验证/授权：** 保护您的端点安全。具体做法取决于您的部署环境和要求（API 密钥、OAuth、JWT 令牌、网络级限制）。框架通常针对常见模式提供扩展或中间件支持。
- **异步处理：** LLM 调用可能很慢。使用异步请求处理（FastAPI 中的 `async`/`await`，或 Flask 通过 ASGI 实现的异步支持）来防止您的 API 服务器在等待 LLM 时被阻塞，从而允许它并发处理其他传入请求。
- **状态管理：** 决定如何管理 LLM 组件（链、代理、索引连接）的状态。它们应该在 API 服务器启动时初始化一次，还是按请求创建？全局初始化通常更高效，但需要仔细处理共享资源。

在 FastAPI 和 Flask 之间选择通常取决于项目需求。FastAPI 内置的数据验证、异步支持和自动文档功能对于复杂的 API 很有吸引力，特别是那些严重依赖 I/O 操作（如 LLM 调用）的 API。Flask 的简洁性和灵活性使其成为小型服务或当您更喜欢手动选择和集成组件时的很好选择。两者都为创建使您的已部署 LLM 应用程序可用的 API 层提供了坚实的基础。

## 参考资料

- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  提供FastAPI的全面指南，包含使用Pydantic进行数据验证、异步编程和自动API文档生成。
- [Flask Documentation](https://flask.palletsprojects.com/) — Pallets (2024)
  学习Flask的官方资料，涵盖路由、请求处理和其构建Web服务的模块化设计。
- [asyncio - Asynchronous I/O, event loop, coroutines and tasks](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  Python官方关于`asyncio`的文档，对于理解高性能Web框架（如FastAPI）中用于I/O密集型任务的异步编程模式很有价值。
- [Pydantic Documentation](https://docs.pydantic.dev/) — Samuel Colvin and Pydantic Contributors (2024)
  Pydantic的官方文档，这是一个对数据验证、序列化和设置管理很重要的库，是FastAPI请求和响应处理的基础。

---

[上一节](03-%E9%80%89%E6%8B%A9%E9%83%A8%E7%BD%B2%E7%AD%96%E7%95%A5.md) · [下一节](05-%E7%9B%91%E6%8E%A7%E5%B7%B2%E9%83%A8%E7%BD%B2%E7%9A%84%E5%BA%94%E7%94%A8%E7%A8%8B%E5%BA%8F.md)
