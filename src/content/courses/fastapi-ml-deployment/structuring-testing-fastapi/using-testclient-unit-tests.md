---
course: "fastapi-ml-deployment"
chapter: "structuring-testing-fastapi"
lesson: "using-testclient-unit-tests"
sourceId: 5304
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-4-structuring-testing-fastapi/using-testclient-unit-tests"
title: "使用 TestClient 进行单元测试"
description: "使用内置的 TestClient 为您的 FastAPI 应用程序编写单元测试。"
order: 5
plots: []
sourceHash: "7985ae1ae93082df674e42658aee096ec338720f6eafeb656e1bbaacc6c405d9"
sourceCorrections: []
---

对 FastAPI 应用程序进行测试是构建可靠服务的一个基本环节。通过 `curl` 或浏览器界面等工具手动测试端点可能耗时且容易出错，特别是随着应用程序的增长。FastAPI 提供了一种便捷的方式，使用 `TestClient` 为您的 API 编写自动化测试。

`TestClient` 基于出色的 `httpx` 库构建，该库提供了一个现代、支持异步的 HTTP 客户端。然而，当用于测试 FastAPI 应用程序时，`TestClient` 直接与您的应用程序代码交互，*无需* 运行像 Uvicorn 这样的实时 Web 服务器。这使得测试更快、更可靠，并且更容易在持续集成 (CI) 管道等自动化环境中运行。它有效地模拟向您的应用程序发送 HTTP 请求，并允许您检查响应。

### 配置 TestClient

要使用 `TestClient`，您首先需要导入它，并通过传入您的 FastAPI 应用程序实例来实例化它。通常，您会在测试文件中执行此操作。

假设您在名为 `main.py` 的文件中定义了一个简单的 FastAPI 应用程序：

```python
# main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.post("/items/")
async def create_item(item: Item):
    return {"item_name": item.name, "item_price": item.price}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

现在，您可以创建一个测试文件（例如 `test_main.py`）并配置 `TestClient`：

```python
# test_main.py
from fastapi.testclient import TestClient
from main import app # 导入您的 FastAPI 应用程序实例

# 实例化 TestClient
client = TestClient(app)

def test_read_main():
    # 向根路径 "/" 发送 GET 请求
    response = client.get("/")
    # 断言 HTTP 状态码为 200 (OK)
    assert response.status_code == 200
    # 断言响应的 JSON 与预期的字典匹配
    assert response.json() == {"message": "Hello World"}
```

在此示例中：

1. 我们从 `fastapi.testclient` 导入 `TestClient`。
2. 我们从 `main.py` 文件中导入 `app` 实例。
3. 我们创建了一个 `TestClient` 实例，并将我们的 `app` 传递给它。
4. 在 `test_read_main` 函数内部（测试函数通常以 `test_` 开头），我们使用 `client.get("/")` 来模拟向根端点发送 GET 请求。
5. 然后，我们使用标准 `assert` 语句检查 `response.status_code` 是否为 `200` (HTTP OK)，以及 `response.json()` 内容是否与端点应返回的内容匹配。

### 测试不同的 HTTP 方法和参数 (parameter)

`TestClient` 支持所有标准 HTTP 方法，例如 `POST`、`PUT`、`DELETE` 等，与 `httpx` API 保持一致。

#### 测试 POST 请求

要测试在请求体中接收数据的端点（例如我们的 `/items/` POST 端点），您可以将字典作为请求方法的 `json` 参数传入：

```python
# test_main.py（续）
def test_create_item():
    item_data = {"name": "Test Item", "price": 10.99}
    # 向 "/items/" 发送带 JSON 数据的 POST 请求
    response = client.post("/items/", json=item_data)
    # 断言状态码为 200 (OK)
    assert response.status_code == 200
    # 断言响应的 JSON 与预期输出匹配
    # 注意：该端点返回 'item_name' 和 'item_price'
    assert response.json() == {"item_name": "Test Item", "item_price": 10.99}

def test_create_item_invalid_data():
    # 发送缺少必需 'price' 字段的数据
    invalid_item_data = {"name": "Incomplete Item"}
    response = client.post("/items/", json=invalid_item_data)
    # FastAPI 会自动为验证错误返回 422
    assert response.status_code == 422
    # 您可以选择检查验证错误的详细信息
    # assert "detail" in response.json() # 可以添加更具体的检查
```

第二个测试 `test_create_item_invalid_data` 演示了如何测试 Pydantic 处理的验证逻辑。发送不完整的数据会导致 `422 Unprocessable Entity` 状态码，对此我们进行了断言。

#### 测试路径和查询参数

测试带有路径和查询参数的端点非常直接。路径参数直接包含在 URL 字符串中，而查询参数可以作为字典传递给 `params` 参数。

```python
# test_main.py（续）
def test_read_item():
    item_id = 5
    # 向 "/items/5" 发送 GET 请求
    response = client.get(f"/items/{item_id}")
    assert response.status_code == 200
    assert response.json() == {"item_id": item_id, "q": None}

def test_read_item_with_query_param():
    item_id = 10
    query_string = "some query"
    # 向 "/items/10?q=some%20query" 发送 GET 请求
    response = client.get(f"/items/{item_id}", params={"q": query_string})
    assert response.status_code == 200
    assert response.json() == {"item_id": item_id, "q": query_string}
```

### 与测试框架集成（例如 Pytest）

虽然您可以使用 `TestClient` 运行简单的脚本，但它与 `pytest` 等测试框架集成得非常好。Pytest 提供了测试发现、fixture（用于设置/拆卸）、断言和报告等功能，使您的测试工作流程更高效。

一个典型的 `pytest` 结构可能如下所示：

```python
# test_main.py（使用 pytest 结构）
import pytest
from fastapi.testclient import TestClient
from main import app # 假设 main.py 包含您的 FastAPI 应用程序

# 使用 pytest fixture 为多个测试创建一次客户端
@pytest.fixture(scope="module")
def test_client():
    client = TestClient(app)
    yield client # 将客户端提供给测试

# 测试现在接受 fixture 名称作为参数
def test_read_main(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item(test_client):
    response = test_client.post("/items/", json={"name": "Test Item", "price": 10.99})
    assert response.status_code == 200
    assert response.json() == {"item_name": "Test Item", "item_price": 10.99}

# ... 其他使用 test_client 的测试 ...
```

使用像 `test_client` 这样的 fixture 有助于整洁地管理设置代码。您通常会在终端中使用 `pytest` 命令运行这些测试。

通过运用 `TestClient`，您可以为 FastAPI 端点编写全面的单元测试，确保您的 API 逻辑、数据验证和响应结构按预期准确运行。这构成了构建可靠且可维护的机器学习 (machine learning)部署服务的重要组成部分。在接下来的章节中，我们将讨论如何测试更复杂的场景，包括涉及数据库交互或外部依赖的场景，这通常需要像模拟这样的技术。

## 参考资料

- [Testing FastAPI](https://fastapi.tiangolo.com/tutorial/testing/) — Sebastián Ramírez (2024)
  官方文档提供了使用FastAPI的TestClient进行自动化测试的权威指南，涵盖了其设置、各种HTTP方法的用法以及集成。
- [pytest documentation](https://docs.pytest.org/en/stable/) — The pytest development team (2024)
  此官方文档对于理解如何有效地使用pytest框架组织和运行测试至关重要，该框架与FastAPI的TestClient集成。
- [HTTPX: A next-generation HTTP client for Python](https://www.python-httpx.org/) — Encode OSS (2024)
  提供了对TestClient所基于的底层异步HTTP客户端库的见解，更深入地理解其请求和响应处理能力。
- [Building Web APIs with FastAPI: Design, build, and deploy production-ready high-performance APIs](https://www.packtpub.com/product/building-web-apis-with-fastapi-design-build-and-deploy-production-ready-high-performance-apis/9781803248384) — Brian Okken (2023)
  Publisher: Packt Publishing
  一本涵盖FastAPI应用程序开发的综合书籍，包括专门介绍使用TestClient和pytest进行测试的部分，提供了实用且更广泛的背景。
