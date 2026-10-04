---
course: "fastapi-ml-deployment"
chapter: "intro-fastapi-api-fundamentals"
lesson: "practice-simple-endpoints"
sourceId: 5246
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-1-intro-fastapi-api-fundamentals/practice-simple-endpoints"
title: "实践：构建简单端点"
description: "动手练习使用FastAPI构建基础GET和POST端点。"
order: 8
plots: []
sourceHash: "09b26c51fd5fefcc6c34ec4723d1e43a5c72bc472b349394f6e0bf073f2617e4"
sourceCorrections: []
---

构建几个基础API端点，提供FastAPI的实践经验。这展示了路径操作、HTTP方法以及FastAPI如何处理请求。

我们将继续使用上一节中介绍的`main.py`文件。请记住在终端中使用`uvicorn main:app --reload`来运行您的应用程序。在开发过程中，`--reload`标志很方便，因为它会在您保存代码更改时自动重启服务器。

### 构建基础GET端点

最常用的HTTP方法是GET，用于从指定资源获取数据。在FastAPI中，您可以使用`@app.get()`装饰器在一个`async`函数上方定义GET端点。

让我们修改您的`main.py`，加入一个返回简单JSON消息的根GET端点：

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    """
    此端点返回一条消息。
    它作为API的根或入口。
    """
    return {"message": "To the ML Model API"}
```

说明如下：

1. `@app.get("/")`：此装饰器告知FastAPI，其下方的`read_root`函数负责处理对路径`/`（即根路径）发出的GET请求。
2. `async def read_root():`：这定义了一个名为`read_root`的异步函数。FastAPI支持同步（`def`）和异步（`async def`）路由处理器。使用`async def`可以借助Python的`asyncio`实现并发操作，我们将在第5章更多讨论。目前，请知晓这是在FastAPI中定义路径操作函数的标准方式。
3. `return {"message": "To the ML Model API"}`：FastAPI会自动将此Python字典转换为JSON响应。

保存文件。如果`uvicorn`正在以`--reload`运行，它应该会自动重启。打开您的网页浏览器并访问`http://127.0.0.1:8000/`。您应该看到JSON响应：`{"message":"Hello to the ML Model API"}`。

您也可以通过访问`http://127.0.0.1:8000/docs`来访问自动生成的交互式API文档。此界面（Swagger UI）允许您直接在浏览器中查看和测试您的API端点。找到`/`端点，展开它，然后点击“Try it out”，再点击“Execute”。

### 添加路径参数 (parameter)

通常，您需要根据URL路径中包含的标识符来获取特定项目。这些被称为路径参数。让我们构建一个端点来获取特定项目ID的信息。

```python
# main.py (将此函数添加在前面函数下方)
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "To the ML Model API"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    """
    根据路径中提供的ID获取项目。
    """
    # 在实际应用中，你会根据item_id获取数据
    return {"item_id": item_id, "description": f"Details for item {item_id}"}
```

重要修改：

1. `@app.get("/items/{item_id}")`：路径现在包含`{item_id}`。这表示一个名为`{item_id}`的路径参数。
2. `async def read_item(item_id: int):`：函数现在接受一个名为`item_id`的参数。关键是，我们添加了类型提示`: int`。FastAPI使用此提示来：
   - 验证为`item_id`在路径中接收到的值是否可以转换为整数。
   - 转换值（最初是URL中的字符串）为整数。
   - 提供编辑器支持（如自动补全和类型检查）。

保存文件并测试此端点。在浏览器中访问`http://127.0.0.1:8000/items/42`。您应该看到：`{"item_id":42,"description":"Details for item 42"}`。

现在，尝试访问`http://127.0.0.1:8000/items/abc`。FastAPI会在请求到达您的函数代码之前自动拦截此请求，因为“abc”无法转换为整数。它会返回一个有用的JSON错误响应，指示验证失败：

```json
{
  "detail": [
    {
      "loc": [
        "path",
        "item_id"
      ],
      "msg": "value is not a valid integer",
      "type": "type_error.integer"
    }
  ]
}
```

此自动数据验证是使用FastAPI和类型提示的一个主要优点。

### 添加查询参数 (parameter)

查询参数是附加在URL中问号（`?`）后面的可选键值对，常用作过滤、排序或分页。与路径参数不同，它们不是路径结构本身的一部分。

让我们修改`/items/`端点（或构建一个新端点）以接受可选的`skip`和`limit`查询参数用于分页。

```python
# main.py (根据需要修改或添加)
from fastapi import FastAPI
from typing import Optional # 导入 Optional

app = FastAPI()

# ... (之前的端点) ...

@app.get("/users/")
async def read_users(skip: int = 0, limit: int = 10):
    """
    获取用户列表，支持可选分页。
    使用查询参数'skip'和'limit'。
    """
    # 模拟从数据源获取用户
    all_users = [{"user_id": i, "name": f"User {i}"} for i in range(100)]
    return all_users[skip : skip + limit]

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    """
    根据ID获取项目，带有一个可选查询参数'q'。
    """
    response = {"item_id": item_id, "description": f"Details for item {item_id}"}
    if q:
        response.update({"query_param": q})
    return response
```

说明：

1. `read_users(skip: int = 0, limit: int = 10)`：
   - 参数`skip`和`limit`定义为函数参数，但*不*在路径字符串（`/users/`）中。FastAPI将它们识别为查询参数。
   - 类型提示（`: int`）提供验证和转换。
   - 默认值（`= 0`，`= 10`）使这些参数成为可选的。如果客户端未在URL中提供它们，将使用这些默认值。
2. `read_item(item_id: int, q: Optional[str] = None)`：
   - 我们添加了一个新参数`q`。
   - `Optional[str]`表示`q`可以是字符串或`None`。
   - `= None`将其默认值设置为`None`，使其成为可选的。

测试这些：

- `http://127.0.0.1:8000/users/`：使用默认值（`skip=0`，`limit=10`）。
- `http://127.0.0.1:8000/users/?skip=10&limit=5`：使用提供的值。
- `http://127.0.0.1:8000/items/5`：没有查询参数`q`。
- `http://127.0.0.1:8000/items/5?q=somequery`：包含查询参数`q`。

再次尝试提供无效类型，例如`skip=abc`，以查看自动验证错误。

### 构建基础POST端点

POST方法用于向服务器发送数据以创建或更新资源。定义POST端点与GET相似，使用`@app.post()`装饰器。

处理POST请求的*主体*（正在发送的数据）通常涉及Pydantic模型，我们将在第2章详细介绍。对于本次实践，让我们构建一个简单的POST端点，它仅确认请求，暂不处理复杂的输入数据。

```python
# main.py (添加此函数)
from fastapi import FastAPI
from typing import Optional

app = FastAPI()

# ... (所有之前的端点) ...

@app.post("/items/")
async def create_item():
    """
    创建新项目的占位端点。
    目前仅返回确认消息。
    数据接收将在第2章处理。
    """
    # 在第2章，我们将学习如何在此处接收数据
    return {"message": "Item received (but not processed yet)"}
```

由于浏览器通常只通过地址栏直接发出GET请求，您需要不同的工具来轻松测试此POST端点：

1. **FastAPI Docs**：访问`http://127.0.0.1:8000/docs`，找到`POST /items/`端点，展开它，点击“Try it out”，然后点击“Execute”。您应该看到成功响应。
2. **curl（命令行）**：打开您的终端并运行：

   ```bash
   curl -X POST http://127.0.0.1:8000/items/ -H "accept: application/json"
   ```

   `-X POST`标志指定了HTTP方法。`-H`添加了一个标头，表示我们接受JSON响应。您将收到`{"message":"Item received (but not processed yet)"}`。

### 完整代码示例

以下是包含我们构建的所有端点的完整`main.py`：

```python
# main.py
from fastapi import FastAPI
from typing import Optional

app = FastAPI(title="Simple API Practice", version="0.1.0")

@app.get("/")
async def read_root():
    """
    返回消息的根端点。
    """
    return {"message": "To the ML Model API"}

@app.get("/users/")
async def read_users(skip: int = 0, limit: int = 10):
    """
    获取用户列表，支持可选分页。
    使用查询参数'skip'和'limit'。
    """
    all_users = [{"user_id": i, "name": f"User {i}"} for i in range(100)]
    return all_users[skip : skip + limit]

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    """
    根据ID（路径参数）获取项目，带有一个可选查询参数'q'。
    """
    response = {"item_id": item_id, "description": f"Details for item {item_id}"}
    if q:
        response.update({"query_param": q})
    return response

@app.post("/items/")
async def create_item():
    """
    通过POST创建新项目的占位端点。
    """
    return {"message": "Item received (but not processed yet)"}

# 运行方式: uvicorn main:app --reload
```

您现在已成功构建了使用不同HTTP方法（GET、POST）的端点，并学会了如何使用FastAPI类型提示系统提供的自动验证来处理路径和查询参数 (parameter)。您还看到了交互式API文档的用处。在下一章中，我们将大幅提升我们处理数据的能力，特别是针对POST请求的复杂请求体，我们将使用Pydantic模型。

## 参考资料

- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2023)
  这是学习和使用FastAPI的首要且最全面的资源，涵盖了本节介绍的所有方面，如路径操作、HTTP方法、路径/查询参数以及自动文档。
- [typing - Support for type hints](https://docs.python.org/3/library/typing.html) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  解释了Python的类型提示系统，FastAPI在数据验证、转换和交互式文档生成方面严重依赖它，如路径和查询参数所示。
- [RESTful API design guide](https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design) — Microsoft (2025)
  Publisher: Microsoft
  提供了RESTful API设计的最佳实践和基本原则，包括HTTP方法（GET, POST）、路径参数和查询参数的正确使用，这些内容在本节中有所介绍。
- [asyncio - Asynchronous I/O](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  介绍了Python使用async/await语法编写并发代码的框架，为理解FastAPI端点中使用的async def函数提供了基础知识。
