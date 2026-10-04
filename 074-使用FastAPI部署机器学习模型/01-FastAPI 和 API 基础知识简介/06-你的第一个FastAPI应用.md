# 你的第一个FastAPI应用

来源：[原文](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-1-intro-fastapi-api-fundamentals/first-fastapi-application)

[返回章节目录](README.md) · [返回课程目录](../README.md)

创建并运行你的第一个最小的FastAPI应用，需要配置好的开发环境和已安装的库。本练习展示了FastAPI程序的基本结构。

在你的项目目录中创建一个名为`main.py`的新文件，并添加以下Python代码：

```python
# main.py
from fastapi import FastAPI

# 创建FastAPI类的实例
app = FastAPI()

# 定义根路径 ("/") 的路径操作装饰器
# 这会告诉FastAPI，下面的函数处理指向 "/" 的GET请求
@app.get("/")
async def read_root():
  """
  这是API的根端点。
  它返回一个简单的问候消息。
  """
  return {"message": "Hello from the FastAPI ML Service!"}

# 定义另一个简单的端点
@app.get("/status")
async def get_status():
  """
  一个简单的状态端点。
  """
  return {"status": "API is running"}
```

我们来解析一下这段代码：

1. `from fastapi import FastAPI`：我们导入`FastAPI`类，它为你的API提供所有核心功能。
2. `app = FastAPI()`：我们创建一个`FastAPI`类的实例。这个`app`变量将是创建API路由的主要交互点。
3. `@app.get("/")`：这是一个Python装饰器。装饰器用于修改或增强函数。在这里，`@app.get`告诉FastAPI，它下面的函数（`read_root`）负责处理使用`GET` HTTP方法且指向路径`/`（根路径）的请求。路径（`/`）和HTTP方法（`GET`）的这种组合常被称为“操作”，而处理它的函数是“路径操作函数”。
4. `async def read_root():`：这定义了一个名为`read_root`的异步函数。FastAPI基于Python的`asyncio`库构建，让你能够为端点定义`async`函数。这使得可以并发处理多个请求，对于与模型或外部服务交互时常遇到的I/O密集型任务尤其有益。即使你的函数不执行显式`await`操作，将其定义为`async def`也允许FastAPI在其异步事件循环中正确运行它。
5. `return {"message": "Hello from the FastAPI ML Service!"}`：在函数内部，我们返回一个Python字典。FastAPI自动将此字典转换为JSON响应，并发送回客户端。这种自动数据序列化（以及后续将讲到的传入数据的反序列化）是一个很大的便利。
6. `@app.get("/status")` 和 `async def get_status():`：这定义了路径`/status`处的第二个端点，同样用一个简单的JSON状态消息响应`GET`请求。

要运行此应用，请在终端中导航到你的项目目录（包含`main.py`的目录），并执行以下命令：

```bash
uvicorn main:app --reload
```

我们来分析一下这条命令：

- `uvicorn`：这是运行Uvicorn ASGI服务器的命令，我们之前安装过它。Uvicorn负责通过HTTP实际服务你的FastAPI应用。
- `main:app`：这告诉Uvicorn在哪里找到你的FastAPI应用实例。
  - `main`：指代Python文件`main.py`。
  - `app`：指代`main.py`中创建的`app = FastAPI()`对象。
- `--reload`：这个标志告诉Uvicorn，当它检测到代码文件有更改时，自动重启服务器。这在开发过程中非常有用，因为你无需在每次修改代码后手动停止和启动服务器。

如果一切设置正确，你应该在终端中看到类似这样的输出：

```
INFO:     Will watch for changes in directory '{your_project_directory}'.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [{process_id}] using StatReload
INFO:     Started server process [{process_id}]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

这表示你的FastAPI应用正在运行，并可在`http://127.0.0.1:8000`访问。

打开你的网页浏览器并访问`http://127.0.0.1:8000`。你应该看到`read_root`函数返回的JSON响应：

```json
{"message":"Hello from the FastAPI ML Service!"}
```

现在，尝试访问`http://127.0.0.1:8000/status`。你应该看到：

```json
{"status":"API is running"}
```

### 自动交互式API文档

FastAPI的一个突出特点是其内置的自动交互式文档。它使用诸如OpenAPI（以前称为Swagger）和JSON Schema之类的标准，直接从你的代码生成文档，包括你的路径操作、参数 (parameter)和数据模型（我们将在下一章介绍）。

在你的应用运行期间，访问以下两个URL：

1. `http://127.0.0.1:8000/docs`：这提供Swagger UI界面。它是一个交互式环境，你可以在其中查看所有API端点、它们预期的参数、响应，甚至直接从浏览器中尝试。
2. `http://127.0.0.1:8000/redoc`：这提供一个使用ReDoc的备选文档界面。它提供API规范的清晰、层级化视图。

> 你的代码、FastAPI、服务器与自动生成的文档界面之间的关系。

试用这些界面。你会看到你的`/`和`/status`端点被列出。这种自动文档是一个显著的生产力提升工具，让你和他人更容易理解和使用你的API，尤其当它变得复杂时。

你现在已成功创建、运行并使用了你的第一个FastAPI应用。这个简单的例子奠定了基础，我们将在其上构建更复杂的服务，这些服务能够处理数据验证并提供机器学习 (machine learning)模型预测。

## 参考资料

- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  FastAPI框架的官方且最新资源，涵盖基础概念、API创建和高级功能。
- [Uvicorn Documentation](https://www.uvicorn.org/) — Tom Christie and Contributors (2024)
  Uvicorn的官方文档，Uvicorn是一个运行FastAPI应用的ASGI服务器，详细说明了安装、配置和命令行选项，例如--reload。
- [asyncio - Asynchronous I/O](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  Python `asyncio` 库的官方文档，该库为FastAPI中的并发编程和异步请求处理提供了基础。
- [OpenAPI Specification](https://www.openapis.org/) — OpenAPI Initiative (2024)
  Publisher: The Linux Foundation
  为描述RESTful API提供了标准，FastAPI利用此标准自动生成Swagger UI和ReDoc等交互式文档。

---

[上一节](05-%E8%AE%BE%E7%BD%AE%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83.md) · [下一节](07-FastAPI%20%E8%AF%B7%E6%B1%82-%E5%93%8D%E5%BA%94%E5%91%A8%E6%9C%9F%E7%9A%84%E6%9E%84%E6%88%90.md)
