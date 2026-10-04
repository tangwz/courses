---
course: "fastapi-ml-deployment"
chapter: "structuring-testing-fastapi"
lesson: "logging-fastapi-applications"
sourceId: 5310
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-4-structuring-testing-fastapi/logging-fastapi-applications"
title: "FastAPI 应用中的日志记录"
description: "实现有效的日志记录，以监控请求、错误和应用行为。"
order: 7
plots: []
sourceHash: "03b011ef9391145b4a0638f8c61e768b5f9c0126366a7040d38f1165519a3077"
sourceCorrections: []
---

有效的日志记录是构建和维护可靠应用不可或缺的一部分，尤其是在将机器学习 (machine learning)模型部署为API时。当您的预测服务在生产环境中运行时，日志成为您了解其行为、诊断问题、监控性能和追踪使用模式的主要工具。如果日志记录不足，排查诸如意外预测结果、响应时间慢或推理 (inference)过程中出现错误等问题将变得异常困难。

FastAPI 应用与任何 Python 应用一样，可以使用 Python 内置的 `logging` 模块。这个标准库为从您的应用组件发出日志消息提供了灵活且功能强大的框架。`logging` 模块的核心涉及几个主要组成部分：

- **记录器（Loggers）：** 这些是您的应用代码直接交互的对象，用于发出消息（如 `logger.info(...)`、`logger.error(...)` 等）。记录器通常采用分层结构命名，常反映您项目的模块结构（例如 `logging.getLogger(__name__)`）。
- **处理器（Handlers）：** 处理器决定日志消息发送到何处。常见的处理器包括 `StreamHandler`（将日志发送到 `stderr` 或 `stdout`，这在容器化应用中很常见）、`FileHandler`（将日志写入文件）、`SysLogHandler`、`HTTPHandler` 等。
- **格式化器（Formatters）：** 格式化器控制日志记录的最终输出格式。您可以包含时间戳、日志级别、模块名称、特定消息内容等。
- **过滤器（Filters）：** 过滤器对哪些日志记录从记录器传递到处理器提供细粒度控制。

尽管运行 FastAPI 应用的 ASGI 服务器（如 Uvicorn）通常提供基本的访问日志（显示传入请求、状态码和耗时），但应用级别的日志记录能让您更详尽地了解 API 的内部运作机制，特别是机器学习推理逻辑。

### 基本日志配置

您可以使用 Python 代码配置日志记录。对于简单设置，`logging.basicConfig` 可能足够，但对于复杂应用，使用更结构化的配置方法通常是更好的做法。

要在 FastAPI 应用中开始日志记录，您首先需要获取一个记录器实例，通常以当前模块的名称命名：

```python
import logging
from fastapi import FastAPI

# 配置基本日志记录（可选，根据需要自定义）
# 这通常在应用启动时执行一次
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 获取当前模块的记录器实例
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def read_root():
    logger.info("Root endpoint was accessed.")
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict(data: dict): # 为简洁起见，假设输入是一个简单的字典
    logger.info(f"Prediction request received with data keys: {list(data.keys())}")
    try:
        # 模型推理的占位符
        prediction_result = {"prediction": "example_class", "probability": 0.95} 
        logger.info(f"Prediction successful: {prediction_result}")
        return prediction_result
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True) # exc_info=True 会记录堆栈跟踪
        # 重新抛出或返回适当的错误响应
        raise e 
```

在此示例中：

1. 我们导入了 `logging` 模块。
2. `logging.basicConfig` 将最低日志级别设置为 `INFO` 并定义了基本的日志消息格式。低于 `INFO` 级别（如 `DEBUG`）的消息将被忽略。
3. `logging.getLogger(__name__)` 获取一个特定于其调用模块的记录器。
4. 在路由处理程序内部，`logger.info()` 和 `logger.error()` 用于记录事件。在 `logger.error` 中使用 `exc_info=True` 会自动包含异常信息和堆栈跟踪，这对调试极有价值。

### 记录什么以及如何记录

决定 *记录什么* 很重要。目标是在详细程度和实用性之间取得平衡。日志记录过多会占用过多存储空间，并使查找相关信息变得困难。日志记录过少则在问题发生时让您无法了解情况。请考虑记录以下内容：

- **请求信息：** 关于传入请求的重要细节（例如，相关参数 (parameter)、请求ID）。除非绝对必要且已妥善保护/匿名化，否则请格外小心，*不要* 记录敏感信息，如密码、API 密钥或个人身份信息 (PII)。
- **ML 推理 (inference)：** 记录输入特征（如果输入很大，则记录摘要/哈希）、预测输出以及可能的模型置信度分数。
- **外部交互：** 对数据库、其他 API 或文件系统的调用。
- **错误和异常：** 始终记录错误，并附带足够多的上下文 (context)，包括堆栈跟踪。
- **重要的状态变化：** 记录应用状态中的重要事件或变化。

### 结构化日志记录

虽然人类可读的日志格式在开发期间很有用，但结构化日志（通常是 JSON 格式）在生产环境中非常有益。它们允许日志消息被日志聚合和监控系统（如 Elasticsearch/Logstash/Kibana (ELK)、Splunk、Datadog 等）方便地解析、索引和分析。

您可以配置 Python 的日志模块以输出 JSON 格式。像 `python-json-logger` 这样的库可以简化此过程。

```python
# 使用 python-json-logger 的示例设置（安装命令：pip install python-json-logger）
import logging
from pythonjsonlogger import jsonlogger

# 获取根记录器
log = logging.getLogger() 
log.setLevel(logging.INFO)

# 创建一个输出到控制台（stderr）的处理器
logHandler = logging.StreamHandler()

# 使用 JSON 格式化器
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
logHandler.setFormatter(formatter)

# 将处理器添加到根记录器
# 如果在其他地方配置，请注意不要多次添加处理器
if not log.hasHandlers():
    log.addHandler(logHandler)

# 获取您应用模块的记录器实例
logger = logging.getLogger(__name__)

# --- FastAPI 应用定义 ---
app = FastAPI()

@app.get("/status")
async def get_status():
    extra_data = {"service_version": "1.2.3", "uptime_seconds": 12345}
    logger.info("Status check performed", extra=extra_data) 
    return {"status": "OK"}

# 示例日志输出 (JSON):
# {"asctime": "...", "levelname": "INFO", "name": "__main__", "message": "Status check performed", "service_version": "1.2.3", "uptime_seconds": 12345} 
```

使用 `extra` 字典，您可以方便地为结构化日志记录添加自定义字段。

### 关联 ID

在处理多个并发请求的系统中，将属于同一请求的日志消息关联起来可能很困难。一种常见模式是为每个传入请求分配一个唯一的 *关联 ID*，并在处理该请求时生成的每条日志消息中包含此 ID。这使得通过日志追踪特定请求的整个生命周期变得简单得多。

FastAPI 中间件是生成和管理关联 ID 的合适位置，您可以将它们存储在请求上下文 (context)中或显式传递。像 `asgi-correlation-id` 这样的库可以帮助实现此模式。

实施日志记录最初可能看起来是额外的精力，但在您需要理解、排查问题和维护生产环境中的机器学习 (machine learning) API 时，它会带来巨大的回报。通过配置适当的日志级别、格式（最好是结构化），并在代码中的重要时点记录相关信息，您可以提升您的 FastAPI 应用的可观测性和可靠性。

## 参考资料

- [\`logging\` - Logging facility for Python](https://docs.python.org/3/library/logging.html) — Python Software Foundation (2023)
  Python标准日志库的官方文档，详细介绍了其架构和应用程序日志记录的用法。
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2023)
  FastAPI框架的官方文档，是API开发中整合日志记录实践的重要资源。
- [Site Reliability Engineering: How Google Runs Production Systems](https://www.oreilly.com/library/view/site-reliability-engineering/9781491929110/) — Niall Richard Murphy, Betsy Beyer, Chris Jones, Jennifer Petoff (2016)
  Publisher: O'Reilly Media, Inc.
  一本关于运行可靠生产系统的基础著作，涵盖了部署的日志记录、监控和可观察性原则。
