---
course: "building-advanced-llm-agent-tools"
chapter: "llm-tool-testing-monitoring-maintenance"
lesson: "practice-basic-tool-logging"
sourceId: 6625
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-6-llm-tool-testing-monitoring-maintenance/practice-basic-tool-logging"
title: "实践：为工具使用设置基本日志记录"
description: "一个动手实践环节，讲解如何为代理工具实施基本日志记录，以跟踪其使用时间和方式。"
order: 7
plots: []
sourceHash: "93411e6f49066a9348290f9fa0b163ab670be303f68cc0f1d6256ca455258911"
sourceCorrections: []
---

本次实践环节指导您为自定义工具设置基本日志记录。有效的日志记录是维护可靠且可观察的LLM代理系统的一个主要方面。通过记录工具调用的主要信息，您可以大幅简化调试工作，监控工具状态，并了解您的代理如何发挥其作用。这直接有助于提高您的工具增强型代理的可靠性和长期可用性。

### 为何要记录工具使用情况？

在编写任何代码之前，让我们简要回顾一下记录工具活动为何如此重要。当LLM代理使用工具时，会发生以下几件事：代理决定使用工具，提供输入，工具执行，并返回输出（或错误）。记录这些事件有助于您：

- **调试故障**：当工具出现异常或代理未能获得预期结果时，日志会提供事件发生轨迹，包括导致问题的确切输入。
- **监控性能**：您可以跟踪工具被调用的频率、执行时长以及成功/失败率。这些信息对于找出瓶颈或不可靠的工具有很大价值。
- **理解代理行为**：日志可以展示代理选择和使用工具的模式，这对于评估代理的推理 (inference)过程或工具描述的有效性很有用。
- **审计与合规**：在某些应用中，保留工具操作记录可能是为了审计或合规目的所必需的。

### 介绍 Python 的 `logging` 模块

Python 内置的 `logging` 模块是一个灵活而强大的框架，用于从应用程序发出日志消息。它是大多数 Python 项目的标准选择，包括我们为 LLM 代理构建的工具。

`logging` 模块允许您使用不同级别按严重程度对消息进行分类：

- **DEBUG**：详细信息，通常仅在诊断问题时才有用。
- **INFO**：确认一切正常。这通常用于跟踪一般的工具调用和成功结果。
- **WARNING**：表明发生了意外情况，或近期可能出现问题（例如，‘磁盘空间不足’）。软件仍在正常运行。
- **ERROR**：由于更严重的问题，软件未能执行某些功能。这适用于记录工具执行期间的异常。
- **CRITICAL**：非常严重的错误，表明程序本身可能无法继续运行。

对于工具日志记录，`INFO` 通常适用于成功的调用及其结果，而 `ERROR` 用于工具内部的异常或故障。

### 设计日志消息

为了使日志有用，您需要决定包含哪些信息。对于LLM代理工具，常见要素有：

- **时间戳**：工具被调用的时间。
- **工具名称**：执行了哪个工具。
- **输入参数 (parameter)**：传递给工具的参数。*在此处务必小心：如果工具处理敏感信息，您可能需要从日志中遮蔽或省略某些输入。*
- **输出/结果**：工具返回了什么。对于大型输出，可以考虑记录摘要或成功指示。
- **执行状态**：工具执行是成功还是失败。
- **错误详情**：如果发生错误，记录错误消息，理想情况下包括堆栈跟踪。
- **执行时长**：工具运行了多长时间（对性能监控很有用）。
- **关联ID**：如果您的代理系统使用关联ID（例如，会话ID或请求ID），在日志中包含此信息可以帮助跟踪单个代理在多次工具调用中的行为。

### 动手实践：为示例工具实现日志记录

让我们为一个简单工具实现日志记录。我们将创建一个基本的“天气查询”工具，在此练习中，它只会返回一个模拟响应。

首先，确保您的 Python `logging` 模块已准备就绪（它是标准库的一部分，因此无需安装）。

#### 1. 配置基本日志记录

我们将从设置基本日志配置开始。这会告诉 Python 将日志消息发送到何处（例如，到控制台），要显示的最低严重级别是什么，以及如何格式化消息。将其添加到 Python 脚本的开头或初始化模块中：

```python
import logging
import time
from functools import wraps

# 配置基本日志记录
# 此设置将 INFO 级别及以上的消息记录到控制台。
# 格式包含时间戳、日志器名称、日志级别和消息。
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # 确保日志输出到控制台
)

# 获取我们工具模块的日志器实例
# 使用 __name__ 是常见做法，它将日志器名称设置为模块的名称。
logger = logging.getLogger(__name__)
```

此配置会将日志发送到标准输出（您的控制台）并清晰地格式化它们。

#### 2. 创建示例工具

让我们定义一个简单工具。此工具将模拟获取天气信息。

```python
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    模拟获取给定地点的当前天气。
    """
    logger.info(f"Tool 'get_current_weather' called. Location: {location}, Unit: {unit}")
    try:
        if not isinstance(location, str) or not location.strip():
            logger.error("Location must be a non-empty string.")
            raise ValueError("Location cannot be empty.")

        # 模拟 API 调用延迟
        time.sleep(0.5)

        if location.lower() == "errorland":
            logger.error(f"Simulated error fetching weather for {location}.")
            raise RuntimeError(f"Could not retrieve weather for {location}")

        # 模拟响应
        weather_report = f"The weather in {location} is 22\u00b0{unit.upper()[0]} and sunny."
        logger.info(f"Tool 'get_current_weather' successfully returned: {weather_report}")
        return weather_report
    except Exception as e:
        # exc_info=True 标志会将异常信息（如堆栈跟踪）添加到日志中。
        logger.error(f"Exception in 'get_current_weather': {str(e)}", exc_info=True)
        # 重新抛出异常或返回错误指示器
        # 以便代理框架能够处理它。
        raise
```

在此版本中，我们直接将 `logger.info()` 和 `logger.error()` 调用嵌入 (embedding)到 `get_current_weather` 工具中。这对于简单的工具来说很直接。

#### 3. 使用装饰器增强日志记录（推荐）

对于更复杂的场景，或者当您有许多工具时，直接在每个工具中添加日志语句会变得重复并使核心逻辑混乱。Python 装饰器是统一添加日志功能的优雅方式。

这是一个日志装饰器的示例：

```python
def tool_logger_decorator(tool_function):
    @wraps(tool_function) # 保留原始函数的元数据
    def wrapper(*args, **kwargs):
        tool_name = tool_function.__name__
        # 为日志创建参数表示
        # 在实际应用中请注意敏感数据
        arg_str_parts = [f"{arg!r}" for arg in args]
        kwarg_str_parts = [f"{key}={value!r}" for key, value in kwargs.items()]
        all_args_str = ", ".join(arg_str_parts + kwarg_str_parts)

        logger.info(f"Tool '{tool_name}' called. Args: ({all_args_str})")
        start_time = time.time()
        try:
            result = tool_function(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Tool '{tool_name}' completed successfully in {duration:.4f}s. Result: {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(
                f"Tool '{tool_name}' failed after {duration:.4f}s. Args: ({all_args_str}). Error: {str(e)}",
                exc_info=True # 添加堆栈跟踪
            )
            raise # 重新抛出异常
    return wrapper
```

现在，您可以将此装饰器应用于您的工具：

```python
@tool_logger_decorator
def get_current_weather_decorated(location: str, unit: str = "celsius") -> str:
    """
    模拟获取给定地点的当前天气（装饰器版本）。
    """
    # 注意：工具的核心逻辑内部现在没有直接的日志调用了！
    if not isinstance(location, str) or not location.strip():
        raise ValueError("Location cannot be empty.")

    time.sleep(0.5) # 模拟工作

    if location.lower() == "errorland":
        raise RuntimeError(f"Could not retrieve weather for {location}")

    return f"The weather in {location} is 22\u00b0{unit.upper()[0]} and sunny."

@tool_logger_decorator
def calculate_sum(a: int, b: int) -> int:
    """一个计算两个整数和的简单工具。"""
    time.sleep(0.1) # 模拟工作
    if not (isinstance(a, int) and isinstance(b, int)):
        raise TypeError("Both inputs must be integers.")
    return a + b
```

使用装饰器可以使您的工具函数更整洁，专注于它们的主要任务，而日志记录相关的功能则集中处理。

#### 4. 模拟工具使用并观察日志

让我们调用我们已添加装饰器的工具，并查看日志输出：

```python
if __name__ == "__main__":
    print("--- 测试 get_current_weather_decorated (成功) ---")
    try:
        weather = get_current_weather_decorated("London", unit="C")
        # print(f"报告: {weather}") # 可选：将结果打印到控制台
    except Exception as e:
        # print(f"捕获到异常: {e}") # 可选
        pass # 日志装饰器处理日志输出

    print("\n--- 测试 get_current_weather_decorated (输入验证错误) ---")
    try:
        get_current_weather_decorated("") # 无效输入
    except ValueError as e:
        # print(f"捕获到预期 ValueError: {e}") # 可选
        pass

    print("\n--- 测试 get_current_weather_decorated (模拟运行时错误) ---")
    try:
        get_current_weather_decorated("Errorland")
    except RuntimeError as e:
        # print(f"捕获到预期 RuntimeError: {e}") # 可选
        pass

    print("\n--- 测试 calculate_sum (成功) ---")
    try:
        total = calculate_sum(5, 7)
        # print(f"和: {total}") # 可选
    except Exception as e:
        # print(f"捕获到异常: {e}") # 可选
        pass

    print("\n--- 测试 calculate_sum (类型错误) ---")
    try:
        calculate_sum(10, "20") # 'b' 的类型无效
    except TypeError as e:
        # print(f"捕获到预期 TypeError: {e}") # 可选
        pass
```

当您运行此脚本时，您应该在控制台中看到类似以下的输出（时间戳会有所不同）：

```text
--- 测试 get_current_weather_decorated (成功) ---
2023-10-27 10:00:00,123 - __main__ - INFO - 工具 'get_current_weather_decorated' 被调用。参数：('London', unit='C')
2023-10-27 10:00:00,625 - __main__ - INFO - 工具 'get_current_weather_decorated' 成功完成，耗时 0.5012秒。结果：The weather in London is 22°C and sunny.

--- 测试 get_current_weather_decorated (输入验证错误) ---
2023-10-27 10:00:00,626 - __main__ - INFO - 工具 'get_current_weather_decorated' 被调用。参数：('',)
2023-10-27 10:00:00,627 - __main__ - ERROR - 工具 'get_current_weather_decorated' 失败，耗时 0.0001秒。参数：('',)。错误：Location cannot be empty.
回溯 (最近一次调用):
  ... (ValueError 的堆栈跟踪) ...

--- 测试 get_current_weather_decorated (模拟运行时错误) ---
2023-10-27 10:00:00,628 - __main__ - INFO - 工具 'get_current_weather_decorated' 被调用。参数：('Errorland',)
2023-10-27 10:00:01,130 - __main__ - ERROR - 工具 'get_current_weather_decorated' 失败，耗时 0.5015秒。参数：('Errorland',)。错误：Could not retrieve weather for Errorland
回溯 (最近一次调用):
  ... (RuntimeError 的堆栈跟踪) ...

--- 测试 calculate_sum (成功) ---
2023-10-27 10:00:01,131 - __main__ - INFO - 工具 'calculate_sum' 被调用。参数：(5, 7)
2023-10-27 10:00:01,232 - __main__ - INFO - 工具 'calculate_sum' 成功完成，耗时 0.1005秒。结果：12

--- 测试 calculate_sum (类型错误) ---
2023-10-27 10:00:01,233 - __main__ - INFO - 工具 'calculate_sum' 被调用。参数：(10, '20')
2023-10-27 10:00:01,334 - __main__ - ERROR - 工具 'calculate_sum' 失败，耗时 0.1002秒。参数：(10, '20')。错误：Both inputs must be integers.
回溯 (最近一次调用):
  ... (TypeError 的堆栈跟踪) ...
```

此输出清楚地显示了每个工具何时被调用、接收到的参数 (parameter)、是成功还是失败、结果（如果很长则截断）、持续时间以及错误的堆栈跟踪。

### 解释和使用您的日志

您生成的日志条目提供了有价值的记录：

- **成功调用**：`INFO` 消息确认工具名称、输入以及输出片段和执行时间。这对于验证正常操作和基本性能跟踪很有用。
- **失败调用**：`ERROR` 消息，连同 `exc_info=True`，提供了工具名称、导致失败的输入、错误消息以及完整的堆栈跟踪。这对于调试必不可少。您可以准确查看工具代码中问题发生的位置。

这些日志数据是进行更高级监控的前提。通过解析这些日志，您可以，例如，计算每个工具被调用的次数，计算平均执行时间，或跟踪特定错误的发生频率。

以下图表说明了使用装饰器时的日志记录流程：

> 此图显示了调用装饰器工具时的操作顺序，着重说明了日志装饰器如何在工具核心逻辑执行前后拦截调用以记录信息。

### 日志记录的进一步考虑

虽然本次实践设置了基本日志记录，但随着您的工具系统发展，有几点需要考虑：

- **结构化日志记录**：为了更方便的自动化解析和分析（例如，通过 Elasticsearch 或 Splunk 等日志管理系统），考虑以 JSON 等结构化格式记录消息。`python-json-logger` 等库可以提供帮助。
- **日志记录到文件**：为了持久存储，配置您的日志器将日志写入文件。为此使用 `logging.FileHandler` 类。实施日志轮转（`RotatingFileHandler` 或 `TimedRotatingFileHandler`）以管理日志文件大小。
- **集中式日志记录**：在生产环境中，特别是对于多个服务或分布式代理，将日志发送到集中式日志系统。这允许您在一个地方搜索、分析和监控所有组件的日志。
- **敏感数据**：再次强调谨慎处理敏感数据的重要性。对不应出现在日志中的参数 (parameter)或结果实施过滤或遮蔽。
- **配置**：对于更复杂的应用程序，通过外部文件（例如，INI、YAML 或 JSON）或字典管理日志配置，而不是使用 `basicConfig` 硬编码。Python 的 `logging.config` 模块支持此功能。

通过实施本次实践环节中展示的基本日志记录，您在为LLM代理构建更易于维护、可观察且最终更可靠的工具方面迈出了重要一步。这在您开发和部署日益复杂的代理系统时必不可少。

## 参考资料

- [logging - Logging facility for Python](https://docs.python.org/3/library/logging.html) — Python Software Foundation (2024)
  Python 标准日志模块的官方文档，详细介绍了其架构、级别、处理程序和格式化程序，对设置基本日志至关重要。
- [Fluent Python: Clear, Concise, and Effective Programming](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH5vwqR19xn_VqDBv-fwx672dOaIjXTZXy4q_1SMZzTaTm_RqkCE5O-g6qCwRVMMMrd8GdIEJlQE4yBI1s7e-W0jCl4z33yOX67MuVfnGCej7V49yGOzA4_1UxvDG3Ckq4lIYeMWghiG2Bwlnha6B8-LpLm8I6bYSl5N4OdRLOZdshQrnwZMyYB-VZgoXAClETCzAL1RRIg1HMmHgvhwRDKCzQo6WwgE1e15K9VL5GCKFASYGzD3RRiCpCIv1G3BA49JTFCU94Kng==) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  一本全面的 Python 惯用法指南，包含对装饰器及其在日志等横切关注点中实际应用的深入说明。
- [Site Reliability Engineering: How Google Runs Production Systems](https://www.oreilly.com/library/view/site-reliability-engineering/9781491929124/) — Betsy Beyer, Chris Jones, Jennifer Petoff, Niall Richard Murphy (2016)
  Publisher: O'Reilly Media
  关于构建和运行可靠系统的基础文本，其中包含监控、日志和事件管理原则的章节，适用于可观察的 LLM 智能体系统。
