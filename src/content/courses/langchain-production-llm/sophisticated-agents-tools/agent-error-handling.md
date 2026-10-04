---
course: "langchain-production-llm"
chapter: "sophisticated-agents-tools"
lesson: "agent-error-handling"
sourceId: 3864
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-2-sophisticated-agents-tools/agent-error-handling"
title: "处理工具错误与智能体恢复"
description: "使智能体在工具执行期间更能应对故障的策略。"
order: 3
plots: []
sourceHash: "b73edbe49d49e72cc18c5180fd997972232b3ac70506553e2386912ddb0272ec"
sourceCorrections: []
---

智能体本质上通过工具与外部系统进行交互。无论是查询数据库、调用网络API、执行代码还是访问文件系统，这些交互都发生在智能体核心逻辑的受控环境之外。因此，在生产环境中的智能体系统中，工具执行是故障的常见原因。API可能不可用，数据库可能遇到连接问题，输入可能无效，或者工具本身可能包含程序错误。构建能够妥善处理这些不可避免的错误并尝试恢复的智能体，对于创建可靠的应用是极其重要的。

如果没有可靠的错误处理，智能体在遇到单个工具故障时可能会完全崩溃，过早放弃任务，或者更糟的是，进入一个无效的循环。本节将阐述在LangChain智能体中检测、管理和从工具错误中恢复的策略。

### 工具执行错误的来源

故障可能源于智能体与工具交互生命周期中的不同阶段：

1. **网络与连接：** 超时、DNS解析失败，或无法访问支持工具的服务（例如，外部API）。
2. **外部服务错误：** 工具成功连接到服务，但服务返回错误（例如，HTTP 4xx客户端错误，如 `401 Unauthorized`、`429 Too Many Requests`，或HTTP 5xx服务器错误）。
3. **输入验证问题：** LLM为工具生成的参数 (parameter)不符合工具预期的架构或限制。这对于结构化工具尤其相关。
4. **输出处理失败：** 工具成功执行，但返回的数据格式异常，智能体的解析逻辑或LLM无法正确解读。
5. **工具内部逻辑错误：** 实现工具功能的自定义代码中存在错误，导致在执行过程中出现未处理的异常。
6. **资源限制：** 工具执行可能超出分配的时间、内存或其他资源限制。

### 在LangChain中检测并报告错误

LangChain提供了捕获和报告源自工具的错误的机制。但是，默认情况下，工具中未处理的异常通常会停止智能体的执行。为了避免这种情况并让智能体能够恢复，您必须在工具实例或类上明确启用错误处理。

当工具上的 `handle_tool_error` 设置为 `True`（或指定自定义错误处理函数）时，执行器会捕获执行方法（例如 `_run`）引发的异常。随后，它将异常（通常是 `ToolException`）格式化为一个观察字符串。此字符串会在下一个推理 (inference)步骤中返回给LLM，告知智能体其尝试的操作失败了。

例如，如果一个用于获取天气数据的工具因网络超时而失败，提供给LLM的观察结果可能如下所示：

```text
观察结果：错误：工具'weather_api'失败，错误信息：连接天气服务API时请求超时。
思考：天气API工具因超时而失败。我应该再试一次，也许使用更短的查询，或者考虑是否有其他方法获取信息。如果再次失败，我可能需要告知用户目前无法获取天气。
```

有效的错误检测也依赖于全面的日志记录和追踪。观察智能体的执行轨迹，包括输入、思考、工具调用以及由此产生的错误或观察结果，是调试智能体为何失败以及它是如何尝试处理这种情况的根本。像LangSmith（第五章会提及）这样的工具对于在开发和生产环境中捕获和分析这些轨迹非常有价值。

### 智能体内部的错误处理策略

一旦检测到错误，智能体就需要一个处理策略。主要机制是让LLM根据观察结果中提供的错误信息进行推理 (inference)。

#### 告知LLM

对于简单情况，默认行为通常已足够。LLM接收错误信息并将其纳入其推理过程。根据其指令和错误上下文 (context)，它可能会决定：

- **重试同一工具：** 如果错误表明是输入问题，可能会修改参数 (parameter)后重试。
- **使用不同工具：** 如果有替代方案可以达成相同的子目标。
- **请求澄清：** 如果错误模糊不清或需要用户输入。
- **修改其计划：** 如果失败的步骤表明当前方法存在缺陷。
- **报告失败：** 如果它已用尽所有选项或判定任务无法完成。

传递回LLM的错误信息质量非常重要。过分冗长的堆栈跟踪会占用宝贵的上下文窗口空间，并可能使LLM感到困惑。简洁、信息丰富的错误信息通常更受欢迎。您可以通过对智能体或执行器进行子类化，或通过封装工具来定制工具异常的格式。

#### 智能体执行器错误处理参数

`AgentExecutor` 自身提供了一些参数来影响错误处理：

- `handle_parsing_errors`：此参数专门处理当智能体无法解析LLM输出（例如，LLM的响应未能正确格式化所需的工具名称或参数）时发生的错误。它不直接处理*来自*工具执行本身的错误。将其设置为 `True` 会向LLM提供一个默认的错误消息。您还可以提供一个自定义字符串或函数，以获得更具体的反馈，引导LLM修正其输出格式。
- `max_iterations`：限制智能体可以执行的步骤数（LLM调用 + 工具调用）。这可以防止无限循环，这种循环有时可能由失败的工具调用和无效重试的循环触发。
- `max_execution_time`：为整个智能体运行设置时间限制，防止智能体无限期地卡住，这可能是由于重复的工具超时导致的。

#### 实现自定义错误处理逻辑

为了获得更精细的控制，您可以实现自定义错误处理，而不是仅仅依赖LLM对错误字符串的反应。

1. **重试机制：** 使用重试装饰器或函数封装您的工具执行逻辑。这对于网络中断或临时速率限制等瞬时问题特别有效。指数退避（在每次重试之间逐步延长等待时间）是一种常规做法。

   ```python
   import time
   import random
   from typing import Type
   from requests.exceptions import RequestException
   from langchain_core.tools import BaseTool
   from pydantic import BaseModel, Field

   def retry_with_backoff(retries=3, initial_delay=1, backoff_factor=2, jitter=0.1):
       def decorator(func):
           def wrapper(*args, **kwargs):
               delay = initial_delay
               for i in range(retries):
                   try:
                       return func(*args, **kwargs)
                   except RequestException as e: # 示例：捕获特定的瞬时错误
                       if i == retries - 1:
                           raise # 重新抛出最后一个异常
                       # 对延迟应用抖动
                       actual_delay = delay + random.uniform(-jitter * delay, jitter * delay)
                       time.sleep(actual_delay)
                       delay *= backoff_factor
                   except Exception as e:
                       # Handle non-retryable errors differently or re-raise
                       raise e
           return wrapper
       return decorator

   class SearchInput(BaseModel):
       query: str = Field(description="要搜索的查询字符串")

   class MyApiTool(BaseTool):
       name: str = "my_api"
       description: str = "调用我的特殊API"
       args_schema: Type[BaseModel] = SearchInput
       
       # 必要：将其设置为True，以便将错误呈现给LLM而不是导致崩溃
       handle_tool_error: bool = True

       @retry_with_backoff(retries=3, initial_delay=1)
       def _run(self, query: str) -> str:
           # 替换为使用'requests'或类似库的实际API调用逻辑
           print(f"正在尝试使用查询：{query} 进行API调用")
           # 模拟潜在故障
           if random.random() < 0.5:
                raise RequestException("模拟的网络错误")
           return f"API对查询：{query} 调用成功"

       # 注意：如果未实现_arun，LangChain将默认在线程池中运行_run。

   # 在智能体中使用时，将涉及创建实例：
   # tool = MyApiTool()
   ```
2. **备用工具：** 设计智能体的提示或逻辑，以识别特定错误类型并明确尝试替代工具。例如，如果主要 `search_internal_docs` 工具失败，智能体可能会被指示尝试更通用的 `web_search` 工具。
3. **优雅降级：** 如果工具失败且没有替代方案，智能体可以被设计为提供部分响应，或指示特定信息不可用，而不是导致整个任务失败。
4. **结构化错误报告：** 不要只返回字符串，让工具在失败时返回结构化错误对象。这需要在智能体循环中进行自定义处理，但允许LLM或自定义逻辑根据错误代码或类型做出更准确的反应。

### 设计有弹性的工具

主动预防错误通常比被动处理错误更有效。在开发自定义工具时：

- **添加输入验证：** 在工具定义中（特别是对于 `StructuredTool`），使用Pydantic等库来验证LLM提供的参数 (parameter)，*在*尝试执行之前。如果验证失败，返回信息丰富的错误消息。
- **处理预期异常：** 在工具代码中，使用 `try...except` 块封装外部调用（API、数据库查询）。捕获特定异常（例如 `requests.exceptions.Timeout`、`sqlalchemy.exc.OperationalError`），并返回清晰、可操作的错误消息，而不是让原始异常冒泡上浮。
- **提供清晰的错误消息：** 确保工具返回的错误消息对LLM（或自定义逻辑）来说足够清晰，以便理解失败的性质。
- **考虑幂等性：** 如果工具会修改外部状态，请考虑在可能的情况下使其幂等（可以安全重试而不会产生意外副作用）。

### 智能体恢复流程示例

一个涉及错误处理的典型流程可能如下所示：

> 智能体执行流程，包含潜在的工具故障和错误处理分支。

有效地处理工具错误，能将智能体从一个脆弱的原型转变为一个更具弹性的系统，能够应对交互中的不确定性。通过结合向LLM提供信息丰富的错误反馈、策略性地运用执行器参数 (parameter)、自定义处理逻辑（如重试）以及妥善设计工具，您可以显著提升LangChain智能体在生产环境中的可靠性和表现。

## 参考资料

- [LangChain Agents Documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF5ZTb9SMV6L6IHFNepRuHhVrXc6J90Xu3uVX9EAtZnEyCzIG8KHGwf0o_JjKIQxMrHqAP25MARSduFF82BMYNYJPXDRPRc8rYY70SePdqIFN0PMylRWs59-hVT2V723ueq8urVXyppIaS8k6zNAh33yQfNOA==) — LangChain Contributors (2024)
  Publisher: LangChain
  提供关于构建和配置代理的全面信息，包括AgentExecutor参数（如handle_parsing_errors和max_iterations）的详细信息，以及框架内工具交互和错误报告的机制。
- [Release It!: Design and Deploy Production-Ready Software](https://pragprog.com/titles/mcrelease2/) — Michael T. Nygard (2018)
  Publisher: The Pragmatic Programmers
  一本关于构建健壮、有弹性和容错软件系统的基础著作，涵盖重试、超时和断路器等模式，与通用错误处理和恢复策略直接相关。
- [Building Microservices: Designing Fine-Grained Systems](https://books.google.com/books?id=Pj9gEAAAQBAJ) — Sam Newman (2021)
  Publisher: O'Reilly Media
  讨论了服务及其交互的设计，包括错误处理、弹性模式和健壮的API契约，适用于将工具视为代理系统内的组件。
