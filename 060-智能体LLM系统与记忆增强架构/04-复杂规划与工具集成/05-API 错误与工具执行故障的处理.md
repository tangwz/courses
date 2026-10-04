# API 错误与工具执行故障的处理

来源：[原文](https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-4-complex-planning-tool-integration/handling-tool-errors)

[返回章节目录](README.md) · [返回课程目录](../README.md)

集成外部工具和API能极大地增强智能体的功能，使其能与动态信息源交互，并执行超出大型语言模型内部知识范围的操作。然而，这种交互也带来了潜在的故障点。网络连接可能不稳定，API可能无预警变更，服务可能出现停机，速率限制可能被突破，智能体本身也可能生成无效请求。构建可靠的智能体系统，需要将错误处理视为核心设计原则，而非可有可无的附加项。忽视这一点会导致智能体脆弱不堪，在遇到外部依赖的必然缺陷时，会不可预测地发生故障。

本节细述了方法，用于在工具执行过程中检测、管理和从故障中恢复，让智能体能更可靠、更具适应性地运行。

### 了解工具交互中的故障模式

在实施处理机制前，识别潜在故障的不同特点是很重要的：

1. **网络问题：** 这些问题通常是暂时的，与智能体和API端点之间的连接有关。示例包括：

   - **超时：** 服务器未在指定时间内响应。
   - **连接错误：** 无法建立连接（例如，`ConnectionRefusedError`）。
   - **DNS解析失败：** 无法解析API的主机名。
2. **API服务器错误（HTTP 5xx）：** 这些表明托管API的服务器端存在问题。

   - `500 内部服务器错误`：一个通用服务器错误。
   - `502 错误网关`：通常与代理或网关问题一同出现。
   - `503 服务不可用`：服务器暂时过载或因维护而停机。
   - `504 网关超时`：作为网关的服务器未从上游服务器收到及时响应。
3. **客户端错误（HTTP 4xx）：** 这些表明智能体发送的请求存在问题。

   - `400 错误请求`：请求格式不正确或包含无效语法。通常由大型语言模型生成不正确参数 (parameter)引起。
   - `401 未经授权`：缺少或无效的认证凭据。
   - `403 禁止访问`：认证成功，但已认证用户缺少请求资源的权限。
   - `404 未找到`：请求的资源不存在。
   - `429 请求过多`：智能体已超出API的速率限制。
4. **数据处理错误：** 在接收到响应后发生的故障。

   - **解析错误：** 响应格式非预期（例如，无效的JSON/XML）。
   - **架构验证错误：** 响应结构与预期架构不匹配。
5. **智能体生成的无效输入：** 大型语言模型本身可能为工具调用生成逻辑上不正确或格式错误的参数，这会导致可预测的API错误（通常是`400 Bad Request`）或意外的工具行为。

### 检测故障

有效的处理始于可靠的检测。在您的工具执行封装器中实现检测逻辑：

- **检查HTTP状态码：** 这是检测API级别错误的主要机制。明确检查状态码是否在4xx或5xx范围内。
- **实现超时：** 为所有外部请求配置合理的超时，以防止智能体无限期挂起。特别捕获超时异常。
- **使用`try-except`块：** 将外部调用（网络请求、数据解析）封装在`try-except`块中，以捕获诸如`requests.exceptions.Timeout`、`requests.exceptions.ConnectionError`、`json.JSONDecodeError` 或自定义API客户端异常。
- **验证响应架构：** 对于重要工具，在将接收到的数据传回智能体或后续处理步骤之前，根据预定义的架构（例如，在Python中使用Pydantic模型）验证其结构。这可以捕获意外的格式变化。
- **解析错误负载：** 许多API会在4xx/5xx错误的响应体中返回结构化的错误消息（通常是JSON）。解析这些消息以获取有关故障的更具体细节。

### 工具执行故障的处理方法

一旦检测到故障，智能体或其控制系统需要一个响应方法。

#### 1. 重试机制

重试对于网络瞬时故障或临时服务器不可用（5xx错误）等瞬时问题有效。

- **简单重试：** 在固定的延迟后，尝试再次调用固定次数。这对于一些情况通常不够用。
- **指数退避：** 以指数方式增加重试之间的延迟。一个常用公式是 $延迟 = 基础延迟 \times 因子^{尝试次数}$。例如，当$基础延迟 = 1秒$且$因子 = 2$时，延迟可能是1秒、2秒、4秒、8秒……这可以防止压垮暂时有问题的服务。
- **带抖动的指数退避：** 为退避延迟添加少量随机变化（$延迟 = random(0, 基础延迟 \times 因子^{尝试次数})$ 或 $延迟 = (基础延迟 \times 因子^{尝试次数}) + random(0, 抖动量)$）。抖动有助于防止在广泛的瞬时故障后，多个智能体实例同时重试（即“惊群效应”问题）。
- **条件重试：** 配置重试逻辑，使其仅针对特定错误类型触发。在未纠正根本问题（凭据或请求参数 (parameter)）的情况下，对`401 未经授权`或`400 错误请求`进行重试通常是徒劳的。重试最适用于超时、连接错误、`429 请求过多`和`5xx`服务器错误。

```python
import time
import random
import requests

def execute_api_call_with_retry(url, params, headers, max_attempts=5, base_delay=1.0, factor=2.0, jitter=0.1):
    """
    执行带指数退避和抖动的API GET请求。仅在特定瞬时错误时重试。
    """
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10) # 示例超时

            # 成功或不可重试的客户端错误
            if response.status_code < 500 and response.status_code != 429:
                 response.raise_for_status() # 立即为 4xx 状态码抛出 HTTPError
                 return response.json() # 或者返回响应对象

            # 可重试的服务器错误或速率限制
            if response.status_code >= 500 or response.status_code == 429:
                print(f"Attempt {attempts}: Received status {response.status_code}. Retrying...")
                # 继续执行重试逻辑

        except requests.exceptions.Timeout:
            print(f"Attempt {attempts}: Request timed out. Retrying...")
            # 继续执行重试逻辑
        except requests.exceptions.ConnectionError as e:
            print(f"Attempt {attempts}: Connection error ({e}). Retrying...")
            # 继续执行重试逻辑
        except requests.exceptions.RequestException as e:
            # 捕获其他与请求相关的异常（例如，由 raise_for_status 抛出的 4xx HTTPError）
            print(f"Attempt {attempts}: Non-retriable request error: {e}")
            raise # 立即重新抛出异常

        if attempts < max_attempts:
            # 计算带指数退避和抖动的延迟
            delay = base_delay * (factor ** (attempts - 1))
            delay = random.uniform(delay - delay * jitter, delay + delay * jitter)
            print(f"Waiting {delay:.2f} seconds before next attempt.")
            time.sleep(delay)
        else:
            print(f"Attempt {attempts}: Max retries reached.")
            # 选项：抛出自定义异常或返回特定的错误指示
            raise MaxRetriesExceededError(f"API call failed after {max_attempts} attempts.")

    # 如果抛出了 MaxRetriesExceededError，则不应到达此处
    return None

class MaxRetriesExceededError(Exception):
    pass

# 示例用法（请替换为实际API详情）
# try:
#     data = execute_api_call_with_retry("https://api.example.com/data", params={"id": 123}, headers={"Authorization": "Bearer token"})
#     # 处理数据
# except MaxRetriesExceededError as e:
#     # 处理重试后的最终故障
#     print(e)
# except requests.exceptions.HTTPError as e:
#     # 处理不可重试的 4xx 错误
#     print(f"Client error: {e.response.status_code}")
# except Exception as e:
#     # 处理其他意外错误
#     print(f"An unexpected error occurred: {e}")
```

> 此Python函数示例展示了API调用执行，其中包含指数退避、抖动以及基于HTTP状态码和请求异常的条件重试。

#### 2. 备用方法

当重试失败或不适用时，考虑替代操作：

- **使用备用工具：** 如果主要工具（例如，某个天气API）持续发生故障，智能体可以被设计为尝试次要天气API。这需要定义智能体规划模块可访问的工具等效性或层级关系。
- **使用缓存数据：** 如果稍微过时的数据可接受，智能体可以使用工具先前缓存的响应。这要求为工具输出实现一个缓存层。
- **优雅降级：** 智能体在信息不完整的情况下继续执行任务，并承认故障。它可能会告知用户（“我无法获取最新股票价格，但根据昨日收盘价……”）或做出合理的假设。

#### 3. 告知智能体并触发重新规划

重要的是，故障信息通常必须反馈给大型语言模型核心，以便其进行智能适应：

- **错误反馈：** 在智能体的上下文 (context)或暂存区中包含错误的简明摘要（例如，“工具‘GetWeatherAPI’故障：503 服务不可用，重试3次后”或“工具‘SearchEngine’故障：400 错误请求 - 查询格式无效”）。
- **提示工程 (prompt engineering)：** 指导大型语言模型如何应对特定错误。例如，“如果工具调用导致404错误，请考虑拓宽搜索词或使用不同的搜索工具。如果导致429错误，请等待后再试，或在可用时使用备用工具。”
- **触发重新规划：** 一个重要的工具故障，特别是对当前计划步骤至关重要的故障，应该触发重新规划周期。智能体接收错误上下文，并必须生成一个修改后的计划以达成其目标，这可能包括绕过故障工具或尝试不同方法。这直接联系到“自我修正与计划完善”一节中的内容。

#### 4. 断路器模式

对于反复发生故障的工具，持续重试会浪费资源和大型语言模型上下文。断路器模式提供了一种更有条理的方法：

- **闭合（Closed）：** 初始状态。请求通过并到达工具。计数器跟踪近期故障。如果在时间窗口内故障超出阈值，断路器会跳闸至开启状态。
- **开启（Open）：** 请求在配置的“冷却”期内立即被拒绝，不尝试工具调用。这可以防止持续打击一个故障服务。冷却期结束后，断路器转换到半开启状态。
- **半开启（Half-Open）：** 允许有限数量的测试请求通过。如果这些请求成功，断路器会重置为闭合状态。如果它们失败，则会再次跳闸至开启状态，重新开始冷却。

实现断路器通常涉及在直接工具调用之外维护状态，这通常在智能体框架或专门的工具管理服务中进行。

> 断路器模式中的状态转换，用于管理频繁发生故障的工具交互。

### 实施良好实践

- **工具封装器：** 在封装函数或类中抽象核心工具交互逻辑。这些封装器应封装错误检测、重试逻辑（如`execute_api_call_with_retry` 示例）、架构验证，并标准化返回给智能体控制器的成功和错误输出格式。
- **清晰的错误传播：** 确保错误信息（类型、消息、状态码、重试次数）从工具封装器清晰地传播回智能体的推理 (inference)循环。避免使用通用错误消息。
- **可观测性：** 为所有工具交互实现全面的日志记录和追踪。记录输入、输出（或错误详情）、状态码、时间以及重试次数。这对于调试涉及多次工具调用的复杂智能体行为是必不可少的。可以集成OpenTelemetry等工具进行分布式追踪。
- **配置：** 使重试次数、延迟、超时和断路器阈值等参数 (parameter)可配置，以便根据特定的API行为和应用需求进行微调 (fine-tuning)。
- **测试：** 专门为您的错误处理逻辑开发单元和集成测试。使用模拟API（例如，在Python中使用`requests-mock`等库）来模拟不同的故障场景（超时、特定状态码、格式错误的响应），并验证您的重试、备用和错误报告机制按预期工作。考虑在预生产环境中进行故障注入测试。

通过系统地处理外部交互中潜在的故障，您可以构建出在动态环境中可靠执行复杂多步任务的、更具韧性和能力的智能体系统。这种韧性是生产就绪型智能体应用的标志。

## 参考资料

- [Release It! Design and Deploy Production-Ready Software](https://books.google.com/books/about/Release_It_Design_and_Deploy_Production.html?id=l6JRDwAAQBAJ) — Michael T. Nygard (2018)
  Publisher: The Pragmatic Bookshelf
  一本关于设计和部署弹性软件系统的基础书籍，涵盖了在分布式环境中处理故障的重试、超时和优雅降级等模式。
- [Retrying on errors with exponential backoff](https://cloud.google.com/storage/docs/exponential-backoff) — Google Cloud Documentation (2024)
  Publisher: Google Cloud
  官方的Google Cloud文档，提供了在分布式系统中为瞬时错误实现指数退避和抖动重试机制的最佳实践。

---

[上一节](04-%E5%B7%A5%E5%85%B7%E6%8F%8F%E8%BF%B0%E4%B8%8E%E9%80%89%E6%8B%A9%E6%9C%BA%E5%88%B6.md) · [下一节](06-%E8%87%AA%E6%88%91%E6%A0%A1%E6%AD%A3%E4%B8%8E%E8%AE%A1%E5%88%92%E4%BC%98%E5%8C%96.md)
