---
course: "building-advanced-llm-agent-tools"
chapter: "integrating-external-apis-tools"
lesson: "api-rate-limits-retries-tools"
sourceId: 6590
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-4-integrating-external-apis-tools/api-rate-limits-retries-tools"
title: "处理 API 调用频率限制与重试"
description: "在您的基于 API 的工具中，管理 API 调用频率限制和实现可靠重试机制的策略。"
order: 3
plots: []
sourceHash: "d1dcd94e54822c52d2c8e4dc4d79ae9a1343e3d7705e63197b067c1b4f265ff0"
sourceCorrections: []
---

当您的 LLM 代理工具与外部 API 交互时，它们是他人系统中的“访客”。API 通常会实施使用策略，称为调用频率限制，以确保系统稳定和所有用户得到公平访问。达到这些限制，或遇到暂时的网络故障，都可能导致您的工具失效。为构建可靠的基于 API 的工具，您必须采取管理频率限制的策略，并智能地重试失败的请求。这不仅使您的工具更具弹性，而且确保它们在 API 环境中表现得像个“好公民”。

### 了解 API 调用频率限制

API 调用频率限制定义了客户端在特定时间段内可以向 API 发送请求的数量。例如，一个 API 可能允许每个用户每分钟发送 100 个请求，或每个 IP 地址每小时发送 1000 个请求。超出这些限制通常会导致 HTTP `429 Too Many Requests` 错误。API 通常通过 HTTP 响应头传达当前的频率限制状态。常见请求头包括：

- `X-RateLimit-Limit`: 当前时间窗口内允许的总请求数。
- `X-RateLimit-Remaining`: 当前时间窗口内剩余的请求数。
- `X-RateLimit-Reset`: 当前时间窗口重置并请求配额补足的时间（通常是 Unix 时间戳或剩余秒数）。

主动监测这些请求头能让您的工具预判并主动调整以适应频率限制。务必查阅 API 的文档，因为请求头名称和频率限制方案可能差别很大。

### 主动管理调用频率限制

首要的防护措施是设计您的工具来遵守已公布的频率限制。如果一个 API 允许每分钟 60 个请求，您的工具就不应尝试在几秒钟内发送 100 个请求。

- **客户端限速：** 您可以在请求之间引入延迟。例如，如果限制是每分钟 60 个请求，可以考虑在每个请求后引入 1 秒的延迟。
- **动态限速：** 一种更精巧的办法是根据 `X-RateLimit-Remaining` 和 `X-RateLimit-Reset` 请求头调整请求速率。如果剩余请求数很少，您的工具可以增加延迟。
- **请求排队：** 对于可能生成大量突发请求的工具，一个本地队列可以保存请求，并以符合 API 限制的速率分发它们。

缓存 API 响应，特别是对于不常变动的数据，可以大大减少发出的调用次数，从而帮助您的工具保持在频率限制内。

### 重试机制的必要性

即使采取了仔细的限速措施，请求仍然可能失败。短暂的网络问题、临时的服务器不可用（HTTP `503 Service Unavailable`）或达到频率限制（HTTP `429`）都是常见的原因。仅仅因为第一次错误就让工具操作失败，可能导致代理不够稳定。一个设计良好的重试机制可以显著提高基于 API 的工具的可靠性。

然而，并非所有错误都值得重试。如果不解决根本问题，重试因认证错误（HTTP `401 Unauthorized`）或格式错误的请求（HTTP `400 Bad Request`）而失败的请求通常毫无意义。重试通常适用于：

- 频率限制错误（例如，HTTP `429`）。
- 服务器端错误（例如，HTTP `500 Internal Server Error`、`502 Bad Gateway`、`503 Service Unavailable`、`504 Gateway Timeout`）。
- 网络连接错误。

在实现重试之前，一个重要的考量是**幂等性**。幂等操作是指无论执行多少次，其效果都与只执行一次相同。`GET`、`PUT` 和 `DELETE` 请求通常是幂等的。`POST` 请求通常创建新资源，因此通常不是幂等的。重试非幂等的 `POST` 请求可能导致意外的副作用，例如创建重复条目。如果您必须重试非幂等请求，请确保 API 提供了去重机制，或者您的应用逻辑可以处理潜在的重复数据。

### 实现带抖动的指数退避

一种简单的重试策略可能涉及每隔几秒重试一次失败的请求。这可能会有问题，特别是当许多客户端都在做同样的事情时，可能使服务器不堪重负（一个“惊群效应”问题）。一种更有效且被广泛采用的策略是**指数退避**。

采用指数退避策略，每次失败尝试后，重试前的延迟会呈指数级增长。例如，第一次重试可能等待 1 秒，第二次 2 秒，第三次 4 秒，以此类推，通常直到达到最大退避时间。

为进一步完善此策略，会加入**抖动**。抖动会为退避延迟引入一个小的随机时间量。这有助于防止多个可能同时开始退避的客户端同时重试，从而更均匀地分配负载。带有指数退避和抖动的延迟计算公式通常如下所示：

`delay = min(max_backoff, base_delay * (2 ** attempt_number)) + random_between(0, jitter_amount)`

以下是一个说明此逻辑的 Python 代码片段：

```python
import time
import random
# 假设使用 requests 库处理 HTTPError，尽管为简洁起见未在此代码片段中直接使用
# import requests 

MAX_RETRIES = 5
INITIAL_DELAY = 1  # 秒
MAX_DELAY = 60     # 秒
JITTER_FACTOR = 0.5 # 添加当前延迟部分的最多 50% 作为抖动

def make_api_call_with_retry(api_function, *args, **kwargs):
    retries = 0
    current_base_delay_component = INITIAL_DELAY
    
    while retries < MAX_RETRIES:
        try:
            # 这里是您进行实际 API 调用的地方
            # 例如：
            # response = api_function(*args, **kwargs) # 例如：requests.get(...)
            # response.raise_for_status() # 对 HTTP 错误代码（4xx 或 5xx）抛出异常
            
            # 为了演示，我们模拟一次调用
            print(f"Attempt {retries + 1}: Calling API...")
            if retries < 2: # 模拟前两次尝试失败
                # 在实际场景中，您会捕获特定的异常，例如
                # requests.exceptions.HTTPError, requests.exceptions.ConnectionError 等。
                # 并检查状态码（例如，如果 e.response.status_code == 429）
                raise ConnectionError(f"Simulated API failure on attempt {retries + 1}") 
            print("API call successful!")
            return {"data": "some_data_from_api"} # 模拟成功响应
        
        except (ConnectionError) as e: # 捕获特定的可重试异常。根据需要添加其他异常。
                                       # 例如，requests.exceptions.Timeout，特定的 HTTPError 状态码
            retries += 1
            if retries >= MAX_RETRIES:
                print(f"Error: {e}. Max retries reached. Failing.")
                raise Exception("API call failed after multiple retries") from e

            # 指数退避计算
            # current_base_delay_component = INITIAL_DELAY * (2 ** (retries -1)) # 这是一种方法
                                                                             # 或者直接将前一次加倍，以更平缓地开始
            if retries == 1:
                 current_base_delay_component = INITIAL_DELAY
            else:
                 current_base_delay_component = min(MAX_DELAY, current_base_delay_component * 2)


            # 添加抖动：当前基本延迟组件的随机部分
            jitter = random.uniform(0, JITTER_FACTOR * current_base_delay_component)
            
            actual_delay = min(MAX_DELAY, current_base_delay_component + jitter)
            
            print(f"Error: {e}. Retrying in {actual_delay:.2f} seconds (attempt {retries + 1}/{MAX_RETRIES})...")
            
            # 如果是 429 错误，并且 X-RateLimit-Reset 请求头可用，
            # 如果解析并认为合适，您可能选择休眠直到该特定重置时间
            # 这可能会覆盖 'actual_delay'。
            # 例如：
            # if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
            #     reset_header = e.response.headers.get('X-RateLimit-Reset')
            #     if reset_header:
            #         try:
            #             # 解析 reset_header（例如，Unix 时间戳或等待秒数）
            #             # seconds_to_wait_from_header = parse_reset_time(reset_header)
            #             # actual_delay = max(actual_delay, seconds_to_wait_from_header) 
            #             # 或者，如果重置时间很长，您仍然可以设置上限或记录警告。
            #         except ValueError:
            #             pass # 无法解析请求头，继续使用计算出的退避

            time.sleep(actual_delay)

    # 如果循环因达到最大重试次数而结束，则会到达此处，但已在内部处理
    # 为确保万无一失，如果它在没有返回或抛出异常的情况下退出循环，则重新抛出
    raise Exception("API call failed after exhausting retries without explicit success or failure handling in loop.")

# 实际 API 调用函数的示例占位符
# def my_actual_api_call(endpoint_url):
#   import requests
#   response = requests.get(endpoint_url, timeout=10) # 为请求本身设置超时
#   response.raise_for_status() # 对不良响应（4xx 或 5xx）抛出 HTTPError
#   return response.json()

# try:
#   # 将 'my_actual_api_call' 及其参数替换为您真实的 API 调用
#   result = make_api_call_with_retry(my_actual_api_call, "https://api.example.com/data")
#   print("最终结果:", result)
# except Exception as e:
#   print("最终错误:", e)
```

此示例模拟了 API 调用失败并应用了带抖动的指数退避策略。在一个工具中，您会将此逻辑与 `requests` 等 HTTP 客户端库结合起来，仔细检查 HTTP 状态码和特定的异常类型，以判断是否适合重试。许多 HTTP 客户端库也提供对各种重试策略的内置支持，这可以简化您的实现。

### 与 LLM 代理沟通

当一个 API 工具，尽管有重试逻辑，最终仍未能获得响应或执行操作时，它需要将此失败情况清晰地传达给 LLM 代理。工具返回的错误消息应该足够有信息量，以便 LLM 了解问题的性质（例如，“API 频率限制已超，重试后仍未恢复”或“外部服务多次尝试后仍暂时不可用”）。

这使得 LLM 能够做出更明智的决定。它可能会：

- 尝试使用替代工具。
- 推迟需要此工具的任务。
- 告知用户此问题并请求指导。
- 修改其计划，以应对不可用的数据或操作。

例如，如果一个旨在获取当前股票价格的工具因其 API 上持续的频率限制而失败，LLM 代理应被告知此情况。代理可以决定是否继续使用可能过时的数据（如果可用）、通知用户，或稍后重试，而不是反复调用失败的工具。

### 重试逻辑图

一个图形化的表示可以阐明重试决策过程：

> 这是处理带重试机制的 API 请求的决策流程图。它概述了检查成功、评估错误是否可重试、管理重试次数以及应用带抖动的退避延迟，并说明了 `X-RateLimit-Reset` 请求头的潜在用途。

通过认真实现频率限制处理和重试机制，您可以将您的 API 封装器从可能不稳定的组件转变为更可靠的工具。这会大大提高 LLM 代理在需要与外部环境交互时的效率和恢复能力。

## 参考资料

- [Error Retries and Exponential Backoff in AWS](https://docs.aws.amazon.com/general/latest/gr/api-retries.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  来自主要云提供商的实用指南，详细介绍了实现重试逻辑（包括指数退避和抖动）的最佳实践，以构建有弹性的API客户端。
- [Tenacity: Retry decorator for Python](https://github.com/jd/tenacity) — The Tenacity Developers (2024)
  一个流行的Python库的文档，它简化了向函数调用添加重试功能，并具有可配置的退避策略和抖动。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  这本书为构建稳健且容错的分布式系统提供了全面的原则，这些原则是弹性API交互策略的基础。
