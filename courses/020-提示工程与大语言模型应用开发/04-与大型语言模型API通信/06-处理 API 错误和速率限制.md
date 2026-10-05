# 处理 API 错误和速率限制

来源：[原文](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-4-interacting-with-llm-apis/handling-api-errors-rate-limits)

[返回章节目录](README.md) · [返回课程目录](../README.md)

与 LLM API 等外部服务交互会带来不确定性。网络问题、服务器端问题、无效请求或仅仅发送过多请求过快都可能导致失败。好的应用程序会预见这些问题，并包含妥善处理它们的机制。未能如此可能导致应用程序崩溃、用户体验不佳以及潜在的数据丢失。

### 常见的 API 错误

当你使用 Python 的 `requests` 等库进行 API 调用时，响应对象包含有用的信息，最值得注意的是 HTTP 状态码。这些代码表明了你的请求结果。虽然你理想情况下希望得到 `200 OK` 状态，但你需要为其他情况做好准备：

- **4xx 客户端错误：** 这些表明你的请求存在问题。

  - `400 错误请求`: 通常表示你的请求负载格式不正确、缺少必需字段或包含无效参数 (parameter)值（例如，不支持的模型名称、无效的温度设置）。请仔细检查 API 文档和你的请求结构。响应正文通常包含有关错误的具体信息。
  - `401 未授权`: 你的 API 密钥丢失、无效或已过期。请仔细检查你提供认证令牌或密钥的方式。确保它没有被撤销或更改。
  - `403 禁止访问`: 你已通过身份验证，但没有权限访问请求的资源或执行该操作。这可能与你的订阅级别或特定的 API 端点访问规则有关。
  - `404 未找到`: 请求的端点或资源不存在。请验证 API 端点 URL。
  - `429 请求过多`: 你在给定时间内超出了允许的请求数量。这是一个速率限制错误，下面会详细介绍。
- **5xx 服务器错误：** 这些表明 API 提供方存在问题。

  - `500 内部服务器错误`: 一个通用错误，表明服务器端出现问题。你通常无法立即做什么，只能稍后重试。
  - `502 错误网关` / `503 服务不可用` / `504 网关超时`: 这些表明服务器暂时过载、停机维护或无法与上游服务通信。延迟后重试通常是最好的方法。

以下是使用 `requests` 库检查状态码的一个基本 Python 示例：

```python
import requests
import os
import json

# 假设 API_ENDPOINT 和 API_KEY 已在其他地方定义
# API_ENDPOINT = "https://api.example-llm-provider.com/v1/completions"
# API_KEY = os.getenv("LLM_API_KEY") 

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "some-model-name",
    "prompt": "Translate 'hello' to French:",
    "max_tokens": 10
}

try:
    response = requests.post(API_ENDPOINT, headers=headers, json=data, timeout=30) # 设置超时

    # 检查请求是否成功（状态码 2xx）
    if response.status_code == 200:
        result = response.json()
        print("成功:", result)
    # 处理特定客户端错误
    elif response.status_code == 400:
        print(f"错误 400: 错误请求. 响应: {response.text}")
        # 记录错误详情以便调试
    elif response.status_code == 401:
        print("错误 401: 未授权. 检查你的 API 密钥.")
        # 可能停止应用程序或通知管理员
    elif response.status_code == 429:
        print("错误 429: 请求过多. 请稍后重试.")
        # 实现重试逻辑（见下文）
    # 处理服务器错误
    elif response.status_code >= 500:
        print(f"错误 {response.status_code}: 服务器错误. 响应: {response.text}")
        # 实现重试逻辑
    # 处理其他意外错误
    else:
        print(f"意外错误: {response.status_code}. 响应: {response.text}")
        response.raise_for_status() # 对其他非 200 代码抛出异常

except requests.exceptions.RequestException as e:
    print(f"网络或请求错误发生: {e}")
    # 处理网络问题、超时等
```

务必检查响应正文以获取详细的错误消息，特别是对于 `400 Bad Request` 错误，因为 API 提供方通常会包含有关问题原因的具体信息。

### 了解速率限制

API 提供商会实施速率限制，以保证公平使用、防止滥用，并维持所有用户的服务稳定。这些限制规定了你在特定时间段内可以发出的请求数量或可以处理的数据量（令牌）（例如，每分钟请求数、每天令牌数）。

超出速率限制通常会导致 `429 Too Many Requests` 错误。API 响应还可能包含指示何时可以重试的标头（例如，`Retry-After` 标头指定等待的秒数，或 `X-RateLimit-Reset` 指示限制重置的时间）。

### 处理错误和速率限制的策略

仅仅在发生错误时失败是很少能接受的。你需要采取策略来增强应用程序的韧性。

#### 1. 错误检查和日志记录

如 Python 示例所示，第一步始终是检查响应的 HTTP 状态码。根据状态码，你可以决定采取适当的行动。记录错误，包括状态码、响应正文（如果可用）以及导致错误的请求。这些信息对于调试非常有用。

#### 2. 重试机制

对于像 `5xx` 服务器问题或 `429` 速率限制这样的瞬时错误，延迟后自动重试请求是一种常见且有效的方法。然而，立即重试通常适得其反，特别是对于速率限制。

**指数退避：** 一种广泛使用的重试策略是指数退避。你不是在固定延迟后重试，而是在每次失败尝试后以指数方式增加等待时间。这可以避免在 API 服务出现问题时使其过载。

这是基本逻辑：

1. 发出 API 请求。
2. 如果它因可重试错误而失败（例如 429, 500, 503）：
   - 等待计算出的延迟（例如，$基础延迟 \times 2^{尝试次数}$）。
   - 可选地在延迟中添加一些随机抖动，以避免多个客户端同时重试（例如，$延迟 = 基础延迟 \times 2^{尝试次数} + 随机数(0, 1)$ 秒）。
   - 增加尝试次数。
   - 如果尚未达到最大重试次数，则返回第 1 步。
3. 如果请求成功或因不可重试错误（例如 400, 401）而失败，则停止。
4. 如果达到最大重试次数，报告持续的失败。

> 流程图说明了使用指数退避处理 API 错误的常见重试逻辑。

这是一个指数退避的 Python 实现：

```python
import time
import random
import requests

def call_llm_api_with_retry(api_endpoint, headers, data, max_retries=5, base_delay=1.0):
    """在可重试错误发生时，使用指数退避调用 LLM API。"""
    retries = 0
    delay = base_delay
    while retries < max_retries:
        try:
            response = requests.post(api_endpoint, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                return response.json() # 成功

            # 检查可重试错误
            if response.status_code == 429 or response.status_code >= 500:
                print(f"可重试错误 {response.status_code}. 在 {delay:.2f} 秒后重试...")
                time.sleep(delay)
                retries += 1
                delay = base_delay * (2 ** retries) + random.uniform(0, 1) # 带抖动的指数退避
                continue # 重试循环
            else:
                # 不可重试的客户端错误（4xx，429除外）或意外代码
                print(f"不可重试错误: {response.status_code}. 响应: {response.text}")
                response.raise_for_status() # 抛出异常
                return None # 如果 raise_for_status() 有效，不应到达此处

        except requests.exceptions.RequestException as e:
            print(f"网络错误: {e}. 在 {delay:.2f} 秒后重试...")
            time.sleep(delay)
            retries += 1
            delay = base_delay * (2 ** retries) + random.uniform(0, 1) # 带抖动的指数退避

    print(f"已超过最大重试次数 ({max_retries}). API 调用失败.")
    return None # 指示所有重试后失败

# 示例用法（假设 headers 和 data 如前所述已定义）
# result = call_llm_api_with_retry(API_ENDPOINT, headers, data)
# if result:
#     print("重试后 API 调用成功:", result)
# else:
#     print("API 调用永久失败.")
```

#### 3. 遵守速率限制标头

请注意 API 响应中特定的速率限制标头，例如：

- `Retry-After`: 指定在发出下一个请求之前需要等待的秒数（通常与 429 或 503 错误一起发送）。
- `X-RateLimit-Limit`: 在时间窗口内允许的最大请求数。
- `X-RateLimit-Remaining`: 当前窗口中剩余的请求数。
- `X-RateLimit-Reset`: 速率限制窗口重置的时间（通常为 Unix 时间戳格式）。

你可以使用这些标头来实施更智能的速率限制逻辑，在接近限制时主动减慢请求速度，或精确等待 `Retry-After` 指定的所需持续时间。

#### 4. 监控

监控应用程序的 API 使用情况、成本和错误率。大多数 API 提供商都提供仪表板来跟踪使用情况。为高错误率或接近速率限制设置警报可以帮助你及早发现和处理问题。

通过实施错误检查、像指数退避这样的智能重试策略，并注意速率限制，你可以构建出更可靠的应用程序，它们能够妥善处理与第三方 API 交互时不可避免的问题。

## 参考资料

- [HTTP response status codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status) — MDN Web Docs contributors (2024)
  HTTP状态码的全面指南，对理解API响应结果至关重要。
- [Release It! Design and Deploy Production-Ready Software](https://pragprog.com/titles/rr2/release-it-second-edition/) — Michael T. Nygard (2018)
  Publisher: The Pragmatic Programmers
  一本关于构建弹性软件系统的基础著作，涵盖故障处理模式、重试以及分布式环境中稳定性设计。
- [Requests: HTTP for Humans](https://requests.readthedocs.io/en/latest/) — Kenneth Reitz and others (2024)
  `requests`库的官方文档，涵盖Python中的异常处理、超时和通用的HTTP客户端使用。

---

[上一节](05-%E5%A4%84%E7%90%86API%E5%93%8D%E5%BA%94.md) · [下一节](07-%E6%B5%81%E5%BC%8F%E5%93%8D%E5%BA%94.md)
