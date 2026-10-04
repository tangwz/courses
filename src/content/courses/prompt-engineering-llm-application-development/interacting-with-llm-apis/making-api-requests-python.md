---
course: "prompt-engineering-llm-application-development"
chapter: "interacting-with-llm-apis"
lesson: "making-api-requests-python"
sourceId: 3025
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-4-interacting-with-llm-apis/making-api-requests-python"
title: "使用 Python 发送 API 请求"
description: "使用 Python 库（如 requests 或特定提供商的 SDK）调用 LLM API。"
order: 3
plots: []
sourceHash: "16727809c88dbd987e3820915771fd36f39f4de643f29ec2a3131902503281d4"
sourceCorrections: []
---

通过编程发送提示并接收回复，需要从你的应用程序向 LLM 提供商的 API 端点发出 HTTP 请求。Python 及其丰富的库生态系统为此类交互提供了好方法。

在 Python 中进行 API 调用主要有两种方法：

1. **使用通用 HTTP 库：** 像 `requests` 这样的库允许你手动构建和发送 HTTP 请求。这让你能精确控制请求的细节。
2. **使用特定提供商的软件开发工具包 (SDK)：** 大多数主要的 LLM 提供商（如 OpenAI、Anthropic、Google）都提供官方 Python SDK。这些库封装了底层的 API 调用，提供了针对其特定服务定制的便捷功能和抽象。

### 使用 `requests` 库进行调用

`requests` 库因其简洁和强大，是 Python 中进行 HTTP 请求的常用选择。如果你尚未安装它，可以使用 pip 来安装：

```bash
pip install requests
```

要使用 `requests` 与 LLM API 交互，你通常需要发送一个 HTTP POST 请求。此请求包含：

1. **URL：** LLM 服务提供的特定 API 端点（例如，`https://api.openai.com/v1/chat/completions`）。
2. **Headers（请求头）：** 关于请求的信息，其中最重要的是：
   - `Authorization`：包含你的 API 密钥，用于认证（通常以 `Bearer` 为前缀）。
   - `Content-Type`：指定你正在发送的数据格式，几乎总是 `application/json`。
3. **Body (Payload)（请求体）：** 发送给 API 的实际数据，格式为 JSON。这包括提示、模型选择以及 `temperature` 和 `max_tokens` 等生成参数 (parameter)。

让我们用一个针对 LLM API 端点的例子来说明。假设你的 API 密钥存储在一个名为 `LLM_API_KEY` 的环境变量中。

```python
import requests
import os
import json

# 从环境变量中获取 API 密钥
api_key = os.getenv("LLM_API_KEY")
if not api_key:
    raise ValueError("未找到 API 密钥。请设置 LLM_API_KEY 环境变量。")

# 定义 API 端点
api_url = "https://api.example-llm-provider.com/v1/generate" # 替换为实际端点

# 设置请求头
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# 构建请求体 (payload)
payload = {
    "model": "example-model-v2", # 指定所需的 LLM
    "prompt": "解释 HTTP GET 和 POST 请求之间的区别。",
    "max_tokens": 150,
    "temperature": 0.7,
    "stop_sequences": ["\n"] # 可选：生成应停止的序列
}

try:
    # 发送 POST 请求
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    # 检查请求是否成功（状态码 200 OK）
    response.raise_for_status() # 对于不良响应（4xx 或 5xx）引发 HTTPError

    # 解析 JSON 响应
    result = response.json()
    print("API 响应：")
    print(result)

    # 提取生成的文本（结构取决于 API）
    if "choices" in result and len(result["choices"]) > 0:
        generated_text = result["choices"][0].get("text", "未找到文本")
        print("\n生成的文本：")
        print(generated_text.strip())
    else:
        print("\n无法从响应中提取生成的文本。")

except requests.exceptions.RequestException as e:
    print(f"API 请求期间发生错误：{e}")
except json.JSONDecodeError:
    print(f"未能解码 JSON 响应：{response.text}")
except Exception as e:
    print(f"发生意外错误：{e}")
```

在此示例中：

- 我们导入了必要的库（`requests`、`os`、`json`）。
- 我们从环境变量中安全地获取 API 密钥。切勿将 API 密钥直接硬编码在源代码中。
- 我们定义了 `headers` 字典，包括 `Authorization` 令牌和 `Content-Type`。
- 我们创建了包含提示和模型参数的 `payload` 字典。确切的结构和参数名称会因具体的 LLM API 提供商而异。请查阅他们的文档。
- `requests.post()` 发送请求。我们传递 URL、请求头和 JSON 序列化的请求体（`json.dumps(payload)`）。
- `response.raise_for_status()` 对于基本的错误检查很重要。如果 API 返回错误状态码（如 401 Unauthorized、429 Rate Limit Exceeded 或 500 Internal Server Error），它将引发异常。
- 如果成功，`response.json()` 会将 JSON 响应体解析为一个 Python 字典。
- 随后我们尝试提取相关的生成文本，并考虑到 API 返回的特定结构。
- 错误处理（`try...except`）被包含在内，以捕获潜在的网络问题、JSON 解析问题或其他异常。

使用 `requests` 提供了透明度，并且适用于任何基于 HTTP 的 API，但它要求你手动处理 JSON 序列化和请求头构建等细节。

### 使用提供商 SDK 进行调用

LLM 提供商通常提供 Python SDK，以简化与其 API 的交互。这些库抽象了底层的 HTTP 请求细节，提供了更具 Python 风格的接口。

例如，使用 OpenAI Python 库（通过 `pip install openai` 安装），进行聊天完成请求的代码如下所示：

```python
import os
from openai import OpenAI, OpenAIError

# SDK 通常会自动在环境变量中查找 API 密钥
# （例如 OPENAI_API_KEY），或者你可以在客户端初始化时传入。
# 请确保你的环境中设置了 OPENAI_API_KEY。
try:
    client = OpenAI() # API 密钥会从 OPENAI_API_KEY 环境变量中隐式读取

    # 使用 SDK 的方法进行 API 调用
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "解释编程中递归的概念。",
            }
        ],
        model="gpt-3.5-turbo", # 指定模型
        max_tokens=100,
        temperature=0.8,
    )

    # 通过 SDK 对象访问响应内容
    print("API 响应对象：")
    # print(chat_completion) # 原始 SDK 对象可能很冗长

    if chat_completion.choices:
        response_content = chat_completion.choices[0].message.content
        print("\n生成的文本：")
        print(response_content.strip())
    else:
        print("未找到响应选项。")

    # 如果可用，打印使用信息
    if chat_completion.usage:
        print("\n令牌用量：")
        print(f"提示令牌数：{chat_completion.usage.prompt_tokens}")
        print(f"完成令牌数：{chat_completion.usage.completion_tokens}")
        print(f"总令牌数：{chat_completion.usage.total_tokens}")

except OpenAIError as e:
    print(f"发生 OpenAI API 错误：{e}")
except Exception as e:
    print(f"发生意外错误：{e}")
```

与使用 `requests` 相比，请注意以下区别：

- 我们从 `openai` 库实例化一个 `client` 对象。认证通常由 SDK 隐式处理（例如，从环境变量中读取 `OPENAI_API_KEY`）。
- 我们调用一个为聊天完成端点设计的特定方法（`client.chat.completions.create`）。
- `model`、`messages`、`max_tokens` 和 `temperature` 等参数 (parameter)作为函数参数传递。
- SDK 处理 HTTP 请求的形成、请求头的设置以及数据的序列化。
- 响应作为 Python 对象返回（通常是 Pydantic 模型或类似结构），使得访问 `choices[0].message.content` 或用量统计等特定字段更加容易。
- SDK 通常定义特定的异常类型（如 `OpenAIError`），以便更好地处理与 API 相关的错误。

### 你应该使用哪种方法？

- **在以下情况下使用 `requests`：**
  - 你需要最大限度地控制 HTTP 请求的细节。
  - 你正在与没有专用 Python SDK 的 API 交互。
  - 你偏好更少的依赖，或者想更清楚地理解底层协议。
- **在以下情况下使用提供商 SDK：**
  - 你的目标 LLM 提供商有可用的 SDK。
  - 你偏好更高层、更便捷的抽象。
  - 你希望获得内置的认证处理、响应解析和提供商特定功能。
  - 你正在与常常构建在这些 SDK 之上的框架（如 LangChain）进行集成。

对于大多数应用程序开发而言，推荐使用官方提供商 SDK，因为它能简化开发和维护。然而，了解如何使用 `requests` 对于调试、处理不太常见的 API，或当你需要直接控制网络交互时非常有价值。无论采用哪种方法，安全处理 API 密钥和错误处理都是可靠应用程序开发的重要组成部分。

## 参考资料

- [\`requests\` Documentation](https://requests.readthedocs.io/en/latest/) — Kenneth Reitz and `requests` Contributors (2024)
  Python `requests`库的官方文档，提供了关于发起HTTP请求的全面指南和API参考。
- [OpenAI Python Library](https://platform.openai.com/docs/libraries/python-sdk) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI Python客户端库的官方文档，详细介绍了如何通过其API与各种LLM功能进行交互。
- [HTTP | MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP) — MDN Contributors (2024)
  Publisher: Mozilla
  关于超文本传输协议（HTTP）的全面指南，涵盖了请求方法、头部、状态码以及Web通信的基本原理。
- [Automate the Boring Stuff with Python, 2nd Edition: Practical Programming for Total Beginners](https://automatetheboringstuff.com/2e/chapter12/) — Al Sweigart (2019)
  Publisher: No Starch Press; Pages: Chapter 12
  一份实用指南，其中包含使用Python的`requests`库与Web API交互的部分，适合初学者。
