---
course: "python-llm-workflows"
chapter: "interacting-llm-apis-python"
lesson: "api-authentication-methods"
sourceId: 5125
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-3-interacting-llm-apis-python/api-authentication-methods"
title: "认证方式"
description: "了解安全认证 API 请求的不同方式。"
order: 5
plots: []
sourceHash: "e38ad00e357ee63466eb701b1f3c1b2a1ddd4bf756d98859651757310855a127"
sourceCorrections: []
---

当你的 Python 应用与大型语言模型（LLM）提供商的 API 沟通时，它需要证明自己有权限这样做。API 通常不是开放的通道；它们需要身份凭证来管理访问、追踪计费使用情况并保证安全。这个过程称为认证。没有正确的认证，你的 API 请求很可能会被拒绝。

大多数 LLM 提供商使用 **API 密钥** 作为服务器到服务器认证的主要方式。API 密钥通常是一个在你注册服务时分配给你的唯一、长的字符串。可以把它看作是专门用于你应用程序的密码。

### 在 HTTP 请求头中使用 API 密钥

发送 API 密钥最常见的方式是在你的请求的 HTTP 请求头中。许多 API 遵循使用 `Authorization` 请求头和 `Bearer` 令牌方案的惯例。

以下是使用 `requests` 库在请求中包含 API 密钥的示例：

```python
import requests
import os

# 从环境变量加载 API 密钥以提高安全性
API_KEY = os.getenv("LLM_PROVIDER_API_KEY")
API_ENDPOINT = "https://api.example-llm-provider.com/v1/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "model-name",
    "prompt": "Explain API authentication simply.",
    "max_tokens": 50
}

try:
    response = requests.post(API_ENDPOINT, headers=headers, json=data)
    response.raise_for_status() # 对于错误响应（4XX 或 5XX）抛出 HTTPError

    # 处理成功响应
    result = response.json()
    print(result)

except requests.exceptions.RequestException as e:
    print(f"发生 API 请求错误：{e}")
except Exception as e:
    print(f"发生意外错误：{e}")
```

在此示例中：

1. 我们从环境变量 `LLM_PROVIDER_API_KEY` 中获取 API 密钥。**切勿将 API 密钥直接硬编码在你的源代码中。** 这会带来很大的安全风险。请参阅第二章，了解管理密钥的安全方式。
2. 我们创建一个 `headers` 字典。
3. 字典中的 `Authorization` 键被设置为字符串 `Bearer` ，后面跟着实际的 API 密钥。`Bearer` 后面的空格很重要。
4. 这个 `headers` 字典被传递给 `requests.post` 函数。

一些 API 可能使用不同的请求头名称，例如 `X-API-Key` 或在其文档中说明的自定义名称。务必查阅具体的 LLM 提供商的 API 文档，以了解正确的请求头和格式。

> 涉及通过 Authorization 请求头进行 API 密钥认证的请求流程。

### 使用客户端库进行认证

正如上一节所述，LLM 提供商通常会提供官方的 Python 客户端库（SDK）以简化交互。这些库通常会为你处理认证的细节。

通常，你只需配置一次库，提供你的 API 密钥，这通常是在初始化客户端对象时完成的。然后，该库会自动以正确的格式将密钥包含在所有后续请求中。

许多库都设计为自动在特定的环境变量中查找 API 密钥（例如，OpenAI 库使用 `OPENAI_API_KEY`，Anthropic 库使用 `ANTHROPIC_API_KEY`）。

```python
# 使用 'llm_provider_sdk' 库的示例
import os
# pip install llm_provider_sdk
# 假设已设置 LLM_PROVIDER_API_KEY 环境变量
# （请查阅具体库的文档以获取确切的变量名）

try:
    # 库可能会自动从环境中获取标识符
    # 或者你可以明确地初始化它：
    # client = llm_provider_sdk.Client(api_key=os.getenv("LLM_PROVIDER_API_KEY"))

    # 如果库默认使用环境变量：
    import llm_provider_sdk
    client = llm_provider_sdk.Client() # 隐式使用环境变量

    response = client.completions.create(
        model="model-name",
        prompt="Explain API authentication simply using a client library.",
        max_tokens=60
    )
    print(response)

except ImportError:
    print("请安装所需的 SDK: pip install llm_provider_sdk")
except Exception as e:
    # 捕获可能的错误，如缺少 API 密钥或连接问题
    print(f"发生错误：{e}")
```

使用官方客户端库通常是推荐的方式，因为它封装了 HTTP 请求和认证请求头的底层细节，带来更清晰、更易于维护的代码。它也保证你遵循了提供商预设的交互方式。

### 其他认证机制

尽管 API 密钥在直接访问 LLM API 中普遍存在，但你偶尔也会遇到其他方式，尤其是在企业或云集成场景中：

- **OAuth 2.0:** 当应用程序需要代表最终用户操作时（例如，聊天机器人通过 Google Drive API 访问用户文档），这种方式更常见。它涉及用户同意和令牌交换。对于本课程关注的直接服务器到 LLM 通信来说，它不那么常见。
- **云提供商 IAM/服务账户：** 当使用集成到主要云平台（AWS Bedrock、Google Vertex AI、Azure OpenAI Service）的 LLM 时，认证通常通过云提供商的身份和访问管理（IAM）系统处理，使用服务账户凭证或托管身份。对应的云 SDK 通常会管理此认证流程。

对于从你的 Python 后端调用标准 LLM API 的主要任务，通过环境变量安全管理并可能通过客户端库封装的 API 密钥将是你的主要方式。始终优先考虑安全：避免在代码或版本控制中暴露密钥，并使用环境变量或合适的密钥管理方案，尤其是在生产环境中。

## 参考资料

- [Requests: HTTP for Humans™](https://docs.python-requests.org/en/latest/) — Kenneth Reitz and the Requests Community (2023)
  Python `requests`库的官方文档，解释了如何发起HTTP请求、设置请求头和处理响应。
- [Hypertext Transfer Protocol (HTTP/1.1): Authentication](https://datatracker.ietf.org/doc/html/rfc7235) — R. Fielding and J. Reschke (2014)
  Journal: RFC 7235; Publisher: Internet Engineering Task Force (IETF); DOI: [10.17487/RFC7235](https://doi.org/10.17487/RFC7235)
  此RFC定义了HTTP的`Authorization`请求头和认证框架，包括用于API密钥认证的`Bearer`令牌方案。
- [OWASP API Security Top 10](https://owasp.org/API-Security/) — Erez Yalon, Inon Shkedy, Paulo Silva (2023)
  Publisher: OWASP Foundation
  API安全风险和缓解策略的指南，包含了API认证和安全凭证处理的基本实践。
