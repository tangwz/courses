---
course: "python-llm-workflows"
chapter: "interacting-llm-apis-python"
lesson: "api-requests-with-requests"
sourceId: 5115
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-3-interacting-llm-apis-python/api-requests-with-requests"
title: "使用requests库发送API请求"
description: "使用Python `requests` 库向LLM API发送提示。"
order: 2
plots: []
sourceHash: "1c27b31d2f18426b5658a854267768762e31d01f4749f8ea7d68c474d83fae54"
sourceCorrections: []
---

从Python与LLM API交互时，一个主要任务是发送包含提示的请求并接收LLM的响应。这种通信通常依赖于发出HTTP请求。实现这一目标的直接且常见方法之一是使用标准的Python `requests` 库。该库提供了一个简单的接口用于发出HTTP请求，这对于与大多数基于网络的LLM API通信必不可少。

尽管许多LLM提供商提供了专用的Python客户端库（我们稍后会介绍），但直接使用 `requests` 有其价值，原因如下：

- 这有助于你理解底层的通信协议（HTTP）。
- 它能普遍适用于任何基于HTTP的API，即使没有专用库或库未及时更新。
- 它通常足以应对简单的交互或快速测试。

### `requests` API请求的构成

要进行API调用，我们通常需要四个主要信息：

1. **URL：** 我们想要交互的API服务的具体网址（端点）。这由LLM供应商提供。
2. **HTTP方法：** 请求的类型。对于发送数据以生成文本，这几乎总是 `POST`。
3. **请求头：** 随请求发送的附加信息。这通常包括：
   - `Content-Type`：指定发送数据的格式（例如，`application/json`）。
   - `Authorization`：包含用于验证请求的凭据，通常是API密钥（例如，`Bearer YOUR_API_KEY`）。请记住，要像第2章讨论的那样，安全地处理API密钥。
4. **请求体（Body）：** 发送给API的实际数据。对于LLM，这包含提示、模型选择和其他生成参数 (parameter)，通常格式为JSON。

### 发送POST请求

`requests` 库使用 `requests.post()` 函数使发送 `POST` 请求变得简单。让我们通过一个例子来说明。假设我们有一个API端点 `https://api.example-llm-provider.com/v1/completions` 和我们的API密钥。

```python
import requests
import os
import json

# 最佳实践：从环境变量或安全配置中加载API密钥
# 假设你的环境中已设置API_KEY（参见第2章）
api_key = os.getenv("EXAMPLE_LLM_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the EXAMPLE_LLM_API_KEY environment variable.")

# 1. 定义API端点URL
api_url = "https://api.example-llm-provider.com/v1/completions" # 替换为实际的端点

# 2. 设置请求头
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 3. 准备请求的载荷（数据）
# 参数名称（例如，'model'、'prompt'、'max_tokens'）因提供商而异
payload = {
    "model": "model-name-123", # 指定所需的LLM模型
    "prompt": "Translate the following English text to French: 'Hello, world!'",
    "max_tokens": 50,         # 限制响应长度
    "temperature": 0.7        # 控制创造性（0.0=确定性，>1.0=更具创造性）
}

# 4. 发送POST请求
try:
    response = requests.post(api_url, headers=headers, json=payload) # 使用 json= 参数可自动序列化载荷，并设置Content-Type（如果尚未设置）

    # 5. 检查响应状态并处理结果
    response.raise_for_status() # 对于错误的响应（4xx或5xx），会引发HTTPError

    # 假设API返回JSON格式数据
    result = response.json()
    print("API Response:")
    print(json.dumps(result, indent=2)) # 漂亮打印JSON响应

    # 提取生成的文本（结构取决于具体的API）
    # 这是一种结构；请查看你的提供商文档
    if "choices" in result and len(result["choices"]) > 0:
        generated_text = result["choices"][0].get("text", "No text found")
        print("\nGenerated Text:")
        print(generated_text.strip())
    else:
        print("\nCould not find generated text in the expected format.")

except requests.exceptions.RequestException as e:
    print(f"An error occurred during the API request: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

在此示例中：

- 我们导入了必要的库（`requests`、`os`、`json`）。
- 我们从环境变量中安全地获取API密钥。
- 我们定义了 `api_url`、`headers` 和 `payload` 字典。请注意，这里使用f-string将API密钥插入到 `Authorization` 请求头中。
- 核心操作是 `requests.post(api_url, headers=headers, json=payload)`。直接使用 `json` 参数 (parameter)会告诉 `requests` 自动将 `payload` 字典转换为JSON字符串，并将 `Content-Type` 请求头设置为 `application/json`（尽管为了清晰起见，我们也明确设置了它）。
- `response.raise_for_status()` 是一个重要步骤。它检查HTTP状态码是否表示成功（例如200 OK）。如果API返回错误代码（例如401 Unauthorized、429 Rate Limit Exceeded、500 Internal Server Error），此方法将引发 `requests.exceptions.HTTPError`。
- 如果请求成功，`response.json()` 会将响应体中的JSON内容解析为Python字典。
- 然后，我们尝试提取响应的相关部分，这通常涉及处理嵌套结构（例如 `result["choices"][0]["text"]`）。确切的结构因API而异，因此请务必查阅提供商的文档。
- 使用 `try...except` 进行的基本错误处理会捕获潜在的网络问题（`requests.exceptions.RequestException`）或其他问题。

### 常见的请求体参数 (parameter)

`payload` 字典是你控制LLM行为的地方。虽然具体名称可能不同，但常见参数包括：

- `model`：（字符串）你想要使用的特定语言模型的标识符（例如，`gpt-3.5-turbo`、`claude-2.1`）。
- `prompt` 或 `messages`：（字符串或列表）输入文本。一些API使用 `prompt` 进行简单的文本续写，而另一些则使用 `messages` 列表进行类似聊天的交互，通常包含角色（`user`、`assistant`、`system`）。
- `max_tokens`：（整数）模型在响应中应生成的最大标记 (token)数（大致相当于单词或单词的一部分）。有助于控制成本和响应长度。
- `temperature`：（浮点数，通常为0.0到2.0）控制输出的随机性。较低的值（例如0.2）使输出更集中和确定，而较高的值（例如0.8）使其更具创造性和多样性。
- `top_p`：（浮点数，0.0到1.0）是温度参数的替代方案，用于控制随机性（核采样）。模型只考虑累积概率质量超过 `top_p` 的标记。
- `stop`：（字符串或字符串列表）API应停止生成后续标记的字符序列。

始终参考你正在使用的特定LLM提供商的文档，以了解可用的参数、它们的名称、期望值和默认行为。

使用 `requests` 库提供了一种与LLM API交互的基本方法。它让你直接控制HTTP通信，并且即使在使用更专业的库时也是一项有价值的技能，因为这些库通常基于这些基本原理构建。在接下来的部分中，我们将更详细地研究如何处理不同类型的API响应和错误，然后考察主要LLM提供商提供的官方客户端库。

## 参考资料

- [OpenAI Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat/create) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI 聊天补全端点的官方 API 参考，展示了请求和响应结构、常见负载参数（如 `model`、`messages`、`max_tokens`、`temperature`）以及用于 LLM 交互的身份验证方法。
