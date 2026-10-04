---
course: "intro-large-language-models"
chapter: "using-pre-trained-llms"
lesson: "sending-first-api-request"
sourceId: 3738
sourceUrl: "https://apxml.com/zh/courses/intro-large-language-models/chapter-5-using-pre-trained-llms/sending-first-api-request"
title: "发送您的第一个API请求"
description: "一份关于如何向LLM服务发起简单API调用的分步指南。"
order: 5
plots: []
sourceHash: "2011188b24927b54c88be8ffa2f2ce057765dffe8fedc523f7e98a2b471df659"
sourceCorrections: []
---

应用程序编程接口（API）使代码能够与远程服务通信，包括那些托管强大大型语言模型的服务。通过这些“桥梁”发送请求，即可建立这种互动。

可以把发起API请求想象成向LLM服务发送一封非常具体、有条理的“信件”。这封“信件”需要有目的地地址、处理方式的指示、身份证明（认证）以及实际消息（您的提示词 (prompt)）。

### API请求的组成部分

与LLM API进行的大多数交互都使用网络标准协议，特别是HTTP请求。一个典型请求包含以下几个部分：

1. **端点URL：** 这是您发送请求的特定网址（URL）。每个LLM服务提供商都有自己的端点URL，用于不同的操作（例如生成文本）。例如，它可能看起来像 `https://api.example-llm-provider.com/v1/generate`。
2. **HTTP方法：** 这告诉服务器您想执行什么操作。向LLM发送数据以获取响应时，方法几乎总是 `POST`。
3. **头部：** 这些包含关于您请求的元信息。两个重要的头部是：
   - `Content-Type`：告诉服务器您在请求体中发送的数据格式。这通常是 `application/json`。
   - `Authorization`：提供您的凭据，通常是API密钥，以证明您有权限使用该服务。稍后会详细说明。
4. **正文（Payload）：** 这包含您要发送的实际数据，通常以JSON（JavaScript对象表示法）格式。它包含您的提示词 (prompt)和任何控制LLM行为的参数 (parameter)。

### 认证：您的API密钥

在发送请求之前，您几乎总是需要一个API密钥。您通常通过在LLM服务提供商处注册账户获得此密钥。

**重要：** 像对待密码一样对待您的API密钥。保守其秘密并确保其安全。切勿将其直接嵌入 (embedding)到可能公开共享的代码中（例如在版本控制系统中）。环境变量或安全的密钥管理工具是应用程序中处理密钥的更好方法。

API密钥通常包含在 `Authorization` 头部中。常见格式是 `Bearer YOUR_API_KEY`，其中 `YOUR_API_KEY` 替换为从提供商处获得的实际密钥。

### 构建请求正文（Payload）

请求正文是您放置LLM指令的地方。它是一个包含键值对的JSON对象。常见元素包括：

- `prompt`：（字符串）您希望LLM处理或响应的文本输入。
- `model`：（字符串）指定您想使用的特定LLM（例如，`example-model-v2` 或 `llama-3-8b-instruct`）。提供商通常提供具有不同能力和规模的多种模型。
- `max_tokens`：（整数）您希望生成响应包含的最大token数量（大致相当于单词或单词的一部分）。这有助于控制长度和成本。
- `temperature`：（浮点数，通常在0到1之间）控制输出的随机性。较低的值（例如0.2）使输出更集中和确定，而较高的值（例如0.8）使其更具创造性和多样性。

以下是一个简单JSON有效载荷的示例：

```json
{
  "model": "example-model-v1-instruct",
  "prompt": "Translate the following English text to French: 'Hello, world!'",
  "max_tokens": 50,
  "temperature": 0.7
}
```

### 发送请求：使用 `curl` 的示例

`curl` 是一个用于通过URL传输数据的命令行工具。它非常适合快速测试API端点。以下是您如何发送上述请求的示例（将 `YOUR_API_KEY` 和URL替换为实际值）：

```bash
# 将 YOUR_OPENAI_API_KEY 替换为您的实际API密钥
# 如果需要，将 gpt-3.5-turbo 替换为您想要的模型
curl https://api.openai.com/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": "Translate the following English text to French: 'Hello!'"
      }
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

让我们分析一下这个命令：

- `curl https://.../v1/generate`：指定工具 (`curl`) 和端点URL。
- `-X POST`：将HTTP方法设置为 `POST`。
- `-H "Content-Type: application/json"`：添加内容类型头部。
- `-H "Authorization: Bearer YOUR_API_KEY"`：添加授权头部（请记住使用您的真实密钥）。
- `-d '{...}'`：指定要在请求体中发送的数据（有效载荷）。请注意，在命令行中，JSON 使用单引号括起来，并且提示字符串内部的单引号需要进行转义 (`\'`)。

### 发送请求：使用 Python 的示例

在实际操作中，您通常会在脚本或应用程序中进行API调用。Python的 `requests` 库是实现此目的的常用选择。

首先，请确保您已安装 `requests`（`pip install requests`）。

```python
import requests
import json
import os # 推荐用于处理API密钥

# --- 配置 ---
# 最佳实践：从环境变量或安全存储加载API密钥
# api_key = os.getenv("YOUR_API_KEY")
api_key = "YOUR_API_KEY" # 如果不使用环境变量，请替换为您的实际API密钥
if not api_key:
    raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

api_endpoint = "https://api.openai.com/v1/chat/completions"

# --- 请求头部 ---
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# --- 请求有效载荷（OpenAI聊天补全格式）---
payload = {
    "model": "gpt-3.5-turbo", # 指定您想使用的模型
    "messages": [
        {
            "role": "user", # 角色可以是'system'、'user'或'assistant'
            "content": "Translate the following English text to French: 'Hello, world!'" # 用户的提示词
        }
        # 您可以在此处添加更多消息以提供对话历史，例如：
        # {"role": "assistant", "content": "Bonjour le monde!"},
        # {"role": "user", "content": "Now translate 'goodbye'"}
    ],
    "max_tokens": 50,     # 要生成的最大token数量
    "temperature": 0.7    # 控制随机性（0.0到2.0）
}

# --- 发送请求 ---
try:
    response = requests.post(api_endpoint, headers=headers, json=payload)

    # --- 检查成功响应 ---
    response.raise_for_status() # 对于错误响应（4xx或5xx）抛出HTTPError异常

    # 如果成功，继续处理响应
    print("请求成功！")
    response_data = response.json()
    # 访问生成文本的示例（结构可能略有不同）
    # print(json.dumps(response_data, indent=2)) # 格式化打印完整响应
    if response_data.get("choices"):
        print("生成文本：", response_data["choices"][0]["message"]["content"])
    else:
        print("响应中未找到choices。")

except requests.exceptions.RequestException as e:
    print(f"API请求期间发生错误：{e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"状态码：{e.response.status_code}")
        try:
            # 尝试解析并打印来自OpenAI的错误详情（如果可用）
            error_details = e.response.json()
            print(f"响应体：{json.dumps(error_details, indent=2)}")
        except json.JSONDecodeError:
            # 如果响应不是JSON，则打印原始文本
            print(f"响应体：{e.response.text}")
    else:
        print("未从服务器收到响应。")

except Exception as ex:
    print(f"发生意外错误：{ex}")
```

这段Python代码与 `curl` 命令做的事情基本相同：它定义了URL、头部和有效载荷，然后发送一个 `POST` 请求。我们还添加了基本的错误检查，使用了 `try...except` 块和 `response.raise_for_status()`，这是一个好习惯。使用 `requests.post` 时，`json=`参数 (parameter)会自动将Python字典 `payload` 转换为JSON字符串并正确设置 `Content-Type` 头部。

### 请求流程图

为了便于理解，请查看您的请求路径：

> 此图表显示您的客户端向API服务器发送一个HTTP POST请求，其中包含必要的信息（URL、头部、正文）。服务器验证请求，将指令传递给实际的LLM，接收生成的文本，并通过HTTP响应将其发送回您的客户端。

您现在已经了解了API请求的结构，以及如何使用常用工具来构建和发送请求。下一步是理解LLM服务返回的内容。

## 参考资料

- [HTTP Overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview) — Mozilla Contributors (2024)
  Journal: MDN Web Docs
  提供超文本传输协议 (HTTP) 的全面介绍，包括请求方法、头部和状态码，这些是理解 API 交互的基础。
- [Requests: HTTP for Humans™](https://requests.readthedocs.io/en/latest/) — Kenneth Reitz and The Requests Project Contributors (2024)
  `requests` 库的官方文档，提供了在 Python 中进行 HTTP 请求的详细指南和 API 参考，与 Python 示例直接相关。
- [API Reference - Chat Completions](https://platform.openai.com/docs/api-reference/chat/completions) — OpenAI (2024)
  Journal: OpenAI Documentation; Publisher: OpenAI
  OpenAI Chat Completions API 的官方文档，详细说明了本节中演示的 `model`、`messages`、`temperature` 和 `max_tokens` 等请求参数。
