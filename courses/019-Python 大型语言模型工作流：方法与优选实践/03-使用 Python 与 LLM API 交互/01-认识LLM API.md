# 认识LLM API

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-3-interacting-llm-apis-python/understanding-llm-apis)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了连接您的Python应用程序与大型语言模型的强大功能，我们依赖于应用程序编程接口（API）。您可以将API视为LLM服务提供的一个明确的约定或菜单。您的应用程序（客户端）使用此菜单向LLM提供方的服务器请求特定服务（如生成文本或回答问题）。服务器处理请求并返回响应，通常包含LLM的输出。

这种通过互联网进行的客户端-服务器交互通常遵循Web API的原则，主要使用HTTP(S)。让我们分析一下典型的LLM API交互的主要构成部分。

### API调用的构成

当您的Python代码与LLM API交互时，它会构建一个包含特定信息的HTTP请求。LLM提供方的服务器接收此请求，处理后返回HTTP响应。

1. **端点URL:** 这是您发送请求的特定网络地址。LLM提供方会发布不同的端点以支持不同功能（例如，文本补全、嵌入 (embedding)、模型列表）。例如，一个端点可能看起来像 `https://api.example-llm-provider.com/v1/chat/completions`。`v1` 通常表示API版本。
2. **HTTP方法:** 这个动词告诉服务器客户端希望执行什么操作。发送提示并获取补全时，几乎普遍使用 `POST` 方法，因为您正在向服务器发送数据（提示和参数 (parameter)）以创建新资源（即补全）。其他方法如 `GET` 可能用于获取信息，例如可用模型。
3. **请求头:** 这些提供关于请求的元数据。常见请求头包括：

   - `Authorization`: 包含凭据（通常是API密钥或令牌）以验证您的请求。我们将在“安全设置API密钥”一节中介绍如何安全地获取和管理这些信息。
   - `Content-Type`: 指定请求体中发送数据的格式，LLM API通常是 `application/json`。
   - `Accept`: 告诉服务器客户端偏好的响应格式（通常也是 `application/json`）。
4. **请求体:** 对于 `POST` 请求，这包含发送到服务器的实际数据负载。对于LLM API，这通常以JSON格式组织，并包含：

   - 针对LLM的提示或消息。
   - 您想使用的特定模型（例如，`gpt-3.5-turbo`，`claude-3-opus-20240229`）。
   - 用于控制生成过程的参数，例如 `temperature`（随机性）、`max_tokens`（响应长度限制）、`stop` 序列等。

   *最小请求体示例 (JSON):*

   ```json
   {
     "model": "example-model-v1",
     "messages": [
       {"role": "user", "content": "解释API调用的含义。"}
     ],
     "max_tokens": 100,
     "temperature": 0.7
   }
   ```
5. **响应:** 处理请求后，服务器返回一个HTTP响应，包含：

   - **状态码:** 表示结果的标准HTTP代码（例如，`200 OK` 表示成功，`401 Unauthorized` 表示API密钥无效，`429 Too Many Requests` 表示速率限制，`500 Internal Server Error` 表示服务器内部错误）。
   - **请求头:** 类似于请求头，提供关于响应的元数据。
   - **响应体:** 通常包含LLM生成的输出，通常也是JSON格式。这可能包括文本补全、使用统计（令牌计数）以及生成停止的原因。

   *最小响应体示例 (JSON):*

   ```json
   {
     "id": "chatcmpl-12345abcde",
     "object": "chat.completion",
     "created": 1700000000,
     "model": "example-model-v1",
     "choices": [
       {
         "index": 0,
         "message": {
           "role": "assistant",
           "content": "An API call is like ordering from a restaurant menu..."
         },
         "finish_reason": "stop"
       }
     ],
     "usage": {
       "prompt_tokens": 15,
       "completion_tokens": 55,
       "total_tokens": 70
     }
   }
   ```

### 请求-响应流程

这种交互可以看作是一场直接对话：您的应用程序发送一个结构化请求，API服务器返回一个结构化响应。

> 图表显示了客户端应用程序与LLM API服务器之间的基本交互。

理解这种结构非常重要。它明确了您需要发送哪些信息（提示、参数 (parameter)、API密钥）以及您可以预期收到哪些信息（LLM的响应、使用数据）。尽管不同提供方（如OpenAI、Anthropic、Google）的具体细节略有不同，但使用HTTP(S)端点、方法、请求头和JSON负载的核心原则保持不变。

这种标准化方法使开发者能够与复杂的AI模型交互，而无需管理底层基础设施，只需使用熟悉的Web协议和数据格式。既然我们已经了解了API调用涉及的内容，后续章节将向您展示如何使用Python库构建和发送这些请求。

## 参考资料

- [Chat completions API reference](https://platform.openai.com/docs/api-reference/chat/create) — OpenAI (2025)
  Publisher: OpenAI
  提供了构建请求和解释响应的详细官方文档，适用于最常用的LLM聊天API之一，展示了所讨论概念的实际应用。
- [HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP) — MDN Contributors (2024)
  提供了关于HTTP协议基础的全面且易懂的解释，包括方法、头部、状态码和客户端-服务器交互，这些都是LLM API的构建基础。
- [Requests: HTTP for Humans™](https://requests.readthedocs.io/en/latest/) — Kenneth Reitz and the Requests Community (2024)
  流行Python `requests` 库的官方文档，该库是进行HTTP请求和与API交互的标准工具，直接支持Python中的实际应用。

---

[上一节](../02-%E9%85%8D%E7%BD%AE%E7%94%A8%E4%BA%8E%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%20%28LLM%29%20%E5%BC%80%E5%8F%91%E7%9A%84%20Python%20%E7%8E%AF%E5%A2%83/06-%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%E5%AE%9E%E8%B7%B5.md) · [下一节](02-%E4%BD%BF%E7%94%A8requests%E5%BA%93%E5%8F%91%E9%80%81API%E8%AF%B7%E6%B1%82.md)
