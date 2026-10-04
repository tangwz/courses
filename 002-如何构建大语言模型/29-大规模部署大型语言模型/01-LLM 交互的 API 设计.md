# LLM 交互的 API 设计

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-29-serving-llms-at-scale/api-design-llm-interaction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要使训练和优化好的大型语言模型（LLM）可供应用使用，需要一个定义清晰的应用程序编程接口（API）。此 API 作为服务基础设施与客户端应用之间的约定，规定了请求的格式以及预期的响应。周全地设计此接口对于易用性、性能和可维护性而言十分重要。不良的 API 设计可能导致资源使用效率低下、客户端集成困难以及用户体验不佳。

### 选择 API 协议和风格

多数 LLM 交互通过网络进行，因此标准的网络协议成为常见选择。

- **REST/HTTP 与 JSON**：这是最普遍的方法，因为它简单、普遍且易于与标准网络技术和库配合使用。请求和响应通常以 JSON 格式通过 HTTP(S) 进行编码。它适用于许多应用，特别是面向用户的应用。
- **gRPC**：对于内部微服务或对性能要求高的应用，gRPC 是更优的选择。它使用 Protocol Buffers 进行序列化，并使用 HTTP/2 进行传输，与 REST/JSON 相比，通常能带来更低的延迟和更高的吞吐量 (throughput)。然而，它需要特定的客户端库，并可能增加复杂性。

对于多数通用 LLM 服务，设计良好的 REST/HTTP API 能在性能和可用性之间取得良好平衡。

### 设计请求负载

客户端需要向模型发送足够的信息来生成文本。典型的请求负载可能包含：

- **提示词（Prompt）**：LLM 应进行响应的输入文本或消息序列（针对聊天模型）。
- **生成参数 (parameter)**：用于影响输出文本的控制项。常见参数包含：
  - `max_new_tokens`：要生成的最大令牌数量。防止生成失控并控制响应长度。
  - `temperature`：控制输出的随机性。较低的值（例如 0.2）使输出更具确定性和集中性，而较高的值（例如 0.8）则增加多样性和创造性。值为 0 实际上使贪婪解码成为标准。
  - `top_p` (核采样)：规定一个概率阈值 $p$。模型只考虑累计概率超过 $p$ 的最小令牌集合。这通常比单独使用 `temperature` 效果更好。值通常介于 0.8 到 1.0 之间。
  - `top_k`：将采样限制在最有可能的 $k$ 个后续令牌。可与 `top_p` 一起使用或替代 `top_p` 使用。
  - `repetition_penalty`：惩罚已在提示词或生成序列中出现过的令牌，阻止重复的输出。值大于 1.0 会增加惩罚。
  - `stop_sequences`：如果生成，则表示响应结束的字符串列表。
- **流式传输标志**：一个布尔值，表明客户端是希望响应逐令牌流式传输，还是作为一个单个完整块接收。
- **模型标识符**（可选）：如果 API 端点提供多个模型或模型版本服务，则需要一个标识符来指定使用哪个模型。

以下是一个可能的文本补全 API 的 JSON 请求体：

```json
{
  "prompt": "解释 LLM 推理中的键值缓存原理：",
  "model": "llm-engine-v1.2",
  "max_new_tokens": 250,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop_sequences": ["\n\n", "---"],
  "stream": false
}
```

### 设计响应负载

响应应提供生成的文本和相关的元数据。

- **非流式响应**：对于 `stream` 为 `false` 的请求，响应通常是一个包含以下内容的 JSON 对象：
  - `generated_text`：模型输出的完整字符串。
  - `finish_reason`：表明生成停止的原因（例如，如果达到 `max_new_tokens` 则为 `length`，如果生成了停止序列则为 `stop_sequence`，如果模型自然完成则为 `eos_token`）。
  - `usage`：令牌计数（例如，`prompt_tokens`、`completion_tokens`、`total_tokens`）。有助于追踪成本或资源消耗。

非流式响应示例：

```json
{
  "id": "cmpl-xyz123",
  "object": "text_completion",
  "created": 1677652288,
  "model": "llm-engine-v1.2",
  "choices": [
    {
      "text": " 键值（KV）缓存是一种重要的优化技术，在基于 Transformer 的大型语言模型（LLM）的自回归解码过程中使用。在生成过程中，每个新令牌都依赖于涉及所有先前令牌的注意力计算。自注意力层中为先前令牌计算的键（K）和值（V）在生成新令牌时保持不变。KV 缓存将这些 K 和 V 张量存储在内存中（通常是 GPU HBM），而不是在每一步重新计算整个序列的这些张量。在生成下一个令牌时，模型只需计算最新令牌的 K 和 V，并重用所有先前令牌的缓存值。这显著减少了每个生成令牌的计算成本，使得推理速度更快，特别是对于长序列。",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop_sequence"
    }
  ],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 151,
    "total_tokens": 164
  }
}
```

- **流式响应**：当 `stream` 为 `true` 时，服务器会发送回一个事件序列，通常使用服务器发送事件（SSE）。每个事件通常包含生成文本的一个片段。最后一个事件包含 `finish_reason` 和 `usage` 统计信息。这使得客户端能够逐步显示响应，从而提升用户感受到的性能。

SSE 事件序列示例（简化版）：

```
event: token
data: {"text": " "}

event: token
data: {"text": "-值"}

event: token
data: {"text": " (KV"}

event: token
data: {"text": ") 缓存"}

... 更多令牌事件 ...

event: token
data: {"text": "."}

event: done
data: {"finish_reason": "stop_sequence", "usage": {"prompt_tokens": 13, "completion_tokens": 151, "total_tokens": 164}}
```

> 流式 API 请求和响应的流程。客户端发起请求，服务按生成顺序返回文本片段。

### 处理同步和异步操作

使用 LLM 生成文本可能需要相当长的时间，从几秒到几分钟，特别是对于长输出。

- **同步 API**：客户端发送请求并等待（阻塞）直到收到完整响应。这很简单，但对于长时间生成容易出现 HTTP 超时，并可能导致客户端应用无响应。仅适用于非常短且可预测的生成。
- **异步 API**：客户端发送请求后，服务器立即使用任务 ID 进行确认。然后客户端可以使用任务 ID 轮询一个端点，或等待回调（webhook）来接收结果。这避免了超时，但增加了客户端逻辑的复杂性。
- **流式 API**：如前所述，流式传输提供了一个中间方案。初始连接保持开放，逐步交付结果。这避免了超时并提升了感受到的响应速度，使其成为交互式应用的首选方法。

### 客户端交互示例 (Python)

以下是客户端如何使用 Python 的 `requests` 库与这些 API 端点进行交互的示例：

**标准（非流式）请求：**

```python
import requests
import json

API_URL = "http://your-llm-service.com/generate"
API_KEY = "your_api_key_here" # 应安全处理

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
  "prompt": "What is the capital of Malaysia?",
  "max_new_tokens": 50,
  "temperature": 0.5,
  "stream": False
}

try:
    response = requests.post(API_URL,
                             headers=headers,
                             json=payload,
                             timeout=30) # 设置超时
    response.raise_for_status() # 对于不好的状态码（4xx 或 5xx）抛出异常

    result = response.json()
    generated_text = result['choices'][0]['text']
    print(f"生成的文本: {generated_text}")
    print(f"用量: {result['usage']}")

except requests.exceptions.RequestException as e:
    print(f"API 请求失败: {e}")
except KeyError as e:
    print(f"解析响应失败: 缺少 {e}")
```

**流式请求：**

```python
import requests
import json

API_URL = "http://your-llm-service.com/generate"
API_KEY = "your_api_key_here" # 应安全处理

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream" # 表明偏好 SSE
}

payload = {
  "prompt": "Write a short story about a robot learning to paint:",
  "max_new_tokens": 300,
  "temperature": 0.8,
  "stream": True
}

try:
    # 在 requests 中使用 stream=True 来处理流式连接
    with requests.post(
        API_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=180
    ) as response:
        response.raise_for_status()
        print("流式响应:")
        full_response = ""
        final_data = None

        # 逐行遍历流
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('event: token'):
                    # 获取下一行，应为数据
                    data_line = next(response.iter_lines()).decode('utf-8')
                    if data_line.startswith('data:'):
                        try:
                            content = json.loads(data_line[len('data: '):])
                            text_chunk = content.get('text', '')
                            print(text_chunk, end='', flush=True)
                            full_response += text_chunk
                        except json.JSONDecodeError:
                            print(f"\n解码 JSON 错误: {data_line}")
                elif decoded_line.startswith('event: done'):
                     data_line = next(response.iter_lines()).decode('utf-8')
                     if data_line.startswith('data:'):
                         try:
                            final_data = json.loads(
                                data_line[len('data: '):]
                            )
                         except json.JSONDecodeError:
                            print(f"\n解码最终 JSON 错误: {data_line}")
                     break # 收到 done 事件后退出循环

        print("\n--- 流式传输结束 ---")
        if final_data:
            print(f"结束原因: {final_data.get('finish_reason')}")
            print(f"用量: {final_data.get('usage')}")
        else:
            print("未收到最终数据。")

except requests.exceptions.RequestException as e:
    print(f"\nAPI 请求失败: {e}")
```

### 错误处理和状态码

API 定义了清晰的错误响应。请使用标准的 HTTP 状态码：

- `200 OK`：请求成功（非流式或流式传输的最终事件）。
- `400 Bad Request`：无效输入（例如，JSON 格式错误、缺少必需参数 (parameter)、参数值无效）。在响应体中包含描述性的错误消息。
- `401 Unauthorized`：缺少或无效的身份验证凭据。
- `403 Forbidden`：已认证用户没有权限。
- `429 Too Many Requests`：超出速率限制。如果可能，包含 `Retry-After` 标头。
- `500 Internal Server Error`：生成过程中服务器端发生意外错误。
- `503 Service Unavailable`：服务暂时过载或因维护而停机。

### 安全考量

LLM API 与任何网络服务一样，必须加以保护。实施适当的身份验证（例如，API 密钥、OAuth 2.0）和授权机制来控制访问。使用 HTTPS 加密流量。如果提示词 (prompt)包含不受信任的用户输入，请注意潜在的提示注入攻击。输入验证也是一种安全形式，可以防止格式错误的请求在系统内部引发问题。

设计清晰、文档齐全的 API 是使您的 LLM 可用且实用的基础。在定义接口时，请考虑客户端应用的需求、LLM 生成的特性（可能慢速、长度可变）以及标准网络实践。

## 参考资料

- [Google Cloud API Design Guide](https://cloud.google.com/apis/design/) — Google Cloud (2024)
  Publisher: Google Cloud
  提供了设计RESTful API的全面最佳实践，涵盖协议、资源命名、错误处理和安全等方方面面。
- [Hugging Face Transformers: Generation Configuration](https://huggingface.co/docs/transformers/main_classes/text_generation#generation-parameters) — Hugging Face (2024)
  Publisher: Hugging Face
  详细介绍了LLM的各种生成参数（例如温度、top_p、top_k、重复惩罚），解释了它们对输出的影响。
- [gRPC Core Concepts](https://grpc.io/docs/what-is-grpc/core-concepts/) — gRPC Authors (2024)
  介绍了gRPC的核心概念和架构，这是一个开源的RPC框架，用于高性能微服务通信。

---

[上一节](../28-%E9%AB%98%E6%95%88%E6%8E%A8%E7%90%86%E7%AD%96%E7%95%A5/05-%E6%8E%A8%E6%B5%8B%E8%A7%A3%E7%A0%81.md) · [下一节](02-%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E6%A1%86%E6%9E%B6%20%28Triton%2C%20TorchServe%29.md)
