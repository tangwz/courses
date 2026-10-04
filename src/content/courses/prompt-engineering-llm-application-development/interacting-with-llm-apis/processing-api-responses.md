---
course: "prompt-engineering-llm-application-development"
chapter: "interacting-with-llm-apis"
lesson: "processing-api-responses"
sourceId: 3032
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-4-interacting-with-llm-apis/processing-api-responses"
title: "处理API响应"
description: "解析并获取 LLM API 响应中的有用信息。"
order: 5
plots: []
sourceHash: "e604eeae935b151c555ca98895a7c55707ded3bfc5f05780ad7554cf5bc5fd16"
sourceCorrections: []
---

处理LLM API返回的响应是与这些模型交互的基本组成部分。LLM API通常返回JSON格式的数据，这种格式机器可读，并且在包括Python在内的大多数编程语言中都相对容易处理。理解这些响应的结构和内容，对将LLM输出整合到你的应用中非常重要。

补全或聊天API的典型响应包含几部分信息。我们来查看一个你可能遇到的常见结构（细节在OpenAI、Anthropic、Google等提供商之间略有差异，但核心思想是相似的）：

```json
{
  "id": "resp-a1b2c3d4e5",
  "object": "text_completion",
  "created": 1680000000,
  "model": "model-name-v1.0",
  "choices": [
    {
      "index": 0,
      "text": "The capital of France is Paris.",
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 7,
    "total_tokens": 17
  }
}
```

或者对于基于聊天的API：

```json
{
  "id": "chatresp-f6g7h8i9j0",
  "object": "chat.completion",
  "created": 1680000100,
  "model": "chat-model-v2.1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "\n\nThe capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 9,
    "total_tokens": 24
  }
}
```

我们来分析一下这些重要字段：

1. **`id`**: API响应的唯一标识符。便于日志记录和跟踪。
2. **`object`**: 返回对象的类型（例如，`text_completion`，`chat.completion`）。
3. **`created`**: 一个Unix时间戳，指示响应生成的时间。
4. **`model`**: 生成响应时使用的具体LLM版本。这对于复现性和理解模型版本之间可能存在的行为差异非常重要。
5. **`choices`**: 这通常是最核心的部分。它是一个数组，包含模型生成的一个或多个可能的补全或消息。
   - **`index`**: 此选项在数组中的位置（通常从0开始）。
   - **`text`** 或 **`message.content`**: LLM生成的实际文本。这是你通常会在应用中使用的核心输出。请注意补全和聊天API之间结构上的细微差异。聊天API通常返回一个包含`role`（`assistant`）和`content`的`message`对象。
   - **`finish_reason`**: 解释模型停止生成文本的原因。常见的值包括：
     - `stop`：模型自然完成了响应或遇到了预定义的停止序列。
     - `length`：生成达到了请求中指定的tokens最大数量（`max_tokens`）。输出可能被截断。
     - `content_filter`：生成内容被提供商的安全过滤器标记 (token)。
     - `tool_calls`：（在较新的API中）模型决定调用你提供的一个函数或工具。这需要特定的处理，稍后会讨论。
     - `null`：生成仍在进行中（与流式传输相关）。
   - **`logprobs`**: （可选）提供生成tokens的对数概率，有助于高级分析，但默认通常为`null`。
6. **`usage`**: 提供有关请求中token消耗的信息。
   - **`prompt_tokens`**: 你输入prompt中的token数量。
   - **`completion_tokens`**: 响应中生成的token数量。
   - **`total_tokens`**: prompt和completion tokens的总和。这与API调用的费用直接相关。

### 在Python中获取响应数据

假设你已使用`requests`等库接收到响应，你首先需要解析JSON主体。大多数LLM客户端库或SDK会自动处理此过程，返回一个Python字典或对象。如果直接使用`requests`：

```python
import requests
import json
import os

# 假设API_ENDPOINT、HEADERS和PAYLOAD已正确定义
# API_ENDPOINT = "YOUR_LLM_API_ENDPOINT"
# HEADERS = {"Authorization": f"Bearer {os.getenv('LLM_API_KEY')}", "Content-Type": "application/json"}
# PAYLOAD = { "model": "your-model-name", "prompt": "What is the capital of France?", "max_tokens": 50 }

# 用于演示的模拟响应 - 请替换为实际的API调用
simulated_response_json = """
{
  "id": "chatresp-simulated",
  "object": "chat.completion",
  "created": 1680000100,
  "model": "chat-model-v2.1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "\\n\\nThe capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 9,
    "total_tokens": 24
  }
}
"""
try:
    # 在实际应用中：
    # response = requests.post(API_ENDPOINT, headers=HEADERS, json=PAYLOAD)
    # response.raise_for_status() # 检查HTTP错误（4xx, 5xx）
    # response_data = response.json()

    # 使用模拟数据：
    response_data = json.loads(simulated_response_json)

    # 基本验证：检查'choices'是否存在且不为空
    if 'choices' in response_data and len(response_data['choices']) > 0:
        # 根据预期结构（例如，聊天补全）获取内容
        if 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
            generated_content = response_data['choices'][0]['message']['content'].strip()
            finish_reason = response_data['choices'][0].get('finish_reason', 'unknown') # 为安全起见使用.get

            print(f"生成内容: {generated_content}")
            print(f"结束原因: {finish_reason}")

            # 如果可用，获取用量信息
            if 'usage' in response_data:
                usage = response_data['usage']
                print(f"Tokens用量 - Prompt: {usage.get('prompt_tokens', 'N/A')}, Completion: {usage.get('completion_tokens', 'N/A')}, 总计: {usage.get('total_tokens', 'N/A')}")
            else:
                print("响应中未找到用量数据。")

        # 如果需要用于不同API类型，添加对'text'等其他结构的检查
        elif 'text' in response_data['choices'][0]:
             generated_content = response_data['choices'][0]['text'].strip()
             finish_reason = response_data['choices'][0].get('finish_reason', 'unknown')
             print(f"生成内容: {generated_content}")
             print(f"结束原因: {finish_reason}")
             # 获取用量信息...（与上面类似）
        else:
             print("在第一个选项中未找到'message'/'content'或'text'。")

    else:
        print("错误：响应中未找到'choices'数组或其为空。")
        # 如果需要，记录完整的response_data用于调试
        # print(f"完整响应: {response_data}")

except json.JSONDecodeError:
    print("错误：未能解码API的JSON响应。")
# except requests.exceptions.RequestException as e:
#     print(f"API请求出错: {e}") # 处理请求错误
except (KeyError, IndexError, TypeError) as e:
    print(f"解析API响应的预期结构时出错: {e}")
    # 同样可以在此处记录response_data，以便调试意外结构
```

### 处理多个选项和结束原因

虽然大多数请求要求单个补全（`n=1`），但你可以请求多个选项。如果`len(response_data['choices']) > 1`，你需要遍历数组或根据一些标准（例如，索引0通常是默认或概率最高的）选择一个选项。

`finish_reason`对应用逻辑很重要。

- 如果它是`length`，响应可能不完整。你可能需要通知用户、妥善截断，或者可能进行另一次API调用以继续生成（尽管这需要仔细的状态管理）。
- 如果它是`content_filter`，你应妥善处理，例如显示一个通用消息而非被过滤的内容。
- 如果它是`tool_calls`，你的应用需要解析请求的工具调用，执行相应的函数，并可能在后续的API调用中将结果发送回模型。

处理API响应不仅仅是获取文本。它需要解析结构化数据，理解元数据（如结束原因和token用量），并谨慎处理潜在的差异或缺失字段。这种仔细的处理确保你的应用可以可靠地使用LLM生成的输出。

## 参考资料

- [Gemini API reference: Text generation (REST)](https://ai.google.dev/docs/reference/rest) — Google (2024)
  Publisher: Google
  展示Google生成式模型的响应结构，使读者能够比较不同LLM提供商的API响应格式。
- [\`json\` - JSON encoder and decoder](https://docs.python.org/3/library/json.html) — Python Software Foundation (2024)
  Python中解析和序列化JSON数据的官方文档，对处理以JSON格式接收的LLM API响应至关重要。
- [Completions API Reference](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHlw9NGQKkESsDAg979qvyHYIjFNiZCsXvG3ePIZYLrG94rp2smZ6jWbgGdUPrV2fty4R3dMaq9k0GU7CfpHb60agpyj1GIaPVvEP64ymqclLg3q4K8E4ckawegHYmdd917_sTVz_wSkhscpjaZKY-XN85lgCM=) — OpenAI (2024)
  Publisher: OpenAI
  描述OpenAI传统文本补全模型的响应结构，有助于理解本节中提供的初始JSON示例。
