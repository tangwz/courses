# 使用官方Python客户端库

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-3-interacting-llm-apis-python/official-python-client-libraries)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管使用 `requests` 库是与 HTTP API（包括大型语言模型 API）交互的基础方式，但它通常涉及编写重复的样板代码来处理身份验证、构造请求体和解析响应。对于许多常用的大语言模型 (LLM)提供商，一个更方便、更有条理的方法是使用其官方 Python 客户端库（也称为 SDK - 软件开发工具包）。

这些库是对底层 HTTP API 调用的封装。它们提供以下几个优点：

1. **抽象化：** 它们隐藏了构建 HTTP 请求和解析 JSON 响应的底层细节。您通过 Python 对象和方法进行交互，使代码更简洁、更易读。
2. **易用性：** 身份验证、参数 (parameter)设置和处理不同 API 端点等常见任务通过专用函数或类得以简化。
3. **类型提示与验证：** 精心设计的库通常包含类型提示，有助于更好的静态分析和编辑器支持，从而在运行时之前捕获潜在错误。有些还可能包含参数的基本验证。
4. **功能集成：** 客户端库通常使使用提供商提供的特定功能（如流式响应、函数调用或管理异步请求）变得更容易，这些功能如果手动使用 `requests` 实现可能会很复杂。
5. **错误处理：** 它们通常提供自定义异常类，与特定的 API 错误情况相对应，使错误处理比捕获通用 HTTP 错误更具体。

### 示例：使用 OpenAI Python 库

OpenAI 提供了一个常用且维护良好的 Python 库，用于与其模型（如 GPT-4 和 GPT-3.5 Turbo）交互。

首先，您需要安装该库：

```bash
pip install openai
```

回顾第 2 章中安全管理 API 密钥的重要性。假设您已将 OpenAI API 密钥设置为环境变量 (`OPENAI_API_KEY`)，初始化客户端并发出简单的聊天完成请求如下所示：

```python
import os
from openai import OpenAI

# 客户端自动获取 OPENAI_API_KEY 环境变量
# 您也可以直接传入：client = OpenAI(api_key="your_api_key")
client = OpenAI()

try:
    # 向聊天完成端点发送请求
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 指定模型
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Malaysia?"}
        ],
        max_tokens=50, # 限制响应长度
        temperature=0.7 # 控制随机性（0=确定性，1=更随机）
    )

    # 提取响应内容
    # 响应对象是结构化的，方便访问
    if response.choices:
        message_content = response.choices[0].message.content
        print(f"Assistant: {message_content.strip()}")
    else:
        print("No response choices received.")

    # 打印使用信息（对成本追踪有用）
    print(f"Usage: {response.usage}")

except Exception as e:
    print(f"An API error occurred: {e}")
```

将此与使用 `requests` 构造等效请求进行比较。您需要手动执行以下操作：

- 设置 `Authorization: Bearer <your_api_key>` 请求头。
- 将 `messages` 列表和其他参数 (parameter)格式化为 JSON 有效载荷。
- 指定正确的 API 端点 URL (`https://api.openai.com/v1/chat/completions`)。
- 解析 JSON 响应字符串。
- 为不同的 HTTP 状态码和 OpenAI 错误格式实现特定错误处理。

`openai` 库处理这些细节，让您专注于核心逻辑：定义提示 (`messages`)、选择模型以及处理结构化的 `response` 对象。

### 示例：使用 Anthropic Python 库

Anthropic，Claude 系列模型的提供商，也提供了一个 Python 客户端库。其模式类似，突显了这些库如何为其各自的服务提供一致的开发者体验。

安装该库：

```bash
pip install anthropic
```

假设您的 Anthropic API 密钥已设置为 `ANTHROPIC_API_KEY`：

```python
import os
from anthropic import Anthropic, AnthropicError

# 客户端自动获取 ANTHROPIC_API_KEY 环境变量
# 或使用以下方式初始化：client = Anthropic(api_key="your_api_key")
client = Anthropic()

try:
    message = client.messages.create(
        model="claude-3-opus-20240229", # 指定模型
        max_tokens=100,
        temperature=0.7,
        system="You are a helpful assistant.", # 系统提示
        messages=[
            {
                "role": "user",
                "content": "What is the main purpose of the Python requests library?"
            }
        ]
    )

    # 从结构化对象中访问响应内容
    if message.content:
        print(f"Assistant: {message.content[0].text.strip()}") # 注意与 OpenAI 的结构差异
    else:
        print("No response content received.")

    # 打印使用情况（结构与 OpenAI 不同）
    print(f"Usage: Input Tokens={message.usage.input_tokens}, Output Tokens={message.usage.output_tokens}")

except AnthropicError as e:
    # 库提供的特定错误处理
    print(f"An Anthropic API error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

注意其相似之处：客户端初始化、方法调用 (`messages.create`)、模型和参数 (parameter)指定以及通过结构化响应对象 (`message`) 获取结果。然而，具体的方法名、参数名（`system` 提示是独立的）和响应结构（`message.content[0].text`）有所不同，这反映了 Anthropic 的 API 设计。

### 选择原始请求与客户端库

通常，当官方客户端库可用且维护良好时，建议使用它。通过抽象化 HTTP 细节并提供特定于供应商的便利和错误处理，它使得代码更易于维护、更易读。

您可能会在以下情况下坚持使用直接的 `requests` 调用：

- 您的目标大语言模型 (LLM)提供商或语言没有官方库。
- 您需要对库抽象化的 HTTP 请求细节进行细粒度控制（这种情况较不常见）。
- 您在非常受限的环境中尽量减少依赖。
- 您正在与一个非常简单的 API 端点交互，使用库的开销感觉没有必要。

对于 Python 生态系统中的大多数开发，使用这些官方库将大幅加快您的工作流程并减少与大语言模型 API 交互时出错的可能性。它们构成了我们将在后续章节中进一步查看的工具的重要部分。

## 参考资料

- [OpenAI Python Library](https://github.com/openai/openai-python) — OpenAI (2020)
  Publisher: OpenAI
  OpenAI Python客户端库的官方文档，详细介绍了安装、配置和使用方法，并提供示例。
- [Anthropic Python SDK](https://docs.anthropic.com/claude/reference/python) — Anthropic (2024)
  Anthropic Python SDK的官方文档，说明了如何使用该库与Claude模型交互。
- [Requests: HTTP for Humans™](https://requests.readthedocs.io/en/latest/) — Kenneth Reitz and individual contributors (2024)
  Requests库的综合文档，提供Python中HTTP交互的核心知识。
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) — OpenAI (2024)
  OpenAI服务的官方API参考，列出了客户端库封装的HTTP端点和数据结构。

---

[上一节](03-%E5%A4%84%E7%90%86API%E5%93%8D%E5%BA%94%E5%92%8C%E9%94%99%E8%AF%AF.md) · [下一节](05-%E8%AE%A4%E8%AF%81%E6%96%B9%E5%BC%8F.md)
