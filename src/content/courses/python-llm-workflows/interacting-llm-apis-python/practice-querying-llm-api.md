---
course: "python-llm-workflows"
chapter: "interacting-llm-apis-python"
lesson: "practice-querying-llm-api"
sourceId: 5131
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-3-interacting-llm-apis-python/practice-querying-llm-api"
title: "练习：调用LLM API"
description: "实际操作练习，向LLM API发送请求并处理结果。"
order: 7
plots: []
sourceHash: "dc0d8ff770d521df211498fe45ab9dfec8c9d29cd9710c75f6b9c5923895bbda"
sourceCorrections: []
---

提供实际操作步骤，指导您使用Python向LLM API发送请求并解读结果。建议您熟练掌握LLM API的基础知识、身份验证和响应处理。在继续之前，请确保您的开发环境已设置好，并且API密钥已妥善保护。

### 前提条件

继续之前，请确保您已具备以下条件：

1. **API密钥：** 从LLM提供商（如OpenAI、Anthropic、Cohere或其他）获取API密钥。
2. **API密钥的保护：** 将您的API密钥设置为环境变量（例如，`OPENAI_API_KEY`）。这对于安全很重要，也能避免在脚本中硬编码密钥。如果您需要回顾环境变量的设置，请查阅第二章。
3. **库的安装：** 安装必要的Python库。我们将使用标准`requests`库和供应商特定库（以OpenAI的`openai`库为例）来展示示例。

   ```bash
   pip install requests python-dotenv openai
   ```
4. **`.env` 文件：** 在您的项目目录中创建一个名为`.env`的文件，并按以下方式添加您的API密钥：

   ```
   OPENAI_API_KEY='your_api_key_here'
   ```

### 使用`requests`库进行查询

`requests`库让您可以直接控制HTTP请求。这对于理解底层通信很有用，但可能比使用专用客户端库更繁琐。

```python
import requests
import os
import json
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv()

# 从环境变量中获取API密钥
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")

# 定义API端点（以OpenAI的聊天补全端点为例）
api_url = "https://api.openai.com/v1/chat/completions"

# 准备带有授权的请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 准备数据负载
data = {
    "model": "gpt-3.5-turbo", # 指定您想要使用的模型
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the difference between an API and an SDK in simple terms."}
    ],
    "temperature": 0.7, # 控制随机性（0=确定性，1=更随机）
    "max_tokens": 100 # 限制响应的长度
}

try:
    # 发送POST请求
    response = requests.post(api_url, headers=headers, json=data, timeout=30) # 添加了超时设置

    # 检查请求是否成功
    response.raise_for_status() # 对于不好的响应（4XX或5XX）抛出HTTPError

    # 解析JSON响应
    result = response.json()

    # 打印响应的相关部分
    print("Response using requests:")
    if result.get("choices"):
        print(result["choices"][0]["message"]["content"])
    else:
        print("No choices found in the response.")
    # 可选：如果可用，打印使用信息
    if result.get("usage"):
        print("\nUsage Info:", result["usage"])

except requests.exceptions.RequestException as e:
    print(f"An error occurred during the API request: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

该脚本首先安全地加载API密钥。然后它定义目标API端点并构建所需的请求头，包括授权令牌。`data`字典包含提示（“解释API和SDK的区别...”）以及模型名称、温度和最大令牌数等参数 (parameter)。最后，它发送请求，检查错误，解析JSON响应，并打印生成的文本。

### 使用官方客户端库进行查询（OpenAI示例）

官方客户端库（SDK）抽象化了大部分HTTP请求的样板代码，提供更符合Python习惯的接口。

```python
import os
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv()

# OpenAI库会自动查找OPENAI_API_KEY环境变量。
# 您也可以显式初始化它：client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
try:
    client = OpenAI() # 初始化客户端，自动使用OPENAI_API_KEY环境变量

    # 进行API调用
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain the difference between an API and an SDK in simple terms."}
        ],
        temperature=0.7,
        max_tokens=100
    )

    # 打印响应的相关部分
    print("\nResponse using OpenAI client library:")
    if completion.choices:
        print(completion.choices[0].message.content)
    else:
        print("No choices found in the response.")

    # 可选：打印使用信息
    if completion.usage:
        print("\nUsage Info:", completion.usage)

except OpenAIError as e:
    print(f"An OpenAI API error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

请注意这段代码的简洁程度。`openai`库在内部处理了请求头的设置和请求的构建。您只需实例化客户端并调用相应的方法（`chat.completions.create`）。针对OpenAI API的错误处理也通常内置于库中。

### 理解交互流程

这个过程包括您的Python脚本向LLM提供商的API端点发送结构化数据（您的提示和参数 (parameter)），该端点随后处理请求并返回生成的文本和元数据。

> 一个简单的图表，说明了您的Python脚本与LLM API服务之间的请求-响应循环。

### 实验与观察

现在轮到您进行实验了。修改代码示例：

- **改变提示：** 提出不同的问题或给出不同的指令。
- **调整`temperature`：** 尝试`0.1`（更集中）和`1.0`（更具创造性/随机性）等值，看看输出如何变化。
- **修改`max_tokens`：** 设置一个较小的值（例如，`20`）或一个较大的值（在模型限制内）来控制响应长度。
- **使用不同的模型：** 如果您的API密钥允许，请尝试指定另一个模型（例如，如果您使用OpenAI并有权限，可以指定`"gpt-4"`）。
- **引入错误：** 临时使用无效的API密钥或构造格式错误的请求数据，看看错误处理（`try...except`）块如何运行。

进行这些实验将巩固您通过API参数 (parameter)控制LLM行为的理解。请记住之前讨论的关于速率限制和成本管理的问题；频繁或复杂的查询将消耗您的配额并可能产生费用。

通过完成本次练习，您已成功使用Python直接与LLM API进行交互。您现在可以程序化地发送提示、配置生成参数并接收响应。这构成了构建更复杂的LLM工作流程的基本技能，我们将在后续章节中使用LangChain等库开始构建这些工作流程。

## 参考资料

- [Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat/create) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI Chat Completions API的官方文档，详细介绍了端点结构、请求参数（如模型、消息、温度、最大token数）和预期的响应格式。
- [OpenAI Python Library Documentation](https://platform.openai.com/docs/) — OpenAI (2025)
  Publisher: OpenAI
  OpenAI Python客户端库的官方文档，展示了与OpenAI API交互的高级Pythonic接口，简化了开发过程。
