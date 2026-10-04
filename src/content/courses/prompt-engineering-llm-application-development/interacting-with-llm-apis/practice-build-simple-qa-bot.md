---
course: "prompt-engineering-llm-application-development"
chapter: "interacting-with-llm-apis"
lesson: "practice-build-simple-qa-bot"
sourceId: 3041
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-4-interacting-with-llm-apis/practice-build-simple-qa-bot"
title: "动手实践：搭建一个简单的问答机器人"
description: "使用Python进行动手实践，通过LLM API构建一个基础的问答应用。"
order: 8
plots: []
sourceHash: "56824e4a50b972fac7c132ff029b133cbf0b62a7d50283892fede79b05f89d55"
sourceCorrections: []
---

构建一个简单的问答机器人，从而展示实际应用。本次练习旨在培养调取API、发送提示词 (prompt)及以程序方式处理回复的能力。将创建一个命令行程序，用于接收用户提问，通过LLM的API发送给LLM，并显示回答。

### 前提条件

开始前，请确保已进行以下设置：

1. **Python环境**：已安装并可运行的Python 3。
2. **LLM API权限**：从LLM服务提供商（如OpenAI、Anthropic、Google）获取的API密钥。本例中，代码结构是基于OpenAI API的，但其运作方式对其他提供商也类似。
3. **所需库文件**：你需要一个用于进行HTTP请求的库，或一个特定于服务提供商的库。为图方便，我们将使用 `openai` 库。通过pip安装：

   ```bash
   pip install openai
   ```
4. **API密钥安全**：**切勿**将你的API密钥直接写死在脚本中。请使用环境变量。设置一个名为 `OPENAI_API_KEY` 的环境变量，并填入你的实际密钥。

- 在Linux/macOS上： `export OPENAI_API_KEY='your_api_key_here'`
- 在Windows（命令提示符）上： `set OPENAI_API_KEY=your_api_key_here`
- 在Windows（PowerShell）上： `$env:OPENAI_API_KEY='your_api_key_here'`

### 问答机器人的主要运行逻辑

我们的机器人将循环执行以下步骤：

1. **提示用户**：要求用户输入一个问题。
2. **获取输入**：从命令行读取问题。
3. **检查退出指令**：如果用户输入“quit”、“exit”或类似指令，则终止程序。
4. **构建提示词 (prompt)**：创建一个简单的提示词，其中包含用户的问题。我们将指示LLM充当一个有帮助的助手。
5. **调用LLM API**：将提示词发送到LLM API，并指定模型和任何相关参数 (parameter)（如 `temperature` 或 `max_tokens`）。
6. **处理回复**：从API回复中提取回答文本。
7. **错误处理**：包含API调用的基本错误处理（例如，网络问题、认证问题、速率限制）。
8. **打印回答**：将LLM的回复打印到控制台。
9. **循环**：重复上述过程。

### Python代码实现

我们来编写Python代码。创建一个名为 `qa_bot.py` 的文件。

```python
import os
import openai
from openai import OpenAI # Use the updated OpenAI library structure

# --- 配置 ---
# 尝试从环境变量加载API密钥
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except openai.OpenAIError as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure the OPENAI_API_KEY environment variable is set.")
    exit()

MODEL_NAME = "gpt-3.5-turbo" # 或选择其他合适的模型

# --- 核心功能 ---
def get_llm_response(user_question):
    """
    将用户的问题发送到LLM API并返回回复。
    """
    system_prompt = "You are a helpful assistant. Answer the user's question clearly and concisely."

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.7, # 调整以平衡创造性和确定性
            max_tokens=150   # 限制回复长度
        )

        # 提取回复文本
        # 注意：确切的结构可能因API版本更新而略有不同
        if completion.choices and len(completion.choices) > 0:
            response_text = completion.choices[0].message.content.strip()
            return response_text
        else:
            return "Error: No response received from the API."

    except openai.APIConnectionError as e:
        return f"API Connection Error: {e}"
    except openai.RateLimitError as e:
        return f"API Rate Limit Exceeded: {e}"
    except openai.AuthenticationError as e:
        return f"API Authentication Error: {e}. Check your API key."
    except openai.APIError as e:
        return f"API Error: {e}"
    except Exception as e:
        # 捕获任何其他意外错误
        return f"An unexpected error occurred: {e}"

# --- 主交互循环 ---
def main():
    print("Simple Q&A Bot (type 'quit' or 'exit' to stop)")
    print("-" * 30)

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit"]:
            print("Bot: Goodbye!")
            break

        if not user_input:
            continue

        print("Bot: Thinking...") # 等待时提供反馈

        llm_answer = get_llm_response(user_input)

        print(f"Bot: {llm_answer}")
        print("-" * 30)

if __name__ == "__main__":
    main()
```

### 代码说明

1. **导入模块**：我们导入 `os` 用于访问环境变量，以及 `openai` 用于API交互。
2. **配置**：我们初始化 `OpenAI` 客户端。API密钥从 `OPENAI_API_KEY` 环境变量中获取。此处包含基本的错误处理，以捕获初始化问题。我们还定义了 `MODEL_NAME`。
3. **`get_llm_response(user_question)` 函数**：
   - 接收 `user_question` 作为输入。
   - 定义一个 `system_prompt`，为LLM提供上下文 (context)或指令（在此处是设置其角色）。
   - 使用 `client.chat.completions.create` 调用API。这是用于像GPT-3.5 Turbo或GPT-4这样的聊天型模型的标准方法。
   - `messages` 参数 (parameter)遵循聊天格式，包含角色（“system”、“user”）。
   - `temperature` 和 `max_tokens` 作为本章前面讨论的参数示例被包含在内。`temperature=0.7` 允许回复具有一定的变化性，而 `max_tokens=150` 则限制长度以避免回答过长。
   - **回复解析**：它通过 `completion.choices[0].message.content` 获取回复内容。此处回复对象的结构很重要；如果你使用不同的库或模型，在开发过程中可能需要查看实际的回复对象（`print(completion)`）。
   - **错误处理**：`try...except` 代码块捕获特定的 `openai` 异常（如连接错误、速率限制、认证问题）以及通用异常，并返回有用的错误信息，而不是导致机器人崩溃。
4. **`main()` 函数**：
   - 打印一条消息。
   - 进入一个无限的 `while True` 循环以进行持续交互。
   - `input("You: ")` 提示用户并读取其输入。
   - 检查“quit”或“exit”命令以跳出循环。
   - 使用用户的输入调用 `get_llm_response`。
   - 打印返回的回答（或错误消息），并以“Bot: ”为前缀。
5. **`if __name__ == "__main__":`**：这个标准的Python结构可确保 `main()` 函数仅在脚本直接执行时才被调用。

### 运行机器人

1. 将代码保存为 `qa_bot.py`。
2. 确保你的 `OPENAI_API_KEY` 环境变量在终端会话中已设置。
3. 从终端运行脚本：

   ```bash
   python qa_bot.py
   ```

### 交互示例

```
简单问答机器人（输入“quit”或“exit”停止）
------------------------------
你：法国的首都在哪里？
机器人：思考中...
机器人：法国的首都是巴黎。
------------------------------
你：解释一下LLM API中温度（temperature）的含义。
机器人：思考中...
机器人：在LLM API中，“温度”（temperature）参数控制输出的随机性。较低的温度（例如0.2）会使输出更集中、更确定，通常选择最有可能的下一个词。较高的温度（例如0.8）会增加随机性，导致更具创造性或多样化的回复，但也可能不够连贯。它本质上调整了下一个token预测的概率分布。
------------------------------
你：quit
机器人：再见！
```

### 后续步骤和改进

这个机器人功能基本，但它呈现了核心的交互模式。以下是一些你可以扩展它的方法：

- **对话历史（记忆）**：修改 `get_llm_response` 函数，使其接受并将之前的对话轮次包含在 `messages` 列表中。这使得LLM能够了解之前讨论过的内容。（在后续关于框架的章节中有所介绍）。
- **流式回复**：实现流式回复（如果API支持），以便在答案生成时逐字显示，从而提升感知的响应速度。（本章前面已介绍）。
- **不同提示技巧**：试用第2章中更高级的提示词 (prompt)（例如，角色提示、要求特定的输出格式）。
- **参数 (parameter)调整**：调整 `temperature`、`max_tokens`、`top_p` 等，以观察它们如何影响不同类型问题的输出质量和风格。
- **模型选择**：试用通过API可用的不同模型。
- **用户界面**：搭建一个简单的网页界面（使用Flask或Streamlit），而非命令行交互。

这个动手练习提供了一个实际范例，展示了本章介绍的API交互思路如何转化为一个可运行的应用。成功搭建并运行这个机器人，标志着你在将LLM融入自己的软件项目中迈出了重要一步。

## 参考资料

- [OpenAI Python Library](https://github.com/openai/openai-python) — OpenAI (2024)
  Publisher: OpenAI
  `openai` Python库的官方文档，包括客户端设置和API密钥管理。
- [Prompt Engineering Guide](https://www.promptingguide.ai/) —  (2024)
  一份被广泛认可的资源，提供设计和优化LLM提示的全面策略。
- [Building Large Language Model Applications](https://www.amazon.com/Building-Large-Language-Model-Applications/dp/1098150492) — Reza Shabani, Greg Lou, Paul R. Schram, Kevin H. Smith (2023)
  Publisher: O'Reilly Media
  一本指导读者使用LLM开发实际应用程序的书籍，涵盖架构考量和高级功能。
