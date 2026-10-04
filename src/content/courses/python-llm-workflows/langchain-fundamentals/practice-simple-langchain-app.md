---
course: "python-llm-workflows"
chapter: "langchain-fundamentals"
lesson: "practice-simple-langchain-app"
sourceId: 5149
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-4-langchain-fundamentals/practice-simple-langchain-app"
title: "动手实践：构建一个简单的LangChain应用"
description: "使用LangChain的核心组件构建您的第一个应用程序。"
order: 6
plots: []
sourceHash: "6ee8fb4259a2ff67a91654b12e6d8fc5bacade2653cd82aed38063d98ac12746"
sourceCorrections: []
---

将LangChain的基本构成部分——模型、提示和输出解析器——组合起来，创建一个简单、实用的应用程序。这项动手练习将巩固您对这些组件如何使用LangChain表达式语言（LCEL）在基本工作流程中协同工作的理解。

我们的目标是制作一个小型应用程序，它接收一个主题作为输入，并生成一条关于此主题的短小、幽默的推文。

### 前提条件

开始之前，请确保您已具备：

1. 安装了所需的库：`langchain` 和一个特定于提供商的库，如 `langchain_openai`、`langchain_anthropic` 或 `langchain_community`（用于通过Hugging Face Hub等与N其他模型交互）。如果您尚未安装，请安装它们：

   ```bash
   pip install langchain langchain_openai python-dotenv
   # 或根据需要安装其他提供商库
   ```
2. 设置您选择的LLM提供商的API密钥。如第2章所述，通常的做法是安全地存储您的密钥，通常在`.env`文件中，并将其作为环境变量加载。例如，如果使用OpenAI，您的`.env`文件可能包含：

   ```
   OPENAI_API_KEY="your_api_key_here"
   ```

   确保您的Python脚本可以访问此变量，例如，可以使用`python-dotenv`这样的库。

### 构建推文生成器

我们将逐步构建我们的应用程序。

1. **模型：** 首先，我们需要实例化我们想要使用的语言模型。此对象充当底层LLM服务的接口。在此示例中，我们将使用OpenAI的聊天模型，但您可以将其替换为LangChain支持的其他模型。

   ```python
   import os
   from dotenv import load_dotenv
   from langchain_openai import ChatOpenAI

   # 加载环境变量（确保您的OPENAI_API_KEY已设置）
   load_dotenv()

   # 初始化LLM
   # 我们使用0.7的temperature，以平衡创造性和连贯性
   llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
   ```

   这里，`ChatOpenAI` 表示与特定模型（`gpt-3.5-turbo`）的连接。`temperature` 参数 (parameter)影响输出的随机性；较低的值使其更具确定性，较高的值使其更具创造性。
2. **提示模板：** 接下来，我们定义我们希望LLM如何执行任务。我们将使用 `ChatPromptTemplate` 来组织我们的请求，清楚地说明所需的格式并包含用户的输入主题。

   ```python
   from langchain_core.prompts import ChatPromptTemplate

   # 定义提示结构
   prompt_template = ChatPromptTemplate.from_messages([
       ("system", "你是一个机智的助手，可以生成短小、有趣的推文。"),
       ("human", "生成一条关于：{topic} 的推文（最多140个字符）")
   ])
   ```

   此模板使用两种消息类型：

   - `system`：提供AI角色的背景或指令。
   - `human`：代表用户的输入，包含将动态填充的占位符`{topic}`。
3. **输出解析器：** LLM通常在特定的对象结构（如 `AIMessage`）中返回其响应。通常，我们只想要纯文本内容。`StrOutputParser` 为我们处理这种提取。

   ```python
   from langchain_core.output_parsers import StrOutputParser

   # 初始化输出解析器
   output_parser = StrOutputParser()
   ```
4. **使用LCEL连接组件：** LangChain表达式语言（LCEL）允许我们使用管道操作符（`|`）巧妙地连接这些组件。此操作符将一个组件的输出作为下一个组件的输入，从而创建一个处理流程。

   ```python
   # 使用LCEL创建链
   tweet_chain = prompt_template | llm | output_parser
   ```

   这行代码定义了我们应用程序的工作流程：输入数据首先进入`prompt_template`以格式化请求，然后格式化后的提示发送给`llm`，最后，`llm`的输出由`output_parser`处理以获得最终的字符串结果。

   > 数据流经简单的LangChain表达式语言（LCEL）链。
5. **运行链：** 现在我们可以使用一个特定的主题来调用我们的链。

   ```python
   # 定义输入主题
   input_data = {"topic": "procrastinating squirrels"}

   # 调用链并获取结果
   generated_tweet = tweet_chain.invoke(input_data)

   # 打印结果
   print("生成的推文：")
   print(generated_tweet)
   ```

### 完整代码示例

这是结合了所有步骤的完整Python脚本：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def main():
    # 从.env文件加载环境变量
    # 确保您的OPENAI_API_KEY已在.env文件中设置
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None:
        print("错误：OPENAI_API_KEY环境变量未设置。")
        print("请创建一个包含 OPENAI_API_KEY='您的api密钥' 的 .env 文件")
        return

    # 1. 初始化LLM
    # 使用temperature=0.7以获得有创造性但连贯的输出
    print("正在初始化LLM...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 2. 定义提示模板
    print("正在定义提示模板...")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个机智的助手，可以生成短小、有趣的推文（最多140个字符）。"),
        ("human", "生成一条关于：{topic} 的推文")
    ])

    # 3. 初始化输出解析器
    print("正在初始化输出解析器...")
    output_parser = StrOutputParser()

    # 4. 使用LCEL管道操作符创建链
    print("正在创建处理链...")
    tweet_chain = prompt_template | llm | output_parser

    # 5. 定义输入并调用链
    input_topic = "procrastinating squirrels"
    print(f"\n正在为主题“{input_topic}”生成推文...")
    input_data = {"topic": input_topic}

    try:
        generated_tweet = tweet_chain.invoke(input_data)

        # 打印结果
        print("\n--- 生成的推文 ---")
        print(generated_tweet)
        print("-----------------------")
        print(f"长度：{len(generated_tweet)}个字符")

    except Exception as e:
        print(f"\n发生错误：{e}")
        print("请检查您的API密钥、网络连接以及模型可用性。")

if __name__ == "__main__":
    main()
```

### 执行与输出

当您运行此脚本（假设您的环境已正确设置）时，它会：

1. 初始化与LLM的连接。
2. 定义提示结构和输出解析器。
3. 组合链。
4. 获取输入主题（`"拖延的松鼠"`）。
5. 将此主题通过链传递：
   - 提示模板将其格式化为LLM的请求。
   - LLM根据提示生成响应。
   - 输出解析器提取文本内容。
6. 将生成的推文打印到您的控制台。

输出可能类似于此（确切的文本会因LLM的性质而异）：

```
正在初始化LLM...
正在定义提示模板...
正在初始化输出解析器...
正在创建处理链...

正在为主题“拖延的松鼠”生成推文...

--- 生成的推文 ---
My plan for domination via nut hoarding is solid. Execution starts... right after this nap. And maybe one more acorn. #SquirrelLife #ProcrastinatorsUnite #MaybeTomorrow
-----------------------
长度：138个字符
```

### 试验

此示例体现了核心工作流程。您可以轻松修改它：

- 更改 `input_topic` 变量以生成关于不同主题的推文。
- 调整 `ChatPromptTemplate` 中的 `system` 或 `human` 消息以更改所需的风格或任务。
- 尝试调整 `ChatOpenAI` 中的 `temperature` 参数 (parameter)，看看它如何影响输出的创造性。
- 如果您可以访问其他模型，尝试将 `ChatOpenAI` 替换为不同的LangChain模型集成。

这项实践练习为构建更复杂的应用程序奠定了基础。在下一章中，我们将了解如何创建更复杂的序列，并介绍可以使用工具与其环境交互的代理。

## 参考资料

- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) — LangChain (2023)
  LangChain 表达式语言 (LCEL) 的官方文档。
- [LangChain Core: Model I/O (Models, Prompts, Output Parsers)](https://python.langchain.com/v0.1/docs/modules/model_io/) — LangChain (2023)
  LangChain 核心模型输入/输出组件的官方文档。
- [Chat completions API guide](https://platform.openai.com/docs/api-reference/chat/create) — OpenAI (2023)
  Publisher: OpenAI
  OpenAI 聊天补全 API 的官方指南。
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — Elvis Han, Shawn Chang, and other contributors (2023)
  关于提示工程技术的综合性教育资源。
