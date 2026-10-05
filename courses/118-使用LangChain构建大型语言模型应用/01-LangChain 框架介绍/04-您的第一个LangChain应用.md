# 您的第一个LangChain应用

来源：[原文](https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-1-intro-to-langchain-framework/first-langchain-application)

[返回章节目录](README.md) · [返回课程目录](../README.md)

这个初步例子呈现了如何将LangChain的模块化部件组合成一个可用的应用。它是一个实际范例，展现了LangChain的核心功能：将模块化部件连接成一个有逻辑、可运行的图。我们将构建一个简单程序，它从用户获取一个主题，将其格式化为提示，发送给语言模型，并返回一个有结构的回复。

这个过程呈现了用LangChain构建应用的主要模式。你定义数据流动，通过连接部件来构建它，然后用你的输入来运行。

### 基本部件：模型、提示与解析器

我们的第一个应用将使用三个必要部件：

1. **一个聊天模型：** 这是一个将处理我们请求的语言模型。我们将使用一个围绕OpenAI模型的封装，它提供了一个一致的交互界面。
2. **一个提示模板：** 提示模板提供了一种有结构且可重复使用的方式来生成提示，而不是将用户输入直接放入字符串中。这使得指令与用户提供的数据之间能清晰地分开。
3. **一个输出解析器：** 大型语言模型（LLMs）以特定数据结构返回响应，例如 `AIMessage` 对象。输出解析器是一个简单部件，它将这种原始输出转换为更可用的格式，如一个普通字符串。

### 使用LCEL组合应用

LangChain表达式语言（LCEL）是将这些部件链接在一起的标准方式。它使用一种类似于管道 (`|`) 运算符的语法，其中一个部件的输出作为输入传递给下一个。这为你的应用逻辑创建了一个清晰易懂的定义。

我们简单应用的数据流动可以被看作一个序列，其中数据在每一步都被转换。

> 这是第一个应用的数据管道。输入字典用于格式化提示，该提示被发送到模型。模型的响应随后被解析成一个最终字符串。

### 用代码构建应用

让我们把这种结构转换为Python脚本。请确保你已按照上一节所示在环境中配置了OpenAI API凭据。

首先，我们导入所需类。`ChatOpenAI` 是我们的模型接口，`ChatPromptTemplate` 用于我们的提示，而 `StrOutputParser` 将整理最终输出。

```python
# main.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 从.env文件加载环境变量
load_dotenv()

# 确保已设置OPENAI_API_KEY
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY 环境变量未设置。")
```

接下来，我们实例化这三个部件。我们的提示模板将包含一个占位符 `{topic}`，它将被用户输入填充。

```python
# 1. 初始化模型
# 我们将使用gpt-4o-mini，因其速度和能力。
# temperature参数控制随机性；0表示更确定的输出。
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. 定义提示模板
# 此模板指示模型扮演一个乐于助人的助手，并生成
# 关于给定主题的一个有趣事实。
prompt = ChatPromptTemplate.from_template(
    "你是一个乐于助人的助手。告诉我一个关于{topic}的有趣事实。"
)

# 3. 初始化输出解析器
# 这将把模型的邮件对象转换为一个简单字符串。
parser = StrOutputParser()
```

定义好部件后，我们现在可以使用LCEL的 `|` 运算符将它们链接在一起，形成一个链。这个 `chain` 对象是一个可运行的管道。

```python
# 4. 通过将部件连接起来构建链
chain = prompt | model | parser
```

这一行代码定义了整个工作流。`prompt` 部件将接收初始输入，其输出将发送给 `model`，而模型的输出将由 `parser` 处理。

### 执行链

要运行该链，我们调用其 `invoke()` 方法。链的输入必须与第一个部件（在此例中是提示模板）预期的变量匹配。由于我们的模板预期一个 `{topic}`，我们传递一个包含主题的字典。

```python
# 5. 用输入字典调用链
input_data = {"topic": "the Eiffel Tower"}
response = chain.invoke(input_data)

print(response)
```

运行此脚本将产生模型的一个响应，例如：

```text
埃菲尔铁塔最初是为1889年世博会建造的一个临时设施，并差点在1909年被拆除，但因被改作一个巨型无线电报天线而得以保存。
```

当 `invoke()` 被调用时，LangChain 处理整个操作序列：

1. `input_data` 字典被传递给 `prompt` 对象。
2. `ChatPromptTemplate` 将 `{topic}` 占位符替换为“埃菲尔铁塔”，并创建一个格式化提示。
3. 这个格式化提示被发送给 `model` (`ChatOpenAI`)。
4. 模型与OpenAI API通信并接收一个响应，该响应封装在一个 `AIMessage` 对象中。
5. `AIMessage` 对象被传递给 `parser` (`StrOutputParser`)。
6. 解析器从 `AIMessage` 中提取字符串内容。
7. 最终字符串作为 `invoke()` 调用的结果返回。

你现在已成功构建并运行了你的第一个LangChain应用。这个简单的 `prompt | model | parser` 结构构成了你将构建的几乎所有应用的起点。在接下来的章节中，我们将通过添加更精致的部件来扩展此模式，用于管理记忆、连接外部数据，并实现更复杂的、多步推理 (inference)。

## 参考资料

- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) — LangChain Developers (2024)
  Publisher: LangChain
  LangChain表达式语言的官方文档，详细介绍了如何将模块化组件组合成可运行的链。
- [OpenAI API Reference: Models](https://platform.openai.com/docs/models/overview) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI语言模型的官方参考资料，包括GPT-3.5 Turbo，用于理解模型功能和API交互。
- [LangChain Core Concepts](https://python.langchain.com/docs/concepts/) — LangChain Developers (2024)
  Publisher: LangChain
  LangChain核心架构概念的官方概述，包括可运行对象、提示、模型和解析器等组件，以及它们如何相互作用。
- [Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI关于设计有效提示以优化模型性能和响应的最佳实践指南。

---

[上一节](03-%E8%AE%BE%E7%BD%AE%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83.md) · [下一节](../02-%E6%A8%A1%E5%9E%8B%E3%80%81%E6%8F%90%E7%A4%BA%E4%B8%8E%E8%A7%A3%E6%9E%90%E5%99%A8/01-%E4%B8%8ELLM%E5%92%8C%E8%81%8A%E5%A4%A9%E6%A8%A1%E5%9E%8B%E4%BA%A4%E4%BA%92.md)
