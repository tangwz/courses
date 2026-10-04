---
course: "building-llm-apps-with-langchain"
chapter: "constructing-chains"
lesson: "using-simple-llm-chain"
sourceId: 7418
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-3-constructing-chains/using-simple-llm-chain"
title: "使用简单的LLM链"
description: "学习如何使用核心LLM链将模型、提示模板和输出解析器组合成一个单一组件。"
order: 2
plots: []
sourceHash: "8ddaebe89e1e4e3eeed6168a0c22e6fabea3e35aa0fd02e5fa453196dc621590"
sourceCorrections: []
---

为了将与语言模型交互的独立组件组合成一个单一的、可重用的单元，链序列作为核心模式。它将提示模板直接连接到模型，构成最简流程。它接收用户输入，使用模板进行格式化，然后返回模型的输出。

本质上，此序列封装了接收输入变量、借助它们构建提示、将提示发送给LLM并返回结果的过程。此工作流是几乎所有LLM驱动应用的核心机制。

### 链的工作流程

数据在链中的流动是直接且可预测的。输入，通常是一个字典，为提示模板提供变量。模板随后将这些变量格式化成完整的提示字符串，并将其传递给语言模型。模型生成响应，即最终输出。

> 基本链内的数据流动，从输入变量到模型的最终输出。

### 实现基本链

让我们构建一个简单链，根据其产品简短描述为一家科技初创公司生成一个创意名称。这需要一个提示模板来引导LLM，以及一个模型来执行生成。

首先，请确保您已安装必要的库，并按照第1章所述配置了您的环境变量（例如 `OPENAI_API_KEY`）。

```python
#
# main.py
#
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. 实例化语言模型
# 我们将使用OpenAI的gpt-3.5-turbo模型，并设置较低的temperature参数
# 以鼓励更可预测、更少随机的输出。
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# 2. 定义提示模板
# 模板预期一个输入变量：'product_description'。
template = """
你是一位创意命名专家。为一家专注于{product_description}的科技初创公司生成一个引人注目的单一名称。

名称:
"""
prompt = PromptTemplate(
    input_variables=["product_description"],
    template=template
)

# 3. 创建链
# 我们使用管道运算符 (|) 连接提示和模型。
# 此语法被称为LangChain表达式语言（LCEL）。
name_generation_chain = prompt | llm

# 4. 使用输入运行链
# 我们使用.invoke()方法并传入一个包含
# 我们的提示模板所需变量的字典。
product_description = "一个用于个人财务管理的AI驱动平台"
result = name_generation_chain.invoke({"product_description": product_description})

print(result)
```

运行此代码将产生类似如下的输出：

```text
content='Finara' response_metadata={...} id='...'
```

输出是一个 `AIMessage` 对象（因为我们使用的是聊天模型）。生成的文本存储在 `content` 属性中。此对象结构保留了关于生成过程的元数据，这对于调试或日志记录很有帮助。

### 使用LCEL扩展链

尽管上述基本链运行良好，但我们通常只想要原始字符串输出，而不是完整的消息对象。为了实现此目的，我们可以使用LangChain表达式语言（LCEL）来扩展我们的序列。

LCEL 允许我们使用管道运算符 (`|`) 组合链，类似于Unix系统上的shell命令。我们可以在链的末尾添加一个输出解析器，以自动提取字符串内容。

```python
#
# main_lcel.py
#
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 组件保持不变
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
prompt = PromptTemplate(
    input_variables=["product_description"],
    template="""你是一位创意命名专家。为一家专注于{product_description}的科技初创公司生成一个引人注目的单一名称。

名称:"""
)

# StrOutputParser从消息中提取文本内容。
output_parser = StrOutputParser()

# 使用管道运算符构建链
# prompt -> llm -> output_parser
name_generation_chain_lcel = prompt | llm | output_parser

# 使用相同的输入调用链
product_description = "一个用于个人财务管理的AI驱动平台"
result = name_generation_chain_lcel.invoke({"product_description": product_description})

print(result)
```

此更新版本产生更清晰的输出：

```text
Finara
```

通过将模型的输出通过管道传递给 `StrOutputParser`，我们直接获得字符串响应。LCEL 语法简化了代码，使其更易于添加、移除或交换组件。例如，添加不同的输出解析器，正如您在第2章中学习到的，就像更改序列中的最后一个元素一样简单。

使用这种可组合模式带来以下显著益处：

- **模块化**：它将提示-模型组合打包成一个逻辑序列。
- **标准化**：所有可运行的LangChain对象都共享一个通用接口（`invoke`、`batch`、`stream`），使其可互换。
- **可组合性**：这些简单链是您将用于构建更复杂、多步骤序列的核心单元，这也是下一节的重心。

## 参考资料

- [Chains](https://python.langchain.com/docs/concepts/#chains) — LangChain (2024)
  LangChain官方文档，解释了链（包括LLMChain）作为LLM应用核心构建块的概念。
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) — LangChain (2024)
  LangChain官方文档，详细介绍了LCEL，这是组合复杂链和可运行组件的推荐方法，包括本节中使用的管道操作符。
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — DAIR.AI (2024)
  一个全面且由社区驱动的资源，为有效设计大型语言模型提示提供指导和技术。
- [Introduction to LangChain](https://python.langchain.com/docs/get_started/introduction/) — LangChain (2024)
  Publisher: LangChain
  LangChain框架的入门概述，包括其组件和帮助开发者构建LLM应用的使命。
