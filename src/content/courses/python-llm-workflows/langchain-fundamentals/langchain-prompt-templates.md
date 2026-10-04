---
course: "python-llm-workflows"
chapter: "langchain-fundamentals"
lesson: "langchain-prompt-templates"
sourceId: 5143
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-4-langchain-fundamentals/langchain-prompt-templates"
title: "创建和使用提示模板"
description: "学习使用 LangChain 创建灵活且可复用的提示结构。"
order: 4
plots: []
sourceHash: "21e5fed789f6f4bb192ec16c36842976826f18e24c0db97ed5e4f2605620384e"
sourceCorrections: []
---

与大型语言模型进行交互，通常需要向它们发送精心编写的文本，这些文本被称为提示（prompts）。虽然你可以在 Python 代码中直接将这些提示写成字符串，但随着应用程序复杂度的增加，这种方式很快就会变得繁琐且难以管理。想象一下，如果每次都需要相同的基本指令格式，但输入不同，那么硬编码这些变体将导致重复，并使更新变得困难。

LangChain 通过\*\*提示模板（Prompt Templates）\*\*为这个问题提供了一个简洁的解决方案。提示模板本质上是一种预设的、用于生成提示的方法。它包含一个带有占位符的文本字符串，这些占位符用于稍后填充动态值。这使你能够将提示的固定结构与可变部分分离，从而提高可复用性和可维护性。

### 定义一个简单的提示模板

LangChain 中处理基于文本的提示的主要类是 `PromptTemplate`。让我们看看如何使用它。

首先，你需要导入该类：

```python
from langchain.prompts import PromptTemplate
```

接下来，定义你的模板字符串。使用花括号 `{}` 表示稍后将插入变量的占位符。让我们创建一个模板，用于根据产品名称及其特性生成产品描述摘要：

```python
template_string = """
总结产品“{product_name}”的主要特性。
关注以下方面：{features}。
提供一份适合电商列表的简洁摘要。
"""
```

现在，创建一个 `PromptTemplate` 实例。你需要提供两个主要参数 (parameter)：

1. `input_variables`：一个字符串列表，指定模板字符串中使用的变量名称（不带花括号）。
2. `template`：模板字符串本身。

```python
prompt_template = PromptTemplate(
    input_variables=["product_name", "features"],
    template=template_string
)

print(prompt_template)
```

这将输出关于模板对象的信息，确认它期望的输入变量并呈现模板结构。

### 格式化提示

一旦你有了 `PromptTemplate` 对象，就可以通过为输入变量提供值来生成一个完整的提示字符串。这可以通过使用 `.format()` 方法并以关键字参数 (parameter)的形式传递值来完成：

```python
# 定义此实例的具体值
product = "Quantum Laptop X"
product_features = "16-inch OLED display, 12-core CPU, 32GB RAM, 1TB SSD, 20-hour battery life"

# 格式化提示
formatted_prompt = prompt_template.format(product_name=product, features=product_features)

print(formatted_prompt)
```

输出将是完整格式化的提示字符串，已准备好发送给 LLM：

```text
总结产品“Quantum Laptop X”的主要特性。
关注以下方面：16英寸 OLED 显示屏、12核 CPU、32GB 内存、1TB SSD、20小时电池续航。
提供一份适合电商列表的简洁摘要。
```

### 为什么要使用提示模板？

使用 `PromptTemplate` 相对于手动构建提示字符串具有以下几个优点：

- **可复用性：** 定义一个模板一次，然后用不同的输入重复使用它。
- **一致性：** 确保所有从模板生成的提示都遵循完全相同的结构和指令，这对于从 LLM 获取可靠的输出很重要。
- **可维护性：** 如果你需要更改提示的指令或结构，只需在一个地方更新模板即可。
- **清晰性：** 将静态提示指令与动态数据分离，使你的代码更易于阅读和理解。
- **参数 (parameter)化：** 明确定义提示的哪些部分是可变的，从而减少插入数据时出错的可能性。

### 与 LangChain 模型集成

虽然我们一直专注于创建和格式化模板，但最终目标是将这个格式化的提示与 LLM 一起使用。如前所述，LangChain 为各种语言模型（如 `LLM` 或 `ChatModel` 包装器）提供了接口。一个 `PromptTemplate` 对象可以顺畅地融入这些工作流程中。

这是一个简化示例，展示了格式化提示的典型用法（假设你已有一个名为 `llm` 的 LLM 对象，并已按“核心理念”章节所述配置好）：

```python
# 假设 'llm' 是一个已初始化的 LangChain LLM 接口
# from langchain_openai import OpenAI # 示例导入
# llm = OpenAI(api_key="YOUR_API_KEY") # 示例初始化

# 1. 定义模板
template_string = "将以下英文文本翻译成法语：{english_text}"
prompt_template = PromptTemplate(
    input_variables=["english_text"],
    template=template_string
)

# 2. 使用特定输入格式化提示
formatted_prompt = prompt_template.format(english_text="Hello, how are you?")

# 3. 将格式化提示发送给 LLM
# response = llm.invoke(formatted_prompt) # 适用于较新 LangChain 版本的 invoke 方法

# print(response)
# 预期输出可能为：“Bonjour, comment ça va ？”
```

这种结构，即模板格式化输入，然后将输入传递给模型，是构建更复杂序列的根本，这些序列被称为**链（Chains）**，我们将在下一章中介绍它们。

### 有效模板设计的建议

创建好的提示模板既是艺术也是科学。以下是一些指导原则：

- **明确具体：** 清晰说明你希望 LLM 执行的任务。避免模棱两可。
- **提供背景信息：** 如果有必要，为 LLM 提供背景信息或定义一个角色（例如，“你是一个擅长技术写作的有用助手。”）。
- **结构化指令：** 如果有助于说明多部分指令，可以在模板内使用项目符号或编号列表等格式。
- **引导输出格式：** 如果你需要特定格式的输出（如 JSON 或列表），请在模板中明确要求。输出解析器（稍后会讲到）可以帮助规范化这一点。
- **迭代改进：** 不要期望你的第一个模板是完美的。进行实验，用不同的输入测试输出，并根据结果改进模板。

掌握提示模板是有效使用 LangChain 的一个重要步骤。它们提供了可靠指导 LLM 所需的结构，是创建复杂应用程序的一个构成部分。

## 参考资料

- [Prompt templates | LangChain](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFSXRZNLQE17-fusnt--gnsWfmMIKcN5TziOSgRtvkbRMy-Hv-_pqHUV1KOcGxq9cgWYmBKEffGf3zTX4RCSgMks5Dq7bGyXFC-pmunrWwfQP3A2n76dIXSb-64AcuB6HKW6EnJ2e8nrIRxMiRa4f8br4z0zAvStQ==) — LangChain (2024)
  LangChain PromptTemplate 类的官方文档，解释了其API和用法。
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — Prompt Engineering Guide Community (2024)
  一个社区维护的资源，详细介绍了提示工程技术，包括结构化提示设计原则。
- [Learning LangChain: Build and deploy powerful LLM applications](https://www.packtpub.com/product/learning-langchain) — Jonathon Gordon (2023)
  Publisher: Packt Publishing
  一本关于使用LangChain构建LLM应用程序的实用书籍，包含提示工程和模板实现的章节。
