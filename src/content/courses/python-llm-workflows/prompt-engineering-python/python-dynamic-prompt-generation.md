---
course: "python-llm-workflows"
chapter: "prompt-engineering-python"
lesson: "python-dynamic-prompt-generation"
sourceId: 5201
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-8-prompt-engineering-python/python-dynamic-prompt-generation"
title: "使用 Python 进行动态指令生成"
description: "运用 Python 根据上下文或数据以编程方式创建指令。"
order: 4
plots: []
sourceHash: "5b0af29382c504cca3f8c5970d075005febd58537314d1f1be238b9763913155"
sourceCorrections: []
---

许多大型语言模型（LLM）应用需要指令能够适应不断变化的情况、用户输入或获取的数据。尽管静态指令可以作为一种初始方法，但硬编码所有可能的变体是不切实际且不灵活的。因此，使用 Python 的功能动态生成指令变得必不可少。动态指令生成使您能够即时为大型语言模型（LLM）构建定制的说明，从而实现更相关、更个性化、更有效的交互。

设想一下需要生成一篇新闻文章的摘要，但同时又希望大型语言模型根据用户偏好采用特定的语气（正式、非正式、乐观）。或者考虑一个聊天机器人，需要将用户的姓名和之前的对话轮次纳入其下一条指令中。这些场景需要指令在发送给模型之前即时组合，融入具体、最新的信息。

### Python 中的动态指令生成方法

Python 提供了几种简单的方法来动态构建指令。

#### 字符串格式化 (f-strings)

Python 的 f-strings（格式化字符串字面量）通常是将变量和表达式嵌入 (embedding)到指令文本中最直接、最易读的方式。它们允许您创建模板，其中占位符在运行时填充。

```python
user_name = "Alex" # 用户名
topic = "the future of renewable energy" # 话题
difficulty = "beginner" # 难度

# 使用 f-string 插入变量
prompt = f"""
Explain {topic} to me.
Assume I am a {difficulty} and my name is {user_name}.
Keep the explanation concise and focus on the main benefits.
"""

print(prompt)
```

这段代码会生成以下指令字符串：

```text
请向我解释可再生能源的未来。
假设我是一名初学者，我的名字是 Alex。
请保持解释简洁，并侧重于主要益处。
```

您可以在 f-string 中的花括号 `{}` 内嵌入任何有效的 Python 表达式，使其成为进行简单动态调整的功能强大的工具。

#### 条件逻辑 (if/else)

通常，您会希望根据特定条件改变指令的一部分。标准的 Python `if/elif/else` 语句非常适合此目的。您可以构建不同的指令片段，并根据应用逻辑组合它们。

```python
user_skill_level = "expert" # 用户技能水平（可来自用户资料）
query = "Explain the transformer architecture." # 查询
prompt_base = f"Question: {query}\nAnswer:"
prompt_suffix = "" # 初始化空后缀

if user_skill_level == "beginner":
    prompt_suffix = "\n请简单解释，避免技术术语。"
elif user_skill_level == "intermediate":
    prompt_suffix = "\n假设您对基本机器学习概念有一定了解。"
elif user_skill_level == "expert":
    prompt_suffix = "\n请随意包含技术细节和数学符号。"

final_prompt = prompt_base + prompt_suffix
print(final_prompt)
```

`user_skill_level = "expert"` 的输出：

```text
问题：解释 Transformer 架构。
回答：
请随意包含技术细节和数学符号。
```

这种方法允许根据运行时条件对指令进行显著的结构变化。

#### 使用循环整合数据

当您需要包含多条数据时，例如项目列表、用户评论或搜索结果，Python 的 `for` 循环是不可或缺的。您可以遍历数据并在指令中进行适当格式化。

```python
product_name = "Quantum Leap Laptop" # 产品名称
reviews = [
    "Amazing speed, boots up in seconds!", # 惊人的速度，几秒钟内启动！
    "Battery life could be better, but overall solid.", # 电池续航可以更好，但整体表现稳定。
    "A bit pricey, but the performance justifies it.", # 有点贵，但性能物有所值。
    "Screen resolution is fantastic for graphic design work." # 屏幕分辨率对于图形设计工作来说非常棒。
]

# 将评论格式化为带序号的列表，用于指令
formatted_reviews = "\n".join([f"{i+1}. {review}" for i, review in enumerate(reviews)])

prompt = f"""
Summarize the main pros and cons for the product "{product_name}" based on these user reviews:

{formatted_reviews}

Provide the summary as bullet points under 'Pros' and 'Cons'.
"""

print(prompt)
```

这会生成：

```text
请根据这些用户评论，总结产品“Quantum Leap Laptop”的主要优点和缺点：

1. 惊人的速度，几秒钟内启动！
2. 电池续航可以更好，但整体表现稳定。
3. 有点贵，但性能物有所值。
4. 屏幕分辨率对于图形设计工作来说非常棒。

请以“优点”和“缺点”为标题，提供要点总结。
```

#### 与工作流库集成 (LangChain 示例)

LangChain 等库在设计时考虑了动态指令。它们的 `PromptTemplate` 对象明确定义了输入变量，使动态生成清晰且有条理。您在第四章中了解了 `PromptTemplate`；下面是集成动态数据的方法：

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI # 示例大型语言模型集成

# 假设 llm 已初始化：llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

template_string = """
Generate a short product description for a product with the following features.
Product Name: {product_name}
Category: {category}
Important Features:
{features_list}

Target Audience: {audience}
Tone: {tone}
"""

prompt_template = PromptTemplate(
    input_variables=["product_name", "category", "features_list", "audience", "tone"],
    template=template_string
)

# 动态数据
product_data = {
    "product_name": "EcoGrow Smart Garden",
    "category": "Home & Garden",
    "features": ["Automated watering", "LED grow lights", "App connectivity", "Soil sensors"],
    "audience": "Urban dwellers with limited space",
    "tone": "Enthusiastic and eco-conscious"
}

# 使用循环格式化功能列表，然后传递给模板
formatted_features = "\n".join([f"- {feature}" for feature in product_data["features"]])

# 使用模板和动态数据生成最终指令
final_prompt_string = prompt_template.format(
    product_name=product_data["product_name"],
    category=product_data["category"],
    features_list=formatted_features, # 传递格式化后的列表
    audience=product_data["audience"],
    tone=product_data["tone"]
)

print("--- 生成的指令 ---")
print(final_prompt_string)

# 示例：在简单链中使用此指令
# chain = LLMChain(llm=llm, prompt=prompt_template)
# response = chain.run(product_name=product_data["product_name"], ...)
# print("\n--- 大型语言模型响应 (示例) ---")
# print(response) # 这部分需要 API 密钥和实际执行
```

此示例展示了如何结合 Python 的字符串格式化（在 `features_list` 的循环内）与 LangChain 的 `PromptTemplate` 来创建结构化的动态指令。`PromptTemplate` 明确定义了预期的输入，Python 代码在填充模板之前准备数据。

> 此图展示了应用逻辑如何使用 Python 结合数据和模板生成供大型语言模型使用的最终指令的流程。

### 动态生成时的注意事项

- **可读性：** 尽管是动态生成的，发送给大型语言模型的最终指令必须清晰且明确。确保您的 Python 逻辑生成结构良好的指令。
- **复杂性：** 非常复杂的条件逻辑或深度嵌套的格式化会使指令难以调试。力求生成指令的 Python 代码清晰明了。
- **调试：** 当出现问题时，检查实际生成并发送给大型语言模型的指令字符串十分重要。在开发过程中，在 API 调用前打印或记录最终指令。
- **安全性 (指令注入)：** 如果用户提供的文本直接插入到指令中，请注意指令注入的风险。恶意用户可能会精心构造输入来劫持指令的原始意图。对用于动态指令构建的用户输入进行清理或谨慎处理，尤其是在对安全性要求高的应用中。这是一个复杂的问题，但提高认识是第一步。

通过掌握 Python 动态指令生成，您能很大程度上控制大型语言模型交互，从而构建更精密、适应性更强、以用户为中心的应用。它将指令从静态说明转变为由实时上下文 (context)和数据塑造的灵活沟通工具。

## 参考资料

- [The Python Tutorial](https://docs.python.org/3/tutorial/) — Python Software Foundation (2024)
  提供 Python 核心功能的基础指南，包括字符串格式化（f-string）、条件语句和循环结构，这些对于动态提示生成非常重要。
- [Prompt templates (LangChain Documentation)](https://python.langchain.com/v0.2/docs/concepts/#prompt-templates) — LangChain (2024)
  Publisher: LangChain
  官方文档，详细介绍了 LangChain 应用程序中 `PromptTemplate` 对象的创建和使用，用于结构化和动态提示构建。
- [Prompt engineering (OpenAI Documentation)](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI 提供的一份指南，涵盖了设计有效提示、提供上下文和优化大型语言模型输出的核心原则。
- [LLM01: Prompt Injection (OWASP Top 10 for Large Language Model Applications)](https://llmtop10.com/LLM01/) — OWASP Foundation (2023)
  Publisher: OWASP Foundation
  讨论提示注入，这是使用大型语言模型的应用程序中的一项重要安全漏洞，概述其性质和缓解策略。
- [Hands-On Large Language Models with Python: Build and deploy powerful NLP applications with the latest LLM techniques and frameworks](https://www.packtpub.com/product/hands-on-large-language-models-with-python/9781837637841) — Ben Auffarth (2024)
  Publisher: Packt Publishing
  一本实践性书籍，展示了如何使用 Python 构建和集成 LLM 驱动的应用程序，其中包含与动态提示构建和使用相关的章节。
