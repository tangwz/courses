---
course: "python-llm-workflows"
chapter: "advanced-langchain-chains-agents"
lesson: "langchain-chains-sequential-ops"
sourceId: 5151
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-5-advanced-langchain-chains-agents/langchain-chains-sequential-ops"
title: "理解用于顺序操作的链"
description: "了解链如何连接多个LLM调用或构成部分形成序列。"
order: 1
plots: []
sourceHash: "a67cf7c2ed587f44da27c39aca973bed936e5260d9d9344697669c7f6ebab312"
sourceCorrections: []
---

LangChain包含主要的构成部分，例如与语言模型交互、编写有效的提示以及解析生成的结果。尽管这些单独的元素本身已很强大，但许多应用需要多次调用大型语言模型。通常，你需要执行一系列操作，其中一步的输出会作为下一步的输入。这就是LangChain链式机制发挥作用的地方。

可以将链视为你的大型语言模型工作流的流水线。它们提供了一种结构化的方式，将多个构成部分（例如大型语言模型、提示模板、输出解析器，甚至其他链）连接成一个连贯的序列。通过将这些元素连接起来，你可以构建执行多步推理 (inference)、数据转换或任务分解的应用。

### 链的顺序特性

其核心是，链按照预定的顺序执行一系列步骤。其显著特点是，序列中一个步骤生成的输出通常会成为后续步骤的输入。这允许你通过将复杂的任务分解为更小、更易管理的部分来组织它们。

例如，设想你想要：

1. 获取产品描述。
2. 根据描述生成一个简短、吸引人的营销口号。
3. 将该口号翻译成法语。

这需要两次独立的LLM调用。链允许你按顺序执行这些调用，自动将第一步生成的口号传递给翻译步骤。

### 简单链：LLMChain

最基本的链类型是`LLMChain`，我们之前提到过。它结合了提示模板、语言模型，以及可选的输出解析器。虽然简单，但`LLMChain`实例是常用于更复杂的顺序链中的构成单元。

```python
# 假设 'llm' 是一个已初始化的语言模型实例
# 假设 'prompt_template' 是一个已初始化的 PromptTemplate 实例

from langchain.chains import LLMChain

# LLMChain 接收一个提示模板和一个 LLM
basic_chain = LLMChain(llm=llm, prompt=prompt_template)

# 你可以通过为提示提供输入变量来运行它
input_data = {"product": "A durable, lightweight hiking backpack"}
response = basic_chain.run(input_data)
print(response)
# 预期输出：由 LLM 根据提示生成的字符串
```

### 连接链：SimpleSequentialChain

对于每个步骤接收一个字符串输入并产生一个字符串输出的直接序列，LangChain提供了`SimpleSequentialChain`。它接收一个链列表并按顺序执行它们，将一个的输出直接作为下一个的输入。

让我们来实践口号生成和翻译的例子：

```python
# 假设 'llm' 是一个已初始化的语言模型实例
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# 链 1：生成口号
template1 = "Generate a catchy, short marketing slogan for a {product_description}."
prompt1 = PromptTemplate(input_variables=["product_description"], template=template1)
chain_one = LLMChain(llm=llm, prompt=prompt1)

# 链 2：将口号翻译成法语
template2 = "Translate the following slogan into French: {slogan}"
prompt2 = PromptTemplate(input_variables=["slogan"], template=template2)
chain_two = LLMChain(llm=llm, prompt=prompt2)

# 使用 SimpleSequentialChain 组合它们
overall_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)

# 运行组合链
product_info = "waterproof, solar-powered camping lantern"
french_slogan = overall_chain.run(product_info)
print(french_slogan)
```

当你运行此代码时，`SimpleSequentialChain`会首先使用 `product_info` 执行 `chain_one`。输出字符串（口号）随后会自动作为输入（`slogan`）传递给 `chain_two`。最终输出是 `chain_two` 的结果（法语翻译）。`verbose=True` 参数 (parameter)通过打印中间步骤来帮助呈现此流程。

### 处理多个输入/输出：SequentialChain

`SimpleSequentialChain` 方便但有局限，仅限于步骤之间进行单个字符串输入/输出。如果后续步骤需要多个先前步骤生成的信息，或者一个步骤产生多个输出怎么办？这时就用到更通用的 `SequentialChain`。

`SequentialChain` 允许你明确定义序列中每个链的输入和输出变量。这提供更大的控制权，以便管理数据如何在步骤间流动。

考虑一个场景：

1. 给出主题，生成简要说明。
2. 给出说明，识别主要关键词。
3. 给出主题和关键词，撰写一个简短的引言段落。

请注意，步骤3需要来自步骤1（原始主题）和步骤2（关键词）的输入。

```python
# 假设 'llm' 是一个已初始化的语言模型实例
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# 链 1：生成解释
template_explain = "Provide a brief explanation of the topic: {topic}"
prompt_explain = PromptTemplate(input_variables=["topic"], template=template_explain)
# 输出默认为 'text'，我们将其重命名以增加清晰度
chain_explain = LLMChain(llm=llm, prompt=prompt_explain, output_key="explanation")

# 2：识别关键词
template_keywords = "Identify the main keywords in the following text:\n{explanation}"
prompt_keywords = PromptTemplate(input_variables=["explanation"], template=template_keywords)
chain_keywords = LLMChain(llm=llm, prompt=prompt_keywords, output_key="keywords")

# 链 3：撰写引言
template_intro = "Write a short introductory paragraph about {topic} using these keywords: {keywords}"
prompt_intro = PromptTemplate(input_variables=["topic", "keywords"], template=template_intro)
chain_intro = LLMChain(llm=llm, prompt=prompt_intro, output_key="intro_paragraph")

# 使用 SequentialChain 组合
complex_chain = SequentialChain(
    chains=[chain_explain, chain_keywords, chain_intro],
    input_variables=["topic"], # 整个序列的输入
    # 序列的输出变量（指定你想要的）
    output_variables=["explanation", "keywords", "intro_paragraph"],
    verbose=True
)

# 运行复杂链
input_topic = "Quantum Computing"
result = complex_chain({"topic": input_topic})
print("\n--- 结果 ---")
print(f"解释:\n{result['explanation']}")
print(f"\n关键词:\n{result['keywords']}")
print(f"\n引言段落:\n{result['intro_paragraph']}")
```

在 `SequentialChain` 中，你为整个序列定义 `input_variables`，并定义希望在最后返回的 `output_variables`。LangChain根据每个 `LLMChain` 中指定的 `output_key` 和后续链期望的 `input_variables`，自动管理中间输出（`explanation` 和 `keywords`）。

### 链流程可视化

理解数据流很重要，尤其当链变得更复杂时。这是一个简单的图表，表示 `SimpleSequentialChain` 示例（口号生成和翻译）：

> 此图表展示了对产品描述进行顺序处理，通过两个LLM链来生成翻译后的口号。

### 使用链的好处

在LangChain应用中使用链带来多项益处：

- **模块化：** 链鼓励将大型、复杂的任务分解为更小、独立且更易管理的步骤。
- **可重用性：** 独立的链（如翻译链或摘要链）可以在你的应用的不同部分甚至不同项目中重用。
- **可读性和可维护性：** 明确定义操作序列，使你的应用逻辑更容易理解、调试和修改，与手动编写复杂、嵌套的LLM调用相比，有明显优势。
- **抽象性：** 链隐藏了管理中间输入和输出的底层复杂性，让你能专注于整体工作流逻辑。

链提供了一种构建顺序工作流的强大机制。然而，有些任务需要更动态的行为，其中下一步不是预先确定的，而是取决于前一步的结果。对于这类情况，LangChain 提供了代理（Agents），它们发挥LLM的推理 (inference)能力来决定下一步要执行的动作。我们将在后续章节中讨论代理（Agents）。

## 参考资料

- [Chains](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGPzn1iW6eHH6lSS6zlkkojSfl2T8Byg7nolL13nzn6xq5d5O8fNtBo_pE7YxWZbyR4hsQTIQn_jEhJnj-5vFMLlrkRE0L0fgRc65hfLc-1tJ8xT97Bg3SD7iECWwwYPmKHtAQuMVCzLw9VA4E4FygZ9WUogkuzWQd7Jzk=) — LangChain Authors (2024)
  Publisher: LangChain
  LangChain链的官方文档，包含其目的、类型和实现细节。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  介绍了“思维链”提示技术，这在概念上支撑了将复杂问题分解为大型语言模型顺序推理步骤的思想。
