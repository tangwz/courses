---
course: "building-llm-apps-with-langchain"
chapter: "constructing-chains"
lesson: "creating-sequential-chains"
sourceId: 7419
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-3-constructing-chains/creating-sequential-chains"
title: "创建顺序链"
description: "通过将多个链条连接起来构建多步工作流，其中一个链条的输出成为下一个链条的输入。"
order: 3
plots: []
sourceHash: "d09e8fe841d061590cd8b8ba1e9d158bd624442a653f2ad0c997a58f6cf5a9de"
sourceCorrections: []
---

尽管单个链条对独立任务有效，但大多数应用都涉及一系列操作。例如，要生成一篇技术博客文章，你可能需要先创建一个大纲，然后根据大纲撰写引言，最后生成社交媒体帖子来宣传。每一步都依赖于前一步的输出。LangChain允许你使用LangChain表达式语言 (LCEL) 将这些操作按顺序组合成管道。

### 线性工作流

连接操作最直接的方式是将其组合成一个线性序列，其中每一步都将其输出传递给下一步。

考虑一个两步过程：

1. **生成标题：** 给定一个主题，一个LLM为一部戏剧创建一个吸引人的标题。
2. **撰写剧情梗概：** 给定戏剧标题，另一个LLM撰写一个简短的剧情梗概。

数据流向为直线：`主题` -> `标题` -> `剧情梗概`。

> 数据以线性序列流动。每个组件都将其输出提供给下一个。

让我们来实现这一点。我们将使用提示词 (prompt)和一个聊天模型来设置两个链条。我们使用 `StrOutputParser` 来确保模型的输出是一个干净的字符串，以便进行下一步。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 假设 OPENAI_API_KEY 已在你的环境中设置
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# 链条1: 为一部戏剧生成标题
prompt_title = ChatPromptTemplate.from_template(
    "你是一位剧作家。给定主题'{topic}'，为一部戏剧撰写一个吸引人的标题。"
)
chain_title = prompt_title | llm | StrOutputParser()

# 链条2: 为这部戏剧生成剧情梗概
prompt_synopsis = ChatPromptTemplate.from_template(
    "你是一位剧评人。给定戏剧标题'{title}'，撰写一个简短的、一段的剧情梗概。"
)
chain_synopsis = prompt_synopsis | llm | StrOutputParser()
```

为了连接这些，我们需要确保 `chain_title` 的输出与 `chain_synopsis` 的预期输入匹配。由于 `chain_synopsis` 预期一个带有 `title` 键的字典，而 `chain_title` 返回一个字符串，我们使用 `RunnablePassthrough` 通过一个字典明确地映射输出。

```python
# 使用管道操作符创建序列
overall_chain = (
    chain_title 
    | {"title": RunnablePassthrough()} 
    | chain_synopsis
)

# 使用初始主题运行链条
topic = "一个20世纪20年代爵士乐音乐家的兴衰"
final_synopsis = overall_chain.invoke({"topic": topic})

print(final_synopsis)
```

当你运行这个链条时，LangChain会执行 `chain_title`，获取结果字符串，将其映射到 `title` 键，并将其传递给 `chain_synopsis`。

严格线性流的主要限制是，中间数据通常会丢失或无法供后续步骤使用，除非明确地传递。

### 使用 RunnablePassthrough 管理状态

对于更复杂的工作流，你通常需要在多个步骤中保持状态。这在以下情况中是必要的：

- 早期链条产生多个后续链条所需的输出。
- 后续链条需要访问中间输出 *和* 原始输入。

`RunnablePassthrough.assign()` 方法通过向流经链条的字典添加新值来管理此过程，而不覆盖现有数据。它将工作流视为一个累积状态。

让我们修改之前的例子。假设剧情梗概链条需要同时知道生成的 `标题` 和原始 `主题`，以添加更多上下文 (context)。

> 带有状态管理的数据流。`assign` 将输出添加到状态中，使其可用于后续步骤。

为此，我们串联 `assign` 调用。每一步都会计算一个值并将其附加到状态。

```python
# 设置LLM和提示词
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# 链条1定义: 仅获取标题的逻辑
prompt_title = ChatPromptTemplate.from_template(
    "你是一位剧作家。给定主题'{topic}'，为一部戏剧撰写一个吸引人的标题。"
)
chain_title = prompt_title | llm | StrOutputParser()

# 链条2定义: 预期输入'title'和'topic'
prompt_synopsis = ChatPromptTemplate.from_template(
    "为一部名为'{title}'、关于'{topic}'的戏剧撰写一个简短的一段式剧情梗概。"
)
chain_synopsis = prompt_synopsis | llm | StrOutputParser()
```

现在我们构建管道。我们使用 `assign` 捕获每个阶段的输出。

```python
# 创建带有状态管理的链条
overall_chain = (
    # 步骤1: 计算标题并将其添加到数据流中。 
    # 数据流现在包含 {'topic': ..., 'title': ...}
    RunnablePassthrough.assign(title=chain_title)
    | 
    # 步骤2: 计算剧情梗概。 
    # 它可以访问数据流中的'topic'和'title'。
    RunnablePassthrough.assign(synopsis=chain_synopsis)
)

# 运行链条
input_data = {"topic": "一个20世纪20年代爵士乐音乐家的兴衰"}
result = overall_chain.invoke(input_data)

print(result)
```

此调用的输出是一个包含原始输入和所有已赋值变量的字典。

```json
{
  "topic": "一个20世纪20年代爵士乐音乐家的兴衰",
  "title": "蓝调尾声",
  "synopsis": "以20世纪20年代充满活力、混乱的“咆哮的二十年代”为背景，《蓝调尾声》记录了萨克斯手利奥“国王”克里奥尔的迅速崛起和令人心碎的衰落……"
}
```

通过使用 `RunnablePassthrough.assign`，你能够对数据管道进行精细控制。这种结构允许你保存和重用序列中任何位置的信息，这对于精密的多步骤推理 (inference)过程来说必不可少。

## 参考资料

- [How to: Sequential Chains](https://python.langchain.com/docs/how_to/sequential_chains/) — LangChain Developers (2024)
  Publisher: LangChain
  LangChain 官方文档，介绍了如何使用 SimpleSequentialChain 和 SequentialChain 构建多步骤应用程序。
- [Generative AI with Python and TensorFlow](https://www.oreilly.com/library/view/generative-ai-with/9781098145229/) — Leonie Watson, Tom Nick (2023)
  Publisher: O'Reilly Media
  一本构建生成式人工智能应用程序的实用指南，涵盖将大型语言模型功能组合成工作流的架构模式。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: arXiv preprint arXiv:2201.11903; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  介绍了思维链提示技术，该技术通过将复杂问题分解为中间推理步骤，使大型语言模型能够解决复杂问题，这与顺序操作在概念上相似。
