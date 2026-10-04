---
course: "building-llm-apps-with-langchain"
chapter: "models-prompts-parsers"
lesson: "implementing-few-shot-prompting"
sourceId: 7414
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-2-models-prompts-parsers/implementing-few-shot-prompting"
title: "实现少样本提示"
description: "通过使用 FewShotPromptTemplate 和示例选择器提供上下文示例，提升模型表现。"
order: 3
plots: []
sourceHash: "4d1249ee8085e67c903a1a73e469355799bef79746069d2cb65be58269db9be0"
sourceCorrections: []
---

尽管一套结构良好的指令可以有效引导模型，但某些任务所需的不只是指令。它们需要通过示范来学习。例如，当您需要模型遵循非常具体的输出格式，或执行具有特定类别的分类任务时，在提示中直接提供示例可以显著提升性能。这种方法称为少样本提示。

少样本提示依据的是上下文 (context)学习的原理。通过加入少量输入-输出示例（即“样本”），您是在让模型识别一种模式。模型随后将这种学到的模式应用于您提供的新输入，从而获得更准确、更稳定的结果。

### 构建一个基础少样本提示

实现少样本提示最直接的办法是将示例直接嵌入 (embedding)到您的提示字符串中。当您拥有一组小型、固定的、普遍适用于您任务的示例时，此方法很有效。

让我们以一个情感分析任务为例，我们需要将文本归为以下三类之一：`正面`、`负面`或`中性`。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 假设llm已初始化，例如：llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt_with_examples = PromptTemplate(
    input_variables=["input"],
    template="""
Classify the sentiment of the following text.

Example 1:
Text: "I'm so excited for the new product launch! It's going to be amazing."
Sentiment: Positive

Example 2:
Text: "The delivery was delayed and the item arrived damaged."
Sentiment: Negative

Example 3:
Text: "The system is functioning as expected."
Sentiment: Neutral

Now, classify this text:
Text: "{input}"
Sentiment:
"""
)

# 创建链
chain = prompt_with_examples | llm

# 使用新输入运行链
response = chain.invoke({"input": "I'm not sure how I feel about the new update."})
print(response.content)
```

```text
Neutral
```

在这个示例中，模型从提供的三个样本中学习了预期的 `文本 -> 情感` 格式以及每个类别的特点。这种硬编码方法对于固定需求来说简单而实用，但缺少灵活性。如果您有大量示例，或者需要针对不同输入选择不同的示例，这种做法就会变得不方便。

### 使用 FewShotPromptTemplate 实现动态示例

LangChain 通过 `FewShotPromptTemplate` 提供了一种更具扩展性和灵活性的方法。这个类从一组示例构建提示，并按照指定模板对其进行格式化。这种逻辑分离使得您可以独立于主提示结构来管理您的示例。

`FewShotPromptTemplate` 需要几个主要部分：

- `examples`：一个包含您示例数据的字典列表。
- `example_prompt`：一个 `PromptTemplate`，用于定义 `examples` 列表中每个示例的格式。
- `prefix`：出现在格式化示例之前的文本。
- `suffix`：出现在格式化示例之后的文本，通常包含最终的输入变量。

让我们采用这种更具结构的方法来重新构建我们的情感分类器。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# 1. 定义示例列表
examples = [
    {
        "text": "The new feature is incredibly intuitive and has improved my workflow.",
        "sentiment": "Positive"
    },
    {
        "text": "I've been on hold for over an hour. This is unacceptable.",
        "sentiment": "Negative"
    },
    {
        "text": "The package arrived on the scheduled day.",
        "sentiment": "Neutral"
    },
]

# 2. 创建一个模板来格式化每个示例
example_template = """
Text: "{text}"
Sentiment: "{sentiment}"
"""

example_prompt = PromptTemplate(
    input_variables=["text", "sentiment"],
    template=example_template
)

# 3. 组装 FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Classify the sentiment of the following text based on the examples.",
    suffix='Text: "{input}"\nSentiment:',
    input_variables=["input"],
    example_separator="\n\n"
)

# 假设 llm 已初始化
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
chain = few_shot_prompt | llm

# 运行链
response = chain.invoke({"input": "The documentation is clear, but I found a small typo."})
print(response.content)
```

```text
Neutral
```

这种方法更整洁且更易于维护。您的示例以结构化字典列表的形式存储，可以轻松地从文件或数据库加载，并且格式化它们的逻辑与主提示指令分开。

### 智能示例选择

如果您有成百上千个示例怎么办？将它们全部包含进来会超出大型语言模型的上下文 (context)窗口限制，并且效率低下。一个更好的策略是针对给定输入只选择最相关的示例。LangChain 的 `ExampleSelector` 对象就是为此而设计的。

最有效的一个选择器是 `SemanticSimilarityExampleSelector`。它会找出语义上与用户输入最接近的示例。这是通过将所有示例和输入嵌入 (embedding)到向量 (vector)空间中，然后执行相似性搜索来完成的。

要使用 `SemanticSimilarityExampleSelector`，您需要三项：

1. 一个**嵌入模型**，用于将文本转换为数值向量。
2. 一个**向量存储**，用于保存示例向量并执行高效的相似性搜索。
3. `examples` 列表。

下图说明了此选择器如何动态地构建提示。

> 使用语义相似性进行动态示例选择的流程。

让我们将此应用于一个稍微复杂一些的任务：对用户支持工单进行分类。通过选择相似的旧工单示例，我们可以帮助模型更准确地分类新工单。

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 示例支持工单
examples = [
    {"ticket": "My account is locked and I can't log in.", "category": "Account Access"},
    {"ticket": "How do I reset my password?", "category": "Account Access"},
    {"ticket": "The app crashed after the latest update on my phone.", "category": "Technical Issue"},
    {"ticket": "I am getting a 'connection error' message.", "category": "Technical Issue"},
    {"ticket": "I was charged twice for my subscription this month.", "category": "Billing"},
    {"ticket": "Can I get a refund for my last purchase?", "category": "Billing"},
    {"ticket": "What are your business hours?", "category": "General Inquiry"},
]

# 格式化每个示例的模板
example_prompt = PromptTemplate(
    input_variables=["ticket", "category"],
    template="Ticket: {ticket}\nCategory: {category}"
)

# 初始化示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(), # 嵌入模型
    Chroma,             # 向量存储
    k=2                 # 要选择的示例数量
)

# 使用选择器创建 FewShotPromptTemplate
similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Classify the support ticket based on similar past tickets.",
    suffix="Ticket: {input}\nCategory:",
    input_variables=["input"]
)

# 测试选择器
new_ticket = "I can't find the invoice for my last payment."
print(similar_prompt.format(input=new_ticket))
```

当您运行此代码时，`example_selector` 将发现关于“发票”的新工单与“账单”相关的示例最相似。因此，输出提示将只包含那两个示例，使其具有很强的上下文感知能力：

```text
Classify the support ticket based on similar past tickets.

Ticket: I was charged twice for my subscription this month.
Category: Billing

Ticket: Can I get a refund for my last purchase?
Category: Billing

Ticket: I can't find the invoice for my last payment.
Category:
```

这种动态选择使您的应用更高效。它为模型提供了针对每个特定输入最相关的上下文，这比静态、硬编码的示例有了显著的提升。通过掌握少样本提示，您可以更好地控制模型行为，并使其能够生成结构化输出，这是我们接下来将通过输出解析器讨论的话题。

## 参考资料

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: NeurIPS; DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  本文介绍了上下文学习，并展示了少量样本提示在大型语言模型（如GPT-3）中的有效性，为该技术奠定了学术基础。
- [Few-Shot Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/few_shot_examples) — LangChain Team (2024)
  LangChain官方文档，提供了关于使用`FewShotPromptTemplate`和`ExampleSelector`进行动态少量样本提示的实用指南。
- [Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2023)
  OpenAI官方提示工程指南，涵盖了与LLM交互的原则和方法，包括少量样本示例如何提高模型性能和控制力。
- [Embeddings](https://openai.com/blog/new-embedding-models-and-api-updates) — OpenAI (2024)
  OpenAI的这份指南解释了嵌入、其生成方式以及在语义搜索和示例选择等任务中的应用，这是`SemanticSimilarityExampleSelector`的基础。
