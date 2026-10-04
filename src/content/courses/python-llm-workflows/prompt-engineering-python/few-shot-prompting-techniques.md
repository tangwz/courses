---
course: "python-llm-workflows"
chapter: "prompt-engineering-python"
lesson: "few-shot-prompting-techniques"
sourceId: 5197
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-8-prompt-engineering-python/few-shot-prompting-techniques"
title: "少量样本提示技巧"
description: "在提示中使用示例（少量样本学习）来引导LLM行为。"
order: 2
plots: []
sourceHash: "528ee214bfcb909d6a5cd0b64c9ff3ca714cd8743ddf05b6d79583b7bce3ba5d"
sourceCorrections: []
---

大型语言模型（LLM）具有很强的能力，但相比只被告知*做什么*，当被展示*如何*完成任务时，它们的表现通常更好。少量样本提示就派上用场了。少量样本提示是一种直接在提示中包含几个完成任务的示例（样本）的技术。这与零样本提示不同，零样本提示是一种只提供指令而没有任何示例的方法。

可以把它看作是在要求模型解决新问题之前，给它一个迷你教程。这些示例作为演示，引导模型达到预期的输出格式、风格或推理 (inference)过程。这种方法对于期望输出结构具体明确的任务，或者当任务本身需要某种模式而仅凭指令无法立即看出时，尤其有效。

### 零样本、单样本与少量样本

让我们明确一下这些区别：

- **零样本：** 提示只包含指令和输入。模型完全依靠其预训练 (pre-training)的知识。
  - *示例：* `Translate to French: Hello world`
- **单样本：** 提示在实际输入前包含一个示例。这为模型提供了一次演示。
  - *示例:*

    ```
    Translate to French:
    sea otter => loutre de mer
    Hello world =>
    ```
- **少量样本：** 提示提供多个示例（通常是2到5个，有时会更多）。这提供了更强的引导，并帮助模型更好地归纳模式。
  - *示例:*

    ```
    Translate to French:
    sea otter => loutre de mer
    cheese => fromage
    blue sky => ciel bleu
    Hello world =>
    ```

### 为什么使用少量样本提示？

提供多个示例有以下几个好处：

1. **准确性提升：** 对于许多任务，特别是涉及特定格式或分类的任务，少量样本提示的表现明显优于零样本提示。这些示例限制了模型可能的输出。
2. **格式控制：** 如果您需要精确的输出格式（如JSON、特定的首字母大写或某种句子结构），少量样本示例是说明这一点的有效方法。
3. **任务明确：** 复杂的指令有时可能模糊不清。示例可以具体说明预期结果。
4. **处理新任务：** 当要求模型执行它在训练期间可能未明确见过的任务（但涉及它*确实*理解的模式）时，少量样本学习有助于模型快速适应。

### 在Python中构建少量样本提示

在Python中，您可以使用简单的字符串格式化方法，或者使用LangChain等库的更结构化方法来构建少量样本提示。

#### 基本字符串格式化

您可以使用f-strings或`str.format()`来动态构建提示。

```python
# 输入/输出对的示例
examples = [
    {"input": "A friendly water mammal.", "output": "sea otter"},
    {"input": "A dairy product made from milk.", "output": "cheese"},
    {"input": "The color of the atmosphere on a clear day.", "output": "blue sky"}
]

# 我们希望模型处理的新输入
new_input = "A large, gray animal with a trunk."

# 构建提示字符串
prompt_parts = ["Identify the object from the description.\n"]
for example in examples:
    prompt_parts.append(f"Description: {example['input']}")
    prompt_parts.append(f"Object: {example['output']}\n") # 添加换行符用于分隔

# 添加最终输入
prompt_parts.append(f"Description: {new_input}")
prompt_parts.append("Object:") # 提示模型给出最终输出

final_prompt = "\n".join(prompt_parts)

print(final_prompt)

# 预期输出:
# Identify the object from the description.
#
# Description: A friendly water mammal.
# Object: sea otter
#
# Description: A dairy product made from milk.
# Object: cheese
#
# Description: The color of the atmosphere on a clear day.
# Object: blue sky
#
# Description: A large, gray animal with a trunk.
# Object:
```

这个`final_prompt`字符串随后会被发送到LLM API。模型会识别示例中的模式，并很可能输出“elephant”。

#### 使用LangChain构建结构化提示

像LangChain这样的框架提供了管理提示的方法，特别是少量样本提示。您可以使用`FewShotPromptTemplate`之类的类。（我们已经在第4章了解了`PromptTemplate`；`FewShotPromptTemplate`在此基础上构建）。

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 定义示例
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

# 定义每个示例的格式模板
example_prompt = PromptTemplate.from_template("Input: {input}\nOutput: {output}")

# 定义整体的少量样本提示模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of the input word.",
    suffix="Input: {user_input}\nOutput:", # {user_input} 是最终输入的变量
    input_variables=["user_input"], # 指定最终输入的变量名
    example_separator="\n\n" # 示例之间的分隔符
)

# 为新输入格式化提示
final_prompt = few_shot_prompt.format(user_input="hot")

print(final_prompt)

# 预期输出:
# Give the antonym of the input word.
#
# Input: happy
# Output: sad
#
# Input: tall
# Output: short
#
# Input: hot
# Output:
```

使用`FewShotPromptTemplate`可以使示例管理、格式化和整体提示结构更清晰，尤其当提示变得更复杂时。

### 选择有效的示例

示例的质量非常重要。糟糕的示例可能会让模型困惑或导致它重复错误。请记住以下几点：

- **相关性：** 示例应与您希望模型执行的实际任务紧密匹配。
- **清晰度：** 每个示例中的输入-输出关系应清晰明确，没有歧义。
- **一致性：** 确保示例中输出的格式和风格与您对新输入的期望保持一致。
- **多样性（如适用）：** 如果任务涉及变体，请尝试包含涵盖不同方面或边缘情况的示例。但不要让示例相互矛盾。
- **准确性：** 仔细检查您的示例是否正确。模型会从中学习，包括任何错误。

### 何时使用少量样本提示

少量样本提示在以下情况中特别有效：

- **文本分类：** 引导模型使用特定的类别标签。
- **格式转换：** 将数据从一种结构转换为另一种结构（例如，文本到JSON）。
- **代码生成：** 提供特定模式的代码片段示例。
- **风格控制：** 确保响应遵循特定的语气或风格（例如，正式、非正式、诗意）。
- **复杂指令：** 当简单指令不足时，通过演示来分解任务。

### 考量与限制

少量样本提示虽然功能强大，但也有一些方面需要考虑：

- **上下文 (context)窗口限制：** 每个示例都会给您的提示增加token。LLM有最大的上下文窗口大小（它们一次可以处理的总token数）。过多的示例可能会超出此限制。
- **成本：** LLM API调用通常根据输入和输出token的数量定价。包含大量示例的更长提示将增加每次API调用的成本。
- **示例选择工作量：** 寻找或创建好的示例需要仔细思考，并可能需要一些实验。

少量样本提示是实际提示工程 (prompt engineering)中的一种基本方法。通过在提示中提供具体示例，您可以大大增强引导LLM行为的能力，并获得更可靠、准确且格式正确的结果，尤其是在使用Python以编程方式实现这些提示时。

## 参考资料

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2005.14165; DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  介绍了大型语言模型（LLM）中的上下文学习概念，展示了像GPT-3这样的模型如何在不更新参数的情况下，通过少量示例执行新任务，这构成了少样本提示的基础。
- [Prompt Engineering Guide](https://www.promptingguide.ai/techniques/fewshot) — Learn Prompting (2023)
  一份关于提示工程技术的全面指南，其中有一个专门的章节解释了少样本提示的益处和实际应用，并经常引用相关的研究论文。
- [Few-shot prompt templates](https://python.langchain.com/v0.2/docs/how_to/few_shot_examples/) — LangChain documentation contributors (2024)
  LangChain 的 `FewShotPromptTemplate` 官方文档，详细介绍了其使用方法、参数以及在 Python 中编程构建少样本提示的示例。
- [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837) — Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer (2022)
  Journal: EMNLP 2022 (long); DOI: [10.48550/arXiv.2202.12837](https://doi.org/10.48550/arXiv.2202.12837)
  探究了上下文学习（少样本提示）有效性的深层原因，阐明了输入输出格式和标签空间的重要性，而非示例的表面形式或事实正确性。
