---
course: "prompt-engineering-llm-application-development"
chapter: "advanced-prompting-strategies"
lesson: "chain-of-thought-prompting"
sourceId: 2987
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-2-advanced-prompting-strategies/chain-of-thought-prompting"
title: "思维链提示"
description: "鼓励大型语言模型在复杂问题解决中分步推理。"
order: 6
plots: []
sourceHash: "b01dec9f244d5abf0f3278df217c197e9c81edd6fa74b59aa3f42f8bed120fc0"
sourceCorrections: []
---

大型语言模型可以生成流畅的文本，回答问题，并遵循指令，但它们有时在需要多步推理 (inference)的任务上表现不佳，例如算术应用题或逻辑谜题。当你直接给模型一个复杂问题时，它可能会根据模式猜测答案，但可能无法执行所需的实际推理，从而导致错误的结果。

大型语言模型（LLMs）有时在需要多步骤推理的任务（例如算术应用题或逻辑谜题）上表现不佳。当直接提问复杂问题时，它们可能基于模式猜测答案，而非执行实际推理，从而导致结果不正确。思维链（CoT）提示是一种旨在解决这类问题的方法。它指示模型生成其达到解决方案所需的中间推理步骤，而不仅仅是要求最终答案。这就像要求某人在数学问题中“展示其解题过程”一样。

### 思维链提示的工作方式

核心想法很简单：修改提示，明确要求模型分解其思考过程。这通常涉及添加一个简单的短语，如“让我们一步一步地思考”，或者提供呈现这种逐步方法的例子（在少样本提示中）。

为什么这能奏效？大型模型经过大量文本的训练，其中包括人们解释其推理 (inference)过程的文本。通过提示步骤，你引导模型生成类似于这些解释的序列，这些序列在过程中通常包含更准确的计算或逻辑推导。

考虑一个简单的算术问题：

**标准提示：**

```text
Q: John has 5 apples. He buys 3 more boxes of apples, and each box contains 4 apples. How many apples does John have in total?
A:
```

模型可能会很快猜测一个答案，可能犯错（例如，5 + 3 + 4 = 12）。

**思维链提示（零样本）：**

```text
Q: John has 5 apples. He buys 3 more boxes of apples, and each box contains 4 apples. How many apples does John have in total?
A: Let's think step by step.
1. John starts with 5 apples.
2. He buys 3 boxes of apples.
3. Each box contains 4 apples.
4. The total number of apples in the boxes is 3 * 4 = 12 apples.
5. John's total number of apples is his initial amount plus the apples from the boxes: 5 + 12 = 17 apples.
So the final answer is 17.
```

通过添加“让我们一步一步地思考”，我们鼓励模型清晰表达中间计算过程：首先计算盒子中苹果的数量，*然后*将其加到最初的数量上。这种结构化的方法明显提高了获得正确答案17的可能性。

### 何时使用思维链提示

思维链在推理 (inference)过程不简单或不明显时特别有效：

1. **算术和数学应用题：** 涉及多步计算或步骤的问题。
2. **逻辑推理：** 需要根据已知事实进行推导的谜题或问题（例如，“如果A比B高，B比C高，那么A比C高吗？请解释原因。”）。
3. **多步指令：** 模型需要遵循一系列动作或规则的复杂任务。
4. **调试模型错误：** 当模型在使用标准提示时，持续地对某类问题给出错误答案时，思维链可以帮助找出其推理失败的*地方*，或引导其走向正确的路径。

### 变体与考量

- **零样本思维链：** 如上所示，简单添加一个触发短语，如“让我们一步一步地思考”，就能很好地运作，特别是对于能力强的模型。它不需要任何例子。
- **少样本思维链：** 对于更复杂的任务，在提示中提供呈现所需逐步推理 (inference)格式的例子通常更有效。你向模型展示针对类似问题，一个好的思维链应该是什么样子。

  ```text
  Q: [例题 1]
  A: [步骤 1 推理...] [步骤 2 推理...] 最终答案: [答案 1]

  Q: [例题 2]
  A: [步骤 1 推理...] [步骤 2 推理...] 最终答案: [答案 2]

  Q: [你想解决的实际问题]
  A:
  ```

  这为推理过程*和*输出格式提供了更强的指导。
- **输出长度和成本：** 思维链提示自然会生成更长的响应，因为它包含推理步骤。这会增加生成的token数量，从而影响API成本和延迟。
- **评估推理：** 虽然思维链提高了准确性，但重要的是要记住模型在推理步骤中仍可能犯错。有时它甚至可能产生错误的推理，但偶然得到正确的最终答案，或者反之。对于要求高可靠性的应用程序来说，评估*步骤*本身而不仅仅是最终输出可能是必要的。

### 示例：为API调用构建思维链提示

假设你正在构建一个使用大型语言模型API来解决稍微复杂一些问题的应用程序。以下是如何使用Python f-string来构建提示：

```python
problem = "一家杂货店出售袋装橘子，每袋8个，售价4.50美元；散装橘子每个售价0.60美元。如果你买一袋，每个橘子的成本是多少？比散装的便宜吗？"

prompt_cot = f"""
问题: {problem}

一步一步地分析问题，以确定袋装橘子的每个成本，并将其与散装橘子的价格进行比较。

步骤 1: 确定一袋橘子的成本和数量。
一袋的成本是 $4.50。
一袋有8个橘子。

步骤 2: 计算购买一袋时每个橘子的成本。
每个橘子的成本 = 总成本 / 橘子数量
每个橘子的成本 = $4.50 / 8

步骤 3: 执行除法运算。
4.50 / 8 = $0.5625

步骤 4: 确定一个散装橘子的成本。
一个散装橘子的成本是 $0.60。

步骤 5: 比较袋装橘子的每个成本与散装橘子的成本。
$0.5625 (袋装) 对比 $0.60 (散装)
$0.5625 小于 $0.60。

步骤 6: 得出哪个选项每个橘子更便宜的结论。
购买袋装橘子每个更便宜。

最终答案: 袋装橘子的每个成本是 $0.5625。购买袋装比购买每个售价 $0.60 的散装橘子更便宜。
"""

# 之后你就可以将这个 'prompt_cot' 字符串发送给大型语言模型 API。
# 注意: 上面的例子展示了你希望大型语言模型模仿的*理想*输出格式。
# 一个更典型的零样本提示会在问题之后结束，并添加：
# "让我们一步一步地分解这个问题："
# 或者，对于少样本提示，你会在提出新问题之前提供一个或两个像上面那样的完整例子。
```

思维链提示是你提示工程 (prompt engineering)工具包中一种有用的方法。它鼓励大型语言模型进行更有条理的处理，通常能在需要仔细推理 (inference)或计算的任务上带来更好的表现。它也使模型的“思考”过程更加透明，这对于理解和调试其行为很有帮助。正如你稍后会看到的，思维链生成的推理路径也可以与其他方法（如自洽性）结合使用，以进一步提高结果的可靠性。

## 参考资料

- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) — Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa (2022)
  Journal: NeurIPS2022; DOI: [10.48550/arXiv.2205.11916](https://doi.org/10.48550/arXiv.2205.11916)
  预印本。提出了零样本思维链提示技术，表明只需添加“让我们一步一步思考”就能在没有示例的情况下显著提升推理能力。
- [Prompt engineering](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGwwR2-03n8YtWUpvYZkgMBhi7_7CAOycUzy7E7xdYrasSvZa8XDNS3rg-GywbTeY8kLGvGE-Sz5AzgOiNzyWDhZrQmQNn1jX6DKemsg5OBIOq6L6zj_wDE1J7AUZIztQy-c4MrT834aWejg6PHg1mZmjm8MfSLewSCgNsiwrWjVotaRnDtVIihu-2iKdZLylG6QKM=) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI官方指南，提供有效的提示工程实用建议和策略，包括推理技术。
