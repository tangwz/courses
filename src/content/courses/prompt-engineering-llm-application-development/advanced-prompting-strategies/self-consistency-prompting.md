---
course: "prompt-engineering-llm-application-development"
chapter: "advanced-prompting-strategies"
lesson: "self-consistency-prompting"
sourceId: 2990
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-2-advanced-prompting-strategies/self-consistency-prompting"
title: "自洽性提示"
description: "通过采样多个推理路径并选择最一致的答案来提高结果准确性。"
order: 7
plots: []
sourceHash: "faba94a3c04ff1ec046c57698a0fc324e8347575f5bdc38436840eb743fb50af"
sourceCorrections: []
---

思维链 (CoT) 提示方法促使大型语言模型 (LLM) 生成分步推理 (inference)，这项技术常用于提升复杂任务上的表现。然而，它生成的推理路径并非总是唯一或正确。如果多次运行相同的 CoT 提示，尤其是在温度（`temperature`）设置大于零的情况下，可能会得到不同的推理路径，从而导致潜在的不同最终答案。有些路径可能存在缺陷，而另一些则可能合理。如何能增加对最终答案的信心？

自洽性是一种高阶提示技巧，旨在解决这种变异性。它借鉴了思维链等方法，通过为同一问题生成多个推理路径，然后选择出现频率最高的答案作为最终结果。其背后的理念是，一个问题通常有多个有效的推理路径导向正确答案，而不正确的答案通常是通过更多样、一致性较低的逻辑缺陷得出的。

### 核心思想：通过多样性达成共识

自洽性不依赖于单一的 CoT 输出，而是按以下方式运作：

1. **生成多样化路径：** 使用思维链提示（或任何旨在引出推理 (inference)的提示），但利用采样技术生成多个完成内容。这通常涉及将 API 调用中的 `temperature` 参数 (parameter)设置为大于 0 的值（例如 0.5 或 0.7），以促进输出的多样性。
2. **提取答案：** 从每个生成的完成内容中，提取最终答案。这通常需要解析文本以找到一个结论性语句（例如，“最终答案是 X”）。
3. **多数投票：** 统计提取的答案。在不同推理路径中出现频率最高的答案被选作最终、最可靠的输出。

可以将其想象成咨询几位独立工作的专家。如果他们中的大多数人得出相同的结论，即使是通过略微不同的方法，您也会对该结论更有信心。

### 自洽性如何运作：逐步示例

让我们用一个简单的算术应用题来说明：

**提示：**

```text
Q: John has 5 apples. He buys 3 more boxes of apples. Each box contains 4 apples. He then gives away 6 apples. How many apples does John have left? Show your reasoning step-by-step.

A:
```

现在，假设我们使用此提示并在 `temperature=0.7` 的设置下生成了 3 个不同的回应：

**回应 1：**

```text
Step 1: Calculate the number of apples in the boxes. John buys 3 boxes * 4 apples/box = 12 apples.
Step 2: Calculate the total number of apples John has initially. John starts with 5 apples + 12 new apples = 17 apples.
Step 3: Calculate the final number of apples after giving some away. John has 17 apples - 6 apples = 11 apples.
The final answer is: 11
```

**回应 2：**

```text
Step 1: John starts with 5 apples.
Step 2: He buys 3 boxes, and each has 4 apples, so that's 3 * 4 = 12 apples.
Step 3: His total apples become 5 + 12 = 17 apples.
Step 4: He gives away 6 apples, so he has 17 - 6 = 11 apples left.
Final Answer: 11
```

**回应 3：** （包含一个推理 (inference)错误）

```text
Step 1: Calculate apples bought: 3 boxes * 4 apples = 12 apples.
Step 2: Add initial apples: 5 + 12 = 17 apples.
Step 3: Subtract apples given away: 17 - 6 = 10 apples. (Calculation error here)
So, John has 10 apples left.
The final answer is: 10
```

**应用自洽性：**

1. **生成路径：** 我们生成了 3 个回应。
2. **提取答案：** 答案是 11、11 和 10。
3. **多数投票：** 答案“11”出现了两次，而“10”出现了一次。

自洽性答案是 **11**。尽管其中一个路径包含错误，但生成路径间的共识指向了正确结果。

以下是该过程的视觉呈现：

> 一张图表展示了自洽性工作流程：一个单一的提示生成多个多样化的LLM推理路径。从每个路径中提取最终答案，并通过多数投票决定最一致的结果。

### 优点与考量

**优点：**

- **准确性提升：** 与单次 CoT 相比，通常能大幅提升在需要复杂推理 (inference)、算术或逻辑推导任务上的表现。
- **可靠性提高：** 对 LLM 生成的单一、可能存在缺陷的推理路径不那么敏感。

**考量：**

- **计算成本：** 需要多次调用 LLM API（每个路径一次），与生成单个回应相比，会增加延迟和金钱成本。
- **温度参数 (parameter)调整：** 选择合适的温度参数很重要。如果太低，路径将不够多样。如果太高，推理可能变得毫无意义。通常需要进行实验。
- **路径数量：** 更多的路径（$N$）通常能增强稳定性，但也会线性增加成本。找到一个好的平衡点（例如，$N=5$ 到 $N=10$）很常见。
- **答案提取：** 您需要一个可靠的方法来从不同推理路径中可能各异的文本格式中解析最终答案。设计明确要求答案以一致格式呈现的提示会有帮助。
- **处理平局：** 如果没有明显的多数答案，则需要确定一个策略（例如，从前几名中随机选择，或发出不确定信号）。

自洽性是一种强大的技术，当 LLM 推理输出的可靠性和准确性很重要，并且增加的计算成本可以接受时，它非常适用。它利用 LLM 的生成能力通过共识进行自我修正，使其成为构建更可靠 AI 应用的有价值工具。

## 参考资料

- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) — Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou (2023)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2203.11171](https://doi.org/10.48550/arXiv.2203.11171)
  介绍了自洽性技术，通过聚合多个思维链路径来增强大型语言模型（LLM）的推理能力。
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) — Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan (2023)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 36; DOI: [10.48550/arXiv.2305.10601](https://doi.org/10.48550/arXiv.2305.10601)
  介绍了一种更结构化的多路径推理方法，通过探索思维树的步骤，构建了与自洽性类似的思想。
