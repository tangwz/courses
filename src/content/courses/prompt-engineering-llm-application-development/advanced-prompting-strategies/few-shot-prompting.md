---
course: "prompt-engineering-llm-application-development"
chapter: "advanced-prompting-strategies"
lesson: "few-shot-prompting"
sourceId: 2976
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-2-advanced-prompting-strategies/few-shot-prompting"
title: "小样本提示"
description: "在提示词中提供示例（上下文学习），以引导大型语言模型的响应。"
order: 2
plots: []
sourceHash: "f57314b3728fab73703cecd35db6b934e6c6167dd94b6ca13692c61097ab3260"
sourceCorrections: []
---

小样本提示是一种高级技术，用于引导大型语言模型（LLM），它与那些完全依赖LLM已有知识和指令遵循能力的方法形成对比。它利用大型语言模型内部被称为**上下文 (context)学习**的机制来实现其效果。您不再仅仅告诉模型*做什么*，而是通过在提示词 (prompt)中直接提供少量示例（$k$ 个示例，其中 $k$ 通常很小，例如 1 到 5 个）来*展示*它如何做。

这种方法在以下情况特别有效：

1. 目标任务新颖或具体，可能无法通过简单指令准确说明。
2. 您需要特定格式的输出，这种格式通过展示比描述更容易传达。
3. 您希望引导模型形成特定风格、语气或推理 (inference)模式。
4. 零样本表现无法满足所需的准确性或一致性。

### 上下文 (context)学习的工作原理

需要理解的是，小样本提示*不*会更新大型语言模型的底层参数 (parameter)（权重 (weight)）。这种“学习”是动态发生的，在单次 API 调用的上下文窗口内。模型分析提供的示例，识别连接示例输入与示例输出的模式。然后，它将这种推断出的模式应用到示例后实际的输入查询上。

把它想象成在某人执行一项特定任务之前，您立即为他们提供简短、即时的培训。他们利用这些示例来理解即时需求，但不会从根本上改变其长期知识。

### 小样本提示词 (prompt)的构建方式

一个典型的小样本提示词遵循一致的结构：

1. **（可选）任务说明：** 概述总体目标的简要指令。
2. **示例 1：**
   - 输入标签（例如，`Text:`，`Input:`）+ 示例输入 1
   - 输出标签（例如，`Sentiment:`，`Output:`）+ 示例输出 1
3. **示例 2：**
   - 输入标签 + 示例输入 2
   - 输出标签 + 示例输出 2
4. **... (重复 k 个示例)**
5. **实际任务输入：**
   - 输入标签 + 实际输入
6. **输出指示符：**
   - 输出标签（大型语言模型会在此之后补全文本）

在示例和最终查询之间保持标签和格式的一致性对成功很重要。

### 示例：情感分类

假设我们希望将客户反馈分类为 `正面`、`负面`或`中立`，但我们希望确保模型准确使用这些标签。

```text
对以下客户评论进行情感分类。

文本: "这款设备的电池续航太棒了！"
情感: 正面

文本: "屏幕太容易刮花了。"
情感: 负面

文本: "它按时送达了。"
情感: 中立

文本: "客户支持很有帮助，但响应速度较慢。"
情感:
```

通过提供示例，我们引导模型：

- 理解任务是情感分类。
- 识别我们期望的特定标签（`正面`、`负面`、`中立`）。
- 推断出对最终输入进行分类的模式。

大型语言模型遵循该模式，很可能会用 `负面` 或 `中立` 补全提示，这取决于它在类似数据上进行的学习以及示例的指引。

### 示例：简单代码生成（从文档字符串到函数）

小样本提示有助于引导代码生成任务。假设您想根据描述生成简单的 Python 函数。

```python
# 根据描述生成一个Python函数。

描述: """将两个数字相加。"""
函数:
def add(a, b):
  """将两个数字相加。"""
  return a + b

描述: """连接两个字符串，用空格分隔。"""
函数:
def concatenate_strings(s1, s2):
  """连接两个字符串，用空格分隔。"""
  return s1 + " " + s2

描述: """计算矩形的面积。"""
函数:
```

模型会观察到模式：获取描述，将其用作文档字符串，定义一个具有合适参数 (parameter)的函数，并实现其逻辑。它很可能会生成如下内容：

```python
def rectangle_area(length, width):
  """计算矩形的面积。"""
  return length * width
```

### 选择有效示例

示例的质量极大地影响结果。在选择时请考虑以下几点：

- **相关性：** 示例必须与目标任务和您将提供的输入类型直接相关。
- **清晰度和格式：** 使用清晰的输入/输出标签，并在所有示例和最终查询中保持一致的格式。分隔符（如换行符）很重要。
- **准确性：** 确保示例输出根据所需逻辑或格式与示例输入正确对应。不正确的示例会使模型混淆。
- **多样性（如果需要）：** 如果任务涉及不同情况（例如，不同情感类型），包含覆盖该范围的示例。
- **简洁性：** 虽然展示模式是必需的，但过长的示例会占用宝贵的上下文 (context)窗口空间。

通常，少量示例（$k=1$ 到 $k=5$）就足够了。这就是为什么它被称为“小样本”提示。添加过多示例可能不会提升表现，并且可能超出模型的上下文长度限制。

### 图示：小样本提示结构

以下图示说明了为翻译设计的小样本提示中的流程。

> 此图展示了指令、示例（输入/输出对）和最终查询是如何构建在发送给大型语言模型的提示词 (prompt)中，然后大型语言模型根据学习到的模式生成结果。

### 考量与权衡

- **上下文 (context)窗口：** 小样本提示由于包含示例，比零样本提示消耗更多令牌。对于上下文窗口较小的模型或处理长输入/输出时，这是一个重要考量。您可能需要限制示例的数量（$k$）或缩短它们。
- **示例敏感性：** 模型表现有时对所选的具体示例、其格式甚至其顺序出人意料地敏感。通常需要实验才能找到最有效的示例组合。
- **成本：** 由于提示词 (prompt)更长，使用小样本提示的 API 调用通常比相同任务的零样本调用成本更高。
- **无法替代微调 (fine-tuning)：** 尽管功能强大，小样本学习提供临时的、特定于上下文的指导。如果某项任务需要针对各类输入重复执行并保持高准确性，微调模型本身可能是一种更有效的长期方案，尽管这需要更多数据和计算资源。

小样本提示是您提示工程 (prompt engineering)工具箱中一种有价值的技术，提供了一种有效引导大型语言模型完成特定任务和输出格式的方法，无需修改模型本身。与零样本提示相比，它提升了控制力，使得更复杂的交互成为可能。

## 参考资料

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; Pages: 1877-1901; DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  这篇基础性论文首次介绍并广泛论证了大型语言模型，特别是GPT-3，通过上下文实现的强大少样本学习能力。
- [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586) — Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig (2021)
  Journal: ACM Computing Surveys; DOI: [10.48550/arXiv.2107.13586](https://doi.org/10.48550/arXiv.2107.13586)
  对自然语言处理中各种提示方法进行了系统性概述，其中包括少样本提示及其理论基础。
- [Prompt engineering techniques](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2024)
  Publisher: OpenAI
  提供关于有效提示工程的实用指导和示例，明确将少样本提示作为一项指导LLM的策略。
- [Prompt Engineering: A Primer for Large Language Models](https://www.packtpub.com/product/prompt-engineering-a-primer-for-large-language-models/9781805105779) — Paul Daou, David Khoury (2023)
  Publisher: Packt Publishing
  本书介绍了提示工程技术，涵盖了少样本提示及其在LLM开发中的应用。
