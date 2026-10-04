---
course: "prompt-engineering-llm-application-development"
chapter: "foundations-prompt-engineering"
lesson: "llm-temperature-parameters"
sourceId: 2967
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-1-foundations-prompt-engineering/llm-temperature-parameters"
title: "了解LLM温度及其他参数"
description: "学习温度、top-p和最大token数等参数如何作用于LLM生成。"
order: 5
plots: ["plots/2967-0.json"]
sourceHash: "ab0d071c95155716ca215a1b1e44d835df6d854f5d291e35479e23142772fe6e"
sourceCorrections: []
---

提示为大型语言模型（LLM）提供了生成输出的方向。了解并使用生成参数 (parameter)能让您微调 (fine-tuning)模型“如何”得出回复。这些参数就像控制文本生成过程各个方面的旋钮，作用着创造力、随机性和输出长度。熟练掌握它们对于在您的应用中取得一致且预期的结果十分重要。

可以把LLM想象成一次生成一个标记 (token)（单词或子单词）。在每一步，它会计算每个可能的下一个标记的概率。参数会更改模型如何从这个概率分布中选择“实际”的下一个标记。

### 温度

温度或许是最常调整的参数 (parameter)。它直接控制输出的随机性。

- **低温度（例如，$T = 0.1$ 至 $0.4$）：** 接近0的值会使模型更加确定和集中。它会持续选择概率最高的标记 (token)。这会使输出更可预测、更连贯，并严格遵循训练数据中最可能出现的模式。在需要事实准确性、一致性或可预测性的任务中，使用低温度，比如：

  - 事实问答
  - 代码生成
  - 事实文本总结
  - 分类
- **高温度（例如，$T = 0.8$ 至 $1.5$ 或更高）：** 接近1（甚至略高于1）的值会增加随机性。模型更有可能选择概率较低的标记，从而产生更多样化、更有创意、有时甚至出乎意料的输出。然而，极高的温度也可能导致文本连贯性差或毫无意义。在需要创意、尝试或多变性的任务中，使用高温度：

  - 头脑风暴想法
  - 创意写作（故事、诗歌）
  - 生成多个不同选项
  - 更具对话性或多样化的聊天机器人回复

从技术上讲，温度会修改应用于模型原始输出分数（logits）的 **softmax 函数**，使其在成为概率之前发生改变。低温度会使概率分布变得更尖锐，将概率质量集中在靠前的选择上。高温度则会使分布变得更平坦，从而使不太可能的标记变得更有可能被选中。



![温度对下一个标记概率的影响](plots/2967-0.json)



> 较低的温度会使概率分布变得更尖锐，使最有可能的标记概率大大增加。较高的温度会使分布变得更平坦，增加选择可能性较小的标记的机会。

使用API时，设置温度可能如下所示（具体语法取决于API提供商）：

```python
# API调用示例片段
response = client.completions.create(
  model="some-llm-model",
  prompt="Write a short story about a futuristic city.",
  temperature=0.9, # 较高温度，用于创意内容
  max_tokens=150
)

response_factual = client.completions.create(
  model="some-llm-model",
  prompt="What is the capital of Malaysia?",
  temperature=0.1, # 较低温度，用于事实召回
  max_tokens=50
)
```

### Top-p（核心采样）

Top-p，也称为核心采样，提供了另一种控制随机性的方法，它从一组更小、动态的、最有可能的标记 (token)中进行选择。

Top-p不会考虑“所有”可能的标记（像温度那样，只是调整了概率），而是只考虑累积概率超过特定阈值 $p$ 的最小标记集。然后，模型只从这个高概率标记的“核心”中进行采样。

- **示例：** 如果 `top_p = 0.9`，模型会按降序对最可能标记的概率进行求和，直到总和达到或超过0.9。然后，它只在这些选定的标记中重新分配概率，并从该子集中进行采样。如果单个最可能的标记的概率已经达到0.95，那么 `top_p=0.9` 将有效意味着只考虑该标记。如果前10个标记的总和为0.92，那么只考虑这10个标记。

Top-p通常被视为温度的替代或补充。它根据每一步概率分布的形状自适应地调整所考虑的标记数量。这可以防止模型选择极不可能的“尾部”标记（即使在适中温度下，如果分布非常平坦也可能发生），同时在存在多个好的选项时仍能保持多样性。

许多实践者建议“要么”使用温度“要么”使用top-p，并将未使用的参数 (parameter)设置为中性值（例如，如果使用 `top_p` 则 `temperature=1.0`，如果使用 `temperature` 则 `top_p=1.0`）。请查阅具体的API文档以获取最佳实践建议。

```python
# API调用示例片段
response = client.completions.create(
  model="some-llm-model",
  prompt="Suggest three innovative uses for graphene.",
  top_p=0.85, # 从前85%的概率质量中采样
  temperature=1.0, # 如果主要使用top_p，则将温度设置为中性值
  max_tokens=100
)
```

### Top-k

Top-k采样是一种更简单的替代方法。它将模型限制为只从最有可能的 $k$ 个下一个标记 (token)中进行采样。

- **示例：** 如果 `top_k = 5`，模型会识别出下一步概率最高的五个标记。然后，它会忽略所有其他标记，只从这前五个中进行采样，通常是在重新分配它们的概率之后。

尽管容易理解，但如果真正的概率质量集中在少于 $k$ 个标记中，`top_k` 有时可能会过于严格；或者如果分布非常平坦，且第 $k$ 个标记仍然可能性很小，则它可能不够严格。Top-p通常提供更具适应性的控制。

```python
# API调用示例片段
response = client.completions.create(
  model="some-llm-model",
  prompt="List common types of renewable energy.",
  top_k=10, # 仅从最有可能的10个下一个标记中采样
  max_tokens=75
)
```

### 最大标记 (token)数（最大长度 / 最大新标记数）

此参数 (parameter)对生成的回复长度设置了硬性限制，以标记为单位进行衡量。标记可以是单词、单词的一部分、标点符号或空格，具体取决于模型的标记器。

- **作用：** `max_tokens`（或类似名称如 `max_length`、`max_new_tokens`）可以防止模型生成过长、冗余或可能成本高昂的回复。
- **重要性：** 对于管理API成本（通常基于标记计数）、确保回复符合UI限制以及防止失控生成非常重要。它还会与模型的上下文 (context)窗口限制有所关联——模型可以处理的标记总数（提示 + 生成）。
- **注意事项：**
  - 设置过低可能会过早地截断回复，切断有价值的信息。
  - 设置过高可能导致不必要的冗长和更高的成本。
  - 请留意该参数是限制“总”标记数（提示 + 生成）还是只限制“新生成”的标记数（`max_new_tokens`）。API文档是您在此方面的指南。较新的API通常倾向于使用 `max_new_tokens`。

```python
# API调用示例片段
response = client.completions.create(
  model="some-llm-model",
  prompt="Explain the concept of recursion in programming. Be concise.",
  temperature=0.3,
  max_tokens=100 # 将生成的解释限制在100个标记
)
```

### 其他常用参数 (parameter)

您可能会遇到其他一些参数：

- **频率惩罚：** 一个数值（例如，0.0到2.0），它根据标记 (token)在已生成文本中出现的频率进行惩罚。值越高，越不鼓励重复。
- **存在惩罚：** 类似于频率惩罚，但仅因标记出现就进行惩罚，无论频率如何。这鼓励引入新的话题或想法。
- **停止序列：** 一个字符串列表（例如，`["\n", " Human:", " AI:"]`）。当模型生成这些序列中的一个时，它会立即停止生成，即使 `max_tokens` 尚未达到。这对于构建对话或防止模型在自然终点后生成不需要的文本很有用。

### 选择合适的参数 (parameter)

没有单一的“最佳”参数设置；最佳配置很大程度上取决于您的具体任务：

- **事实性/确定性任务：** 从低 `temperature`（例如，0.1-0.4）开始，并可能设置 `top_p=1.0`。
- **创意性/多样化任务：** 从较高 `temperature`（例如，0.7-1.0）或适中 `top_p`（例如，0.8-0.95）开始。如果可以接受一些重复，您也可以稍微降低惩罚。
- **平衡性任务：** 适中 `temperature`（例如，0.5-0.7）通常是一个好的起点。

**试用必不可少。** 了解这些参数如何影响特定用例输出的最佳方法是尝试不同的值并观察结果。从默认的API设置或上述建议值开始，然后根据您是否需要更多创意、更强的集中性、更少重复或不同的输出长度进行增量调整。记录下哪些组合对不同类型的提示效果最好。

了解这些参数能让您对LLM的输出有更细致的控制，而不仅仅是提示本身。在接下来的练习中，您将能够直接体验这些设置并亲眼看到它们的效果。

## 参考资料

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1904.09751](https://doi.org/10.48550/arXiv.1904.09751)
  引入了原子采样（top-p），并分析了其与包括温度和top-k在内的其他神经文本生成解码策略的有效性。
- [API Reference: Chat Completions](https://platform.openai.com/docs/api-reference/chat/completions) — OpenAI (2024)
  Publisher: OpenAI
  官方文档，详细介绍了OpenAI API中可用的生成参数，包括温度、top-p、最大token数和各种惩罚项。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  自然语言处理的经典参考文献，提供了关于语言模型、文本生成以及token概率和采样机制的全面背景知识。
- [Generation Strategies for Language Models](https://huggingface.co/docs/transformers/main/en/generation_strategies) — Hugging Face (2024)
  Transformers库中实现的各种文本生成策略的实用指南，提供了对温度、top-k和top-p等参数的清晰解释和示例。
