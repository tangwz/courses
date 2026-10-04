---
course: "prompt-engineering-llm-application-development"
chapter: "interacting-with-llm-apis"
lesson: "understanding-api-request-parameters"
sourceId: 3029
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-4-interacting-with-llm-apis/understanding-api-request-parameters"
title: "理解API请求参数"
description: "详细介绍常见的API参数，如模型选择、提示词、温度和最大生成字数。"
order: 4
plots: ["plots/3029-0.json"]
sourceHash: "04e16d27835d24419c03b487c42d458515427c42ecf42cfdf8424093442b987e"
sourceCorrections: []
---

与LLM API交互时，您不仅仅发送提示文本；您还会附带一组参数 (parameter)，这些参数指示模型*如何*生成其响应。可以将这些参数视为LLM行为的控制开关。尽管不同API提供商（如OpenAI、Anthropic、Google Cloud Vertex AI或Cohere）之间的具体名称和可用选项可能略有不同，但核心原理大致相同。掌握这些参数对于根据您的特定应用需求调整LLM输出很有帮助。

现在，让我们来了解您会遇到的最常用且效果显著的参数。

### 重要识别和输入参数 (parameter)

这些参数明确了您正在使用*哪个*模型以及您正在提供*什么*输入。

- **`model`**: 这个字符串参数指定了您希望使用的具体大型语言模型。例如 `gpt-4o-mini`、`claude-3-sonnet-20240229` 或 `gemini-1.5-pro-latest`。您的选择会影响模型的能力（例如，推理 (inference)能力、指令遵循）、知识截止日期、上下文 (context)窗口大小（模型可以考虑的文本量）、性能（延迟）和成本。根据任务要求选择合适的模型通常需要权衡这些因素。例如，对于简单的摘要任务，一个更简单的模型可能就足够且更具成本效益，而对于复杂的推理或创造性生成，则可能需要一个能力更强的模型。
- **`prompt` 或 `messages`**: 这是您为LLM提供实际输入文本的地方。

  - 对于更简单的、完成风格的模型或任务，您可以使用一个包含指令和上下文的`prompt`参数，作为一个单一字符串。
  - 更常见的是，特别是对于聊天优化的模型，您会使用`messages`参数。它接受一个消息对象列表，通常是字典，每个字典指定一个`role`（`system`、`user`或`assistant`）和`content`（消息文本）。这种结构化格式使您能够有效地提供上下文、指令（通常通过`system`角色）和对话历史记录。

  ```python
  # 聊天模型API调用中'messages'结构的示例
  messages_payload = [
      {"role": "system", "content": "您是一个能将英语翻译成法语的有用助手。"},
      {"role": "user", "content": "翻译以下句子：'Hello, how are you?'"}
  ]
  # 此列表将是发送到API的整体请求体的一部分。
  ```

### 控制生成长度和停止条件

这些参数 (parameter)管理生成响应的大小和终止点。

- **`max_tokens` （或类似名称，如`max_output_tokens`、`max_tokens_to_sample`）**: 这个整数参数设置了模型在其响应中被允许生成的最大token数量。Token是单词的片段；对于英文文本，一个token大约是四分之三个单词。设置此限制对于以下方面很重要：

  - **成本管理**: API使用通常按token（输入+输出）计费。限制输出token可以避免意外高额账单。
  - **延迟控制**: 更长的响应需要更多时间生成。
  - **可预测性**: 确保输出不会超出下游处理或显示限制的预期长度。
  - **防止失控生成**: 如果模型没有自然地达到停止点，则阻止其漫无目的地生成。

  请注意，如果模型的自然响应长于`max_tokens`，输出将被截断，可能会在句子或思维中间中断。您需要选择一个足够大以获得预期有用响应的值，但又要足够小以保证效率。
- **`stop` （或 `stop_sequences`）**: 此参数允许您指定一个或多个文本序列，如果模型生成了这些序列，将导致模型立即停止生成后续token。这对于以下方面很有用：

  - **结构化输出**: 例如，如果您要求模型生成一个列表并希望它在列表结束后停止，您可以包含一个停止序列，如`\n\n`或`---END---`。
  - **模仿格式**: 如果您正在提示模型生成代码或遵循特定模板，您可以使用代表该结构结束的停止序列（例如，一个闭合大括号`}`或一个特定的分隔符）。
  - **控制对话轮次**: 在聊天机器人中，您可以使用`User:`或`Human:`等停止序列，以防止模型虚构下一个用户轮次。

  API将返回生成文本，*直到但不包括*停止序列。

### 调整创造性和确定性

最常调整的参数 (parameter)可能是那些控制输出随机性和风格的参数。

- **`temperature`**: 这个浮点参数（通常范围为0.0到1.0，有时为2.0）调整输出的随机性。从技术上讲，它在应用softmax函数之前，会对logits（下一个Token的原始概率分数）进行缩放。

  - **较低的`temperature`（例如0.0 - 0.3）**: 使模型更具确定性和集中性。它会始终选择概率最高的Token。将低值用于需要事实准确性、可预测性或遵守特定格式的任务，例如代码生成、事实问答或数据提取。温度为0旨在获得最有可能的单一输出（尽管由于计算因素可能仍存在细微差异）。
  - **较高的`temperature`（例如0.7 - 1.0+）**: 增加随机性和创造性。模型更可能选择概率较低的Token，从而产生更多样、更出人意料或更有想象力的输出。将高值用于创意写作、头脑风暴、生成多个选项或模拟更像人类的对话等任务。

  
  
  ![温度对Token概率的影响](plots/3029-0.json)
  
  

  > 在较低的温度下，概率分布更集中，倾向于选择最有可能的Token。较高的温度会使分布变平，增加选择概率较低Token的机会。
- **`top_p` （核采样）**: 这个参数也是一个浮点数，通常介于0.0和1.0之间，提供了另一种控制随机性的方式。`top_p`不基于温度缩放的概率考虑所有Token，而是将选择范围限制在累积概率质量大于或等于`p`的最小Token集合中。

  - 例如，如果`top_p`为0.9，模型仅考虑那些加起来构成下一步概率分布90%的最有可能的Token。然后它从这个缩小后的集合中进行采样（通常与温度结合使用）。
  - `top_p`为1.0时考虑所有Token。非常低的`top_p`（例如0.1）会使输出高度确定性，类似于低温度。
  - `top_p`在防止模型选择非常低概率、可能无意义的Token方面特别有效，即使在较高温度下也是如此，因为它会根据概率分布的形状动态调整候选池的大小。

  **建议**: API提供商通常建议调整`temperature`*或*`top_p`其中之一，但不要同时显著调整两者。一种常见的方法是将`top_p`设置为一个较高的值（如0.9或0.95），然后调整`temperature`以获得所需的创造性与集中度水平。

### 其他可能有用的参数 (parameter)

尽管以上参数是最常见的，但API文档可能会列出其他参数：

- **`n`**: 指定为给定提示生成多少个独立的完成。这对于获取多个建议或选项很有用。请注意，这会使您的Token使用成本成倍增加。
- **`presence_penalty`**: 一个浮点数（通常为-2.0到2.0），根据Token是否*已在*当前生成的文本中出现而对其进行惩罚。正值会抑制重复。
- **`frequency_penalty`**: 类似于`presence_penalty`，但根据Token在文本中*出现的频率*对其进行惩罚。正值会降低频繁重复相同词语的可能性。
- **`logit_bias`**: 允许您手动增加或减少特定Token在输出中出现的可能性。这是一个高级功能，通常用于对生成进行细粒度控制。
- **`user`**: 一个代表发出请求的最终用户的唯一标识符字符串。模型本身不使用它进行生成，但它有助于API提供商监控潜在的滥用情况，并且在您的应用程序中追踪个人用户的使用情况也很有用。

### 找到合适的设置

了解这些参数 (parameter)是第一步。下一步是实验。最佳设置在很大程度上取决于特定的LLM、任务和所需的输出风格。

- **从默认值开始**: 从API提供商的默认设置开始。
- **系统迭代**: 一次更改一个参数（例如，调整`temperature`同时保持其他参数不变），并观察其对输出质量和特征的影响。
- **查阅文档**: 始终参考您正在使用的模型的具体API文档。它将提供准确的参数名称、有效范围、默认值，有时还会提供具体建议。
- **考虑任务**: 对于事实性任务使用低温度/`top_p`，对于创意性任务使用高温度/`top_p`。使用`stop_sequences`来强制输出结构。适当设置`max_tokens`以控制成本和完成度。

通过仔细选择和调整这些参数，您可以对LLM的输出获得显著控制，从简单的提示转向在应用程序中设计更可靠和有效的交互。

## 参考资料

- [OpenAI API Reference - Chat Completions](https://platform.openai.com/docs/api-reference/chat/create) — OpenAI (2024)
  Publisher: OpenAI
  详细介绍了如何使用OpenAI语言模型的核心API参数，包括模型选择、输入格式和生成控制。
- [The Curious Case of Neural Text Generation](https://arxiv.org/abs/1904.09751) — Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi (2020)
  Journal: ICLR; DOI: [10.48550/arXiv.1904.09751](https://doi.org/10.48550/arXiv.1904.09751)
  介绍并分析了核采样（top_p），这是一种控制语言模型生成文本多样性和质量的基本技术。
- [Anthropic API Reference - Messages API](https://docs.anthropic.com/claude/reference/messages_post) — Anthropic (2024)
  提供了来自另一领先供应商的LLM API常见参数的视角，展示了跨平台的一致性和特定参数变体。
- [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) — Isa Fulford, Andrew Ng (2023)
  Publisher: DeepLearning.AI
  一门实用速成课程，探讨了提示工程（包括参数调整）如何影响LLM行为和应用程序开发。
