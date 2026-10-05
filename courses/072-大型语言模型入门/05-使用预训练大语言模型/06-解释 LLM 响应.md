# 解释 LLM 响应

来源：[原文](https://apxml.com/zh/courses/intro-large-language-models/chapter-5-using-pre-trained-llms/interpreting-llm-responses)

[返回章节目录](README.md) · [返回课程目录](../README.md)

与大型语言模型交互时，尤其是在使用应用程序编程接口（API）时，会返回一个响应。此响应包含模型生成的文本及其他相关信息。理解这些响应的结构和内容，是有效利用模型输出的必要条件。

### API 响应的结构

与通过简单网页聊天界面交互不同（在这种情况下，您通常只看到生成的文本），API 响应通常是结构化的。这种结构的一种常见格式是 JSON（JavaScript 对象表示法），它是一种轻量级、基于文本的格式，计算机易于解析和生成，人类也相对易于阅读。

虽然确切的字段可能因具体的 LLM 服务提供商而略有不同，但大多数 API 响应都包含几项重要信息：

1. **生成的文本：** 这是您主要想要获得的输出。它通常嵌套在 JSON 结构中，可能在 `choices`、`text`、`content` 或 `output` 等键下。如果您提出要求，可能会有多个选项，但通常您会侧重于第一个或主要的那个。
2. **元数据：** 关于请求和响应处理的信息。这可以包括请求的唯一标识符（`id`）、所用模型的具体版本（`model`）和时间戳（`created`）。
3. **用量信息：** 关于所消耗计算资源的详细情况，通常以 token 计量。您通常会看到：
   - `prompt_tokens`：您发送的输入提示中的 token 数量。
   - `completion_tokens`：模型生成的文本中的 token 数量。
   - `total_tokens`：提示和生成文本的 token 总和。这对于了解成本有重要意义，因为许多服务都基于 token 用量收费。
4. **完成原因：** 一个指示，解释模型为何停止生成文本。常见原因包括达到指定的长度限制（`length`）、遇到预设的停止序列（`stop`），或被内容过滤器标记 (token)（`content_filter`）。

### API 响应示例（简化版）

让我们来看一个 JSON 响应可能是什么样子的简化示例：

```json
{
  "id": "cmpl-a1b2c3d4e5f6g7h8",
  "object": "text_completion",
  "created": 1678886400,
  "model": "example-model-v1",
  "choices": [
    {
      "text": "\nThe Large Language Model generated this sentence as a response.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 14,
    "total_tokens": 24
  }
}
```

在此示例中：

- 主要的生成文本在 `choices[0].text` 中。请注意有时会出现的前导换行符（`\n`）。
- `finish_reason` 为 "stop"，表示模型自然地完成了其响应或达到了指定的停止词。
- `usage` 部分告诉我们，提示有 10 个 token，响应有 14 个 token，本次 API 调用总共使用了 24 个 token。

### 提取核心输出

在使用 API 时，您的代码需要解析此 JSON 结构以提取您实际需要的部分，通常是生成的文本本身。在编程方面，这通常涉及访问嵌套元素，例如 `response['choices'][0]['text']`（以 Python 字典访问为例）。您可能还需要修剪提取文本开头或结尾的额外空白或格式字符。

### 用量信息为何重要

追踪 `prompt_tokens` 和 `completion_tokens` 对于管理成本和遵守 API 提供商设定的任何用量限制很重要。理解不同提示如何转化为 token 数量有助于您优化与模型的交互。

### 理解完成原因

`finish_reason` 提供关于生成过程的背景信息：

- `stop`：模型达到了自然停止点或遇到了您定义为停止信号的特定字符序列。
- `length`：模型停止，因为它达到了允许生成文本的最大 token 数量（无论是默认限制还是您在请求中指定的限制）。如果您看到此情况，生成的文本可能在思考或句子中间被截断。如果需要更长的响应，您可能需要在下一次请求中调整最大长度参数 (parameter)。
- `content_filter`：模型的输出可能被安全系统标记 (token)。在这种情况下，响应文本可能为空或被修改。

### 网页界面中的响应

当您使用网页界面（如聊天机器人网站）时，大部分结构化信息是隐藏的。界面通常直接显示响应的 `text` 部分，以便于使用。然而，其基本原理是相同的。模型仍然会处理 token，生成文本，并根据特定条件停止，即使您没有看到详细的 JSON 响应。

解释 LLM 的响应，无论是直接通过 API 还是通过网页界面，都是有效使用这些模型的基本步骤。通过理解其结构、核心文本输出、用量数据和完成原因，您可以更好地控制模型的行为，并将其功能应用于您的任务或程序中。

## 参考资料

- [OpenAI API Reference](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHNg9DpsWPHr6WVqfwjty4GqastZOHY2ncx8pyccfCXIVrnnKC2avmNkZFXYH7jzOBVDS3HTrJCWKkPjzZM8PRFORAnZlYyGD8mT86Vtuuk7VMCrHF07V75Bbqkp1f94NAmoGcvtaRfdxnfLwpEwS10OS_8bLFn) — OpenAI (2024)
  提供与OpenAI LLM API交互的详细规范，包括响应结构、token用量和各种参数。
- [Generative AI on Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview) — Google Cloud (2024)
  提供通过Vertex AI使用Google LLM服务的全面指南，涵盖API响应格式、输入/输出结构和成本考量。
- [What are tokens?](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) — OpenAI (2024)
  Publisher: OpenAI
  解释了token作为LLM处理基本单元的概念，它们在提示和完成长度中的作用，以及对使用成本的影响。

---

[上一节](05-%E5%8F%91%E9%80%81%E6%82%A8%E7%9A%84%E7%AC%AC%E4%B8%80%E4%B8%AAAPI%E8%AF%B7%E6%B1%82.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%AE%80%E5%8D%95%E7%9A%84%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90%E4%BB%BB%E5%8A%A1.md)
