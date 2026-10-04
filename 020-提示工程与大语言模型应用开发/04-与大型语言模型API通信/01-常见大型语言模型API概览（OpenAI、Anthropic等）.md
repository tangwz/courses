# 常见大型语言模型API概览（OpenAI、Anthropic等）

来源：[原文](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-4-interacting-with-llm-apis/overview-common-llm-apis)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将大型语言模型 (LLM) 集成到软件应用程序中，是利用其生成能力的基础过程。这种集成主要通过与各类机构提供的应用程序编程接口 (API) 通信来实现，这些机构运行并提供这些高效模型。API充当连接器，您的应用可借此向大型语言模型服务发送提示及其他指令，并按程序接收生成内容。

许多供应商通过API提供先进的大型语言模型访问。尽管具体细节有所不同，但核心思想保持一致：您发送包含提示和配置参数 (parameter)的请求，供应商的服务会使用其大型语言模型进行处理，然后给出结果。

以下是一些知名的供应商及其产品：

### 主要大型语言模型API供应商

- **OpenAI:** 或许是知名度很高的供应商，OpenAI提供API以使用GPT-4、GPT-4o和GPT-3.5-Turbo等模型。他们的API文档完善且得到广泛采纳，这使其成为许多开发者的常用起点。他们通常使用“聊天补全”格式，交流内容按角色（系统、用户、助手）以消息序列组织。
- **Anthropic:** Anthropic为其Claude系列模型（例如Claude 3 Opus、Sonnet和Haiku）提供API。他们非常注重AI安全性和实用性，通常依据“宪法”中阐明的原则构建模型。其API结构在思路方面与OpenAI相似，但有自己的特定请求和响应格式。
- **Google:** Google通过Google Cloud的Vertex AI平台和Google AI Studio提供使用其Gemini系列模型（例如Gemini Pro、Gemini 1.5 Pro）。与更广阔的Google Cloud生态系统集成，对于已在该环境中运行的应用来说，是一大优势。Vertex AI提供一套MLOps工具，有助于模型和部署的管理。
- **Cohere:** Cohere为其Command模型提供API，通常面向企业应用。除文本生成外，Cohere的平台常包含用于文本嵌入 (embedding)和分类等任务的专用端点，这表明其侧重于信息获取和业务流程自动化。
- **其他平台与开源模型：** 除这些主要商业供应商外，还有一些平台提供API，可使用种类更丰富的模型，包括流行的开源替代品（如Llama、Mistral）。
  - **Hugging Face:** 提供推理 (inference)端点，可从其模型库中运行和提供许多模型。
  - **Amazon Foundation:** AWS上的一项托管服务，提供API以使用来自不同供应商的基础模型（包括Anthropic、Cohere、Meta和亚马逊自家的Titan模型）。
  - **Together AI / Anyscale / Fireworks AI:** 这些平台提供优化的推理服务，通常注重流行开源模型的速度和成本效益，这些模型可通过与OpenAI标准兼容的API使用。

### 常见API功能

尽管供应商和模型各异，大多数大型语言模型API都通过特定端点提供一套核心功能：

1. **模型选择：** 指明您想使用哪种大型语言模型（例如 `gpt-4o`、`claude-3-sonnet-20240229`、`gemini-1.5-pro-latest`）。
2. **提示输入：** 提供实际提示文本，通常以结构化格式（如消息列表）。
3. **参数 (parameter)配置：** 设置参数以控制生成过程，例如 `temperature`（随机性）、`max_tokens`（输出长度限制）、`top_p` 等。（本章稍后将详细说明这些参数）。
4. **请求提交：** 向供应商的API端点发送结构化请求（通常以JSON格式通过HTTPS）。
5. **响应处理：** 接收响应（通常为JSON格式），其中包含大型语言模型生成的文本，以及使用信息或潜在安全标记 (token)等元数据。

> 应用与大型语言模型供应商API之间的基本通信流程。

### 选择供应商

选择API供应商通常取决于您项目需求的一些特定因素：

- **模型能力：** 供应商提供的模型是否擅长您的应用所要求的特定任务（例如，编程、创意写作、复杂推理 (inference)、多语言能力）？
- **性能与成本：** 延迟特点和定价模式（通常按token输入/输出计费）如何？不同模型和供应商之间，成本可能有很大差别。
- **API设计与文档：** API使用起来是否直观易懂？文档是否清晰和完整？
- **安全与可靠性：** 有哪些安全功能（例如，内容过滤）可用？供应商的正常运行时间和可靠性保证如何？
- **工具与生态：** 供应商是否提供额外工具（如SDK、评估框架），或者与您现有云基础设施的集成是否良好？

理解API供应商的这方面内容，是按程序控制大型语言模型的初步工作。后续部分将带您了解使用Python进行认证、发出请求和处理响应的实际操作方法。

## 参考资料

- [OpenAI API Documentation](https://platform.openai.com/docs) — OpenAI (2024)
  Publisher: OpenAI
  提供与OpenAI模型（包括GPT-4o和GPT-3.5-Turbo）交互的全面指南和API参考，涵盖聊天补全和参数配置。
- [Anthropic API Documentation](https://docs.anthropic.com/en/) — Anthropic (2024)
  Publisher: Anthropic
  Anthropic Claude模型访问的官方文档，详细说明其特定的API格式和实现安全AI交互的最佳实践。
- [Gemini API (Google Cloud Vertex AI) Documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini) — Google Cloud (2024)
  Publisher: Google Cloud
  提供通过Vertex AI平台集成Google Gemini模型的信息，包括API使用和Google Cloud生态系统内的功能。
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html) — Amazon Web Services (AWS) (2024)
  Amazon Bedrock的官方指南，该托管服务提供对不同提供商基础模型的API访问，着重于模型部署和管理。

---

[上一节](../03-%E6%8F%90%E7%A4%BA%E8%AF%8D%E8%AE%BE%E8%AE%A1%E3%80%81%E8%BF%AD%E4%BB%A3%E4%B8%8E%E8%AF%84%E4%BC%B0/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%8F%90%E7%A4%BA%E8%AF%8D%E4%BC%98%E5%8C%96%E6%8C%91%E6%88%98.md) · [下一节](02-API%20%E8%AE%A4%E8%AF%81%E4%B8%8E%E5%AE%89%E5%85%A8.md)
