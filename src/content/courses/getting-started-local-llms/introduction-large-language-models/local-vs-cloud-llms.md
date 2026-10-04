---
course: "getting-started-local-llms"
chapter: "introduction-large-language-models"
lesson: "local-vs-cloud-llms"
sourceId: 4205
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-1-introduction-large-language-models/local-vs-cloud-llms"
title: "本地LLM与云端LLM"
description: "了解在本地运行大语言模型与使用云端API之间的主要差异和权衡。"
order: 5
plots: ["plots/4205-0.json"]
sourceHash: "276df6f59552f95b34a7cb2340f2c06654d2c04a6afec4fc586fd749020cc448"
sourceCorrections: []
---

既然您已了解大语言模型 (LLM)的基本原理以及在本地运行它们的一些吸引人之处，接下来我们直接比较与这些功能强大的工具交互的两种主要方式：使用云端服务和在您自己的本地计算机上运行它们。每种方式都有各自的特点、优势和劣势。理解这些差异将帮助您决定哪种方式或哪种方式组合适合您的需求。

### 云端大语言模型 (LLM)：远程使用计算资源

云端大语言模型是由公司托管并维护在强大服务器基础设施上的模型。可以想象一下OpenAI的ChatGPT（由GPT模型驱动）、Google的Gemini或Anthropic的Claude等服务。您通常通过网页界面与这些模型交互，对于开发者来说，更常见的方式是通过应用程序编程接口（API）。API充当协议或中介，允许您的应用程序向云服务发送请求（您的提示），并通过互联网接收响应（大语言模型生成的文本）。

**云端大语言模型的优势：**

- **可使用先进模型：** 云服务提供商通常提供对可用最大、能力最强模型的访问，这些模型在普通消费级硬件上是无法运行的。
- **无硬件负担：** 您无需功能强大的计算机。所有繁重计算都在提供商的服务器上进行。通常，一台具备互联网接入的简单设备就足够了。
- **易于上手：** 通常，开始使用非常简单，只需注册账户并获取API密钥。技术设置要求最低。
- **可伸缩性：** 云服务旨在处理大量请求，并能根据需求自动扩展。

**云端大语言模型的劣势：**

- **费用：** 大多数云端大语言模型服务采用按使用量付费模式，通常根据处理的令牌数量（包括输入和输出）计费。这可能变得昂贵，特别是对于频繁或大规模使用而言。
- **隐私顾虑：** 当您使用云API时，您的提示和生成的数据会发送到提供商的服务器。尽管公司有隐私政策，但您的数据会脱离您的直接控制，这对于敏感信息可能无法接受。
- **依赖互联网：** 您需要稳定的互联网连接才能使用云端大语言模型。无法离线使用。
- **控制受限：** 与在本地运行模型相比，您对模型的特定参数 (parameter)或行为的控制通常较少。自定义选项可能受到限制。
- **审查和限制：** 云服务提供商可能会根据其政策实施内容过滤器或使用限制。

### 本地大语言模型 (LLM)：在您的机器上运行模型

正如我们所讨论的，本地大语言模型是您可以直接下载并在您自己的计算机硬件（笔记本电脑或台式机）上运行的模型。您可以使用管理模型加载和处理的特定软件工具与它们交互。

**本地大语言模型的优势：**

- **隐私和安全：** 您的数据永不离开您的计算机。提示和生成的文本完全保持私密，这对于保密工作或个人信息来说是一个重要益处。
- **使用成本效益：** 一旦您拥有合适的硬件，运行这些模型通常是免费的。使用模型本身没有按令牌计费或订阅费用（尽管某些模型在商业使用时可能需要许可费用）。
- **离线能力：** 您可以随时随地使用您的大语言模型，无需互联网连接。
- **控制和自定义：** 您对模型的选择、配置（如温度设置，稍后讨论）以及如何将其整合到您的工作流程中拥有更大的控制权。
- **不受外部限制：** 您不受第三方提供商内容政策或潜在审查的约束。

**本地大语言模型的劣势：**

- **硬件要求：** 在本地运行大语言模型需要大量的计算机资源，主要是RAM（内存），通常还需要一块性能良好的GPU（显卡）和足够的VRAM（显存 (VRAM)）以获得可接受的性能。我们将在下一章详细讨论这一点。
- **设置复杂性：** 开始使用需要更多精力，包括安装软件、下载大型模型文件以及可能进行的配置设置。
- **性能限制：** 大语言模型生成文本的速度完全取决于您的硬件。在低端机器上，它可能比顶级云服务慢。
- **模型可用性：** 尽管有许多优秀的开源模型可用，但绝对最大和最前沿的模型可能对本地硬件要求过高，或者未以可运行格式公开发布。

### 简要对比：本地与云端

本地大语言模型 (LLM)和云端大语言模型之间的选择通常涉及权衡。以下是主要因素的比较：



![本地与云端大语言模型对比](plots/4205-0.json)



> 这些因素的评分标准为5分代表高/好，1分代表低/差，均从用户角度衡量。对于“硬件需求”，5分表示用户硬件要求低，而1分表示用户硬件要求高。对于“费用（使用）”，5分表示持续成本低，而2分表示潜在持续成本高。

### 您应选择哪种方式？

- **如果满足以下条件，请选择云端：**
  - 您需要绝对最高的性能和能力。
  - 您没有强大的硬件，或者不想管理设置。
  - 您的数据隐私要求允许第三方处理。
  - 您的使用量下成本可控，或者您只是简单尝试。
- **如果满足以下条件，请选择本地：**
  - 数据隐私和安全是您的首要考虑。
  - 您想避免持续的使用费用。
  - 您需要离线访问。
  - 您喜欢调试，并希望对模型及其配置拥有最大控制权。
  - 您拥有或愿意投入购买合适的硬件。

许多人发现两种方式结合使用的价值。他们可能会将云服务用于需要最大计算能力的任务，将本地模型用于对隐私敏感的任务、离线工作或试验。

本课程专门侧重于**本地**方式。我们将指导您了解所需的硬件、设置软件、查找模型以及在您的机器上直接与它们交互。接下来，我们开始准备您的计算机。

## 参考资料

- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint arXiv:2303.18223; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  一篇广泛的学术综述，讨论了LLM的现状，包括其能力、架构、应用和操作挑战，这为部署决策提供了背景信息。
- [Generative AI with Transformers: Build your own Large Language Model (LLM)](https://www.packtpub.com/product/generative-ai-with-transformers-build-your-own-large-language-model-llm/9781803230613) — Ben Auffarth (2023)
  Publisher: Packt Publishing
  一本实践指南，介绍了LLM的原理，并提供了构建和部署LLM的指导，包括针对本地与云端不同环境的考量。
- [OpenAI API Reference](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFcxA6f2L26fvXGp4l-rcNTTcLSnL2uEaMbvWArX6waa-l9YwY9AUvQw2sTzRbY1sdefuu9KAXG45heSLjWntzR2xIj__b_pW9XRQsx7R4WoGxg8hNUv6P1_ges2sjOZiRyNA8ixQM1DP99gvQnkyaa99Qh28xc) — OpenAI (2024)
  Publisher: OpenAI
  与云端LLM交互的官方文档，展示了本节讨论的API驱动的访问模型和核心功能。
- [🤗 Transformers Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  Publisher: Hugging Face
  运行和试验本地开源LLM的主要资源，提供了模型加载、推理和硬件兼容性的信息。
