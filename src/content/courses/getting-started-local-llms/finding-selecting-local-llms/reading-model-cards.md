---
course: "getting-started-local-llms"
chapter: "finding-selecting-local-llms"
lesson: "reading-model-cards"
sourceId: 4244
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-3-finding-selecting-local-llms/reading-model-cards"
title: "了解模型卡片中的信息"
description: "了解模型卡片中提供的信息，例如预期用途、局限性及要求。"
order: 5
plots: []
sourceHash: "feca0f8b076f77f04a489d15fd299c3be01bf6ed312bc734d35e45f63825e394"
sourceCorrections: []
---

在使用大型语言模型时，在Hugging Face等平台上查找模型通常是第一步。了解模型大小、GGUF等格式以及量化 (quantization)作用等技术方面也很有帮助。然而，仅仅下载遇到的第一个模型并非总是最佳做法。评估具体模型是确保它们适合特定需求、与硬件兼容并允许用于预期目的的必要步骤。这就是**模型卡片**发挥作用的地方。

可以把模型卡片看作LLM的官方文档或数据表。它是模型创建者提供模型特性、功能、局限性及预期用途等重要信息的标准化方式。研究人员将这种思路规范化，以提高透明度和负责任的AI实践。在下载或使用模型前阅读模型卡片，是做出明智选择的必要一步。

### 为什么要仔细阅读模型卡片？

花几分钟时间查看模型卡片，可以节省你大量时间，并避免之后可能出现的问题。以下是它们有价值的原因：

1. **了解预期用途：** 创建者通常会说明模型的用途（例如通用对话、代码生成、摘要）以及它擅长的地方。这能帮助你根据任务找到合适的模型。
2. **确认局限性与风险：** 模型并非完美无缺。卡片通常会列出已知弱点、潜在偏见，或模型可能产生不准确、无意义甚至有害输出的情况。了解这些有助于你设定实际期望并谨慎使用模型。
3. **检查要求：** 尽管我们已经讨论了通用硬件需求，但模型卡片可能会说明特定版本（如不同的量化 (quantization)级别）所需的最小RAM或VRAM，或者提到必要的软件库。
4. **核实许可：** 如同下一节所说，许可规定了你可以如何合法使用模型（例如个人测试、研究或商业产品）。模型卡片是找到此信息的主要途径。
5. **评估性能（相对）：** 卡片通常包含标准基准测试或评估的结果。尽管基准分数不能说明一切，但它们可以提供不同模型之间的粗略比较依据，尤其是在类似条件下进行评估时。
6. **鼓励负责任的使用：** 通过了解模型的背景、训练数据（简要提及）和潜在的伦理问题，你可以更负责任地使用LLM。

### 模型卡片中常见的信息

尽管结构可能略有不同，但模型卡片，尤其是在Hugging Face这样的平台上的，通常包含几个标准部分。让我们看看会看到什么：

- **模型细节：** 此部分通常提供核心技术信息。你会找到关于模型架构（例如Llama 2, Mistral）、参数 (parameter)数量（如7B, 13B）、可用的特定文件格式（例如GGUF, Safetensors）的详细信息，以及通常关于任何已应用量化 (quantization)（如Q4\_K\_M）的详细信息。这直接与之前讨论的硬件要求和性能特征有关。
- **预期用途：** 在这里，创建者描述了模型的主要目的。它是用于聊天？编写代码？翻译语言？还是遵循复杂指令？此部分帮助你确定模型的设计是否与你的目标一致。它还可能提及模型表现良好的范围（例如创意写作、技术问答）。
- **局限性与超出范围的用途：** 这是一个非常重要的部分。它概述了模型*不*是为什么设计的，或已知其表现不佳的地方。例子包括可靠地生成事实陈述、进行复杂的数学推理 (inference)或避免有害刻板印象。务必仔细阅读此部分，以避免误用模型或对其输出感到失望。
- **训练数据：** 通常，此部分会概要介绍用于训练模型的数据（例如，“来自互联网的大量文本和代码语料库”）。训练数据的类型极大地影响模型的知识、能力和固有偏见。尽管专有模型的详细信息通常很少，但开源模型可能会提供更多信息。
  "\* **评估结果：** 模型创建者经常报告在标准学术基准测试（例如，用于通用知识的MMLU，用于常识推理的HellaSwag，用于编码的HumanEval）上的表现。这些分数提供了一种量化比较模型的方法，尽管性能可能有所不同。不必过于纠结于每个基准测试的细节；如果比较类似模型，请关注相对分数。"
- **伦理考量与偏见：** 此部分涉及从训练数据中学到的潜在偏见，并讨论公平性或安全性问题。它可能会描述创建者采用的缓解策略。尽管有时简短，但它的存在表明了对技术更广泛影响的认识。
- **如何使用/使用示例：** 通常，你会找到实用的说明或代码片段，展示如何使用流行的库（如Python用户使用的`transformers`）或工具（如Ollama的命令或LM Studio的设置）加载和运行模型。当你准备开始试验时，这会非常有用。
- **许可：** 模型许可的明确声明（例如Apache 2.0、MIT、Llama 2社区许可）。这对于理解你的使用权利和义务非常必要。我们将在下一节更详细地说明许可。

> 模型卡片中常见的部分，突出显示了各部分提供的信息类型。

### 查找模型卡片

在Hugging Face上，模型卡片通常是模型仓库页面上显示的主要内容（即仓库中的`README.md`文件）。通常，当你进入某个模型页面时，它是你首先看到的内容。请寻找“Model Card”、“Model Details”、“Intended Use”等标题。

### 做出明智的选择

假设你正在寻找一个适用于16GB内存笔记本电脑的通用聊天小型模型。你在Hugging Face上找到了一个看起来不错的模型。通过阅读其模型卡片，你可能会了解到：

- **细节：** 这是一个70亿参数 (parameter)模型，提供GGUF格式，推荐使用Q4\_K\_M量化 (quantization)级别，适用于至少有8GB模型可用内存的系统。（很好，符合你的硬件！）
- **预期用途：** 设计用于会话式AI、指令遵循和摘要。（符合你的目标！）
- **局限性：** 已知在处理复杂数学问题时表现不佳，有时可能生成重复文本。不建议用于安全关键型应用。（了解这些很好，有助于管理预期。）
- **许可：** Apache 2.0。（允许广泛使用，包括个人和商业用途，需注明出处。）

根据这些信息，你可以自信地判断这个模型是否是你需求的良好起点。如果没有阅读卡片，你可能下载了一个对你的系统来说过大的模型，或者当你想要聊天时，却下载了一个主要用于编码任务的模型。

总之，将模型卡片视为必要阅读内容。它提供背景信息，设定预期，并引导你选择一个与你的硬件、目标以及其许可定义的允许用途相符的模型。花时间了解这些信息，是在运行你的第一个本地LLM之前的一个重要步骤。

## 参考资料

- [Model Cards for Model Reporting](https://doi.org/10.1145/3287560.3287596) — Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Timnit Gebru (2019)
  Journal: Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*); Publisher: Association for Computing Machinery; Pages: 220-229; DOI: [10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596)
  提出了模型卡的概念，作为一种标准化的报告框架，旨在促进人工智能开发的透明度和责任。
- [Model cards](https://huggingface.co/docs/hub/model-cards) — Hugging Face (2024)
  官方文档解释了Hugging Face Hub上的模型卡是什么，以及如何创建和解读它们。
- [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research (TMLR); DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提出了一个用于全面评估语言模型的框架，涵盖了广泛的指标，为模型卡中提及的简单基准分数提供了更深入的见解。
