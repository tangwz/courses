---
course: "getting-started-local-llms"
chapter: "basic-interaction-prompting"
lesson: "what-is-a-prompt"
sourceId: 4275
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-5-basic-interaction-prompting/what-is-a-prompt"
title: "什么是提示？"
description: "说明什么是提示及其在引导大型语言模型回应中的作用。"
order: 1
plots: []
sourceHash: "758c4e36602467e64e07590baa34afa8ad75ba5a35c59abbce4c99131766b612"
sourceCorrections: []
---

把大型语言模型想象成一个功能强大的文本生成器。它对语言、语法、事实甚至不同的写作风格都非常熟悉，但它不知道你想要它做什么，直到你给出指令。这时，提示就派上用场了。

\*\*提示就是你提供给大型语言模型（LLM）的文本输入，用于让它开始工作。\*\*它可以是指令、问题、待补充的文本片段，或者你希望模型撰写的主题。这是你与模型对话的那一方。

可以这样打个比方：

- 当你使用搜索引擎时，你输入关键词或问题。那就是你的提示。
- 当你使用计算器时，你输入数字和运算（$2 + 2$）。那就是你的提示。
- 当你与数字助理对话时，你说的短语（“嘿，助理，天气怎么样？”）就是你的提示。

对于大型语言模型，提示的作用相似，但通常更灵活、更具开放性。它引导模型进行文本生成。大型语言模型接收你的提示，根据其训练数据进行处理，然后生成相关的后续内容或答案。

**输入（提示） -> 大型语言模型处理 -> 输出（回应）**

例如，如果你提供提示：

```text
解释在本地运行大型语言模型的主要优势。
```

模型会理解你正在寻求关于特定议题的解释。它会根据其掌握的知识生成文本，以直接回答你的问题。

你的提示的质量和清晰度明显影响大型语言模型回应的质量和关联性。一个结构清晰的提示会清楚地说明任务或问题，使模型更容易生成有益的输出。反之，一个模糊或模棱两可的提示可能导致意想不到或没有帮助的结果。

提示的类型可以从非常简单到相当复杂：

- **提问：** `法国的首都是什么？`
- **指令：** `总结以下文本：`，后接文本内容。
- **续写开头：** `很久很久以前，在一个遥远的地方，住着一个...`
- **创意文本请求：** `写一首关于机器人学画画的短诗。`

在本课程中，学习如何编写有效的提示是你与在计算机上运行的本地大型语言模型进行沟通并掌控其表现的主要途径。接下来的部分将向你展示如何针对不同任务来组织这些提示。

## 参考资料

- [Introduction to prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2023)
  Publisher: OpenAI
  官方指南，解释了提示工程的概念，从提示的基本定义开始。
- [Guide to prompt design](https://ai.google.dev/gemini-api/docs/prompting-intro) — Google for Developers (2024)
  谷歌官方文档，概述了提示设计，包括提示在与大型语言模型交互中的作用。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  这项全面的综述对大型语言模型进行了广泛概述，包括其交互机制，如提示和上下文学习。它确立了提示如何使用的背景。
