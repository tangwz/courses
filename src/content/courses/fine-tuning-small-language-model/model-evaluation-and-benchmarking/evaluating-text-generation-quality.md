---
course: "fine-tuning-small-language-model"
chapter: "model-evaluation-and-benchmarking"
lesson: "evaluating-text-generation-quality"
sourceId: 8522
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-6-model-evaluation-and-benchmarking/evaluating-text-generation-quality"
title: "评估文本生成质量"
description: "对模型输出进行定性检查，以确定其对预期任务的执行情况。"
order: 1
plots: []
sourceHash: "08d3e7fa90618fe31e555b0930dd088fec6147efd53b9b94052aaf5359b1b94c"
sourceCorrections: []
---

通过监控损失曲线和保存检查点来进行数学优化，可以确认模型正在学习预测下一个标记 (token)，但这并不能保证针对特定任务生成高质量的文本。虽然损失值下降是一个正面指标，但它无法衡量连贯性、语气或对格式规范的遵守情况。在计算自动化指标之前，必须进行定性评估，以直接检查模型的输出结果。

单靠数学指标可能会产生误导。模型可能会通过记忆常见的标记序列而获得较低的损失值，但并没有学会指令背后的结构。通过阅读生成的文本，你可以发现逻辑不一致、重复循环以及公式经常忽略的结构性问题。为了有效地做到这一点，你需要一种结构化的方法，而不是在提示词 (prompt)中输入随机查询。

创建一个专门用于人工审查的目标验证集。该集合应包含 20 到 50 个提示词，分为三类。首先，包含与训练数据结构和主题完全匹配的标准案例。其次，包含使用意外表述或要求异常复杂响应的边缘案例。第三，包含与训练数据完全无关的分布外案例。测试分布外提示词有助于确定微调 (fine-tuning)过程是否破坏了模型的通用知识。

> 比较基础模型和微调模型输出的并排定性评估工作流。

在评估生成质量时，传递给推理 (inference)引擎的参数 (parameter)会对输出产生很大影响。如果输出效果很差，问题可能在于生成设置，而不是模型权重 (weight)。温度（Temperature）参数在应用 softmax 函数之前对 logits 进行缩放。较低的温度（如 $T = 0.1$）会使模型更加确定，并对其首选选项更有信心。对于 JSON 生成或数据提取等结构化任务，强烈建议使用这种设置。较高的温度（如 $T = 0.7$）则会鼓励对话任务使用更多样化的词汇。你还应该配置 top-p 采样，它将模型限制在累积概率超过值 $p$ 的动态标记池中进行选择。

在检查过程中，应主动寻找微调后的小语言模型中常见的特定失效模式。第一个问题是格式失败。检查模型是否在应停止时停止。如果模型在回答提示词后继续喋喋不休，很可能是因为它在训练期间没能学会结束标记（end-of-sequence token）。小语言模型对提示词模板非常敏感，数据集中缺失填充或序列标记通常会导致这种行为。

第二个问题是幻觉 (hallucination)。在高度结构化的数据集上进行微调有时会让模型优先考虑格式而非事实准确性。它学会了正确答案的形态，并会自信地插入虚假信息来填充该形态。第三个问题是模式崩塌。如果模型对完全不同的提示词返回完全相同的模板或答案，则学习率可能过高，或者模型训练的轮次过多，导致其过拟合 (overfitting)到单一模式中。

为了高效地执行此评估，可以编写一个 Python 脚本，并排生成原始基础模型和新微调适配器的响应。通过使用相同的生成参数对完全相同的提示词进行处理，你可以隔离出训练数据带来的影响。你可以观察微调过程究竟如何改变了模型的行为，确保这些变化在进入自动化基准测试算法之前符合你的项目目标。

## 参考资料

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1904.09751](https://doi.org/10.48550/arXiv.1904.09751)
  介绍了核采样（Top-p sampling），并解释了为什么基于最大概率的解码方式经常导致文本重复或质量低下。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, and Yiannis Pappas et al. (2022)
  Journal: arXiv preprint arXiv:2211.09110; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  一个全面评估模型的框架，涵盖了准确性、校准和公平性等多个指标。
- [Text Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies) — Hugging Face (2024)
  Publisher: Hugging Face
  技术文档，解释了在常见推理引擎中 temperature 和 Top-p 等参数的实现方式及其影响。
- [Survey of Hallucination in Natural Language Generation](https://dl.acm.org/doi/10.1145/3571730) — Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung (2023)
  Journal: ACM Computing Surveys; Volume: 55; Pages: 1-38; DOI: [10.1145/3571730](https://doi.org/10.1145/3571730)
  对生成文本中的不同幻觉类型和失败模式进行了分类，为定性评审提供了专业术语。
