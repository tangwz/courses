---
course: "fine-tuning-small-language-model"
chapter: "model-evaluation-and-benchmarking"
lesson: "testing-prompt-generalization"
sourceId: 8524
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-6-model-evaluation-and-benchmarking/testing-prompt-generalization"
title: "测试提示词泛化能力"
description: "评估微调后的模型如何处理训练期间未见过的表述和指令。"
order: 3
plots: []
sourceHash: "083909996b1d1a9a820e6898167dfdcc521cc177b369744edb46b2d5a05cc6d0"
sourceCorrections: []
---

困惑度（Perplexity）和 ROUGE 等数值指标为我们提供了模型复现验证数据准确度的基础参考。然而，实际用户很少会完全按照训练数据集中的示例来组织请求。测试提示词 (prompt)泛化能力是为了衡量微调 (fine-tuning)后的模型在处理陌生的措辞、不同的结构以及训练阶段未明确出现的指令时的实际表现。

如果您使用严格的模板微调了一个模型，用于从会议记录中提取行动事项，那么当提示词完全匹配“提取行动事项：”这一字符串时，它的表现可能非常出色。但如果同样是将问题改为“根据这些笔记，我们接下来要做什么？”，模型就彻底失效了，这说明微调过程产生了一种脆弱的表达方式。模型只是死记硬背了模板，而没有学会背后的任务逻辑。衡量这种灵活性是评估语言模型的一个组成部分。

为了有条理地测试泛化能力，您可以构建一个评估集，其中包含针对同一基本任务的多组变体。您可以从数据集中的原始提示词结构开始，生成多个改写版本。

> 通过将不同的提示词结构输入微调模型并比较输出的一致性，来测试提示词泛化能力的工作流程。

在构建这些变体时，有三种常用的技术：

1. **指令改写**：使用同义词和不同的句子结构重写系统提示词或主要指令。将正式的请求改为口语化的提问。
2. **上下文 (context)扰动**：改变提供给模型的上下文。调整段落顺序，引入微小的拼写错误，或者在提示词中加入完全无关的句子，观察模型是否仍能聚焦于正确的信息。
3. **格式切换**：要求模型以训练数据中不常见的格式输出。如果您的模型是训练输出 JSON 的，试着要求它输出 HTML 表格。这可以测试模型是否能将任务逻辑与输出格式解耦。

评估这些测试结果需要从精确匹配指标转向语义指标。像 ROUGE 这样的标准指标对特定词汇的选择非常敏感，如果输出格式发生细微变化，得分就会下降。相反，您可以使用外部嵌入 (embedding)模型（如预训练 (pre-training)的 sentence transformer），将生成的文本转换为稠密向量 (vector)表示。

获得向量嵌入后，您可以计算原始提示词生成的输出与变体提示词生成的输出之间的余弦相似度。如果 $\mathbf{u}$ 代表原始输出的嵌入向量，$\mathbf{v}$ 代表变体输出的嵌入向量，则余弦相似度计算如下：

$\text{余弦相似度} (\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$

较高的余弦相似度得分表明，无论提示词如何措辞，输出的语义含义都保持一致。这说明模型泛化良好。如果提示词变化时相似度得分大幅下降，则说明模型对输入格式过于敏感。

您可以通过编写 Python 评估循环来自动化此过程。加载您的微调模型和另一个独立的 sentence transformer 模型。遍历包含同义提示词列表的 JSON 文件。为列表中的每个提示词生成响应，将响应转换为嵌入向量，并计算每组相似度得分的方差。

如果自动化评估显示泛化能力较差，这通常直接指向数据集准备或训练参数 (parameter)中的问题。一个只对单一特定提示词格式有响应的模型，很可能在训练时对该字符串产生了过拟合 (overfitting)。观察到这些失败情况后，自然会进入下一个评估阶段，即通过将这些输出与原始基座模型的能力进行对比，来检测过拟合和灾难性遗忘的迹象。

## 参考资料

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers, Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982–3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  介绍了用于计算文本输出之间语义相似度的 Sentence Transformers 框架。
- [BERTScore: Evaluating Text Generation with BERT](https://openreview.net/forum?id=SkeHuCVFDr) — Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi (2020)
  Journal: International Conference on Learning Representations (ICLR)
  提出了一种利用预训练嵌入来评估生成文本质量的指标，为 ROUGE 等词汇指标提供了替代方案。
- [PromptBench: A Unified Library for Robustness Evaluation of Language Models](https://arxiv.org/abs/2306.04528) — Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, Xing Xie (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2306.04528](https://doi.org/10.48550/arXiv.2306.04528)
  一项关于语言模型如何应对提示词扰动和对抗性攻击的综合研究及软件库。
