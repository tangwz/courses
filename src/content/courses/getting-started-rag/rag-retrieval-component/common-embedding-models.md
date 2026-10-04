---
course: "getting-started-rag"
chapter: "rag-retrieval-component"
lesson: "common-embedding-models"
sourceId: 3640
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-2-rag-retrieval-component/common-embedding-models"
title: "常用嵌入模型"
description: "简要介绍用于生成文本嵌入的流行预训练模型（例如，Sentence-BERT 变体）。"
order: 3
plots: []
sourceHash: "7b05f94f56a643eb8418b589e03dd53950e8ba3aede3057c65bcc871974967a9"
sourceCorrections: []
---

现在我们已了解了向量 (vector)嵌入 (embedding)是*什么*，下一步自然就是了解它们是如何创建的。虽然理论上你可以从零开始训练自己的嵌入模型，但绝大多数应用都依赖于强大的**预训练 (pre-training)模型**。这些模型已在海量文本数据上进行过训练，使它们能够有效捕捉词语和句子之间复杂的语义关联 (semantic relationship)，从而大大节省时间和计算资源。

目标是找到那些擅长生成嵌入的模型，使得语义相似的句子在向量空间中对应的向量距离相近（例如，具有高余弦相似度，$cos(\theta)$）。这正是 RAG 中检索步骤所需要的。

### 基于 Transformer 的句子嵌入 (embedding)

当今许多最成功且应用广泛的嵌入模型都基于**Transformer 架构**，它彻底改变了自然语言处理 (NLP)。然而，直接使用 BERT 等标准 Transformer 模型的原始输出进行句子相似度任务，通常会产生不理想的结果。这些模型主要针对遮蔽语言建模等任务进行预训练 (pre-training)，不一定直接用于生成即用型、可比较的句子级嵌入。

为了解决这个问题，已发展出专门的架构和微调 (fine-tuning)策略。一个专门为生成高质量句子嵌入而设计的知名模型系列是 **Sentence-BERT (SBERT)** 及其众多变体。

#### Sentence-BERT (SBERT)

SBERT 通过使用**孪生网络**结构修改了标准 BERT 架构。在这种设置下，两个相同的预训练 Transformer 网络并行处理两个输入句子。输出（通常是池化后的句子嵌入）随后使用相似度指标进行比较。SBERT 在大量标注了相似度（例如，语义文本相似度 (STS) 基准）的句子对数据集上进行微调。此训练过程专门优化模型，使其生成的嵌入中，相似句子具有高余弦相似度分数。

基于 SBERT 的模型的主要优势包括：

1. **语义质量：** 它们生成能有效捕捉句子含义的嵌入，与简单平均标准 BERT 的词嵌入相比，在相似度搜索任务上表现更好。
2. **计算效率：** 虽然微调需要资源，但*使用*预训练的 SBERT 模型为新句子生成嵌入，在计算上比对所有句子对进行全面的交叉编码器比较快得多。

#### 流行实现与变体

`sentence-transformers` 库建立在 PyTorch 和 TensorFlow 等框架之上，提供了对各种预训练 SBERT 和其他句子嵌入模型的便捷访问。一些你将遇到的常见例子包括：

- **`all-MiniLM-L6-v2`**：一个流行且均衡的模型，性能良好，体积相对较小，推理 (inference)速度快。它是许多通用任务的良好起点。
- **`multi-qa-mpnet-base-dot-v1`**：此模型专门为语义搜索任务进行微调，特别适用于问答场景，即根据查询（问题）查找相关段落（答案）。它在非对称搜索任务（查询和文档形式不同）中通常表现良好。
- **`paraphrase-multilingual-mpnet-base-v2`**：这是一个多语言模型的例子，能够跨不同语言生成可比较的嵌入。如果你的知识库包含多语言文档，这很有价值。

这些模型通过 Hugging Face Hub 等平台易于获取，并且通常直接集成到 RAG 框架中。

### 选择嵌入 (embedding)模型

选择合适的嵌入模型是构建 RAG 系统的重要一步。请考虑以下因素：

1. **任务类型：** 你是在执行对称语义搜索（查找相似句子，例如文章聚类）还是非对称搜索（根据查询查找相关文档，RAG 中的常见情况）？有些模型更适合其中一种。
2. **性能与资源限制：** 大型模型通常提供更好的嵌入质量，但生成嵌入需要更多计算资源（内存、处理时间），并可能需要更多存储空间。请在性能需求与可用硬件和延迟要求之间取得平衡。像 `all-MiniLM-L6-v2` 这样的模型在许多用例中能取得良好平衡。
3. **特定文本类别：** 通用模型通常表现良好，但如果你的文档属于非常特定的文本类别（例如，法律文本、生物医学研究），在类似特定类别数据上微调 (fine-tuning)的模型可能会产生更好的结果，尽管这些模型市面上较不常见。
4. **语言覆盖：** 如果你的应用需要处理多种语言，请确保选择多语言模型。

通常需要进行实验。你可以从通用模型开始，并评估它在你特定任务和数据上的表现。如果检索质量不佳，你可以再尝试更专业或更大的模型。

这些预训练 (pre-training)模型为将你的文本文档和用户查询转换为有意义的向量 (vector)表示提供了支撑，这些向量表示是接下来将讨论的相似度搜索机制所必需的。它们是现代 RAG 系统中有效信息检索的重要组成部分。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  本文介绍了Transformer架构，它是BERT和SBERT等现代NLP模型的基础。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Volume: 1; Pages: 4171–4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  介绍了BERT模型，这是一种基础的预训练语言模型，后来成为SBERT等专门模型的基础。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers, Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982–3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  这篇开创性论文介绍了Sentence-BERT，它专门用于生成适用于相似性任务的高质量句子嵌入。
- [Sentence-Transformers Documentation](https://www.sbert.net/) — Nils Reimers (2024)
  Publisher: Hugging Face
  `sentence-transformers`库的官方文档，提供了实用指南、模型示例和实现细节。
