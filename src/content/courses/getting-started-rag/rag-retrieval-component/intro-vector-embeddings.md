---
course: "getting-started-rag"
chapter: "rag-retrieval-component"
lesson: "intro-vector-embeddings"
sourceId: 3639
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-2-rag-retrieval-component/intro-vector-embeddings"
title: "向量嵌入简介"
description: "介绍将文本表示为数字向量（嵌入）的理念，以实现语义理解。"
order: 2
plots: ["plots/3639-0.json"]
sourceHash: "1c6bf480c68c62b7dd8e7defe4fe5064b164ebce555f5dc722bea5c432d93372"
sourceCorrections: []
---

如章节开头所述，检索器是否好用，取决于它能否理解用户查询和知识库中文档的*含义*。然而，计算机不像人类那样天生理解语言，它们处理的是数字。这种基本差异使得我们需要一种方法，将文本转换为能捕捉其语义核心的数字形式。这时，向量 (vector)嵌入 (embedding)就派上用场了。

向量嵌入是文本（可以是词、句子，甚至整个文档）在多维数学空间中的紧密数字表示。可以想象，每段文本都被映射到这个空间中的一个特定点或向量。一个向量本质上是一串数字，例如 `[0.05, -0.21, 0.98, ..., 1.52]`。“多维”表示这些向量可以有许多分量，通常是几百甚至几千个（例如，768或1024维很常见）。

这些嵌入的一个显著特性是，它们旨在捕捉语义关系。具有相似含义的文本片段，其向量在这个高维空间 (high-dimensional space)中预期会彼此“接近”。反之，含义不相似的文本，其向量会距离更远。例如，“机器学习 (machine learning)”的嵌入可能比“股票市场”的嵌入更接近“人工智能”的嵌入。

考虑一个简化的二维空间：



![Interactive chart](plots/3639-0.json)



> 一个二维可视化图，其中“狗”和“小狗”或“猫”和“小猫”等相关词语的位置比“苹果”等不相关词语更接近。实际的嵌入存在于高得多的维度中。

文本之间的相似性并非基于简单的关键词重叠，而是基于在训练专门的嵌入模型时，通过分析大量文本数据所学到的语境理解。这些模型，通常基于像Transformer这样的神经网络 (neural network)架构，学习词语在语境中的用法，并将这种理解编码成数字向量。随后将介绍具体的模型类型。

这种表示为何对RAG如此重要？当用户提交查询时，RAG系统首先将该查询转换为其向量嵌入。然后，检索器组件使用这个查询向量来搜索向量数据库（其中存储了所有文档片段的预计算嵌入）。目的是在向量空间中找到与查询嵌入*最接近*的文档片段嵌入。这种接近度通常使用数学相似性度量来衡量，例如余弦相似度，我们很快会讨论。

通过使用嵌入，检索过程超越了简单的关键词匹配。它能识别与查询相关的文档，即使这些文档没有使用完全相同的词语。这种理解语义的能力，对于为生成器大型语言模型获取真正有用的上下文 (context)非常重要，从而得到更准确和相关的最终答案。

## 参考资料

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean (2013)
  Journal: arXiv preprint arXiv:1301.3781; DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  介绍了Word2Vec，一种学习捕获语义和句法关系的密集词嵌入的基础方法，阐明了将词映射到连续空间中向量的核心思想。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS); Volume: 30; DOI: [10.5555/3295222.3295349](https://doi.org/10.5555/3295222.3295349)
  提出了Transformer架构，它已成为许多现代嵌入模型的基础，实现了上下文文本表示的学习。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Pages: 4171–4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  介绍了BERT，这是基于Transformer的预训练语言模型的一项重要进展，对于生成上下文感知的嵌入至关重要。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://doi.org/10.18653/v1/D19-1410) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982–3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  描述了Sentence-BERT，一种从预训练的类BERT模型创建语义化句子和文档嵌入的方法，并针对语义相似性搜索等任务进行了优化。
