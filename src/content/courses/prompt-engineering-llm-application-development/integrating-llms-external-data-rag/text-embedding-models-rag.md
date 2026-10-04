---
course: "prompt-engineering-llm-application-development"
chapter: "integrating-llms-external-data-rag"
lesson: "text-embedding-models-rag"
sourceId: 3061
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-6-integrating-llms-external-data-rag/text-embedding-models-rag"
title: "文本嵌入模型"
description: "了解文本嵌入模型如何表示文本以进行语义搜索。"
order: 4
plots: []
sourceHash: "b7864c7c82ffdcc6c80ef560477ae41eec463c1b07e90c5b24f1bea403676060"
sourceCorrections: []
---

在前面一步将文档分割成可管理的片段后，我们面临一个新问题：如何高效地在成千上万甚至数百万个文本片段中查找与用户查询最相关的那些？直接比较原始文本字符串效率低下，并且经常错过语义关联 (semantic relationship)（例如，“狗品种信息”与“关于金毛猎犬的事实”）。我们需要一种方法来以数值方式表示文本的*含义*。这就是文本嵌入 (embedding)模型发挥作用的地方。

### 理解文本嵌入 (embedding)

文本嵌入模型是专门的神经网络 (neural network)，用于将文本片段（词语、句子、段落或整个文档）转换为称为向量 (vector)的数值表示。可以将这些向量看作是在高维“含义空间”中定位文本的坐标。

有效文本嵌入背后的核心思想是：**语义相似性对应于空间邻近性**。含义相似的文本将被映射到该向量空间中彼此靠近的向量，而含义不同的文本则会相距更远。这种数值表示使得我们能够通过计算进行有意义的比较和搜索。

例如，像“流感的症状是什么？”和“我怎么知道自己是否得了流感？”这样的句子很可能产生彼此非常接近的嵌入向量。相比之下，像“法国的首都是哪里？”这样的句子会得到一个与流感相关句子相距很远的向量。

> 一个简化的二维高维嵌入空间表示。像“狗”和“小狗”这样相似的事物被映射到一起很近，而不相关的“汽车”则相距很远。

### 嵌入 (embedding)如何生成

文本的数值表示，即向量 (vector)表示，并非随意创建。它们是复杂深度学习 (deep learning)模型的产物，通常基于Transformer架构（该架构也驱动了许多大型语言模型，如GPT和BERT）。这些嵌入模型在海量文本数据上进行训练，学习捕捉词语和事物之间的上下文 (context)细节、语义和关系。

生成嵌入的常见方式包括：

1. **基于API的服务：** 像OpenAI（例如，`text-embedding-ada-002`、`text-embedding-3-small`、`text-embedding-3-large`）、Cohere和Google等服务商通过API调用提供嵌入生成服务。这简化了集成，但会涉及网络延迟和每次API调用可能产生的费用。
2. **开源模型：** 像`sentence-transformers`（基于Hugging Face的`transformers`构建）这样的库提供了大量预训练 (pre-training)模型的访问（例如，`all-MiniLM-L6-v2`、`multi-qa-mpnet-base-dot-v1`），你可以在本地或自己的基础设施上运行这些模型。这提供了更多控制，对于大量数据而言具有成本效益，但需要管理模型和计算资源。

### 选择嵌入 (embedding)模型

嵌入模型的选择会很大程度上影响RAG系统的性能。考虑的因素有：

- **性能：** 模型在您的特定应用场景或任务中捕捉语义相似性的效果如何？有些模型擅长处理短句，有些擅长处理较长的文档，还有一些为问答等特定任务进行了调优。通常需要进行基准测试和在您的数据上进行实证测试。
- **维度：** 嵌入是具有特定维度数量的向量 (vector)（例如，768、1024、1536，甚至更高）。更高的维度可能捕捉更多信息，但会导致更大的存储需求和更慢的相似性计算。
- **计算成本与速度：** API调用有相关成本和延迟。在本地运行模型需要适当的硬件（可能需要GPU以提高速度），并会带来维护开销。
- **上下文 (context)长度：** 嵌入模型对它们一次可以处理的文本量有限制。请确保模型的上下文长度适用于您的文档片段大小。

### 衡量相似性：余弦相似性

一旦我们将文本表示为向量 (vector)，如何在这一高维空间 (high-dimensional space)中衡量“接近度”呢？最常用的指标是**余弦相似性**。余弦相似性不是测量向量端点之间的欧几里得距离（这对向量大小敏感），而是测量两个向量之间夹角的余弦。它能很好地说明这些向量是否指向同一方向。

对于两个向量 $\mathbf{A}$ 和 $\mathbf{B}$，余弦相似性计算公式如下：


$$
\text{相似性} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
$$


其中：

- $\mathbf{A} \cdot \mathbf{B}$ 是向量 $\mathbf{A}$ 和 $\mathbf{B}$ 的点积。
- $\|\mathbf{A}\|$ 和 $\|\mathbf{B}\|$ 是向量 $\mathbf{A}$ 和 $\mathbf{B}$ 的大小（或L2范数）。
- $n$ 是向量的维度数量。

结果范围从-1到1：

- **1：** 向量指向完全相同的方向（最大相似性）。
- **0：** 向量正交，通常表示不相关。
- **-1：** 向量指向相反方向（最大不相似性）。

实际中，对于许多常见模型生成的嵌入 (embedding)，其值通常介于0和1之间，因为模型常被训练为在此范围内表示语义相似性。

### 嵌入 (embedding)在RAG中的作用

文本嵌入是RAG中“检索”步骤的核心。该过程如下：

1. **索引：** 从您的外部数据源生成的所有文档片段都由嵌入模型处理。每个片段被转换为一个向量 (vector)嵌入。
2. **存储：** 这些嵌入（及其原始文本片段的引用）存储在专为高效向量搜索设计的专业数据库中（将在下一节介绍）。
3. **查询：** 当用户提交查询时，该查询文本也由*相同的*嵌入模型处理，以生成查询向量。
4. **搜索：** 系统计算查询向量与所有已存储的文档片段向量之间的相似性（通常是余弦相似性）。
5. **检索：** 识别出与查询向量相似性得分最高的向量（及其对应的文本片段），作为最相关的信息。

这些检索到的信息随后用于“增强”发送给大型语言模型的提示，为其提供所需上下文 (context)，以便根据外部数据回答用户查询。

现在我们了解了如何将文本片段表示为可搜索的向量，我们需要一种有效的方法来存储这些向量并快速执行相似性搜索，尤其是在处理大型数据集时。这引向了向量数据库这个方向。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，它是现代大型语言模型（LLMs）和许多文本嵌入模型的基础。
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) — OpenAI (2024)
  OpenAI官方的基于API的文本嵌入模型文档，涵盖了可用的模型和使用方法。
