---
course: "agentic-llm-memory-architectures"
chapter: "designing-memory-systems"
lesson: "long-term-memory-vector-stores"
sourceId: 4467
sourceUrl: "https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-3-designing-memory-systems/long-term-memory-vector-stores"
title: "长期记忆：向量数据库与嵌入"
description: "使用向量数据库（例如，Pinecone、Chroma）进行语义信息检索。"
order: 3
plots: ["plots/4467-0.json"]
sourceHash: "0b0edfb0ee585b70c1301b3df335808321880ca6c1e92dd242143dab204a085c"
sourceCorrections: []
---

虽然短期记忆机制处理即时上下文 (context)，但智能体系统通常需要访问大量信息，或回溯远超LLM上下文窗口 $L_{context}$ 的经验。这就需要持久的长期记忆方案。与简单的关键词搜索不同，智能体经常需要根据相似性或含义来检索信息。向量 (vector)数据库和文本嵌入 (embedding)在此扮演重要角色。

### 语义搜索与嵌入 (embedding)

LLM智能体现代长期记忆的根基在于语义搜索。语义搜索并非匹配精确词语，而是依据含义相似度寻找信息。这通过将文本转换为名为嵌入的数值表示来实现。

嵌入是高维空间 (high-dimensional space)中的一个稠密向量 (vector) $e$，由嵌入模型 $f$ 生成，即 $e = f(text)$。这些模型经过训练，使得含义相似的文本在向量空间中映射到彼此接近的点。例如，“智能体记忆系统”和“为自主AI存储信息”的嵌入可能会比“智能体记忆系统”和“金融市场分析”的嵌入更接近。

常见的嵌入模型包括Sentence-BERT (SBERT)变体、OpenAI的Ada嵌入或Cohere的嵌入模型。它们在维度（例如384、768、1536或更多维度）、训练目标以及捕捉语义相似度的细微差别上有所不同。

两个嵌入之间的“接近度”，以及原始文本之间的语义相似度，通常使用距离度量方法进行衡量，例如欧几里得距离，或更普遍的余弦相似度。余弦相似度衡量两个向量之间夹角的余弦值，范围从-1（含义相反）到1（含义相同），0表示正交或不相关。对于两个非零嵌入向量 $A$ 和 $B$:


$$
\text{相似度}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
$$


余弦相似度得分越高，表示语义关联 (semantic relationship)性越大。



![简化的2D嵌入空间可视化](plots/4467-0.json)



> 文本嵌入的简化2D投影。聚在一起的点代表语义相似的文档。查询向量（星形）检索最近的文档向量（蓝色簇）。

### 向量 (vector)数据库：嵌入 (embedding)的专用数据库

存储和高效查询数百万或数十亿高维向量需要专用的数据库，这类数据库被称为向量数据库。实例包括Pinecone等托管服务，以及Chroma、Milvus、Weaviate等自部署选项，或FAISS（Facebook AI相似度搜索）等库。

这些数据库针对近似最近邻 (ANN) 搜索进行了优化。给定一个查询向量 $e_{query}$，目标是根据所选距离度量（如余弦相似度或欧几里得距离）找到数据库中与 $e_{query}$ 最接近的 $k$ 个向量。

**索引与查询流程：**

1. **数据准备**：长文档或数据源通常被分割成更小、易于处理的块。选择合适的块分割策略（例如，固定大小、句子分割、递归分割）对检索质量有重要影响。
2. **嵌入**：每个文本块通过嵌入模型生成其向量表示。
3. **索引**：嵌入向量、原始文本块以及可能的关联元数据（例如，源文档ID、创建时间戳、章节引用）存储在向量数据库中。数据库构建索引结构（例如，HNSW图、IVF索引）以支持快速搜索。
4. **查询**：当智能体需要信息时，其查询（例如，“智能体设计中的挑战是什么？”）使用*相同*的嵌入模型进行嵌入。
5. **ANN搜索**：向量数据库使用其索引高效地找到与查询嵌入最接近的 $k$ 个嵌入向量。这一步通常涉及ANN算法，这些算法牺牲完美精度以换取显著的速度提升，这对于实时智能体交互非常重要。常见的ANN算法包括分层可导航小世界（HNSW）图和倒排文件索引（IVF）。
6. **检索**：数据库返回与识别出的最近邻向量对应的原始文本块（及其元数据）。

> 智能体系统中从向量数据库检索信息的工作流程。离线索引过程涉及嵌入并存储文本块。运行时，智能体根据任务形成查询，将其嵌入，使用ANN搜索向量数据库，检索相关块，并使用它们来增强核心LLM的上下文 (context)。

向量数据库能够简化管理ANN索引的复杂性，并提供API以便轻松插入、删除和查询向量数据。它们是为LLM提供可扩展、可搜索长期记忆的根本。特定向量数据库的选择通常取决于可扩展性要求、托管偏好（云端与本地）、期望的一致性保证以及搜索期间对元数据过滤的支持等因素。后续的“高级检索策略”一节将论述提升从这些系统检索信息质量的方法。

## 参考资料

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers, Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982-3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  介绍了一种获取语义丰富句子嵌入的方法。
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) — Yu. A. Malkov, D. A. Yashunin (2018)
  Journal: Proceedings of the 2018 SIAM International Conference on Data Mining (SDM); Publisher: SIAM; Pages: 462-470; DOI: [10.1137/1.9781611975348.51](https://doi.org/10.1137/1.9781611975348.51)
  描述了用于高效近似最近邻搜索的层次可导航小世界（HNSW）算法。
- [Billion-scale similarity search with GPUs](https://ieeexplore.ieee.org/document/8099859) — Jeff Johnson, Matthijs Douze, Hervé Jégou (2017)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 6228-6236; DOI: [10.1109/CVPR.2017.653](https://doi.org/10.1109/CVPR.2017.653)
  介绍FAISS，一个用于在大型向量数据集中进行高效相似性搜索的库。
- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings) — OpenAI (2024)
  Publisher: OpenAI
  提供使用 OpenAI 嵌入模型的指南和 API 参考。
