---
course: "large-scale-distributed-rag"
chapter: "advanced-distributed-retrieval-strategies"
lesson: "graph-retrieval-distributed-rag"
sourceId: 6174
sourceUrl: "https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-2-advanced-distributed-retrieval-strategies/graph-retrieval-distributed-rag"
title: "分布式环境中的图检索"
description: "在分布式RAG架构中使用图检索方法进行复杂信息查找。"
order: 4
plots: []
sourceHash: "07a5bb9abea0cd1d176fe552cdd41cddec3049ea68ab4b8187efd78cbdef5e2a"
sourceCorrections: []
---

尽管向量 (vector)检索擅长查找语义相似的独立信息块，但许多复杂的信息需求需要理解数据片段“之间”的关系。图检索处理此类需求，通过将数据建模为实体和联系互联的网络。利用图能够让分布式RAG系统实现复杂的查询功能，可以通过明确连接、路径遍历和邻域分析来识别相关信息，通常对密集检索方法起到补充作用。

### 大规模知识图谱的构建与管理

图检索的核心是一个结构良好的知识图谱。对于大规模RAG，此类图通常包含：

- **节点**：表示实体，例如文档、文档片段、命名实体（人物、组织、地点）、主题，甚至用户。节点特征可以包含嵌入 (embedding)、文本描述或元数据。
- **边**：表示节点之间的关系。这些可以是显式的（例如，引用、超链接、作者关系、片段的“部分-整体”关系）或隐式的（例如，文档嵌入之间的强语义相似性、用户会话中的共现、推断出的主题关联）。边的属性可以包含权重 (weight)、时间戳或关系类型。

从大量异构数据集中构建此类图需要分布式处理。采用了Apache Spark（GraphX或GraphFrames）、Apache Flink（Gelly）或专用图处理引擎等框架。此过程通常包括：

1. **实体和关系提取**：在源文档上使用NLP技术，包括命名实体识别（NER）和关系提取（RE）。
2. **节点和边定义**：将提取的信息和文档结构映射到图元素。
3. **分布式加载**：将节点和边摄取到分布式图数据库或可伸缩图处理系统中。此步骤必须处理高吞吐量 (throughput)和潜在的并发更新。
4. **图增强**：使用从图结构（例如，中心性分数）或外部知识库得来的信息来扩充图。

维护这些图也需要强大的数据管道，能够以分布式方式处理更新、删除和模式演变，确保图反映底层数据的最新状态。

### 分布式图数据库与查询技术

高效存储和查询大型图需要专门的分布式图数据库。选项包括：

- **属性图数据库**：像Neo4j（及其因果集群）、JanusGraph（由ScyllaDB、Cassandra或HBase支持）、ArangoDB（具备图能力的多模型数据库）或TigerGraph等系统，专为事务性图工作负载和复杂遍历而设计。它们提供Cypher（用于Neo4j、ArangoDB）或Gremlin（用于JanusGraph、TinkerPop支持的系统）等查询语言。
- **RDF三元组存储**：尽管对于典型的RAG文档检索不那么常见，但当数据自然表示为RDF三元组时，分布式RDF存储（如Amazon Neptune，也支持属性图）或Virtuoso可以使用，特别是在与公共链接数据集成时。

分布式图数据库的一项主要挑战是**数据分区（分片）**。有效的分区旨在最大程度减少查询执行期间的分区间通信。常见策略包括：

- **边切割分区**：节点被分配到分区，跨分区的边被“切割”。这可能导致高度节点的高度复制。
- **顶点切割分区**：边被分配到分区，连接到多个分区中边的顶点被“切割”（即，顶点状态被复制或分割）。

分区策略的选择取决于图结构和查询模式。分布式图数据库中的查询优化包括最大程度减少数据移动和跨分区并行遍历。

### 用于检索的图遍历和搜索算法

图存储好后，可使用多种算法进行检索，并适应分布式执行：

- **个性化PageRank (PPR)**：给定一组查询节点（例如，从用户查询中提取的实体），PPR识别图中与查询节点高度“连接”的其他节点。在分布式环境中，像PPR这样的迭代算法需要仔细管理跨分区的状态和高效的消息传递。节点 $v$ 对起始集 $S$ 的分数更新规则 $PPR_k(v|S)$ 通常涉及聚合其邻居的分数：
  
  $$
  PPR_k(v|S) = (1-\alpha) \cdot \mathbb{1}_{v \in S} + \alpha \sum_{w \in N_{in}(v)} \frac{PPR_{k-1}(w|S)}{deg_{out}(w)}
  $$
  
  分布式地计算此和并管理跨分片的收敛存在一定难度。
- **带重启的随机游走 (RWR)**：类似于PPR，RWR模拟从查询节点开始的随机游走，每一步都有可能回到查询节点。这有助于在查询实体的局部邻域进行查找。分布式实现面临管理游走器状态和高效访问跨分区邻居信息的挑战。
- **最短路径和K跳邻居**：查找查询实体和潜在答案节点之间的最短路径，或检索查询实体 $k$ 跳内的所有节点。使用分布式广度优先搜索（BFS）或深度优先搜索（DFS）变体，需要协调来管理机器间的已访问集合和前沿。
- **子图模式匹配**：识别图中表示相关性的特定模式或主题。例如，通过“由…创作”边找到与“项目X”节点连接到“团队Y”节点的文档。分布式子图同构计算密集，但功能强大。

这些算法可以使用图数据库的查询语言（例如，Cypher的可变长度路径模式、Gremlin遍历）实现，或者，如果需要更复杂的迭代计算，可以通过图处理框架实现。

### 将图增强检索集成到RAG流水线中

图检索在大规模RAG系统中很少单独起作用。它通常与其他检索方法集成：

1. **候选来源**：图遍历可以识别一组初始的候选文档或实体。例如，从查询中提及的实体开始，检索2-3跳内的连接文档。这些候选者随后可以送入更精确的重排序器，可能是基于神经网络 (neural network)的。
2. **查询扩展/优化**：通过图遍历识别的实体（例如，查询词的邻居）可以用于扩展或优化原始查询，以便进行后续的密集或稀疏检索阶段。
3. **上下文 (context)丰富**：检索高度相关节点（通过向量 (vector)搜索或初始图处理识别）周围的子图，可以为LLM提供丰富、结构化的上下文。此上下文可能包括相关实体、它们的属性及其相互连接，使LLM能够生成更全面、更具见地的回答。
4. **混合检索得分**：来自图检索的信号（例如，路径存在、相关子图中节点的中心性、PPR得分）可以与向量搜索和稀疏检索方法的得分在一个整合层中结合，然后再进行最终的重排序。

以下图表展示了图检索如何集成到混合分布式RAG架构中：

> 混合RAG架构，结合了基于图和基于向量的检索，在LLM生成前送入融合和重排序阶段。

### 运行考量与可伸缩性挑战

在分布式环境中部署图检索带来了独特的运行挑战：

- **图数据摄取与更新**：高效地摄取新数据并近实时地反映更新（节点/边的添加、修改、删除）而不影响查询性能是困难的。变更数据捕获（CDC）机制和增量图更新策略是重要的。
- **遍历的可伸缩性**：深度遍历或涉及超级节点（度极高的节点）的查询可能成为性能瓶颈。策略包括限制遍历深度的查询规划、对高度节点的专门索引或采样技术。
- **一致性模型**：分布式图数据库提供多种一致性模型（例如，最终一致性、因果一致性）。选择会影响数据新鲜度与查询性能和复杂性。对于RAG，最终一致性对于图数据通常是可以接受的，但这需要根据用例对陈旧信息的敏感度进行仔细评估。
- **成本管理**：大规模分布式图数据库在存储、内存（用于缓存图结构）和计算（用于遍历）方面可能资源密集。优化数据布局、查询模式和硬件配置对于经济高效的运行是必要的。
- **模式管理**：尽管某些图数据库具有模式灵活性，但随着图的复杂性增加和集成更多数据源，定义和演进连贯的模式或本体变得越来越重要。

通过仔细处理这些系统设计和运行方面，图检索可以显著提升大规模分布式RAG系统的能力，使它们能够基于对知识库内关系更全面的理解来回答问题和生成内容。

## 参考资料

- [The PageRank Citation Ranking: Bringing Order to the Web](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) — Lawrence Page, Sergey Brin, Rajeev Motwani, Terry Winograd (1999)
  Journal: Stanford InfoLab Technical Report
  介绍PageRank算法的奠基性论文，该算法是许多基于图的排名和遍历方法（包括本节讨论的个性化PageRank）的基础。
- [A Survey on Knowledge Graphs: From Construction to Applications](https://doi.org/10.1109/TKDE.2019.2928844) — Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, Guodong Long (2021)
  Journal: IEEE Transactions on Knowledge and Data Engineering; Publisher: IEEE; Volume: 34; Pages: 1888-1910; DOI: [10.1109/TKDE.2019.2928844](https://doi.org/10.1109/TKDE.2019.2928844)
  一份关于知识图谱构建、管理和应用的综合性综述，包括实体和关系提取，这些都是在分布式环境中大规模图谱创建的基础。
- [GraphX: A Resilient Distributed Graph System on Spark](https://doi.org/10.1145/2484425.2484427) — Reynold S. Xin, Joseph E. Gonzalez, Michael J. Franklin, Ion Stoica (2013)
  Journal: Proceedings of the 1st International Workshop on Graph Data Management Experiences and Systems (GRADES '13); Publisher: ACM; Pages: 2:1-2:6; DOI: [10.1145/2484425.2484427](https://doi.org/10.1145/2484425.2484427)
  介绍了GraphX，一个基于Apache Spark构建的分布式图处理框架，与本节中关于分布式图构建和PageRank等算法的讨论直接相关。
