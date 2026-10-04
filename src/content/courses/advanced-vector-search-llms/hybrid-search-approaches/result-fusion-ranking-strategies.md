---
course: "advanced-vector-search-llms"
chapter: "hybrid-search-approaches"
lesson: "result-fusion-ranking-strategies"
sourceId: 3876
sourceUrl: "https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-3-hybrid-search-approaches/result-fusion-ranking-strategies"
title: "结果合并与排序方法"
description: "用于合并和重新排序来自多种检索方法（向量、关键词）结果的技术。"
order: 3
plots: ["plots/3876-0.json"]
sourceHash: "9516ed2103dcc153926edb5f9f528e0396dfa74ea5be68dfed41185cd0b7390a"
sourceCorrections: []
---

结合来自不同系统（如向量 (vector)检索和传统关键词检索（例如BM25））的结果是混合检索的一个主要实践。并行运行这些检索会得到两个潜在结果列表，每个列表都有其文档集和相应的分数。目标是将它们归并成一个统一且排序合理的列表，该列表能够反映基于语义相似度和词项匹配的相关度。

### 分数无法直接比较的问题

合并结果时的一个主要困难是，不同检索方法产生的分数通常无法直接比较。

- **向量 (vector)检索分数：** 这些分数通常代表相似度或距离。余弦相似度分数通常落在[-1, 1]或[0, 1]的范围内，而欧氏距离为非负值（$[0, \infty)$）。这些分数衡量嵌入 (embedding)空间中的语义接近程度。
- **关键词检索分数：** 像BM25这样的算法产生的分数反映了词频、逆文档频率和文档长度。这些分数通常是非负值（$[0, \infty)$），但没有固定上限，并且其分布会根据查询词项和语料库统计数据显著变化。

尝试直接将0.85的余弦相似度与15.2的BM25分数相加或比较是没有意义的。它们代表着不同尺度上的不同概念。此外，分数与相关性之间的关系可能是非线性的，并且在两个系统之间有所不同。余弦相似度的微小增加可能代表显著的相关性提升，而高BM25分数的类似绝对值增加可能效果不那么明显。



![分数分布示例 (标准化前)](plots/3876-0.json)



> 向量检索（余弦相似度，底部坐标轴）和关键词检索（BM25，顶部坐标轴）中前10个文档的分数分布。分数越高通常越好，但其尺度和分布差异显著。

因此，有效的合并方法必须首先解决这种分数不兼容问题。

### 分数标准化方法

一种常用方法是在结合分数之前，将每个检索系统的分数标准化到统一量纲，通常是[0, 1]的范围。

1. **最小-最大缩放（Min-Max Scaling）：** 这种技术根据每个系统在*当前结果列表内*观察到的最小和最大分数，将分数线性缩放到[0, 1]的范围。

   
   $$
   Score_{norm} = \frac{Score - Score_{min}}{Score_{max} - Score_{min}}
   $$
   
   - **优点：** 实现简单，确保分数介于0和1之间。
   - **缺点：** 对异常值高度敏感（单个非常高或低的分数会压缩其他所有分数的范围），假设分数与相关性之间存在线性关系。需要获取足够的结果来确定合理的$Score_{min}$和$Score_{max}$。
2. **Z-分数标准化（标准分数）：** 这种方法根据结果列表内分数的均值（$\mu$）和标准差（$\sigma$）缩放分数。

   
   $$
   Score_{norm} = \frac{Score - \mu}{\sigma}
   $$
   
   - **优点：** 与最小-最大缩放相比，对极端异常值不那么敏感，并考虑了分数的分布。
   - **缺点：** 产生的分数没有边界（通常范围在-3到+3之间，但可能超出），需要计算均值和标准差。如果潜在分布差异很大（例如，高度偏斜），得到的分数可能仍然无法直接比较。可能需要进一步转换（例如应用sigmoid函数）才能映射到[0, 1]。

这些统计数据（$Score_{min}$，$Score_{max}$，$\mu$，$\sigma$）的计算通常是每次查询根据检索到的结果集（例如，每个系统的前100个文档）进行的。

### 加权和结合

分数标准化后（例如，使用最小-最大缩放得到$Score_{vec\_norm}$和$Score_{kw\_norm}$），一种直接的结合方式是通过加权和。对于一个由任一或两个系统检索到的文档$d$，其混合分数可以计算为：


$$
Score_{hybrid}(d) = w_{vector} \times Score_{vec\_norm}(d) + w_{keyword} \times Score_{kw\_norm}(d)
$$


这里：

- $Score_{vec\_norm}(d)$ 是文档$d$的标准化向量 (vector)分数。如果文档$d$未被向量检索发现，该分数为0。
- $Score_{kw\_norm}(d)$ 是文档$d$的标准化关键词分数。如果文档$d$未被关键词检索发现，该分数为0。
- $w_{vector}$ 和 $w_{keyword}$ 是决定每个检索系统相对重要性的权重 (weight)。通常，$w_{vector} + w_{keyword} = 1$。

**确定权重：** 权重$w_{vector}$和$w_{keyword}$是显著影响最终排序的超参数 (parameter) (hyperparameter)。它们的最优值高度依赖于具体的用例、数据特性和用户期望。

- 对于语义含义最重要的应用，$w_{vector}$可能会更高。
- 对于匹配特定词项或代码很重要的应用，$w_{keyword}$可能会更高。
- 这些权重通常通过以下方法凭经验进行调整：
  - **离线评估：** 在标注数据集（真实数据）上计算NDCG（归一化 (normalization)折损累积增益）或MAP（平均精度均值）等指标，以此优化权重。
  - **A/B测试：** 将不同的权重组合部署到不同的用户群体，并衡量点击率或任务成功率等在线指标。

### 基于排名的结合方法

分数标准化和加权和的替代方法是使用文档在每个列表中的排名位置。这种方法本质上避免了比较不同尺度分数的困难。其思路是，文档在排序列表中的位置比其原始分数更能稳定地指示相关性。我们将在下一节讨论的倒数排名结合（RRF）等方法便是基于此原理运作。它们结合的是排名而非分数，通常对分数变化更具弹性，并且比加权和方法所需的调优更少。

### 重新排序方法

结合方法主要侧重于合并初始结果列表。但是，您可以通过添加一个**重新排序**步骤来进一步优化排序。在使用结合方法（如加权和或RRF）获得一个（例如）前100个候选的结合列表后，可以应用一个更复杂、通常计算成本更高的模型，仅对这些候选进行重新排序。

- **目的：** 借助更丰富的特征或更复杂的模型（这些模型对整个语料库应用会太慢），以提升顶部结果的精度。
- **模型：** 这可能涉及：
  - **交叉编码器：** 联合处理查询和候选文档以产生高精度相关性分数的Transformer模型。
  - **排序学习（LTR）模型：** 基于从查询-文档对中提取的特征（包括原始向量 (vector)/关键词分数、文档元数据、查询特征等）进行训练的模型（例如LambdaMART、梯度提升树）。
- **权衡：** 重新排序可以显著提升相关性，但会增加检索过程的延迟，因为重新排序模型需要评估前N个候选中的每一个。如果顶部少量结果的质量提升被高度重视，这通常是可以接受的。

> 混合检索工作流程：通过向量和关键词检索进行初始获取，随后是结合步骤（例如，标准化和加权和）以创建候选列表，以及一个可选的重新排序步骤，使用更强大的模型进行最终排序。

### 实现考量

- **处理重叠：** 当一个文档同时出现在向量 (vector)和关键词结果列表中时，您需要一种一致的方式在结合过程中处理它。对于加权和，您只需将加权标准化分数相加。对于基于排名的结合方法（如RRF），在多个列表中出现自然有助于结合公式的计算。
- **性能：** 分数标准化和加权和计算通常比初始检索操作快很多。基于排名的结合方法也通常是轻量的。主要的性能考量来自添加重新排序步骤，它会引入额外的延迟，延迟量与重新排序的候选数量和重新排序模型的复杂度成比例。

选择正确的结合和排序方法需要平衡相关性要求、计算预算（延迟、成本）以及调优和维护所需的工作量。简单的加权和更容易上手，但可能需要仔细调优，而基于排名的结合方法可以更稳定，重新排序则提供了更高精度的潜力，但代价是增加了复杂性和延迟。

## 参考资料

- [Reciprocal Rank Fusion Outperforms Other Fusion Methods for Cold Start, Diversification, and Other Tough Search Problems](https://dl.acm.org/doi/10.1145/1571941.1572007) — Gordon V. Cormack, Maura Grossman, Charles L. A. Clarke (2009)
  Journal: Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval; Publisher: Association for Computing Machinery; Pages: 1009–1010; DOI: [10.1145/1571941.1572007](https://doi.org/10.1145/1571941.1572007)
  介绍了倒数排名融合（RRF），一种基于排名的融合方法，用于结合来自不同检索系统的排名列表。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  一本涵盖信息检索基础概念（包括BM25、评分和基本结果组合方法）的标准教科书。
- [Neural Information Retrieval](https://link.springer.com/chapter/10.1007/978-3-030-90981-0_7) — Sean MacAvaney, Andrew Yates, Arman Cohan, Nazir Correia, Ophir Frieder (2022)
  Journal: The Handbook of Information Retrieval; Publisher: Springer, Cham; Pages: 181-209; DOI: [10.1007/978-3-030-90981-0_7](https://doi.org/10.1007/978-3-030-90981-0_7)
  概述了现代神经信息检索技术，包括用于高级检索系统中重新排序的深度学习模型。
