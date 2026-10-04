---
course: "causal-inference-ml-systems"
chapter: "advanced-causal-discovery"
lesson: "discovery-interventional-data"
sourceId: 4327
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-2-advanced-causal-discovery/discovery-interventional-data"
title: "因果结构发现：使用干预数据"
description: "将干预数据与观察性数据结合，提升因果发现精度。"
order: 3
plots: []
sourceHash: "a09a00131af1badaa325d546975291428a592a9a19ea50e92616f3f09ff4f832"
sourceCorrections: []
---

观察性数据为因果发现提供依据，但仅依赖它常会带来根本性不确定。多个有向无环图（DAG）可以表示数据中观察到的一组条件独立性。这些图构成一个马尔可夫等价类（MEC）。若无额外信息，仅凭观察性数据通常无法分辨MEC内的真实因果结构。在此背景下，干预数据变得不可或缺。干预能主动操控系统，提供被动观察无法得到的因果方向直接证据。

### 通过主动操控解决不确定性

设想一个含$X$、$Y$、$Z$三个变量的简单系统。假设观察性数据显示，给定$Y$时$X$和$Z$条件独立（$X \perp Z | Y$），但$X$和$Y$相关，$Y$和$Z$相关。这种独立性结构与同一MEC内的多个DAG兼容，其中包括：

1. **链式结构：** $X \rightarrow Y \rightarrow Z$
2. **分支结构：** $X \leftarrow Y \rightarrow Z$
3. **链式结构（反向）：** $X \leftarrow Y \leftarrow Z$

仅凭观察性数据无法区分这些结构。但进行干预则改变了局面。想象我们可以干预$Y$，将其值设定为一个特定常数$y'$，记作$do(Y=y')$。此操作在真实图中有效移除了所有指向$Y$的因果箭头。

- 如果真实图是$X \rightarrow Y \rightarrow Z$，干预$Y$会切断$X$对$Y$的影响。然而，$Y \rightarrow Z$的连接仍保留。干预后$X$和$Z$将变得独立：$X \perp Z$。
- 如果真实图是$X \leftarrow Y \rightarrow Z$，干预$Y$会移除$Y$对$X$和$Z$的影响。干预后$X$和$Z$将独立：$X \perp Z$。
- 如果真实图是$X \leftarrow Y \leftarrow Z$，干预$Y$会切断$Z$对$Y$的影响。$X \leftarrow Y$的连接仍保留。$X$和$Z$通过$X \leftarrow Y$保持相关。

通过观察干预分布$P(X, Z | do(Y=y'))$中的（不）独立性，我们有可能区分这些结构。例如，如果干预$Y$后$X$和$Z$保持相关，我们可以排除前两种结构。

下图说明了在干预下观察独立性如何帮助区分两个观察性等价图。

> 对比同一MEC中的两个图。观察性数据显示两者均为$X \perp Z | Y$。干预$Y$（红色节点）会切断入向边。在上方结构（$X \rightarrow Y \rightarrow Z$）中，$Y \rightarrow Z$路径仍保持活跃，可能保留相关性。在下方结构（$X \leftarrow Y \rightarrow Z$）中，来自$Y$的两条路径均被干预切断，导致$X \perp Z$。

### 干预类型

干预在性质和范围上可能有所不同，提供不同种类的信息：

- **完美干预：** 它们直接对应于$do$算子。这类干预涉及将变量$X_i$设定为固定值$x_i'$，完全覆盖其自然因果机制并切断所有入向边。这是信息量最大的类型，但在实际中常难以完美实现。
- **不完美（随机）干预：** 不完美干预（随机干预）并非固定$X_i$，而是修改其生成机制。例如，它可能改变条件分布$P(X_i | Pa(X_i))$的参数 (parameter)或添加外部随机影响。这些情况通常更贴近现实（如鼓励而非强制某种行为）。它们仍提供信息，但需要仔细建立干预机制本身的模型。
- **多重干预：** 同时或按序干预多个变量，可展现更高阶的交互效应，并比单变量干预更能解决复杂的模糊性。
- **未知干预目标：** 有时，我们可能知道干预发生了（可能在部分数据中），但不确切知道哪个或哪些变量受到了影响。算法需为此不确定性而设计。

### 集成干预数据的算法

几种因果发现算法旨在利用干预提供的额外信息：

1. **基于约束的扩展（例如，PC/FCI的修改版）：** 标准的基于约束的算法依赖于对观察性数据的条件独立性测试。当有干预数据时，这些测试可应用于对应特定干预的数据子集。对于一次完美干预$do(X_k=x_k')$，我们知道在该子集中$X_k$没有父节点。此信息可用于确定观察性MEC中先前无方向边的方向。例如，如果$X - Y$是基于观察性数据的无向边，但我们观察到在$do(Y=y')$下$X$和$Y$变为独立，这暗示方向为$X \rightarrow Y$。FCI（快速因果推断）处理潜在混杂因素，也能纳入干预数据以完善所得的部分祖先图（PAG）。
2. **基于分数的扩展（例如，GIES）：** 贪婪干预等价搜索（GIES）扩展了贪婪等价搜索（GES）算法。GES使用评分（如BIC）在MEC空间中进行搜索。GIES修改了评分过程以处理混合的观察性数据和干预数据。它假设干预目标已知。候选图的评分通常通过对不同数据状态（观察性及各种干预）的评分求和来计算，其中每个状态的似然项考虑了干预下的修改图结构（即移除完美干预节点的入向边）。

   
   $$
   \text{分数}(G, D) = \sum_{j=0}^{m} \text{局部分数}(G, D_j)
   $$
   

   其中$D_0$是观察性数据集，$D_j$（$j > 0$）是在不同干预状态下的数据集，且$局部分数$评估（可能已修改的）图$G$与数据$D_j$的匹配程度。
3. **不变因果预测（ICP）：** ICP使用来自多个“环境”或“设置”的数据。这些环境可能自然形成（如来自不同地点的数据），或由干预明确生成。核心思想是，目标变量$Y$的*真实*因果预测因子$Pa(Y)$在这些环境中保持不变，即条件分布$P(Y | Pa(Y))$是稳定的。非因果预测因子（如后代或通过混杂相关的变量）可能表现出在不同环境中变化的统计关系。ICP寻找一组预测因子$S$，使得条件分布$P(Y|X_S)$在所有观察到的环境中保持不变。当存在干预但可能未完美描述时，此方法特别有用。

### 结合观察性数据与干预数据

通常，最有效的方法是同时使用观察性数据和干预数据。观察性数据提供广泛覆盖，并辅助识别初始MEC。随后可以设计或应用定向干预，以解决该类中剩余的不确定性。如GIES等算法便是为此混合环境明确设计的。

处理组合数据时，正确组织其结构很重要。通常这包括汇集数据并添加列，以指示每个样本的干预状态。

```python
import pandas as pd

# 示例观察性数据
obs_data = pd.DataFrame({
    'X': [0.1, 1.1, -0.5, ...],
    'Y': [0.5, -0.2, 1.8, ...],
    'Z': [-1.0, 0.8, 0.1, ...],
    'intervention_target': ['None'] * n_obs
})

# 示例干预数据（对Y进行干预）
int_data_Y = pd.DataFrame({
    'X': [0.3, -1.2, 0.7, ...],
    'Y': [10.0] * n_int_Y, # 通过干预固定的值
    'Z': [0.6, -0.9, 1.1, ...],
    'intervention_target': ['Y'] * n_int_Y
})

# 组合数据集
combined_data = pd.concat([obs_data, int_data_Y], ignore_index=True)

# 与发现函数一起使用
# from causallearn.search.ScoreBased.gies import gies

# 假设'gies'函数可接受干预目标信息
# result = gies(combined_data[['X', 'Y', 'Z']].values,
#               intervention_targets=combined_data['intervention_target'].tolist())

# 'result'将包含关于习得图结构的信息
```

### 最优实验设计

如果您有资源进行干预，一个问题便出现：哪些干预信息量最大？因果发现中的最优实验设计，旨在选择一系列干预，以最有效率地解决当前估计MEC或PAG中的不确定性。这通常涉及识别预期能最大限度减少与数据一致的可能因果结构数量的干预。这是一个高级议题，常涉及复杂的评分或信息论标准。

### 实际难题

尽管功能强大，但使用干预数据也伴随难题：

- **可行性与成本：** 进行干预可能成本高昂、不道德，或在许多系统中实际无法实现（例如干预社会经济地位）。
- **不完美干预：** 干预很少能在没有副作用（“脱靶效应”）或完全切断入向影响的情况下，完美地固定变量值。对实际干预机制进行建模变得重要。
- **未知目标：** 确定干预影响了哪些变量可能很困难。
- **可扩展性：** 处理组合观察性数据和干预数据的算法可能计算需求高，特别是当变量和干预类型很多时。

尽管存在这些难题，干预打破对称性并确定边方向的能力，使得干预数据成为因果发现的宝贵资源，克服了纯粹观察性方法的局限。当有干预数据时，集成它能显著增强对推断因果结构的信心。

## 参考资料

- [Causality: Models, Reasoning, and Inference](https://www.cambridge.org/core/books/causality/C352ED5700D8C30B2CE08A1856860E91) — Judea Pearl (2009)
  Publisher: Cambridge University Press; DOI: [10.1017/CBO9780511818311](https://doi.org/10.1017/CBO9780511818311)
  这本基础性著作介绍了do-演算、结构因果模型以及干预在因果发现中的理论作用。
- [Causation, Prediction, and Search](https://mitpress.mit.edu/books/causation-prediction-and-search-second-edition) — Peter Spirtes, Clark Glymour, Richard Scheines (2000)
  Publisher: The MIT Press
  介绍了PC和FCI等基于约束的因果发现算法，并确立了马尔可夫等价类的概念，阐明了干预数据的重要性。
- [Elements of Causal Inference: Foundations and Learning Algorithms](https://mitpress.mit.edu/9780262037310/elements-of-causal-inference/) — Jonas Peters, Dominik Janzing, Bernhard Schölkopf (2017)
  Publisher: The MIT Press; DOI: [10.7551/mitpress/9780262037310.001.0001](https://doi.org/10.7551/mitpress/9780262037310.001.0001)
  一本全面的教材，涵盖了现代因果推断方法，包括使用干预数据进行因果发现、不同干预类型以及不变因果预测。
- [Characterization and greedy search of Markov equivalence classes of interventional DAGs](https://www.jmlr.org/papers/volume13/hauser12a/hauser12a.pdf) — Andreas Hauser, Peter Bühlmann (2012)
  Journal: Journal of Machine Learning Research; Volume: 13; Pages: 2409-2463; DOI: [10.5555/2361093.2361110](https://doi.org/10.5555/2361093.2361110)
  介绍了贪婪干预等价搜索（GIES）算法，这是一种基于分数的因果发现方法，旨在从观测数据和干预数据中学习因果图。
- [Causal inference using invariant prediction: identification and confidence intervals](https://academic.oup.com/jrsssb/article/78/5/947/7000185) — Jonas Peters, Peter Bühlmann, Nicolai Meinshausen (2016)
  Journal: Journal of the Royal Statistical Society Series B: Statistical Methodology; Publisher: Wiley; Volume: 78; Pages: 947-1012; DOI: [10.1111/rssb.12167](https://doi.org/10.1111/rssb.12167)
  介绍了不变因果预测（ICP）的原始研究论文，该方法利用来自多个环境的数据来识别因果预测变量。
