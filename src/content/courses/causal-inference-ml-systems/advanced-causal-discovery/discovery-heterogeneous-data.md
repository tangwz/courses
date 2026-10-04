---
course: "causal-inference-ml-systems"
chapter: "advanced-causal-discovery"
lesson: "discovery-heterogeneous-data"
sourceId: 4330
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-2-advanced-causal-discovery/discovery-heterogeneous-data"
title: "从异构数据中进行因果发现"
description: "整合并进行跨不同数据源或环境的因果发现的方法。"
order: 4
plots: []
sourceHash: "63373e554fdb5f4259d9ba34dc64ff86aa0c06147b8083d9c6b2df8a688815e4"
sourceCorrections: []
---

“数据很少以由单一固定概率分布控制的整洁包形式出现。更常见的是，我们遇到**异构数据**：源自不同来源、条件、环境或实验设置的数据集集合。例如，你可能拥有来自不同工厂的传感器读数、多家医院的患者数据，或跨越不同区域或时间段的经济指标。尽管其基本因果机制*可能*是共享的，但变量的具体分布、执行的干预措施，甚至测量的变量都可能显著不同。

简单地汇集异构数据集并直接应用标准因果发现算法（如PC、FCI或GES）通常会带来问题。数据集之间分布的差异可能会违反这些算法所依赖的假设，特别是忠实性假设。不适当地组合数据可能导致虚假连接或遗漏连接，从而有效地掩盖真正的基本因果结构。设想在一个混杂因素 $Z$ 在两个子组中与 $X$ 和 $Y$ 存在不同关系的数据中；汇集数据可能会模糊甚至颠倒 $X$ 和 $Y$ 之间的表观关系。

### 将异构性转化为优势

先进的因果发现技术不将异构性单纯视为复杂性，而是将其视为有价值的信息来源。如果正确建模，数据集之间的变异可以提供约束，有助于识别从任何单一数据集都可能模糊不清或无法发现的因果关系。其核心思路是，尽管观测分布可能在不同环境中变化，但其基本*因果机制*本身通常是不变的或稳定的。

设想一个简单例子：假设我们有两个测量变量 $X, Y, Z$ 的数据集。在数据集1中，$X$ 的方差较低，而在数据集2中则较高，这可能是由于不同的背景条件或针对 $X$ 的干预措施。如果关系 $X \rightarrow Y$ 是真正的因果关系，我们预期从 $X$（及其其他父变量）生成 $Y$ 的机制在两个数据集中保持一致，即使 $X$ 的分布发生变化。反之，如果 $Y$ 是 $X$ 的原因，那么 $X$ 分布的变化不应影响 $Y \rightarrow X$ 机制本身。通过寻找在这些不同环境中保持稳定的结构模型或关系，我们可以获得更强的特定因果联系证据。

### 不变因果预测 (ICP)

一个利用此原理的著名框架是**不变因果预测 (ICP)**。ICP 旨在通过借助来自多个环境或设置的数据，识别目标变量 $Y$ 的直接原因集。核心假设是：

1. **环境的存在：** 数据可以划分为不同的环境 $\mathcal{E} = \{e_1, e_2, ..., e_m\}$，其中 $m \ge 2$。这些环境可以代表不同的实验条件、时间段、位置或已知干预措施。
2. **基本SCM的稳定性：** 存在一个生成变量的基本结构因果模型 (SCM)，并且在所有环境 $e \in \mathcal{E}$ 中，$Y$ 在其直接原因 $Pa(Y)$ 条件下的条件分布是*相同*的。也就是说，$P(Y | Pa(Y))$ 是不变的。
3. **异构性：** 除了 $Y$ 之外，在给定其父变量或干预措施的情况下，其他变量的分布在不同环境间可能有所不同。这种变异提供了识别信息。

ICP 通过检验假设进行。对于给定的候选预测变量集 $S$，它检验 $S$ 包含真正的因果父变量 $Pa(Y)$ 并在所有环境中产生不变预测模型的零假设。这通常涉及在每个环境中对 $Y$ 进行关于 $S$ 的回归，并检验系数和残差分布是否统计上相同。

ICP 的最终输出是所有*不能被拒绝*的集合 $S$ 的交集。在其假设下，这个交集保证只包含 $Y$ 的真正因果父变量，提供了一个可能保守但高度可靠的因果预测变量集。

### 扩展异构数据的发现算法

其他方法调整了更广泛的发现算法：

- **联合因果推断 (JCI)：** JCI 框架试图同时借助各种数据类型（观测数据、干预数据、分布不同的数据）来估计单个基本因果图。
- **环境特定约束/评分方法：** 一些算法修改了基于约束（如FCI变体）或基于评分的方法（如GES变体），以明确考虑多个环境。它们可能寻找在不同环境中一致的结构，或者使用环境信息来解决不确定性（例如，确定边的方向）。例如，观察到在对 $Z$ 进行干预的环境中 $X$ 和 $Y$ 是独立的，而在其他情况下是相关的，这为涉及 $Z$ 的特定路径提供了有力证据。

### 异构性解决不确定性的例子

考虑三个变量 $X, Y, Z$。从一个环境中纯粹的观测数据，基于约束的方法可能识别出包含 $X \rightarrow Y \rightarrow Z$、$X \leftarrow Y \rightarrow Z$ 和 $X \leftarrow Y \leftarrow Z$ 的等价类。现在，假设我们有来自不同环境的第二个数据集，其中 $X$ 的方差明显提高，这可能是由于某种外部因素，但与第一个环境相比，$P(Y|X)$ 和 $P(Z|Y)$ 的条件分布似乎没有变化。这种不变性，加上 $X$ 分布的变化传播到 $Y$ 和 $Z$，有力地支持了 $X \rightarrow Y \rightarrow Z$ 的结构。

> 该图说明了观察到在 $X$ 属性不同（高亮节点）的环境中存在一致关系（粗箭头），如何支持 $X \rightarrow Y \rightarrow Z$ 因果链。

### 实际考量与实施

使用异构数据进行因果发现时：

1. **数据划分：** 根据已知干预措施、不同时间段、实验条件或子组，仔细定义不同的环境。环境划分的有效性非常重要。
2. **假设验证：** 不变因果机制的假设是强烈的。应使用敏感性分析或领域知识来评估其合理性。是否有理由认为因果律本身在不同环境之间会发生变化？
3. **算法选择：** 选择适合当前异构类型和特定研究问题（例如，ICP 用于识别 $Y$ 的父变量，更宽泛的方法用于完整图发现）的算法。
4. **软件：** Python 中像 `causal-learn` 这样的库提供了处理异构数据或结合环境信息的某些算法的实现。针对ICP等方法的专用软件包也可能直接从研究人员的仓库中获取。

整合来自异构源的数据为更可靠的因果发现提供了一个有效途径。通过将环境变异视为信号而非噪声，我们可以对可能的因果结构施加更强的约束，从而更准确地推断基本数据生成过程。

## 参考资料

- [Causal Inference Using Invariant Prediction: Identification and Search](https://onlinelibrary.wiley.com/doi/10.1111/rssb.12152) — Jonas Peters, Peter Bühlmann, Nicolai Meinshausen (2016)
  Journal: Journal of the Royal Statistical Society: Series B (Statistical Methodology); Publisher: John Wiley & Sons; Volume: 78; Pages: 947-973; DOI: [10.1111/rssb.12152](https://doi.org/10.1111/rssb.12152)
  提出了不变因果预测 (ICP) 的基础理论和算法，利用不同环境中稳定的因果机制来识别直接原因。
- [Elements of Causal Inference: Foundations and Learning Algorithms](https://mitpress.mit.edu/books/elements-causal-inference) — Jonas Peters, Dominik Janzing, Bernhard Schölkopf (2017)
  Publisher: The MIT Press
  一本关于因果推断理论基础的综合教材，包含不变预测和从异构数据中进行因果发现的详细解释。
- [A Survey of Causal Discovery from Non-Stationary/Heterogeneous Data](https://doi.org/10.1109/TPAMI.2020.2995326) — Wei Chen, Xiang Li, Ruichu Cai, Jie Qiao, Zhifeng Hao (2021)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 43; Pages: 2217-2234; DOI: [10.1109/TPAMI.2020.2995326](https://doi.org/10.1109/TPAMI.2020.2995326)
  概述了利用异构或非平稳数据进行因果发现的各种方法，包括对不变方法及其扩展的讨论。
