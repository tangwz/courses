---
course: "mastering-gradient-boosting-algorithms"
chapter: "gradient-boosting-specialized-tasks"
lesson: "survival-analysis-boosting"
sourceId: 2070
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-9-gradient-boosting-specialized-tasks/survival-analysis-boosting"
title: "梯度提升在生存分析中的应用"
description: "调整梯度提升以处理事件发生时间数据和删失观测。"
order: 3
plots: []
sourceHash: "5168a4e8f4da325aca6e9116550b7a2c42f0a9fc74c8a8581b5eee682e6097d5"
sourceCorrections: []
---

生存分析，又称事件发生时间分析，旨在预测特定目标事件发生的时间。这在不同领域都有应用，从医学中预测患者生存时间、工程中预测组件失效时间到商业中预测客户流失。生存数据的一个主要特点和难点是存在**删失**。

### 理解删失

在许多研究中，我们未能观察到所有受试者在观察期内事件的发生。这称作删失。最常见的类型是**右删失**，它发生在：

1. 研究在受试者经历事件之前结束。
2. 受试者在事件发生前退出研究或失访。

我们知道受试者至少存活到删失时间，但在那之后我们不知道他们实际的事件发生时间。直接预测事件时间的标准回归方法，或预测事件是否发生的分类方法，都不够用，因为它们无法正确处理删失观测引入的不确定性。忽略删失或将删失时间视为事件时间会导致有偏差的结果。

以下是事件时间与删失的简单可视化：

> 三名受试者的时间线。受试者1在研究期间内经历了事件(\*)。受试者2和3发生了右删失(O)，原因分别是研究结束或他们失访。

### 生存与风险函数

生存分析不直接预测事件时间$T$，而是侧重于使用以下方法描述其概率分布：

- **生存函数，$S(t)$：** 事件时间$T$大于某个指定时间$t$的概率。
  
  $$
  S(t) = P(T > t)
  $$
  
- **风险函数，$h(t)$（或风险率）：** 在已知生存到时间$t$的情况下，事件在时间$t$发生的瞬时可能性。
  
  $$
  h(t) = \lim_{\Delta t \to 0} \frac{P(t \le T < t + \Delta t \mid T \ge t)}{\Delta t}
  $$
  
  风险函数描述了事件在时间$t$发生的风险。这些函数之间存在关系：$S(t) = \exp\left(-\int_0^t h(u) du\right)$。

### 梯度提升在生存分析中的应用

梯度提升模型可以有效地应用于生存分析。提升算法通常不直接预测事件时间，而是用来建模**风险函数**，或者更常见的是与协变量$X$相关的**对数风险比**。

梯度提升的核心框架保持不变：顺序构建由基学习器（树）组成的加性模型$F(X)$。每棵新树$f_m(X)$旨在拟合一个适当损失函数 (loss function)的负梯度，该负梯度是根据当前模型预测$F_{m-1}(X)$计算的。主要的调整在于使用源自适用于删失事件发生时间数据的统计模型的损失函数。

### Cox比例风险模型与损失函数 (loss function)

Cox比例风险（Cox PH）模型是生存分析中的一个重要组成部分。它假设具有协变量$X_i$的个体$i$的风险为：


$$
h(t | X_i) = h_0(t) \exp(\eta_i)
$$


其中：

- $h_0(t)$是基线风险函数，一个所有个体共有的任意非负时间函数。
- $\eta_i = f(X_i)$是个体$i$的对数风险评分或对数风险比，在传统Cox模型中常被建模为线性组合$\beta^T X_i$。

重要的假设是*比例风险*：任意两个个体的风险比随时间保持不变。协变量$X_i$通过项$\exp(\eta_i)$对基线风险产生乘性作用。

梯度提升模型可以估计潜在的非线性对数风险评分$\eta_i = F(X_i)$。为了在提升框架内做到这一点，我们需要一个损失函数。标准方法使用**Cox偏似然**的负对数。这个似然函数能够巧妙地处理删失数据，而无需估计基线风险$h_0(t)$。

设$t_1 < t_2 < \dots < t_D$是数据中观察到的不同事件时间。设$\mathcal{D}_j$是在时间$t_j$经历事件的个体集合，$\mathcal{R}_j$是*风险集*——在时间$t_j$之前仍在观察中（未经历事件也未被删失）的个体集合。Cox偏似然为：


$$
L = \prod_{j=1}^{D} \frac{\prod_{i \in \mathcal{D}_j} \exp(\eta_i)}{\left( \sum_{k \in \mathcal{R}_j} \exp(\eta_k) \right)^{|\mathcal{D}_j|}}
$$


梯度提升的目标是最小化负对数偏似然：


$$
\text{损失} = -\log L = -\sum_{j=1}^{D} \left( \sum_{i \in \mathcal{D}_j} \eta_i - |\mathcal{D}_j| \log \left( \sum_{k \in \mathcal{R}_j} \exp(\eta_k) \right) \right)
$$


尽管完整的推导过程比较复杂，但可以计算该损失函数相对于预测$\eta_i = F(X_i)$的一阶和二阶导数（梯度和Hessian）。这些梯度和Hessian随后被用于标准的梯度提升算法中（特别是在构建树的步骤中，用于寻找最佳分裂和叶子值）。

像XGBoost这样的库通常内置支持此功能。例如，在XGBoost中指定`objective='survival:cox'`会指示算法使用负对数偏似然作为损失函数，并在内部计算相应的梯度和Hessian。

### 实现细节

- **数据格式：** 生存数据通常每个观测需要至少三条信息：
  - 特征向量 (vector)$X$。
  - 观察到的时间$T_{obs}$（如果事件发生，这是事件时间；否则是删失时间）。
  - 事件指示符$\delta$（例如，如果事件发生，$\delta=1$；如果删失，$\delta=0$）。一些库可能需要特定的编码，例如事件时间为正值，删失时间为负值。请务必查阅库的文档。
- **评估：** 标准指标如准确率或RMSE不适用。生存模型常用的评估指标包括：
  - **一致性指数（C-index）：** 衡量模型正确预测风险评分较高受试者更早经历事件的可比受试者对的比例。它类似于分类中的AUC。
  - 时变AUC。
  - Brier分数。
- **输出：** 模型通常为每个受试者输出对数风险比$\eta = F(X)$。值越高表示预测风险越高。这些风险评分可用于按风险对个体进行排序，或潜在地估计生存概率（尽管估计$S(t)$需要额外步骤，通常涉及基线风险的估计）。

### 优势与注意事项

将梯度提升应用于生存分析有以下几个优点：

- **捕获非线性和相互作用：** 基于树的结构自然能够处理特征与风险之间复杂的关联。
- **特征重要性：** SHAP等方法仍可用于理解特征对预测风险评分的贡献。
- **正则化 (regularization)：** 标准的提升正则化技术（收缩、子抽样、树约束）有助于防止过拟合 (overfitting)。

然而，请注意以下几点：

- **比例风险假设：** 如果使用基于Cox的目标函数，模型本身依赖于比例风险假设。违反此假设会影响模型的有效性。测试此假设或考虑替代模型（如加速失效时间模型，尽管在标准提升包中实现较少）可能很有必要。
- **解释性：** 尽管可以获得特征重要性，但解释特征在时间尺度上的具体影响可能不如参数 (parameter)生存模型那样直观。输出是风险评分，而不是直接的生存时间或概率。

梯度提升为建模删失事件发生时间数据提供了一个强大且灵活的工具，通常通过其建模复杂数据结构的能力实现出色的预测表现。许多现代提升库提供了特定的目标函数，使得将生存分析应用于其中相对简单，只要数据格式正确即可。

## 参考资料

- [Regression Models and Life-Tables](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x) — David R. Cox (1972)
  Journal: Journal of the Royal Statistical Society. Series B (Methodological); Publisher: Royal Statistical Society; Volume: 34; Pages: 187-220; DOI: [10.1111/j.2517-6161.1972.tb00899.x](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x)
  引入Cox比例风险模型的开创性论文，是生存分析的基础。
- [The Statistical Analysis of Failure Time Data](https://www.wiley.com/en-us/The+Statistical+Analysis+of+Failure+Time+Data,+2nd+Edition-p-9781118032985) — John D. Kalbfleisch, Ross L. Prentice (2011)
  Publisher: Wiley-Interscience; DOI: [10.1002/9781118032985](https://doi.org/10.1002/9781118032985)
  关于生存分析理论与应用的综合性教材，第二版。
- [Greedy Function Approximation: A Gradient Boosting Machine](http://www.jstor.org/stable/2699986) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  关于梯度提升机（GBM）算法的奠基性论文，该算法被应用于生存分析。
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍了XGBoost库，该库提供了`survival:cox`目标函数，用于将梯度提升应用于生存分析。
