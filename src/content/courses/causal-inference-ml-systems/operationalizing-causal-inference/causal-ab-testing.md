---
course: "causal-inference-ml-systems"
chapter: "operationalizing-causal-inference"
lesson: "causal-ab-testing"
sourceId: 4417
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-6-operationalizing-causal-inference/causal-ab-testing"
title: "运用因果推断改进 A/B 测试"
description: "运用因果推断方法，改进在线实验和 A/B 测试的设计与分析。"
order: 4
plots: ["plots/4417-0.json"]
sourceHash: "20da9c3055788956e9519359fb02e1845bda6413ffea740bd9918a8552036548"
sourceCorrections: []
---

标准 A/B 测试，或称在线对照实验，是许多机器学习 (machine learning)系统中数据驱动决策的根本。通过将用户或单位随机分配到实验组 (A) 或对照组 (B)，它旨在评估某项干预措施（例如新功能或新算法）的平均处理效应 (ATE)。尽管功能强大，但标准 A/B 测试的实施常在简化假设下进行，这些假设在复杂系统中可能不成立。结合先进的因果推断技术，能显著提升这些实验的设计、分析和解读，提供更周全的理解和更可信的结论。

### 评估实验中的异质效应

尽管 ATE 提供了干预措施平均影响的重要概括，但它可能掩盖不同子群体反应方式的显著差异。理解这种异质性（条件平均处理效应，即 CATE）对于个性化、定向推广或发现集中在特定人群中的不利影响通常非常重要。

我们可以不依赖简单的结果事后细分，这种细分可能导致多重检验问题和虚假发现。我们可以运用专门针对 CATE 设计的估计器，并使用实验数据。第 3 章讨论的方法，例如因果森林和元学习器（S-学习器、T-学习器、X-学习器），直接适用。假设来自 A/B 测试的数据包含处理分配 $W_i \in \{0, 1\}$、结果 $Y_i$ 和处理前协变量向量 (vector) $X_i$，这些方法估计 $\tau(x) = E[Y_i(1) - Y_i(0) | X_i = x]$。

例如，将因果森林应用于 A/B 测试数据，使我们能够识别哪些用户特征（例如，参与度、设备类型、在 $X_i$ 中捕获的人口统计信息）与更大或更小的处理效应相关。这有助于更细致地理解影响。

考虑在实验数据上实施 X-学习器：

1. **阶段 1：** 使用单独的模型估计条件结果。
   - 使用实验组的数据拟合 $\hat{\mu}_1(x) = E[Y | X=x, W=1]$。
   - 使用对照组的数据拟合 $\hat{\mu}_0(x) = E[Y | X=x, W=0]$。
2. **阶段 2：** 归因个体处理效应。
   - 对于受处理单位：$\hat{D}_i^1 = Y_i - \hat{\mu}_0(X_i)$。
   - 对于对照单位：$\hat{D}_i^0 = \hat{\mu}_1(X_i) - Y_i$。
3. **阶段 3：** 通过建模归因效应来估计 CATE。
   - 使用受处理单位拟合 $\hat{\tau}_1(x) = E[\hat{D}^1 | X=x]$。
   - 使用对照单位拟合 $\hat{\tau}_0(x) = E[\hat{D}^0 | X=x]$。
4. **阶段 4：** 使用加权函数 $g(x)$ 组合估计值（通常是倾向得分，在标准 A/B 测试中已知且统一，例如 $g(x)=0.5$）。
   - $\hat{\tau}(x) = g(x) \hat{\tau}_0(x) + (1 - g(x)) \hat{\tau}_1(x)$。

这种结构化方法，在每个步骤中使用灵活的机器学习 (machine learning)模型，比简单的子群体分析提供了更精确的 CATE 估计。



![A/B 测试中的 ATE 与 CATE](plots/4417-0.json)



> 估计的处理效应在不同用户细分群体中可能差异很大（蓝线，CATE），而总体平均效应（红色虚线，ATE）可能会掩盖这些差异。

### 处理对标准假设的违反

"实验常常偏离基本 A/B 分析所假设的理想环境。因果推断提供了应对这些复杂情况的工具。"

#### 不完善的随机化和依从性

尽管平台力求实现完美的随机化，但实施细节、用户行为（例如，使用多个设备）或逐步推广有时会损害分配机制。如果怀疑存在问题，比较各组协变量分布（$P(X|W=1)$ 与 $P(X|W=0)$）的诊断检查非常重要。如果存在不平衡，通常用于观察性研究的方法，例如倾向得分加权或基于实验前协变量的匹配，有时可以调整微小的不平衡。更可靠地说，将双重机器学习 (machine learning)应用于实验数据，使用实验前协变量 $X$ 对结果 $Y$ 和（可能不完善的）处理分配 $W$ 进行建模，可以产生偏差较小的效应估计。不依从性（指分配的处理与实际接受的处理不同）可以使用工具变量 (IV) 方法来处理，将随机分配视为实际处理接受的工具。

#### 溢出效应和网络效应

稳定单位处理值假设 (SUTVA) 认为一个单位的结果只受其自身处理分配的影响，而不受其他单位的影响。在具有网络结构（社交网络、市场）的系统中，此假设经常被违反，因为处理一个用户可能会影响其同伴或市场动态。在此类情况下，标准 A/B 测试可能产生有偏估计。

处理溢出效应需要修改实验设计或分析：

- **基于群组的随机化：** 随机分配用户群组（例如，基于地理位置、社交图社区），而不是个体。这可以将溢出效应控制在群组内部，从而能够估计群组层面的效应。分析时需要考虑这种群组划分（例如，使用群组标准误差）。
- **网络感知估计：** 如果网络结构已知，模型可以尝试明确估计直接和间接（溢出）效应。这通常涉及图分析或结合网络特征的专门回归技术。

> 在群组层面随机分配处理（例如，群组 2 和 3 为实验组，群组 1 和 4 为对照组），有助于管理如果使用个体随机化可能会在跨群组的互联用户之间发生的溢出效应（虚线）。

### 减少方差以提高敏感度

实验通常旨在检测不明显的效应。减少结果指标的方差可以提高统计功效，从而能够检测更小的 ATE 或在更小的样本量或更短的持续时间内达到统计显著性。因果推断技术，特别是那些运用实验前数据的技术，对于减少方差是核心。

主要思路是使用与实验后结果 $Y_i$ 相关但*不受*处理 $W_i$ 影响的实验前信息 $X_i$（例如，实验前的用户活动指标）。通过基于 $X_i$ 调整 $Y_i$，我们可以减少其方差，而不会给处理效应估计带来偏差（因为随机化确保 $W_i$ 独立于 $X_i$）。

一种常用方法是回归调整或 CUPED（使用实验前数据进行对照实验）。我们构建一个调整后的结果：


$$
Y_i^{adj} = Y_i - \hat{E}[Y_i | X_i]
$$


其中 $\hat{E}[Y_i | X_i]$ 是仅基于实验前协变量的结果估计。一个简单的线性版本使用 $\hat{E}[Y_i | X_i] = \hat{\theta} X_i$，其中 $\hat{\theta}$ 通常通过回归（$Y \sim X$）从实验前数据或实验期间的对照组数据中估计得出。

更一般而言，$\hat{E}[Y_i | X_i]$ 可以使用在适当数据（例如，实验前数据或对照组数据）上训练的任何机器学习 (machine learning)模型进行估计。这直接与双重机器学习 (DML) 的原则相关。在用于 ATE 估计的 DML 中，我们对 $E[Y|X]$ 和 $E[W|X]$（倾向得分，在 A/B 测试中已知）进行建模。使用结果模型 $E[Y|X]$ 进行调整正是减少方差的目标。运用 DML 框架提供了一种系统的方法来执行这种调整，即使在高维 $X$ 的情况下，也能确保可靠性。

```python
# 通过 DML 结果模型进行方差缩减的 Python 代码片段
# 假设：
#   df_exp: 包含实验数据（Y, W, X_pre_experiment）的 DataFrame
#   df_pre: 包含实验前数据（Y_pre, X_pre_experiment）的 DataFrame

from sklearn.ensemble import RandomForestRegressor

# 1. 在实验前数据或对照组数据上训练结果模型
# 使用实验前数据：
# outcome_model = RandomForestRegressor()
# outcome_model.fit(df_pre['X_pre_experiment'], df_pre['Y_pre'])
# E_Y_X = outcome_model.predict(df_exp['X_pre_experiment'])

# 备选方案：使用实验中的对照组数据
control_data = df_exp[df_exp['W'] == 0]
outcome_model_ctrl = RandomForestRegressor()
outcome_model_ctrl.fit(control_data['X_pre_experiment'], control_data['Y'])
E_Y_X = outcome_model_ctrl.predict(df_exp['X_pre_experiment'])

# 2. 创建调整后的结果
df_exp['Y_adj'] = df_exp['Y'] - E_Y_X

# 3. 使用调整后的结果估计 ATE
# (例如，Y_adj 的均值差异)
ate_adj = df_exp[df_exp['W'] == 1]['Y_adj'].mean() - df_exp[df_exp['W'] == 0]['Y_adj'].mean()

# 这种调整后的 ATE 通常比 Y 的简单均值差异具有更低的方差
```

### 序贯设计和自适应实验

传统的固定周期 A/B 测试可能效率不高，需要预先确定大的样本量或长的持续时间。序贯测试方法允许在结果累积时进行监测，如果检测到统计显著效应（或缺乏效应），则可以提前停止实验，同时控制整体 I 类错误率（例如，使用 alpha-消耗函数）。

自适应实验通过在实验进行中根据观察到的数据更改实验参数 (parameter)而更进一步。例如，多臂老虎机 (MAB) 算法动态调整分配概率，将更多用户分配给当前表现最佳的臂，以最小化遗憾（机会成本）。

尽管功能强大，但解读序贯和自适应设计的结果需要谨慎。对最终结果的简单分析可能会有偏差，因为停止时间或分配概率是依赖于数据的。因果推断框架在此处很重要：

- 潜在结果表示法有助于精确定义目标可估计量，即使在自适应分配下也是如此。
- 需要纠正自适应停止或分配的方法，以获得 ATE 或 CATE 的无偏估计。
- 在自适应实验中估计 CATE 较有挑战，但可以提供关于分配为何转移的理解（例如，某个臂只对特定用户细分群体表现更优）。

### 连接实验与观察数据

由于成本、伦理问题或技术限制，实验并非总是可行。因果推断使我们能够明智地将现有 A/B 测试的理解与更丰富的观察数据相结合。

- **验证：** 观察性因果估计（使用第 3 和第 4 章的方法得出）有时可以使用小规模、有针对性的 A/B 测试结果进行验证或校准。
- **推广（可迁移性）：** A/B 测试可能在一个特定人群（例如，某个国家的用户）上进行。因果可迁移性方法旨在利用关于这两个人群协变量的观察数据，将这些发现推广到不同的目标人群。这需要理解人群之间的差异以及这些差异如何改变处理效应。

通过运用因果视角，结合实验数据和观察数据的优势，我们可以在机器学习 (machine learning)系统中建立对干预效应更全面的理解。这种整合方法将 A/B 测试视为对被优化系统因果理解的一部分。

## 参考资料

- [Generalized Random Forests](https://doi.org/10.1214/18-AOS1709) — Susan Athey, Julie Tibshirani, Stefan Wager (2019)
  Journal: Ann. Statist.; Volume: 47; Pages: 1148-1178; DOI: [10.1214/18-AOS1709](https://doi.org/10.1214/18-AOS1709)
  介绍广义随机森林（包括因果森林），用于异质处理效应估计。
- [Double/debiased machine learning for treatment and causal parameters](https://arxiv.org/abs/1608.00060) — Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney K. Newey, James M. Robins (2018)
  Journal: The Econometrics Journal; Volume: 21; Pages: C1-C68; DOI: [10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097)
  提出双重/去偏机器学习框架，用于因果参数的稳健和去偏估计，适用于平均处理效应和条件平均处理效应。
- [Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFlVX4xvj_FVn_NOXojly0sA1sSzCaPwwCZ_za0DAJM1Zo1mJEZ8rebB0T9tcR3Bqz-nPzYiEFA_Vd6yEsox7j5JBRo1zQq874xPruORFQbcT_4rKX5mWOOnQ5aQXrGsYH4OYG8aSl6Smx2sEi4QPl3G9SIVkqSlwiKSz6JwY3bNzm-XugnauAblT7iuOlLFHdeGO41NMdLxqi5PBuJ) — Guido W. Imbens, Donald B. Rubin (2015)
  Publisher: Cambridge University Press
  一本关于因果推断的教科书，涵盖潜在结果、平均处理效应、稳定单位处理值假设和工具变量等基础知识。
