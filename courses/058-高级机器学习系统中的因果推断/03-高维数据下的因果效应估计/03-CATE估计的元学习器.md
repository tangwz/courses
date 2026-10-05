# CATE估计的元学习器

来源：[原文](https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-3-high-dimensional-effect-estimation/meta-learners-cate)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管像双重机器学习 (machine learning)这样的方法在估计平均处理效应（ATE）方面表现出色，但许多应用需要对个体差异有更细致的理解：治疗$T$对结果$Y$的影响如何因具有不同特征$X$的个体而变化？这种变化的效应就是条件平均处理效应，或称CATE：


$$
\tau(x) = E[Y(1) - Y(0) | X=x]
$$


在高维环境中估计CATE面临特定的难题。我们需要能够灵活地建模协变量$X$与处理效应本身之间关系的方法，而无需施加严格的参数 (parameter)假设。元学习器通过借助现有的监督机器学习算法（“基础学习器”）来构建CATE估计器，提供了一个有用的框架来实现这一目标。它们被称为“元”学习器，因为它们在标准预测模型之上运行。

我们将审视三种重要的元学习器：S-学习器、T-学习器和X-学习器。每种都提供了一种不同的策略来重新利用监督学习 (supervised learning)算法进行CATE估计。

### S-学习器：通过特征包含实现简洁

S-学习器（单一学习器）采取最直接的方法。它将处理指示器$T$作为常规特征，与协变量$X$一起包含在一个单一模型中，该模型训练用于预测结果$Y$。

令$\mu(x, t) = E[Y | X=x, T=t]$为在协变量和处理分配条件下的预期结果。S-学习器使用完整数据集$(X_i, T_i, Y_i)$训练一个单一的监督学习 (supervised learning)模型$\hat{\mu}$（例如，梯度提升、随机森林、神经网络 (neural network)）来近似$\mu(x, t)$。

The CATE然后通过取模型在给定$x$下，$T$设为1和$T$设为0时的预测值之差来估计：


$$
\hat{\tau}_S(x) = \hat{\mu}(x, T=1) - \hat{\mu}(x, T=0)
$$


**优点：**

- **简洁：** 使用任何标准监督学习库都易于实现。
- **直接结果建模：** 直接建模响应曲面$E[Y|X, T]$。

**缺点：**

- **处理效应正则化 (regularization)：** 如果处理效应$\tau(x)$相对于$X$对$Y$的主要效应较小，基础学习器$\hat{\mu}$内的正则化可能会减弱甚至消除估计的效应。模型优先考虑$Y$的整体预测准确性，而不是特指与$T$相关的*差异*的准确性。
- **模型特有性：** 假设基础学习器适用于使用相同的函数形式和正则化来捕获主要结果关系和处理效应异质性。例如，如果结果$Y$与$X$平滑相关，但处理效应$\tau(x)$具有剧烈的不连续性，则此假设可能不成立。

S-学习器是一个有用的基准，但当处理效应不明显或其结构与基准结果函数显著不同时，其表现通常不佳。

### T-学习器：治疗组和对照组的独立模型

T-学习器（双学习器）采用更直接的方法来建模潜在结果。它构建两个独立的监督学习 (supervised learning)模型：一个用于治疗下的结果，一个用于对照下的结果。

1. **治疗组模型：** 仅使用$T=1$的数据点训练模型$\hat{\mu}_1$来预测$Y$。此模型估计$\mu_1(x) = E[Y | X=x, T=1]$。
2. **对照组模型：** 仅使用$T=0$的数据点训练第二个模型$\hat{\mu}_0$来预测$Y$。此模型估计$\mu_0(x) = E[Y | X=x, T=0]$。

The CATE然后将估计为这两个模型预测值之间的差：


$$
\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)
$$

> T-学习器方法的流程，将数据分割以训练治疗组和对照组的独立预测结果模型。

**优点：**

- **直接潜在结果建模：** 明确地分别为治疗组和对照组建模响应曲面。
- **灵活性：** 允许对$\hat{\mu}_1$和$\hat{\mu}_0$使用不同的基础学习器或超参数 (parameter) (hyperparameter)设置，这可能适应每个组的不同复杂度。

**缺点：**

- **数据稀疏性：** 如果一个治疗组远小于另一个治疗组（处理分配不平衡），则较小治疗组的模型可能由于数据不足而表现不佳。
- **误差传播：** 最终的CATE估计结合了来自两个独立模型的误差。
- **忽视共享信息：** 未能借助两个治疗组之间$X$和$Y$关系中潜在的相似性。$\hat{\mu}_1$学到的关于$X$效应的信息不能直接为$\hat{\mu}_0$提供信息，反之亦然。

当处理效应显著时，T-学习器通常比S-学习器表现更好，但当处理不平衡时，其表现可能会下降。

### X-学习器：借助估算处理效应

X-学习器由Künzel 等人（2019）提出，旨在解决S和T-学习器的不足之处，特别是在处理组不平衡或CATE函数复杂的情况下。它采用多阶段估计策略。

**阶段1：估计结果模型（类似于T-学习器）**
首先，像T-学习器那样，分别使用治疗组（$T=1$）和对照组（$T=0$）的数据，精确估计独立的预测结果模型$\hat{\mu}_1(x)$和$\hat{\mu}_0(x)$。

**阶段2：估算个体处理效应**
接下来，使用阶段1的模型来估算每个个体的*反事实*结果，并计算估算的处理效应：

- 对于*治疗组*中的个体（$T_i=1$），将效应估算为：
  
  $$
  \tilde{D}_i^1 = Y_i^{\text{obs}} - \hat{\mu}_0(X_i)
  $$
  
  这代表观测到的结果减去如果他们*未*接受治疗时的预测结果。
- 对于*对照组*中的个体（$T_i=0$），将效应估算为：
  
  $$
  \tilde{D}_i^0 = \hat{\mu}_1(X_i) - Y_i^{\text{obs}}
  $$
  
  这代表如果他们接受治疗时的预测结果减去他们观测到的结果。

**阶段3：使用估算效应估计CATE**
现在，将估算的效应$\tilde{D}^1$和$\tilde{D}^0$视为目标变量。训练两个新的监督学习 (supervised learning)模型，根据协变量$X$预测这些估算的效应：

- 使用所有$T_i=1$的单元的数据集$(X_i, \tilde{D}_i^1)$训练模型$\hat{\tau}_1(x)$。
- 使用所有$T_i=0$的单元的数据集$(X_i, \tilde{D}_i^0)$训练模型$\hat{\tau}_0(x)$。
  这些模型直接学习协变量$X$与每个组内估计处理效应之间的关系。

**阶段4：通过加权组合估计**
最后，使用加权函数$g(x)$组合两个CATE估计$\hat{\tau}_1(x)$和$\hat{\tau}_0(x)$。$g(x)$的一个常见选择是倾向得分的估计值，即$\hat{e}(x) = P(T=1 | X=x)$：


$$
\hat{\tau}_X(x) = g(x) \hat{\tau}_0(x) + (1 - g(x)) \hat{\tau}_1(x)
$$


这种加权方案在给定$x$时，为来自较大组的估计赋予更多权重 (weight)。例如，如果倾向得分较低（$g(x) \approx 0$），这意味着具有协变量$x$的个体很少接受治疗，则最终估计更多地依赖于$\hat{\tau}_1(x)$，这是使用治疗组（我们有更多关于$Y(1)$的信息）的估算效应学习得到的。相反，如果$g(x) \approx 1$，则估计更多地依赖于$\hat{\tau}_0(x)$。

**优点：**

- **对不平衡数据有效：** 通过估算效应并单独建模，它使用来自较大组的数据来改进与较小组相关的估计。
- **直接CATE建模：** 阶段3专门针对CATE函数$\tau(x)$。
- **渐近性质：** 在某些条件下具有理想的理论性质。

**缺点：**

- **复杂性：** 涉及多个估计步骤，增加了实现复杂度和潜在的故障点。
- **误差传播：** 阶段1模型（$\hat{\mu}_0, \hat{\mu}_1$）的误差传播到估算效应（$\tilde{D}^0, \tilde{D}^1$），并随后传播到阶段3模型（$\hat{\tau}_0, \hat{\tau}_1$）。
- **倾向得分估计：** 加权需要倾向得分$g(x)$的估计，如果倾向模型错误设定，这将引入另一个潜在的误差源。

X-学习器通常是最精巧且表现最佳的元学习器，特别是在处理显著异质性或数据不平衡时，但其复杂性需要仔细的实现和验证。

### 元学习器的选择与实现

S、T和X-学习器之间的选择取决于问题的具体特点：

- **S-学习器：** 一个简单的基准，如果预期处理效应较大且以类似于主要结果函数的方式与协变量平滑相关，则可能适用。
- **T-学习器：** 当治疗组相对平衡且单独建模看起来适用时，是一个不错的选择。
- **X-学习器：** 通常在治疗组不平衡、CATE复杂或最大限度地利用所有可用数据很关键时更受偏爱。

**实现注意事项：**

- **基础学习器：** 任何元学习器的表现都很大程度上取决于底层监督机器学习 (machine learning)模型（$\hat{\mu}$、$\hat{\mu}_0$、$\hat{\mu}_1$、$\hat{\tau}_0$、$\hat{\tau}_1$）的选择和调优。通常使用能够处理高维数据和复杂交互的模型（例如，梯度提升、随机森林、神经网络 (neural network)）。请注意这些基础学习器中的正则化 (regularization)如何影响CATE估计（特别是S-学习器）。
- **倾向得分（针对X-学习器）：** X-学习器中的加权函数$g(x)$需要倾向得分模型。该模型的质量会影响最终的CATE估计。
- **软件：** Python中的`causalml`和`dowhy`等库提供了这些元学习器的实现，简化了其应用。例如，`causalml.inference.meta`提供了`BaseSLearner`、`BaseTLearner`和`BaseXLearner`等类，允许插入scikit-learn兼容的估计器作为基础学习器。

元学习器在标准监督学习 (supervised learning)和因果效应估计之间提供了灵活且有用的桥梁。通过巧妙地构建预测任务，它们使我们能够使用熟悉的机器学习工具估计复杂、异质的处理效应，这是朝着理解高维环境中个体化影响迈出的重要一步。请记住，像所有依赖协变量调整的方法一样，元学习器根本上依赖于**无混杂假设**：$(Y(1), Y(0)) \perp T | X$。它们根据观测到的协变量$X$估计CATE，假设这些协变量足以控制混杂偏倚。评估这些估计的稳定性，如后续所述，仍然是工作流程的重要组成部分。

## 参考资料

- [Meta-Learners for Estimating Heterogeneous Treatment Effects using Supervised Learning](https://www.pnas.org/doi/10.1073/pnas.1804597116) — Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu (2019)
  Journal: Proceedings of the National Academy of Sciences; Publisher: National Academy of Sciences; Volume: 116; Pages: 4153-4162; DOI: [10.1073/pnas.1804597116](https://doi.org/10.1073/pnas.1804597116)
  这篇基础论文介绍了 X-Learner，并为 CATE 估计中的 S-Learner 和 T-Learner 方法提供了一个全面的框架。
- [Causal Inference for The Brave and True](https://matheusfacure.github.io/python-causality-handbook) — Matheus Facure Alves (2022)
  这是一本易于理解的在线教科书，提供了因果推断的实践示例和理论背景，包括对元学习器的详细说明。
- [Estimation and Inference for Heterogeneous Treatment Effects using Causal Forests](https://doi.org/10.1080/01621459.2017.1319839) — Stefan Wager, Susan Athey (2018)
  Journal: Journal of the American Statistical Association; Publisher: Taylor & Francis; Volume: 113; Pages: 1228-1242; DOI: [10.1080/01621459.2017.1319839](https://doi.org/10.1080/01621459.2017.1319839)
  一篇基础性工作，介绍了因果森林，这是一种用于估计异质治疗效果的非参数方法，解决了与元学习器类似的挑战。
- [CausalML Documentation: Meta-Learners](https://causalml.readthedocs.io/en/latest/methodology.html#meta-learner-algorithms) — CausalML Developers (2024)
  提供了使用 Python 的 CausalML 库实现 S-Learner、T-Learner 和 X-Learner 的实际示例和 API 规范。

---

[上一节](02-%E5%9B%A0%E6%9E%9C%E6%A3%AE%E6%9E%97%E4%B8%8E%E5%BC%82%E8%B4%A8%E6%95%88%E5%BA%94.md) · [下一节](04-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9C%A8%E6%95%88%E5%BA%94%E4%BC%B0%E8%AE%A1%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md)
