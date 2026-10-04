---
course: "mastering-gradient-boosting-algorithms"
chapter: "gradient-boosting-specialized-tasks"
lesson: "survival-objective-functions"
sourceId: 2072
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-9-gradient-boosting-specialized-tasks/survival-objective-functions"
title: "生存分析目标函数 (Cox 比例风险模型)"
description: "在提升框架中实现适用于生存分析的目标函数。"
order: 4
plots: []
sourceHash: "3565d48100b3fc9b20d8689b23a70f609cf5c09f96bed30442d4b04bf988843c"
sourceCorrections: []
---

训练用于时间-事件数据的梯度提升模型需要特定的目标函数。这些目标函数通常采用Cox比例风险模型的原理。

## Cox比例风险模型的基本原理

在将其整合到提升方法之前，我们先简要回顾Cox比例风险(Cox PH)模型。它是一种在生存分析中广泛使用的半参数 (parameter)模型。其主要思想是为具有协变量$X$的个体在时间$t$时，对风险率$h(t | X)$进行建模。风险率表示在给定生存到时间$t$的情况下，在时间$t$时发生事件的瞬时风险。

Cox模型假定*比例风险*。这意味着协变量的影响是乘性的，并且相对于基线风险函数$h_0(t)$在时间上保持不变：


$$
h(t | X) = h_0(t) \exp(X \beta)
$$


这里：

- $h_0(t)$ 是基线风险函数，表示所有协变量均为零的个体的风险。在标准Cox模型中，它未被具体指定。
- $X$ 是个体的协变量向量 (vector)。
- $\beta$ 是系数向量，表示每个协变量的影响程度。
- $\exp(X \beta)$ 是风险比。它量化 (quantization)了协变量如何调整基线风险。例如，如果$\exp(\beta_k) = 2$，协变量$X_k$增加一个单位，在任何给定时间$t$都会使风险率加倍，同时保持其他协变量不变。

Cox模型的一个重要特点是，它允许在不估计基线风险$h_0(t)$的情况下估计系数$\beta$。这通过最大化一个*偏似然*函数来实现，该函数将个体在特定时间发生事件的风险与当时仍在风险中的所有个体的风险进行比较。

## 调整Cox PH以用于梯度提升

梯度提升模型学习一个预测结果的函数$F(X)$。在采用类似Cox方法的生存分析中，提升模型$F(X)$扮演传统Cox模型中线性预测因子$X \beta$的角色。风险函数随后被建模为：


$$
h(t | X) = h_0(t) \exp(F(X))
$$


目标是使用提升算法来学习函数$F(X)$。为此，我们需要一个基于Cox偏似然的目标函数。

让我们考虑一个包含$N$个个体的数据集。对于每个个体$i$，我们有：

- $T_i$: 观察到的时间（事件发生时间或删失时间）。
- $\delta_i$: 事件指示器（如果在$T_i$观察到事件则为1，如果删失则为0）。
- $X_i$: 协变量向量 (vector)。

Cox模型的偏似然函数是通过考虑每个事件发生的时间$t_i$（$\delta_i = 1$）来构建的。对于每个这样的事件时间，似然函数将经历事件的个体$i$的风险与在时间$t_i$之前仍处于*风险中*的所有个体的风险总和进行比较。在时间$t_i$处于风险中的个体集合，记为$R_i$，包括所有观察时间$T_j$大于或等于$t_i$的个体$j$（即，他们在$t_i$之前未经历事件也未被删失）。

偏似然函数为：


$$
L = \prod_{i: \delta_i=1} \frac{\exp(F(X_i))}{\sum_{j \in R_i} \exp(F(X_j))}
$$


梯度提升旨在最小化一个损失函数 (loss function)，这通常是负对数似然。对偏似然取负对数，得到需要最小化的目标函数：


$$
Obj = - \log(L) = - \sum_{i: \delta_i=1} \left[ F(X_i) - \log \left( \sum_{j \in R_i} \exp(F(X_j)) \right) \right]
$$


此表达式用作在Cox比例风险框架下训练用于生存分析的梯度提升模型的损失函数。

## 提升方法中的梯度与海森矩阵

为了使用梯度提升优化此目标，我们需要其关于每个观测值$k$的模型输出$F(X_k)$的一阶和二阶导数（梯度和海森矩阵）。这些导数在每次提升迭代中指导弱学习器（通常是决策树）的训练。

计算涉及对负对数偏似然项进行求导。尽管完整推导可能比较复杂，但观测值$k$的所得梯度$g_k$和海森矩阵$h_k$取决于个体$k$是否经历过事件及其与经历事件的个体$i$的风险集$R_i$的关系。具体而言，对于$F(X_k)$的导数涉及对所有事件时间$t_i$（其中个体$k$属于风险集$R_i$）的项$\frac{\exp(F(X_k))}{\sum_{j \in R_i} \exp(F(X_j))}$求和。

当选择Cox目标时，XGBoost和LightGBM等库会在内部实现这些计算。然后，基学习器被训练以预测负梯度，模型$F(X)$则迭代更新。

## 在提升库中的实现

主要的梯度提升库提供内置支持，用于使用Cox目标的生存分析。

- **XGBoost:** 使用目标参数 (parameter) `objective='survival:cox'`。输入标签通常应为事件时间的正值，以及删失时间的负值（例如，事件时间为`t`，删失时间为`-t`）。请查阅您所用版本的XGBoost文档，了解期望的精确格式。
- **LightGBM:** 使用 `objective='coxph'`。LightGBM通常对标签期望两个独立的列：一个用于时间，一个用于事件指示器（0表示删失，1表示事件）。
- **CatBoost:** CatBoost也通过 `loss_function='Cox'` 参数支持Cox回归。与LightGBM类似，它通常期望时间列和事件指示器列。

训练好的提升模型的输出`predict(X)`提供对数相对风险$F(X)$。较高的值表示相对于基线风险，预测的事件发生风险较高。这些值可用于按风险对个体进行排名，或者如果与基线风险的估计值结合，还可以估计生存概率（尽管$h_0(t)$的估计通常在拟合提升模型后单独进行）。

## 需要注意的事项

当使用带有Cox PH目标函数的提升方法时，请记住：

- **比例风险假定：** 梯度提升并不能消除检查比例风险假定的必要性。虽然树模型的灵活性可以在一定程度上捕捉复杂的交互作用，从而近似非比例性，但仍然建议正式检验这一假定（例如，使用Schoenfeld残差）。
- **处理事件时间重叠：** 当多个个体在完全相同的时间发生事件时，标准的偏似然函数需要修改。库通常使用Breslow或Efron等常用近似方法来有效处理重叠时间。请注意您所选库默认使用的处理方法。
- **评估：** 准确率或AUC等标准指标不适用于生存分析。请使用为删失数据设计的指标，例如一致性指数（C-index）或时变AUC，来评估模型性能。许多库允许在训练期间指定这些指标以进行监控或早期停止。

通过实现Cox PH目标函数，梯度提升框架成为分析时间-事件数据的有力工具，将它们的适用范围扩展到医学研究、预测性维护和客户流失分析等删失数据常见的方面。

## 参考资料

- [Regression Models and Life-Tables](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x) — David R. Cox (1972)
  Journal: Journal of the Royal Statistical Society. Series B (Methodological); Publisher: Royal Statistical Society; Volume: 34; Pages: 187-220; DOI: [10.1111/j.2517-6161.1972.tb00899.x](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x)
  首次提出Cox比例风险模型和部分似然概念的原始出版物，构成了生存分析目标函数的基础。
- [The Statistical Analysis of Failure Time Data](https://www.wiley.com/en-us/The+Statistical+Analysis+of+Failure+Time+Data,+2nd+Edition-p-9781118032985) — John D. Kalbfleisch, Ross L. Prentice (2002)
  Publisher: Wiley; DOI: [10.1002/9781118032985](https://doi.org/10.1002/9781118032985)
  一本经典教科书，全面阐述了生存分析，包括对Cox PH模型及其理论基础的详细解释。
- [Survival Analysis with Integrated Machine Learning and Deep Learning: A Tutorial](https://www.jmlr.org/papers/v20/17-824.html) — Håvard Kvamme, Ørnulf Borgan, Daniel Fensom (2019)
  Journal: Journal of Machine Learning Research; Publisher: JMLR, Inc.; Volume: 20; Pages: 1-122; DOI: [10.5555/3305381.3305382](https://doi.org/10.5555/3305381.3305382)
  一份全面的教程，讨论了使用机器学习和深度学习进行生存分析的现代方法，包括在这些背景下应用基于Cox的损失函数。
- [Learning Task Parameters - XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters) — XGBoost Contributors (2023)
  Publisher: XGBoost Project
  XGBoost库中`survival:cox`目标函数实现的官方文档，包括输入数据格式和使用注意事项。
