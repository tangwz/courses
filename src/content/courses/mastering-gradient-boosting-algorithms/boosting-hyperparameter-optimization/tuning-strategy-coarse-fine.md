---
course: "mastering-gradient-boosting-algorithms"
chapter: "boosting-hyperparameter-optimization"
lesson: "tuning-strategy-coarse-fine"
sourceId: 2059
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-8-boosting-hyperparameter-optimization/tuning-strategy-coarse-fine"
title: "调优策略：从粗到细"
description: "为复杂模型调优制定实用流程。"
order: 6
plots: []
sourceHash: "3ba558c95cb627cc90b23e40c61eaf463fd5908ff2d6a28509d897dbfa3b8193"
sourceCorrections: []
---

应对梯度提升模型庞大的超参数 (parameter) (hyperparameter)空间可能令人望而却步。虽然像随机搜索或贝叶斯优化这样的自动化搜索算法是强大的工具，但盲目地将它们应用于每个参数的整个潜在范围通常计算成本高昂且效率低下。一种更结构化且有效的方法涉及多阶段策略，从广泛的粗略搜索进展到更集中的精细调优阶段。这种方法有助于明智地分配计算资源，并增加找到高性能超参数配置的可能性。

### 阶段1：粗略调优（广泛寻优）

最初目标不一定是找到绝对最佳参数 (parameter)，而是快速识别超参数 (hyperparameter)空间中有潜力的*区域*。可以将其视为描绘模型性能的概括区域。

- **目标：** 确定有影响的参数，以及可能存在良好性能的大致范围。
- **方法：** 随机搜索通常是一个不错的选择，因为它在寻找多样组合方面效率较高。贝叶斯优化的早期阶段也可以达到此目的。
- **参数：** 专注于对梯度提升模型有显著影响的超参数：
  - `learning_rate`（或 `eta`）：控制每次提升迭代的步长。
  - `n_estimators`：提升迭代次数（树的数量）。请注意，此参数通常最好通过早期停止来控制，而不是最初在大范围上直接调优，尤其是在 `learning_rate` 也正在调优时。设置一个足够大的最大值，并让早期停止为给定的一组其他参数找到最佳值。
  - `max_depth`：单棵树的最大深度。控制模型复杂度。
  - 采样参数（例如，`subsample`、`colsample_bytree`、`colsample_bylevel`、`colsample_bynode`）：控制用于构建每棵树的数据或特征比例。对正则化 (regularization)很重要。
  - 正则化参数（例如，XGBoost 中的 `lambda`、`alpha`；LightGBM 中的 `reg_lambda`、`reg_alpha`）：树权重 (weight)上的L2和L1正则化项。
- **范围：** 为这些参数定义广泛范围。对于 `learning_rate` 或正则化项等参数，对数尺度通常适用（例如，范围从0.001到0.1）。对于树深度或采样率，线性范围适用。
- **执行：**
  - 使用有限数量的搜索迭代（例如，30-100次，取决于预算）。
  - 采用更简单的验证策略，例如单个保留验证集或计算量较小的交叉验证设置（例如，3折交叉验证），以加快评估速度。
  - 如果处理非常大的数据集，可以考虑在数据的代表性子集上执行此阶段。

此阶段的结果应是粗略认识到哪些参数范围能产生更好性能，以及哪些可能可以从后续研究中排除。例如，您可能会发现低于0.01的学习率持续表现不佳，或者深度大于8会导致过拟合 (overfitting)而没有显著增益。

### 阶段2：精细调优（集中优化）

一旦粗略搜索确定了有潜力的区域，下一步就是聚焦并在此缩小后的范围内进行更密集的搜索。

- **目标：** 在早期确定的高潜力区域内精确确定最佳超参数 (parameter) (hyperparameter)配置。
- **方法：** 贝叶斯优化（使用Optuna或Hyperopt等框架）在此表现出色。它建模目标函数和智能选择要尝试的下一个参数的能力，使其在受限空间内优化解决方案时效率高。如果优化后的搜索空间足够小，集中的网格搜索也可行。
- **参数：** 对最有影响力的参数使用阶段1中确定的缩小范围。您还可以引入在粗略阶段保持不变或影响较小的参数，以进行精细调整（例如，XGBoost 中的 `min_child_weight`、`gamma`）。
- **范围：** 根据粗略结果定义更紧密的搜索空间。例如，如果粗略搜索表明良好的学习率在0.01到0.05之间，则精细调优范围可能变为 `LogUniform(0.01, 0.05)`。
- **执行：**
  - 分配更大的计算预算，允许更多搜索迭代（例如，50-200+次）。
  - 使用更稳健的评估策略，通常在完整训练数据集上进行k折交叉验证（例如，5折或10折交叉验证），以获得可靠的性能估计。
  - 确保在每次评估中有效使用早期停止，以确定正在测试的特定参数组合的最佳 `n_estimators`。

此阶段每次迭代需要更多计算，但能更详细地检查搜索空间中最有希望的区域，显著增加找到接近真实最优配置的可能性。

> 一个流程图，说明了两阶段调优策略，它基于中间分析，从广泛寻优（粗略搜索）转向集中优化（精细调优）。

### 整合早期停止

如前所述，`n_estimators`（树的数量或提升迭代次数）与 `learning_rate` 紧密相关。较低的学习率通常需要更多树才能收敛。与其将 `n_estimators` 作为在广泛范围（例如，100到5000）内搜索的主要超参数 (parameter) (hyperparameter)，更高效的做法是：

1. 在提升器的配置中为 `n_estimators` 设置一个合理大的上限（例如，2000）。
2. 在每次评估（无论是单个验证拆分还是交叉验证的每个折叠内）期间，使用XGBoost、LightGBM和CatBoost提供的早期停止功能。
3. 配置早期停止以监控验证集上的性能指标，并在指标在指定轮数内没有改进时停止训练（例如，`early_stopping_rounds=50`）。

这样，对于搜索算法测试的 *其他* 超参数（如 `learning_rate`、`max_depth` 等）的每个组合，树的最佳数量会自动高效地确定。

### 实用考量

- **计算预算：** 迭代次数、交叉验证的复杂度以及数据子集化的决定，都受到可用时间和计算能力的严重影响。从粗到细的策略有助于在这些限制内最大化获得的值。
- **优化框架：** 像Optuna这样的库非常适合这种策略。您可以运行一个初始的“粗略”研究，分析其结果（Optuna提供有用的可视化工具），然后发起第二次“精细”研究，其搜索空间根据第一次研究进行调整。
- **迭代：** 超参数 (parameter) (hyperparameter)调优并非总是严格线性的过程。精细调优阶段的结果有时会表明最初的粗略搜索太窄或完全错过了潜在更好的区域。如果结果不令人满意，或者精细调优似乎停留在次优区域，请准备好重新审视之前的假设，并可能运行额外的粗略或中间搜索。

通过采用这种结构化的从粗到细调优策略，您将摆脱暴力搜索，并应用更智能、更资源高效的流程来优化您的梯度提升模型，从而获得更好的性能和更可靠的结果。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  提供XGBoost的理论和实践基础，包含其架构、正则化技术以及本节讨论的关键超参数。
- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research Editorial Board; Volume: 13; Pages: 281-305; DOI: [10.1162/JMLR.2012.13.issue.oct.281-305](https://doi.org/10.1162/JMLR.2012.13.issue.oct.281-305)
  介绍并实证展示了随机搜索在超参数优化方面优于网格搜索的有效性，与粗略调整阶段高度相关。
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://doi.org/10.1145/3292500.3330701) — Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, Masanori Koyama (2019)
  Journal: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19); Publisher: ACM; Pages: 2623-2631; DOI: [10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701)
  介绍Optuna，一个支持多阶段调整和贝叶斯优化的优化框架，与本节描述的精细调整阶段一致。
- [Parameter Tuning](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html) — XGBoost Developers (2024)
  官方文档，提供XGBoost超参数调整的实用指导，包含策略、参数交互以及早停的使用。
