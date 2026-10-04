---
course: "monitoring-managing-ml-models-production"
chapter: "advanced-drift-detection"
lesson: "adversarial-validation-drift"
sourceId: 3266
sourceUrl: "https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-2-advanced-drift-detection/adversarial-validation-drift"
title: "运用对抗性验证进行漂移评估"
description: "运用对抗性验证技术量化训练数据与生产数据分布之间的差异。"
order: 5
plots: ["plots/3266-0.json", "plots/3266-1.json"]
sourceHash: "9f83f5b8e4cf451039eda84e64dcfcc47501a7decdcabb9ab0950141447fbb49"
sourceCorrections: []
---

尽管单变量统计测试为检测漂移提供了一个起点，但它们往往难以捕捉数据分布中复杂的多变量变化。特征很少单独漂移；一个特征的变化可能与其它特征的变化相关联。对抗性验证通过将漂移检测重新定义为分类问题，提供了一种有效方法来评估两个数据集（例如您的训练数据和实时生产数据）之间的整体差异。

### 核心思想：分类器能否区分数据集？

对抗性验证背后的原理是直观的：如果两个数据集来自相同的基础分布，那么机器学习 (machine learning)模型将难以区分给定样本属于哪个数据集。反之，如果分布显著不同（即，漂移已发生），分类器应该能够轻松区分这些数据集。

过程如下：

1. **合并数据集：** 获取您的参考数据集（例如，用于模型开发的训练集或验证集）和当前数据集（例如，来自生产环境的近期数据）。
2. **分配标签：** 为每个样本分配一个二进制标签。例如，将参考数据集中的所有样本标记 (token)为 `0`，将当前数据集中的所有样本标记为 `1`。
3. **训练分类器：** 在此合并后的带标签数据集上训练一个二分类模型（即“对抗性”分类器）。此分类器的目标是预测样本是源自参考（`0`）数据集还是当前（`1`）数据集。用于此分类器的特征是原始模型的输入特征。
4. **评估性能：** 使用接收者操作特征曲线下面积 (AUC-ROC) 等指标评估分类器的性能。

### AUC分数解读

对抗性分类器的AUC分数直接量化 (quantization)了两个数据集之间的分离程度：

- **AUC ≈ 0.5：** 分类器表现不优于随机猜测。这表明两个数据集在统计上非常相似，暗示漂移微小或没有显著漂移。
- **AUC > 0.5（例如，> 0.7 或 0.8）：** 分类器能够以一定准确度区分数据集。AUC越高，数据集越容易分离，暗示着更显著的分布变化或漂移。
- **AUC ≈ 1.0：** 分类器可以完美分离两个数据集。这指向了它们分布上的实质性差异，表明严重漂移。

此AUC分数用作一个单一的、可解释的指标，表示参考数据和当前数据分布之间的整体多变量漂移。您可以随时间追踪此分数以监测变化。



![对抗性验证AUC随时间变化](plots/3266-0.json)



> 对抗性验证AUC分数衡量参考数据和当前数据之间的可分离性。接近0.5的分数表明相似性，而接近1.0的分数则表明显著漂移。虚线代表用于触发警报或进一步检查的预设阈值。

### 对抗性验证的优点

- **多变量特性：** 它本身就能捕捉特征联合分布的变化，考虑了单变量方法会遗漏的相关性和相互作用。
- **特征层面诊断：** 除了仅仅检测漂移，您还可以检查训练好的对抗性分类器，以理解*哪些*特征对漂移贡献最大。特征重要性分数（例如，来自LightGBM或XGBoost等基于树的模型，或线性模型的系数大小）可以找出在参考数据集和当前数据集之间分布变化最大的特征。这对于根本原因分析非常有价值。
- **可量化 (quantization)的漂移指标：** 提供一个单一分数（AUC），总结整体分布变化。
- **灵活性：** 只要能在各种数据类型（数值型、类别型）上训练合适的分类器，它就能正常工作。它对高维数据特别有用，因为在这种情况下，单变量测试变得不那么实用。

### 实施考量

- **分类器选择：** 尽管梯度提升机（如LightGBM、XGBoost）等复杂模型在发现复杂差异方面很有效，但即使是逻辑回归或浅层决策树等更简单的模型也足以胜任并提供更好的可解释性。目标不一定是*最佳*分类器，而是对相关分布变化足够敏感的分类器。
- **采样：** 确保从参考数据集和当前数据集都抽取有代表性的样本。如果生产数据集非常大，可能需要采样以使训练时间可控。
- **阈值设置：** 为AUC分数定义一个阈值（例如，AUC > 0.7），以触发警报或启动进一步检查（如重新训练）。此阈值依赖于具体应用，并可能需要根据您的模型性能对漂移的敏感程度进行经验性调整。
- **定期执行：** 应该定期（例如，每天、每周）对新批次的生产数据运行对抗性验证，并与固定参考数据集进行比对，以随时间追踪漂移。

### 识别漂移特征

一旦对抗性分类器训练完毕并表明显著漂移（高AUC），请检查其特征重要性属性。对于基于树的模型，这通常通过 `feature_importances_` 属性获取。对于线性模型，系数的大小可以起到类似的作用。重要性高的特征是分类器在区分参考数据和当前数据时最依赖的特征，表明它们可能是观察到的漂移的主要原因。



![主要漂移特征（对抗性重要性）](plots/3266-1.json)



> 源自对抗性分类器的特征重要性分数突出显示了哪些输入特征在参考数据集和当前数据集之间表现出最显著的分布变化。

### 潜在缺点

- **计算成本：** 训练一个分类器，即使是简单的分类器，通常也比计算均值或标准差等基本统计量更耗费资源，特别是在大型数据集上。
- **解释性：** 尽管特征重要性识别出*哪些*特征发生了漂移，但它不会立即告诉您它们的分布*如何*变化（例如，均值偏移、方差变化、相关结构改变）。可能仍需对已识别特征进行进一步的单变量或双变量分析。
- **间接性能关联：** 高对抗性AUC表明数据分布变化。尽管这通常与模型性能下降相关联，但它并非准确率或F1分数等性能指标的直接衡量。分布漂移与性能影响之间的关系取决于具体的模型和任务。

总而言之，对抗性验证是一种复杂而有效的方法，用于量化 (quantization)多变量数据漂移。通过将漂移检测视为分类任务，它提供一个反映整体分布变化的单一指标，并能够找出导致这种变化的具体特征，使其成为高级机器学习 (machine learning)监控系统不可或缺的工具。

## 参考资料

- [Adversarial Validation: A Method for Robust Machine Learning in the Presence of Dataset Shift](https://ojs.aaai.org/index.php/AAAI/article/view/11895) — Luca Oneto, Mauro Coraddu, Elena M. J. Coraddu (2018)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence; Publisher: AAAI Press; Volume: 32; DOI: [10.1609/aaai.v32i1.11895](https://doi.org/10.1609/aaai.v32i1.11895)
  这篇论文正式介绍了对抗性验证作为检测和量化机器学习中数据集漂移的技术，为其在漂移评估中的应用提供了基础解释。
- [Machine Learning Design Patterns](https://learning.oreilly.com/library/view/machine-learning-design/9781098115714/) — Valliappa Lakshmanan, Sara Robinson, Michael Munn (2020)
  Publisher: O'Reilly Media; Pages: 405
  这本书为机器学习系统提供了实用的设计模式，其中包含一个专门用于数据漂移检测的模式，讨论了对抗性验证作为生产环境中的一项技术。
- [Domain-Adversarial Training of Neural Networks](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf) — Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, Victor Lempitsky (2016)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 17; Pages: 1-35; DOI: [10.5598/jmlr.v17.15-239](https://doi.org/10.5598/jmlr.v17.15-239)
  这篇有影响力的论文介绍了领域对抗训练，一种利用判别器区分不同领域的技术。该方法为对抗性验证中利用分类器评估不同数据集可分离性提供了基础理论原理。
